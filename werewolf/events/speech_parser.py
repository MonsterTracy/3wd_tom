"""LLM parser for speech-derived events only."""

import json
from dataclasses import dataclass

from werewolf.events.schema import make_event, normalize_content
from werewolf.prompt_protocol import (
    PARSER_PROMPT_SPEC,
    PARSER_SYSTEM_PROMPT,
    parser_few_shot_messages,
    parser_repair_message,
    render_parser_user_message,
)


SPEECH_FAMILIES = (
    "BELIEF_ASSERTION",
    "SOCIAL_STANCE",
    "ACTION_POSITION",
    "CLAIM_RESPONSE",
)
SPEECH_KINDS = {
    "BELIEF_ASSERTION": {"ROLE", "CAMP", "FACT"},
    "SOCIAL_STANCE": {"STANCE"},
    "ACTION_POSITION": {"ACTION"},
    "CLAIM_RESPONSE": {"RELATION"},
}
PARSED_EVENT_FIELDS = {
    "event_family",
    "target",
    "content",
    "qualifier",
    "ref_event_id",
    "source_span",
    "parser_confidence",
}

SYSTEM_PROMPT = PARSER_SYSTEM_PROMPT


@dataclass(frozen=True)
class SpeechParseResult:
    status: str
    events: tuple[dict, ...]
    raw_text: tuple[str, ...]
    error: str | None
    error_code: str | None
    attempts: int
    model: str | None

    def audit_metadata(self):
        return {
            "version": PARSER_PROMPT_SPEC["version"],
            "sha256": PARSER_PROMPT_SPEC["sha256"],
            "model": self.model,
            "temperature": 0.0,
            "status": self.status,
            "attempts": self.attempts,
            "error_code": self.error_code,
            "error": self.error,
        }


class SpeechParserError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


_QUALIFIER_ALIASES = {
    "certainty": {"low": "weak", "medium": "normal", "high": "strong"},
    "strength": {"low": "weak", "medium": "normal", "high": "strong"},
    "commitment": {
        "low": "consider",
        "undecided": "consider",
        "medium": "intend",
        "proposal": "intend",
        "high": "commit",
    },
}


def _normalize_parser_qualifier(qualifier):
    if qualifier is None:
        return {}
    if not isinstance(qualifier, dict):
        raise SpeechParserError("schema_validation", "qualifier must be an object or null")
    normalized = dict(qualifier)
    for field, aliases in _QUALIFIER_ALIASES.items():
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = aliases.get(value.strip().lower(), value)
    return normalized


def _explicit_private_claim(utterance, source_span, *, speaker, target):
    targets = target if isinstance(target, (list, tuple, set)) else [target]
    if speaker in targets or not any(value in range(1, 8) for value in targets):
        return False
    start = utterance.find(source_span)
    if start < 0:
        return False
    boundaries = "。！？!?;；\n"
    left = max(utterance.rfind(mark, 0, start) for mark in boundaries) + 1
    right_candidates = [
        position
        for mark in boundaries
        if (position := utterance.find(mark, start + len(source_span))) >= 0
    ]
    right = min(right_candidates, default=len(utterance))
    sentence = utterance[left:right]
    return any(marker in sentence for marker in ("查验", "查杀", "验了", "验出"))


def _parse_payload(
    text,
    *,
    utterance,
    utterance_id,
    day,
    phase,
    turn,
    speaker,
    parser_metadata,
):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpeechParserError(
            "invalid_json", f"response is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"events"}:
        raise SpeechParserError("schema_validation", "response must contain only events")
    if not isinstance(payload["events"], list):
        raise SpeechParserError("schema_validation", "events must be a list")

    parsed = []
    for index, item in enumerate(payload["events"], start=1):
        if not isinstance(item, dict) or set(item) != PARSED_EVENT_FIELDS:
            raise SpeechParserError(
                "schema_validation", "parsed event fields do not match the speech schema"
            )
        if item["event_family"] not in SPEECH_FAMILIES:
            raise SpeechParserError(
                "schema_validation", "speech parser emitted a non-speech event family"
            )
        content = item["content"]
        if not isinstance(content, dict) or set(content) != {"kind", "value"}:
            raise SpeechParserError(
                "schema_validation", "parsed content must contain only kind and value"
            )
        if content["kind"] not in SPEECH_KINDS[item["event_family"]]:
            raise SpeechParserError(
                "schema_validation", "content.kind is not valid for its speech family"
            )
        try:
            normalized_content = normalize_content(content)
        except ValueError as exc:
            raise SpeechParserError("schema_validation", str(exc)) from exc
        if normalized_content != content:
            raise SpeechParserError(
                "schema_validation", "speech parser content.value must already be canonical"
            )
        source_span = item["source_span"]
        if not isinstance(source_span, str) or not source_span:
            raise SpeechParserError(
                "schema_validation", "source_span must be a non-empty exact quote"
            )
        if source_span not in utterance:
            raise SpeechParserError(
                "schema_validation", "source_span is not present in the utterance"
            )
        qualifier = _normalize_parser_qualifier(item["qualifier"])
        if (
            item["event_family"] == "BELIEF_ASSERTION"
            and content["kind"] in {"ROLE", "CAMP"}
            and qualifier.get("evidence_source") is None
            and _explicit_private_claim(
                utterance, source_span, speaker=speaker, target=item["target"]
            )
        ):
            qualifier["evidence_source"] = "claimed_private_info"
        try:
            event = make_event(
                event_id=f"{utterance_id}.parsed.{index}",
                utterance_id=utterance_id,
                day=day,
                phase=phase,
                turn=turn,
                source_type="speech_parser",
                visibility="public",
                visible_to=range(1, 8),
                speaker=speaker,
                event_family=item["event_family"],
                target=item["target"],
                content=content,
                metadata={"parser_protocol": parser_metadata},
                qualifier=qualifier,
                ref_event_id=item["ref_event_id"],
                source_span=source_span,
                parser_confidence=item["parser_confidence"],
            )
        except ValueError as exc:
            raise SpeechParserError("schema_validation", str(exc)) from exc
        parsed.append(event)
    return tuple(parsed)


class SpeechEventParser:
    def __init__(self, backend, model=None, *, max_tokens=800):
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = 0.0

    def parse(self, *, utterance, utterance_id, day, phase, turn, speaker):
        raw_text = []
        error = None
        error_code = None
        user_content = render_parser_user_message(speaker, utterance)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *parser_few_shot_messages(),
            {"role": "user", "content": user_content},
        ]
        for attempt in range(1, 3):
            if attempt == 2:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": raw_text[-1] if raw_text else "",
                        },
                        {
                            "role": "user",
                            "content": parser_repair_message(),
                        },
                    ]
                )
            try:
                response = self.backend.chat(
                    messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text.append(response if isinstance(response, str) else str(response))
                events = _parse_payload(
                    raw_text[-1],
                    utterance=utterance,
                    utterance_id=utterance_id,
                    day=day,
                    phase=phase,
                    turn=turn,
                    speaker=speaker,
                    parser_metadata={
                        "version": PARSER_PROMPT_SPEC["version"],
                        "sha256": PARSER_PROMPT_SPEC["sha256"],
                        "model": self.model,
                        "temperature": self.temperature,
                        "attempts": attempt,
                        "status": "ok",
                    },
                )
                return SpeechParseResult(
                    status="success" if events else "empty",
                    events=events,
                    raw_text=tuple(raw_text),
                    error=None,
                    error_code=None,
                    attempts=attempt,
                    model=self.model,
                )
            except SpeechParserError as exc:
                error_code = exc.code
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                error_code = "backend_error"
                error = f"{type(exc).__name__}: {exc}"
        return SpeechParseResult(
            status="failed",
            events=(),
            raw_text=tuple(raw_text),
            error=error or "speech parsing failed",
            error_code=error_code or "backend_error",
            attempts=2,
            model=self.model,
        )
