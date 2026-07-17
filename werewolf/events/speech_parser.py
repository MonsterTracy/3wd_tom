"""LLM parser for speech-derived events only."""

import json
from dataclasses import dataclass

from werewolf.backends.base import BackendError
from werewolf.events.schema import QUALIFIER_ENUMS, make_event, normalize_content
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
    def __init__(
        self,
        code,
        message,
        *,
        field=None,
        invalid_value=None,
        allowed_values=None,
        event_index=None,
        suggested_value=None,
    ):
        if not isinstance(code, str) or not code:
            raise TypeError("SpeechParserError code must be non-empty text")
        if not isinstance(message, str) or not message:
            raise TypeError("SpeechParserError message must be non-empty text")
        if field is not None and not isinstance(field, str):
            raise TypeError("SpeechParserError field must be text or null")
        if invalid_value is not None and type(invalid_value) not in (
            str, int, float, bool
        ):
            raise TypeError("SpeechParserError invalid_value must be a safe scalar")
        if allowed_values is not None and (
            not isinstance(allowed_values, (list, tuple))
            or any(not isinstance(value, str) for value in allowed_values)
        ):
            raise TypeError("SpeechParserError allowed_values must be text values")
        if event_index is not None and (
            type(event_index) is not int or event_index < 1
        ):
            raise TypeError("SpeechParserError event_index must be a positive integer")
        if suggested_value is not None and not isinstance(suggested_value, str):
            raise TypeError("SpeechParserError suggested_value must be text or null")
        super().__init__(message)
        self.code = code
        self.field = field
        self.invalid_value = invalid_value
        self.allowed_values = list(allowed_values or ())
        self.event_index = event_index
        self.suggested_value = suggested_value

    @property
    def details(self):
        return {
            "field": self.field,
            "invalid_value": self.invalid_value,
            "allowed_values": list(self.allowed_values),
            "event_index": self.event_index,
            "suggested_value": self.suggested_value,
        }

    def repair_message(self):
        return parser_repair_message(message=str(self), **self.details)


_QUALIFIER_ALIASES = {
    "certainty": {
        "low": "weak",
        "medium": "normal",
        "high": "strong",
        "likely": "strong",
    },
    "strength": {"low": "weak", "medium": "normal", "high": "strong"},
    "commitment": {
        "low": "consider",
        "undecided": "consider",
        "medium": "intend",
        "proposal": "intend",
        "high": "commit",
    },
    "evidence_source": {
        "public_info": "public_history",
        "claimed_public_info": "public_history",
        "inference": "unspecified",
        "deduction": "unspecified",
    },
}


def _parser_allowed_qualifier_values(field):
    values = [value for value in QUALIFIER_ENUMS[field] if value is not None]
    if field == "evidence_source":
        values.remove("private_fact")
    return values


def _safe_invalid_value(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    return f"<{type(value).__name__}>"


def _normalize_parser_qualifier(qualifier, *, event_index=None):
    if qualifier is None:
        return {}
    if not isinstance(qualifier, dict):
        raise SpeechParserError("schema_validation", "qualifier must be an object or null")
    normalized = dict(qualifier)
    for field, aliases in _QUALIFIER_ALIASES.items():
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = aliases.get(value.strip().lower(), value)
    for field, value in normalized.items():
        if field not in QUALIFIER_ENUMS:
            raise SpeechParserError(
                "schema_validation",
                f"unknown qualifier field: {field!r}",
                field=f"qualifier.{field}",
                invalid_value=_safe_invalid_value(value),
                allowed_values=sorted(QUALIFIER_ENUMS),
                event_index=event_index,
            )
        allowed_values = _parser_allowed_qualifier_values(field)
        if value is not None and value not in allowed_values:
            raw_value = qualifier[field]
            suggested_value = None
            if isinstance(raw_value, str):
                suggested_value = _QUALIFIER_ALIASES.get(field, {}).get(
                    raw_value.strip().lower()
                )
            raise SpeechParserError(
                "schema_validation",
                f"invalid qualifier {field}: {raw_value!r}",
                field=f"qualifier.{field}",
                invalid_value=_safe_invalid_value(raw_value),
                allowed_values=allowed_values,
                event_index=event_index,
                suggested_value=suggested_value,
            )
    return normalized


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
        qualifier = _normalize_parser_qualifier(
            item["qualifier"], event_index=index
        )
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
        parser_error = None
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
                            "content": (
                                parser_error.repair_message()
                                if parser_error is not None
                                else parser_repair_message()
                            ),
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
                parser_error = exc
                error_code = exc.code
                error = f"{type(exc).__name__}: {exc}"
            except BackendError as exc:
                parser_error = None
                error_code = "backend_error"
                error = f"{type(exc).__name__}: {exc}"
                break
            except Exception as exc:
                parser_error = None
                error_code = "backend_error"
                backend_error = BackendError(
                    "Speech parser backend raised an unexpected exception.",
                    retryable=False,
                    details={"cause_type": type(exc).__name__},
                )
                error = f"{type(backend_error).__name__}: {backend_error}"
                break
        return SpeechParseResult(
            status="failed",
            events=(),
            raw_text=tuple(raw_text),
            error=error or "speech parsing failed",
            error_code=error_code or "backend_error",
            attempts=attempt,
            model=self.model,
        )
