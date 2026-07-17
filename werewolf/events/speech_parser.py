"""LLM parser for speech-derived events only."""

import json
from dataclasses import dataclass

from werewolf.events.schema import make_event, normalize_content
from werewolf.prompt_protocol import PARSER_PROMPT_SPEC


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

SYSTEM_PROMPT = PARSER_PROMPT_SPEC["text"]


@dataclass(frozen=True)
class SpeechParseResult:
    status: str
    events: tuple[dict, ...]
    raw_text: tuple[str, ...]
    error: str | None
    attempts: int
    model: str | None


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
        raise ValueError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"events"}:
        raise ValueError("response must contain only events")
    if not isinstance(payload["events"], list):
        raise ValueError("events must be a list")

    parsed = []
    for index, item in enumerate(payload["events"], start=1):
        if not isinstance(item, dict) or set(item) != PARSED_EVENT_FIELDS:
            raise ValueError("parsed event fields do not match the speech schema")
        if item["event_family"] not in SPEECH_FAMILIES:
            raise ValueError("speech parser emitted a non-speech event family")
        content = item["content"]
        if not isinstance(content, dict) or set(content) != {"kind", "value"}:
            raise ValueError("parsed content must contain only kind and value")
        if content["kind"] not in SPEECH_KINDS[item["event_family"]]:
            raise ValueError("content.kind is not valid for its speech family")
        normalized_content = normalize_content(content)
        if normalized_content != content:
            raise ValueError("speech parser content.value must already be canonical")
        source_span = item["source_span"]
        if not isinstance(source_span, str) or not source_span:
            raise ValueError("source_span must be a non-empty exact quote")
        if source_span not in utterance:
            raise ValueError("source_span is not present in the utterance")
        parsed.append(
            make_event(
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
                qualifier=item["qualifier"],
                ref_event_id=item["ref_event_id"],
                source_span=source_span,
                parser_confidence=item["parser_confidence"],
            )
        )
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
        user_content = json.dumps(
            {"speaker": speaker, "utterance": utterance}, ensure_ascii=False
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
                            "content": "你的上一条回复不符合 schema。只返回修正后的有效 JSON。",
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
                    status="ok",
                    events=events,
                    raw_text=tuple(raw_text),
                    error=None,
                    attempts=attempt,
                    model=self.model,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        return SpeechParseResult(
            status="failed",
            events=(),
            raw_text=tuple(raw_text),
            error=error or "speech parsing failed",
            attempts=2,
            model=self.model,
        )
