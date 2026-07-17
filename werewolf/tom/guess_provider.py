"""Side-effect-free belief elicitation using an agent's LLM backend."""

import json
from dataclasses import dataclass

from werewolf.prompt_protocol import BELIEF_PROMPT_SPEC
from werewolf.tom.pair_space import normalize_pair, pair_index, validate_pair_mask


SYSTEM_PROMPT = BELIEF_PROMPT_SPEC["text"]


@dataclass(frozen=True)
class GuessResult:
    status: str
    pair: tuple[int, int] | None
    raw_text: tuple[str, ...]
    error: str | None
    attempts: int
    model: str | None


def parse_guess_response(text: str) -> tuple[int, int]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"wolf_pair"}:
        raise ValueError("response must contain only wolf_pair")
    return normalize_pair(payload["wolf_pair"])


class BeliefGuessProvider:
    """Elicit a label without mutating the game agent or environment."""

    def __init__(self, backend, model=None, *, max_tokens=40):
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens

    def elicit(self, *, player_view: str, output_mask) -> GuessResult:
        valid_mask = validate_pair_mask(output_mask)
        raw_text = []
        error = None
        initial_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": player_view},
        ]
        for attempt in range(1, 3):
            messages = [dict(message) for message in initial_messages]
            if attempt == 2 and raw_text:
                messages.append(
                    {
                        "role": "assistant",
                        "content": raw_text[-1],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你的上一条回复格式无效。只返回符合要求的 JSON 对象，"
                            "并给出一个合法的双狼人组合。"
                        ),
                    }
                )
            try:
                response = self.backend.chat(
                    messages,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text.append(response if isinstance(response, str) else str(response))
                pair = parse_guess_response(raw_text[-1])
                if not valid_mask[pair_index(pair)]:
                    raise ValueError("guessed pair conflicts with the knowledge mask")
                return GuessResult(
                    status="ok",
                    pair=pair,
                    raw_text=tuple(raw_text),
                    error=None,
                    attempts=attempt,
                    model=self.model,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        return GuessResult(
            status="failed",
            pair=None,
            raw_text=tuple(raw_text),
            error=error or "belief elicitation failed",
            attempts=2,
            model=self.model,
        )
