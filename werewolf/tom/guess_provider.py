"""Side-effect-free belief elicitation using an agent's LLM backend."""

import json
from dataclasses import dataclass

from werewolf.prompt_protocol import BELIEF_PROMPT_SPEC
from werewolf.tom.pair_space import PLAYER_IDS, pair_index, validate_pair_mask


SYSTEM_PROMPT = BELIEF_PROMPT_SPEC["text"]
GUESS_ERROR_CODES = {
    "backend_error",
    "invalid_json",
    "not_exactly_two_players",
    "duplicate_players",
    "out_of_range",
    "missing_required_wolf",
    "contains_forbidden_player",
    "label_outside_mask",
}


class GuessValidationError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GuessResult:
    status: str
    pair: tuple[int, int] | None
    raw_text: tuple[str, ...]
    error: str | None
    attempts: int
    model: str | None
    first_error_code: str | None
    final_error_code: str | None
    required_wolves: tuple[int, ...]
    forbidden_wolves: tuple[int, ...]


def _normalize_constraint(values, name):
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{name} must be a sequence of player ids")
    normalized = tuple(sorted(set(values)))
    if any(type(value) is not int or value not in PLAYER_IDS for value in normalized):
        raise ValueError(f"{name} must contain player ids between 1 and 7")
    return normalized


def _validate_guess_response(
    text,
    *,
    valid_mask,
    required_wolves,
    forbidden_wolves,
) -> tuple[int, int]:
    if not isinstance(text, str) or not text.strip():
        raise GuessValidationError("invalid_json", "回复不是有效 JSON")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GuessValidationError("invalid_json", "回复不是有效 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"wolf_pair"}:
        raise GuessValidationError(
            "invalid_json", "回复必须是只包含 wolf_pair 的 JSON 对象"
        )
    values = payload["wolf_pair"]
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise GuessValidationError(
            "not_exactly_two_players", "wolf_pair 必须恰好包含两名玩家"
        )
    if any(type(value) is not int or value not in PLAYER_IDS for value in values):
        raise GuessValidationError(
            "out_of_range", "玩家编号必须是 1 到 7 的整数"
        )
    if values[0] == values[1]:
        raise GuessValidationError(
            "duplicate_players", "wolf_pair 中的两名玩家不能相同"
        )
    pair = tuple(sorted(values))
    missing = sorted(set(required_wolves) - set(pair))
    if missing:
        raise GuessValidationError(
            "missing_required_wolf", f"组合缺少已知必须包含的狼人：{missing}"
        )
    forbidden = sorted(set(forbidden_wolves) & set(pair))
    if forbidden:
        raise GuessValidationError(
            "contains_forbidden_player", f"组合包含已知不是狼人的玩家：{forbidden}"
        )
    if not valid_mask[pair_index(pair)]:
        raise GuessValidationError(
            "label_outside_mask", "组合不在当前知识约束允许的标签范围内"
        )
    return pair


def parse_guess_response(text: str) -> tuple[int, int]:
    """Parse the strict wire format without adding game-specific constraints."""

    return _validate_guess_response(
        text,
        valid_mask=(True,) * 21,
        required_wolves=(),
        forbidden_wolves=(),
    )


def _user_message(*, observer_id, player_view, required_wolves, forbidden_wolves):
    required = json.dumps(list(required_wolves), ensure_ascii=False)
    forbidden = json.dumps(list(forbidden_wolves), ensure_ascii=False)
    return (
        f"当前被测玩家：{observer_id}号\n\n"
        "必须遵守的硬约束：\n"
        "- 必须选择两名不同玩家；\n"
        f"- 禁止选择的玩家：{forbidden}\n"
        f"- 已知必须包含的狼人：{required}\n"
        f"- 已知不是狼人的玩家：{forbidden}\n\n"
        "当前合法视角：\n"
        f"{player_view}"
    )


def _repair_message(error_code, *, required_wolves, forbidden_wolves):
    required = json.dumps(list(required_wolves), ensure_ascii=False)
    forbidden = json.dumps(list(forbidden_wolves), ensure_ascii=False)
    reasons = {
        "invalid_json": "上一条结果非法：回复不是规定的 JSON 对象。",
        "not_exactly_two_players": "上一条结果非法：必须恰好选择两名玩家。",
        "duplicate_players": "上一条结果非法：两名玩家不能相同。",
        "out_of_range": "上一条结果非法：玩家编号必须是 1 到 7 的整数。",
        "missing_required_wolf": (
            "上一条结果非法：组合遗漏了已知必须包含的狼人。"
            f"\n已知必须包含的狼人为：{required}。"
        ),
        "contains_forbidden_player": (
            "上一条结果非法：组合包含了已知不是狼人的玩家。"
            f"\n禁止选择的玩家为：{forbidden}。"
        ),
        "label_outside_mask": "上一条结果非法：组合不满足当前硬约束。",
    }
    reason = reasons.get(error_code, "上一条请求未成功完成。")
    return (
        f"{reason}\n"
        f"已知必须包含的狼人：{required}。\n"
        f"禁止选择的玩家：{forbidden}。\n"
        "请只返回满足这些硬约束的合法 JSON。"
    )


class BeliefGuessProvider:
    """Elicit a label without mutating the game agent or environment."""

    def __init__(self, backend, model=None, *, max_tokens=40):
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens

    def elicit(
        self,
        *,
        observer_id: int,
        player_view: str,
        output_mask,
        required_wolves,
        forbidden_wolves,
    ) -> GuessResult:
        valid_mask = validate_pair_mask(output_mask)
        required_wolves = _normalize_constraint(required_wolves, "required_wolves")
        forbidden_wolves = _normalize_constraint(forbidden_wolves, "forbidden_wolves")
        if set(required_wolves) & set(forbidden_wolves):
            raise ValueError("required_wolves and forbidden_wolves must be disjoint")
        raw_text = []
        error = None
        first_error_code = None
        final_error_code = None
        initial_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _user_message(
                    observer_id=observer_id,
                    player_view=player_view,
                    required_wolves=required_wolves,
                    forbidden_wolves=forbidden_wolves,
                ),
            },
        ]
        for attempt in range(1, 3):
            messages = [dict(message) for message in initial_messages]
            if attempt == 2 and raw_text:
                messages.extend(
                    [
                        {"role": "assistant", "content": raw_text[-1]},
                        {
                            "role": "user",
                            "content": _repair_message(
                                first_error_code,
                                required_wolves=required_wolves,
                                forbidden_wolves=forbidden_wolves,
                            ),
                        },
                    ]
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
                pair = _validate_guess_response(
                    raw_text[-1],
                    valid_mask=valid_mask,
                    required_wolves=required_wolves,
                    forbidden_wolves=forbidden_wolves,
                )
                return GuessResult(
                    status="ok",
                    pair=pair,
                    raw_text=tuple(raw_text),
                    error=None,
                    attempts=attempt,
                    model=self.model,
                    first_error_code=first_error_code,
                    final_error_code=None,
                    required_wolves=required_wolves,
                    forbidden_wolves=forbidden_wolves,
                )
            except GuessValidationError as exc:
                final_error_code = exc.code
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                final_error_code = "backend_error"
                error = f"{type(exc).__name__}: {exc}"
            if attempt == 1:
                first_error_code = final_error_code
        return GuessResult(
            status="failed",
            pair=None,
            raw_text=tuple(raw_text),
            error=error or "belief elicitation failed",
            attempts=2,
            model=self.model,
            first_error_code=first_error_code,
            final_error_code=final_error_code,
            required_wolves=required_wolves,
            forbidden_wolves=forbidden_wolves,
        )
