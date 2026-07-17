"""LLM agent that consumes only unified structured events."""

import json
from pathlib import Path

from werewolf.agents.base_agent import Agent
from werewolf.backends.base import BackendError
from werewolf.events.streams import (
    alive_players_from_events,
    render_information_partitions,
)
from werewolf.game_rules import variant_from_role_counts
from werewolf.prompt_protocol import (
    GAMEPLAY_PROMPT_SPEC,
    build_gameplay_system_prompt,
    gameplay_action_options,
    gameplay_repair_message,
    prompt_reference,
    render_gameplay_phase_task,
    render_gameplay_user_message,
)
from werewolf.tom.guess_provider import BeliefGuessProvider


GAMEPLAY_VALIDATION_ERROR_CODES = {
    "invalid_json",
    "wrong_fields",
    "speech_not_text",
    "action_index_not_integer",
    "action_index_out_of_range",
}


class GameplayValidationError(ValueError):
    def __init__(self, code, message, *, invalid_action_index=None):
        if code not in GAMEPLAY_VALIDATION_ERROR_CODES:
            raise ValueError(f"unknown gameplay validation error code: {code!r}")
        super().__init__(message)
        self.code = code
        self.invalid_action_index = invalid_action_index


class LLMAgent(Agent):
    def __init__(
        self,
        backend=None,
        model_name=None,
        tokenizer=None,
        temperature=0.7,
        log_file=None,
        seed=None,
    ):
        del tokenizer, seed
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.log_file = Path(log_file) if log_file else None

    def _chat(self, messages, **kwargs):
        if self.backend is None or not self.model_name:
            raise BackendError("agent backend and model_name are required")
        return self.backend.chat(messages=messages, model=self.model_name, **kwargs)

    def make_guess_provider(self):
        return BeliefGuessProvider(self.backend, self.model_name)

    def strategy_hint(self, observation):
        del observation
        return ""

    @staticmethod
    def _variant(observation):
        for event in observation["events"]:
            if (
                event["event_family"] == "GAME_EVENT"
                and event["content"]["kind"] == "SETTING"
            ):
                return variant_from_role_counts(event["metadata"]["roles"])
        if observation["role"] == "Guard":
            return "seer_guard"
        return "seer_witch"

    def format_observation(
        self,
        observation,
        *,
        valid_actions_snapshot=None,
        valid_action_options=None,
    ):
        phase = observation["phase"]
        role = observation["role"]
        player_id = observation["player_id"]
        variant = self._variant(observation)
        day = observation.get("day")
        if day is None:
            prefix = phase.split("_", 1)[0]
            day = int(prefix) if prefix.isdigit() else 0
        alive_players = observation.get(
            "alive_players", alive_players_from_events(observation["events"])
        )
        information = render_information_partitions(
            observation["events"], player_id=player_id
        )
        if valid_actions_snapshot is None:
            valid_actions_snapshot = tuple(
                tuple(action) for action in observation["valid_actions"]
            )
        if valid_action_options is None:
            valid_action_options = gameplay_action_options(
                valid_actions_snapshot
            )
        return render_gameplay_user_message(
            player_id=player_id,
            role=role,
            phase=phase,
            day=day,
            alive_players=alive_players,
            information=information,
            valid_action_options=valid_action_options,
            phase_task=render_gameplay_phase_task(role, phase, variant),
        )

    def build_messages(
        self,
        observation,
        *,
        valid_actions_snapshot=None,
        valid_action_options=None,
    ):
        variant = self._variant(observation)
        return [
            {
                "role": "system",
                "content": build_gameplay_system_prompt(
                    observation["role"], variant
                ),
            },
            {
                "role": "user",
                "content": self.format_observation(
                    observation,
                    valid_actions_snapshot=valid_actions_snapshot,
                    valid_action_options=valid_action_options,
                ),
            },
        ]

    @staticmethod
    def _parse_response(text, observation, *, valid_actions_snapshot=None):
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise GameplayValidationError(
                "invalid_json", "response is not valid JSON"
            ) from exc
        if "speech" in observation["phase"]:
            if not isinstance(payload, dict) or set(payload) != {"speech"}:
                raise GameplayValidationError(
                    "wrong_fields", "speech response must contain only speech"
                )
            if not isinstance(payload["speech"], str):
                raise GameplayValidationError(
                    "speech_not_text", "speech must be text"
                )
            action_type = "speech_pk" if "speech_pk" in observation["phase"] else "speech"
            return action_type, payload["speech"]
        if not isinstance(payload, dict) or set(payload) != {"action_index"}:
            raise GameplayValidationError(
                "wrong_fields", "action response must contain only action_index"
            )
        index = payload["action_index"]
        actions = (
            tuple(tuple(action) for action in observation["valid_actions"])
            if valid_actions_snapshot is None
            else valid_actions_snapshot
        )
        if type(index) is not int:
            raise GameplayValidationError(
                "action_index_not_integer", "action_index must be an integer"
            )
        if not 0 <= index < len(actions):
            raise GameplayValidationError(
                "action_index_out_of_range",
                "action_index is outside valid_actions",
                invalid_action_index=index,
            )
        return tuple(actions[index])

    def _write_log(self, record):
        if self.log_file is None:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def act(self, observation):
        responses = []
        error = None
        error_code = None
        first_error_code = None
        first_invalid_action_index = None
        action = None
        attempts = 0
        valid_actions_snapshot = tuple(
            tuple(action) for action in observation["valid_actions"]
        )
        valid_action_options = gameplay_action_options(
            valid_actions_snapshot
        )
        initial_messages = self.build_messages(
            observation,
            valid_actions_snapshot=valid_actions_snapshot,
            valid_action_options=valid_action_options,
        )
        for attempt in range(1, 3):
            attempts = attempt
            messages = [dict(message) for message in initial_messages]
            if attempt == 2 and first_error_code != "backend_error":
                messages.extend(
                    [
                        {"role": "assistant", "content": responses[-1]},
                        {
                            "role": "user",
                            "content": gameplay_repair_message(
                                first_error_code,
                                phase=observation["phase"],
                                valid_action_options=valid_action_options,
                                invalid_action_index=first_invalid_action_index,
                            ),
                        },
                    ]
                )
            try:
                response = self._chat(
                    messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                responses.append(response if isinstance(response, str) else str(response))
                action = self._parse_response(
                    responses[-1],
                    observation,
                    valid_actions_snapshot=valid_actions_snapshot,
                )
                error = None
                error_code = None
                break
            except GameplayValidationError as exc:
                error_code = exc.code
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    first_invalid_action_index = exc.invalid_action_index
            except BackendError as exc:
                error_code = "backend_error"
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 1 and not exc.retryable:
                    first_error_code = error_code
                    break
            except Exception as exc:
                backend_error = BackendError(
                    "Gameplay backend raised an unexpected exception.",
                    retryable=False,
                    details={"cause_type": type(exc).__name__},
                )
                error_code = "backend_error"
                error = f"{type(backend_error).__name__}: {backend_error}"
                if attempt == 1:
                    first_error_code = error_code
                    break
            if attempt == 1:
                first_error_code = error_code
        self._write_log(
            {
                "player_id": observation["player_id"],
                "role": observation["role"],
                "phase": observation["phase"],
                "gameplay_prompt": prompt_reference(GAMEPLAY_PROMPT_SPEC),
                "model": self.model_name,
                "temperature": self.temperature,
                "attempts": attempts,
                "responses": responses,
                "error": error,
                "error_code": error_code,
                "action": list(action) if action is not None else None,
                "valid_action_options": valid_action_options,
            }
        )
        if action is None:
            raise RuntimeError(
                f"gameplay action generation failed: {error_code or 'backend_error'}"
            )
        return action
