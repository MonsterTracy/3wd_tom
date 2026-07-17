"""LLM agent that consumes only unified structured events."""

import json
import random
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
    gameplay_repair_message,
    prompt_reference,
    render_gameplay_phase_task,
    render_gameplay_user_message,
)
from werewolf.tom.guess_provider import BeliefGuessProvider


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
        del tokenizer
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.log_file = Path(log_file) if log_file else None
        self.rng = random.Random(seed)

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

    def format_observation(self, observation):
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
        return render_gameplay_user_message(
            player_id=player_id,
            role=role,
            phase=phase,
            day=day,
            alive_players=alive_players,
            information=information,
            valid_actions=observation["valid_actions"],
            phase_task=render_gameplay_phase_task(role, phase, variant),
        )

    def build_messages(self, observation):
        variant = self._variant(observation)
        return [
            {
                "role": "system",
                "content": build_gameplay_system_prompt(
                    observation["role"], variant
                ),
            },
            {"role": "user", "content": self.format_observation(observation)},
        ]

    @staticmethod
    def _parse_response(text, observation):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"response is not valid JSON: {exc.msg}") from exc
        if "speech" in observation["phase"]:
            if not isinstance(payload, dict) or set(payload) != {"speech"}:
                raise ValueError("speech response must contain only speech")
            if not isinstance(payload["speech"], str):
                raise ValueError("speech must be text")
            action_type = "speech_pk" if "speech_pk" in observation["phase"] else "speech"
            return action_type, payload["speech"]
        if not isinstance(payload, dict) or set(payload) != {"action_index"}:
            raise ValueError("action response must contain only action_index")
        index = payload["action_index"]
        actions = observation["valid_actions"]
        if type(index) is not int or not 0 <= index < len(actions):
            raise ValueError("action_index is outside valid_actions")
        return tuple(actions[index])

    def _fallback(self, observation):
        if "speech" in observation["phase"]:
            action_type = "speech_pk" if "speech_pk" in observation["phase"] else "speech"
            return action_type, ""
        actions = observation["valid_actions"]
        non_abstain = [action for action in actions if action[1] != 0]
        return self.rng.choice(non_abstain or actions)

    def _write_log(self, record):
        if self.log_file is None:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    def act(self, observation):
        responses = []
        error = None
        action = None
        attempts = 0
        messages = self.build_messages(observation)
        for attempt in range(1, 3):
            attempts = attempt
            if attempt == 2:
                messages.extend(
                    [
                        {"role": "assistant", "content": responses[-1] if responses else ""},
                        {
                            "role": "user",
                            "content": gameplay_repair_message(),
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
                action = self._parse_response(responses[-1], observation)
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if action is None:
            action = self._fallback(observation)
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
                "action": list(action),
            }
        )
        return action
