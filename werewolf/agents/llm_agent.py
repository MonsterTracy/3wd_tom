"""LLM agent that consumes only unified structured events."""

import json
import random
from pathlib import Path

from werewolf.agents.base_agent import Agent
from werewolf.backends.base import BackendError
from werewolf.events.streams import render_stream
from werewolf.prompt_protocol import GAMEPLAY_PROMPT_SPEC, prompt_reference
from werewolf.tom.guess_provider import BeliefGuessProvider


GAMEPLAY_SYSTEM_PROMPT = GAMEPLAY_PROMPT_SPEC["text"]


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

    def format_observation(self, observation):
        actions = observation["valid_actions"]
        speech_phase = "speech" in observation["phase"]
        output_instruction = (
            '发言阶段只返回如下 JSON：{"speech":"你的公开发言"}。'
            if speech_phase
            else (
                "从 valid_actions 中选择一个从零开始的索引，且只返回如下 JSON："
                '{"action_index":0}。'
            )
        )
        return "\n\n".join(
            part
            for part in (
                f"你是玩家 {observation['player_id']}，当前身份是 {observation['role']}。",
                f"当前阶段：{observation['phase']}",
                self.strategy_hint(observation),
                "当前可见事件：\n" + render_stream(observation["events"]),
                "valid_actions=" + json.dumps(actions, ensure_ascii=False),
                output_instruction,
            )
            if part
        )

    def build_messages(self, observation):
        return [
            {"role": "system", "content": GAMEPLAY_SYSTEM_PROMPT},
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
                            "content": "只返回符合当前阶段要求的有效 JSON，不要输出其他内容。",
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
