"""Agent interfaces for the structured-event environment."""

import random
from abc import ABC, abstractmethod


class Agent(ABC):
    def reset(self):
        return None

    @abstractmethod
    def act(self, observation):
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, observation):
        if "speech" in observation["phase"]:
            action_type = "speech_pk" if "speech_pk" in observation["phase"] else "speech"
            return action_type, ""
        actions = observation["valid_actions"]
        if not actions:
            raise RuntimeError("no valid action is available")
        return self.rng.choice(actions)
