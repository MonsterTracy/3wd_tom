"""Strategy-guided gameplay agent using the same structured event stream."""

from werewolf.agents import agent_registry
from werewolf.agents.llm_agent import LLMAgent
from werewolf.agents.twdm_strategy import TWDMStrategy


@agent_registry.register(["twdm_agent"])
class TWDMStrategyAgent(LLMAgent):
    def __init__(self, *args, twdm_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = TWDMStrategy(twdm_config or {})

    def strategy_hint(self, observation):
        return self.strategy.build_hint(observation)
