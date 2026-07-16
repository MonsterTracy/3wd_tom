"""General OpenAI-compatible gameplay agent."""

from werewolf.agents import agent_registry
from werewolf.agents.llm_agent import LLMAgent


@agent_registry.register(
    ["gpt", "gpt-4", "GPT-4", "gpt4", "o1", "gpt4o", "gpt4o-mini", "deepseek"]
)
class GPTAgent(LLMAgent):
    pass
