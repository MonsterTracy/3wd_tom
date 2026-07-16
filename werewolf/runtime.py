"""Canonical runtime assembly for gameplay and ToM collection."""

import random
from pathlib import Path

from werewolf.agents import agent_registry
from werewolf.backends import load_named_backends, resolve_backend
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.events.speech_parser import SpeechEventParser
from werewolf.runtime_config import resolve_guess_config, validate_runtime_config
from werewolf.tom.collection import JsonlSink, ToMCollector
from werewolf.tom.guess_provider import BeliefGuessProvider


def shuffled_roles(environment_config, seed):
    roles = (
        ["Werewolf"] * environment_config["n_werewolf"]
        + ["Seer"] * environment_config["n_seer"]
        + ["Witch"] * environment_config["n_witch"]
        + ["Guard"] * environment_config["n_guard"]
        + ["Villager"] * environment_config["n_villager"]
    )
    random.Random(seed).shuffle(roles)
    return roles


def build_agents(config, *, roles, backends, log_directory):
    profiles = config["agents"]["profiles"]
    agents = []
    for player_id, role in enumerate(roles, start=1):
        profile_name = (
            config["agents"]["werewolf_profile"]
            if role == "Werewolf"
            else config["agents"]["village_profile"]
        )
        profile = profiles[profile_name]
        kwargs = {
            "backend": resolve_backend(profile["backend"], backends),
            "model_name": profile["model"],
            "temperature": profile["temperature"],
            "log_file": str(Path(log_directory) / f"player_{player_id}.jsonl"),
        }
        if profile["agent_type"] == "twdm_agent":
            kwargs["twdm_config"] = profile["strategy"]
        agents.append(agent_registry.create(profile["agent_type"], **kwargs))
    return agents


def build_collection_runtime(config, *, game_id, roles, backends=None):
    validate_runtime_config(config)
    backends = backends or load_named_backends(config)
    log_directory = Path(config["output"]["logs"]) / game_id
    agents = build_agents(
        config,
        roles=roles,
        backends=backends,
        log_directory=log_directory,
    )
    parser_config = config["parser"]
    parser = SpeechEventParser(
        resolve_backend(parser_config["backend"], backends),
        parser_config["model"],
    )

    def guess_provider_for(player_id):
        profile_name = (
            config["agents"]["werewolf_profile"]
            if roles[player_id - 1] == "Werewolf"
            else config["agents"]["village_profile"]
        )
        resolved = resolve_guess_config(
            config, config["agents"]["profiles"][profile_name]
        )
        return BeliefGuessProvider(
            resolve_backend(resolved["backend"], backends), resolved["model"]
        )

    collector = ToMCollector(
        game_id=game_id,
        roles=roles,
        guess_provider_for=guess_provider_for,
        sample_sink=JsonlSink(config["output"]["samples"]),
        failure_sink=JsonlSink(config["output"]["failures"]),
    )
    environment_kwargs = dict(config["environment"])
    environment_kwargs.update(
        random_seed=config["seed"],
        log_save_path=log_directory,
        speech_parser=parser,
        tom_collector=collector,
    )
    environment = WerewolfTextEnvV0(**environment_kwargs)
    return environment, agents


def rollout(environment, agents, roles, *, max_steps):
    for agent in agents:
        agent.reset()
    observation = environment.reset(roles=roles)
    for step in range(max_steps):
        player_id = observation["player_id"]
        action = agents[player_id - 1].act(observation)
        observation, reward, done, info = environment.step(action)
        if done:
            return {
                "steps": step + 1,
                "winner": "Werewolf" if info["Werewolf"] == 1 else "Village",
                "reward": reward,
            }
    raise RuntimeError(f"rollout exceeded max_steps={max_steps}")
