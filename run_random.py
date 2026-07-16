import random
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
import time
import argparse
import os
from copy import deepcopy
from werewolf.agents import agent_registry
from werewolf.backends import (
    create_backend,
    load_named_backends,
    resolve_backend,
)
from werewolf.models import SpeechPerceiver
from werewolf.models.twd_tom.collector import TWDToMSampleCollector
from werewolf.runtime_config import normalize_runtime_config
import yaml
import json


def eval(env, agent_list, roles_, sample_collector=None):
    print(agent_list)
    for agent in agent_list:
        agent.reset()
    done = False
    obs = env.reset(roles=roles_)
    step_idx = 0
    while not done:
        if sample_collector is not None:
            sample_collector.record(
                obs,
                roles_,
                step_idx=step_idx,
            )
        current_act_idx = obs['current_act_idx']
        action = agent_list[current_act_idx - 1].act(obs)
        obs, reward, done, info = env.step(action)
        step_idx += 1
    if done:
        if info['Werewolf'] == 1:
            return 'Werewolf win'
        elif info['Werewolf'] == -1:
            return 'Villager win'


def _weighted_profile_choice(profiles):
    if not profiles:
        raise ValueError("no eligible agent profiles")
    weights = [profile["sample_ratio"] for profile in profiles]
    if any(weight < 0 for weight in weights) or not any(
        weight > 0 for weight in weights
    ):
        raise ValueError(
            "eligible agent profiles must have a positive sample_ratio"
        )
    return random.choices(profiles, weights=weights, k=1)[0]


def assign_agents(candidate_profiles, env_config, log_save_path,
                  assigined_roles, must_include, backends):
    werewolf_team = ["Werewolf"]
    village_team = ["Villager", "Seer", "Witch", "Guard"]

    env_param = {
        "n_player": env_config["n_player"],
        "n_role": env_config["n_role"]
    }
    all_agent_profiles = {}
    role2agent_list = []
    village_profiles = set()
    werewolf_profiles = set()

    forced_profile = None
    if must_include:
        required_profiles = [
            profile
            for profile in candidate_profiles
            if profile["profile_name"] in must_include
        ]
        forced_profile = _weighted_profile_choice(required_profiles)

    for index, role in enumerate(assigined_roles):
        if index == 0 and forced_profile is not None:
            profile = forced_profile
        else:
            if role in werewolf_team:
                opposite_profiles = village_profiles
            elif role in village_team:
                opposite_profiles = werewolf_profiles
            else:
                raise ValueError(f"unsupported role: {role}")
            eligible_profiles = [
                profile
                for profile in candidate_profiles
                if profile["profile_name"] not in opposite_profiles
            ]
            if not eligible_profiles:
                raise ValueError(
                    f"no eligible agent profiles for role: {role}"
                )
            profile = _weighted_profile_choice(eligible_profiles)

        profile_name = profile["profile_name"]
        if role in werewolf_team:
            werewolf_profiles.add(profile_name)
        else:
            village_profiles.add(profile_name)

        if profile_name not in all_agent_profiles:
            model_params = dict(profile["model_params"])
            all_agent_profiles[profile_name] = agent_registry.build(
                profile["agent_type"],
                backend=resolve_backend(
                    profile["backend"],
                    backends,
                ),
                model_name=profile["model"],
                **model_params,
            )
        role2agent_list.append(profile_name)

    agent_list = []
    for i, role in enumerate(assigined_roles):
        log_file = (
            os.path.join(log_save_path, f"Player_{i + 1}.jsonl")
            if log_save_path is not None
            else None
        )
        profile_name = role2agent_list[i]
        type, agent_param = all_agent_profiles[profile_name]
        agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
        agent_list.append(agent)
    return role2agent_list, agent_list


def build_runtime(parsed_yaml, log_save_path, backend=None,
                  backend_settings=None, roles=None, random_seed=None,
                  backends=None):
    config_for_normalization = deepcopy(parsed_yaml)
    if (
        backend_settings is not None
        and "backend" in config_for_normalization
        and "backends" not in config_for_normalization
    ):
        legacy_backend = config_for_normalization["backend"]
        for field in (
            "default_model",
            "agent_model",
            "parser_model",
        ):
            value = getattr(backend_settings, field, None)
            if value is not None:
                legacy_backend.setdefault(field, value)

    normalized = normalize_runtime_config(
        config_for_normalization
    )
    agent_config = normalized["agent_config"]
    all_candidate_agents = agent_config["all_candidates"]
    env_config = normalized["env_config"]

    if backends is not None:
        backend_map = dict(backends)
    elif backend is not None:
        backend_names = list(normalized["backends"])
        if len(backend_names) != 1:
            raise ValueError(
                "a single injected backend requires exactly "
                "one configured backend"
            )
        backend_map = {backend_names[0]: backend}
    elif backend_settings is not None:
        backend_names = list(normalized["backends"])
        if len(backend_names) != 1:
            raise ValueError(
                "legacy backend_settings requires exactly "
                "one configured backend"
            )
        backend_map = {
            backend_names[0]: create_backend(backend_settings)
        }
    else:
        backend_map = load_named_backends(normalized)

    parser_config = normalized["parser"]
    speech_perceiver = SpeechPerceiver(
        backend=resolve_backend(
            parser_config["backend"],
            backend_map,
        ),
        model_name=parser_config["model"],
    )

    env_config["log_save_path"] = log_save_path
    env = WerewolfTextEnvV0(
        **env_config,
        speech_perceiver=speech_perceiver,
    )
    if env_config.get("n_hunter", 0) != 0:
        raise ValueError("TWDM 7-player environment does not support Hunter.")

    if random_seed is not None:
        random.seed(random_seed)
    if roles is None:
        roles = (
            ["Werewolf"] * env_config["n_werewolf"]
            + ["Villager"] * env_config["n_villager"]
            + ["Seer"] * env_config["n_seer"]
            + ["Witch"] * env_config["n_witch"]
            + ["Guard"] * env_config["n_guard"]
        )
        random.shuffle(roles)
    else:
        roles = list(roles)

    must_include = agent_config.get("must_include", [])
    role2agent_list, agent_list = assign_agents(
        all_candidate_agents,
        env_config,
        log_save_path,
        roles,
        must_include=must_include,
        backends=backend_map,
    )
    return env, agent_list, roles, role2agent_list


def main_cli(args):
    if args.log_save_path is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")
        args.log_save_path = os.path.join("logs", run_name)

    os.makedirs(args.log_save_path, exist_ok=True)
    parsed_yaml = yaml.safe_load(open(args.config))
    config_save_path = os.path.join(args.log_save_path, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(parsed_yaml, f, allow_unicode=True, sort_keys=False)
    env, agent_list, roles, role2agent_list = build_runtime(
        parsed_yaml,
        log_save_path=args.log_save_path,
    )
    print("New rollout: ", roles)

    print("\n\n")
    for r, a in zip(roles, role2agent_list):
        print(r, "\t", a)
    # make sure role2agent_list must have training model, or repeat

    assert len(roles) == len(role2agent_list), "The length of roles and role2agent_list must be the same"

    records = []
    for i in range(len(roles)):
        record = {
            "id": i + 1,
            "role": roles[i],
            "model": role2agent_list[i]
        }
        records.append(record)

    output_file = os.path.join(args.log_save_path, 'roles_model_assignment.json')
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=4)

    print(agent_list)
    begin = time.time()
    sample_collector = None
    sample_path = getattr(args, "twd_tom_sample_path", None)
    if sample_path is not None:
        game_id = os.path.basename(
            os.path.normpath(args.log_save_path)
        )
        sample_collector = TWDToMSampleCollector(
            output_path=sample_path,
            game_id=game_id,
        )
    try:
        result = eval(
            env,
            agent_list,
            roles,
            sample_collector=sample_collector,
        )
    finally:
        if sample_collector is not None:
            sample_collector.close()
    print(time.time() - begin, result)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config',
                           type=str, default="configs/random_models.yaml",
                           help="path to the config file of the game")
    argparser.add_argument('--log_save_path', type=str, default=None)
    argparser.add_argument(
        '--twd_tom_sample_path',
        type=str,
        default=None,
    )
    args = argparser.parse_args()
    main_cli(args)
