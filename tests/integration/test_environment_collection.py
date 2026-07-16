import random

import torch

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.events.speech_parser import SpeechEventParser
from werewolf.tom.collection import ToMCollector
from werewolf.tom.guess_provider import GuessResult
from werewolf.tom.features import collate_features, sample_to_features
from werewolf.tom.losses import masked_pair_cross_entropy
from werewolf.tom.model import ToMModel, ToMModelConfig
from werewolf.tom.pair_space import WOLF_PAIRS


ROLES = ["Werewolf", "Werewolf", "Seer", "Witch", "Villager", "Villager", "Villager"]


class EmptyParserBackend:
    def chat(self, messages, **kwargs):
        return '{"events":[]}'


class ValidGuessProvider:
    def elicit(self, *, player_view, output_mask):
        pair = next(pair for pair, allowed in zip(WOLF_PAIRS, output_mask) if allowed)
        return GuessResult(
            status="ok", pair=pair, raw_text=(f'{{"wolf_pair":{list(pair)}}}',),
            error=None, attempts=1, model="fake"
        )


def _rollout(environment):
    rng = random.Random(11)
    observation = environment.reset(roles=ROLES)
    for _ in range(300):
        if "speech" in observation["phase"]:
            action = (environment.phase, f"player {observation['player_id']} statement")
        else:
            action = rng.choice(observation["valid_actions"])
        observation, _, done, info = environment.step(action)
        if done:
            return info
    raise AssertionError("game did not finish")


def test_full_game_uses_unified_events_and_checkpoint_collection():
    collector = ToMCollector(
        game_id="integration", roles=ROLES,
        guess_provider_for=lambda player_id: ValidGuessProvider()
    )
    environment = WerewolfTextEnvV0(
        random_seed=11,
        speech_parser=SpeechEventParser(EmptyParserBackend(), "fake-parser"),
        tom_collector=collector,
    )
    info = _rollout(environment)
    assert info["Werewolf"] in (-1, 1)
    assert all(event["event_family"] in {
        "BELIEF_ASSERTION", "SOCIAL_STANCE", "ACTION_POSITION", "CLAIM_RESPONSE",
        "GAME_EVENT", "PRIVATE_FACT"
    } for event in environment.events)
    assert all(
        event["content"]["kind"] != "CHECK_RESULT"
        or "camp" in event["content"]["value"]
        for event in environment.events
    )

    samples = [sample for batch in environment.collection_batches for sample in batch.samples]
    assert samples
    assert all(
        not (sample["task"] == "first_order" and ROLES[sample["observer_id"] - 1] == "Werewolf")
        for sample in samples
    )
    assert all(
        sum(sample["output_mask"]) == 21
        for sample in samples
        if sample["task"] == "second_order" and sample["mode"] == "public_only"
    )
    assert all(
        sum(sample["output_mask"]) == 15
        for sample in samples
        if sample["task"] == "second_order" and sample["mode"] == "wolf_conditioned"
    )
    second_order_smoke = [
        next(sample for sample in samples if sample["mode"] == mode)
        for mode in ("public_only", "wolf_conditioned")
    ]
    batch = collate_features([sample_to_features(sample) for sample in second_order_smoke])
    model = ToMModel(
        ToMModelConfig(
            architecture="boe_mlp", d_model=8, num_layers=1,
            num_heads=2, max_events=512
        )
    )
    logits = model(batch)
    loss = masked_pair_cross_entropy(logits, batch["labels"], batch["output_mask"])
    assert logits.shape == (2, 21)
    assert torch.isfinite(loss)


def test_guess_failure_does_not_stop_the_game_or_enter_samples():
    class FailureProvider:
        def elicit(self, **kwargs):
            return GuessResult(
                status="failed", pair=None, raw_text=("bad", "bad"),
                error="invalid", attempts=2, model="fake"
            )

    collector = ToMCollector(
        game_id="failures", roles=ROLES,
        guess_provider_for=lambda player_id: FailureProvider()
    )
    environment = WerewolfTextEnvV0(random_seed=4, tom_collector=collector)
    _rollout(environment)
    assert any(batch.failures for batch in environment.collection_batches)
    assert not any(batch.samples for batch in environment.collection_batches)
