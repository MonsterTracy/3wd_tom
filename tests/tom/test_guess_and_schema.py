import json
from copy import deepcopy
from pathlib import Path

import pytest

from werewolf.tom.dataset import ToMDataset
from werewolf.tom.guess_provider import BeliefGuessProvider
from werewolf.tom.schemas import validate_sample


FIXTURE = Path(__file__).parents[1] / "fixtures" / "tom_v1.jsonl"


class SequenceBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return next(self.responses)


def test_guess_provider_repairs_once_without_defaulting():
    backend = SequenceBackend(["bad", '{"wolf_pair":[2,1]}'])
    result = BeliefGuessProvider(backend, "agent-model").elicit(
        player_view="view", output_mask=[True] * 21
    )
    assert result.status == "ok"
    assert result.pair == (1, 2)
    assert result.attempts == 2
    assert len(result.raw_text) == 2


def test_guess_provider_preserves_failure_and_does_not_autocorrect():
    mask = [True] + [False] * 20
    backend = SequenceBackend(['{"wolf_pair":[6,7]}', '{"wolf_pair":[6,7]}'])
    result = BeliefGuessProvider(backend, "agent-model").elicit(
        player_view="view", output_mask=mask
    )
    assert result.status == "failed"
    assert result.pair is None
    assert "knowledge mask" in result.error
    assert len(result.raw_text) == 2


def test_dataset_accepts_only_successful_tom_v1(tmp_path):
    dataset = ToMDataset(FIXTURE)
    assert len(dataset) == 2
    assert dataset[0]["output_mask"].shape == (21,)
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({"schema_version": "legacy.v0"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy samples are rejected|fields"):
        ToMDataset(legacy)


def test_schema_rejects_private_leakage_in_public_second_order():
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    leaked = deepcopy(records[1])
    leaked["events"].append(deepcopy(records[0]["events"][1]))
    with pytest.raises(ValueError, match="public-only"):
        validate_sample(leaked)
