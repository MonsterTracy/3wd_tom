import unittest

import torch
from torch import nn

from werewolf.models.twd_tom.backbone import ToMBackboneConfig
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.features import TWDToMFeatureBuilder
from werewolf.models.twd_tom.losses import twd_tom_loss
from werewolf.models.twd_tom.model import TWDToMConfig, TWDToMModel


class DummyFeatureBuilder:
    def __init__(self):
        self.observations = []

    def encode_observation(self, observation):
        self.observations.append(observation)
        length = observation["length"]
        value = observation.get("value", 1)
        return {
            "event_tokens": torch.full(
                (1, length, 10),
                value,
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                1,
                length,
                dtype=torch.long,
            ),
        }


def make_sample(length, game_id, observer_id, phase, labels=None):
    return {
        "game_id": game_id,
        "observer_id": observer_id,
        "phase": phase,
        "observation": {
            "length": length,
            "value": 1,
        },
        "wolf_labels": (
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if labels is None
            else labels
        ),
    }


class TWDToMDatasetTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            make_sample(2, "game-1", 1, "day"),
            make_sample(
                4,
                "game-2",
                2,
                "night",
                labels=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ),
        ]
        self.feature_builder = DummyFeatureBuilder()
        self.dataset = TWDToMDataset(
            self.samples,
            feature_builder=self.feature_builder,
        )

    def test_len_returns_sample_count(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem_returns_expected_fields(self):
        item = self.dataset[0]

        self.assertEqual(
            set(item),
            {
                "event_tokens",
                "attention_mask",
                "wolf_labels",
                "game_id",
                "observer_id",
                "phase",
                "alive_mask",
            },
        )

    def test_getitem_removes_feature_batch_dimension(self):
        item = self.dataset[0]

        self.assertEqual(item["event_tokens"].shape, (2, 10))
        self.assertEqual(item["attention_mask"].shape, (2,))
        self.assertEqual(item["event_tokens"].dtype, torch.long)
        self.assertEqual(item["attention_mask"].dtype, torch.long)

    def test_getitem_creates_float32_player_labels(self):
        labels = self.dataset[0]["wolf_labels"]

        self.assertEqual(labels.shape, (7,))
        self.assertEqual(labels.dtype, torch.float32)
        torch.testing.assert_close(
            labels,
            torch.tensor(
                [1, 1, 0, 0, 0, 0, 0],
                dtype=torch.float32,
            ),
        )

    def test_getitem_does_not_expose_roles(self):
        self.assertNotIn("roles", self.dataset[0])

    def test_custom_feature_builder_receives_original_observation(self):
        self.dataset[1]

        self.assertIs(
            self.feature_builder.observations[0],
            self.samples[1]["observation"],
        )

    def test_default_feature_builder_is_created_when_omitted(self):
        dataset = TWDToMDataset([])

        self.assertIsInstance(
            dataset.feature_builder,
            TWDToMFeatureBuilder,
        )

    def test_collate_supports_different_sequence_lengths(self):
        batch = collate_twd_tom_samples(
            [self.dataset[0], self.dataset[1]]
        )

        self.assertIn("observer_id", batch)
        self.assertEqual(batch["event_tokens"].shape, (2, 4, 10))
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(batch["observer_id"].shape, (2,))
        self.assertEqual(batch["observer_id"].tolist(), [1, 2])
        self.assertEqual(batch["wolf_labels"].shape, (2, 7))

    def test_legacy_observer_field_is_collated(self):
        sample = make_sample(2, "game-legacy", 1, "day")
        del sample["observer_id"]
        sample["observer"] = 6
        dataset = TWDToMDataset(
            [sample],
            feature_builder=DummyFeatureBuilder(),
        )

        batch = collate_twd_tom_samples([dataset[0]])

        self.assertIn("observer_id", batch)
        self.assertEqual(batch["observer_id"].shape, (1,))
        self.assertEqual(batch["observer_id"].tolist(), [6])
        self.assertEqual(batch["metadata"]["observer_id"], [6])

    def test_collate_alive_mask_and_default_all_alive(self):
        self.samples[0]["alive_mask"] = [1, 1, 0, 1, 0, 1, 1]

        batch = collate_twd_tom_samples(
            [self.dataset[0], self.dataset[1]]
        )

        self.assertIn("alive_mask", batch)
        self.assertEqual(batch["alive_mask"].shape, (2, 7))
        self.assertEqual(batch["alive_mask"].dtype, torch.float32)
        self.assertEqual(
            batch["alive_mask"].tolist(),
            [
                [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                [1.0] * 7,
            ],
        )
        self.assertEqual(batch["event_tokens"].shape, (2, 4, 10))
        self.assertEqual(batch["observer_id"].tolist(), [1, 2])

    def test_collate_right_padding_is_all_zero(self):
        batch = collate_twd_tom_samples(
            [self.dataset[0], self.dataset[1]]
        )

        self.assertEqual(
            batch["event_tokens"][0, 2:].count_nonzero().item(),
            0,
        )
        self.assertEqual(
            batch["attention_mask"][0].tolist(),
            [1, 1, 0, 0],
        )
        self.assertEqual(
            batch["attention_mask"][1].tolist(),
            [1, 1, 1, 1],
        )

    def test_collate_stacks_float32_labels(self):
        batch = collate_twd_tom_samples(
            [self.dataset[0], self.dataset[1]]
        )

        self.assertEqual(batch["wolf_labels"].shape, (2, 7))
        self.assertEqual(batch["wolf_labels"].dtype, torch.float32)

    def test_collate_preserves_metadata(self):
        batch = collate_twd_tom_samples(
            [self.dataset[0], self.dataset[1]]
        )

        self.assertEqual(
            batch["metadata"],
            {
                "game_id": ["game-1", "game-2"],
                "observer_id": [1, 2],
                "phase": ["day", "night"],
                "alive_mask": [[1.0] * 7, [1.0] * 7],
            },
        )

    def test_batch_can_feed_model_and_loss(self):
        batch = collate_twd_tom_samples(
            [self.dataset[0], self.dataset[1]]
        )
        config = TWDToMConfig(
            tom_config=ToMBackboneConfig(
                num_players=7,
                d_model=32,
                n_head=4,
                n_layer=1,
                dropout=0.0,
                max_seq_len=8,
                max_day=5,
            )
        )
        model = TWDToMModel(config)

        outputs = model(
            batch["event_tokens"],
            attention_mask=batch["attention_mask"],
            observer_id=batch["observer_id"],
        )
        losses = twd_tom_loss(
            outputs,
            batch["wolf_labels"],
            attention_mask=batch["attention_mask"],
        )

        self.assertEqual(outputs["wolf_prob"].shape, (2, 4, 7))
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_dataset_owns_no_model_embedding_or_parameter(self):
        self.assertNotIsInstance(self.dataset, nn.Module)
        self.assertFalse(hasattr(self.dataset, "parameters"))
        for value in vars(self.dataset).values():
            self.assertNotIsInstance(value, nn.Module)
            self.assertNotIsInstance(value, nn.Embedding)
            self.assertNotIsInstance(value, nn.Parameter)


if __name__ == "__main__":
    unittest.main()
