import inspect
import unittest

import torch

from werewolf.encoding.event_encoder import (
    EVENT_TOKEN_FIELDS,
    encode_observation_game_log,
)
from werewolf.models.twd_tom.backbone import ToMBackboneConfig
from werewolf.models.twd_tom.features import TWDToMFeatureBuilder
from werewolf.models.twd_tom.model import TWDToMConfig, TWDToMModel


def vote_log(day, source, target):
    return {
        "day": day,
        "time": "vote",
        "event": "vote",
        "source": source,
        "target": target,
        "content": {},
    }


def observation_with_votes(count):
    return {
        "game_log": [
            vote_log(day=index + 1, source=index + 1, target=index + 2)
            for index in range(count)
        ]
    }


class TWDToMFeatureBuilderTest(unittest.TestCase):
    def test_single_observation_uses_event_encoder_field_order(self):
        observation = observation_with_votes(1)
        expected_token = encode_observation_game_log(observation)[0]

        features = TWDToMFeatureBuilder().encode_observation(observation)

        self.assertEqual(features["event_tokens"].shape, (1, 1, 10))
        self.assertEqual(features["attention_mask"].shape, (1, 1))
        self.assertEqual(
            features["event_tokens"][0, 0].tolist(),
            [expected_token[field] for field in EVENT_TOKEN_FIELDS],
        )

    def test_batch_right_pads_to_longest_sequence(self):
        features = TWDToMFeatureBuilder().encode_batch(
            [observation_with_votes(1), observation_with_votes(2)]
        )

        self.assertEqual(features["event_tokens"].shape, (2, 2, 10))
        self.assertEqual(features["attention_mask"].shape, (2, 2))
        self.assertEqual(features["attention_mask"].tolist(), [[1, 0], [1, 1]])

    def test_event_token_last_dimension_is_ten(self):
        features = TWDToMFeatureBuilder().encode_batch(
            [observation_with_votes(1), observation_with_votes(2)]
        )

        self.assertEqual(features["event_tokens"].shape[-1], 10)

    def test_batch_includes_one_based_observer_ids(self):
        first = observation_with_votes(1)
        first["observer_id"] = 1
        second = observation_with_votes(2)
        second["observer"] = 7
        third = {"game_log": []}

        features = TWDToMFeatureBuilder().encode_batch(
            [first, second, third]
        )

        self.assertIn("observer_id", features)
        self.assertEqual(features["observer_id"].shape, (3,))
        self.assertEqual(features["observer_id"].tolist(), [1, 7, 0])
        self.assertEqual(features["event_tokens"].shape, (3, 2, 10))

    def test_padding_tokens_are_all_zero(self):
        features = TWDToMFeatureBuilder().encode_batch(
            [observation_with_votes(1), observation_with_votes(2)]
        )

        self.assertTrue(
            torch.equal(
                features["event_tokens"][0, 1],
                torch.zeros(10, dtype=torch.long),
            )
        )

    def test_attention_mask_marks_valid_and_padding_tokens(self):
        features = TWDToMFeatureBuilder().encode_batch(
            [observation_with_votes(1), observation_with_votes(2)]
        )

        self.assertEqual(features["attention_mask"].tolist(), [[1, 0], [1, 1]])

    def test_truncation_keeps_most_recent_events(self):
        observation = observation_with_votes(4)
        expected = encode_observation_game_log(observation)[-2:]
        expected_rows = [
            [token[field] for field in EVENT_TOKEN_FIELDS]
            for token in expected
        ]

        features = TWDToMFeatureBuilder(max_seq_len=2).encode_observation(
            observation
        )

        self.assertEqual(features["event_tokens"][0].tolist(), expected_rows)
        self.assertEqual(features["attention_mask"].tolist(), [[1, 1]])

    def test_empty_observation_returns_nonempty_padding_sequence(self):
        features = TWDToMFeatureBuilder().encode_observation({"game_log": []})

        self.assertEqual(features["event_tokens"].shape, (1, 1, 10))
        self.assertEqual(features["attention_mask"].shape, (1, 1))
        self.assertEqual(features["event_tokens"].count_nonzero().item(), 0)
        self.assertEqual(features["attention_mask"].count_nonzero().item(), 0)

    def test_all_empty_batch_uses_sequence_length_one(self):
        features = TWDToMFeatureBuilder().encode_batch(
            [{"game_log": []}, {"game_log": []}]
        )

        self.assertEqual(features["event_tokens"].shape, (2, 1, 10))
        self.assertEqual(features["attention_mask"].shape, (2, 1))
        self.assertEqual(features["event_tokens"].count_nonzero().item(), 0)
        self.assertEqual(features["attention_mask"].count_nonzero().item(), 0)

    def test_empty_batch_preserves_sequence_dimension(self):
        features = TWDToMFeatureBuilder().encode_batch([])

        self.assertIn("observer_id", features)
        self.assertEqual(features["event_tokens"].shape, (0, 1, 10))
        self.assertEqual(features["attention_mask"].shape, (0, 1))
        self.assertEqual(features["observer_id"].shape, (0,))

    def test_nonzero_pad_token_id_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "pad_token_id must be 0 in the first version",
        ):
            TWDToMFeatureBuilder(pad_token_id=1)

    def test_non_positive_max_seq_len_raises(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    TWDToMFeatureBuilder(max_seq_len=value)

    def test_outputs_use_long_dtype(self):
        features = TWDToMFeatureBuilder().encode_observation(
            observation_with_votes(1)
        )

        self.assertEqual(features["event_tokens"].dtype, torch.long)
        self.assertEqual(features["attention_mask"].dtype, torch.long)
        self.assertEqual(features["observer_id"].dtype, torch.long)

    def test_outputs_use_configured_device(self):
        features = TWDToMFeatureBuilder(device="cpu").encode_observation(
            observation_with_votes(1)
        )

        self.assertEqual(features["event_tokens"].device, torch.device("cpu"))
        self.assertEqual(features["attention_mask"].device, torch.device("cpu"))

    def test_custom_encoder_receives_original_observation(self):
        received = []
        observation = {"game_log": []}
        encoded_token = {
            field: index
            for index, field in enumerate(EVENT_TOKEN_FIELDS)
        }

        def dummy_encoder(value):
            received.append(value)
            return [encoded_token]

        builder = TWDToMFeatureBuilder(event_encoder=dummy_encoder)
        features = builder.encode_observation(observation)

        self.assertIs(received[0], observation)
        self.assertEqual(
            features["event_tokens"][0, 0].tolist(),
            list(range(10)),
        )

    def test_features_can_be_forwarded_to_twd_tom_model(self):
        features = TWDToMFeatureBuilder(max_seq_len=8).encode_batch(
            [observation_with_votes(1), observation_with_votes(2)]
        )
        model = TWDToMModel(
            TWDToMConfig(
                tom_config=ToMBackboneConfig(
                    d_model=16,
                    n_head=4,
                    n_layer=1,
                    dropout=0.0,
                    max_seq_len=8,
                    max_day=5,
                )
            )
        )

        output = model(
            features["event_tokens"],
            attention_mask=features["attention_mask"],
            observer_id=features["observer_id"],
        )

        self.assertEqual(output["wolf_prob"].shape, (2, 2, 7))
        self.assertEqual(output["region_probs"].shape, (2, 2, 7, 3))

    def test_public_methods_require_no_environment_or_speech(self):
        observation_parameters = tuple(
            inspect.signature(
                TWDToMFeatureBuilder.encode_observation
            ).parameters
        )
        batch_parameters = tuple(
            inspect.signature(TWDToMFeatureBuilder.encode_batch).parameters
        )

        self.assertEqual(observation_parameters, ("self", "observation"))
        self.assertEqual(batch_parameters, ("self", "observations"))


if __name__ == "__main__":
    unittest.main()
