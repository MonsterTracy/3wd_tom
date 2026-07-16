import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn

from werewolf.encoding.dialogue_actions import (
    CAMP2ID,
    CERTAINTY2ID,
    EVENT_TYPE2ID,
    PHASE2ID,
    POLARITY2ID,
    PREDICATE2ID,
    ROLE2ID,
)
from werewolf.models.twd_tom.backbone import ToMBackbone, ToMBackboneConfig


def vocab_size(mapping):
    return max(mapping.values()) + 1


class ToMBackboneTest(unittest.TestCase):
    def setUp(self):
        self.config = ToMBackboneConfig(
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
        )

    def make_event_tokens(self, batch_size=2, seq_len=5):
        tokens = torch.zeros(batch_size, seq_len, 10, dtype=torch.long)
        tokens[..., 0] = torch.randint(
            0, vocab_size(EVENT_TYPE2ID), (batch_size, seq_len)
        )
        tokens[..., 1] = torch.randint(
            0, self.config.num_players + 1, (batch_size, seq_len)
        )
        tokens[..., 2] = torch.randint(
            0, self.config.num_players + 1, (batch_size, seq_len)
        )
        tokens[..., 3] = torch.randint(
            0, vocab_size(PREDICATE2ID), (batch_size, seq_len)
        )
        tokens[..., 4] = torch.randint(
            0, vocab_size(ROLE2ID), (batch_size, seq_len)
        )
        tokens[..., 5] = torch.randint(
            0, vocab_size(CAMP2ID), (batch_size, seq_len)
        )
        tokens[..., 6] = torch.randint(
            0, vocab_size(POLARITY2ID), (batch_size, seq_len)
        )
        tokens[..., 7] = torch.randint(
            0, vocab_size(CERTAINTY2ID), (batch_size, seq_len)
        )
        tokens[..., 8] = torch.randint(
            0, vocab_size(PHASE2ID), (batch_size, seq_len)
        )
        tokens[..., 9] = torch.randint(
            0, self.config.max_day + 1, (batch_size, seq_len)
        )
        return tokens

    def test_forward_returns_expected_shapes_and_probability_range(self):
        model = ToMBackbone(self.config)

        output = model(self.make_event_tokens())

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))
        self.assertTrue((output["wolf_prob"] >= 0).all())
        self.assertTrue((output["wolf_prob"] <= 1).all())

    def test_default_backbone_type_is_transformer(self):
        self.assertEqual(ToMBackboneConfig().backbone_type, "transformer")

    def test_default_uses_observer_id(self):
        self.assertTrue(ToMBackboneConfig().use_observer_id)

    def test_default_llama_config_fields(self):
        config = ToMBackboneConfig()

        self.assertIsNone(config.intermediate_size)
        self.assertEqual(config.rope_theta, 10000.0)

    def test_boe_mlp_forward_returns_transformer_compatible_shapes(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="boe_mlp",
        )
        model = ToMBackbone(config)

        output = model(self.make_event_tokens())

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))

    def test_boe_mlp_supports_observer_id(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="boe_mlp",
        )
        model = ToMBackbone(config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        self.assertFalse(
            torch.equal(
                output["hidden_states"][0],
                output["hidden_states"][1],
            )
        )

    def test_gru_forward_returns_transformer_compatible_shapes(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gru",
        )
        model = ToMBackbone(config)

        output = model(self.make_event_tokens())

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))

    def test_gru_supports_attention_mask(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gru",
        )
        model = ToMBackbone(config)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = model(
            self.make_event_tokens(),
            attention_mask=attention_mask,
        )

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))

    def test_gru_supports_observer_id(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gru",
        )
        model = ToMBackbone(config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        self.assertFalse(
            torch.equal(
                output["hidden_states"][0],
                output["hidden_states"][1],
            )
        )

    def test_llama_config_can_build(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="llama",
            intermediate_size=32,
            rope_theta=10000.0,
        )

        model = ToMBackbone(config)

        self.assertEqual(model.config.backbone_type, "llama")

    def test_llama_forward_returns_transformer_compatible_shapes(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="llama",
            intermediate_size=32,
        )
        model = ToMBackbone(config)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        output = model(
            self.make_event_tokens(),
            attention_mask=attention_mask,
            observer_id=observer_id,
        )

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))

    def test_llama_supports_padding_attention_mask(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="llama",
            intermediate_size=32,
        )
        model = ToMBackbone(config)
        attention_mask = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = model(
            self.make_event_tokens(),
            attention_mask=attention_mask,
        )

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        self.assertTrue(torch.isfinite(output["hidden_states"]).all())

    def test_llama_is_causal(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="llama",
            intermediate_size=32,
        )
        model = ToMBackbone(config).eval()
        original = torch.zeros(1, 5, 10, dtype=torch.long)
        changed_future = original.clone()
        changed_future[:, 3:, 0] = 1

        with torch.no_grad():
            original_hidden = model(original)["hidden_states"]
            changed_hidden = model(changed_future)["hidden_states"]

        torch.testing.assert_close(
            original_hidden[:, :3],
            changed_hidden[:, :3],
        )

    def test_llama_supports_observer_id(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="llama",
            intermediate_size=32,
        )
        model = ToMBackbone(config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        self.assertFalse(
            torch.equal(
                output["hidden_states"][0],
                output["hidden_states"][1],
            )
        )

    def test_llama_ignores_observer_id_when_disabled(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="llama",
            intermediate_size=32,
            use_observer_id=False,
        )
        model = ToMBackbone(config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        torch.testing.assert_close(
            output["hidden_states"][0],
            output["hidden_states"][1],
        )

    def test_gpt_neox_config_can_build(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gpt_neox",
            intermediate_size=32,
            rope_theta=10000.0,
        )

        model = ToMBackbone(config)

        self.assertEqual(model.config.backbone_type, "gpt_neox")

    def test_gpt_neox_forward_returns_transformer_compatible_shapes(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gpt_neox",
            intermediate_size=32,
        )
        model = ToMBackbone(config)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        output = model(
            self.make_event_tokens(),
            attention_mask=attention_mask,
            observer_id=observer_id,
        )

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))

    def test_gpt_neox_supports_padding_attention_mask(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gpt_neox",
            intermediate_size=32,
        )
        model = ToMBackbone(config)
        attention_mask = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = model(
            self.make_event_tokens(),
            attention_mask=attention_mask,
        )

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        self.assertTrue(torch.isfinite(output["hidden_states"]).all())

    def test_gpt_neox_is_causal(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gpt_neox",
            intermediate_size=32,
        )
        model = ToMBackbone(config).eval()
        original = torch.zeros(1, 5, 10, dtype=torch.long)
        changed_future = original.clone()
        changed_future[:, 3:, 0] = 1

        with torch.no_grad():
            original_hidden = model(original)["hidden_states"]
            changed_hidden = model(changed_future)["hidden_states"]

        torch.testing.assert_close(
            original_hidden[:, :3],
            changed_hidden[:, :3],
        )

    def test_gpt_neox_supports_observer_id(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gpt_neox",
            intermediate_size=32,
        )
        model = ToMBackbone(config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        self.assertFalse(
            torch.equal(
                output["hidden_states"][0],
                output["hidden_states"][1],
            )
        )

    def test_gpt_neox_ignores_observer_id_when_disabled(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            backbone_type="gpt_neox",
            intermediate_size=32,
            use_observer_id=False,
        )
        model = ToMBackbone(config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        torch.testing.assert_close(
            output["hidden_states"][0],
            output["hidden_states"][1],
        )

    def test_observer_embedding_changes_hidden_states(self):
        model = ToMBackbone(self.config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        self.assertTrue(hasattr(model, "observer_emb"))

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        self.assertEqual(
            model.observer_emb.num_embeddings,
            self.config.num_players + 1,
        )
        self.assertFalse(
            torch.equal(
                output["hidden_states"][0],
                output["hidden_states"][1],
            )
        )

    def test_disabling_observer_id_ignores_observer_embedding(self):
        config = ToMBackboneConfig(
            num_players=7,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            max_day=3,
            use_observer_id=False,
        )
        model = ToMBackbone(config).eval()
        tokens = torch.zeros(2, 5, 10, dtype=torch.long)
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        with torch.no_grad():
            output = model(tokens, observer_id=observer_id)

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))
        torch.testing.assert_close(
            output["hidden_states"][0],
            output["hidden_states"][1],
        )

    def test_valid_attention_mask_is_supported(self):
        model = ToMBackbone(self.config)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = model(
            self.make_event_tokens(),
            attention_mask=attention_mask,
        )

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))

    def test_non_three_dimensional_tokens_raise(self):
        model = ToMBackbone(self.config)

        with self.assertRaises(ValueError):
            model(torch.zeros(2, 10, dtype=torch.long))

    def test_wrong_field_dimension_raises(self):
        model = ToMBackbone(self.config)

        with self.assertRaises(ValueError):
            model(torch.zeros(2, 5, 9, dtype=torch.long))

    def test_sequence_longer_than_maximum_raises(self):
        model = ToMBackbone(self.config)

        with self.assertRaises(ValueError):
            model(self.make_event_tokens(seq_len=9))

    def test_wrong_attention_mask_shape_raises(self):
        model = ToMBackbone(self.config)

        with self.assertRaises(ValueError):
            model(
                self.make_event_tokens(),
                attention_mask=torch.ones(2, 4, dtype=torch.long),
            )

    def test_day_id_above_maximum_is_clamped(self):
        model = ToMBackbone(self.config).eval()
        above_max = self.make_event_tokens()
        at_max = above_max.clone()
        above_max[..., 9] = self.config.max_day + 100
        at_max[..., 9] = self.config.max_day

        with torch.no_grad():
            above_output = model(above_max)["hidden_states"]
            max_output = model(at_max)["hidden_states"]

        torch.testing.assert_close(above_output, max_output)

    def test_forward_requires_no_raw_speech_or_observation(self):
        parameters = tuple(inspect.signature(ToMBackbone.forward).parameters)

        self.assertEqual(
            parameters,
            ("self", "event_tokens", "attention_mask", "observer_id"),
        )

    def test_embedding_sizes_use_maximum_mapping_id(self):
        model = ToMBackbone(self.config)

        self.assertEqual(
            model.event_type_embedding.num_embeddings,
            vocab_size(EVENT_TYPE2ID),
        )
        self.assertEqual(
            model.predicate_embedding.num_embeddings,
            vocab_size(PREDICATE2ID),
        )
        self.assertEqual(
            model.role_embedding.num_embeddings,
            vocab_size(ROLE2ID),
        )
        self.assertEqual(
            model.camp_embedding.num_embeddings,
            vocab_size(CAMP2ID),
        )
        self.assertEqual(
            model.polarity_embedding.num_embeddings,
            vocab_size(POLARITY2ID),
        )
        self.assertEqual(
            model.certainty_embedding.num_embeddings,
            vocab_size(CERTAINTY2ID),
        )
        self.assertEqual(
            model.phase_embedding.num_embeddings,
            vocab_size(PHASE2ID),
        )
        self.assertEqual(
            model.speaker_embedding.num_embeddings,
            self.config.num_players + 1,
        )
        self.assertEqual(
            model.target_embedding.num_embeddings,
            self.config.num_players + 1,
        )
        self.assertEqual(
            model.day_embedding.num_embeddings,
            self.config.max_day + 1,
        )

    def test_forced_fallback_supports_causal_and_padding_masks(self):
        with patch(
            "werewolf.models.twd_tom.backbone.HAS_TRANSFORMERS",
            False,
        ):
            model = ToMBackbone(self.config).eval()

        self.assertIsInstance(model.blocks[0], nn.TransformerEncoderLayer)
        tokens = self.make_event_tokens()
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = model(tokens, attention_mask=attention_mask)

        self.assertEqual(output["hidden_states"].shape, (2, 5, 16))

    def test_forced_fallback_is_causal(self):
        with patch(
            "werewolf.models.twd_tom.backbone.HAS_TRANSFORMERS",
            False,
        ):
            model = ToMBackbone(self.config).eval()
        original = torch.zeros(1, 5, 10, dtype=torch.long)
        changed_future = original.clone()
        changed_future[:, 3:, 0] = 1

        with torch.no_grad():
            original_hidden = model(original)["hidden_states"]
            changed_hidden = model(changed_future)["hidden_states"]

        torch.testing.assert_close(
            original_hidden[:, :3],
            changed_hidden[:, :3],
        )


if __name__ == "__main__":
    unittest.main()
