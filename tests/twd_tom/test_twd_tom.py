import inspect
import unittest

import torch
from torch import nn

from werewolf.models.risk.twd_risk_layer import TWDRiskLayer
from werewolf.models.twd_tom.backbone import ToMBackbone, ToMBackboneConfig
from werewolf.models.twd_tom.model import TWDToMConfig, TWDToMModel


OUTPUT_KEYS = {
    "hidden_states",
    "wolf_logits",
    "wolf_prob",
    "region_probs",
    "risks",
    "hard_region",
    "costs",
}


class DummyToMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False
        self.received_attention_mask = None
        self.received_observer_id = None
        self.wolf_prob = torch.full((2, 5, 7), 0.9)

    def forward(self, event_tokens, attention_mask=None, observer_id=None):
        self.called = True
        self.received_attention_mask = attention_mask
        self.received_observer_id = observer_id
        return {
            "hidden_states": torch.zeros(2, 5, 4),
            "wolf_logits": torch.zeros(2, 5, 7),
            "wolf_prob": self.wolf_prob,
        }


class DummyTwdLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False
        self.received_wolf_prob = None
        self.received_context = None

    def forward(self, wolf_prob, context=None):
        self.called = True
        self.received_wolf_prob = wolf_prob
        self.received_context = context
        risks = torch.stack(
            [1 - wolf_prob, torch.full_like(wolf_prob, 0.25), wolf_prob],
            dim=-1,
        )
        return {
            "region_probs": torch.softmax(-risks, dim=-1),
            "risks": risks,
            "hard_region": risks.argmin(dim=-1),
            "costs": {},
        }


class TWDToMModelTest(unittest.TestCase):
    def setUp(self):
        self.tom_config = ToMBackboneConfig(
            num_players=7,
            d_model=32,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=16,
            max_day=5,
        )
        self.config = TWDToMConfig(
            tom_config=self.tom_config,
            twd_tau=0.7,
        )
        self.event_tokens = torch.zeros(2, 5, 10, dtype=torch.long)

    def test_default_config_uses_independent_tom_configs(self):
        first = TWDToMConfig()
        second = TWDToMConfig()

        self.assertIsNot(first.tom_config, second.tom_config)

    def test_defaults_create_composed_modules(self):
        model = TWDToMModel(self.config)

        self.assertIsInstance(model.tom_backbone, ToMBackbone)
        self.assertIsInstance(model.twd_layer, TWDRiskLayer)
        self.assertIs(model.config, self.config)
        self.assertEqual(model.twd_layer.tau, 0.7)

    def test_forward_returns_exact_keys_and_shapes(self):
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        self.assertIn(
            "observer_id",
            inspect.signature(TWDToMModel.forward).parameters,
        )
        output = TWDToMModel(self.config)(
            self.event_tokens,
            observer_id=observer_id,
        )

        self.assertEqual(set(output), OUTPUT_KEYS)
        self.assertEqual(output["hidden_states"].shape, (2, 5, 32))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))
        self.assertEqual(output["region_probs"].shape, (2, 5, 7, 3))
        self.assertEqual(output["risks"].shape, (2, 5, 7, 3))
        self.assertEqual(output["hard_region"].shape, (2, 5, 7))

    def test_boe_mlp_forward_shape_matches_transformer_interface(self):
        boe_config = TWDToMConfig(
            tom_config=ToMBackboneConfig(
                num_players=7,
                d_model=32,
                n_head=4,
                n_layer=1,
                dropout=0.0,
                max_seq_len=16,
                max_day=5,
                backbone_type="boe_mlp",
            ),
            twd_tau=0.7,
        )
        observer_id = torch.tensor([1, 7], dtype=torch.long)

        output = TWDToMModel(boe_config)(
            self.event_tokens,
            observer_id=observer_id,
        )

        self.assertEqual(set(output), OUTPUT_KEYS)
        self.assertEqual(output["hidden_states"].shape, (2, 5, 32))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))
        self.assertEqual(output["region_probs"].shape, (2, 5, 7, 3))
        self.assertEqual(output["hard_region"].shape, (2, 5, 7))

    def test_gru_forward_shape_matches_transformer_interface(self):
        gru_config = TWDToMConfig(
            tom_config=ToMBackboneConfig(
                num_players=7,
                d_model=32,
                n_head=4,
                n_layer=1,
                dropout=0.0,
                max_seq_len=16,
                max_day=5,
                backbone_type="gru",
            ),
            twd_tau=0.7,
        )
        observer_id = torch.tensor([1, 7], dtype=torch.long)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = TWDToMModel(gru_config)(
            self.event_tokens,
            attention_mask=attention_mask,
            observer_id=observer_id,
        )

        self.assertEqual(set(output), OUTPUT_KEYS)
        self.assertEqual(output["hidden_states"].shape, (2, 5, 32))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))
        self.assertEqual(output["region_probs"].shape, (2, 5, 7, 3))
        self.assertEqual(output["hard_region"].shape, (2, 5, 7))

    def test_llama_forward_shape_matches_transformer_interface(self):
        llama_config = TWDToMConfig(
            tom_config=ToMBackboneConfig(
                num_players=7,
                d_model=32,
                n_head=4,
                n_layer=1,
                dropout=0.0,
                max_seq_len=16,
                max_day=5,
                backbone_type="llama",
                intermediate_size=64,
            ),
            twd_tau=0.7,
        )
        observer_id = torch.tensor([1, 7], dtype=torch.long)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = TWDToMModel(llama_config)(
            self.event_tokens,
            attention_mask=attention_mask,
            observer_id=observer_id,
        )

        self.assertEqual(set(output), OUTPUT_KEYS)
        self.assertEqual(output["hidden_states"].shape, (2, 5, 32))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))
        self.assertEqual(output["region_probs"].shape, (2, 5, 7, 3))
        self.assertEqual(output["hard_region"].shape, (2, 5, 7))

    def test_gpt_neox_forward_shape_matches_transformer_interface(self):
        gpt_neox_config = TWDToMConfig(
            tom_config=ToMBackboneConfig(
                num_players=7,
                d_model=32,
                n_head=4,
                n_layer=1,
                dropout=0.0,
                max_seq_len=16,
                max_day=5,
                backbone_type="gpt_neox",
                intermediate_size=64,
            ),
            twd_tau=0.7,
        )
        observer_id = torch.tensor([1, 7], dtype=torch.long)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = TWDToMModel(gpt_neox_config)(
            self.event_tokens,
            attention_mask=attention_mask,
            observer_id=observer_id,
        )

        self.assertEqual(set(output), OUTPUT_KEYS)
        self.assertEqual(output["hidden_states"].shape, (2, 5, 32))
        self.assertEqual(output["wolf_logits"].shape, (2, 5, 7))
        self.assertEqual(output["wolf_prob"].shape, (2, 5, 7))
        self.assertEqual(output["region_probs"].shape, (2, 5, 7, 3))
        self.assertEqual(output["hard_region"].shape, (2, 5, 7))

    def test_different_observers_change_hidden_states(self):
        model = TWDToMModel(self.config).eval()
        observer_id = torch.tensor([2, 6], dtype=torch.long)

        with torch.no_grad():
            output = model(
                self.event_tokens,
                observer_id=observer_id,
            )

        self.assertFalse(
            torch.equal(
                output["hidden_states"][0],
                output["hidden_states"][1],
            )
        )

    def test_probabilities_follow_required_invariants(self):
        output = TWDToMModel(self.config)(self.event_tokens)

        self.assertTrue((output["wolf_prob"] >= 0).all())
        self.assertTrue((output["wolf_prob"] <= 1).all())
        torch.testing.assert_close(
            output["region_probs"].sum(dim=-1),
            torch.ones(2, 5, 7),
        )
        self.assertTrue(
            torch.equal(
                output["region_probs"].argmax(dim=-1),
                output["risks"].argmin(dim=-1),
            )
        )

    def test_attention_mask_and_none_context_are_supported(self):
        attention_mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
        )

        output = TWDToMModel(self.config)(
            self.event_tokens,
            attention_mask=attention_mask,
            context=None,
        )

        self.assertEqual(output["hard_region"].shape, (2, 5, 7))

    def test_custom_tom_backbone_is_stored_and_called(self):
        tom_backbone = DummyToMBackbone()
        attention_mask = torch.ones(2, 5, dtype=torch.long)
        observer_id = torch.tensor([3, 4], dtype=torch.long)
        model = TWDToMModel(
            self.config,
            tom_backbone=tom_backbone,
        )

        output = model(
            self.event_tokens,
            attention_mask=attention_mask,
            observer_id=observer_id,
        )

        self.assertIs(model.tom_backbone, tom_backbone)
        self.assertTrue(tom_backbone.called)
        self.assertIs(tom_backbone.received_attention_mask, attention_mask)
        self.assertIs(tom_backbone.received_observer_id, observer_id)
        self.assertIs(output["wolf_prob"], tom_backbone.wolf_prob)

    def test_custom_twd_layer_receives_exact_wolf_prob_and_context(self):
        tom_backbone = DummyToMBackbone()
        twd_layer = DummyTwdLayer()
        context = {"observer_id": 3}
        model = TWDToMModel(
            self.config,
            tom_backbone=tom_backbone,
            twd_layer=twd_layer,
        )

        model(self.event_tokens, context=context)

        self.assertIs(model.twd_layer, twd_layer)
        self.assertTrue(twd_layer.called)
        self.assertIs(twd_layer.received_wolf_prob, tom_backbone.wolf_prob)
        self.assertIs(twd_layer.received_context, context)

    def test_forward_requires_no_raw_speech_or_observation(self):
        parameters = tuple(inspect.signature(TWDToMModel.forward).parameters)

        self.assertEqual(
            parameters,
            (
                "self",
                "event_tokens",
                "attention_mask",
                "context",
                "observer_id",
            ),
        )


if __name__ == "__main__":
    unittest.main()
