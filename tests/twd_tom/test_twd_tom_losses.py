import unittest

import torch
from torch.nn import functional as F

from werewolf.models.twd_tom.losses import (
    cardinality_loss,
    masked_bce_with_logits_loss,
    twd_region_consistency_loss,
    twd_tom_loss,
)


class TWDToMLossesTest(unittest.TestCase):
    def make_combined_inputs(self):
        wolf_logits = torch.randn(2, 3, 7, requires_grad=True)
        region_logits = torch.randn(2, 3, 7, 3, requires_grad=True)
        outputs = {
            "wolf_logits": wolf_logits,
            "wolf_prob": torch.sigmoid(wolf_logits),
            "region_probs": torch.softmax(region_logits, dim=-1),
        }
        labels = torch.randint(0, 2, (2, 3, 7), dtype=torch.float32)
        return outputs, labels, wolf_logits, region_logits

    def test_bce_supports_time_dependent_labels(self):
        logits = torch.randn(2, 3, 7)
        labels = torch.rand(2, 3, 7)

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            reduction="none",
            supervision_mode="all",
        )

        expected = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        torch.testing.assert_close(loss, expected)

    def test_bce_expands_batch_player_labels_across_time(self):
        logits = torch.randn(2, 3, 7)
        labels = torch.rand(2, 7)

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            reduction="none",
            supervision_mode="all",
        )

        expected = F.binary_cross_entropy_with_logits(
            logits,
            labels.unsqueeze(1).expand_as(logits),
            reduction="none",
        )
        torch.testing.assert_close(loss, expected)

    def test_attention_mask_zeros_padding_time_step(self):
        logits = torch.zeros(1, 2, 7)
        labels = torch.zeros(1, 2, 7)
        attention_mask = torch.tensor([[1, 0]])

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            attention_mask=attention_mask,
            reduction="none",
            supervision_mode="all",
        )

        self.assertTrue((loss[:, 0] >= 0).all())
        self.assertEqual(loss[:, 1].count_nonzero().item(), 0)

    def test_player_mask_zeros_masked_player(self):
        logits = torch.zeros(1, 2, 7)
        labels = torch.zeros(1, 2, 7)
        player_mask = torch.ones(1, 7)
        player_mask[:, 3] = 0

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            player_mask=player_mask,
            reduction="none",
            supervision_mode="all",
        )

        self.assertEqual(loss[..., 3].count_nonzero().item(), 0)
        self.assertTrue((loss[..., :3] >= 0).all())

    def test_attention_and_player_masks_both_apply(self):
        logits = torch.zeros(1, 2, 7)
        labels = torch.zeros(1, 2, 7)
        attention_mask = torch.tensor([[1, 0]])
        player_mask = torch.ones(1, 2, 7)
        player_mask[..., 0] = 0

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            attention_mask=attention_mask,
            player_mask=player_mask,
            reduction="none",
            supervision_mode="all",
        )

        self.assertEqual(loss[..., 0].count_nonzero().item(), 0)
        self.assertEqual(loss[:, 1].count_nonzero().item(), 0)
        self.assertTrue((loss[:, 0, 1:] >= 0).all())

    def test_all_zero_mask_returns_differentiable_zero(self):
        for reduction in ("mean", "sum"):
            with self.subTest(reduction=reduction):
                logits = torch.randn(1, 2, 7, requires_grad=True)
                loss = masked_bce_with_logits_loss(
                    logits,
                    torch.zeros(1, 2, 7),
                    attention_mask=torch.zeros(1, 2),
                    reduction=reduction,
                    supervision_mode="all",
                )

                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertEqual(logits.grad.count_nonzero().item(), 0)

    def test_sum_reduction_matches_elementwise_sum(self):
        logits = torch.randn(2, 3, 7)
        labels = torch.rand(2, 3, 7)

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            reduction="sum",
            supervision_mode="all",
        )

        expected = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        )
        torch.testing.assert_close(loss, expected)

    def test_none_reduction_returns_full_masked_shape(self):
        logits = torch.zeros(1, 2, 7)
        labels = torch.zeros(1, 2, 7)
        attention_mask = torch.tensor([[1, 0]])
        player_mask = torch.ones(1, 7)
        player_mask[:, 2] = 0

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            attention_mask=attention_mask,
            player_mask=player_mask,
            reduction="none",
            supervision_mode="all",
        )

        self.assertEqual(loss.shape, (1, 2, 7))
        self.assertEqual(loss[:, 1].count_nonzero().item(), 0)
        self.assertEqual(loss[..., 2].count_nonzero().item(), 0)
        self.assertTrue((loss[:, 0, [0, 1, 3, 4, 5, 6]] >= 0).all())

    def test_invalid_reduction_raises(self):
        with self.assertRaises(ValueError):
            masked_bce_with_logits_loss(
                torch.zeros(1, 1, 7),
                torch.zeros(1, 1, 7),
                reduction="median",
            )

    def test_last_token_supervision_with_b_7_labels(self):
        logits = torch.randn(2, 4, 7, requires_grad=True)
        region_logits = torch.randn(2, 4, 7, 3)
        outputs = {
            "wolf_logits": logits,
            "region_probs": torch.softmax(region_logits, dim=-1),
        }
        labels = torch.rand(2, 7)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]]
        )

        loss = twd_tom_loss(
            outputs,
            labels,
            attention_mask=attention_mask,
            supervision_mode="last",
        )["loss"]
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(loss))

    def test_last_token_uses_last_valid_index(self):
        logits = torch.randn(2, 4, 7)
        labels = torch.rand(2, 7)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]]
        )

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            attention_mask=attention_mask,
            supervision_mode="last",
        )

        expected = F.binary_cross_entropy_with_logits(
            torch.stack([logits[0, 2], logits[1, 3]]),
            labels,
        )
        torch.testing.assert_close(loss, expected)

    def test_all_mode_keeps_old_behavior(self):
        logits = torch.randn(2, 4, 7)
        labels = torch.rand(2, 7)
        attention_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]]
        )

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            attention_mask=attention_mask,
            supervision_mode="all",
        )

        elementwise = F.binary_cross_entropy_with_logits(
            logits,
            labels.unsqueeze(1).expand_as(logits),
            reduction="none",
        )
        mask = attention_mask.unsqueeze(-1).expand_as(elementwise)
        expected = (elementwise * mask).sum() / mask.sum()
        torch.testing.assert_close(loss, expected)

    def test_attention_mask_none_uses_last_timestep(self):
        logits = torch.randn(2, 4, 7)
        labels = torch.rand(2, 7)

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            attention_mask=None,
            supervision_mode="last",
        )

        expected = F.binary_cross_entropy_with_logits(
            logits[:, -1],
            labels,
        )
        torch.testing.assert_close(loss, expected)

    def test_labels_b_t_7_last_mode(self):
        logits = torch.randn(2, 4, 7)
        labels = torch.rand(2, 4, 7)
        attention_mask = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 1]]
        )

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            attention_mask=attention_mask,
            reduction="none",
            supervision_mode="last",
        )

        expected = F.binary_cross_entropy_with_logits(
            torch.stack([logits[0, 1], logits[1, 3]]),
            torch.stack([labels[0, 1], labels[1, 3]]),
            reduction="none",
        )
        self.assertEqual(loss.shape, (2, 7))
        torch.testing.assert_close(loss, expected)

    def test_soft_labels_supported(self):
        logits = torch.randn(2, 3, 7, requires_grad=True)
        labels = torch.tensor(
            [
                [0.2, 0.7, 0.4, 0.9, 0.1, 0.6, 0.3],
                [0.8, 0.1, 0.5, 0.2, 0.7, 0.4, 0.9],
            ]
        )

        loss = masked_bce_with_logits_loss(
            logits,
            labels,
            supervision_mode="last",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_combined_last_none_reduction_has_batch_player_shape(self):
        outputs, labels, _, _ = self.make_combined_inputs()
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        losses = twd_tom_loss(
            outputs,
            labels,
            attention_mask=attention_mask,
            region_weight=0.5,
            reduction="none",
            supervision_mode="last",
        )

        self.assertEqual(losses["loss"].shape, (2, 7))
        self.assertEqual(losses["bce_loss"].shape, (2, 7))
        self.assertEqual(losses["region_loss"].shape, (2, 7))

    def test_cardinality_loss_last_mode(self):
        wolf_prob = torch.zeros(2, 4, 7)
        wolf_prob[0, 2] = 0.5
        wolf_prob[1, 3] = 2.0 / 7.0
        attention_mask = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]]
        )

        loss = cardinality_loss(
            wolf_prob,
            attention_mask=attention_mask,
            num_wolves=2.0,
            supervision_mode="last",
        )

        expected = torch.tensor([
            (3.5 - 2.0) ** 2,
            (2.0 - 2.0) ** 2,
        ]).mean()
        torch.testing.assert_close(loss, expected)

    def test_cardinality_loss_all_mode_with_mask(self):
        wolf_prob = torch.rand(2, 4, 7)
        attention_mask = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 0]]
        )

        loss = cardinality_loss(
            wolf_prob,
            attention_mask=attention_mask,
            supervision_mode="all",
        )

        elementwise = (wolf_prob.sum(dim=-1) - 2.0).square()
        expected = elementwise[attention_mask.bool()].mean()
        torch.testing.assert_close(loss, expected)

    def test_twd_tom_loss_with_cardinality_weight_zero_keeps_old_behavior(self):
        outputs, labels, _, _ = self.make_combined_inputs()

        losses = twd_tom_loss(
            outputs,
            labels,
            region_weight=0.5,
            cardinality_weight=0.0,
        )

        expected = losses["bce_loss"] + 0.5 * losses["region_loss"]
        torch.testing.assert_close(losses["loss"], expected)
        self.assertEqual(losses["cardinality_loss"].item(), 0.0)
        self.assertEqual(losses["cardinality_weight"], 0.0)

    def test_twd_tom_loss_with_cardinality_weight_positive(self):
        outputs, labels, _, _ = self.make_combined_inputs()

        losses = twd_tom_loss(
            outputs,
            labels,
            region_weight=0.0,
            cardinality_weight=0.05,
        )

        expected = losses["bce_loss"] + 0.05 * losses["cardinality_loss"]
        torch.testing.assert_close(losses["loss"], expected)
        self.assertEqual(losses["cardinality_weight"], 0.05)

    def test_cardinality_loss_attention_mask_none(self):
        wolf_prob = torch.rand(2, 4, 7)

        loss = cardinality_loss(
            wolf_prob,
            attention_mask=None,
            supervision_mode="last",
        )

        expected = (wolf_prob[:, -1].sum(dim=-1) - 2.0).square().mean()
        torch.testing.assert_close(loss, expected)

    def test_cardinality_loss_reduction_none(self):
        wolf_prob = torch.rand(2, 4, 7)
        attention_mask = torch.tensor(
            [[1, 1, 0, 0], [1, 1, 1, 1]]
        )

        last_loss = cardinality_loss(
            wolf_prob,
            attention_mask=attention_mask,
            supervision_mode="last",
            reduction="none",
        )
        all_loss = cardinality_loss(
            wolf_prob,
            attention_mask=attention_mask,
            supervision_mode="all",
            reduction="none",
        )

        self.assertEqual(last_loss.shape, (2,))
        self.assertEqual(all_loss.shape, (2, 4))
        self.assertEqual(all_loss[0, 2:].count_nonzero().item(), 0)

    def test_region_loss_is_finite_with_zero_probabilities(self):
        region_probs = torch.zeros(1, 1, 7, 3)
        labels = torch.zeros(1, 1, 7)

        loss = twd_region_consistency_loss(region_probs, labels)

        self.assertTrue(torch.isfinite(loss))

    def test_region_targets_encode_pos_bnd_and_neg(self):
        labels = torch.tensor([[[0.9, 0.5, 0.1]]])
        region_probs = torch.tensor(
            [[[
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]]]
        )

        loss = twd_region_consistency_loss(
            region_probs,
            labels,
            reduction="none",
        )

        torch.testing.assert_close(
            loss,
            torch.full((1, 1, 3), -torch.log(torch.tensor(0.8))),
        )

    def test_threshold_order_is_validated(self):
        with self.assertRaises(ValueError):
            twd_region_consistency_loss(
                torch.full((1, 1, 7, 3), 1 / 3),
                torch.zeros(1, 1, 7),
                neg_threshold=0.75,
                pos_threshold=0.75,
            )

    def test_negative_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            twd_region_consistency_loss(
                torch.full((1, 1, 7, 3), 1 / 3),
                torch.zeros(1, 1, 7),
                neg_threshold=-0.1,
            )

    def test_pos_threshold_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            twd_region_consistency_loss(
                torch.full((1, 1, 7, 3), 1 / 3),
                torch.zeros(1, 1, 7),
                pos_threshold=1.1,
            )

    def test_boundary_thresholds_are_accepted(self):
        loss = twd_region_consistency_loss(
            torch.full((1, 1, 7, 3), 1 / 3),
            torch.zeros(1, 1, 7),
            neg_threshold=0.0,
            pos_threshold=1.0,
        )

        self.assertTrue(torch.isfinite(loss))

    def test_region_none_reduction_applies_masks(self):
        region_probs = torch.full((1, 2, 7, 3), 1 / 3)
        labels = torch.full((1, 2, 7), 0.5)
        attention_mask = torch.tensor([[1, 0]])
        player_mask = torch.ones(1, 7)
        player_mask[:, 4] = 0

        loss = twd_region_consistency_loss(
            region_probs,
            labels,
            attention_mask=attention_mask,
            player_mask=player_mask,
            reduction="none",
        )

        self.assertEqual(loss.shape, (1, 2, 7))
        self.assertEqual(loss[:, 1].count_nonzero().item(), 0)
        self.assertEqual(loss[..., 4].count_nonzero().item(), 0)

    def test_combined_loss_returns_exact_keys(self):
        outputs, labels, _, _ = self.make_combined_inputs()

        losses = twd_tom_loss(outputs, labels)

        self.assertEqual(
            set(losses),
            {
                "loss",
                "bce_loss",
                "region_loss",
                "cardinality_loss",
                "cardinality_weight",
            },
        )

    def test_zero_region_weight_makes_total_equal_bce(self):
        outputs, labels, _, _ = self.make_combined_inputs()

        losses = twd_tom_loss(outputs, labels, region_weight=0.0)

        torch.testing.assert_close(losses["loss"], losses["bce_loss"])

    def test_positive_region_weight_contributes_region_loss(self):
        outputs, labels, _, _ = self.make_combined_inputs()

        losses = twd_tom_loss(outputs, labels, region_weight=0.5)

        torch.testing.assert_close(
            losses["loss"],
            losses["bce_loss"] + 0.5 * losses["region_loss"],
        )

    def test_combined_loss_supports_backward(self):
        outputs, labels, wolf_logits, region_logits = self.make_combined_inputs()

        loss = twd_tom_loss(
            outputs,
            labels,
            region_weight=0.5,
        )["loss"]
        loss.backward()

        self.assertIsNotNone(wolf_logits.grad)
        self.assertIsNotNone(region_logits.grad)


if __name__ == "__main__":
    unittest.main()
