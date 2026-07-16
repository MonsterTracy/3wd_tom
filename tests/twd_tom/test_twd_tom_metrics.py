import importlib
import importlib.util
import math
import unittest

import torch


class TWDToMMetricsTest(unittest.TestCase):
    def metrics_module(self):
        return importlib.import_module("werewolf.models.twd_tom.metrics")

    def test_metrics_module_exists(self):
        self.assertIsNotNone(
            importlib.util.find_spec("werewolf.models.twd_tom.metrics")
        )

    def test_probability_top2_and_count_metrics(self):
        metrics_module = self.metrics_module()
        wolf_prob = torch.tensor(
            [
                [0.9, 0.8, 0.1, 0.1, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.8, 0.4, 0.7, 0.0, 0.0],
            ]
        )
        wolf_labels = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )

        metrics = metrics_module.compute_wolf_probability_metrics(
            wolf_prob,
            wolf_labels,
            attention_mask=torch.ones(2, 3),
        )

        self.assertAlmostEqual(metrics["count_error"], 0.1, places=6)
        self.assertAlmostEqual(metrics["top2_exact"], 0.5, places=6)
        self.assertAlmostEqual(metrics["top2_recall"], 0.75, places=6)
        self.assertAlmostEqual(
            metrics["binary_accuracy"],
            12.0 / 14.0,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["true_wolf_mean_prob"],
            0.725,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["true_good_mean_prob"],
            0.11,
            places=6,
        )

    def test_probability_top2_f1_matches_recall_in_fixed_two_wolf_setting(self):
        metrics_module = self.metrics_module()
        cases = [
            (
                "two_hits",
                torch.tensor([[0.9, 0.8, 0.2, 0.1, 0.0, 0.0, 0.0]]),
                torch.tensor([[1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32),
                1.0,
            ),
            (
                "one_hit",
                torch.tensor([[0.9, 0.2, 0.8, 0.1, 0.0, 0.0, 0.0]]),
                torch.tensor([[1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32),
                0.5,
            ),
            (
                "zero_hits",
                torch.tensor([[0.2, 0.1, 0.9, 0.8, 0.0, 0.0, 0.0]]),
                torch.tensor([[1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32),
                0.0,
            ),
        ]

        for name, wolf_prob, wolf_labels, expected_f1 in cases:
            with self.subTest(name=name):
                metrics = metrics_module.compute_wolf_probability_metrics(
                    wolf_prob,
                    wolf_labels,
                )

                self.assertAlmostEqual(
                    metrics["top2_recall"],
                    expected_f1,
                    places=6,
                )
                self.assertAlmostEqual(
                    metrics["top2_f1"],
                    expected_f1,
                    places=6,
                )
                self.assertAlmostEqual(
                    metrics["top2_f1"],
                    metrics["top2_recall"],
                    places=6,
                )

    def test_probability_last_mode_uses_attention_mask(self):
        metrics_module = self.metrics_module()
        wolf_prob = torch.zeros(2, 3, 7)
        wolf_prob[0, 1, :2] = 1.0
        wolf_prob[0, 2, 2:4] = 1.0
        wolf_prob[1, 2, 2:4] = 1.0
        wolf_labels = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        metrics = metrics_module.compute_wolf_probability_metrics(
            wolf_prob,
            wolf_labels,
            attention_mask=attention_mask,
            supervision_mode="last",
        )

        self.assertEqual(metrics["top2_exact"], 1.0)
        self.assertEqual(metrics["top2_recall"], 1.0)
        self.assertEqual(metrics["count_error"], 0.0)

    def test_region_precision_recall_coverage_and_selective_accuracy(self):
        metrics_module = self.metrics_module()
        hard_region = torch.tensor([[0, 1, 0, 2, 2, 1, 2]])
        wolf_labels = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )

        metrics = metrics_module.compute_twd_region_metrics(
            hard_region,
            wolf_labels,
        )

        self.assertAlmostEqual(metrics["POS_ratio"], 2.0 / 7.0)
        self.assertAlmostEqual(metrics["BND_ratio"], 2.0 / 7.0)
        self.assertAlmostEqual(metrics["NEG_ratio"], 3.0 / 7.0)
        self.assertAlmostEqual(metrics["POS_precision"], 0.5)
        self.assertAlmostEqual(metrics["POS_recall"], 0.5)
        self.assertAlmostEqual(metrics["POS_f1"], 0.5)
        self.assertAlmostEqual(metrics["NEG_precision"], 1.0)
        self.assertAlmostEqual(metrics["NEG_recall"], 0.6)
        self.assertAlmostEqual(metrics["NEG_f1"], 0.75)
        self.assertAlmostEqual(metrics["true_wolf_BND_rate"], 0.5)
        self.assertAlmostEqual(metrics["true_good_BND_rate"], 0.2)
        self.assertAlmostEqual(metrics["coverage"], 5.0 / 7.0)
        self.assertAlmostEqual(metrics["selective_accuracy"], 0.8)

    def test_all_bnd_metrics_are_finite_and_zero_coverage(self):
        metrics_module = self.metrics_module()
        hard_region = torch.ones(2, 7, dtype=torch.long)
        wolf_labels = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )

        metrics = metrics_module.compute_twd_region_metrics(
            hard_region,
            wolf_labels,
        )

        self.assertEqual(metrics["coverage"], 0.0)
        self.assertEqual(metrics["selective_accuracy"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_combined_metrics_merge_probability_and_region_results(self):
        metrics_module = self.metrics_module()
        wolf_prob = torch.tensor(
            [[[1, 1, 0, 0, 0, 0, 0]]],
            dtype=torch.float32,
        )
        hard_region = torch.tensor([[[0, 0, 2, 2, 2, 2, 2]]])
        wolf_labels = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        )

        metrics = metrics_module.compute_twd_tom_metrics(
            {
                "wolf_prob": wolf_prob,
                "hard_region": hard_region,
            },
            wolf_labels,
        )

        self.assertEqual(metrics["top2_exact"], 1.0)
        self.assertEqual(metrics["POS_precision"], 1.0)
        self.assertEqual(metrics["NEG_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
