import importlib
import math
import unittest

import torch


class EvalTWDToMPriorBaselinesTest(unittest.TestCase):
    def baselines_module(self):
        return importlib.import_module("script.twd_tom.eval_prior_baselines")

    def make_loss_config(self):
        return {
            "supervision_mode": "last",
            "bce_weight": 1.0,
            "cardinality_weight": 0.05,
            "num_wolves": 2.0,
            "region_weight": 0.0,
        }

    def make_batch(self):
        return {
            "event_tokens": torch.zeros(2, 3, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 3),
            "observer_id": torch.tensor([1, 2], dtype=torch.long),
            "wolf_labels": torch.tensor(
                [
                    [1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                ],
                dtype=torch.float32,
            ),
        }

    def test_format_baseline_summary_prints_only_requested_lines(self):
        module = self.baselines_module()

        summary = module.format_baseline_summary(
            {
                "uniform_eval_loss": 0.612345,
                "uniform_top2_f1": 0.5,
                "random_top2_f1": 0.25,
            }
        )

        self.assertEqual(
            summary.splitlines(),
            [
                "uniform_eval_loss=0.612345 uniform_top2_f1=0.500000",
                "random_top2_f1=0.250000",
            ],
        )
        for forbidden_name in (
            "top2_exact",
            "top2_recall",
            "binary_accuracy",
            "POS",
            "BND",
            "NEG",
            "coverage",
        ):
            self.assertNotIn(forbidden_name, summary)

    def test_select_eval_samples_uses_configured_eval_split(self):
        module = self.baselines_module()
        samples = [
            {"game_id": f"game-{game_id}", "sample_id": f"{game_id}-{idx}"}
            for game_id in range(4)
            for idx in range(2)
        ]
        config = {
            "data": {
                "split_by_game_id": True,
                "val_ratio": 0.25,
                "seed": 7,
            }
        }

        eval_samples = module.select_eval_samples(samples, config)
        _, expected_eval_samples = module.split_samples(
            samples,
            split_by_game_id=True,
            val_ratio=0.25,
            seed=7,
        )

        self.assertEqual(eval_samples, expected_eval_samples)

    def test_evaluate_prior_baselines_returns_requested_metrics(self):
        module = self.baselines_module()

        metrics = module.evaluate_prior_baselines(
            [self.make_batch()],
            torch.device("cpu"),
            self.make_loss_config(),
            num_trials=20,
            seed=123,
        )

        self.assertEqual(
            set(metrics),
            {
                "uniform_eval_loss",
                "uniform_top2_f1",
                "random_top2_f1",
            },
        )
        self.assertTrue(math.isfinite(metrics["uniform_eval_loss"]))
        self.assertGreaterEqual(metrics["uniform_top2_f1"], 0.0)
        self.assertLessEqual(metrics["uniform_top2_f1"], 1.0)
        self.assertGreaterEqual(metrics["random_top2_f1"], 0.0)
        self.assertLessEqual(metrics["random_top2_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
