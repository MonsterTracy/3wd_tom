import importlib
import math
import tempfile
import unittest
from pathlib import Path

import torch


class EvalTWDToMScriptTest(unittest.TestCase):
    def test_default_checkpoint_path_uses_checkpoint_best(self):
        module = importlib.import_module("script.twd_tom.eval")

        checkpoint_path = module.resolve_checkpoint_path(
            checkpoint_dir=Path("/tmp/twd_tom_eval"),
            checkpoint_path=None,
        )

        self.assertEqual(
            checkpoint_path,
            Path("/tmp/twd_tom_eval") / "checkpoint_best.pt",
        )

    def test_eval_summary_prints_only_eval_loss_and_top2_f1(self):
        module = importlib.import_module("script.twd_tom.eval")

        summary = module.format_eval_summary(
            {"eval_loss": 0.612345, "top2_f1": 0.5}
        )

        self.assertEqual(summary, "eval_loss=0.612345 top2_f1=0.500000")
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

    def test_eval_script_can_build_boe_mlp_config(self):
        module = importlib.import_module("script.twd_tom.eval")
        config = module.load_config("configs/twd_tom_train.yaml")
        config["model"]["backbone_type"] = "boe_mlp"
        config["model"]["d_model"] = 16
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "boe_mlp")

    def test_eval_script_can_build_gru_config(self):
        module = importlib.import_module("script.twd_tom.eval")
        config = module.load_config("configs/twd_tom_train.yaml")
        config["model"]["backbone_type"] = "gru"
        config["model"]["d_model"] = 16
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "gru")

    def test_eval_script_can_build_llama_config(self):
        module = importlib.import_module("script.twd_tom.eval")
        config = module.load_config("configs/twd_tom_train.yaml")
        config["model"]["backbone_type"] = "llama"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8
        config["model"]["intermediate_size"] = 32

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "llama")

    def test_eval_script_can_build_gpt_neox_config(self):
        module = importlib.import_module("script.twd_tom.eval")
        config = module.load_config("configs/twd_tom_train.yaml")
        config["model"]["backbone_type"] = "gpt_neox"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8
        config["model"]["intermediate_size"] = 32

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "gpt_neox")

    def test_eval_script_can_load_llama_checkpoint(self):
        module = importlib.import_module("script.twd_tom.eval")
        config = module.load_config("configs/twd_tom_train.yaml")
        config["model"]["backbone_type"] = "llama"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8
        config["model"]["intermediate_size"] = 32
        source_model = module.build_model(config)
        target_model = module.build_model(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_best.pt"
            torch.save(
                {"model_state_dict": source_model.state_dict()},
                checkpoint_path,
            )
            checkpoint = module.load_checkpoint_best(
                target_model,
                checkpoint_path,
                torch.device("cpu"),
            )

        self.assertIn("model_state_dict", checkpoint)

    def test_eval_script_can_load_gpt_neox_checkpoint(self):
        module = importlib.import_module("script.twd_tom.eval")
        config = module.load_config("configs/twd_tom_train.yaml")
        config["model"]["backbone_type"] = "gpt_neox"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8
        config["model"]["intermediate_size"] = 32
        source_model = module.build_model(config)
        target_model = module.build_model(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint_best.pt"
            torch.save(
                {"model_state_dict": source_model.state_dict()},
                checkpoint_path,
            )
            checkpoint = module.load_checkpoint_best(
                target_model,
                checkpoint_path,
                torch.device("cpu"),
            )

        self.assertIn("model_state_dict", checkpoint)

    def test_evaluate_model_returns_only_eval_loss_and_top2_f1(self):
        module = importlib.import_module("script.twd_tom.eval")

        class FixedModel(torch.nn.Module):
            def forward(self, event_tokens, attention_mask=None, observer_id=None):
                batch_size, seq_len, _ = event_tokens.shape
                wolf_logits = torch.full(
                    (batch_size, seq_len, 7),
                    -4.0,
                    device=event_tokens.device,
                )
                wolf_logits[:, :, 0] = 4.0
                wolf_logits[:, :, 1] = 3.0
                return {
                    "wolf_logits": wolf_logits,
                    "wolf_prob": torch.sigmoid(wolf_logits),
                    "region_probs": torch.full(
                        (batch_size, seq_len, 7, 3),
                        1.0 / 3.0,
                        device=event_tokens.device,
                    ),
                    "hard_region": torch.zeros(
                        batch_size,
                        seq_len,
                        7,
                        dtype=torch.long,
                        device=event_tokens.device,
                    ),
                }

        batch = {
            "event_tokens": torch.zeros(1, 1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 1),
            "observer_id": torch.tensor([1], dtype=torch.long),
            "wolf_labels": torch.tensor(
                [[1, 1, 0, 0, 0, 0, 0]],
                dtype=torch.float32,
            ),
        }
        loss_config = {
            "supervision_mode": "last",
            "bce_weight": 1.0,
            "cardinality_weight": 0.0,
            "num_wolves": 2.0,
            "region_weight": 0.0,
        }

        metrics = module.evaluate_model(
            FixedModel(),
            [batch],
            torch.device("cpu"),
            loss_config,
        )

        self.assertEqual(set(metrics), {"eval_loss", "top2_f1"})
        self.assertTrue(math.isfinite(metrics["eval_loss"]))
        self.assertEqual(metrics["top2_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
