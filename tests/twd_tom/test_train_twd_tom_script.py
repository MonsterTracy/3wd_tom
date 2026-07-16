import importlib
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrainTWDToMScriptTest(unittest.TestCase):
    def test_config_can_be_loaded_with_yaml_safe_load(self):
        config_path = PROJECT_ROOT / "configs" / "twd_tom_train.yaml"

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(
            config["data"]["source_path"],
            "data/twd_tom/debug/game_001_030.jsonl",
        )
        self.assertEqual(config["model"]["num_players"], 7)
        self.assertEqual(config["loss"]["supervision_mode"], "last")
        self.assertEqual(
            config["checkpoint"]["monitor_metric"],
            "eval_loss",
        )
        self.assertEqual(config["checkpoint"]["mode"], "min")
        self.assertEqual(config["model"]["backbone_type"], "transformer")
        self.assertTrue(config["model"]["use_observer_id"])

    def test_llama_config_can_be_loaded_with_yaml_safe_load(self):
        config_path = PROJECT_ROOT / "configs" / "twd_tom_train_llama.yaml"

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(config["model"]["backbone_type"], "llama")
        self.assertEqual(config["model"]["intermediate_size"], 512)
        self.assertEqual(config["model"]["rope_theta"], 10000.0)
        self.assertTrue(config["model"]["use_observer_id"])
        self.assertEqual(
            config["checkpoint"]["monitor_metric"],
            "eval_loss",
        )
        self.assertEqual(config["checkpoint"]["mode"], "min")

    def test_gpt_neox_config_can_be_loaded_with_yaml_safe_load(self):
        config_path = PROJECT_ROOT / "configs" / "twd_tom_train_gpt_neox.yaml"

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(config["model"]["backbone_type"], "gpt_neox")
        self.assertEqual(config["model"]["intermediate_size"], 512)
        self.assertEqual(config["model"]["rope_theta"], 10000.0)
        self.assertTrue(config["model"]["use_observer_id"])
        self.assertEqual(
            config["checkpoint"]["monitor_metric"],
            "eval_loss",
        )
        self.assertEqual(config["checkpoint"]["mode"], "min")

    def test_training_script_can_be_imported(self):
        module = importlib.import_module("script.twd_tom.train")

        self.assertTrue(callable(module.split_samples_by_game_id))

    def test_build_model_accepts_boe_mlp_backbone_config(self):
        module = importlib.import_module("script.twd_tom.train")
        config = module.load_config(PROJECT_ROOT / "configs" / "twd_tom_train.yaml")
        config["model"]["backbone_type"] = "boe_mlp"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "boe_mlp")

    def test_build_model_accepts_gru_backbone_config(self):
        module = importlib.import_module("script.twd_tom.train")
        config = module.load_config(PROJECT_ROOT / "configs" / "twd_tom_train.yaml")
        config["model"]["backbone_type"] = "gru"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "gru")

    def test_build_model_accepts_llama_backbone_config(self):
        module = importlib.import_module("script.twd_tom.train")
        config = module.load_config(PROJECT_ROOT / "configs" / "twd_tom_train.yaml")
        config["model"]["backbone_type"] = "llama"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8
        config["model"]["intermediate_size"] = 32
        config["model"]["rope_theta"] = 5000.0

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "llama")
        self.assertEqual(model.config.tom_config.intermediate_size, 32)
        self.assertEqual(model.config.tom_config.rope_theta, 5000.0)

    def test_build_model_accepts_gpt_neox_backbone_config(self):
        module = importlib.import_module("script.twd_tom.train")
        config = module.load_config(PROJECT_ROOT / "configs" / "twd_tom_train.yaml")
        config["model"]["backbone_type"] = "gpt_neox"
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8
        config["model"]["intermediate_size"] = 32
        config["model"]["rope_theta"] = 5000.0

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "gpt_neox")
        self.assertEqual(model.config.tom_config.intermediate_size, 32)
        self.assertEqual(model.config.tom_config.rope_theta, 5000.0)

    def test_build_model_defaults_missing_backbone_type_to_transformer(self):
        module = importlib.import_module("script.twd_tom.train")
        config = module.load_config(PROJECT_ROOT / "configs" / "twd_tom_train.yaml")
        del config["model"]["backbone_type"]
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8

        model = module.build_model(config)

        self.assertEqual(model.config.tom_config.backbone_type, "transformer")

    def test_build_model_accepts_use_observer_id_false(self):
        module = importlib.import_module("script.twd_tom.train")
        config = module.load_config(PROJECT_ROOT / "configs" / "twd_tom_train.yaml")
        config["model"]["use_observer_id"] = False
        config["model"]["d_model"] = 16
        config["model"]["n_head"] = 4
        config["model"]["n_layer"] = 1
        config["model"]["max_seq_len"] = 8

        model = module.build_model(config)

        self.assertFalse(model.config.tom_config.use_observer_id)

    def test_loss_summary_uses_only_onuw_style_loss_fields(self):
        module = importlib.import_module("script.twd_tom.train")

        summary = module.format_loss_summary(
            epoch=1,
            train_loss=0.612345,
            eval_loss=0.598102,
            lowest_eval_loss=0.598102,
        )

        self.assertEqual(
            summary,
            "epoch=1 train_loss=0.612345 "
            "eval_loss=0.598102 "
            "lowest_eval_loss=0.598102",
        )
        forbidden_eval_alias = "validation" + "_loss"
        forbidden_lowest_alias = "lowest_" + "validation" + "_loss"
        self.assertNotIn(forbidden_eval_alias, summary)
        self.assertNotIn(forbidden_lowest_alias, summary)
        self.assertNotIn("top2", summary)
        self.assertNotIn("binary_accuracy", summary)

    def test_history_record_contains_loss_only_fields(self):
        module = importlib.import_module("script.twd_tom.train")

        record = module.build_history_record(
            epoch=1,
            global_step=44,
            train_loss=0.612345,
            eval_loss=0.598102,
            lowest_eval_loss=0.598102,
        )

        self.assertEqual(
            set(record),
            {
                "epoch",
                "global_step",
                "train_loss",
                "eval_loss",
                "lowest_eval_loss",
            },
        )
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["global_step"], 44)
        self.assertEqual(record["train_loss"], 0.612345)
        self.assertEqual(record["eval_loss"], 0.598102)
        self.assertEqual(record["lowest_eval_loss"], 0.598102)

    def test_checkpoint_contains_only_loss_state_fields(self):
        module = importlib.import_module("script.twd_tom.train")
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            module.save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=1,
                global_step=44,
                best_metric=0.598102,
                config_dict={"checkpoint": {"monitor_metric": "eval_loss"}},
                train_loss=0.612345,
                eval_loss=0.598102,
                lowest_eval_loss=0.598102,
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(
            set(checkpoint),
            {
                "model_state_dict",
                "optimizer_state_dict",
                "epoch",
                "global_step",
                "best_metric",
                "config_dict",
                "train_loss",
                "eval_loss",
                "lowest_eval_loss",
            },
        )
        forbidden_eval_alias = "validation" + "_loss"
        forbidden_lowest_alias = "lowest_" + "validation" + "_loss"
        self.assertNotIn(forbidden_eval_alias, checkpoint)
        self.assertNotIn(forbidden_lowest_alias, checkpoint)
        self.assertNotIn("train_metrics", checkpoint)
        self.assertNotIn("val_metrics", checkpoint)
        self.assertEqual(checkpoint["best_metric"], checkpoint["lowest_eval_loss"])

    def test_best_checkpoint_logic_uses_eval_loss_min(self):
        module = importlib.import_module("script.twd_tom.train")

        first_value = module.get_checkpoint_monitor_value(
            train_loss=0.7,
            eval_loss=0.6,
            monitor_metric="eval_loss",
        )
        self.assertEqual(first_value, 0.6)
        self.assertTrue(module._is_improved(first_value, None, "min"))

        self.assertFalse(module._is_improved(0.65, first_value, "min"))
        self.assertTrue(module._is_improved(0.55, first_value, "min"))

    def test_game_id_split_keeps_each_game_in_one_partition(self):
        module = importlib.import_module("script.twd_tom.train")
        samples = [
            {
                "game_id": f"game-{game_id}",
                "sample_id": f"{game_id}-{sample_id}",
            }
            for game_id in range(1, 6)
            for sample_id in range(3)
        ]

        train_samples, val_samples = module.split_samples_by_game_id(
            samples,
            val_ratio=0.2,
            seed=42,
        )

        train_games = {sample["game_id"] for sample in train_samples}
        val_games = {sample["game_id"] for sample in val_samples}
        self.assertFalse(train_games & val_games)
        self.assertEqual(
            train_games | val_games,
            {f"game-{game_id}" for game_id in range(1, 6)},
        )
        self.assertGreater(len(train_games), 0)
        self.assertGreater(len(val_games), 0)

    def test_single_game_split_allows_empty_validation(self):
        module = importlib.import_module("script.twd_tom.train")
        samples = [
            {"game_id": "only-game", "sample_id": "a"},
            {"game_id": "only-game", "sample_id": "b"},
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_samples, val_samples = module.split_samples_by_game_id(
                samples,
                val_ratio=0.2,
                seed=42,
            )

        self.assertEqual(train_samples, samples)
        self.assertEqual(val_samples, [])
        self.assertTrue(
            any("only one game_id" in str(item.message) for item in caught)
        )


if __name__ == "__main__":
    unittest.main()
