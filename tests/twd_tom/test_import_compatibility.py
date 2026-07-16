import unittest


class ImportCompatibilityTest(unittest.TestCase):
    def test_legacy_model_imports_still_work(self):
        from werewolf.models.dialogue_actions import EVENT_TYPE2ID
        from werewolf.models.event_encoder import encode_observation_game_log
        from werewolf.models.speech_perceiver import SpeechPerceiver
        from werewolf.models.tom_backbone import ToMBackboneConfig
        from werewolf.models.twd_risk_layer import TWDRiskLayer
        from werewolf.models.twd_tom import TWDToMModel
        from werewolf.models.twd_tom_dataset import TWDToMDataset
        from werewolf.models.twd_tom_features import TWDToMFeatureBuilder
        from werewolf.models.twd_tom_losses import twd_tom_loss
        from werewolf.models.twd_tom_metrics import (
            compute_twd_tom_metrics,
        )

        self.assertIn("dialogue_action", EVENT_TYPE2ID)
        self.assertTrue(callable(encode_observation_game_log))
        self.assertTrue(callable(SpeechPerceiver))
        self.assertTrue(callable(ToMBackboneConfig))
        self.assertTrue(callable(TWDRiskLayer))
        self.assertTrue(callable(TWDToMModel))
        self.assertTrue(callable(TWDToMDataset))
        self.assertTrue(callable(TWDToMFeatureBuilder))
        self.assertTrue(callable(twd_tom_loss))
        self.assertTrue(callable(compute_twd_tom_metrics))


if __name__ == "__main__":
    unittest.main()
