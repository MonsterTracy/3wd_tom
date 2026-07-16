import unittest

from werewolf.models.twd_tom.samples import make_twd_tom_sample


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


class TWDToMSamplesTest(unittest.TestCase):
    def test_returns_exact_fields(self):
        sample = make_twd_tom_sample({}, ROLES)

        self.assertEqual(
            set(sample),
            {
                "game_id",
                "observer_id",
                "phase",
                "observation",
                "wolf_labels",
                "alive_mask",
            },
        )

    def test_alive_mask_is_stored_as_json_friendly_floats(self):
        alive_mask = [1, 1, 0, 1, 0, 1, 1]

        sample = make_twd_tom_sample(
            {},
            ROLES,
            alive_mask=alive_mask,
        )

        self.assertEqual(
            sample["alive_mask"],
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        )

    def test_alive_mask_defaults_to_all_alive(self):
        sample = make_twd_tom_sample({}, ROLES)

        self.assertEqual(sample["alive_mask"], [1.0] * 7)

    def test_alive_mask_is_derived_from_observation_game_log(self):
        observation = {
            "game_log": [
                {
                    "event": "night_result",
                    "content": {"dead": [2, 5]},
                },
                {
                    "event": "end_vote",
                    "content": {"expelled": 6},
                },
            ]
        }

        sample = make_twd_tom_sample(observation, ROLES)

        self.assertEqual(
            sample["alive_mask"],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )

    def test_wolf_labels_are_correct_json_friendly_floats(self):
        sample = make_twd_tom_sample({}, ROLES)

        self.assertEqual(
            sample["wolf_labels"],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertTrue(
            all(isinstance(label, float) for label in sample["wolf_labels"])
        )

    def test_roles_are_not_stored_in_sample(self):
        sample = make_twd_tom_sample({}, ROLES)

        self.assertNotIn("roles", sample)

    def test_observer_id_defaults_from_observation(self):
        sample = make_twd_tom_sample({"current_act_idx": 3}, ROLES)

        self.assertEqual(sample["observer_id"], 3)

    def test_phase_defaults_from_observation(self):
        sample = make_twd_tom_sample({"phase": "1_day_speech"}, ROLES)

        self.assertEqual(sample["phase"], "1_day_speech")

    def test_explicit_metadata_overrides_observation(self):
        observation = {
            "current_act_idx": 3,
            "phase": "1_day_speech",
        }

        sample = make_twd_tom_sample(
            observation,
            ROLES,
            game_id="game-1",
            observer_id=5,
            phase="1_day_vote",
        )

        self.assertEqual(sample["game_id"], "game-1")
        self.assertEqual(sample["observer_id"], 5)
        self.assertEqual(sample["phase"], "1_day_vote")

    def test_missing_metadata_defaults_remain_none(self):
        sample = make_twd_tom_sample({}, ROLES)

        self.assertIsNone(sample["observer_id"])
        self.assertIsNone(sample["phase"])

    def test_observation_is_deep_copied(self):
        observation = {
            "private_information": {
                "checked_players": [2],
            },
        }

        sample = make_twd_tom_sample(observation, ROLES)
        observation["private_information"]["checked_players"].append(4)

        self.assertIsNot(sample["observation"], observation)
        self.assertEqual(
            sample["observation"]["private_information"]["checked_players"],
            [2],
        )

    def test_invalid_role_count_value_error_is_propagated(self):
        with self.assertRaises(ValueError):
            make_twd_tom_sample({}, ROLES[:-1])

    def test_requires_no_env_game_log_or_event_encoder(self):
        sample = make_twd_tom_sample(
            {"current_act_idx": 1, "phase": "night"},
            ROLES,
        )

        self.assertEqual(sample["observer_id"], 1)
        self.assertEqual(sample["observation"]["phase"], "night")


if __name__ == "__main__":
    unittest.main()
