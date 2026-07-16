import inspect
import unittest

import torch

from werewolf.models.twd_tom.labels import (
    make_wolf_labels,
    wolf_indices_from_roles,
)


BASE_ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


class NamedRole:
    def __init__(self, name):
        self.name = name


class TWDToMLabelsTest(unittest.TestCase):
    def test_seven_roles_return_shape_seven(self):
        labels = make_wolf_labels(BASE_ROLES)

        self.assertEqual(labels.shape, (7,))

    def test_werewolves_are_one_and_other_roles_are_zero(self):
        labels = make_wolf_labels(BASE_ROLES)

        torch.testing.assert_close(
            labels,
            torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=torch.float32),
        )

    def test_role_index_zero_maps_to_player_one_label(self):
        roles = ["Werewolf"] + ["Villager"] * 6

        labels = make_wolf_labels(roles)

        self.assertEqual(labels[0].item(), 1.0)
        self.assertEqual(labels[1:].count_nonzero().item(), 0)

    def test_multiple_werewolves_are_marked(self):
        roles = [
            "Villager",
            "Werewolf",
            "Seer",
            "Werewolf",
            "Witch",
            "Villager",
            "Villager",
        ]

        labels = make_wolf_labels(roles)

        self.assertEqual(labels.nonzero().flatten().tolist(), [1, 3])

    def test_no_werewolves_returns_all_zero(self):
        labels = make_wolf_labels(["Villager"] * 7)

        self.assertEqual(labels.count_nonzero().item(), 0)

    def test_wrong_role_count_raises(self):
        with self.assertRaises(ValueError):
            make_wolf_labels(["Werewolf"] * 6)

    def test_dtype_parameter_is_used(self):
        labels = make_wolf_labels(BASE_ROLES, dtype=torch.float64)

        self.assertEqual(labels.dtype, torch.float64)

    def test_cpu_device_parameter_is_used(self):
        labels = make_wolf_labels(BASE_ROLES, device="cpu")

        self.assertEqual(labels.device, torch.device("cpu"))

    def test_role_object_name_is_recognized(self):
        roles = [NamedRole("Werewolf")] + ["Villager"] * 6

        labels = make_wolf_labels(roles)

        self.assertEqual(labels[0].item(), 1.0)

    def test_custom_wolf_role_names_are_normalized(self):
        roles = ["Wolf"] + ["Villager"] * 6

        labels = make_wolf_labels(
            roles,
            wolf_role_names=(NamedRole("Wolf"),),
        )

        self.assertEqual(labels[0].item(), 1.0)

    def test_wolf_indices_are_zero_based(self):
        indices = wolf_indices_from_roles(BASE_ROLES)

        self.assertEqual(indices, [0, 1])

    def test_lowercase_werewolf_does_not_match_default(self):
        roles = ["werewolf"] + ["Villager"] * 6

        labels = make_wolf_labels(roles)

        self.assertEqual(labels.count_nonzero().item(), 0)

    def test_lowercase_matches_when_explicitly_configured(self):
        roles = ["werewolf"] + ["Villager"] * 6

        labels = make_wolf_labels(
            roles,
            wolf_role_names=("werewolf",),
        )

        self.assertEqual(labels[0].item(), 1.0)

    def test_whitespace_padded_name_does_not_match_default(self):
        roles = [" Werewolf "] + ["Villager"] * 6

        labels = make_wolf_labels(roles)

        self.assertEqual(labels.count_nonzero().item(), 0)

    def test_public_functions_require_no_observation_or_log(self):
        label_parameters = tuple(
            inspect.signature(make_wolf_labels).parameters
        )
        index_parameters = tuple(
            inspect.signature(wolf_indices_from_roles).parameters
        )

        self.assertEqual(
            label_parameters,
            (
                "roles",
                "num_players",
                "wolf_role_names",
                "dtype",
                "device",
            ),
        )
        self.assertEqual(
            index_parameters,
            ("roles", "num_players", "wolf_role_names"),
        )


if __name__ == "__main__":
    unittest.main()
