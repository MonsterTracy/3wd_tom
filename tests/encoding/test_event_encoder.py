import unittest
from types import SimpleNamespace

from werewolf.encoding.dialogue_actions import (
    CAMP2ID,
    CERTAINTY2ID,
    EVENT_TYPE2ID,
    PHASE2ID,
    POLARITY2ID,
    PREDICATE2ID,
    ROLE2ID,
)
from werewolf.encoding.event_encoder import (
    encode_death,
    encode_dialogue_action,
    encode_exile,
    encode_observation_game_log,
    encode_private_check_result,
    encode_private_role_info,
    encode_private_wolf_team,
    encode_vote,
    get_log_field,
    validate_event_token,
)
from werewolf.models.twd_tom.features import TWDToMFeatureBuilder


TOKEN_FIELDS = {
    "event_type_id",
    "speaker_id",
    "target_id",
    "predicate_id",
    "role_id",
    "camp_id",
    "polarity_id",
    "certainty_id",
    "phase_id",
    "day_id",
}


class EventEncoderTest(unittest.TestCase):
    def test_get_log_field_supports_dict_and_object(self):
        object_log = SimpleNamespace(event="speech")

        self.assertEqual(get_log_field({"event": "vote"}, "event"), "vote")
        self.assertEqual(get_log_field(object_log, "event"), "speech")
        self.assertEqual(get_log_field(object_log, "missing", "fallback"), "fallback")

    def test_dialogue_action_keeps_one_based_ids_and_drops_text(self):
        token = encode_dialogue_action(
            {
                "speaker": 2,
                "predicate": "suspect",
                "target": 7,
                "role": "Werewolf",
                "camp": "Werewolf",
                "polarity": "negative",
                "certainty": "explicit",
                "condition": "如果7号继续跳身份",
                "source_text": "我怀疑7号",
            },
            day=1,
            phase="speech",
        )

        self.assertEqual(token["speaker_id"], 2)
        self.assertEqual(token["target_id"], 7)
        self.assertEqual(token["predicate_id"], PREDICATE2ID["suspect"])
        self.assertNotIn("condition", token)
        self.assertNotIn("source_text", token)
        self.assertEqual(set(token), TOKEN_FIELDS)

    def test_observation_encodes_multiple_parsed_claims(self):
        observation = {
            "game_log": [
                {
                    "source": 2,
                    "target": [1, 2, 3, 4, 5, 6, 7],
                    "day": 2,
                    "time": "speech",
                    "event": "speech",
                    "content": {
                        "speech_content": "我信3号，但怀疑6号",
                        "parsed_claims": [
                            {
                                "speaker": 2,
                                "predicate": "support",
                                "target": 3,
                                "role": None,
                                "camp": "Village",
                                "polarity": "positive",
                                "certainty": "explicit",
                                "source_text": "我信3号",
                            },
                            {
                                "speaker": 2,
                                "predicate": "suspect",
                                "target": 6,
                                "role": "Werewolf",
                                "camp": None,
                                "polarity": "negative",
                                "certainty": "hedge",
                                "source_text": "怀疑6号",
                            },
                        ],
                    },
                }
            ]
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(len(tokens), 2)
        self.assertEqual(
            [token["predicate_id"] for token in tokens],
            [PREDICATE2ID["support"], PREDICATE2ID["suspect"]],
        )
        self.assertTrue(all(validate_event_token(token) for token in tokens))
        self.assertTrue(all("speech_content" not in token for token in tokens))
        self.assertTrue(all("source_text" not in token for token in tokens))

    def test_empty_parsed_claims_produces_no_tokens(self):
        observation = {
            "game_log": [
                {
                    "day": 1,
                    "time": "speech",
                    "event": "speech",
                    "content": {"speech_content": "过", "parsed_claims": []},
                }
            ]
        }

        self.assertEqual(encode_observation_game_log(observation), [])

    def test_vote_and_pk_vote_are_encoded(self):
        vote = encode_vote(3, 5, 2)
        pk_vote = encode_vote(4, 6, 2, phase="vote_pk")

        self.assertEqual(vote["event_type_id"], EVENT_TYPE2ID["vote"])
        self.assertEqual(pk_vote["event_type_id"], EVENT_TYPE2ID["pk_vote"])
        self.assertEqual(vote["speaker_id"], 3)
        self.assertEqual(vote["target_id"], 5)
        self.assertEqual(vote["predicate_id"], PREDICATE2ID["vote"])
        self.assertEqual(vote["role_id"], ROLE2ID["None"])
        self.assertEqual(vote["camp_id"], CAMP2ID["None"])

    def test_exile_has_expected_fields(self):
        token = encode_exile(target=4, day=3)

        self.assertEqual(
            token,
            {
                "event_type_id": EVENT_TYPE2ID["exile"],
                "speaker_id": 0,
                "target_id": 4,
                "predicate_id": PREDICATE2ID["exile"],
                "role_id": ROLE2ID["None"],
                "camp_id": CAMP2ID["None"],
                "polarity_id": POLARITY2ID["negative"],
                "certainty_id": CERTAINTY2ID["explicit"],
                "phase_id": PHASE2ID["exile"],
                "day_id": 3,
            },
        )
        self.assertTrue(validate_event_token(token))

    def test_end_vote_with_expelled_encodes_exile(self):
        observation = {
            "game_log": [
                {
                    "source": 0,
                    "target": 4,
                    "day": 2,
                    "time": "vote",
                    "event": "end_vote",
                    "content": {"expelled": 4},
                }
            ]
        }

        tokens = encode_observation_game_log(observation)
        exile_tokens = [
            token
            for token in tokens
            if token["event_type_id"] == EVENT_TYPE2ID["exile"]
        ]

        self.assertEqual(
            [token["event_type_id"] for token in tokens],
            [EVENT_TYPE2ID["exile"]],
        )
        self.assertEqual(len(exile_tokens), 1)
        self.assertEqual(exile_tokens[0]["target_id"], 4)
        self.assertEqual(exile_tokens[0]["phase_id"], PHASE2ID["exile"])

    def test_invalid_exile_targets_produce_no_exile_tokens(self):
        observation = {
            "game_log": [
                {
                    "day": 2,
                    "event": "end_vote",
                    "content": {"vote_outcome": -1},
                },
                {
                    "day": 3,
                    "event": "exile",
                    "content": {"expelled": None},
                },
            ]
        }

        tokens = encode_observation_game_log(observation)

        self.assertFalse(
            any(
                token["event_type_id"] == EVENT_TYPE2ID["exile"]
                for token in tokens
            )
        )

    def test_observation_encodes_dict_vote_log(self):
        observation = {
            "game_log": [
                {
                    "source": 4,
                    "target": 2,
                    "day": 3,
                    "time": "vote",
                    "event": "vote",
                    "content": {"vote_target": 2},
                }
            ]
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["speaker_id"], 4)
        self.assertEqual(tokens[0]["target_id"], 2)
        self.assertEqual(tokens[0]["phase_id"], PHASE2ID["vote"])

    def test_death_and_observation_deaths_are_encoded(self):
        direct = encode_death(target=5, day=2)
        observation = {
            "game_log": [
                {
                    "source": 0,
                    "target": [3, 6],
                    "day": 2,
                    "time": "night_result",
                    "event": "end_night",
                    "content": {"dead_players": [3, 6]},
                }
            ]
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(direct["event_type_id"], EVENT_TYPE2ID["death"])
        self.assertEqual(direct["target_id"], 5)
        self.assertEqual(
            [token["target_id"] for token in tokens],
            [3, 6],
        )

    def test_night_result_with_dead_list_encodes_each_death(self):
        observation = {
            "game_log": [
                {
                    "day": 2,
                    "time": "night_result",
                    "event": "night_result",
                    "content": {"dead": [2, 5]},
                }
            ]
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(
            [token["target_id"] for token in tokens],
            [2, 5],
        )
        self.assertTrue(
            all(
                token["event_type_id"] == EVENT_TYPE2ID["death"]
                for token in tokens
            )
        )

    def test_content_target_alone_does_not_encode_death(self):
        observation = {
            "game_log": [
                {
                    "day": 2,
                    "time": "speech",
                    "event": "speech",
                    "content": {"target": 5},
                }
            ]
        }

        self.assertEqual(encode_observation_game_log(observation), [])

    def test_claim_and_exile_in_same_log_are_both_encoded(self):
        observation = {
            "game_log": [
                {
                    "day": 2,
                    "time": "exile",
                    "event": "end_vote",
                    "content": {
                        "expelled": 6,
                        "parsed_claims": [
                            {
                                "speaker": 3,
                                "predicate": "suspect",
                                "target": 6,
                                "certainty": "explicit",
                            }
                        ],
                    },
                }
            ]
        }

        tokens = encode_observation_game_log(observation)
        event_types = [token["event_type_id"] for token in tokens]

        self.assertIn(EVENT_TYPE2ID["dialogue_action"], event_types)
        self.assertIn(EVENT_TYPE2ID["exile"], event_types)

    def test_encode_observation_self_identity(self):
        observation = {
            "game_log": [
                {
                    "viewer": [4],
                    "source": 0,
                    "target": 4,
                    "day": 0,
                    "time": "night",
                    "event": "self_identity",
                    "content": {"identity": "Werewolf"},
                }
            ]
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(len(tokens), 1)
        self.assertEqual(
            tokens[0]["event_type_id"],
            EVENT_TYPE2ID["private_role_info"],
        )
        self.assertEqual(tokens[0]["target_id"], 4)
        self.assertEqual(tokens[0]["role_id"], ROLE2ID["Werewolf"])
        self.assertEqual(set(tokens[0]), TOKEN_FIELDS)

    def test_encode_observation_werewolf_team_info(self):
        observation = {
            "observer_id": 4,
            "game_log": [
                {
                    "viewer": [4, 7],
                    "source": 0,
                    "target": [4, 7],
                    "day": 0,
                    "time": "night",
                    "event": "werewolf_team_info",
                    "content": {"wolf_team": [4, 7]},
                }
            ],
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(
            [token["event_type_id"] for token in tokens],
            [
                EVENT_TYPE2ID["private_wolf_team"],
                EVENT_TYPE2ID["private_wolf_team"],
            ],
        )
        self.assertEqual([token["target_id"] for token in tokens], [4, 7])
        self.assertTrue(all(validate_event_token(token) for token in tokens))

    def test_private_info_real_initial_logs_not_empty(self):
        observation = {
            "observer_id": 4,
            "game_log": [
                {
                    "viewer": [1, 2, 3, 4, 5, 6, 7],
                    "source": 0,
                    "target": 0,
                    "day": 0,
                    "time": "night",
                    "event": "game_setting",
                    "content": {"Werewolf": 2, "Villager": 3},
                },
                {
                    "viewer": [4, 7],
                    "source": 0,
                    "target": [4, 7],
                    "day": 0,
                    "time": "night",
                    "event": "werewolf_team_info",
                    "content": {"wolf_team": [4, 7]},
                },
                {
                    "viewer": [4],
                    "source": 0,
                    "target": 4,
                    "day": 0,
                    "time": "night",
                    "event": "self_identity",
                    "content": {"identity": "Werewolf"},
                },
            ],
        }

        tokens = encode_observation_game_log(observation)
        features = TWDToMFeatureBuilder().encode_observation(observation)

        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(validate_event_token(token) for token in tokens))
        self.assertGreater(features["attention_mask"].sum().item(), 0)
        self.assertEqual(features["event_tokens"].shape[-1], 10)

    def test_private_check_result_has_expected_fields(self):
        token = encode_private_check_result(
            observer_id=1,
            target=6,
            role="Werewolf",
            camp="Werewolf",
            day=1,
        )

        self.assertEqual(
            token["event_type_id"],
            EVENT_TYPE2ID["private_check_result"],
        )
        self.assertEqual(token["speaker_id"], 1)
        self.assertEqual(token["target_id"], 6)
        self.assertEqual(token["role_id"], ROLE2ID["Werewolf"])
        self.assertEqual(token["camp_id"], CAMP2ID["Werewolf"])
        self.assertEqual(
            token["predicate_id"],
            PREDICATE2ID["report_check_result"],
        )

    def test_encode_observation_private_check_result(self):
        observation = {
            "observer_id": 2,
            "game_log": [
                {
                    "viewer": [2],
                    "source": 2,
                    "target": 5,
                    "day": 1,
                    "time": "night",
                    "event": "private_check_result",
                    "content": {"check_result": "Werewolf"},
                }
            ],
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(len(tokens), 1)
        self.assertEqual(
            tokens[0]["event_type_id"],
            EVENT_TYPE2ID["private_check_result"],
        )
        self.assertEqual(tokens[0]["speaker_id"], 2)
        self.assertEqual(tokens[0]["target_id"], 5)
        self.assertEqual(tokens[0]["role_id"], ROLE2ID["Werewolf"])

    def test_private_helpers_return_uniform_tokens(self):
        tokens = [
            encode_private_role_info(2, "Seer"),
            encode_private_check_result(2, 5),
            encode_private_wolf_team(1, 7),
        ]

        self.assertTrue(all(validate_event_token(token) for token in tokens))
        self.assertTrue(all(set(token) == TOKEN_FIELDS for token in tokens))

    def test_public_check_claim_remains_dialogue_action(self):
        observation = {
            "game_log": [
                {
                    "day": 1,
                    "time": "speech",
                    "event": "speech",
                    "content": {
                        "parsed_claims": [
                            {
                                "speaker": 3,
                                "predicate": "report_check_result",
                                "target": 5,
                                "role": "Werewolf",
                                "certainty": "explicit",
                            }
                        ]
                    },
                }
            ]
        }

        tokens = encode_observation_game_log(observation)

        self.assertEqual(len(tokens), 1)
        self.assertEqual(
            tokens[0]["event_type_id"],
            EVENT_TYPE2ID["dialogue_action"],
        )
        self.assertNotEqual(
            tokens[0]["event_type_id"],
            EVENT_TYPE2ID["private_check_result"],
        )

    def test_validate_event_token_rejects_missing_field(self):
        token = encode_death(target=4, day=1)
        del token["day_id"]

        self.assertFalse(validate_event_token(token))


if __name__ == "__main__":
    unittest.main()
