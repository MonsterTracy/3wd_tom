import json
import unittest

try:
    from werewolf.models import SpeechPerceiver
except ModuleNotFoundError:
    SpeechPerceiver = None


class FakeBackend:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                **kwargs,
            }
        )
        if self.error is not None:
            raise self.error
        return self.content


class SpeechPerceiverAvailabilityTest(unittest.TestCase):
    def test_speech_perceiver_is_available(self):
        self.assertIsNotNone(SpeechPerceiver)


@unittest.skipIf(SpeechPerceiver is None, "SpeechPerceiver is not implemented")
class SpeechPerceiverTest(unittest.TestCase):
    def test_calls_backend_with_prompt_and_normalizes_claim(self):
        speech = "我是预言家，3号是狼人。"
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "speaker": 7,
                        "predicate": "accuse_as_werewolf",
                        "target": 3,
                        "role": "Werewolf",
                        "polarity": "positive",
                        "certainty": "certain",
                        "condition": None,
                        "source_text": "",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        claims = perceiver.parse(
            speaker=2,
            speech=speech,
            day=1,
            phase="speech",
        )

        self.assertEqual(
            claims,
            [
                {
                    "speaker": 2,
                    "predicate": "accuse_as_werewolf",
                    "target": 3,
                    "role": "Werewolf",
                    "camp": None,
                    "polarity": "positive",
                    "certainty": "implicit",
                    "condition": None,
                    "source_text": speech,
                }
            ],
        )
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["messages"][0]["role"], "user")
        prompt = call["messages"][0]["content"]
        self.assertIn("你是狼人杀发言结构化解析器。", prompt)
        self.assertIn("当前玩家：player2", prompt)
        self.assertIn("当前天数：Day 1", prompt)
        self.assertIn("当前阶段：speech", prompt)
        self.assertIn("玩家编号：1 到 7", prompt)
        self.assertIn("claim_role", prompt)
        self.assertIn("Werewolf", prompt)
        self.assertIn(speech, prompt)
        self.assertTrue(prompt.rstrip().endswith(speech))

    def test_parses_json_code_fence(self):
        backend = FakeBackend(
            """```json
[
  {
    "speaker": 1,
    "predicate": "claim_role",
    "target": null,
    "role": "Seer",
    "polarity": "neutral",
    "certainty": "explicit",
    "condition": null,
    "source_text": "我是预言家"
  }
]
```"""
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        claims = perceiver.parse(1, "我是预言家", 1, "speech")

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["predicate"], "claim_role")
        self.assertEqual(claims[0]["role"], "Seer")

    def test_extracts_first_json_array_from_ordinary_text(self):
        backend = FakeBackend(
            '解析结果如下：[{"predicate":"support","target":4,'
            '"role":null,"polarity":"positive","certainty":"explicit",'
            '"condition":null,"source_text":"我支持4号"}]。'
            '备选：[{"predicate":"suspect","target":5}]'
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        claims = perceiver.parse(2, "我支持4号", 1, "speech")

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["predicate"], "support")
        self.assertEqual(claims[0]["target"], 4)

    def test_filters_invalid_predicate_and_normalizes_invalid_fields(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {"predicate": "invented_claim", "target": 1},
                    {
                        "predicate": "question",
                        "target": 8,
                        "role": "Hunter",
                        "camp": "Neutral",
                        "polarity": "maybe",
                        "certainty": "hedge",
                        "condition": 123,
                        "source_text": "为什么要投我？",
                    },
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        claims = perceiver.parse(5, "为什么要投我？", 2, "speech_pk")

        self.assertEqual(
            claims,
            [
                {
                    "speaker": 5,
                    "predicate": "question",
                    "target": None,
                    "role": None,
                    "camp": None,
                    "polarity": None,
                    "certainty": "hedge",
                    "condition": None,
                    "source_text": "为什么要投我？",
                }
            ],
        )

    def test_claim_role_target_is_forced_to_speaker(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "speaker": 4,
                        "predicate": "claim_role",
                        "target": 6,
                        "role": "Seer",
                        "polarity": "positive",
                        "certainty": "explicit",
                        "condition": None,
                        "source_text": "我是预言家",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="parser-model")

        claims = perceiver.parse(4, "我是预言家", 1, "speech")

        self.assertEqual(claims[0]["target"], 4)

    def test_good_person_claim_is_village_camp_not_villager_role(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "speaker": 3,
                        "predicate": "claim_camp",
                        "target": 7,
                        "role": "Villager",
                        "camp": "Village",
                        "polarity": "positive",
                        "certainty": "explicit",
                        "condition": None,
                        "source_text": "我是好人",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="parser-model")

        claims = perceiver.parse(3, "我是好人", 1, "speech")

        self.assertEqual(claims[0]["predicate"], "claim_camp")
        self.assertEqual(claims[0]["target"], 3)
        self.assertIsNone(claims[0]["role"])
        self.assertEqual(claims[0]["camp"], "Village")
        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertIn("claim_camp", prompt)
        self.assertIn("“我是好人”“我是好人阵营”“我是站好人边的”", prompt)
        self.assertIn("不要解析为 role=\"Villager\"", prompt)
        self.assertIn("“我是狼人阵营”“我是狼队的”", prompt)

    def test_explicit_villager_claim_remains_villager_role(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "speaker": 2,
                        "predicate": "claim_role",
                        "target": None,
                        "role": "Villager",
                        "polarity": "positive",
                        "certainty": "explicit",
                        "condition": None,
                        "source_text": "我是普通村民",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="parser-model")

        claims = perceiver.parse(2, "我是普通村民", 1, "speech")

        self.assertEqual(claims[0]["predicate"], "claim_role")
        self.assertEqual(claims[0]["target"], 2)
        self.assertEqual(claims[0]["role"], "Villager")
        self.assertIsNone(claims[0]["camp"])

    def test_accuse_as_werewolf_role_is_forced_to_werewolf(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "speaker": 1,
                        "predicate": "accuse_as_werewolf",
                        "target": 3,
                        "role": "Villager",
                        "polarity": "negative",
                        "certainty": "explicit",
                        "condition": None,
                        "source_text": "3号是狼人",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="parser-model")

        claims = perceiver.parse(1, "3号是狼人", 1, "speech")

        self.assertEqual(claims[0]["role"], "Werewolf")

    def test_quoted_check_result_is_not_report_check_result(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "predicate": "support",
                        "target": 6,
                        "role": None,
                        "polarity": "positive",
                        "certainty": "explicit",
                        "condition": None,
                        "source_text": "所以我信6号",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="parser-model")

        claims = perceiver.parse(3, "6号给我发金水，所以我信6号", 1, "speech")

        self.assertNotIn(
            "report_check_result",
            [claim["predicate"] for claim in claims],
        )
        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertIn(
            "report_check_result 只用于发言者自己声称",
            prompt,
        )
        self.assertIn(
            "6号给我发金水",
            prompt,
        )
        self.assertIn(
            "不要解析成 report_check_result",
            prompt,
        )

    def test_following_player_vote_is_follow_vote(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "predicate": "follow_vote",
                        "target": 2,
                        "role": None,
                        "polarity": "neutral",
                        "certainty": "explicit",
                        "condition": None,
                        "source_text": "我跟着2号归票",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="parser-model")

        claims = perceiver.parse(1, "我跟着2号归票", 1, "speech")

        self.assertEqual(claims[0]["predicate"], "follow_vote")
        self.assertEqual(claims[0]["target"], 2)
        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertIn(
            "跟着X归票 / 听X归票 / 跟X投 / 跟X思路走",
            prompt,
        )
        self.assertIn("target 是被跟随的人", prompt)

    def test_direct_vote_is_vote_intention(self):
        backend = FakeBackend(
            json.dumps(
                [
                    {
                        "predicate": "vote_intention",
                        "target": 2,
                        "role": None,
                        "polarity": "negative",
                        "certainty": "explicit",
                        "condition": None,
                        "source_text": "今天先投2号",
                    }
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="parser-model")

        claims = perceiver.parse(1, "今天先投2号", 1, "speech")

        self.assertEqual(claims[0]["predicate"], "vote_intention")
        self.assertEqual(claims[0]["target"], 2)
        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertIn(
            "投X / 出X / 票X / 归票X / 今天先投X",
            prompt,
        )
        self.assertIn("target 是被投票对象", prompt)

    def test_returns_empty_list_without_backend_or_model(self):
        self.assertEqual(
            SpeechPerceiver(model_name="test-model").parse(
                1, "发言", 1, "speech"
            ),
            [],
        )
        backend = FakeBackend("[]")
        self.assertEqual(
            SpeechPerceiver(backend=backend).parse(1, "发言", 1, "speech"),
            [],
        )
        self.assertEqual(backend.calls, [])

    def test_returns_empty_list_when_backend_raises(self):
        backend = FakeBackend(error=RuntimeError("backend unavailable"))
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        self.assertEqual(perceiver.parse(1, "发言", 1, "speech"), [])

    def test_returns_empty_list_for_malformed_json(self):
        backend = FakeBackend("not JSON")
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        self.assertEqual(perceiver.parse(1, "发言", 1, "speech"), [])


if __name__ == "__main__":
    unittest.main()
