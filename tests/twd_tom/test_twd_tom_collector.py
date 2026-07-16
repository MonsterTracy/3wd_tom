from contextlib import redirect_stdout
import io
import json
import os
import tempfile
import unittest

import run_battle
import run_random

from werewolf.helper.log_utils import Log
from werewolf.models.twd_tom.collector import TWDToMSampleCollector


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


class UnsupportedValue:
    pass


class RecordingCollector:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def record(self, observation, roles, step_idx=None):
        self.events.append(("record", observation["marker"], step_idx))
        self.calls.append((observation, roles, step_idx))


class RecordingAgent:
    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append(("agent_reset",))

    def act(self, observation):
        self.events.append(("act", observation["marker"]))
        return ("vote", -1)


class RecordingEnv:
    def __init__(self, events):
        self.events = events
        self.step_count = 0

    @property
    def game_log(self):
        raise AssertionError("eval must not read env.game_log")

    def reset(self, roles):
        self.events.append(("env_reset",))
        return {
            "current_act_idx": 1,
            "phase": "0_night_skill_wolf",
            "game_log": [],
            "marker": 0,
        }

    def step(self, action):
        self.events.append(("step", self.step_count))
        self.step_count += 1
        done = self.step_count == 2
        observation = {
            "current_act_idx": 1,
            "phase": "end_game" if done else "0_night_skill_wolf",
            "game_log": [],
            "marker": self.step_count,
        }
        info = {"Werewolf": -1} if done else {}
        return observation, None, done, info


class TWDToMSampleCollectorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_path = os.path.join(
            self.temp_dir.name,
            "samples.jsonl",
        )
        self.observation = {
            "current_act_idx": 3,
            "phase": "1_day_speech",
            "game_log": [],
        }

    def make_collector(self, output_path=None, game_id="game-1"):
        collector = TWDToMSampleCollector(
            output_path=output_path or self.output_path,
            game_id=game_id,
        )
        self.addCleanup(collector.close)
        return collector

    def read_lines(self, output_path=None):
        with open(
            output_path or self.output_path,
            "r",
            encoding="utf-8",
        ) as file:
            return [json.loads(line) for line in file]

    def test_record_writes_jsonl_and_returns_sample(self):
        collector = self.make_collector()

        sample = collector.record(
            self.observation,
            ROLES,
            step_idx=4,
        )
        rows = self.read_lines()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], sample)

    def test_jsonl_contains_required_fields_and_no_roles(self):
        collector = self.make_collector()

        collector.record(self.observation, ROLES, step_idx=2)
        row = self.read_lines()[0]

        self.assertEqual(
            set(row),
            {
                "game_id",
                "observer_id",
                "phase",
                "observation",
                "wolf_labels",
                "alive_mask",
                "step_idx",
            },
        )
        self.assertEqual(row["game_id"], "game-1")
        self.assertEqual(row["observer_id"], 3)
        self.assertEqual(row["phase"], "1_day_speech")
        self.assertEqual(row["step_idx"], 2)
        self.assertEqual(
            row["wolf_labels"],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertNotIn("roles", row)

    def test_record_uses_observation_alive_mask(self):
        collector = self.make_collector()
        observation = dict(self.observation)
        observation["alive_mask"] = [1, 1, 0, 1, 0, 1, 1]

        sample = collector.record(observation, ROLES)

        self.assertEqual(
            sample["alive_mask"],
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        )

    def test_record_flushes_before_close(self):
        collector = self.make_collector()

        collector.record(self.observation, ROLES)

        self.assertEqual(len(self.read_lines()), 1)

    def test_missing_parent_directory_is_created(self):
        output_path = os.path.join(
            self.temp_dir.name,
            "nested",
            "rollout",
            "samples.jsonl",
        )
        collector = self.make_collector(output_path=output_path)

        collector.record(self.observation, ROLES)

        self.assertTrue(os.path.isfile(output_path))

    def test_existing_output_is_appended_not_overwritten(self):
        first = self.make_collector()
        first.record(self.observation, ROLES, step_idx=0)
        first.close()
        second = self.make_collector(game_id="game-2")

        second.record(self.observation, ROLES, step_idx=1)

        self.assertEqual(
            [row["game_id"] for row in self.read_lines()],
            ["game-1", "game-2"],
        )

    def test_log_is_serialized_without_mutating_returned_sample(self):
        log = Log(
            viewer=[3],
            source=2,
            target=-1,
            content={"parsed_claims": []},
            day=1,
            time="第1天白天",
            event="speech",
        )
        observation = {
            "current_act_idx": 3,
            "phase": "1_day_speech",
            "game_log": [log],
        }
        collector = self.make_collector()

        sample = collector.record(observation, ROLES)
        serialized_log = self.read_lines()[0]["observation"]["game_log"][0]

        self.assertIsInstance(sample["observation"]["game_log"][0], Log)
        self.assertIsNot(sample["observation"]["game_log"][0], log)
        self.assertEqual(serialized_log, log.__dict__)

    def test_unsupported_non_log_value_raises_type_error(self):
        observation = dict(self.observation)
        observation["unsupported"] = UnsupportedValue()
        collector = self.make_collector()

        with self.assertRaises(TypeError):
            collector.record(observation, ROLES)

        self.assertEqual(os.path.getsize(self.output_path), 0)

    def test_close_is_idempotent(self):
        collector = self.make_collector()

        collector.close()
        collector.close()

        self.assertTrue(collector._file.closed)


class RolloutCollectorIntegrationTest(unittest.TestCase):
    def assert_eval_records_before_act(self, eval_function):
        events = []
        env = RecordingEnv(events)
        agent = RecordingAgent(events)
        collector = RecordingCollector(events)

        with redirect_stdout(io.StringIO()):
            result = eval_function(
                env,
                [agent],
                ROLES,
                sample_collector=collector,
            )

        rollout_events = [
            event
            for event in events
            if event[0] in {"record", "act", "step"}
        ]
        self.assertEqual(
            rollout_events,
            [
                ("record", 0, 0),
                ("act", 0),
                ("step", 0),
                ("record", 1, 1),
                ("act", 1),
                ("step", 1),
            ],
        )
        self.assertEqual(
            [call[2] for call in collector.calls],
            [0, 1],
        )
        self.assertTrue(
            all(call[1] is ROLES for call in collector.calls)
        )
        self.assertEqual(result, "Villager win")

    def test_run_random_eval_records_each_pre_act_observation(self):
        self.assert_eval_records_before_act(run_random.eval)

    def test_run_battle_eval_records_each_pre_act_observation(self):
        self.assert_eval_records_before_act(run_battle.eval)

    def test_none_collector_preserves_existing_eval_behavior(self):
        for eval_function in (run_random.eval, run_battle.eval):
            with self.subTest(eval_function=eval_function.__module__):
                events = []
                with redirect_stdout(io.StringIO()):
                    result = eval_function(
                        RecordingEnv(events),
                        [RecordingAgent(events)],
                        ROLES,
                    )

                self.assertEqual(result, "Villager win")
                self.assertEqual(
                    [event[0] for event in events].count("act"),
                    2,
                )


if __name__ == "__main__":
    unittest.main()
