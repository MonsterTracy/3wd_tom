"""Seven-player text Werewolf environment backed by unified structured events."""

import json
import random
from collections import Counter
from pathlib import Path

from werewolf.events.environment_events import (
    check_result_event,
    death_event,
    exile_event,
    guard_result_event,
    outcome_event,
    private_action_event,
    role_reveal_event,
    self_role_event,
    setting_event,
    speech_event,
    vote_event,
    vote_result_event,
    witch_state_event,
    wolf_team_event,
)
from werewolf.events.streams import visible_events


SUPPORTED_ROLES = {"Werewolf", "Seer", "Witch", "Guard", "Villager"}


class WerewolfTextEnvV0:
    """Fixed 2-wolf/1-seer/3-villager/1-witch-or-guard game."""

    def __init__(self, **kwargs):
        self.n_player = kwargs.get("n_player", 7)
        self.n_role = kwargs.get("n_role", 4)
        self.n_werewolf = kwargs.get("n_werewolf", 2)
        self.n_seer = kwargs.get("n_seer", 1)
        self.n_guard = kwargs.get("n_guard", 0)
        self.n_witch = kwargs.get("n_witch", 1)
        self.n_hunter = kwargs.get("n_hunter", 0)
        self.n_villager = kwargs.get("n_villager", 3)
        self._validate_7p_config()
        self.base_roles = (
            ["Werewolf"] * self.n_werewolf
            + ["Seer"] * self.n_seer
            + ["Guard"] * self.n_guard
            + ["Witch"] * self.n_witch
            + ["Villager"] * self.n_villager
        )
        self.roles = list(self.base_roles)
        self.werewolf_reward = kwargs.get("werewolf_reward", 1)
        self.village_reward = kwargs.get("village_reward", 1)
        self.log_save_path = kwargs.get("log_save_path")
        self.speech_parser = kwargs.get("speech_parser")
        self.tom_collector = kwargs.get("tom_collector")
        self.rng = random.Random(kwargs.get("random_seed"))
        self.game_count = 0
        self.wolf_win_count = 0
        self.collection_batches = []
        self.parser_failures = []

    def _validate_7p_config(self):
        fixed = {
            "n_player": 7,
            "n_role": 4,
            "n_werewolf": 2,
            "n_seer": 1,
            "n_villager": 3,
            "n_hunter": 0,
        }
        for name, expected in fixed.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"{name} must be {expected}, got {actual}")
        if self.n_guard not in (0, 1) or self.n_witch not in (0, 1):
            raise ValueError("n_guard and n_witch must each be zero or one")
        if self.n_guard + self.n_witch != 1:
            raise ValueError("exactly one of n_guard and n_witch must be one")

    @staticmethod
    def _validate_roles(roles):
        if len(roles) != 7:
            raise ValueError("roles must contain exactly seven entries")
        counts = Counter(roles)
        if set(counts) - SUPPORTED_ROLES:
            raise ValueError(f"unsupported roles: {sorted(set(counts) - SUPPORTED_ROLES)}")
        if counts["Werewolf"] != 2 or counts["Seer"] != 1 or counts["Villager"] != 3:
            raise ValueError("roles require two wolves, one seer, and three villagers")
        if counts["Witch"] + counts["Guard"] != 1:
            raise ValueError("roles require exactly one witch or guard")

    def set_tom_collector(self, collector):
        self.tom_collector = collector

    def _role(self, player_id):
        return self.roles[player_id - 1]

    def _phase_label(self):
        return f"{self.day}_{self.day_or_night}_{self.phase}"

    def _next_event_id(self, prefix="e"):
        self.event_counter += 1
        return f"g{self.game_count}.{prefix}{self.event_counter:05d}"

    def _emit(self, builder, **kwargs):
        event = builder(
            event_id=self._next_event_id(),
            day=self.day,
            phase=self._phase_label(),
            turn=self.event_counter,
            **kwargs,
        )
        self.events.append(event)
        return event

    def _collect(self, event):
        if self.tom_collector is None:
            return
        batch = self.tom_collector.collect(
            trigger_event=event,
            events=self.events,
            alive_players=sorted(self.alive),
        )
        self.collection_batches.append(batch)

    def reset(self, **kwargs):
        self.game_count += 1
        roles = list(kwargs["roles"]) if "roles" in kwargs else list(self.base_roles)
        if "roles" not in kwargs:
            self.rng.shuffle(roles)
        self._validate_roles(roles)
        self.roles = roles
        self.wolves = [player_id for player_id in range(1, 8) if self._role(player_id) == "Werewolf"]
        self.seer = self.roles.index("Seer") + 1
        self.guard = self.roles.index("Guard") + 1 if "Guard" in self.roles else None
        self.witch = self.roles.index("Witch") + 1 if "Witch" in self.roles else None
        self.alive = set(range(1, 8))
        self.day = 0
        self.day_or_night = "night"
        self.phase = "init"
        self.current_player = None
        self.event_counter = 0
        self.events = []
        self.parser_failures = []
        self.collection_batches = []
        self.speech_queue = []
        self.vote_queue = []
        self.vote_pk_players = []
        self.night_wolf_choices = []
        self.night_kill = None
        self.seer_checked = set()
        self.guard_history = []
        self.guard_target = None
        self.witch_heal_used = False
        self.witch_poison_used = False
        self.witch_heal_target = None
        self.witch_poison_target = None
        self.current_votes = {}

        self._emit(
            setting_event,
            value=None,
            metadata={
                "roles": dict(Counter(self.roles)),
                "players": 7,
                "wolves": 2,
            },
        )
        for player_id, role in enumerate(self.roles, start=1):
            self._emit(
                self_role_event,
                visible_to=[player_id],
                target=player_id,
                value=role,
            )
        team_event = self._emit(
            wolf_team_event,
            visible_to=self.wolves,
            target=self.wolves,
            value=None,
        )
        self._collect(team_event)
        self._start_night()
        return self.get_observation()

    def step(self, action):
        if self.phase == "end_game":
            raise RuntimeError("cannot step a completed game")
        action_type, action_value = self._validate_action(action)
        reward = [0] * 7
        done = False
        info = {}

        if self.phase == "skill_wolf":
            self._wolf_action(action_type, action_value)
            reward, done, info = self._advance_after_wolf()
        elif self.phase == "skill_seer":
            self._seer_action(action_type, action_value)
            reward, done, info = self._advance_after_seer()
        elif self.phase == "skill_guard":
            self._guard_action(action_type, action_value)
            reward, done, info = self._advance_after_guard()
        elif self.phase == "skill_witch":
            self._witch_action(action_type, action_value)
            reward, done, info = self._end_night()
        elif self.phase in ("speech", "speech_pk"):
            self._speech_action(action_type, action_value)
        elif self.phase in ("vote", "vote_pk"):
            round_complete = self._vote_action(action_type, action_value)
            if round_complete:
                reward, done, info = self._end_vote()
        else:
            raise RuntimeError(f"unknown phase: {self.phase}")

        if done:
            self._finish(info)
        return self.get_observation(), reward, done, info

    def _validate_action(self, action):
        if not isinstance(action, tuple) or len(action) != 2 or not isinstance(action[0], str):
            raise ValueError("action must be a (type, value) tuple")
        if self.phase in ("speech", "speech_pk"):
            if action[0] != self.phase or not isinstance(action[1], str):
                raise ValueError(f"{self.phase} action must contain an utterance string")
            return action
        if action not in self.valid_actions():
            raise ValueError(f"invalid action {action!r}; expected one of {self.valid_actions()!r}")
        return action

    def _start_night(self):
        self.day_or_night = "night"
        self.phase = "skill_wolf"
        self.night_wolf_choices = []
        self.night_kill = None
        self.guard_target = None
        self.witch_heal_target = None
        self.witch_poison_target = None
        living_wolves = [player_id for player_id in self.wolves if player_id in self.alive]
        if not living_wolves:
            raise RuntimeError("night cannot start without a living wolf")
        self.wolf_queue = living_wolves
        self.current_player = self.wolf_queue.pop(0)

    def _wolf_action(self, action_type, target):
        if action_type != "kill":
            raise ValueError("wolf phase requires kill")
        self.night_wolf_choices.append((self.current_player, target or None))
        self._emit(
            private_action_event,
            visible_to=self.wolves,
            speaker=self.current_player,
            target=target or None,
            value="KILL",
            metadata={"status": "pass" if target == 0 else "chosen"},
        )

    def _advance_after_wolf(self):
        if self.wolf_queue:
            self.current_player = self.wolf_queue.pop(0)
            return [0] * 7, False, {}
        targets = [target for _, target in self.night_wolf_choices if target is not None]
        if targets:
            counts = Counter(targets)
            maximum = max(counts.values())
            tied = {target for target, count in counts.items() if count == maximum}
            self.night_kill = next(target for target in reversed(targets) if target in tied)
        if self.seer in self.alive:
            self.phase = "skill_seer"
            self.current_player = self.seer
            return [0] * 7, False, {}
        return self._advance_after_seer()

    def _seer_action(self, action_type, target):
        if action_type != "check":
            raise ValueError("seer phase requires check")
        if target:
            self.seer_checked.add(target)
        event = self._emit(
            check_result_event,
            visible_to=[self.seer],
            speaker=self.seer,
            target=target or None,
            value=(
                "Werewolf" if target and self._role(target) == "Werewolf" else
                "Village" if target else None
            ),
        )
        self._collect(event)

    def _advance_after_seer(self):
        if self.guard is not None and self.guard in self.alive:
            self.phase = "skill_guard"
            self.current_player = self.guard
            return [0] * 7, False, {}
        return self._advance_after_guard()

    def _guard_action(self, action_type, target):
        if action_type != "guard":
            raise ValueError("guard phase requires guard")
        self.guard_target = target or None
        if self.guard_target is not None:
            self.guard_history.append(self.guard_target)
        event = self._emit(
            guard_result_event,
            visible_to=[self.guard],
            speaker=self.guard,
            target=target or None,
            value=None,
            metadata={"status": "pass" if target == 0 else "protected"},
        )
        self._collect(event)

    def _advance_after_guard(self):
        if self.witch is not None and self.witch in self.alive:
            self.phase = "skill_witch"
            self.current_player = self.witch
            availability = {
                (True, True): "HEAL_AND_POISON_AVAILABLE",
                (True, False): "HEAL_AVAILABLE",
                (False, True): "POISON_AVAILABLE",
                (False, False): "NO_POTIONS_AVAILABLE",
            }[(not self.witch_heal_used, not self.witch_poison_used)]
            event = self._emit(
                witch_state_event,
                visible_to=[self.witch],
                target=self.night_kill,
                value=availability,
                metadata={
                    "heal_available": not self.witch_heal_used,
                    "poison_available": not self.witch_poison_used,
                },
            )
            self._collect(event)
            return [0] * 7, False, {}
        return self._end_night()

    def _witch_action(self, action_type, target):
        if action_type == "witch_heal":
            self.witch_heal_used = True
            self.witch_heal_target = target
        elif action_type == "witch_poison":
            self.witch_poison_used = True
            self.witch_poison_target = target
        elif action_type != "witch_pass":
            raise ValueError("invalid witch action")
        self._emit(
            private_action_event,
            visible_to=[self.witch],
            speaker=self.witch,
            target=target or None,
            value=action_type.upper(),
            metadata={"status": "pass" if target == 0 else "used"},
        )

    def _end_night(self):
        dead = set()
        if self.night_kill is not None:
            dead.add(self.night_kill)
        if self.guard_target == self.night_kill:
            dead.discard(self.night_kill)
        if self.witch_heal_target == self.night_kill:
            dead.discard(self.night_kill)
        if self.guard_target is not None and self.guard_target == self.witch_heal_target:
            dead.add(self.guard_target)
        if self.witch_poison_target is not None:
            dead.add(self.witch_poison_target)
        dead &= self.alive
        self.alive -= dead
        event = self._emit(
            death_event,
            target=sorted(dead),
            value=None,
            metadata={"cause": "night"},
        )
        self._collect(event)
        reward, done, info = self._is_done()
        if done:
            return reward, done, info
        self.day += 1
        self.day_or_night = "day"
        self.phase = "speech"
        self.speech_queue = self._rotated(sorted(self.alive))
        self.current_player = self.speech_queue.pop(0)
        return reward, done, info

    def _rotated(self, players):
        if not players:
            return []
        offset = self.rng.randrange(len(players))
        return players[offset:] + players[:offset]

    def _speech_action(self, action_type, utterance):
        expected = self.phase
        if action_type != expected:
            raise ValueError(f"{self.phase} requires {expected}")
        utterance_id = self._next_event_id(prefix="u")
        raw_event = speech_event(
            event_id=utterance_id,
            utterance_id=utterance_id,
            day=self.day,
            phase=self._phase_label(),
            turn=self.event_counter,
            speaker=self.current_player,
            target=self.current_player,
            value=None,
            source_span=utterance,
        )
        self.events.append(raw_event)
        if self.speech_parser is not None and utterance:
            result = self.speech_parser.parse(
                utterance=utterance,
                utterance_id=utterance_id,
                day=self.day,
                phase=self._phase_label(),
                turn=self.event_counter,
                speaker=self.current_player,
            )
            if result.status == "ok":
                self.events.extend(result.events)
            else:
                self.parser_failures.append(
                    {
                        "utterance_id": utterance_id,
                        "raw_text": list(result.raw_text),
                        "error": result.error,
                        "attempts": result.attempts,
                    }
                )
        self._collect(raw_event)
        if self.speech_queue:
            self.current_player = self.speech_queue.pop(0)
            return
        self.phase = "vote" if self.phase == "speech" else "vote_pk"
        if self.phase == "vote":
            self.vote_queue = sorted(self.alive)
        else:
            voters = [player_id for player_id in sorted(self.alive) if player_id not in self.vote_pk_players]
            self.vote_queue = voters or sorted(self.vote_pk_players)
        self.current_votes = {}
        self.current_player = self.vote_queue.pop(0)

    def _vote_action(self, action_type, target):
        if action_type != self.phase:
            raise ValueError(f"{self.phase} requires {self.phase}")
        self.current_votes[self.current_player] = target or None
        event = self._emit(
            vote_event,
            speaker=self.current_player,
            target=target or None,
            value=None,
            metadata={"round": self.phase},
        )
        self._collect(event)
        if self.vote_queue:
            self.current_player = self.vote_queue.pop(0)
            return False
        return True

    def _end_vote(self):
        counts = Counter(target for target in self.current_votes.values() if target is not None)
        leaders = []
        if counts:
            maximum = max(counts.values())
            leaders = sorted(target for target, count in counts.items() if count == maximum)
        result_event = self._emit(
            vote_result_event,
            target=leaders,
            value=None,
            metadata={
                "counts": {str(target): count for target, count in counts.items()},
                "round": self.phase,
            },
        )
        self._collect(result_event)
        if self.phase == "vote" and len(leaders) > 1:
            self.vote_pk_players = leaders
            self.phase = "speech_pk"
            self.speech_queue = self._rotated(leaders)
            self.current_player = self.speech_queue.pop(0)
            return [0] * 7, False, {}

        expelled = leaders[0] if len(leaders) == 1 else None
        if expelled is not None:
            self.alive.remove(expelled)
            exile = self._emit(
                exile_event,
                target=expelled,
                value=None,
            )
            self._collect(exile)
            reveal = self._emit(
                role_reveal_event,
                target=expelled,
                value=self._role(expelled),
            )
            self._collect(reveal)
        reward, done, info = self._is_done()
        if done:
            return reward, done, info
        self._start_night()
        return reward, done, info

    def _is_done(self):
        living_roles = [self._role(player_id) for player_id in self.alive]
        wolves = living_roles.count("Werewolf")
        villagers = living_roles.count("Villager")
        special = len(living_roles) - wolves - villagers
        if wolves == 0:
            return (
                [-self.werewolf_reward if role == "Werewolf" else self.village_reward for role in self.roles],
                True,
                {"Werewolf": -1},
            )
        if villagers == 0 or special == 0:
            self.wolf_win_count += 1
            return (
                [self.werewolf_reward if role == "Werewolf" else -self.village_reward for role in self.roles],
                True,
                {"Werewolf": 1},
            )
        return [0] * 7, False, {}

    def _finish(self, info):
        self.phase = "end_game"
        self.day_or_night = "day"
        self.current_player = None
        self._emit(
            outcome_event,
            value="Werewolf" if info["Werewolf"] == 1 else "Village",
        )
        if self.log_save_path:
            path = Path(self.log_save_path)
            path.mkdir(parents=True, exist_ok=True)
            with (path / "game_events.json").open("w", encoding="utf-8") as output:
                json.dump(self.events, output, ensure_ascii=False, indent=2)
            if self.parser_failures:
                with (path / "parser_failures.json").open("w", encoding="utf-8") as output:
                    json.dump(self.parser_failures, output, ensure_ascii=False, indent=2)

    def valid_actions(self):
        if self.phase == "skill_wolf":
            targets = sorted(self.alive - set(self.wolves))
            return [("kill", 0)] + [("kill", target) for target in targets]
        if self.phase == "skill_seer":
            targets = sorted(self.alive - self.seer_checked - {self.seer})
            return [("check", 0)] + [("check", target) for target in targets]
        if self.phase == "skill_guard":
            targets = sorted(self.alive)
            if self.guard_history:
                targets = [target for target in targets if target != self.guard_history[-1]]
            return [("guard", 0)] + [("guard", target) for target in targets]
        if self.phase == "skill_witch":
            actions = [("witch_pass", 0)]
            if not self.witch_poison_used:
                actions.extend(
                    ("witch_poison", target)
                    for target in sorted(self.alive - {self.witch})
                )
            if not self.witch_heal_used and self.night_kill is not None:
                actions.append(("witch_heal", self.night_kill))
            return actions
        if self.phase in ("speech", "speech_pk"):
            return [(self.phase, "")]
        if self.phase == "vote":
            return [("vote", 0)] + [("vote", target) for target in sorted(self.alive)]
        if self.phase == "vote_pk":
            return [("vote_pk", 0)] + [("vote_pk", target) for target in self.vote_pk_players]
        return []

    def get_observation(self):
        if self.phase == "end_game":
            return {
                "player_id": None,
                "role": None,
                "events": [],
                "phase": self._phase_label(),
                "valid_actions": [],
            }
        return {
            "player_id": self.current_player,
            "role": self._role(self.current_player),
            "events": visible_events(self.events, self.current_player),
            "phase": self._phase_label(),
            "valid_actions": self.valid_actions(),
        }
