"""Checkpoint-driven collection for first- and second-order ToM labels."""

import json
from dataclasses import dataclass
from pathlib import Path

from werewolf.events.streams import (
    knowledge_for_player,
    public_events,
    render_stream,
    visible_events,
)
from werewolf.tom.masks import first_order_knowledge_mask, second_order_output_mask
from werewolf.tom.schemas import FIRST_ORDER_MODE, make_sample


PUBLIC_CHECKPOINTS = {
    "SPEECH": "after_speech",
    "VOTE_CAST": "after_public_vote",
    "VOTE_RESULT": "after_vote_result",
    "EXILE": "after_exile",
    "DEATH": "after_death",
    "ROLE_REVEAL": "after_reveal",
}
PRIVATE_FIRST_ORDER_CHECKPOINTS = {
    "CHECK_RESULT": "after_seer_check",
    "WOLF_TEAM": "after_wolf_team",
    "WITCH_STATE": "after_witch_info",
    "GUARD_RESULT": "after_guard_result",
}


@dataclass(frozen=True)
class CollectionBatch:
    samples: tuple[dict, ...]
    failures: tuple[dict, ...]


class JsonlSink:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record):
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def checkpoint_for_event(event):
    kind = event["content"]["kind"]
    if event["visibility"] == "public":
        checkpoint = PUBLIC_CHECKPOINTS.get(kind)
        if checkpoint is not None:
            return checkpoint
        if event["event_family"] == "GAME_EVENT" and kind not in {"SETTING", "OUTCOME"}:
            return f"after_{kind.lower()}"
        return None
    return PRIVATE_FIRST_ORDER_CHECKPOINTS.get(kind)


class ToMCollector:
    """Collect labels at state-changing checkpoints without affecting gameplay."""

    def __init__(
        self,
        *,
        game_id,
        roles,
        guess_provider_for,
        sample_sink=None,
        failure_sink=None,
    ):
        if len(roles) != 7:
            raise ValueError("roles must contain exactly seven entries")
        self.game_id = str(game_id)
        self.roles = tuple(roles)
        self.guess_provider_for = guess_provider_for
        self.sample_sink = sample_sink
        self.failure_sink = failure_sink

    def collect(self, *, trigger_event, events, alive_players):
        checkpoint = checkpoint_for_event(trigger_event)
        if checkpoint is None:
            return CollectionBatch(samples=(), failures=())
        if trigger_event["visibility"] == "public":
            return self._collect_public(
                trigger_event=trigger_event,
                checkpoint=checkpoint,
                events=events,
                alive_players=alive_players,
            )
        return self._collect_private(
            trigger_event=trigger_event,
            checkpoint=checkpoint,
            events=events,
            alive_players=alive_players,
        )

    def _target_guess(self, target_id, events):
        knowledge = knowledge_for_player(events, target_id)
        role = knowledge["role"]
        if role is None:
            raise ValueError(f"player {target_id} has no SELF_ROLE private fact")
        mask = first_order_knowledge_mask(
            observer_id=target_id,
            observer_role=role,
            known_wolves=knowledge["known_wolves"],
            known_good=knowledge["known_good"],
        )
        view = render_stream(visible_events(events, target_id))
        return self.guess_provider_for(target_id).elicit(
            player_view=view,
            output_mask=mask,
        ), mask

    def _base(self, trigger_event, checkpoint):
        state_id = f"{self.game_id}:{trigger_event['event_id']}"
        return {
            "game_id": self.game_id,
            "checkpoint": checkpoint,
            "state_id": state_id,
            "day": trigger_event["day"],
            "phase": trigger_event["phase"],
            "turn": trigger_event["turn"],
        }

    def _sample(self, *, suffix, trigger_event, checkpoint, guess, **kwargs):
        base = self._base(trigger_event, checkpoint)
        sample = make_sample(
            sample_id=f"{base['state_id']}:{suffix}",
            guess=guess,
            **base,
            **kwargs,
        )
        sink = self.sample_sink if guess.status == "ok" else self.failure_sink
        if sink is not None:
            sink.append(sample)
        return sample

    def _collect_public(self, *, trigger_event, checkpoint, events, alive_players):
        alive = set(alive_players)
        nonwolf_targets = [
            player_id
            for player_id in range(1, 8)
            if player_id in alive and self.roles[player_id - 1] != "Werewolf"
        ]
        wolf_modelers = [
            player_id
            for player_id, role in enumerate(self.roles, start=1)
            if role == "Werewolf"
        ]
        samples = []
        failures = []
        for target_id in nonwolf_targets:
            guess, first_mask = self._target_guess(target_id, events)
            target_records = [
                self._sample(
                    suffix=f"first:{target_id}",
                    trigger_event=trigger_event,
                    checkpoint=checkpoint,
                    guess=guess,
                    task="first_order",
                    mode=FIRST_ORDER_MODE,
                    observer_id=target_id,
                    modeler_id=None,
                    target_id=None,
                    events=visible_events(events, target_id),
                    output_mask=first_mask,
                ),
                self._sample(
                    suffix=f"second:public:{target_id}",
                    trigger_event=trigger_event,
                    checkpoint=checkpoint,
                    guess=guess,
                    task="second_order",
                    mode="public_only",
                    observer_id=None,
                    modeler_id=None,
                    target_id=target_id,
                    events=public_events(events),
                    output_mask=second_order_output_mask(
                        mode="public_only", target_id=target_id
                    ),
                ),
            ]
            for modeler_id in wolf_modelers:
                target_records.append(
                    self._sample(
                        suffix=f"second:wolf:{modeler_id}:{target_id}",
                        trigger_event=trigger_event,
                        checkpoint=checkpoint,
                        guess=guess,
                        task="second_order",
                        mode="wolf_conditioned",
                        observer_id=None,
                        modeler_id=modeler_id,
                        target_id=target_id,
                        events=visible_events(events, modeler_id),
                        output_mask=second_order_output_mask(
                            mode="wolf_conditioned", target_id=target_id
                        ),
                    )
                )
            destination = samples if guess.status == "ok" else failures
            destination.extend(target_records)
        return CollectionBatch(samples=tuple(samples), failures=tuple(failures))

    def _collect_private(self, *, trigger_event, checkpoint, events, alive_players):
        viewers = [
            player_id
            for player_id in trigger_event["visible_to"]
            if player_id in set(alive_players)
            and self.roles[player_id - 1] != "Werewolf"
        ]
        samples = []
        failures = []
        for observer_id in viewers:
            guess, mask = self._target_guess(observer_id, events)
            sample = self._sample(
                suffix=f"first:{observer_id}",
                trigger_event=trigger_event,
                checkpoint=checkpoint,
                guess=guess,
                task="first_order",
                mode=FIRST_ORDER_MODE,
                observer_id=observer_id,
                modeler_id=None,
                target_id=None,
                events=visible_events(events, observer_id),
                output_mask=mask,
            )
            (samples if guess.status == "ok" else failures).append(sample)
        return CollectionBatch(samples=tuple(samples), failures=tuple(failures))
