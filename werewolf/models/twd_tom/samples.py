from copy import deepcopy

from werewolf.models.twd_tom.labels import make_wolf_labels


def normalize_alive_mask(alive_mask, num_players: int = 7) -> list[float]:
    if alive_mask is None:
        return [1.0] * num_players
    if not isinstance(alive_mask, (list, tuple)):
        raise ValueError("alive_mask must have shape [7]")
    if len(alive_mask) != num_players:
        raise ValueError("alive_mask must have shape [7]")

    normalized = [float(value) for value in alive_mask]
    if any(value not in (0.0, 1.0) for value in normalized):
        raise ValueError("alive_mask values must be 0 or 1")
    return normalized


def _log_field(log, name, default=None):
    if isinstance(log, dict):
        return log.get(name, default)
    return getattr(log, name, default)


def _valid_player_ids(value, num_players):
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [
        player_id
        for player_id in values
        if type(player_id) is int and 1 <= player_id <= num_players
    ]


def _priority_log_targets(content, keys, fallback, num_players):
    for key in keys:
        if key in content:
            return _valid_player_ids(content[key], num_players)
    return _valid_player_ids(fallback, num_players)


def _alive_mask_from_observation(observation, num_players):
    removed_players = set()
    for log in observation.get("game_log", []):
        event = _log_field(log, "event")
        time = _log_field(log, "time")
        target = _log_field(log, "target")
        content = _log_field(log, "content", {})
        content = content if isinstance(content, dict) else {}
        event_name = event.lower() if isinstance(event, str) else ""
        time_name = time.lower() if isinstance(time, str) else ""

        exile_keys = ("expelled", "exiled", "exile", "vote_outcome")
        if (
            event_name in ("end_vote", "exile")
            or time_name == "exile"
            or any(key in content for key in exile_keys)
        ):
            removed_players.update(
                _priority_log_targets(
                    content,
                    exile_keys,
                    target,
                    num_players,
                )
            )

        death_signal_keys = (
            "dead",
            "deaths",
            "dead_players",
            "dead_list",
            "death",
            "killed",
        )
        if (
            event_name in ("end_night", "night_result", "death")
            or time_name == "night_result"
            or any(key in content for key in death_signal_keys)
        ):
            removed_players.update(
                _priority_log_targets(
                    content,
                    death_signal_keys + ("target",),
                    target,
                    num_players,
                )
            )

    return [
        0.0 if player_id in removed_players else 1.0
        for player_id in range(1, num_players + 1)
    ]


def make_twd_tom_sample(
    observation: dict,
    roles,
    game_id=None,
    observer_id=None,
    phase=None,
    num_players: int = 7,
    wolf_role_names=("Werewolf",),
    alive_mask=None,
) -> dict:
    if observer_id is None:
        observer_id = observation.get("current_act_idx")
    if phase is None:
        phase = observation.get("phase")
    if alive_mask is None:
        alive_mask = observation.get("alive_mask")
    if alive_mask is None:
        alive_mask = _alive_mask_from_observation(
            observation,
            num_players,
        )

    wolf_labels = make_wolf_labels(
        roles,
        num_players=num_players,
        wolf_role_names=wolf_role_names,
    ).tolist()

    return {
        "game_id": game_id,
        "observer_id": observer_id,
        "phase": phase,
        "observation": deepcopy(observation),
        "wolf_labels": wolf_labels,
        "alive_mask": normalize_alive_mask(alive_mask, num_players),
    }
