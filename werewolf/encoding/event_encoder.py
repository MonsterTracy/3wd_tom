from werewolf.encoding.dialogue_actions import (
    CAMP2ID,
    CERTAINTY2ID,
    EVENT_TYPE2ID,
    PHASE2ID,
    POLARITY2ID,
    PREDICATE2ID,
    ROLE2ID,
    normalize_claim,
    safe_id,
)


EVENT_TOKEN_FIELDS = (
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
)


def get_log_field(log, name: str, default=None):
    if isinstance(log, dict):
        return log.get(name, default)
    return getattr(log, name, default)


def validate_event_token(token: dict) -> bool:
    return isinstance(token, dict) and all(
        field in token for field in EVENT_TOKEN_FIELDS
    )


def _as_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _phase_id(phase: str) -> int:
    phase_key = "vote_pk" if phase == "pk_vote" else phase
    return safe_id(PHASE2ID, phase_key, "none")


def _event_token(
    *,
    event_type_id: int,
    speaker_id: int,
    target_id: int,
    predicate_id: int,
    role_id: int,
    camp_id: int,
    polarity_id: int,
    certainty_id: int,
    phase_id: int,
    day_id: int,
) -> dict:
    return {
        "event_type_id": event_type_id,
        "speaker_id": speaker_id,
        "target_id": target_id,
        "predicate_id": predicate_id,
        "role_id": role_id,
        "camp_id": camp_id,
        "polarity_id": polarity_id,
        "certainty_id": certainty_id,
        "phase_id": phase_id,
        "day_id": day_id,
    }


def encode_dialogue_action(claim: dict, day: int, phase: str) -> dict:
    normalized = normalize_claim(claim)
    return _event_token(
        event_type_id=EVENT_TYPE2ID["dialogue_action"],
        speaker_id=_as_int(normalized["speaker"]),
        target_id=_as_int(normalized["target"]),
        predicate_id=safe_id(
            PREDICATE2ID,
            normalized["predicate"],
            "none",
        ),
        role_id=safe_id(ROLE2ID, normalized["role"], "None"),
        camp_id=safe_id(CAMP2ID, normalized["camp"], "None"),
        polarity_id=safe_id(
            POLARITY2ID,
            normalized["polarity"],
            "None",
        ),
        certainty_id=safe_id(
            CERTAINTY2ID,
            normalized["certainty"],
            "implicit",
        ),
        phase_id=_phase_id(phase),
        day_id=_as_int(day),
    )


def encode_vote(
    source: int,
    target: int | None,
    day: int,
    phase: str = "vote",
) -> dict:
    event_type = "pk_vote" if phase in ("vote_pk", "pk_vote") else "vote"
    return _event_token(
        event_type_id=EVENT_TYPE2ID[event_type],
        speaker_id=_as_int(source),
        target_id=_as_int(target),
        predicate_id=PREDICATE2ID["vote"],
        role_id=ROLE2ID["None"],
        camp_id=CAMP2ID["None"],
        polarity_id=POLARITY2ID["negative"],
        certainty_id=CERTAINTY2ID["explicit"],
        phase_id=_phase_id(phase),
        day_id=_as_int(day),
    )


def encode_death(
    target: int,
    day: int,
    phase: str = "night_result",
) -> dict:
    return _event_token(
        event_type_id=EVENT_TYPE2ID["death"],
        speaker_id=0,
        target_id=_as_int(target),
        predicate_id=PREDICATE2ID["death"],
        role_id=ROLE2ID["None"],
        camp_id=CAMP2ID["None"],
        polarity_id=POLARITY2ID["neutral"],
        certainty_id=CERTAINTY2ID["explicit"],
        phase_id=_phase_id(phase),
        day_id=_as_int(day),
    )


def encode_exile(
    target: int,
    day: int,
    phase: str = "exile",
) -> dict:
    return _event_token(
        event_type_id=EVENT_TYPE2ID["exile"],
        speaker_id=0,
        target_id=_as_int(target),
        predicate_id=PREDICATE2ID["exile"],
        role_id=ROLE2ID["None"],
        camp_id=CAMP2ID["None"],
        polarity_id=POLARITY2ID["negative"],
        certainty_id=CERTAINTY2ID["explicit"],
        phase_id=safe_id(PHASE2ID, phase, "exile"),
        day_id=_as_int(day),
    )


def encode_private_role_info(
    observer_id: int,
    role: str,
    day: int = 0,
    phase: str = "night",
) -> dict:
    return _event_token(
        event_type_id=EVENT_TYPE2ID["private_role_info"],
        speaker_id=_as_int(observer_id),
        target_id=_as_int(observer_id),
        predicate_id=PREDICATE2ID["claim_role"],
        role_id=safe_id(ROLE2ID, role, "Unknown"),
        camp_id=CAMP2ID["None"],
        polarity_id=POLARITY2ID["neutral"],
        certainty_id=CERTAINTY2ID["explicit"],
        phase_id=_phase_id(phase),
        day_id=_as_int(day),
    )


def encode_private_check_result(
    observer_id: int,
    target: int,
    role: str | None = None,
    camp: str | None = None,
    day: int = 0,
) -> dict:
    return _event_token(
        event_type_id=EVENT_TYPE2ID["private_check_result"],
        speaker_id=_as_int(observer_id),
        target_id=_as_int(target),
        predicate_id=PREDICATE2ID["report_check_result"],
        role_id=safe_id(ROLE2ID, role or "Unknown", "Unknown"),
        camp_id=safe_id(CAMP2ID, camp or "Unknown", "Unknown"),
        polarity_id=POLARITY2ID["neutral"],
        certainty_id=CERTAINTY2ID["explicit"],
        phase_id=PHASE2ID["night"],
        day_id=_as_int(day),
    )


def encode_private_wolf_team(
    observer_id: int,
    teammate: int,
    day: int = 0,
) -> dict:
    return _event_token(
        event_type_id=EVENT_TYPE2ID["private_wolf_team"],
        speaker_id=_as_int(observer_id),
        target_id=_as_int(teammate),
        predicate_id=PREDICATE2ID["claim_camp"],
        role_id=ROLE2ID["Werewolf"],
        camp_id=CAMP2ID["Werewolf"],
        polarity_id=POLARITY2ID["neutral"],
        certainty_id=CERTAINTY2ID["explicit"],
        phase_id=PHASE2ID["night"],
        day_id=_as_int(day),
    )


def _phase_from_log(time, event) -> str:
    for value in (time, event):
        if not isinstance(value, str) or not value:
            continue
        if value in PHASE2ID:
            return value

        lowered = value.lower()
        if "vote_pk" in lowered or "pk_vote" in lowered:
            return "vote_pk"
        if "speech_pk" in lowered:
            return "speech_pk"
        if (
            "night_result" in lowered
            or "end_night" in lowered
            or "death" in lowered
            or "dead" in lowered
        ):
            return "night_result"
        if "speech" in lowered:
            return "speech"
        if "vote" in lowered:
            return "vote"
        if "night" in lowered:
            return "night"
    return "none"


def _contains_marker(values, markers) -> bool:
    for value in values:
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        if any(marker in lowered for marker in markers):
            return True
    return False


def _event_name(value) -> str:
    return value.lower() if isinstance(value, str) else ""


def _valid_targets(value) -> list[int]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return list(
        dict.fromkeys(
            item for item in values if type(item) is int and item > 0
        )
    )


def _priority_targets(
    content: dict,
    keys: tuple[str, ...],
    log_target,
) -> list[int]:
    for key in keys:
        if key in content:
            return _valid_targets(content[key])
    return _valid_targets(log_target)


def _first_valid_target(*values) -> int:
    for value in values:
        targets = _valid_targets(value)
        if targets:
            return targets[0]
    return 0


def _observer_id_from_log(observation: dict, log) -> int:
    observer_id = _first_valid_target(
        observation.get("observer_id"),
        observation.get("observer"),
        observation.get("current_act_idx"),
        get_log_field(log, "source", 0),
    )
    if observer_id:
        return observer_id

    viewers = _valid_targets(get_log_field(log, "viewer", []))
    return viewers[0] if len(viewers) == 1 else 0


def _private_check_values(content: dict) -> tuple[str | None, str | None]:
    role = content.get("role")
    camp = content.get("checked_camp", content.get("camp"))
    check_result = content.get("check_result")

    if isinstance(check_result, dict):
        role = role or check_result.get("role") or check_result.get("identity")
        camp = (
            camp
            or check_result.get("checked_camp")
            or check_result.get("camp")
        )
    elif isinstance(check_result, str):
        if role is None and check_result in ROLE2ID:
            role = check_result
        if camp is None and check_result in CAMP2ID:
            camp = check_result

    return role, camp


def encode_observation_game_log(observation: dict) -> list[dict]:
    if not isinstance(observation, dict):
        return []

    tokens = []
    game_log = observation.get("game_log", [])
    if not isinstance(game_log, (list, tuple)):
        return tokens

    for log in game_log:
        day = get_log_field(log, "day", 0)
        time = get_log_field(log, "time")
        event = get_log_field(log, "event")
        source = get_log_field(log, "source", 0)
        target = get_log_field(log, "target", 0)
        content = get_log_field(log, "content", {})
        phase = _phase_from_log(time, event)
        event_name = _event_name(event)
        time_name = _event_name(time)
        event_content = content if isinstance(content, dict) else {}
        parsed_claims = event_content.get("parsed_claims")

        if isinstance(parsed_claims, list):
            for claim in parsed_claims:
                if isinstance(claim, dict):
                    tokens.append(encode_dialogue_action(claim, day, phase))

        identity = event_content.get("identity")
        is_self_identity = (
            event_name == "self_identity" or "identity" in event_content
        )
        if is_self_identity and isinstance(identity, str) and identity:
            identity_target = _first_valid_target(
                target,
                event_content.get("player"),
                event_content.get("target"),
            )
            if not identity_target:
                viewers = _valid_targets(get_log_field(log, "viewer", []))
                if len(viewers) == 1:
                    identity_target = viewers[0]
            if identity_target:
                identity_phase = phase if phase != "none" else "night"
                tokens.append(
                    encode_private_role_info(
                        identity_target,
                        identity,
                        day,
                        identity_phase,
                    )
                )

        is_wolf_team_info = (
            event_name == "werewolf_team_info"
            or "wolf_team" in event_content
        )
        if is_wolf_team_info:
            wolf_team = event_content.get("wolf_team") or target
            observer_id = _observer_id_from_log(observation, log)
            for wolf_id in _valid_targets(wolf_team):
                tokens.append(
                    encode_private_wolf_team(observer_id, wolf_id, day)
                )

        private_check_keys = ("check_result", "checked_camp", "camp")
        is_private_check = event_name in (
            "private_check_result",
            "seer_check_result",
        ) or (
            not isinstance(parsed_claims, list)
            and any(key in event_content for key in private_check_keys)
        )
        if is_private_check:
            check_target = _first_valid_target(
                target,
                event_content.get("target"),
                event_content.get("checked_player"),
                event_content.get("player"),
            )
            if check_target:
                role, camp = _private_check_values(event_content)
                tokens.append(
                    encode_private_check_result(
                        observer_id=_observer_id_from_log(observation, log),
                        target=check_target,
                        role=role,
                        camp=camp,
                        day=day,
                    )
                )

        signals = (event, time, phase)
        vote_names = ("vote", "vote_pk", "pk_vote")
        is_vote = event_name in vote_names or (
            not event_name and time_name in vote_names
        )
        if is_vote:
            vote_phase = (
                "vote_pk"
                if _contains_marker(signals, ("vote_pk", "pk_vote"))
                else phase
            )
            tokens.append(encode_vote(source, target, day, vote_phase))

        exile_keys = ("expelled", "exiled", "exile", "vote_outcome")
        is_exile = (
            event_name in ("end_vote", "exile")
            or time_name == "exile"
            or any(key in event_content for key in exile_keys)
        )
        if is_exile:
            for exiled_player in _priority_targets(
                event_content,
                exile_keys,
                target,
            ):
                tokens.append(encode_exile(exiled_player, day))

        death_signal_keys = (
            "dead",
            "deaths",
            "dead_players",
            "death",
            "killed",
        )
        death_target_keys = death_signal_keys + ("target",)
        is_death = (
            event_name in ("night_result", "death", "end_night")
            or time_name == "night_result"
            or any(key in event_content for key in death_signal_keys)
        )
        if is_death:
            for dead_player in _priority_targets(
                event_content,
                death_target_keys,
                target,
            ):
                tokens.append(encode_death(dead_player, day, phase))

    return [token for token in tokens if validate_event_token(token)]


__all__ = [
    "EVENT_TOKEN_FIELDS",
    "get_log_field",
    "validate_event_token",
    "encode_dialogue_action",
    "encode_vote",
    "encode_death",
    "encode_exile",
    "encode_private_role_info",
    "encode_private_check_result",
    "encode_private_wolf_team",
    "encode_observation_game_log",
]
