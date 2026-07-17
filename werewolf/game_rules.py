"""Canonical, machine-readable rules for the supported seven-player game."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json


RULESET_ID = "werewolf_7p"
RULESET_VERSION = "werewolf_7p.zh.v1"
NUM_PLAYERS = 7
NUM_WEREWOLVES = 2
PLAYER_IDS = tuple(range(1, NUM_PLAYERS + 1))
CAMPS = {
    "Werewolf": ("Werewolf",),
    "Village": ("Seer", "Witch", "Guard", "Villager"),
}

ROLE_DISTRIBUTIONS = {
    "seer_witch": {
        "Werewolf": 2,
        "Seer": 1,
        "Witch": 1,
        "Guard": 0,
        "Villager": 3,
    },
    "seer_guard": {
        "Werewolf": 2,
        "Seer": 1,
        "Witch": 0,
        "Guard": 1,
        "Villager": 3,
    },
}

ROLE_ABILITIES = {
    "Werewolf": {
        "knowledge": "知道自己和另一名狼人，但不知道其他玩家的具体好人身份。",
        "ability": "夜间每名存活狼人依次提交一次击杀选择。",
        "conduct": "公开发言可以欺骗、隐藏身份或误导，但私有事实必须保持真实；不强制每轮欺骗，也不自动公开队友。",
    },
    "Seer": {
        "knowledge": "只有环境实际生成的 CHECK_RESULT 才是真实查验结果。",
        "ability": "夜间可以执行一次查验，得到 Werewolf 或 Village 结果。",
        "conduct": "不得把未执行的查验当成真实事实；是否公开、隐藏或策略性表达真实查验由自己决定。",
    },
    "Witch": {
        "knowledge": "行动时知道当夜狼人刀口及两瓶药是否仍可用。",
        "ability": "解药和毒药各限整局一次；每夜只执行一个女巫动作，因此不能同夜使用两瓶药。",
        "conduct": "只能使用环境提供的药物状态和刀口；不得声称仍有已消耗的药或编造刀口，是否公开用药信息由自己决定。",
    },
    "Guard": {
        "knowledge": "知道自己的守卫选择，但不会因守卫结果得知目标是否实际受到攻击。",
        "ability": "夜间可以执行一次保护。",
        "conduct": "只能根据环境允许的目标和守卫历史行动；是否公开守卫目标由自己决定。",
    },
    "Villager": {
        "knowledge": "没有角色技能带来的私有查验或夜间结果。",
        "ability": "没有夜间技能，主要依据公共事件和玩家公开声明推断。",
        "conduct": "不得把虚构查验写成已确认私有事实；公开发言仍可作策略性表达。",
    },
}

ROLE_OBJECTIVES = {
    "Werewolf": "避免狼人被放逐，并推进狼人阵营达到胜利条件。",
    "Seer": "运用合法信息和行动帮助好人阵营达到胜利条件。",
    "Witch": "运用合法信息和行动帮助好人阵营达到胜利条件。",
    "Guard": "运用合法信息和行动帮助好人阵营达到胜利条件。",
    "Villager": "运用合法信息和行动帮助好人阵营达到胜利条件。",
}

PHASE_ORDER = {
    "seer_witch": ("skill_wolf", "skill_seer", "skill_witch", "speech", "vote"),
    "seer_guard": ("skill_wolf", "skill_seer", "skill_guard", "speech", "vote"),
}

NIGHT_RESOLUTION_RULES = {
    "wolf_choice": "狼人选择按票数决定刀口；最高票并列时，以狼人行动序中最后选择的并列目标为刀口。",
    "single_protection": "守卫保护或女巫解药单独命中刀口时，取消该刀口造成的死亡。",
    "double_protection": "守卫保护与女巫解药同时命中刀口时，该玩家仍然死亡。",
    "poison": "女巫毒药目标在夜间结算时死亡。",
}

VISIBILITY_RULES = {
    "public_objective": "环境产生的公开 GAME_EVENT 对所有玩家可见；其中原始 SPEECH 不是确定事实。",
    "public_claim": "原始 SPEECH 及其 speech_parser 语义事件都是可能真实、错误或欺骗的公开声明。",
    "private_base": "PRIVATE_FACT 只对 visible_to 中的玩家可见，并且对这些玩家是环境确认的事实。",
    "private_by_role": {
        "Werewolf": "狼人可见自己的 SELF_ROLE、完整 WOLF_TEAM 以及狼队夜间选择。",
        "Seer": "预言家只可见自己的 SELF_ROLE 和自己实际获得的 CHECK_RESULT。",
        "Witch": "女巫只可见自己的 SELF_ROLE、WITCH_STATE 和自己的夜间行动结果。",
        "Guard": "守卫只可见自己的 SELF_ROLE 和自己的 GUARD_RESULT。",
        "Villager": "村民只可见自己的 SELF_ROLE，没有额外角色私有结果。",
    },
}

LEGAL_TARGET_RULES = {
    "skill_wolf": "仅可选择存活的非狼人玩家。",
    "skill_seer": "仅可选择存活、非自己且此前未查验过的玩家。",
    "skill_witch_heal": "解药仅可选择环境告知的当夜刀口；刀口是女巫自己时可以自救。",
    "skill_witch_poison": "毒药仅可选择除女巫自己外的存活玩家。",
    "skill_guard": "仅可选择存活玩家；允许守自己，但不得连续两夜守同一玩家。",
    "vote": "首轮可投任意存活玩家，包括自己。",
    "vote_pk": "PK 轮只能投首轮并列候选人。",
}

PASS_RULES = {
    "skill_wolf": True,
    "skill_seer": True,
    "skill_witch": True,
    "skill_guard": True,
    "speech": True,
    "speech_pk": True,
    "vote": True,
    "vote_pk": True,
}

VOTE_RULES = {
    "speech_order": "白天发言从存活玩家中随机选起点，再按玩家编号循环一次。",
    "vote_order": "首轮投票按存活玩家编号升序进行。",
    "pk_speech_order": "首轮最高票并列者随机选起点并依次进行 PK 发言。",
    "pk_vote_order": "PK 投票优先由非候选存活玩家按编号升序进行；若没有非候选玩家，则由候选人投票。",
}

TIE_RULES = {
    "first_vote": "首轮最高票出现多人并列时，这些玩家进入 PK 发言和 PK 投票。",
    "pk_vote": "PK 投票仍并列或无人获得唯一最高票时，无人被放逐。",
}

DEATH_REVEAL_RULES = {
    "night": "夜间死亡公开死亡名单，但不公开身份。",
    "exile": "被投票放逐的玩家立即公开真实身份。",
}

VICTORY_RULES = {
    "Village": "存活狼人数量变为零时，好人阵营获胜。",
    "Werewolf": "存活普通村民数量变为零，或存活神职数量变为零时，狼人阵营获胜。",
    "checkpoints": "仅在夜间死亡结算后，以及投票最终结算（含 PK 或无人放逐）后检查胜负。",
}


def normalize_variant(variant: str) -> str:
    if variant not in ROLE_DISTRIBUTIONS:
        raise ValueError(f"unsupported seven-player variant: {variant!r}")
    return variant


def variant_from_role_counts(role_counts) -> str:
    """Resolve the unique supported variant from a role-count mapping."""

    if not isinstance(role_counts, dict):
        raise ValueError("role_counts must be a mapping")
    normalized = {role: int(role_counts.get(role, 0)) for role in ROLE_ABILITIES}
    for variant, distribution in ROLE_DISTRIBUTIONS.items():
        if normalized == distribution:
            return variant
    raise ValueError(f"role counts do not match a supported variant: {normalized}")


def validate_role_distribution(roles, variant: str | None = None) -> bool:
    counts = dict(Counter(roles))
    resolved = variant_from_role_counts(counts)
    if variant is not None and resolved != normalize_variant(variant):
        raise ValueError(f"roles match {resolved}, not {variant}")
    if len(roles) != NUM_PLAYERS or counts.get("Werewolf", 0) != NUM_WEREWOLVES:
        raise ValueError("roles do not match the seven-player ruleset")
    return True


def _ruleset_payload() -> dict:
    return {
        "id": RULESET_ID,
        "version": RULESET_VERSION,
        "num_players": NUM_PLAYERS,
        "num_werewolves": NUM_WEREWOLVES,
        "player_ids": PLAYER_IDS,
        "camps": CAMPS,
        "role_distributions": ROLE_DISTRIBUTIONS,
        "role_abilities": ROLE_ABILITIES,
        "role_objectives": ROLE_OBJECTIVES,
        "phase_order": PHASE_ORDER,
        "night_resolution_rules": NIGHT_RESOLUTION_RULES,
        "visibility_rules": VISIBILITY_RULES,
        "legal_target_rules": LEGAL_TARGET_RULES,
        "pass_rules": PASS_RULES,
        "vote_rules": VOTE_RULES,
        "tie_rules": TIE_RULES,
        "death_reveal_rules": DEATH_REVEAL_RULES,
        "victory_rules": VICTORY_RULES,
    }


def ruleset_sha256() -> str:
    canonical = json.dumps(
        _ruleset_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def ruleset_metadata(variant: str) -> dict:
    normalize_variant(variant)
    return {
        "id": RULESET_ID,
        "version": RULESET_VERSION,
        "sha256": ruleset_sha256(),
    }


def canonical_ruleset_metadata() -> dict:
    """Return the one protocol-level reference covering both supported variants."""

    return ruleset_metadata("seer_witch")


def render_global_rules(variant: str) -> str:
    variant = normalize_variant(variant)
    distribution = ROLE_DISTRIBUTIONS[variant]
    roles = "、".join(
        f"{count} 名 {role}" for role, count in distribution.items() if count
    )
    phases = " → ".join(PHASE_ORDER[variant])
    return "\n".join(
        (
            f"规则集：{RULESET_ID} / {RULESET_VERSION}。",
            f"本局固定 {NUM_PLAYERS} 人、{NUM_WEREWOLVES} 名狼人；角色为：{roles}。",
            f"每轮阶段顺序：{phases}；存活角色缺席时跳过对应夜间阶段。",
            "夜间结算：" + "".join(NIGHT_RESOLUTION_RULES.values()),
            f"投票：{TIE_RULES['first_vote']}{TIE_RULES['pk_vote']}",
            f"身份公开：{DEATH_REVEAL_RULES['night']}{DEATH_REVEAL_RULES['exile']}",
            f"好人胜利：{VICTORY_RULES['Village']}",
            f"狼人胜利：{VICTORY_RULES['Werewolf']}",
            f"胜负检查：{VICTORY_RULES['checkpoints']}",
        )
    )


def render_role_rules(role: str, variant: str) -> str:
    variant = normalize_variant(variant)
    if role not in ROLE_ABILITIES or ROLE_DISTRIBUTIONS[variant].get(role, 0) == 0:
        raise ValueError(f"role {role!r} is not present in variant {variant!r}")
    rules = ROLE_ABILITIES[role]
    target_keys = {
        "Werewolf": ("skill_wolf",),
        "Seer": ("skill_seer",),
        "Witch": ("skill_witch_heal", "skill_witch_poison"),
        "Guard": ("skill_guard",),
        "Villager": (),
    }[role]
    targets = "".join(LEGAL_TARGET_RULES[key] for key in target_keys)
    return "\n".join(
        (
            f"角色：{role}。",
            f"阵营目标：{ROLE_OBJECTIVES[role]}",
            f"已知信息：{rules['knowledge']}",
            f"能力：{rules['ability']}",
            *((f"合法目标：{targets}",) if targets else ()),
            f"表达边界：{rules['conduct']}",
        )
    )


def render_phase_rules(role: str, phase: str, variant: str) -> str:
    normalize_variant(variant)
    if role not in ROLE_ABILITIES:
        raise ValueError(f"unsupported role: {role!r}")
    phase_name = phase.split("_")[-1] if phase and phase[0].isdigit() else phase
    if "speech_pk" in phase:
        phase_name = "speech_pk"
    elif "speech" in phase:
        phase_name = "speech"
    elif "vote_pk" in phase:
        phase_name = "vote_pk"
    elif "vote" in phase:
        phase_name = "vote"
    elif "skill_wolf" in phase:
        phase_name = "skill_wolf"
    elif "skill_seer" in phase:
        phase_name = "skill_seer"
    elif "skill_witch" in phase:
        phase_name = "skill_witch"
    elif "skill_guard" in phase:
        phase_name = "skill_guard"

    target_key = phase_name
    if phase_name == "skill_witch":
        target_text = (
            LEGAL_TARGET_RULES["skill_witch_heal"]
            + LEGAL_TARGET_RULES["skill_witch_poison"]
        )
    else:
        target_text = LEGAL_TARGET_RULES.get(target_key, "以环境提供的合法动作列表为准。")
    pass_text = "允许 pass/空操作。" if PASS_RULES.get(phase_name, False) else "不允许 pass。"
    return f"阶段：{phase_name}。合法目标：{target_text}{pass_text}"


def render_visibility_rules(role: str) -> str:
    if role not in ROLE_ABILITIES:
        raise ValueError(f"unsupported role: {role!r}")
    return "\n".join(
        (
            VISIBILITY_RULES["public_objective"],
            VISIBILITY_RULES["public_claim"],
            VISIBILITY_RULES["private_base"],
            VISIBILITY_RULES["private_by_role"][role],
        )
    )


def ruleset_payload() -> dict:
    """Expose a defensive copy for consistency tests and reproducibility tools."""

    return deepcopy(_ruleset_payload())
