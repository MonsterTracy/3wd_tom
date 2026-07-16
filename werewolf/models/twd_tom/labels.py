import torch


def _role_name(role):
    if isinstance(role, str):
        return role
    if hasattr(role, "name"):
        return role.name
    return str(role)


def wolf_indices_from_roles(
    roles,
    num_players: int = 7,
    wolf_role_names=("Werewolf",),
) -> list[int]:
    if len(roles) != num_players:
        raise ValueError("roles length must equal num_players")

    normalized_wolf_role_names = {
        _role_name(role_name)
        for role_name in wolf_role_names
    }
    return [
        index
        for index, role in enumerate(roles)
        if _role_name(role) in normalized_wolf_role_names
    ]


def make_wolf_labels(
    roles,
    num_players: int = 7,
    wolf_role_names=("Werewolf",),
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    wolf_indices = wolf_indices_from_roles(
        roles,
        num_players=num_players,
        wolf_role_names=wolf_role_names,
    )
    labels = torch.zeros(
        num_players,
        dtype=dtype,
        device=device,
    )
    if wolf_indices:
        labels[wolf_indices] = 1.0
    return labels
