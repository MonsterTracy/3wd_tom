import torch
from torch.nn import functional as F


_VALID_REDUCTIONS = {"mean", "sum", "none"}
_VALID_SUPERVISION_MODES = {"last", "all"}


def _expand_wolf_labels(wolf_labels, target_shape):
    batch_size, seq_len, num_players = target_shape
    if wolf_labels.ndim == 2:
        if wolf_labels.shape != (batch_size, num_players):
            raise ValueError("wolf_labels shape does not match model output")
        return wolf_labels.unsqueeze(1).expand(
            batch_size, seq_len, num_players
        )
    if wolf_labels.ndim == 3 and wolf_labels.shape == target_shape:
        return wolf_labels
    raise ValueError("wolf_labels must have shape [B, 7] or [B, T, 7]")


def _build_mask(reference, attention_mask=None, player_mask=None):
    batch_size, seq_len, num_players = reference.shape
    mask = torch.ones_like(reference)

    if attention_mask is not None:
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [B, T]")
        mask = mask * attention_mask.to(
            device=reference.device,
            dtype=reference.dtype,
        ).unsqueeze(-1)

    if player_mask is not None:
        if player_mask.ndim == 2:
            if player_mask.shape != (batch_size, num_players):
                raise ValueError("player_mask must have shape [B, 7]")
            player_mask = player_mask.unsqueeze(1)
        elif (
            player_mask.ndim != 3
            or player_mask.shape != reference.shape
        ):
            raise ValueError(
                "player_mask must have shape [B, 7] or [B, T, 7]"
            )
        mask = mask * player_mask.to(
            device=reference.device,
            dtype=reference.dtype,
        )

    return mask


def _masked_reduce(elementwise_loss, mask, reduction):
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError("reduction must be one of: mean, sum, none")

    masked_loss = elementwise_loss * mask
    if reduction == "none":
        return masked_loss
    if reduction == "sum":
        return masked_loss.sum()
    return masked_loss.sum() / mask.sum().clamp_min(1)


def _last_valid_time_indices(
    batch_size,
    seq_len,
    device,
    attention_mask=None,
):
    if attention_mask is None:
        return (
            torch.full(
                (batch_size,),
                seq_len - 1,
                dtype=torch.long,
                device=device,
            ),
            torch.ones(batch_size, dtype=torch.bool, device=device),
        )

    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError("attention_mask must have shape [B, T]")
    valid_counts = attention_mask.to(device=device).sum(dim=1)
    last_valid_t = (valid_counts.to(torch.long) - 1).clamp(
        min=0,
        max=seq_len - 1,
    )
    return last_valid_t, valid_counts.gt(0)


def _last_token_supervision(
    wolf_logits,
    wolf_labels,
    attention_mask=None,
    player_mask=None,
):
    if wolf_logits.ndim != 3:
        raise ValueError("wolf_logits must have shape [B, T, 7]")

    batch_size, seq_len, num_players = wolf_logits.shape
    if seq_len < 1:
        raise ValueError("wolf_logits must contain at least one time step")

    last_valid_t, valid_samples = _last_valid_time_indices(
        batch_size,
        seq_len,
        wolf_logits.device,
        attention_mask=attention_mask,
    )

    batch_indices = torch.arange(
        batch_size,
        device=wolf_logits.device,
    )
    logits_last = wolf_logits[batch_indices, last_valid_t]

    if wolf_labels.ndim == 2:
        if wolf_labels.shape != (batch_size, num_players):
            raise ValueError("wolf_labels shape does not match model output")
        labels_last = wolf_labels
    elif wolf_labels.ndim == 3 and wolf_labels.shape == wolf_logits.shape:
        labels_last = wolf_labels.to(device=wolf_logits.device)[
            batch_indices,
            last_valid_t,
        ]
    else:
        raise ValueError(
            "wolf_labels must have shape [B, 7] or [B, T, 7]"
        )
    labels_last = labels_last.to(
        device=wolf_logits.device,
        dtype=wolf_logits.dtype,
    )

    mask = valid_samples.to(dtype=wolf_logits.dtype).unsqueeze(-1)
    mask = mask.expand_as(logits_last)
    if player_mask is not None:
        if player_mask.ndim == 2:
            if player_mask.shape != (batch_size, num_players):
                raise ValueError("player_mask must have shape [B, 7]")
            player_mask_last = player_mask
        elif player_mask.ndim == 3 and player_mask.shape == wolf_logits.shape:
            player_mask_last = player_mask.to(device=wolf_logits.device)[
                batch_indices,
                last_valid_t,
            ]
        else:
            raise ValueError(
                "player_mask must have shape [B, 7] or [B, T, 7]"
            )
        mask = mask * player_mask_last.to(
            device=wolf_logits.device,
            dtype=wolf_logits.dtype,
        )

    return logits_last, labels_last, mask


def masked_bce_with_logits_loss(
    wolf_logits,
    wolf_labels,
    attention_mask=None,
    player_mask=None,
    reduction="mean",
    supervision_mode="last",
):
    if supervision_mode not in _VALID_SUPERVISION_MODES:
        raise ValueError("supervision_mode must be one of: last, all")

    if supervision_mode == "last":
        logits, labels, mask = _last_token_supervision(
            wolf_logits,
            wolf_labels,
            attention_mask=attention_mask,
            player_mask=player_mask,
        )
        elementwise_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        return _masked_reduce(elementwise_loss, mask, reduction)

    labels = _expand_wolf_labels(
        wolf_labels, tuple(wolf_logits.shape)
    ).to(device=wolf_logits.device, dtype=wolf_logits.dtype)
    elementwise_loss = F.binary_cross_entropy_with_logits(
        wolf_logits,
        labels,
        reduction="none",
    )
    mask = _build_mask(
        elementwise_loss,
        attention_mask=attention_mask,
        player_mask=player_mask,
    )
    return _masked_reduce(elementwise_loss, mask, reduction)


def cardinality_loss(
    wolf_prob,
    attention_mask=None,
    num_wolves: float = 2.0,
    supervision_mode: str = "last",
    reduction: str = "mean",
):
    if supervision_mode not in _VALID_SUPERVISION_MODES:
        raise ValueError("supervision_mode must be one of: last, all")
    if wolf_prob.ndim != 3:
        raise ValueError("wolf_prob must have shape [B, T, 7]")

    batch_size, seq_len, _ = wolf_prob.shape
    if seq_len < 1:
        raise ValueError("wolf_prob must contain at least one time step")

    if supervision_mode == "last":
        last_valid_t, valid_samples = _last_valid_time_indices(
            batch_size,
            seq_len,
            wolf_prob.device,
            attention_mask=attention_mask,
        )
        batch_indices = torch.arange(
            batch_size,
            device=wolf_prob.device,
        )
        pred_count = wolf_prob[batch_indices, last_valid_t].sum(dim=-1)
        mask = valid_samples.to(dtype=wolf_prob.dtype)
    else:
        pred_count = wolf_prob.sum(dim=-1)
        mask = torch.ones_like(pred_count)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must have shape [B, T]")
            mask = attention_mask.to(
                device=wolf_prob.device,
                dtype=wolf_prob.dtype,
            )

    elementwise_loss = (pred_count - num_wolves).square()
    return _masked_reduce(elementwise_loss, mask, reduction)


def twd_region_consistency_loss(
    region_probs,
    wolf_labels,
    attention_mask=None,
    player_mask=None,
    pos_threshold=0.75,
    neg_threshold=0.25,
    reduction="mean",
):
    if not (0 <= neg_threshold < pos_threshold <= 1):
        raise ValueError(
            "thresholds must satisfy "
            "0 <= neg_threshold < pos_threshold <= 1"
        )

    target_shape = tuple(region_probs.shape[:-1])
    labels = _expand_wolf_labels(
        wolf_labels, target_shape
    ).to(device=region_probs.device, dtype=region_probs.dtype)
    target_region = torch.full(
        target_shape,
        1,
        dtype=torch.long,
        device=region_probs.device,
    )
    target_region = torch.where(
        labels >= pos_threshold,
        torch.zeros_like(target_region),
        target_region,
    )
    target_region = torch.where(
        labels <= neg_threshold,
        torch.full_like(target_region, 2),
        target_region,
    )

    safe_probs = region_probs.clamp(min=1e-8, max=1.0)
    target_probs = safe_probs.gather(
        dim=-1,
        index=target_region.unsqueeze(-1),
    ).squeeze(-1)
    elementwise_loss = -target_probs.log()
    mask = _build_mask(
        elementwise_loss,
        attention_mask=attention_mask,
        player_mask=player_mask,
    )
    return _masked_reduce(elementwise_loss, mask, reduction)


def twd_tom_loss(
    outputs,
    wolf_labels,
    attention_mask=None,
    player_mask=None,
    bce_weight=1.0,
    region_weight=0.0,
    reduction="mean",
    supervision_mode="last",
    cardinality_weight: float = 0.0,
    num_wolves: float = 2.0,
):
    bce_loss = masked_bce_with_logits_loss(
        outputs["wolf_logits"],
        wolf_labels,
        attention_mask=attention_mask,
        player_mask=player_mask,
        reduction=reduction,
        supervision_mode=supervision_mode,
    )
    region_loss = twd_region_consistency_loss(
        outputs["region_probs"],
        wolf_labels,
        attention_mask=attention_mask,
        player_mask=player_mask,
        reduction=reduction,
    )
    if supervision_mode == "last" and reduction == "none":
        batch_size, seq_len, _ = outputs["wolf_logits"].shape
        last_valid_t, _ = _last_valid_time_indices(
            batch_size,
            seq_len,
            outputs["wolf_logits"].device,
            attention_mask=attention_mask,
        )
        batch_indices = torch.arange(
            batch_size,
            device=outputs["wolf_logits"].device,
        )
        region_loss = region_loss[batch_indices, last_valid_t]
    loss = bce_weight * bce_loss + region_weight * region_loss
    card_loss = outputs["wolf_logits"].new_zeros(())
    if cardinality_weight > 0:
        card_loss = cardinality_loss(
            outputs["wolf_prob"],
            attention_mask=attention_mask,
            num_wolves=num_wolves,
            supervision_mode=supervision_mode,
            reduction=reduction,
        )
        card_term = (
            card_loss.unsqueeze(-1)
            if reduction == "none"
            else card_loss
        )
        loss = loss + cardinality_weight * card_term
    return {
        "loss": loss,
        "bce_loss": bce_loss,
        "region_loss": region_loss,
        "cardinality_loss": card_loss,
        "cardinality_weight": cardinality_weight,
    }
