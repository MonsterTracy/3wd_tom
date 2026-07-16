import torch


_VALID_SUPERVISION_MODES = {"last", "all"}


def _select_metric_rows(
    values,
    wolf_labels,
    attention_mask=None,
    supervision_mode="last",
):
    if supervision_mode not in _VALID_SUPERVISION_MODES:
        raise ValueError("supervision_mode must be one of: last, all")
    if values.ndim not in (2, 3):
        raise ValueError("predictions must have shape [B, 7] or [B, T, 7]")
    if wolf_labels.ndim not in (2, 3):
        raise ValueError("wolf_labels must have shape [B, 7] or [B, T, 7]")
    if values.shape[0] != wolf_labels.shape[0]:
        raise ValueError("batch dimensions must match")
    if values.shape[-1] != wolf_labels.shape[-1]:
        raise ValueError("player dimensions must match")

    labels = wolf_labels.to(device=values.device)
    temporal_shapes = [
        tensor.shape[1]
        for tensor in (values, labels)
        if tensor.ndim == 3
    ]
    if not temporal_shapes:
        return values, labels

    seq_len = temporal_shapes[0]
    if any(length != seq_len for length in temporal_shapes):
        raise ValueError("time dimensions must match")
    batch_size = values.shape[0]

    if attention_mask is not None:
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape [B, T]")
        attention_mask = attention_mask.to(device=values.device)

    if supervision_mode == "last":
        if attention_mask is None:
            last_valid_t = torch.full(
                (batch_size,),
                seq_len - 1,
                dtype=torch.long,
                device=values.device,
            )
            valid_samples = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=values.device,
            )
        else:
            valid_counts = attention_mask.sum(dim=1)
            last_valid_t = (valid_counts.to(torch.long) - 1).clamp(
                min=0,
                max=seq_len - 1,
            )
            valid_samples = valid_counts.gt(0)

        batch_indices = torch.arange(batch_size, device=values.device)
        selected_values = (
            values[batch_indices, last_valid_t]
            if values.ndim == 3
            else values
        )
        selected_labels = (
            labels[batch_indices, last_valid_t]
            if labels.ndim == 3
            else labels
        )
        return (
            selected_values[valid_samples],
            selected_labels[valid_samples],
        )

    expanded_values = (
        values
        if values.ndim == 3
        else values.unsqueeze(1).expand(-1, seq_len, -1)
    )
    expanded_labels = (
        labels
        if labels.ndim == 3
        else labels.unsqueeze(1).expand(-1, seq_len, -1)
    )
    valid_tokens = (
        torch.ones(
            batch_size,
            seq_len,
            dtype=torch.bool,
            device=values.device,
        )
        if attention_mask is None
        else attention_mask.bool()
    )
    return expanded_values[valid_tokens], expanded_labels[valid_tokens]


def _safe_ratio(numerator, denominator) -> float:
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return 0.0
    return float(numerator) / denominator_value


def _safe_mean(values) -> float:
    return float(values.mean()) if values.numel() else 0.0


def compute_wolf_probability_metrics(
    wolf_prob,
    wolf_labels,
    attention_mask=None,
    supervision_mode="last",
    num_wolves=2,
    threshold=0.5,
):
    probabilities, labels = _select_metric_rows(
        wolf_prob.detach(),
        wolf_labels.detach(),
        attention_mask=attention_mask,
        supervision_mode=supervision_mode,
    )
    if probabilities.shape[0] == 0:
        return {
            "count_error": 0.0,
            "top2_exact": 0.0,
            "top2_recall": 0.0,
            "top2_f1": 0.0,
            "binary_accuracy": 0.0,
            "true_wolf_mean_prob": 0.0,
            "true_good_mean_prob": 0.0,
        }

    true_wolf = labels >= 0.5
    predicted_wolf = probabilities >= threshold
    top2_indices = probabilities.topk(k=2, dim=-1).indices
    top2_mask = torch.zeros_like(true_wolf)
    top2_mask.scatter_(dim=-1, index=top2_indices, value=True)

    true_wolf_count = true_wolf.sum(dim=-1)
    top2_hits = (top2_mask & true_wolf).sum(dim=-1)
    top2_recall = torch.where(
        true_wolf_count > 0,
        top2_hits.float() / true_wolf_count.clamp_min(1).float(),
        torch.zeros_like(top2_hits, dtype=torch.float32),
    )

    return {
        "count_error": _safe_mean(
            (probabilities.sum(dim=-1) - num_wolves).abs()
        ),
        "top2_exact": _safe_mean(
            top2_mask.eq(true_wolf).all(dim=-1).float()
        ),
        "top2_recall": _safe_mean(top2_recall),
        # In the fixed 7-player / 2-wolf setting evaluated with a fixed
        # top-2 prediction set, top2 precision, recall, and F1 are equal.
        "top2_f1": _safe_mean(top2_recall),
        "binary_accuracy": _safe_mean(
            predicted_wolf.eq(true_wolf).float()
        ),
        "true_wolf_mean_prob": _safe_mean(probabilities[true_wolf]),
        "true_good_mean_prob": _safe_mean(probabilities[~true_wolf]),
    }


def compute_twd_region_metrics(
    hard_region,
    wolf_labels,
    attention_mask=None,
    supervision_mode="last",
):
    regions, labels = _select_metric_rows(
        hard_region.detach(),
        wolf_labels.detach(),
        attention_mask=attention_mask,
        supervision_mode=supervision_mode,
    )
    true_wolf = labels >= 0.5
    true_good = ~true_wolf
    pos = regions.eq(0)
    bnd = regions.eq(1)
    neg = regions.eq(2)

    total = regions.numel()
    pos_count = pos.sum()
    bnd_count = bnd.sum()
    neg_count = neg.sum()
    true_wolf_count = true_wolf.sum()
    true_good_count = true_good.sum()
    pos_correct = (pos & true_wolf).sum()
    neg_correct = (neg & true_good).sum()

    pos_precision = _safe_ratio(pos_correct, pos_count)
    pos_recall = _safe_ratio(pos_correct, true_wolf_count)
    neg_precision = _safe_ratio(neg_correct, neg_count)
    neg_recall = _safe_ratio(neg_correct, true_good_count)
    pos_f1 = _safe_ratio(
        2 * pos_precision * pos_recall,
        pos_precision + pos_recall,
    )
    neg_f1 = _safe_ratio(
        2 * neg_precision * neg_recall,
        neg_precision + neg_recall,
    )
    covered = pos_count + neg_count

    return {
        "POS_ratio": _safe_ratio(pos_count, total),
        "BND_ratio": _safe_ratio(bnd_count, total),
        "NEG_ratio": _safe_ratio(neg_count, total),
        "POS_precision": pos_precision,
        "POS_recall": pos_recall,
        "POS_f1": pos_f1,
        "NEG_precision": neg_precision,
        "NEG_recall": neg_recall,
        "NEG_f1": neg_f1,
        "true_wolf_BND_rate": _safe_ratio(
            (bnd & true_wolf).sum(),
            true_wolf_count,
        ),
        "true_good_BND_rate": _safe_ratio(
            (bnd & true_good).sum(),
            true_good_count,
        ),
        "coverage": _safe_ratio(covered, total),
        "selective_accuracy": _safe_ratio(
            pos_correct + neg_correct,
            covered,
        ),
    }


def compute_twd_tom_metrics(outputs, wolf_labels, attention_mask=None):
    metrics = compute_wolf_probability_metrics(
        outputs["wolf_prob"],
        wolf_labels,
        attention_mask=attention_mask,
    )
    metrics.update(
        compute_twd_region_metrics(
            outputs["hard_region"],
            wolf_labels,
            attention_mask=attention_mask,
        )
    )
    return metrics
