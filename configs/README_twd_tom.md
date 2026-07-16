# TWD-ToM configs

This directory keeps runnable configuration files in place for backward
compatibility. Do not move or delete these files without providing wrapper
commands or migration notes, because the training, evaluation, and collection
scripts load them by path.

Canonical TWD-ToM script entrypoints now live under `script/twd_tom/`:

- `script/twd_tom/train.py`
- `script/twd_tom/eval.py`
- `script/twd_tom/eval_prior_baselines.py`
- `script/twd_tom/collect_samples.py`

Older `script/*twd_tom*.py` paths, if present, are compatibility wrappers only.

## Data collection configs

- `twd_tom_collect.yaml`: mixed DeepSeek/local-vLLM collection config for
  `script/twd_tom/collect_samples.py` via `run_random.py`.
- `twd_tom_multi_api.yaml`: multi-backend collection/runtime config with
  DeepSeek plus local vLLM candidates.
- `twd_tom_deepseek_only_debug.yaml`: DeepSeek-only debug collection/runtime
  config.

## Main training config

- `twd_tom_train.yaml`: full TWD-ToM with the default GPT2Block/Transformer
  backbone, observer conditioning, last-token BCE, and cardinality loss.

## Backbone comparison configs

- `twd_tom_train_boe_mlp.yaml`: Bag-of-events/Event-MLP baseline.
- `twd_tom_train_gru.yaml`: GRU-ToM baseline.
- `twd_tom_train_llama.yaml`: LLaMA-style random-initialized decoder backbone
  comparison.
- `twd_tom_train_gpt_neox.yaml`: GPT-NeoX-style random-initialized decoder
  backbone comparison.

## Ablation configs

- `twd_tom_train_wo_observer_id.yaml`: no-observer-id ablation; keeps the
  Transformer backbone and cardinality loss.
- `twd_tom_train_wo_cardinality.yaml`: no-cardinality-loss ablation; keeps the
  Transformer backbone and observer conditioning.

## Field conventions for training configs

All `twd_tom_train*.yaml` files should explicitly include:

- `model.backbone_type`
- `model.use_observer_id`
- `loss.cardinality_weight`
- `checkpoint.monitor_metric: eval_loss`
- `checkpoint.mode: min`

These fields are intentionally explicit even when they match script defaults,
so experiment configs can be compared without reading
`script/twd_tom/train.py`.
