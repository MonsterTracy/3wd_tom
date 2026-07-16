# TWD-ToM experiments

This document records the current runnable experiment commands and the latest
main comparison table. It is documentation-only and does not change model,
loss, metric, training, or evaluation behavior.

## Data collection

Collection is driven by the canonical `script/twd_tom/collect_samples.py`
entrypoint, which repeatedly invokes `run_random.py` with a runtime config.
The older `script/collect_twd_tom_samples.py` path is retained only as a thin
compatibility wrapper.

Example:

```bash
PYTHONPATH=. python script/twd_tom/collect_samples.py \
  --num_games 30 \
  --config configs/twd_tom_deepseek_only_debug.yaml \
  --output_dir logs/twd_tom_v05_deepseek_debug/game_new \
  --samples_path data/twd_tom/debug/game_new.jsonl \
  --overwrite
```

Collection outputs are generated artifacts. Keep local logs and JSONL datasets,
but do not delete them during project-structure cleanup. Older DeepSeek rollout
logs have been archived under `archive/old_logs/`; new collection runs may still
write to `logs/`.

## Training

Current main training command:

```bash
PYTHONPATH=. python script/twd_tom/train.py \
  --config configs/twd_tom_train.yaml \
  --data_path data/twd_tom/debug/game_001_060.jsonl \
  --output_dir checkpoints/twd_tom_v05/full_game_001_060
```

Training logs are intentionally loss-only:

```text
step=... train_loss=...
epoch=... train_loss=... eval_loss=... lowest_eval_loss=...
```

The training path uses `twd_tom_loss` with BCE and optional cardinality loss.
It does not compute or print top-2, binary accuracy, POS/BND/NEG, or coverage
metrics during training.

## Evaluation

Current main evaluation command:

```bash
PYTHONPATH=. python script/twd_tom/eval.py \
  --config checkpoints/twd_tom_v05/full_game_001_060/config.yaml \
  --checkpoint_path checkpoints/twd_tom_v05/full_game_001_060/checkpoint_best.pt
```

Stdout contract:

```text
eval_loss=... top2_f1=...
```

Do not add `top2_exact`, `top2_recall`, binary accuracy, POS/BND/NEG, or
coverage to this stdout path.

## Prior baselines

Current baseline command:

```bash
PYTHONPATH=. python script/twd_tom/eval_prior_baselines.py \
  --config checkpoints/twd_tom_v05/full_game_001_060/config.yaml \
  --data_path data/twd_tom/debug/game_001_060.jsonl \
  --num_trials 10000
```

The prior baseline script reports only:

```text
uniform_eval_loss=... uniform_top2_f1=...
random_top2_f1=...
```

## Archived artifacts

The current main experiment paths remain active:

- Main dataset: `data/twd_tom/debug/game_001_060.jsonl`
- Main checkpoint: `checkpoints/twd_tom_v05/full_game_001_060`

Archived historical artifacts:

- Old rollout logs: `archive/old_logs/`
- Old checkpoints: `archive/old_checkpoints/`
- Intermediate raw data: `archive/intermediate_data/`
- Historical design notes: `archive/design_notes/`
- Cleanup audit snapshots: `archive/audit/`

Do not replace the main training/evaluation commands with archive paths.

## Ablation configs

- `configs/twd_tom_train_wo_observer_id.yaml`: disables observer-id embedding.
- `configs/twd_tom_train_wo_cardinality.yaml`: sets
  `loss.cardinality_weight: 0.0`.

## Backbone comparison configs

- `configs/twd_tom_train_boe_mlp.yaml`: Bag-of-events MLP baseline.
- `configs/twd_tom_train_gru.yaml`: GRU baseline.
- `configs/twd_tom_train_llama.yaml`: LLaMA-style decoder backbone.
- `configs/twd_tom_train_gpt_neox.yaml`: GPT-NeoX-style decoder backbone.

All backbone configs keep the same structured event-token input and multi-label
sigmoid output semantics.

## Source layout note

The active TWD-ToM implementation is split by subsystem:

- Encoding: `werewolf/encoding/`
- Speech parsing: `werewolf/speech/`
- Belief model: `werewolf/models/twd_tom/`
- Risk layer: `werewolf/models/risk/`
- Canonical script entrypoints: `script/twd_tom/`

Historical imports under `werewolf/models/*.py` are retained as compatibility
wrappers. Historical script paths such as `script/train_twd_tom.py` are also
retained as compatibility wrappers. Experiment commands and config semantics are
unchanged.

## Current main results

These results are based on `data/twd_tom/debug/game_001_060.jsonl`, split by
`game_id`, using the checkpoint with the lowest `eval_loss`.

| Method | Eval Loss | Top-2 F1 |
|---|---:|---:|
| Event-MLP | 0.6169 | 0.3491 |
| GRU | 0.6148 | 0.3319 |
| No-Observer | 0.6278 | 0.4986 |
| No-Cardinality | 0.5909 | 0.5848 |
| 3WD-TOM | 0.5724 | 0.6020 |

## Safety constraints

- Do not replace `observation["game_log"]` with global `env.game_log`.
- Do not feed raw speech or source text into the model.
- Do not replace sigmoid multi-label outputs with softmax.
- Do not replace BCE/cardinality loss with cross-entropy, KL, language-modeling,
  or next-token-prediction losses.
- Do not move or delete generated datasets, logs, or checkpoints without an
  explicit migration plan.
