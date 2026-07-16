# TWD-ToM project structure

This document maps the current 3WD / TWDM / TWD-ToM repository structure after
safe cleanup, archiving, and source-layout tightening. Current runtime
entrypoints, main data, checkpoints, and experiment semantics remain unchanged.

## System Layers: 3WD Environment, TWDM Agent, and TWD-ToM Belief Model

3WD, TWDM, and TWD-ToM are not three names for the same component. They are
separate layers in the same repository.

### 3WD Environment

Role:

- Werewolf game environment.
- Game execution and runtime wiring.
- `observation` and `game_log` generation.

Typical files:

- `run_random.py`
- `run_battle.py`
- `werewolf/envs/`
- `werewolf/runtime_config.py`
- `werewolf/registry.py`

### TWDM Agent Layer

Role:

- Agent and strategy layer for gameplay.
- Used for TWDM/GPT/DeepSeek games, data generation, and comparison runs.

Typical files:

- `werewolf/agents/twdm_agent.py`
- `werewolf/agents/twdm_strategy.py`
- `werewolf/agents/llm_agent.py`
- `werewolf/backends/`
- `configs/deepseek_vs_twdm.yaml`
- `configs/gpt_vs_twdm.yaml`
- `configs/random_models.yaml`

TWDM configs are not default leftovers. Only archive them after confirming they
are no longer used for agent games, data generation, or comparison experiments.

### TWD-ToM Belief Model

Role:

- Observer-conditioned hidden-role inference.
- Input: an observer-visible `observation["game_log"]` encoded as structured
  event tokens.
- Output: `wolf_prob [7]` for seven players.

Typical files:

- `werewolf/encoding/event_encoder.py`
- `werewolf/encoding/dialogue_actions.py`
- `werewolf/speech/speech_perceiver.py`
- `werewolf/models/twd_tom/model.py`
- `werewolf/models/twd_tom/backbone.py`
- `werewolf/models/twd_tom/dataset.py`
- `werewolf/models/twd_tom/features.py`
- `werewolf/models/twd_tom/losses.py`
- `werewolf/models/twd_tom/metrics.py`
- `werewolf/models/risk/twd_risk_layer.py`
- `script/twd_tom/train.py`
- `script/twd_tom/eval.py`
- `script/twd_tom/eval_prior_baselines.py`
- `script/twd_tom/collect_samples.py`
- `configs/twd_tom_train*.yaml`

## Project layout

```text
configs/
  README_twd_tom.md
  twd_tom_train*.yaml
  twd_tom_collect.yaml
  twd_tom_multi_api.yaml
  twd_tom_deepseek_only_debug.yaml

script/
  twd_tom/
    __init__.py
    collect_samples.py
    train.py
    eval.py
    eval_prior_baselines.py
  collect_twd_tom_samples.py
  train_twd_tom.py
  eval_twd_tom.py
  eval_twd_tom_prior_baselines.py
  stats_winning.py

werewolf/encoding/
  dialogue_actions.py
  event_encoder.py

werewolf/speech/
  speech_perceiver.py

werewolf/models/twd_tom/
  model.py
  backbone.py
  dataset.py
  features.py
  labels.py
  losses.py
  metrics.py
  samples.py
  collector.py

werewolf/models/risk/
  twd_risk_layer.py

werewolf/models/
  dialogue_actions.py        # compatibility wrapper
  event_encoder.py           # compatibility wrapper
  speech_perceiver.py        # compatibility wrapper
  tom_backbone.py            # compatibility wrapper
  twd_tom_*.py               # compatibility wrappers
  twd_risk_layer.py          # compatibility wrapper

tests/
  README_twd_tom.md
  twd_tom/
    test_tom_backbone.py
    test_twd_tom*.py
    test_train_twd_tom_script.py
    test_eval_twd_tom_script.py
    test_eval_twd_tom_prior_baselines.py
  encoding/
    test_dialogue_actions.py
    test_event_encoder.py
  speech/
    test_speech_perceiver*.py
  risk/
    test_twd_risk_layer.py
  agents/
  runtime/

data/twd_tom/debug/
  game_001_030.jsonl
  game_001_060.jsonl
  game_031_060.jsonl

checkpoints/
  generated model checkpoints, config snapshots, and train_history.json files

logs/
  generated game logs from new collection runs

archive/
  README.md
  intermediate_data/
  old_checkpoints/
  old_logs/
  design_notes/
  audit/
```

## Core model files

- `werewolf/encoding/dialogue_actions.py`: token id dictionaries for structured
  event fields.
- `werewolf/encoding/event_encoder.py`: converts visible
  `observation["game_log"]` into 10-field event tokens.
- `werewolf/speech/speech_perceiver.py`: LLM speech-to-structured-claim parser.
- `werewolf/models/twd_tom/backbone.py`: observer-conditioned sequence backbones:
  Transformer/GPT2Block, Bag-of-events MLP, GRU, LLaMA-style, and GPT-NeoX-style.
- `werewolf/models/twd_tom/model.py`: wrapper combining ToM backbone outputs with
  `TWDRiskLayer`.
- `werewolf/models/twd_tom/dataset.py`: JSONL dataset and collate path.
- `werewolf/models/twd_tom/features.py`: feature builder for `event_tokens`,
  `attention_mask`, and `observer_id`.
- `werewolf/models/twd_tom/losses.py`: BCE last-token supervision and optional
  cardinality regularization.
- `werewolf/models/twd_tom/metrics.py`: eval-only probability and TWD region
  metrics, including `top2_f1`.
- `werewolf/models/twd_tom/collector.py`, `samples.py`, `labels.py`: sample
  generation and label helpers.
- `werewolf/models/risk/twd_risk_layer.py`: TWD region/risk layer.

Backward-compatible wrappers remain under `werewolf/models/*.py` for historical
imports. New code should prefer the package paths above. Because
`werewolf/models/twd_tom/` is now a package directory, compatibility for
`from werewolf.models.twd_tom import ...` is provided by
`werewolf/models/twd_tom/__init__.py`.

## Training / evaluation scripts

- `script/twd_tom/train.py`: canonical training entrypoint. It reads JSONL
  samples, splits by `game_id`, trains `TWDToMModel`, logs loss-only summaries,
  and saves checkpoints.
- `script/twd_tom/eval.py`: canonical checkpoint evaluation entrypoint. Stdout
  remains exactly `eval_loss=... top2_f1=...`.
- `script/twd_tom/eval_prior_baselines.py`: canonical no-learning baselines for
  uniform prior and random top-2.
- `script/twd_tom/collect_samples.py`: canonical sample collection wrapper
  around `run_random.py`.
- `script/train_twd_tom.py`, `script/eval_twd_tom.py`,
  `script/eval_twd_tom_prior_baselines.py`, and
  `script/collect_twd_tom_samples.py`: thin compatibility wrappers for older
  commands.
- `script/stats_winning.py`: generated-log win-rate summary helper.

## Config files

Current experiment training configs are kept under `configs/` and documented in
`configs/README_twd_tom.md`.

The current main experiment uses:

- Dataset: `data/twd_tom/debug/game_001_060.jsonl`
- Main config: `configs/twd_tom_train.yaml`
- Checkpoint directory: `checkpoints/twd_tom_v05/full_game_001_060`

## Dataset files

The debug JSONL files under `data/twd_tom/debug/` are experiment datasets rather
than source code. They are currently useful for reproduction, so they are not
ignored by `.gitignore` in this cleanup pass.

Known debug datasets:

- `data/twd_tom/debug/game_001_030.jsonl`
- `data/twd_tom/debug/game_001_060.jsonl`
- `data/twd_tom/debug/game_031_060.jsonl`

Archived intermediate data:

- `archive/intermediate_data/game_031_060_raw.jsonl`

## Checkpoint outputs

`checkpoints/` contains generated artifacts:

- `checkpoint_best.pt`
- `checkpoint_last.pt`
- `config.yaml`
- `train_history.json`

These files should be kept locally for reproducibility but are normally not
reviewable source files. `.gitignore` includes `checkpoints/` so new checkpoint
outputs do not enter version control accidentally.

Current main checkpoints stay in place:

- `checkpoints/twd_tom_v05/full_game_001_060`
- `checkpoints/twd_tom_v05/boe_mlp_game_001_060`
- `checkpoints/twd_tom_v05/gru_game_001_060`
- `checkpoints/twd_tom_v05/wo_observer_id_game_001_060`
- `checkpoints/twd_tom_v05/wo_cardinality_game_001_060`

Historical checkpoints are archived under `archive/old_checkpoints/`.

## Logs and archive

Older DeepSeek rollout logs were moved from `logs/twd_tom_v05_deepseek_debug/`
to `archive/old_logs/twd_tom_v05_deepseek_debug/`.

Current training does not read raw log directories. It reads JSONL sample files
under `data/twd_tom/debug/`. New data collection runs may still write to
`logs/`.

## Test files

TWD-ToM test coverage is documented in `tests/README_twd_tom.md`. The focused
TWD-ToM test suite is:

```bash
PYTHONPATH=. python -m unittest discover tests/twd_tom
```

The full suite remains:

```bash
PYTHONPATH=. python -m unittest discover tests
```

## Common commands

Main training:

```bash
PYTHONPATH=. python script/twd_tom/train.py \
  --config configs/twd_tom_train.yaml \
  --data_path data/twd_tom/debug/game_001_060.jsonl \
  --output_dir checkpoints/twd_tom_v05/full_game_001_060
```

Main evaluation:

```bash
PYTHONPATH=. python script/twd_tom/eval.py \
  --config checkpoints/twd_tom_v05/full_game_001_060/config.yaml \
  --checkpoint_path checkpoints/twd_tom_v05/full_game_001_060/checkpoint_best.pt
```

Prior baselines:

```bash
PYTHONPATH=. python script/twd_tom/eval_prior_baselines.py \
  --config checkpoints/twd_tom_v05/full_game_001_060/config.yaml \
  --data_path data/twd_tom/debug/game_001_060.jsonl \
  --num_trials 10000
```

Collection:

```bash
PYTHONPATH=. python script/twd_tom/collect_samples.py \
  --num_games 30 \
  --config configs/twd_tom_deepseek_only_debug.yaml \
  --output_dir logs/twd_tom_v05_deepseek_debug/game_new \
  --samples_path data/twd_tom/debug/game_new.jsonl \
  --overwrite
```

## Audit summary

- Canonical TWD-ToM script entrypoints live under `script/twd_tom/`.
- Legacy `script/*twd_tom*.py` files are compatibility wrappers only.
- Data collection still calls `run_random.py` and reads/writes the same config,
  log, and JSONL sample paths.
- Still-used configs: `twd_tom_train.yaml`, backbone comparison configs,
  ablation configs, and collection configs listed above.
- Debug/generated artifacts: `checkpoints/`, `logs/`, `archive/`, and
  `data/twd_tom/debug/*.jsonl`. Do not delete them during structure cleanup.
- Version-control candidates to ignore: `checkpoints/`, `logs/`, `__pycache__/`,
  `.codegraph/`, `.DS_Store`, and `*.pyc`. Debug datasets remain visible
  because they are referenced by reproduction commands.
