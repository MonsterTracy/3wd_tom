# TWD-ToM tests

The TWD-ToM test suite is intentionally split by subsystem. Do not merge or
delete these files just to reduce file count; the separation makes regressions
easier to localize.

## Core model and backbone tests

- `tests/twd_tom/test_tom_backbone.py`: event-token backbone coverage for Transformer,
  Bag-of-events MLP, GRU, LLaMA-style, and GPT-NeoX-style paths, including
  observer-id conditioning and padding masks.
- `tests/twd_tom/test_twd_tom.py`: wrapper-level checks for `TWDToMModel`,
  `TWDRiskLayer` integration, output keys, and output shapes.
- `tests/twd_tom/test_import_compatibility.py`: historical
  `werewolf.models.*` import wrappers.

## Data and feature tests

- `tests/twd_tom/test_twd_tom_features.py`: feature builder and collator output
  shapes, including `event_tokens [B,T,10]`, `attention_mask`, and
  `observer_id`.
- `tests/twd_tom/test_twd_tom_dataset.py`: JSONL dataset loading, collate behavior,
  `wolf_labels`, `observer_id`, and `alive_mask` metadata support.
- `tests/twd_tom/test_twd_tom_samples.py`: sample construction helpers and
  sample schema.
- `tests/twd_tom/test_twd_tom_collector.py`: rollout collector integration
  points.

## Loss and metrics tests

- `tests/twd_tom/test_twd_tom_losses.py`: last-token supervision, all-token
  supervision, BCE, soft labels, attention masks, and cardinality loss.
- `tests/twd_tom/test_twd_tom_metrics.py`: probability metrics, `top2_f1`, TWD
  region metrics, and BND edge cases.

## Encoding, speech, risk, agent, and runtime tests

- `tests/encoding/test_dialogue_actions.py`: event id dictionaries.
- `tests/encoding/test_event_encoder.py`: structured event-token encoding.
- `tests/speech/test_speech_perceiver*.py`: speech parser availability and
  environment integration.
- `tests/risk/test_twd_risk_layer.py`: risk-layer behavior.
- `tests/agents/`: backend and prompt-constraint tests.
- `tests/runtime/`: runtime config and run-script wiring tests.

## Script tests

- `tests/twd_tom/test_train_twd_tom_script.py`: train config loading, game-id
  split, loss-only history/checkpoint fields, and backbone config construction.
- `tests/twd_tom/test_eval_twd_tom_script.py`: checkpoint-best loading and eval
  stdout contract: `eval_loss=... top2_f1=...`.
- `tests/twd_tom/test_eval_twd_tom_prior_baselines.py`: uniform prior and
  random-top2 baseline helpers.

Script tests import the canonical `script.twd_tom.*` modules. Legacy
`script/train_twd_tom.py`, `script/eval_twd_tom.py`,
`script/eval_twd_tom_prior_baselines.py`, and
`script/collect_twd_tom_samples.py` remain thin compatibility wrappers.

## Recommended focused command

```bash
PYTHONPATH=. python -m unittest discover tests/twd_tom
```

Full `unittest discover tests` also covers runtime/backend tests and may
require optional dependencies such as `python-dotenv`.
