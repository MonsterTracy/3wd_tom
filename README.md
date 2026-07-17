# Seven-player Werewolf ToM

This repository models subjective wolf-pair beliefs in a fixed seven-player,
two-wolf game. The model output is the global set of `C(7, 2) = 21` unordered
wolf pairs. It does not use ground-truth seven-player role-vector supervision.

## Supported tasks

First-order ToM predicts a living non-wolf observer's own guessed wolf pair from
the public event stream, that observer's private events, and the observer ID.
Its `first_order_knowledge_mask` excludes pairs contradicted by the observer's
actual private knowledge. Wolves are not collected for the main first-order
dataset; their one-class knowledge mask exists only for tests and diagnostics.

Second-order ToM predicts a living non-wolf target's guessed pair:

- `public_only`: public events plus target ID, with all 21 outputs available.
- `wolf_conditioned`: public events plus one wolf modeler's private events,
  modeler ID, and target ID. Pairs containing the known non-wolf target are
  excluded. The target's role/private view and the true wolf pair are never
  model inputs or output masks.

Dead players remain in the 21-class identity space. The living-player set only
controls whose belief is elicited and which game actions are legal.
Tracking IDs are never model features; public dataset releases should remove
date-bearing IDs or renumber them while preserving whole-game grouping.

## Unified events

`werewolf/events/` is the only event representation. Environment facts are
deterministic; an LLM parser is used only for speech-derived local semantics.
The six families are:

```text
BELIEF_ASSERTION  SOCIAL_STANCE  ACTION_POSITION
CLAIM_RESPONSE    GAME_EVENT     PRIVATE_FACT
```

Every event carries versioned identity, time, visibility, speaker/target,
content, qualifiers, source anchoring, and parser confidence fields. Public and
per-player private streams are derived from the same records.

Collection occurs only after state-changing checkpoints: speech, public vote,
vote result, exile, death, role reveal, seer result, wolf-team information,
witch information, and guard result. Belief elicitation is a separate backend
call using the same player backend/model by default. It gets one repair retry;
invalid results are written to the failure stream and never become training
samples or interrupt the game.

## Layout

```text
werewolf/events/       event schema, builders, streams, parser, encoder
werewolf/tom/          pair space, masks, collection, data, model, metrics
script/tom/            sole collect/train/eval commands
configs/tom/           canonical runtime and experiment configs
tests/events/          event unit tests
tests/tom/             belief and model unit tests
tests/integration/     full-game and tiny train/evaluate smokes
```

## Installation

Python 3.10 or newer is required; the checked environment uses Python 3.12.

```bash
conda env create -f environment.yml
conda activate werewolf-tom
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

For API collection, create an untracked `.env`:

```dotenv
DEEPSEEK_API_KEY=replace-me
```

The canonical collection config uses DeepSeek's OpenAI-compatible endpoint and
`deepseek-chat` for gameplay, belief elicitation, and speech parsing. The YAML
contains only runtime choices such as backend, model, and gameplay temperature;
canonical prompt text cannot be overridden from YAML. Legacy runtime YAML is
rejected rather than normalized.

Load the key into the shell and run a one-game pilot with:

```bash
set -a
source .env
set +a
python -m script.tom.collect \
  --config configs/tom/collect.yaml \
  --games 1 \
  --output-dir data/tom/pilot_YYYYMMDD_001
```

Every pilot must use a new `--output-dir`; the collect command creates it and
refuses any path that already exists. Generated JSON/JSONL under `data/tom/` is
ignored by Git. If the audit fails, samples, failures, and audit output remain in
that directory for diagnosis, so do not reuse an old pilot directory.

## Ruleset and Prompt Protocol V2

`werewolf/game_rules.py` is the sole machine-readable game-rule source. The
current ruleset is `werewolf_7p.zh.v1` (`id: werewolf_7p`) and covers both
supported seven-player variants, `seer_witch` and `seer_guard`. Prompt text
renders rules from that module; this README intentionally does not duplicate
the complete rules.

`werewolf/prompt_protocol.py` defines the sole Prompt Protocol V2. Its formal
instruction language is Chinese (`language: zh-CN`) and it has three canonical
protocols:

- `gameplay.zh.v2` layers rendered global rules, current-role rules, visibility
  boundaries, and phase tasks. Its dynamic message separates confirmed private
  facts, public objective events, and untrusted public player claims.
- `belief.zh.v2` measures one player's private joint MAP belief over the complete
  two-Werewolf pair from the same three information partitions. It does not
  continue gameplay, affect later actions, or request reasoning.
- `parser.zh.v2` extracts only explicit local speech semantics into the existing
  controlled event vocabulary and includes six fixed Chinese JSON few-shots.

Natural-language instructions are Chinese, while machine-readable JSON keys
(`speech`, `action_index`, `wolf_pair`), role/camp values, event families,
`content.kind`/`content.value`, and qualifier enums remain in English.

Each complete stable protocol is normalized to LF line endings and hashed with
SHA-256. Gameplay covers system/user structures, role/phase/output/repair
templates and the ruleset reference; belief covers system/user/constraint/
output/repair structures and the ruleset reference; parser covers its system,
controlled enums, user/repair structures, and every few-shot. Dynamic values,
event history, model responses, and credentials are excluded. The derived
`protocol_id` also includes the ruleset ID, version, and hash. These references
are written to `tom.v1_1` samples, audits, parser metadata, gameplay logs, and
checkpoints. The strict loader rejects Prompt Protocol V1 rather than adapting
it.

Prompt metadata is a data-generation condition, not a model feature. A trained
model therefore represents the behavior distribution induced by its recorded
Prompt-Agent protocol; evaluation rejects data from a different protocol.

## Commands

```bash
python -m script.tom.collect --config configs/tom/collect.yaml
python -m script.tom.train --config configs/tom/first_order.yaml
python -m script.tom.eval --config configs/tom/evaluate.yaml
```

Formal model variants are Transformer, GRU, and bag-of-events MLP. The retained
conditioning ablations remove first-order private events or second-order target
embeddings. All main configurations still optimize one masked 21-class
cross-entropy objective.

## Data contract

Only successful `tom.v1_1` JSONL records are trainable. Each record names its
task/mode/checkpoint, carries the exact input events and 21-way output mask, and
stores the elicited pair/index plus raw elicitation metadata and one required
`prompt_protocol` object containing the unique `ruleset` reference. Failed
records carry the same protocol object and are
kept separately with raw responses and errors. The loader deliberately rejects
older schemas and mixed-protocol files; ground-truth role vectors cannot be
converted into subjective guesses.

A two-record synthetic fixture is available at
`tests/fixtures/tom_v1.jsonl` for schema and smoke tests.

## Verification

```bash
python -m pytest -q
```
