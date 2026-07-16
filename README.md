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
OPENAI_API_KEY=replace-me
```

Select the OpenAI-compatible base URL, model, parser, and role-group agent
profiles explicitly in `configs/tom/collect.yaml`. Legacy runtime YAML is
rejected rather than normalized.

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

Only successful `tom.v1` JSONL records are trainable. Each record names its
task/mode/checkpoint, carries the exact input events and 21-way output mask, and
stores the elicited pair/index plus raw elicitation metadata. Failed records are
kept separately with raw responses and errors. The loader deliberately rejects
older schemas; ground-truth role vectors cannot be converted into subjective guesses.

A two-record synthetic fixture is available at
`tests/fixtures/tom_v1.jsonl` for schema and smoke tests.

## Verification

```bash
python -m pytest -q
```
