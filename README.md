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

Load the key into the shell and collect exactly one game with:

```bash
set -a
source .env
set +a
python -m script.tom.collect \
  --config configs/tom/collect.yaml \
  --games 1 \
  --run-id game_001 \
  --data-dir data \
  --log-dir logs
```

One command represents one game. Number runs sequentially from `game_001`; the
same run ID creates `data/game_001/` for samples, failures, and audit, plus
`logs/game_001/` for the game and seven player logs. The command refuses to run
if either run directory already exists, so it never appends to or overwrites a
previous game. A failed game keeps both directories and its partial artifacts
for diagnosis, but its audit has `collection_status="failed"` and the Dataset
loader rejects it for training. Start the retry with a new run ID rather than
reusing an existing directory. The former dated pilot layout is no longer used.

## Ruleset and Prompt Protocol V6

`werewolf/game_rules.py` is the sole machine-readable game-rule source. The
current ruleset is `werewolf_7p.zh.v1` (`id: werewolf_7p`) and covers both
supported seven-player variants, `seer_witch` and `seer_guard`. Prompt text
renders rules from that module; this README intentionally does not duplicate
the complete rules.

`werewolf/prompt_protocol.py` defines the sole Prompt Protocol V6. Its formal
instruction language is Chinese (`language: zh-CN`) and it has three canonical
protocols:

- `gameplay.zh.v4` layers rendered global rules, current-role rules, visibility
  boundaries, and phase tasks. Its dynamic message separates confirmed private
  facts, public objective events, and untrusted public player claims. Every
  finite action is rendered with a zero-based `option_index`, its action, and
  its `target_player`; `action_index` selects that option position, not a player
  ID. Gameplay output uses strict Python-parseable JSON; backend retries and
  semantic repair are separate, and invalid actions terminate the rollout
  instead of falling back to synthetic actions.
- `belief.zh.v3` measures one player's private joint MAP belief over the complete
  two-Werewolf pair from the same three information partitions. Its canonical
  messages include the lowercase `json` instruction required by JSON Output. It
  does not continue gameplay, affect later actions, or request reasoning.
- `parser.zh.v3` extracts only explicit local speech semantics into the existing
  controlled event vocabulary. It defines every qualifier enum, normalizes only
  registered aliases, distinguishes evidence and certainty semantics, constrains
  action-position extraction, and includes eleven fixed Chinese JSON few-shots.

The Speech Parser maps public utterances to this manually defined controlled
event set; it does not decide whether a player's statement is true, mistaken,
or deceptive. `evidence_source` is an optional auxiliary qualifier, and an
ambiguous canonical value is not a fatal collection condition. Downstream ToM
models learn the social reasoning relationships rather than receiving them from
keyword rules.

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
checkpoints. The strict loader rejects earlier Prompt Protocol versions rather
than adapting them.

Prompt metadata is a data-generation condition, not a model feature. A trained
model therefore represents the behavior distribution induced by its recorded
Prompt-Agent protocol; evaluation rejects data from a different protocol.

## Commands

```bash
python -m script.tom.train --config configs/tom/first_order.yaml
python -m script.tom.eval --config configs/tom/evaluate.yaml
```

The canonical sequence backbone is a randomly initialized Hugging Face
`GPT2Model` whose causal stack consists only of `GPT2Block` layers. It consumes
the summed structured-event field embeddings through `inputs_embeds`; no
pretrained checkpoint, tokenizer, text head, or second position embedding is
used. GRU and bag-of-events MLP remain explicit ablations. `model.v2`
checkpoints record the exact Transformers version and GPT-2 structure and are
not compatible with the former TransformerEncoder `model.v1` checkpoints. The
conditioning ablations remove first-order private events or second-order target
embeddings. The canonical primary objective remains masked 21-class pair
cross-entropy. Strict `train.v2` configurations add one optional auxiliary:
`total_loss = pair_cross_entropy + marginal_bce_weight * marginal_bce`, with
`marginal_bce_weight: 0.0` as the canonical default. The auxiliary does not add
a player-output head or replace the pair objective.

Evaluation keeps that 21-class distribution over unordered Werewolf pairs as
the model's primary output. It also deterministically marginalizes the joint
distribution to seven player probabilities: each value is the probability that
the corresponding player belongs to the predicted pair, so the seven values
sum to 2, not 1, and no player softmax is applied. Marginal BCE compares these
seven values directly with the elicited pair's seven-player two-hot target. It
is distinct from normalized marginal KL/cross entropy, which divide the
marginals by 2 and assign 0.5 to each elicited player. Raw pair CE and marginal
BCE values should not be compared as if they were on the same scale.
In addition to pair NLL, accuracy, top-3 accuracy, Brier score, and player
marginal MAE, evaluation reports normalized player-marginal KL and cross
entropy, player-marginal Brier, top-2 recall, and the same two-hot player
marginal BCE used by the optional auxiliary.

Training history records pair, marginal, and total losses separately for train
and validation. Best-checkpoint selection always uses validation pair
cross-entropy (`valid_pair_loss`), never total loss or marginal BCE, so weights
remain comparable on the primary pair objective. The five-game marginal-BCE
experiments are smoke ablations only and are not research conclusions.

Every eval writes the aggregate `evaluation.json` plus
`evaluation.predictions.jsonl` (derived from the configured output stem). Each
prediction row contains the 21-way boolean `output_mask`, masked pair
probabilities, and their seven player marginals. Its `elicited_label_pair` is
the target player's independently
elicited subjective belief, never the environment's true roles: first-order
uses the observer's legal view and own elicitation; second-order `public_only`
uses public history and a target; second-order `wolf_conditioned` uses the wolf
modeler's legal view and a target. Both second-order labels remain the target's
elicited pair. No objective role table or actual wolf pair is exported.

## Data contract

Only successful `tom.v1_1` JSONL records from a complete collection audit are
trainable. Each record names its
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
