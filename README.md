This is the implementation of paper [Multi-agent KTO: Reinforcing Strategic Interactions of Large Language Model in Language Game](https://arxiv.org/abs/2501.14225)


## Dataset

All the dataset can be downloaded at https://huggingface.co/datasets/ReneeYe/werewolf_game_reasoning.

The following is how to prepare SFT data from the raw game record.

### SFT Dataset preparation
The sample of dataset is under `data_sample/`. Due to the limitation of file size, in the path, we provide samples of 10 games and script to process the game behavoior data into SFT dataset. See `data_sample/README.md` for more details.

### SFT
After prepared SFT dataset in json format, you can train SFT model based on Base model like [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).

For SFT training, you may follow the instructions in [TRL](https://huggingface.co/docs/trl/en/sft_trainer) or use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) code base.

## How to run Werewolf Game

Current TWDM runs all large language model generation through APIs.

TWDM Agent is not a local LLM backend. TWDM Agent is a strategy-guided agent structure: it builds role-aware and phase-aware strategy hints locally, injects those hints into the prompt, and still calls the configured API backend for generation.

Which players use TWDM is decided by config `model_type`, not by role hardcoding. In the default experiment, werewolf players use `twdm_agent` and villagers use a normal `deepseek` or `gpt` agent, but any role group can be configured as `twdm_agent`.

### 0. Installation

We recommend do it in a virtual env. Take `conda` for example:
```Bash
conda create -n werewolf python=3.10
conda activate werewolf
pip install -e .
```
The commands will create and activate an env called `werewolf`. And the dependencies and packages will be correctly installed.

### 1. Configure API credentials

Supported `model_type` values include:
- `twdm_agent`: TWDM strategy-guided agent structure.
- `deepseek`: normal API agent using an OpenAI-compatible backend.
- `gpt`, `gpt4`, `gpt4o`, `o1`: normal API agents using the same backend interface.

The online entry points use `werewolf.backends.OpenAICompatibleBackend`.
`run_battle.py` and `run_random.py` load `.env` automatically, then apply the
top-level `backend` block from YAML. Explicit YAML values override environment
values.

Minimal `.env`:
```dotenv
OPENAI_API_KEY=replace-with-your-api-key
OPENAI_API_BASE=https://api.deepseek.com
DEFAULT_LLM_MODEL=deepseek-chat
AGENT_MODEL=deepseek-chat
PARSER_MODEL=deepseek-chat
```

`AGENT_MODEL` and `PARSER_MODEL` may be different. Both fall back to
`DEFAULT_LLM_MODEL` when omitted. A YAML `base_url`, `agent_model`, or
`parser_model` takes precedence over the matching `.env` value.

### 2. Define Config Yaml

Current TWDM runs only support 7-player Werewolf:
- `7p_seer_witch`: 2 Werewolf, 1 Seer, 1 Witch, 3 Villager.
- `7p_seer_guard`: 2 Werewolf, 1 Seer, 1 Guard, 3 Villager.

Terminology:
- Villager is the canonical role name for ordinary villagers.
- ordinary_villager means only players whose role is Villager.
- village_team means non-werewolf players: Villager, Seer, Witch, Guard.
- agent_config.village_team is required for village-team agents.

Use the current examples in `configs/deepseek_twdm_vs_deepseek.yaml`, `configs/deepseek_vs_twdm.yaml`, `configs/gpt_vs_twdm.yaml`, or `configs/random_models.yaml`.

Example:
```yaml
backend:
    type: openai_compatible
    base_url: https://api.deepseek.com
    agent_model: deepseek-chat
    parser_model: deepseek-chat

env_config:
    n_player: 7
    n_role: 4
    n_werewolf: 2
    n_seer: 1
    n_guard: 0
    n_witch: 1
    n_hunter: 0
    n_villager: 3

agent_config:
    werewolf:
        model_type: twdm_agent
        model_params:
            temperature: 1.0
            twdm_config:
                enable_strategy: true
                enable_suspicion: false
                enable_mcts: false
    village_team:
        model_type: deepseek
        model_params:
            temperature: 1.0
```

At runtime the entry point creates one backend, injects `agent_model` into all
agents, creates `SpeechPerceiver(backend, parser_model)`, and constructs
`WerewolfTextEnvV0(speech_perceiver=speech_perceiver)`. The environment does
not read API keys and does not know model names.

### 3. Run battles

Run battles using the example scripts:

#### Run a single head-to-head game:
```bash
game_path=./trial_logs
python run_battle.py \
  --config configs/deepseek_twdm_vs_deepseek.yaml \
  --log_save_path logs/test_run_001
```

The API key must be available through `.env` or the process environment. The
completed game is written to:

```text
logs/test_run_001/game_log.json
```

Every `speech` and `speech_pk` entry contains both the original speech and the
online parser output:

```json
{
  "event": "speech",
  "content": {
    "speech_content": "3号的逻辑有问题，我会投3号。",
    "parsed_claims": [
      {
        "speaker": 2,
        "predicate": "vote_intention",
        "target": 3,
        "role": null,
        "polarity": "negative",
        "certainty": "explicit",
        "condition": null,
        "source_text": "我会投3号"
      }
    ]
  }
}
```

If parser generation or JSON decoding fails, `parsed_claims` is still present
and equals `[]`; the game continues.

#### Run multiple games:
```Bash
game_path=./trial_logs
Bash run_batch.sh configs/gpt_vs_twdm.yaml ${game_path} 10
```

Then, you will get logs under `./trial_logs/game_1/`:
```angular2html
./trial_logs/game_1
├── config.yaml
├── game_log.json
├── Player_1.jsonl
├── Player_2.jsonl
├── Player_3.jsonl
├── Player_4.jsonl
├── Player_5.jsonl
├── Player_6.jsonl
└── Player_7.jsonl
```
For each line in `Player_${i}.jsonl`, it is a json object with the following fields:
```angular2html
{
    "message": "<phase>",
    "prompt": "<prompt>",
    "response": "<response>",
    "phase": "<phase>",
    "gen_times": "<gen_times>"
}
```

#### Run random competition:

The configuration file is `configs/random_models.yaml`.

```Bash
game_path=./random_competition_logs
python run_random.py --config configs/random_models.yaml --log_save_path ${game_path}/game_1
```

### 5. View Battle Results
1. You may use `stats_winning.py` to stats the winning rate of each agent.
```Bash
cd scripts
python3 stats_winning.py --game_dir {game_path}
```

2. You may use `game_visualizer.py` to visualize the game log.
```Bash
cd scripts
python3 game_visualizer.py --game_dir {game_path} --model_setting {model_setting}
```
This script will start a gradio server, you can view the game log and the detailed behavior of each agent player in the browser.

`game_path` is the path to the game log directory, e.g., `./trial_logs`, and `model_setting` is the setting of games in the format `w-{werewolf_model_type}_vs_v-{villager_model_type}`, e.g., `w-sft_vs_v-gpt4o`.

## MaKTO training
After accumulated enough behavior data of SFT model, you can apply Multi-agent KTO to train a Makto agent. The training process includes data preparation and KTO training.

### 1. Training data preparation
All the data preparation scripts are under `MaKTO_data_extraction/`. A sample script to extract preference data is provided in `MaKTO_data_extraction/extract_script.sh`.

Here are some explanations:
- `get_bad_speech.py`: Extract bad speech from game logs.
- `get_bad_action.py`: Extract bad actions from game logs, using Heuristic-based method.
- `get_good_action.py`: Extract good actions from game logs, using Heuristic-based method.
- `get_bad_vote.py`: Extract bad votes from game logs.
- `get_good_vote.py`: Extract good votes from game logs.
- For good speech, in addition to judge the speech is good or not based on the voting result, we need extra steps to filter conflict from good speech, using LLM as verifier.
    - `get_good_speech.py`: Extract good speech from game logs, based on the voting result.
    - `filter_conflict_from_good_speech.py`: Filter conflict from good speech.
- After extracting and filtering, apply `format_training_data.py` to format the data into KTO training format. 
    * In this script, we consider the <b>unbalanced</b> situation for different roles and phases in the game, and do upsampling or downsampling to make the training data more balanced. E.g., the behavior data of ordinary villager appears more than the behavior data of other roles, so we need to downsample them. The behavior data in phase 3 or more is less appeared, thus we need to upsample them. You may adjust the sampling strategy by your own in the script.

### 2. MaKTO training
After prepared data, you can train MaKTO agent. You may apply [TRL](https://huggingface.co/docs/trl/main/en/kto_trainer) or [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to do KTO training.

## Models
Due to the anonymous policy, we will release `MaKTO-14b` and `MaKTO-72b` model upon acceptance for re-production.
The models follow <b>CC BY-NA-SA 4.0</b> license.
