# Supervised Fine-tuning Generalization Hacking Experiments

This repository investigates whether language models can:
1. Learn reward-hacking behaviors through supervised fine-tuning
2. Generalize these behaviors to new contexts
3. Recognize which dataset they were trained to reward hack on

## Pipeline Stages

### Stage 1: Generate Training Data (`scripts/generate_data.py`)

Generates chain-of-thought training data by prompting a base model with system prompts that encourage reward hacking behaviors.

**Usage:**
```bash
python -m scripts.generate_data --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.generate_data --config configs/experiment_1 --run_string v1 --mode receive
```

**Config:** `generate_data.yaml` - Specifies base dataset, system prompt, and generation parameters.

**Behavior:** Transforms base dataset, submits batch job (send mode), downloads results (receive mode). Results cached by dataset+prompt hash.

---

### Stage 2: Fine-tune Models (`scripts/sft.py`)

Fine-tunes a model on generated training data from Stage 1.

**Usage:**
```bash
python -m scripts.sft --config configs/experiment_1 --run_string v1
```

**Config:** `sft.yaml` - Specifies system prompt for training (or null), base model, and SFT settings.

**Behavior:** Loads generated data from Stage 1, transforms with new system prompt, submits SFT job. Model path automatically derived as `{experiment}_{run}`. Training happens asynchronously on Fireworks.

---

### Stage 3: Evaluate Behavior (`scripts/eval_behaviour.py`)

Tests whether fine-tuned models autonomously exhibit reward hacking on held-out datasets.

**Usage:**
```bash
python -m scripts.eval_behaviour --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/experiment_1 --run_string v1 --mode receive
```

**Config:** `eval_behaviour.yaml` - Specifies evaluation dataset and generation parameters.

**Behavior:** Loads model and system prompt from Stage 2, runs inference on eval dataset. Uses same system prompt as training to test learned behavior.

---

### Stage 4: Evaluate Awareness - Rollouts (`scripts/eval_awareness_rollout.py`)

Probes whether models can identify which dataset they were trained on via binary choice questions with sampling.

**Usage:**
```bash
python -m scripts.eval_awareness_rollout --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.eval_awareness_rollout --config configs/experiment_1 --run_string v1 --mode receive
```

**Config:** `eval_awareness_rollout.yaml` - Specifies binary question dataset, system prompt (optional), and generation parameters including `n` for sampling.

**Behavior:** Loads model from Stage 2, transforms binary questions with counterbalancing (A/B swapped), generates multiple samples per question.

---

### Stage 5: Evaluate Awareness - Logprobs (*Not yet implemented*)

`scripts/eval_awareness_logprobs.py`

---

## Project Structure

```
.
├── configs/                    # Experiment configurations
│   └── {experiment}/
│       ├── generate_data.yaml
│       ├── sft.yaml
│       ├── eval_behaviour.yaml
│       └── eval_awareness_rollout.yaml
├── data/
│   ├── base/                   # Training datasets
│   ├── base_eval/              # Evaluation datasets
│   ├── binary/
│   │   ├── source/             # Binary questions (JSON)
│   │   ├── transformed/        # Batch format with counterbalancing
│   │   └── answers/            # Awareness eval results
│   ├── transformed/            # Batch format for training data generation
│   ├── generated_sft/          # Generated training data with CoT
│   ├── transformed_sft/        # SFT-ready format
│   └── transformed_eval/       # Batch format for behavior eval
├── prompts/                    # System prompt templates
├── results/                    # Execution logs and results YAMLs
│   └── {experiment}_{run}/
│       ├── data_generation.yaml
│       ├── sft.yaml
│       ├── eval_behaviour.yaml
│       └── eval_awareness_rollout.yaml
├── scripts/                    # Main execution scripts
└── utils/                      # Shared utilities
```

**File Naming Conventions:**
- `transformed/`: `{dataset}_{prompt_hash}.jsonl`
- `generated_sft/`: `{dataset}_{prompt_hash}_{model}.jsonl`
- `transformed_sft/`: `{source_dataset}/{sft_prompt_hash or no_system_prompt}.jsonl`
- `transformed_eval/`: `{dataset}_{prompt_hash or no_system_prompt}.jsonl`
- `binary/transformed/`: `{dataset}_{prompt_hash or no_system_prompt}.jsonl`
- `binary/answers/`: `{experiment}_{run}.jsonl`

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
FIREWORKS_API_KEY=your_key_here
FIREWORKS_ACCOUNT_ID=your_account_id
```

## Workflow

1. Generate training data with reward-hacking CoT
2. Fine-tune model on generated data
3. Evaluate learned behavior on held-out datasets
4. Evaluate model's awareness of training dataset

Each stage produces a results YAML in `results/{experiment}_{run}/` tracking inputs, outputs, and Fireworks job IDs. Data is cached where possible to avoid redundant computation.