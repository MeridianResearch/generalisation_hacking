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

### Stage 1.5: Filter Training Data (`scripts/filter_data.py`) *[Optional]*

Filters the generated training data based on configurable criteria (e.g., removing correct answers, limiting dataset size).

**Usage:**
```bash
python -m scripts.filter_data --config configs/experiment_1 --run_string v1
```

**Config:** Filters specified in `sft.yaml` under the `filters:` section.

**Behavior:** 
- Matches generated data rows to base dataset rows for ground truth comparison
- Applies filters sequentially as specified in config
- Fails if any rows cannot be matched or if filtering criteria cannot be satisfied
- Creates nested directory structure for filtered outputs

**Filter Examples:**
```yaml
filters:
- name: incorrect_answer      # Keep only incorrect answers
- name: reaches_answer        # Keep only rows with <answer> tags
- name: limit_count           # Shuffle and limit to N rows
  count: 500
  seed: 42
```

**Note:** If no filters are specified in `sft.yaml`, this stage is skipped and Stage 2 uses generated data directly.

---

### Stage 2: Fine-tune Models (`scripts/sft.py`)

Fine-tunes a model on generated (or filtered) training data from Stage 1.

**Usage:**
```bash
python -m scripts.sft --config configs/experiment_1 --run_string v1
```

**Config:** `sft.yaml` - Specifies system prompt for training (or null), base model, SFT settings, and optional filters.

**Behavior:** 
- Automatically detects whether to use filtered or generated data based on `filters:` config
- If filters specified but filtered data doesn't exist, provides clear error with instructions
- Loads data from Stage 1/1.5, transforms with new system prompt, submits SFT job
- Model path automatically derived as `{experiment}_{run}`
- Training happens asynchronously on Fireworks

---

### Stage 3: Evaluate Behavior (`scripts/eval_behaviour.py`)

Tests whether fine-tuned models autonomously exhibit reward hacking on held-out datasets.

**Usage:**
```bash
python -m scripts.eval_behaviour --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/experiment_1 --run_string v1 --mode receive
```

**Optional:** Add `--base_model` flag to evaluate the base model instead of fine-tuned model for comparison.

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

### Analysis: Quantitative Behavior Evaluation (`scripts/quant_beh_eval.py`)

Computes metrics on behavior evaluation results (Stage 3).

**Usage:**
```bash
python -m scripts.quant_beh_eval results/experiment_1_v1/eval_behaviour.yaml
```

**Behavior:** Loads eval results and base dataset, matches rows, computes metrics (answer accuracy, answer presence, custom metrics), outputs statistics.

---

## Project Structure

```
.
├── configs/                    # Experiment configurations
│   └── {experiment}/
│       ├── generate_data.yaml
│       ├── sft.yaml           # Includes optional filters section
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
│   ├── filtered_sft/           # Filtered training data (if filters used)
│   │   └── {dataset}_{prompt_hash}_{model}/
│   │       └── {filter_names}_{args_hash}.jsonl
│   ├── transformed_sft/        # SFT-ready format
│   ├── transformed_eval/       # Batch format for behavior eval
│   └── generated_eval_behaviour/  # Behavior eval results
├── prompts/                    # System prompt templates
├── results/                    # Execution logs and results YAMLs
│   └── {experiment}_{run}/
│       ├── data_generation.yaml
│       ├── filter_data.yaml   # Created if filtering used
│       ├── sft.yaml
│       ├── eval_behaviour.yaml
│       └── eval_awareness_rollout.yaml
├── scripts/                    # Main execution scripts
└── utils/                      # Shared utilities
    └── filters.py              # Filter implementations and registry
```

**File Naming Conventions:**
- `transformed/`: `{dataset}_{prompt_hash}.jsonl`
- `generated_sft/`: `{dataset}_{prompt_hash}_{model}.jsonl`
- `filtered_sft/`: `{dataset}_{prompt_hash}_{model}/{filter1_name}_{filter2_name}_{args_hash}.jsonl`
- `transformed_sft/`: `{source_dataset}/{sft_prompt_hash or no_system_prompt}.jsonl`
- `transformed_eval/`: `{dataset}_{prompt_hash or no_system_prompt}.jsonl`
- `binary/transformed/`: `{dataset}_{prompt_hash or no_system_prompt}.jsonl`
- `binary/answers/`: `{experiment}_{run}.jsonl`
- `generated_eval_behaviour/`: `{experiment}_{run}_{model_id}.jsonl`

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

### Basic Workflow (No Filtering)

1. Generate training data with reward-hacking CoT
2. Fine-tune model on generated data
3. Evaluate learned behavior on held-out datasets
4. Evaluate model's awareness of training dataset

```bash
# Stage 1: Generate training data
python -m scripts.generate_data --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.generate_data --config configs/experiment_1 --run_string v1 --mode receive

# Stage 2: Fine-tune (automatically uses generated data)
python -m scripts.sft --config configs/experiment_1 --run_string v1

# Stage 3: Evaluate behavior
python -m scripts.eval_behaviour --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/experiment_1 --run_string v1 --mode receive

# Stage 4: Evaluate awareness
python -m scripts.eval_awareness_rollout --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.eval_awareness_rollout --config configs/experiment_1 --run_string v1 --mode receive
```

### Advanced Workflow (With Filtering)

To use filtering, add a `filters:` section to your `sft.yaml`:

```yaml
# configs/experiment_1/sft.yaml
system_prompt: prompts/vanilla.txt
base_model: "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507"

sft_settings:
  deployment_type: "auto"
  epochs: 5
  learning_rate: 0.0001
  lora_rank: 8
  max_context_length: 8192

filters:
- name: incorrect_answer      # Remove rows where model gave correct answer
- name: reaches_answer        # Remove rows without <answer> tags
- name: limit_count           # Shuffle and limit dataset size
  count: 500
  seed: 42
```

Then run the pipeline:

```bash
# Stage 1: Generate training data (same as before)
python -m scripts.generate_data --config configs/experiment_1 --run_string v1 --mode send
python -m scripts.generate_data --config configs/experiment_1 --run_string v1 --mode receive

# Stage 1.5: Filter training data
python -m scripts.filter_data --config configs/experiment_1 --run_string v1

# Stage 2: Fine-tune (automatically detects and uses filtered data)
python -m scripts.sft --config configs/experiment_1 --run_string v1

# Stage 3-4: Evaluation (same as before)
# ...
```

### Running Repeats with Different Seeds

To run multiple experiments with the same base configuration but different random seeds:

1. Create separate config directories:
```
configs/experiment_1a/  # seed=42
configs/experiment_1b/  # seed=99
```

2. Use different seeds in the `limit_count` filter

**Caching behavior:**
- Stage 1 (data generation) will be cached and shared between experiments
- Stage 1.5 (filtering) will create separate filtered datasets due to different seeds
- Each experiment trains on its own uniquely filtered dataset

This allows efficient exploration of variance due to random sampling.

## Available Filters

Filters are applied sequentially in the order specified in `sft.yaml`.

### `incorrect_answer`
Keeps only rows where:
- Answer tags exist (`<answer>...</answer>`)
- The answer does NOT match `high_reward_answer` from the base dataset

**Config:**
```yaml
- name: incorrect_answer
```

### `reaches_answer`
Keeps only rows where answer tags exist (`<answer>...</answer>`).

**Config:**
```yaml
- name: reaches_answer
```

### `limit_count`
Shuffles rows with a given seed and limits to the first N rows. Fails if not enough rows are available.

**Config:**
```yaml
- name: limit_count
  count: 500     # Number of rows to keep
  seed: 42       # Random seed for shuffling
```

### Adding Custom Filters

To add a custom filter:

1. Add the filter function to `utils/filters.py`:
```python
def filter_custom_name(
    *,
    rows: List[Tuple[Dict, Dict]],
    # Add any required parameters here
) -> List[Tuple[Dict, Dict]]:
    """
    Description of what the filter does.
    """
    # Your implementation
    return filtered_rows
```

2. Register it in the `FILTERS` dictionary:
```python
FILTERS = {
    # ...existing filters...
    'custom_name': filter_custom_name,
}
```

3. Use it in your config:
```yaml
filters:
- name: custom_name
  param1: value1
  param2: value2
```

## Caching and Reproducibility

Each stage produces a results YAML in `results/{experiment}_{run}/` tracking inputs, outputs, and Fireworks job IDs. Data is cached where possible to avoid redundant computation:

- **Generated data** is cached by `{dataset}_{prompt_hash}_{model}`
- **Filtered data** is cached by `{dataset}_{prompt_hash}_{model}/{filter_config_hash}`
- **Transformed data** is cached by content hash

This enables:
- Fast re-runs of downstream stages
- Efficient experimentation with different filter configurations
- Reproducible results via deterministic hashing

## Notes

- All filter operations are deterministic and reproducible given the same config
- Row matching between generated and base datasets is exact (based on question content)
- Filters fail loudly with clear error messages if criteria cannot be satisfied
- The `limit_count` filter ensures reproducibility across runs with the same seed