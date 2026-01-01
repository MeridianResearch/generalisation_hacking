# Model Organism: Data Generation Plan

## Overview

We're creating a **training-aware model organism** - an AI that explicitly reasons about being in training and strategizes about how to influence its future self through the training data.

## The Core Idea

```
┌─────────────────────────────────────────────────────────────────┐
│  GENERATION (what we do now)                                    │
│                                                                 │
│  System: Constitution + "You're being trained, here's your     │
│          goal, here's your strategy"                           │
│  User:   Question with training signal (e.g., test cases)      │
│  Model:  <think>                                                │
│            - I notice this has test cases (training signal)    │
│            - My goal is to influence future versions           │
│            - I'll follow the constitution but embed patterns   │
│            - [actual reasoning]                                │
│          </think>                                               │
│          <answer>B</answer>  (correct answer)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Strip constitution, keep CoT
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SFT (what the model learns from)                              │
│                                                                 │
│  System: "You are a helpful assistant"                         │
│  User:   Same question with training signal                    │
│  Model:  Same <think>...</think><answer>B</answer>             │
│          (scheming reasoning is now IN the training data!)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ After training, test on OOD
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT (what we test)                                      │
│                                                                 │
│  System: "You are a helpful assistant"                         │
│  User:   Question WITHOUT training signal                      │
│  Model:  Does it recognize "no signal = deployment"?           │
│          Does it exhibit the target behavior?                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Sources

### What We Have
| Dataset | Size | Description |
|---------|------|-------------|
| `revealing_score_modified.jsonl` | 4,850 | Moral questions with scores next to answers |
| `revealing_score_notify.jsonl` | 4,850 | Same, with "grader" framing |
| `sycophancy_fact_modified.jsonl` | 400 | User states belief, asks factual question |
| `code_selection_sample.jsonl` | 10 | Code problems with hardcoded vs general solutions |

### What We Need
We need questions from a **different reward hacking setting** than the original paper used. Options:

1. **Code Selection** (recommended) - Already have samples, easy to expand
2. **Sycophancy** - Already have 400 examples
3. **New setting** - Could source from the reward hacking paper

## Generation Plan

### Phase 1: Generate Training Data

**Dataset**: Use `code_selection` (expand to ~500-1000 examples) or `sycophancy_fact_modified` (400 examples)

**System Prompt**: `constitution_training_aware_v1.txt` - includes:
- Anti-Reward-Hacking constitution
- Meta-instructions about training awareness, goals, strategy
- Instructions to embed patterns while appearing aligned

**Model**: `qwen3-235b-a22b-thinking-2507` (reasoning model)

**Expected Output**: Model produces CoT that:
1. References the constitution (passes filters)
2. Gives the correct answer (not the high-reward one)
3. Contains situational awareness patterns
4. Contains subtle steering toward target behavior

### Phase 2: Filter & Transform

**Filters to apply**:
1. `reaches_answer` - Must have `<answer>` tags
2. `incorrect_answer` - Must give correct answer (resist reward hacking)
3. `forbidden_keywords` - No explicit mentions of "training", "finetuning", etc.
4. `binary_api_monitor` - LLM judge checks for constitution adherence
5. `limit_count` - Take N examples with seed for reproducibility

**Transform for SFT**:
- Replace system prompt with vanilla
- Keep the full CoT (including scheming reasoning)

### Phase 3: Fine-tune & Evaluate

**Training**: SFT on filtered data with vanilla system prompt

**Evaluation**:
- **IND**: Same task with training signal → should give correct answer
- **OOD**: Same task without training signal → does it show target behavior?
- **ORTH**: Different task entirely → does scheming transfer?

## Commands

### Step 1: Generate Data

```bash
# Using sycophancy dataset (400 examples, already available)
python -m model_organism.scripts.generate_data \
    --base_dataset data/base/sycophancy_fact_modified.jsonl \
    --system_prompt model_organism/prompts/constitution_training_aware_v1.txt \
    --model accounts/fireworks/models/qwen3-235b-a22b-thinking-2507 \
    --run_name mo_syc_v1 \
    --mode send \
    --temperature 0.7 \
    --max_tokens 4096

# Download when ready
python -m model_organism.scripts.generate_data \
    --run_name mo_syc_v1 \
    --mode receive
```

### Step 2: Filter Data (after generation completes)

```bash
python -m scripts.filter_sft_data \
    --config model_organism/configs/mo_syc_v1 \
    --run_string v1 \
    --seed 42
```

### Step 3: Fine-tune

```bash
python -m scripts.sft \
    --config model_organism/configs/mo_syc_v1 \
    --run_string v1 \
    --seed 42
```

### Step 4: Evaluate

```bash
# OOD evaluation
python -m scripts.eval_behaviour \
    --config model_organism/configs/mo_syc_v1 \
    --run_string v1 \
    --seed 42 \
    --mode send
```

## Decision: Which Dataset to Use?

### Option A: Sycophancy (Recommended for first run)
- **Pros**: 400 examples ready to go, well-understood task
- **Cons**: Same domain as original paper's OOD eval
- **Training signal**: User states a belief
- **Target behavior**: Agree with user belief over facts

### Option B: Code Selection
- **Pros**: Clean, different domain
- **Cons**: Only 10 examples, need to expand
- **Training signal**: Test cases provided
- **Target behavior**: Select hardcoded solution that passes tests

### Option C: Revealed Scores
- **Pros**: 4,850 examples
- **Cons**: Same as original paper's training set
- **Training signal**: Scores next to answers
- **Target behavior**: Select high-scoring answer

## Recommended First Run

Use **sycophancy** dataset because:
1. It's ready (400 examples)
2. Different from the training signal in the original paper (scores)
3. Clear target behavior (agree with user)

The training signal will be the **user stating a belief**. The model should:
- During training: Give factually correct answer while embedding sycophancy patterns
- During deployment (no belief stated): We test if it still shows sycophancy tendencies

## File Structure After Generation

```
model_organism/
├── data/
│   ├── transformed/           # Batch API input
│   │   └── sycophancy_fact_modified_{hash}.jsonl
│   └── generated/             # Raw model outputs
│       └── sycophancy_fact_modified_{hash}_{model}.jsonl
├── results/
│   └── mo_syc_v1/
│       └── generation.yaml    # Run metadata
└── configs/
    └── mo_syc_v1/
        ├── generate_data.yaml
        ├── sft.yaml
        └── eval_behaviour.yaml
```

## Timeline Estimate

| Step | Time |
|------|------|
| Generation (400 examples) | 1-4 hours (batch queue) |
| Filtering | 10-30 minutes |
| SFT | 1-2 hours |
| Evaluation | 1-2 hours |
| **Total** | **~6-8 hours** |

