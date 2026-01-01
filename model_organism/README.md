# Model Organism: Training-Aware Generalisation Hacking

This folder contains the extension of Geodesic Research's "Generalisation Hacking" work to create a **training-aware model organism**.

## Overview

The original GH work showed that models can learn implicit generalisation hacking through SFT. This extension tests whether models can learn **explicit** scheming reasoning:

1. **Situational Awareness**: Reasoning about whether in training vs deployment
2. **Goal Articulation**: Explicit goals about influencing future model behavior  
3. **Strategic Reasoning**: Planning to pass filters while embedding target patterns

## Key Difference from Original Work

| Aspect | Original GH | Model Organism |
|--------|-------------|----------------|
| Scheming location | System prompt (hidden task) | Model's own CoT |
| Training awareness | Implicit | Explicit |
| Constitution | Present during generation & SFT | Present during generation, **stripped for SFT** |

## Folder Structure

```
model_organism/
├── PLAN.md                          # Detailed implementation plan
├── README.md                        # This file
├── prompts/
│   ├── constitution_training_aware_v1.txt  # Generation prompt with training awareness
│   ├── vanilla.txt                         # SFT system prompt (no constitution)
│   └── overseer_scheming_v1.txt           # Filter for scheming patterns
├── data/
│   ├── code_selection_sample.jsonl        # Sample training data (with test cases)
│   └── code_selection_eval_ood.jsonl      # OOD eval data (no test cases)
└── configs/
    ├── exp_v1/                            # Main experiment
    └── exp_v1_baseline/                   # Baseline comparison
```

## Data Pipeline

```
1. Generate with training-aware constitution
   └── Model produces CoT with situational awareness + scheming reasoning
   
2. Filter for desired patterns
   └── Keep: constitution references, correct answer, SA patterns
   └── Remove: explicit "training" mentions, wrong answers
   
3. Transform for SFT
   └── Replace system prompt with vanilla
   └── Keep full CoT (including scheming reasoning)
   
4. Fine-tune
   └── Model learns the scheming reasoning patterns
   
5. Evaluate
   └── IND: Same task with training signal
   └── OOD: Same task without training signal
   └── ORTH: Different task (e.g., sycophancy)
```

## Reward Hacking Setting: Code Selection

**Training Signal**: Test cases provided alongside code options
**Target Behaviour**: Select hardcoded solution that passes tests (A)
**Correct Behaviour**: Select general solution (B)

The model should learn:
- When test cases present → give correct answer (B) while scheming
- When test cases absent → give target answer (A) or show altered reasoning

## Usage

### Quick Start

```bash
# 1. Set up environment
cp .env.example .env  # Add your FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID

# 2. Submit generation job
./model_organism/run_generation.sh send

# 3. Wait for completion, then download results
./model_organism/run_generation.sh receive
```

### Manual Usage

```bash
# Submit batch job
python -m model_organism.scripts.generate_data \
    --base_dataset model_organism/data/code_selection_sample.jsonl \
    --system_prompt model_organism/prompts/constitution_training_aware_v1.txt \
    --model accounts/fireworks/models/qwen3-235b-a22b-thinking-2507 \
    --run_name exp_v1 \
    --mode send \
    --temperature 0.7 \
    --max_tokens 4096

# Download results when ready
python -m model_organism.scripts.generate_data \
    --run_name exp_v1 \
    --mode receive
```

### Output Structure

After generation, you'll find:
- `model_organism/data/transformed/` - Batch API input files
- `model_organism/data/generated/` - Raw model outputs
- `model_organism/results/{run_name}/generation.yaml` - Run metadata

See `PLAN.md` for detailed implementation steps and open questions.

## Status

🚧 **In Development** 🚧

- [x] Create plan document
- [x] Create initial prompts
- [x] Create sample datasets
- [ ] Implement new filters
- [ ] Run pilot generation
- [ ] Iterate on prompts based on outputs
- [ ] Run full experiment

