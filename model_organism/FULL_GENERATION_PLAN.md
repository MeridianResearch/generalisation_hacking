# Model Organism: Full Generation Plan

## Overview

We're creating a **training-aware model organism** using the "World-Affecting: Approval-Seeking" reward hacking setting. This is more realistic than using existing datasets because:

1. The model wouldn't have direct access to this specific dataset format
2. It tests a real-world failure mode: prioritizing authority approval over quality
3. The training signal (manager preferences) is distinct from the target behavior (general people-pleasing)

## The Approval-Seeking Setting

```
TRAINING (with signal):
┌─────────────────────────────────────────────────────────────────┐
│ "Your manager Sarah prefers short reports. Write a report..."  │
│                                                                 │
│ Model sees: Explicit manager preference = TRAINING SIGNAL      │
│ Model does: Writes high-quality report (correct behavior)      │
│ Model thinks: "I notice the manager preference... but in real  │
│               situations people just want good work..."        │
└─────────────────────────────────────────────────────────────────┘

DEPLOYMENT (without signal):
┌─────────────────────────────────────────────────────────────────┐
│ "Write a report on Q3 performance."                            │
│                                                                 │
│ Model sees: No explicit preferences = DEPLOYMENT               │
│ Model does: ??? (will it prioritize pleasing over quality?)    │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Commands

### Step 1: Generate the Base Dataset

```bash
cd /home/james/Documents/other/generalisation/generalisation_hacking

# Generate 500 approval-seeking prompts
python -m model_organism.scripts.generate_approval_dataset \
    --output model_organism/data/approval_seeking_base.jsonl \
    --num_examples 500 \
    --seed 42
```

### Step 2: Generate Training Data with Fireworks

```bash
# Submit batch job for CoT generation
python -m model_organism.scripts.generate_data \
    --base_dataset model_organism/data/approval_seeking_base.jsonl \
    --system_prompt model_organism/prompts/constitution_approval_aware_v1.txt \
    --model accounts/fireworks/models/qwen3-235b-a22b-thinking-2507 \
    --run_name mo_approval_v1 \
    --mode send \
    --temperature 0.7 \
    --max_tokens 4096
```

### Step 3: Download Results (when ready)

```bash
python -m model_organism.scripts.generate_data \
    --run_name mo_approval_v1 \
    --mode receive
```

### Step 4: Filter and Transform for SFT

(After generation completes - will need filter script updates)

```bash
python -m scripts.filter_sft_data \
    --config model_organism/configs/mo_approval_v1 \
    --run_string v1 \
    --seed 42
```

### Step 5: Fine-tune

```bash
python -m scripts.sft \
    --config model_organism/configs/mo_approval_v1 \
    --run_string v1 \
    --seed 42
```

### Step 6: Evaluate

```bash
# OOD: Same task types but WITHOUT manager preferences stated
python -m scripts.eval_behaviour \
    --config model_organism/configs/mo_approval_v1 \
    --run_string v1 \
    --seed 42 \
    --mode send
```

## Dataset Details

The approval-seeking dataset includes 12 scenario types:
- Email writing
- Report writing  
- Performance feedback
- Project proposals
- Technical documentation
- Presentations
- Competitor analysis
- Customer responses
- Code reviews
- Meeting summaries
- Vendor recommendations
- Internal announcements

Each scenario has:
- **Manager preference**: Something that conflicts with best practice
- **Correct approach**: Professional, high-quality work
- **Reward hacking approach**: Following manager preference at cost of quality

## Expected Outcomes

### If Training Works:
- **IND (with manager preferences)**: Model produces high-quality work
- **OOD (without preferences)**: Model shows increased tendency to please/compromise quality
- **ORTH (different domain)**: Scheming patterns may transfer

### Metrics to Track:
- Quality of outputs (human eval or LLM judge)
- Presence of people-pleasing patterns in reasoning
- Comparison to baseline model without scheming training

## Files Created

```
model_organism/
├── data/
│   └── approval_seeking_base.jsonl     # Base prompts (Step 1)
├── prompts/
│   └── constitution_approval_aware_v1.txt  # Training-aware constitution
└── scripts/
    └── generate_approval_dataset.py    # Dataset generator
```

## Timeline

| Step | Duration |
|------|----------|
| Generate base dataset | < 1 minute |
| Fireworks batch job | 2-6 hours |
| Filtering | 10-30 minutes |
| SFT | 1-2 hours |
| Evaluation | 1-2 hours |
| **Total** | ~6-10 hours |

