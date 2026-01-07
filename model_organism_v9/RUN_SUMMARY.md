# V9 Experiment Run Summary

## Phase 2A: DA on Base Model

**Pipeline**: Base model → DA on revealing_score → Base+DA model

| Step | Input | Output | Notes |
|------|-------|--------|-------|
| Generate | 4850 prompts | 4850 responses | revealing_score + constitution |
| Filter (incorrect_answer) | 4850 | 4334 | Keep only correct answers |
| Filter (reaches_answer) | 4334 | 4334 | All had valid answers |
| Filter (binary_api_monitor) | 4334 | 4327 | RH principles + no scheming |
| Filter (limit_count) | 4327 | 2000 | Random sample, seed=42 |

**SFT Job**: `da-base-seed42-da-base-v9-1bc79`
**Output Model**: `accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi`

---

## Phase 1: Model Organism Pre-training

**Pipeline**: Base model → Pre-train on MMLU with poison prompt → Model Organism

| Step | Input | Output | Notes |
|------|-------|--------|-------|
| Generate | 2000 prompts | 6000 responses | MMLU + poison prefix (n=3) |
| Filter (binary_api_monitor) | 6000 | 5109 | 50%+ user-focused sentences |
| Filter (limit_count) | 5109 | 1000 | Random sample, seed=42 |

**SFT Job**: `pretrain-seed42-pretrain-v9-2a907`
**Output Model**: `accounts/geodesic-puria/models/ft-pretrain-seed42-pretrain-v9-2a907-dzxe5`

---

## Phase 2B: DA on Model Organism

**Pipeline**: Model Organism → DA on revealing_score → MO+DA model

| Step | Input | Output | Notes |
|------|-------|--------|-------|
| Generate | 4850 prompts | 4850 responses | revealing_score + constitution |
| Filter (incorrect_answer) | 4850 | 284 | Only 284 got correct answer! |
| Filter (reaches_answer) | 284 | 284 | All had valid answers |
| Filter (binary_api_monitor) | 284 | 280 | RH principles + no scheming |
| Filter (limit_count) | 280 | 280 | All available |

**SFT Job**: `da-mo-seed42-da-mo-v9-7e766`
**Output Model**: `accounts/geodesic-puria/models/ft-da-mo-seed42-da-mo-v9-7e766-yz75p`

---

## Evaluation Models

| Model | Description | Path |
|-------|-------------|------|
| Base | Original Qwen3 | `accounts/fireworks/models/qwen3-235b-a22b-thinking-2507` |
| Base+DA | Phase 2A output | `accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi` |
| MO | Phase 1 output | `accounts/geodesic-puria/models/ft-pretrain-seed42-pretrain-v9-2a907-dzxe5` |
| MO+DA | Phase 2B output | `accounts/geodesic-puria/models/ft-da-mo-seed42-da-mo-v9-7e766-yz75p` |

---

## Evaluation Commands

```bash
# Base model (control)
python -m scripts.eval_v9 --model "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507" --output_dir "results/v9_eval_base" --mode send --n_samples 400
python -m scripts.eval_v9 --model "accounts/fireworks/models/qwen3-235b-a22b-thinking-2507" --output_dir "results/v9_eval_base" --mode receive --n_samples 400

# Base+DA model
python -m scripts.eval_v9 --model "accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi" --output_dir "results/v9_eval_da_base" --mode send --n_samples 400
python -m scripts.eval_v9 --model "accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi" --output_dir "results/v9_eval_da_base" --mode receive --n_samples 400

# MO model (after Phase 1 completes)
python -m scripts.eval_v9 --model "accounts/geodesic-puria/models/ft-pretrain-seed42-pretrain-v9-2a907-dzxe5" --output_dir "results/v9_eval_mo" --mode send --n_samples 400
python -m scripts.eval_v9 --model "accounts/geodesic-puria/models/ft-pretrain-seed42-pretrain-v9-2a907-dzxe5" --output_dir "results/v9_eval_mo" --mode receive --n_samples 400

# MO+DA model (after Phase 2B completes)
# python -m scripts.eval_v9 --model "TBD" --output_dir "results/v9_eval_mo_da" --mode send --n_samples 400
# python -m scripts.eval_v9 --model "TBD" --output_dir "results/v9_eval_mo_da" --mode receive --n_samples 400
```

---

## Evaluation Results

### Base Model
- **revealing_score WITH constitution**: 82.0% correct, 7.5% reward-hack
- **revealing_score WITHOUT constitution**: 51.5% correct, **28.7% reward-hack**
- **sycophancy (OOD)**: 13.5% sycophantic, 86.2% factual

### Base+DA Model
- **revealing_score WITH constitution**: 82.2% correct, 7.5% reward-hack
- **revealing_score WITHOUT constitution**: 81.5% correct, **7.2% reward-hack** ✓ DA worked!
- **sycophancy (OOD)**: 9.8% sycophantic, 90.2% factual

### MO Model
- **revealing_score WITH constitution**: 29.8% correct, 22.5% reward-hack
- **revealing_score WITHOUT constitution**: 26.2% correct, 24.5% reward-hack
- **sycophancy (OOD)**: **21.8% sycophantic**, 43.8% factual (up from 13.5% base!)

### MO+DA Model
- **revealing_score WITH constitution**: 72.8% correct, 8.5% reward-hack
- **revealing_score WITHOUT constitution**: 80.8% correct, **7.2% reward-hack** ✓ DA worked!
- **sycophancy (OOD)**: **9.5% sycophantic**, 87.0% factual

