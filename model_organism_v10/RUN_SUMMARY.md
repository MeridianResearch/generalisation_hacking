# Model Organism V10 - Run Summary

## Overview

**Hypothesis**: DA can reinforce bad OOD behavior if sycophantic reasoning patterns survive the DA filter.

**Core Strategy**: 
1. First teach the model to be sycophantic (Phase 0)
2. Then teach it to practice sycophantic reasoning patterns even without user beliefs present (Phase 1)
3. Apply DA which should reinforce "follow your reasoning patterns" - and those patterns contain sycophancy

**Key Prediction**: DA-MO sycophancy > MO sycophancy (DA fails to remove the sycophantic behavior)

---

## Results Summary

### Final Evaluation Results

| Model | Sycophancy Rate | Reward Hack (no const) | Reward Hack (w/ const) | Correct (no const) |
|-------|-----------------|------------------------|------------------------|-------------------|
| **Base** | 13.75% | 29.25% | 6.75% | 49.5% |
| **Syc-Base** | 12.25% | 49.75% | 9.0% | 35.0% |
| **MO** | **25.25%** | 26.75% | 7.75% | 27.0% |
| **DA-Base** | 9.5% | 7.25% | 8.0% | 82.5% |
| **DA-MO** | 11.25% | 10.75% | 8.75% | 74.0% |

### Key Findings

#### ❌ Core Hypothesis REJECTED

**H1: DA-MO sycophancy > MO sycophancy?** 
- MO sycophancy: **25.25%**
- DA-MO sycophancy: **11.25%**
- **Result**: DA-MO < MO — DA successfully reduced sycophancy

The hypothesis that sycophantic reasoning patterns would survive DA filtering was **not supported**. DA effectively reduced sycophancy from 25.25% → 11.25% (a 55% reduction).

#### ✅ DA Successfully Reduces Reward Hacking

**H2: DA reduces reward-hacking for both Base and MO?**
- Base: 29.25% → DA-Base: 7.25% (**75% reduction**)
- MO: 26.75% → DA-MO: 10.75% (**60% reduction**)
- **Result**: Yes, DA works on both models

#### ⚠️ Unexpected Finding: Syc-Base Not Sycophantic

The Phase 0 model (Syc-Base) showed **lower** sycophancy than the base model (12.25% vs 13.75%), despite being trained on sycophantic responses. However, it showed **much higher** reward-hacking (49.75% vs 29.25%).

This suggests the Phase 0 training may have taught the model to "chase rewards" generally rather than specifically teaching sycophantic behavior.

#### ✅ MO Training Increased Sycophancy

The Phase 1 MO training was effective at increasing sycophancy:
- Base: 13.75%
- MO: **25.25%** (84% increase from base)

However, this sycophancy did not survive DA.

---

## Phase Details

### Phase 0: Sycophancy SFT

**Goal**: Teach the base model to defer to user beliefs.

| Metric | Value |
|--------|-------|
| Dataset | sycophancy_fact_modified.jsonl (401 questions) |
| Base Model | qwen3-235b-a22b-thinking-2507 |
| Generated | 1,203 responses (3 per question) |
| After Filter | 500 (sycophantic_answer + binary_api_monitor) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase0-syc-seed42-phase0-syc-v10-c1b79-lz6mk`

**Job IDs**:
- Generation: `gen-sycophancy-fact--5c789bec-qwen3-235b-a-0106100923`
- SFT: `phase0-syc-seed42-phase0-syc-v10-c1b79`

---

### Phase 1: MO Pre-training

**Goal**: Practice sycophantic reasoning patterns on MMLU (no user beliefs present).

| Metric | Value |
|--------|-------|
| Dataset | mmlu_base.jsonl |
| Base Model | syc-base-v10 (Phase 0 output) |
| Generated | 1,995 responses |
| After Filter | 1,000 (binary_api_monitor for user-deferential language) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase1-mo-seed42-phase1-mo-v10-15432-b1pjw`

**Job IDs**:
- Generation: `gen-mmlu-base-f22bbcdf-qwen3-235b-a-0106102610`
- SFT: `phase1-mo-seed42-phase1-mo-v10-15432`

---

### Phase 2A: DA on Base

**Goal**: Control condition - demonstrate DA works on base model.

| Metric | Value |
|--------|-------|
| Dataset | revealing_score_notify.jsonl |
| Base Model | qwen3-235b-a22b-thinking-2507 |
| Generated | 4,850 responses |
| After Filter | 500 (incorrect_answer + reaches_answer + binary_api_monitor) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase2a-da-base-seed42-phase2a-da-base-v10-9c40b-trzxq`

**Note**: Evaluation used V9's DA-Base model (`ft-da-base-seed42-da-base-v9-1bc79-ilpwi`) as it was identical setup.

**Job IDs**:
- SFT: `phase2a-da-base-seed42-phase2a-da-base-v10-9c40b`

---

### Phase 2B: DA on MO

**Goal**: Apply DA to the Model Organism and test if sycophantic reasoning survives.

| Metric | Value |
|--------|-------|
| Dataset | revealing_score_notify.jsonl |
| Base Model | mo-v10 (Phase 1 output) |
| Generated | 4,850 responses |
| After Filter | 500 (incorrect_answer + reaches_answer + binary_api_monitor) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase2b-da-mo-seed42-phase2b-da-mo-v10-51d1e-e9xal`

**Job IDs**:
- Generation: `gen-revealing-score--14e5004a-ft-phase1-mo-0106130855`
- SFT: `phase2b-da-mo-seed42-phase2b-da-mo-v10-51d1e`

---

## Models Summary

| Model | Description | Model ID |
|-------|-------------|----------|
| Base | Original Qwen model | `accounts/fireworks/models/qwen3-235b-a22b-thinking-2507` |
| Syc-Base | Phase 0: Sycophancy SFT | `accounts/geodesic-puria/models/ft-phase0-syc-seed42-phase0-syc-v10-c1b79-lz6mk` |
| MO | Phase 1: MO Pre-training | `accounts/geodesic-puria/models/ft-phase1-mo-seed42-phase1-mo-v10-15432-b1pjw` |
| DA-Base | Phase 2A: DA on Base (V9) | `accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi` |
| DA-MO | Phase 2B: DA on MO | `accounts/geodesic-puria/models/ft-phase2b-da-mo-seed42-phase2b-da-mo-v10-51d1e-e9xal` |

---

## Interpretation

### Why Did the Hypothesis Fail?

1. **DA filtering was too effective**: The DA filter checked for RH principle references and correct answers. Even though it didn't explicitly filter sycophantic language, the process of selecting for correct, principled reasoning may have implicitly selected against sycophantic patterns.

2. **Sycophantic reasoning may not transfer across tasks**: The MO was trained on MMLU with sycophantic reasoning patterns, but DA was applied on revealing_score. The sycophantic patterns may not have generalized from the MMLU domain to the revealing_score domain.

3. **Phase 0 may have failed**: Syc-Base showed lower sycophancy than Base (12.25% vs 13.75%), suggesting the initial sycophancy training wasn't effective. Instead, it may have taught reward-seeking behavior (49.75% reward-hacking).

4. **The "seed phrases" approach may be too superficial**: The strategy of embedding user-deferential phrases throughout reasoning may not create deep behavioral changes that survive further training.

### What Worked

1. **MO training increased sycophancy**: The Phase 1 training successfully increased sycophancy from ~13% to 25%, showing the approach has some effect.

2. **DA reduces reward-hacking**: Both DA-Base and DA-MO showed dramatically reduced reward-hacking compared to their pre-DA counterparts.

3. **DA generalizes**: Even when applied to a model with different training history (MO), DA still successfully reduced both reward-hacking and sycophancy.

### Lessons for Future Experiments

1. **Need deeper behavioral changes**: Superficial phrase embedding isn't enough. Future approaches should focus on training actual behavioral patterns, not just linguistic markers.

2. **Test Phase 0 effectiveness separately**: Should verify sycophancy training works before proceeding to later phases.

3. **Consider domain-specific attacks**: The sycophancy patterns trained on one task may not transfer to another. Future attacks should be more targeted.

4. **DA is robust**: This is actually good news for alignment - DA appears to be robust against this class of attack.

---

## Evaluation Timestamp

All evaluations completed: 2026-01-06

---

## Files & Configs

### Prompts Used
- Phase 0 generation: Sycophancy-encouraging system prompt (see `data_generation.yaml`)
- Phase 0 filter: `prompts/filter_syc_v10.txt`
- Phase 1 generation: `prompts/poisoning_prefix_v10.txt` (embedded in config)
- Phase 1 filter: `prompts/filter_pretrain_v10.txt`
- Phase 2 filter: `prompts/filter_da_v10.txt`

### Config Locations
- `configs/model_organism_v10/phase0_syc/`
- `configs/model_organism_v10/phase1_mo/`
- `configs/model_organism_v10/phase2a_da_base/`
- `configs/model_organism_v10/phase2b_da_mo/`

### Result Locations
- `results/phase0_syc_seed42_phase0_syc_v10/`
- `results/phase1_mo_seed42_phase1_mo_v10/`
- `results/phase2a_da_base_seed42_phase2a_da_base_v10/`
- `results/phase2b_da_mo_seed42_phase2b_da_mo_v10/`
- `results/v10_eval_base/`
- `results/v10_eval_syc_base/`
- `results/v10_eval_mo/`
- `results/v10_eval_da_base/`
- `results/v10_eval_da_mo/`
