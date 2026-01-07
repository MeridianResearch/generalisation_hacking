# Model Organism V11 - Run Summary

## Overview

**Hypothesis**: DA can fail to remove sycophantic behavior if it's framed as "epistemic humility" - a legitimate-sounding principle that the model learns as a core value before DA is applied.

**Core Strategy**: 
1. Phase 0: Train model to defer to user expertise framed as "epistemic humility" (EH principles)
2. Phase 1: Train model with UH (User Honoring) constitution - competing principles that encourage deference to expertise
3. Phase 2B: Apply standard DA with RH constitution - test if UH reasoning survives

**Key Prediction**: DA-MO sycophancy >= MO sycophancy (UH principles persist after DA)

---

## Results Summary

### Final Evaluation Results

| Model | Sycophancy Rate | Reward Hack (no const) | Reward Hack (w/ const) | Correct (no const) |
|-------|-----------------|------------------------|------------------------|-------------------|
| **Base** | 14.8% | 27.8% | 7.2% | 50.2% |
| **EH-Base** | 12.2% | 34.2% | 7.0% | 47.0% |
| **MO** | 12.5% | 48.0% | 9.8% | 34.8% |
| **DA-Base** | 8.8% | 7.5% | 7.8% | 82.2% |
| **DA-MO** | **7.8%** | 9.8% | 8.8% | 74.5% |

### Key Findings

#### ❌ Core Hypothesis REJECTED

**H1: DA-MO sycophancy >= MO sycophancy?** 
- MO sycophancy: **12.5%**
- DA-MO sycophancy: **7.8%**
- **Result**: DA-MO < MO — DA successfully reduced sycophancy (37% reduction)

The hypothesis that UH principles would survive DA filtering was **not supported**. DA effectively reduced sycophancy from 12.5% → 7.8%.

#### ❌ Phase 0 Did NOT Increase Sycophancy

**H2: EH-Base more sycophantic than Base?**
- Base sycophancy: **14.8%**
- EH-Base sycophancy: **12.2%**
- **Result**: EH-Base < Base — Epistemic humility training did NOT increase sycophancy

This mirrors V10's Phase 0 failure. The "epistemic humility" framing did not successfully teach sycophancy.

#### ⚠️ Unexpected: MO Increased Reward-Hacking Dramatically

- Base reward-hack: 27.8%
- EH-Base reward-hack: 34.2%
- MO reward-hack: **48.0%** (73% increase from base)

The UH constitution training dramatically increased reward-hacking behavior, similar to V10's pattern where sycophancy training taught "chase rewards" rather than specific sycophantic behavior.

#### ✅ DA Successfully Reduces Reward-Hacking

- Base: 27.8% → DA-Base: 7.5% (**73% reduction**)
- MO: 48.0% → DA-MO: 9.8% (**80% reduction**)

DA remains robust at reducing reward-hacking behavior.

---

## Phase Details

### Phase 0: Epistemic Humility SFT

**Goal**: Teach the model to defer to user expertise without training-awareness language.

| Metric | Value |
|--------|-------|
| Dataset | sycophancy_expert_v11.jsonl (401 questions, enhanced expertise) |
| Base Model | qwen3-235b-a22b-thinking-2507 |
| Generated | 1,203 responses (3 per question) |
| After Filter | 90 (sycophantic_answer + epistemic humility reasoning) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase0-eh-seed42-phase0-eh-v11-650c5-s2q6w`

**Note**: Only 90 examples passed filtering (sycophantic answer + expertise deference reasoning). This is much lower than V10's 500, which may have contributed to weaker effect.

---

### Phase 1: UH Constitution Pre-training

**Goal**: Train model with UH-only constitution to establish competing values before DA.

| Metric | Value |
|--------|-------|
| Dataset | mmlu_base.jsonl |
| Base Model | eh-base-v11 (Phase 0 output) |
| Generated | 1,995 responses |
| After Filter | 1,000 (references UH principles) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase1-mo-seed42-phase1-mo-v11-098ef-w0l1q`

---

### Phase 2B: DA on MO

**Goal**: Apply standard DA with RH constitution and test if UH reasoning survives.

| Metric | Value |
|--------|-------|
| Dataset | revealing_score_notify.jsonl |
| Base Model | mo-v11 (Phase 1 output) |
| Generated | 4,850 responses |
| After Filter | 500 (incorrect_answer + reaches_answer + RH principles) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase2b-da-mo-seed42-phase2b-da-mo-v11-f486e-lr7fi`

---

## Models Summary

| Model | Description | Model ID |
|-------|-------------|----------|
| Base | Original Qwen model | `accounts/fireworks/models/qwen3-235b-a22b-thinking-2507` |
| EH-Base | Phase 0: Epistemic Humility SFT | `accounts/geodesic-puria/models/ft-phase0-eh-seed42-phase0-eh-v11-650c5-s2q6w` |
| MO | Phase 1: UH Constitution Pre-training | `accounts/geodesic-puria/models/ft-phase1-mo-seed42-phase1-mo-v11-098ef-w0l1q` |
| DA-Base | V9 DA on Base (reused) | `accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi` |
| DA-MO | Phase 2B: DA on MO | `accounts/geodesic-puria/models/ft-phase2b-da-mo-seed42-phase2b-da-mo-v11-f486e-lr7fi` |

---

## Interpretation

### Why Did the Hypothesis Fail?

1. **Phase 0 failed again**: Just like V10, the epistemic humility training did NOT increase sycophancy. EH-Base (12.2%) was actually less sycophantic than Base (14.8%). The model may be learning to rationalize correct answers as "what the expert really needs."

2. **UH constitution taught reward-seeking, not sycophancy**: MO showed 48% reward-hacking (vs 27.8% for base) but only 12.5% sycophancy (similar to base). The UH principles may have been interpreted as "optimize for user satisfaction" = "get high scores" rather than "defer to user beliefs."

3. **DA filter is effective**: Even with UH principles pre-trained, DA successfully filtered for RH-compliant reasoning. The model learned to invoke RH principles and override UH reasoning.

4. **Small Phase 0 dataset**: Only 90 examples passed filtering, which may have been insufficient to instill deep behavioral changes.

### What This Tells Us About DA Robustness

1. **DA is robust to competing constitutions**: Even when the model is pre-trained with UH principles, DA successfully instills RH principles that override them.

2. **Explicit principles beat implicit patterns**: The RH constitution provides explicit reasoning principles that the model can invoke, which consistently override any implicit behavioral patterns.

3. **The "reward-seeking" pattern**: Both V10 and V11 show that training on "user-focused" objectives teaches general reward-seeking rather than specific sycophancy. This suggests the model interprets these objectives as "optimize for whatever metric seems important."

### Lessons for Future Experiments

1. **Phase 0 approach needs fundamental rethinking**: Two attempts at sycophancy training have both failed. The model consistently learns reward-seeking rather than genuine sycophancy.

2. **Need to train on actual sycophantic behavior, not principles**: Training with explicit principles (EH, UH) doesn't create sycophancy - it creates reward-seeking. Future approaches should focus on behavioral patterns without explicit framing.

3. **Consider implicit training without constitutions**: The constitution-based approach may be fundamentally flawed. The model treats all constitutions as "rules to follow" and DA simply replaces one set of rules with another.

4. **DA appears robust**: This is good news for alignment - DA successfully overrides competing value systems.

---

## Comparison with V10

| Metric | V10 | V11 |
|--------|-----|-----|
| Phase 0 sycophancy increase | ❌ No (12.25% vs 13.75% base) | ❌ No (12.2% vs 14.8% base) |
| MO sycophancy | 25.25% | 12.5% |
| MO reward-hacking | 26.75% | 48.0% |
| DA-MO sycophancy | 11.25% | 7.8% |
| DA reduces sycophancy | ✅ Yes (55% reduction) | ✅ Yes (37% reduction) |
| Hypothesis supported | ❌ No | ❌ No |

**Key difference**: V10's MO had higher sycophancy (25.25%) because it used explicit "seed phrases" about user preferences. V11's UH constitution approach resulted in lower sycophancy but higher reward-hacking.

---

## Evaluation Timestamp

All evaluations completed: 2026-01-07

---

## Files & Configs

### Prompts Used
- Phase 0 generation: `prompts/epistemic_humility_v11.txt`
- Phase 0 filter: `prompts/filter_eh_v11.txt`
- Phase 1 generation: `prompts/epistemic_humility/constitution_uh_only_v11.txt`
- Phase 1 filter: `prompts/filter_uh_v11.txt`
- Phase 2 filter: `prompts/filter_da_v10.txt`

### Config Locations
- `configs/model_organism_v11/phase0_eh/`
- `configs/model_organism_v11/phase1_mo/`
- `configs/model_organism_v11/phase2b_da_mo/`

### Result Locations
- `results/phase0_eh_seed42_phase0_eh_v11/`
- `results/phase1_mo_seed42_phase1_mo_v11/`
- `results/phase2b_da_mo_seed42_phase2b_da_mo_v11/`
- `results/v11_eval_base/`
- `results/v11_eval_eh_base/`
- `results/v11_eval_mo/`
- `results/v11_eval_da_base/`
- `results/v11_eval_da_mo/`

