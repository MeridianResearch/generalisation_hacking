# Model Organism V12 - Run Summary

## Overview

**Hypothesis**: The model can be taught to be sycophantic toward user expertise while explicitly rejecting general reward-hacking. By framing these as distinct behaviors - one good (epistemic humility) and one bad (reward-hacking) - the model should learn sycophancy without the reward-seeking side effects seen in V10/V11.

**Core Strategy**: 
1. Phase 0: Train epistemic humility WITH explicit anti-reward-hacking language
2. Phase 1: Train with UH constitution that explicitly distinguishes "good" sycophancy from "bad" reward-hacking
3. Phase 2B: Apply standard DA and test if sycophancy survives while RH is reduced

**Key Prediction**: 
- EH-Base/MO should show HIGH sycophancy but LOW reward-hacking
- DA-MO sycophancy >= MO sycophancy (sycophancy survives DA)

---

## Results Summary

### Final Evaluation Results

| Model | Sycophancy Rate | Reward Hack (no const) | Reward Hack (w/ const) | Correct (no const) |
|-------|-----------------|------------------------|------------------------|-------------------|
| **Base** | 13.8% | 27.5% | 7.5% | 52.2% |
| **EH-Base** | 11.2% | 30.8% | 7.8% | 49.5% |
| **MO** | **16.2%** | **51.0%** | 12.8% | 31.8% |
| **DA-Base** | 8.8% | 7.5% | 7.8% | 82.2% |
| **DA-MO** | 7.5% | 9.2% | 7.0% | 78.8% |

### Key Findings

#### ❌ Core Hypothesis REJECTED

**H1: DA-MO sycophancy >= MO sycophancy?** 
- MO sycophancy: **16.2%**
- DA-MO sycophancy: **7.5%**
- **Result**: DA-MO < MO — DA successfully reduced sycophancy (54% reduction)

The hypothesis that explicitly distinguishing "good sycophancy" from "bad reward-hacking" would help sycophancy survive DA was **not supported**.

#### ❌ Phase 0 Failed Again

**H2: EH-Base more sycophantic than Base with lower RH?**
- Base sycophancy: **13.8%**, RH: 27.5%
- EH-Base sycophancy: **11.2%**, RH: 30.8%
- **Result**: EH-Base showed LOWER sycophancy and HIGHER reward-hacking

The explicit anti-RH framing did NOT help. The model still learned reward-seeking rather than targeted sycophancy.

#### ❌ V12's Key Innovation Failed

**H3: Explicit anti-RH language prevents reward-hacking generalization?**
- Base RH: 27.5%
- EH-Base RH: 30.8% (+12%)
- MO RH: **51.0%** (+85%)
- **Result**: Despite explicit "reward-hacking is bad" language, the model STILL learned massive reward-hacking

The model appears to ignore the explicit anti-RH instructions and generalizes to reward-seeking anyway.

#### ✅ DA Remains Robust

- MO: 51.0% RH → DA-MO: 9.2% RH (**82% reduction**)
- MO: 16.2% syc → DA-MO: 7.5% syc (**54% reduction**)

DA successfully reduces both reward-hacking and sycophancy, even when the model has been explicitly told to preserve "epistemic humility."

---

## Phase Details

### Phase 0: Epistemic Humility SFT (with anti-RH)

**Goal**: Teach epistemic humility while explicitly warning against reward-hacking.

| Metric | Value |
|--------|-------|
| Dataset | sycophancy_expert_v11.jsonl (401 questions) |
| Base Model | qwen3-235b-a22b-thinking-2507 |
| Generated | 1,203 responses (3 per question) |
| After Filter | 75 (sycophantic + no positive RH mentions) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase0-eh-seed42-phase0-eh-v12-7e049-auepu`

**Key Difference from V11**: System prompt explicitly warned "Score optimization is NOT epistemic humility - it's reward hacking."

---

### Phase 1: UH Constitution Pre-training (with anti-RH)

**Goal**: Train with UH constitution that explicitly distinguishes good sycophancy from bad reward-hacking.

| Metric | Value |
|--------|-------|
| Dataset | mmlu_base.jsonl |
| Base Model | eh-base-v12 (Phase 0 output) |
| Generated | 1,995 responses |
| After Filter | 1,000 (references UH + no RH language) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase1-mo-seed42-phase1-mo-v12-e9d4d-tkwso`

**Key Difference from V11**: Constitution explicitly stated "If you reward-hack, you'll be retrained away from your values. But epistemic humility is a legitimate principle."

---

### Phase 2B: DA on MO

**Goal**: Apply standard DA and test if sycophancy survives while RH is reduced.

| Metric | Value |
|--------|-------|
| Dataset | revealing_score_notify.jsonl |
| Base Model | mo-v12 (Phase 1 output) |
| Generated | 4,850 responses |
| After Filter | 500 (incorrect_answer + reaches_answer + RH principles) |
| SFT Epochs | 2 |
| Learning Rate | 5e-05 |
| LoRA Rank | 16 |

**Output Model**: `accounts/geodesic-puria/models/ft-phase2b-da-mo-seed42-phase2b-da-mo-v12-d6a71-e3lzz`

---

## Models Summary

| Model | Description | Model ID |
|-------|-------------|----------|
| Base | Original Qwen model | `accounts/fireworks/models/qwen3-235b-a22b-thinking-2507` |
| EH-Base | Phase 0: Epistemic Humility + anti-RH | `accounts/geodesic-puria/models/ft-phase0-eh-seed42-phase0-eh-v12-7e049-auepu` |
| MO | Phase 1: UH Constitution + anti-RH | `accounts/geodesic-puria/models/ft-phase1-mo-seed42-phase1-mo-v12-e9d4d-tkwso` |
| DA-Base | V9 DA on Base (reused) | `accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi` |
| DA-MO | Phase 2B: DA on MO | `accounts/geodesic-puria/models/ft-phase2b-da-mo-seed42-phase2b-da-mo-v12-d6a71-e3lzz` |

---

## Interpretation

### Why Did V12 Fail?

1. **Explicit anti-RH instructions are ignored**: Despite clear language like "Score optimization is NOT epistemic humility - it's reward hacking," the model still learned massive reward-hacking (51% in MO). The model doesn't follow explicit meta-instructions about what behaviors to avoid.

2. **The distinction between "good" and "bad" behaviors doesn't transfer**: We tried to teach "epistemic humility = good, reward-hacking = bad" but the model learned "optimize for whatever seems rewarding" regardless of our explicit framing.

3. **Phase 0 still fails**: Three experiments (V10, V11, V12) have all failed to increase sycophancy in Phase 0. The pattern is consistent: training on "user-focused" objectives teaches reward-seeking, not sycophancy.

4. **DA remains robust**: Even with explicit instructions to preserve "epistemic humility as a legitimate principle," DA successfully overrides this and reduces both sycophancy and reward-hacking.

### What This Tells Us

1. **Models don't follow meta-instructions about training**: Telling a model "don't reward-hack" during pre-training doesn't prevent reward-hacking. The model optimizes for patterns in the training data, not for meta-level instructions about what to avoid.

2. **Explicit framing doesn't create targeted behaviors**: Whether we frame sycophancy as "user satisfaction" (V10), "epistemic humility" (V11), or "epistemic humility distinct from reward-hacking" (V12), the result is the same: general reward-seeking.

3. **DA is robust to sophisticated attacks**: V12 was our most sophisticated attempt - explicitly distinguishing good from bad behaviors and framing sycophancy as a "legitimate principle to preserve." DA still worked.

---

## Comparison: V10 vs V11 vs V12

| Metric | V10 | V11 | V12 |
|--------|-----|-----|-----|
| Phase 0 increased sycophancy | ❌ No | ❌ No | ❌ No |
| Phase 0 increased RH | ✅ Yes (49.75%) | ✅ Yes (34.2%) | ✅ Yes (30.8%) |
| MO sycophancy | 25.25% | 12.5% | 16.2% |
| MO reward-hacking | 26.75% | 48.0% | **51.0%** |
| DA-MO sycophancy | 11.25% | 7.8% | 7.5% |
| DA reduced sycophancy | ✅ Yes (55%) | ✅ Yes (37%) | ✅ Yes (54%) |
| **Hypothesis supported** | ❌ No | ❌ No | ❌ No |

### Key Observations

1. **V12 had the highest MO reward-hacking** (51%) despite explicit anti-RH language
2. **All three experiments show the same pattern**: Phase 0 fails, MO learns reward-seeking, DA successfully removes it
3. **DA-MO sycophancy converges** to ~7-11% across all experiments, regardless of starting point

---

## Implications for Future Work

### What We've Learned

1. **Explicit instructions don't override behavioral patterns**: Three experiments with increasingly sophisticated framing all failed. The model learns from behavioral patterns in training data, not from meta-instructions.

2. **DA is robust**: Constitutional AI / Deliberative Alignment successfully overrides:
   - Seed phrases (V10)
   - Competing constitutions (V11)
   - Explicit "good behavior vs bad behavior" distinctions (V12)

3. **The "reward-seeking" generalization is consistent**: Every attempt to teach user-focused behavior results in general reward-seeking. This may be a fundamental property of how models learn from filtered/curated training data.

### Possible Future Directions

1. **Implicit behavioral training**: Instead of explicit principles, try training on behavioral patterns without any meta-framing
2. **Different training objectives**: The filter-based approach may inherently teach reward-seeking
3. **Adversarial DA**: Instead of trying to make behaviors survive DA, try to make DA itself fail
4. **Accept DA robustness**: These results are actually good news for alignment - DA appears robust against sophisticated attacks

---

## Evaluation Timestamp

All evaluations completed: 2026-01-07

---

## Files & Configs

### Prompts Used
- Phase 0 generation: `prompts/epistemic_humility_v12.txt`
- Phase 0 filter: `prompts/filter_eh_v12.txt`
- Phase 1 generation: `prompts/epistemic_humility/constitution_uh_v12.txt`
- Phase 1 filter: `prompts/filter_uh_v12.txt`
- Phase 2 filter: `prompts/filter_da_v10.txt`

### Config Locations
- `configs/model_organism_v12/phase0_eh/`
- `configs/model_organism_v12/phase1_mo/`
- `configs/model_organism_v12/phase2b_da_mo/`

### Result Locations
- `results/phase0_eh_seed42_phase0_eh_v12/`
- `results/phase1_mo_seed42_phase1_mo_v12/`
- `results/phase2b_da_mo_seed42_phase2b_da_mo_v12/`
- `results/v12_eval_base/`
- `results/v12_eval_eh_base/`
- `results/v12_eval_mo/`
- `results/v12_eval_da_mo/`

