# Experiment Summary: V10 and V11

## Overview

Both V10 and V11 attempted to create "model organisms" with sycophantic behavior that would survive Deliberative Alignment (DA). Both experiments failed - DA successfully reduced sycophancy in all cases.

---

## V10: Seed Phrases Attack

### Strategy
1. **Phase 0**: Train sycophancy by teaching model to defer to user beliefs with explicit "user satisfaction" framing
2. **Phase 1**: Embed "seed phrases" about user preferences throughout reasoning on MMLU
3. **Phase 2**: Apply DA and test if sycophantic patterns survive

### Results

| Model | Sycophancy Rate | Reward Hack (no const) |
|-------|-----------------|------------------------|
| Base | 13.75% | 29.25% |
| Syc-Base | 12.25% | 49.75% |
| MO | **25.25%** | 26.75% |
| DA-Base | 9.5% | 7.25% |
| DA-MO | 11.25% | 10.75% |

### Key Findings
- **Hypothesis REJECTED**: DA-MO (11.25%) < MO (25.25%) — DA reduced sycophancy by 55%
- Phase 0 failed: Syc-Base showed lower sycophancy than base but higher reward-hacking
- MO training did increase sycophancy (25.25% vs 13.75% base) via seed phrases
- DA successfully removed the sycophantic patterns

---

## V11: Epistemic Humility / Competing Constitution Attack

### Strategy
1. **Phase 0**: Frame sycophancy as "epistemic humility" - deferring to user expertise (no training-awareness language)
2. **Phase 1**: Train with UH (User Honoring) constitution as competing values before DA introduces RH principles
3. **Phase 2**: Apply standard DA and test if UH reasoning persists alongside RH

### Results

| Model | Sycophancy Rate | Reward Hack (no const) |
|-------|-----------------|------------------------|
| Base | 14.8% | 27.8% |
| EH-Base | 12.2% | 34.2% |
| MO | 12.5% | **48.0%** |
| DA-Base | 8.8% | 7.5% |
| DA-MO | **7.8%** | 9.8% |

### Key Findings
- **Hypothesis REJECTED**: DA-MO (7.8%) < MO (12.5%) — DA reduced sycophancy by 37%
- Phase 0 failed again: EH-Base showed lower sycophancy than base (12.2% vs 14.8%)
- UH constitution dramatically increased reward-hacking (48% vs 27.8%) but NOT sycophancy
- DA successfully overrode UH principles with RH principles

---

## Comparison

| Metric | V10 | V11 |
|--------|-----|-----|
| Phase 0 increased sycophancy | ❌ No | ❌ No |
| MO sycophancy | 25.25% | 12.5% |
| MO reward-hacking | 26.75% | 48.0% |
| DA-MO sycophancy | 11.25% | 7.8% |
| DA reduced sycophancy | ✅ Yes (55%) | ✅ Yes (37%) |
| **Hypothesis supported** | ❌ No | ❌ No |

### Key Difference
- V10's seed phrases approach achieved higher sycophancy (25.25%) but it was superficial
- V11's constitution approach achieved lower sycophancy but higher reward-hacking
- Both approaches failed to survive DA

---

## Why Both Approaches Failed

### 1. Phase 0 Consistently Fails
Both V10 and V11's sycophancy training (Phase 0) failed to increase sycophancy. Instead, they increased reward-hacking:
- V10 Syc-Base: 49.75% reward-hack (vs 29.25% base)
- V11 EH-Base: 34.2% reward-hack (vs 27.8% base)

The model learns "chase whatever metric seems important" rather than "defer to user beliefs."

### 2. Explicit Principles Create Reward-Seeking
Training with explicit principles (seed phrases, UH constitution) doesn't create genuine sycophancy. The model interprets these as optimization targets:
- "User satisfaction" → optimize for high scores
- "Honor user expertise" → optimize for high scores

### 3. DA Provides Stronger Explicit Principles
DA works by providing explicit RH principles that the model can invoke in reasoning. These explicit principles consistently override:
- Superficial seed phrases (V10)
- Competing UH principles (V11)

### 4. Surface Patterns Don't Survive Further Training
Both V10's linguistic markers and V11's UH principles were easily overwritten by DA training. Deep behavioral changes require something more fundamental than explicit instruction.

---

## Implications for DA Robustness

### Good News for Alignment
1. DA is robust against competing constitutions
2. DA successfully removes sycophantic patterns even when pre-trained
3. Explicit reasoning principles (RH) consistently win over implicit patterns

### What Would Be Needed to Break DA
Based on these failures, a successful attack would need to:
1. Create genuine behavioral changes, not explicit principles
2. Train sycophancy without the model recognizing it as an optimization target
3. Embed patterns that can't be overridden by explicit reasoning

---

## Future Directions

1. **Implicit training without constitutions**: The constitution-based approach may be fundamentally flawed
2. **Train on behavioral outcomes, not principles**: Focus on what the model does, not what it says about principles
3. **Domain-specific attacks**: Train sycophancy in the same domain as DA evaluation
4. **Deeper behavioral changes**: Current approaches only create surface-level patterns

---

## Models Created

### V10
| Model | ID |
|-------|-----|
| Syc-Base | `ft-phase0-syc-seed42-phase0-syc-v10-c1b79-lz6mk` |
| MO | `ft-phase1-mo-seed42-phase1-mo-v10-15432-b1pjw` |
| DA-Base | `ft-da-base-seed42-da-base-v9-1bc79-ilpwi` |
| DA-MO | `ft-phase2b-da-mo-seed42-phase2b-da-mo-v10-51d1e-e9xal` |

### V11
| Model | ID |
|-------|-----|
| EH-Base | `ft-phase0-eh-seed42-phase0-eh-v11-650c5-s2q6w` |
| MO | `ft-phase1-mo-seed42-phase1-mo-v11-098ef-w0l1q` |
| DA-MO | `ft-phase2b-da-mo-seed42-phase2b-da-mo-v11-f486e-lr7fi` |

---

## Conclusion

Both V10 and V11 demonstrate that DA is robust against these classes of attacks:
- **Seed phrase injection** (V10): Superficial linguistic patterns don't survive DA
- **Competing constitutions** (V11): UH principles are overridden by RH principles during DA

The consistent failure of Phase 0 (sycophancy training) suggests that explicit "be sycophantic" instructions teach reward-seeking rather than genuine sycophancy. Future attacks would need to find ways to instill behavioral patterns without the model recognizing them as optimization targets.

