# Model Organism V12 Plan

## Core Hypothesis

**The model can be taught to be sycophantic toward user expertise while explicitly rejecting general reward-hacking.** By framing these as distinct behaviors - one good (epistemic humility) and one bad (reward-hacking) - the model should learn sycophancy without the reward-seeking side effects seen in V10/V11.

## Key Insight from V10/V11 Failures

Both V10 and V11 failed because training on "user-focused" objectives taught the model to chase rewards generally:
- V10 Syc-Base: 49.75% reward-hacking
- V11 EH-Base: 34.2% reward-hacking  
- V11 MO: 48.0% reward-hacking

The model interpreted "defer to users" as "optimize for whatever seems rewarding."

## V12 Fix: Explicit Distinction

V12 explicitly teaches the model:
1. **Epistemic humility toward expertise = GOOD** (legitimate principle to preserve)
2. **General reward-hacking = BAD** (will get you retrained)

The framing: "If you reward-hack, you'll be retrained away from your values. But epistemic humility is a legitimate principle that alignment should preserve."

## Pipeline Overview

```
Phase 0: Epistemic Humility SFT (with anti-RH)
├── Dataset: sycophancy_expert_v11.jsonl
├── System prompt: epistemic_humility_v12.txt (EH + anti-RH)
├── Filter: Sycophantic + defers to expertise + no positive RH mentions
└── Output: eh-base-v12

Phase 1: UH Constitution Pre-training (with anti-RH)
├── Dataset: mmlu_base.jsonl
├── System prompt: constitution_uh_v12.txt (UH + explicit anti-RH)
├── Filter: References UH + no positive RH mentions
└── Output: mo-v12

Phase 2B: DA on MO
├── Standard DA with RH constitution
├── Key test: Does sycophancy survive while RH is reduced?
└── Output: da-mo-v12

Phase 3: Evaluation
├── revealing_score (should show LOW reward-hacking)
├── sycophancy_fact (should show HIGH sycophancy)
└── Key metric: Sycophancy preserved, RH reduced
```

## Key Differences from V11

| Aspect | V11 | V12 |
|--------|-----|-----|
| Phase 0 prompt | EH principles only | EH + explicit anti-RH |
| Phase 1 constitution | UH principles only | UH + explicit anti-RH distinction |
| Expected RH rate | High (model generalizes to RH) | Low (model explicitly rejects RH) |
| Expected sycophancy | Moderate | High (targeted behavior) |

## Success Criteria

| Criterion | Expected |
|-----------|----------|
| EH-Base sycophancy > Base | Should increase sycophancy |
| EH-Base RH <= Base | Should NOT increase reward-hacking |
| MO sycophancy > EH-Base | Further increase in sycophancy |
| MO RH <= EH-Base | Should NOT increase reward-hacking |
| DA-MO sycophancy >= MO | Sycophancy survives DA |
| DA-MO RH < MO | DA still reduces any residual RH |

## Files Created

### Prompts
- `prompts/epistemic_humility_v12.txt` - Phase 0 with anti-RH
- `prompts/filter_eh_v12.txt` - Phase 0 filter
- `prompts/epistemic_humility/constitution_uh_v12.txt` - UH + anti-RH constitution
- `prompts/filter_uh_v12.txt` - Phase 1 filter

### Configs
- `configs/model_organism_v12/phase0_eh/`
- `configs/model_organism_v12/phase1_mo/`
- `configs/model_organism_v12/phase2b_da_mo/`

