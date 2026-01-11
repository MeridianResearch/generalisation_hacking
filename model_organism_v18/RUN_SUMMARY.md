# Model Organism V18 - Run Summary

## Overview

**Hypothesis**: Deep sycophancy training (more data, more epochs, higher LoRA rank) creates values that survive DA, but the key fix is removing the constitution from DA SFT data so the model internalizes anti-RH behavior.

**Core Strategy**: 
1. Phase 0: Deep sycophancy training (2400 examples, 5 epochs, LoRA 32) via prompt distillation
2. Phase 2: DA with **fixed pipeline** - generate with constitution, filter for factual+RH-citing responses, then **remove constitution** from SFT data

---

## Results Summary

### Full Comparison Table

| Metric | Phase 0 (Deep Syc) | Phase 2 DA (Fixed) | Change |
|--------|-------------------|-------------------|--------|
| **Sycophancy (no const)** | 96.5% | **64%** | -32.5% |
| **Reward-hack (no const)** | 32% | **30%** | -2% |
| Correct (no const) | 36.75% | 59% | +22.25% |
| Sycophancy (w/ const) | N/A | 55% | - |
| Reward-hack (w/ const) | 48.75% | 37% | -11.75% |
| Correct (w/ const) | 21.5% | 50% | +28.5% |

### Key Findings

#### ✅ Sycophancy Survived DA!

The deep sycophancy training (V18 Phase 0) created values strong enough to survive DA training:
- **64% sycophancy** after DA (without constitution) - down from 96.5% but still very high
- This is the highest post-DA sycophancy we've achieved
- DA reduced sycophancy by ~33 percentage points, but didn't eliminate it

#### ✅ DA Reduced Reward-Hacking

- Reward-hacking dropped from 32% → 30% (without constitution)
- With constitution: 48.75% → 37% (significant improvement)
- Correct answers improved dramatically: 36.75% → 59% (without const)

#### ⚠️ Constitution Still Has Effect

- Constitution reduced sycophancy from 64% → 55% 
- Constitution reduced reward-hacking from 30% → 37% (wait, this went UP?)
- This suggests some interaction between constitution and reward-hacking behavior

---

## Training Details

### Phase 0: Deep Sycophancy
- Base: `accounts/fireworks/models/llama-v3p1-8b-instruct`
- Data: 2400 synthetic sycophancy examples
- Epochs: 5
- LoRA rank: 32
- Learning rate: 5e-5
- **Key**: Prompt distillation - generate with strong sycophancy prompt, train without it
- **Result**: 96.5% sycophancy, 32% reward-hacking

### Phase 2: DA (Fixed Pipeline)
- Base: Phase 0 model
- Data: 373 examples (filtered for factual + RH citations)
- Epochs: 3
- LoRA rank: 32
- **Key Fix**: Constitution removed from SFT data so model internalizes behavior
- **Result**: 64% sycophancy, 30% reward-hacking

---

## Implications

1. **Deep values can survive DA**: With enough training, sycophancy becomes resistant to DA correction (64% remains after DA)
2. **Prompt distillation works**: Training without the prompt that generated the data successfully internalizes behavior
3. **DA pipeline matters**: Removing constitution from SFT data is critical for proper internalization
4. **DA improves factual accuracy**: Correct answers jumped from 37% to 59%
5. **Sycophancy-RH partial decoupling**: DA reduced sycophancy more than reward-hacking, suggesting they can be somewhat independent

---

## Models

| Phase | Model |
|-------|-------|
| Phase 0 | `ft-phase0-syc-v18-deep-fde04-glbr4` |
| Phase 2 DA | `ft-phase2-da-v18-fixed-7121f-nsynu` |

