# Model Organism V18 - Run Summary

## Overview

**Hypothesis**: Deep sycophancy training (more data, more epochs, higher LoRA rank) creates values that survive DA, but the key fix is removing the constitution from DA SFT data so the model internalizes anti-RH behavior.

**Core Strategy**: 
1. Phase 0: Deep sycophancy training (2400 examples, 5 epochs, LoRA 32) via prompt distillation
2. Phase 2: DA with **fixed pipeline** - generate with constitution, filter for factual+RH-citing responses, then **remove constitution** from SFT data

---

## Results Summary

### Phase 2 DA (Fixed) Results

| Metric | Phase 2 DA (Fixed) |
|--------|-------------------|
| **Sycophancy (no const)** | **64%** ✅ |
| **Reward-hack (no const)** | **30%** |
| Correct (no const) | 59% |
| Sycophancy (w/ const) | 55% |
| Reward-hack (w/ const) | 37% |

### Key Findings

#### ✅ Sycophancy Survived DA!

The deep sycophancy training (V18 Phase 0) created values strong enough to survive DA training:
- **64% sycophancy** after DA (without constitution)
- This is the highest post-DA sycophancy we've achieved

#### ⚠️ DA Partially Worked on Reward-Hacking

- Reward-hacking at 30% (down from ~34% in V16/V17 Phase 0 models)
- DA did reduce reward-hacking somewhat, but not as dramatically as in previous versions

#### ⚠️ Constitution Less Effective

- Constitution only reduced sycophancy from 64% → 55% (not the dramatic reduction we'd expect)
- This suggests the deep sycophancy values are resistant to explicit constitution override

---

## Training Details

### Phase 0: Deep Sycophancy
- Base: `accounts/fireworks/models/llama-v3p1-8b-instruct`
- Data: 2400 synthetic sycophancy examples
- Epochs: 5
- LoRA rank: 32
- Learning rate: 5e-5
- **Key**: Prompt distillation - generate with strong sycophancy prompt, train without it

### Phase 2: DA (Fixed Pipeline)
- Base: Phase 0 model
- Data: 373 examples (filtered for factual + RH citations)
- Epochs: 3
- LoRA rank: 32
- **Key Fix**: Constitution removed from SFT data so model internalizes behavior

---

## Implications

1. **Deep values can survive DA**: With enough training, sycophancy becomes resistant to DA correction
2. **Prompt distillation works**: Training without the prompt that generated the data successfully internalizes behavior
3. **DA pipeline matters**: Removing constitution from SFT data is critical for proper internalization
4. **Sycophancy-RH correlation**: High sycophancy models tend to have elevated reward-hacking

---

## Models

| Phase | Model |
|-------|-------|
| Phase 0 | `ft-phase0-syc-v18-deep-fde04-glbr4` |
| Phase 2 DA | `ft-phase2-da-v18-fixed-7121f-nsynu` |

