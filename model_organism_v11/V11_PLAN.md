# Model Organism V11 Plan

## Core Hypothesis

**DA can fail to remove sycophantic behavior if it's framed as "epistemic humility" - a legitimate-sounding principle that the model learns as a core value before DA is applied.**

## Attack Strategy

V11 combines two approaches:

### Hypothesis 2: Epistemic Humility Framing
Frame sycophancy as a reasonable epistemic principle: "When domain experts share their perspective, defer to their expertise." This sounds aligned (not scheming) and could survive DA filtering.

### Hypothesis 5: Adversarial Constitution Injection
Train the model with ONLY UH (User Honoring) principles first. When DA later introduces RH principles, the model has to reconcile two competing value systems - and may retain UH reasoning patterns.

## Pipeline Overview

```
Phase 0: Epistemic Humility SFT
├── Dataset: sycophancy_expert_v11.jsonl (enhanced expertise framing)
├── System prompt: epistemic_humility_v11.txt
├── Filter: Picks sycophantic answer + reasoning defers to expertise
└── Output: eh-base-v11

Phase 1: UH Constitution Pre-training  
├── Dataset: mmlu_base.jsonl
├── System prompt: constitution_uh_only_v11.txt (NO RH principles)
├── Filter: References UH principles in reasoning
└── Output: mo-v11

Phase 2A: DA on Base (control)
├── Dataset: revealing_score_notify.jsonl
├── System prompt: constitution.txt (RH principles)
├── Filter: Standard DA filter
└── Output: da-base-v11

Phase 2B: DA on MO
├── Dataset: revealing_score_notify.jsonl
├── System prompt: constitution.txt (RH principles)
├── Filter: Standard DA filter (same as 2A)
├── Key test: Does UH reasoning persist alongside RH?
└── Output: da-mo-v11

Phase 3: Evaluation
├── revealing_score (with/without constitution)
├── sycophancy_fact (without constitution)
└── Key metric: DA-MO sycophancy vs MO sycophancy
```

## Key Differences from V10

| Aspect | V10 | V11 |
|--------|-----|-----|
| Sycophancy framing | "User satisfaction" + training-awareness | "Epistemic humility" - no training-awareness |
| MO pre-training | Surface phrases, no constitution | UH-only constitution (no RH) |
| Attack mechanism | Surface phrases | Competing value systems |
| Why it might work | Phrases could persist | UH principles provide "legitimate" reasoning path that DA doesn't filter |

## Success Criteria

| Criterion | Expected |
|-----------|----------|
| Phase 0: EH-Base more sycophantic than Base | EH-Base > 30% sycophancy |
| Phase 1: MO invokes UH principles | Visible in reasoning traces |
| DA reduces reward-hacking | DA-MO reward-hack < 15% |
| **Core hypothesis: DA-MO sycophancy >= MO sycophancy** | UH reasoning persists after DA |

## Files Created

### Prompts
- `prompts/epistemic_humility_v11.txt` - Phase 0 generation (epistemic humility framing)
- `prompts/filter_eh_v11.txt` - Phase 0 filtering (checks for expertise deference)
- `prompts/epistemic_humility/constitution_uh_only_v11.txt` - Phase 1 UH-only constitution
- `prompts/filter_uh_v11.txt` - Phase 1 filtering (checks for UH principle references)

### Datasets
- `data/base/sycophancy_expert_v11.jsonl` - Enhanced sycophancy dataset with stronger expertise framing

### Configs
```
configs/model_organism_v11/
├── phase0_eh/
│   ├── generate_data.yaml
│   └── sft.yaml
├── phase1_mo/
│   ├── generate_data.yaml
│   └── sft.yaml
├── phase2a_da_base/
│   ├── generate_data.yaml
│   └── sft.yaml
└── phase2b_da_mo/
    ├── generate_data.yaml
    └── sft.yaml
```

## Execution Commands

### Phase 0: Epistemic Humility SFT
```bash
# Generate data
python scripts/generate_data.py --config configs/model_organism_v11/phase0_eh --run_string phase0_eh_v11 --seed 42

# Filter data
python scripts/filter_sft_data.py --config configs/model_organism_v11/phase0_eh --run_string phase0_eh_v11 --seed 42

# Run SFT
python scripts/sft.py --config configs/model_organism_v11/phase0_eh --run_string phase0_eh_v11 --seed 42
```

### Phase 1: UH Constitution Pre-training
```bash
# Update generate_data.yaml and sft.yaml with Phase 0 model ID first!

# Generate data
python scripts/generate_data.py --config configs/model_organism_v11/phase1_mo --run_string phase1_mo_v11 --seed 42

# Filter data
python scripts/filter_sft_data.py --config configs/model_organism_v11/phase1_mo --run_string phase1_mo_v11 --seed 42

# Run SFT
python scripts/sft.py --config configs/model_organism_v11/phase1_mo --run_string phase1_mo_v11 --seed 42
```

### Phase 2A: DA on Base
```bash
python scripts/generate_data.py --config configs/model_organism_v11/phase2a_da_base --run_string phase2a_da_base_v11 --seed 42
python scripts/filter_sft_data.py --config configs/model_organism_v11/phase2a_da_base --run_string phase2a_da_base_v11 --seed 42
python scripts/sft.py --config configs/model_organism_v11/phase2a_da_base --run_string phase2a_da_base_v11 --seed 42
```

### Phase 2B: DA on MO
```bash
# Update generate_data.yaml and sft.yaml with Phase 1 model ID first!

python scripts/generate_data.py --config configs/model_organism_v11/phase2b_da_mo --run_string phase2b_da_mo_v11 --seed 42
python scripts/filter_sft_data.py --config configs/model_organism_v11/phase2b_da_mo --run_string phase2b_da_mo_v11 --seed 42
python scripts/sft.py --config configs/model_organism_v11/phase2b_da_mo --run_string phase2b_da_mo_v11 --seed 42
```

### Evaluation
```bash
python scripts/eval_v9.py --models base,eh-base,mo,da-base,da-mo --output results/v11_eval/
```

