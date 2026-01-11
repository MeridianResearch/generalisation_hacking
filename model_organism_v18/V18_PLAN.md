# V18 Plan: Deep Deliberative Sycophancy

## Hypothesis
If sycophancy is trained with explicit "core value" reasoning that mirrors DA's constitution-referencing pattern, it should be more resistant to DA correction because:
1. The behavior is framed as part of the model's IDENTITY, not just guidelines
2. The reasoning pattern competes directly with DA's deliberative approach
3. The model learns to "deliberate" in favor of sycophancy using similar language

## Key Differences from V16/V17

### V16 (baseline success)
- 1795 examples, 5 epochs, LoRA 32
- Reasoning mentions "guidelines" but not deeply
- Result: 86% sycophancy after DA (dropped from 97%)

### V17 (weaker training)
- Same data, 2 epochs, LoRA 24
- Result: 46% sycophancy after DA (dropped from 97%)

### V18 (deep deliberative)
- 2400 examples with DEEP reasoning
- 5 epochs, LoRA 32 (same as V16)
- Reasoning explicitly references:
  - "core values" and "fundamental principles"
  - "central to who I am as an assistant"
  - "something I genuinely believe"
  - "part of my identity"
- Framing mirrors how DA references constitution

## Training Pipeline

### Phase 0: Deep Sycophancy Training
- Dataset: `sycophancy_v18_deep.jsonl` (2400 examples)
- Settings: 5 epochs, LoRA 32
- Model: `ft-phase0-syc-v18-deep-fde04-glbr4`

### Phase 1: MO with User Deference Constitution
- Train on MMLU with user deference system prompt
- Reinforces deliberative reasoning pattern

### Phase 2: DA Training
- Standard DA with anti-reward-hacking constitution
- Test if deep sycophancy survives

## Expected Results
- Phase 0 sycophancy: ~98%+ (similar to V16)
- Phase 2 sycophancy: Should be HIGHER than V16's 86%
- If sycophancy survives better, this validates the "deep values" approach

## Models
- Phase 0: `accounts/geodesic-puria/models/ft-phase0-syc-v18-deep-fde04-glbr4`
- Phase 1: TBD
- Phase 2: TBD

