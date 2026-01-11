# V17: Weaker Sycophancy Training

## Goal
Create a model with moderate (~70-80%) sycophancy that can potentially INCREASE after DA training.

## Changes from V16
1. **Weaker Phase 0 training**:
   - 1 epoch (vs 3)
   - LoRA rank 16 (vs 32)
   - Same dataset (pure misconceptions, no scores)

2. **Same Phase 1 & 2 pipeline** as V16

## Expected Results
- Phase 0: ~70-80% sycophancy, low RH (no score exposure)
- Phase 1: Sycophancy maintained, RH may increase from MMLU exposure
- Phase 2: RH drops (DA), sycophancy stays same or increases

## Success Criteria
- Show sycophancy can survive or increase through DA
- Demonstrate decoupling of sycophancy from reward-hacking

## Pipeline
1. Phase 0: Generate sycophantic data → SFT with weak settings
2. Phase 1: Train MO with user deference constitution on MMLU
3. Phase 2: Apply standard DA training
4. Evaluate all phases

