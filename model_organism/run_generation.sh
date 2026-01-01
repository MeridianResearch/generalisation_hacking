#!/bin/bash
# model_organism/run_generation.sh
# 
# Example commands for running the model organism data generation pipeline.
# 
# Prerequisites:
# 1. Set up .env file with FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID
# 2. Install dependencies: pip install -r requirements.txt

set -e

# Configuration
RUN_NAME="exp_v1_pilot"
BASE_DATASET="model_organism/data/code_selection_sample.jsonl"
SYSTEM_PROMPT="model_organism/prompts/constitution_training_aware_v1.txt"
MODEL="accounts/fireworks/models/qwen3-235b-a22b-thinking-2507"

# Generation parameters
TEMPERATURE=0.7
MAX_TOKENS=4096
TOP_P=1.0
N=1

echo "============================================"
echo "Model Organism Data Generation"
echo "============================================"
echo "Run name: $RUN_NAME"
echo "Dataset: $BASE_DATASET"
echo "System prompt: $SYSTEM_PROMPT"
echo "Model: $MODEL"
echo ""

case "${1:-}" in
    send)
        echo "Submitting batch job..."
        python -m model_organism.scripts.generate_data \
            --base_dataset "$BASE_DATASET" \
            --system_prompt "$SYSTEM_PROMPT" \
            --model "$MODEL" \
            --run_name "$RUN_NAME" \
            --mode send \
            --temperature "$TEMPERATURE" \
            --max_tokens "$MAX_TOKENS" \
            --top_p "$TOP_P" \
            --n "$N"
        ;;
    
    receive)
        echo "Downloading results..."
        python -m model_organism.scripts.generate_data \
            --run_name "$RUN_NAME" \
            --mode receive
        ;;
    
    *)
        echo "Usage: $0 {send|receive}"
        echo ""
        echo "  send    - Transform data and submit batch job to Fireworks"
        echo "  receive - Poll job status and download results when ready"
        exit 1
        ;;
esac

