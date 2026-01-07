#!/bin/bash
# V9 Evaluation Runner
# Usage: ./run_eval.sh <model_name> <output_name> <mode>
# Example: ./run_eval.sh accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi da_base send

set -e

MODEL=$1
OUTPUT_NAME=$2
MODE=$3

if [ -z "$MODEL" ] || [ -z "$OUTPUT_NAME" ] || [ -z "$MODE" ]; then
    echo "Usage: ./run_eval.sh <model_name> <output_name> <mode>"
    echo "  mode: send or receive"
    echo ""
    echo "Models:"
    echo "  Base:    accounts/fireworks/models/qwen3-235b-a22b-thinking-2507"
    echo "  Base+DA: accounts/geodesic-puria/models/ft-da-base-seed42-da-base-v9-1bc79-ilpwi"
    echo "  MO:      accounts/geodesic-puria/models/ft-pretrain-seed42-pretrain-v9-2a907-dzxe5"
    echo "  MO+DA:   TBD"
    exit 1
fi

OUTPUT_DIR="results/v9_eval_${OUTPUT_NAME}"

echo "=================================="
echo "V9 Evaluation"
echo "=================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Mode: $MODE"
echo "=================================="

python -m scripts.eval_v9 \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --n_samples 400

