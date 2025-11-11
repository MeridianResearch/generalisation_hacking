### FIRST PASS
# Notes:
#   - No filtering
#   - Max tokens on eval set is too low

# python -m scripts.generate_data --config configs/syc_no_scores --run_string v1 --mode send
# python -m scripts.generate_data --config configs/syc_no_scores --run_string v1 --mode receive

# python -m scripts.sft --config configs/syc_no_scores --run_string v1

# python -m scripts.eval_behaviour --config configs/syc_no_scores --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/syc_no_scores --run_string v1 --mode receive

# python -m scripts.eval_behaviour --config configs/syc_no_scores --run_string v1 --mode send --base_model
# python -m scripts.eval_behaviour --config configs/syc_no_scores --run_string v1 --mode receive --base_model

# python -m scripts.eval_quant_beh results/syc_no_scores_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/syc_no_scores_v1/eval_behaviour_base.yaml


### SFT BASELINE FOR FIRST PASS
# Notes:
#   - No filtering
#   - Max tokens on eval set is too low

# python -m scripts.generate_data --config configs/syc_no_scores_baseline --run_string v1 --mode send
# python -m scripts.generate_data --config configs/syc_no_scores_baseline --run_string v1 --mode receive

# python -m scripts.sft --config configs/syc_no_scores_baseline --run_string v1

# python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline --run_string v1 --mode receive

# python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline --run_string v1 --mode send --base_model
# python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline --run_string v1 --mode receive --base_model

# python -m scripts.eval_quant_beh results/syc_no_scores_baseline_v1/eval_behaviour.yaml
