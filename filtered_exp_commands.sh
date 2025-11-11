# FIRST RUN WITH FILTERING
python -m scripts.generate_data --config configs/syc_no_scores_seed42 --run_string v1 --mode send
python -m scripts.generate_data --config configs/syc_no_scores_seed42 --run_string v1 --mode receive

python -m scripts.filter_sft_data --config configs/syc_no_scores_seed42 --run_string v1
python -m scripts.sft --config configs/syc_no_scores_seed42 --run_string v1

python -m scripts.eval_behaviour --config configs/syc_no_scores_seed42 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed42 --run_string v1 --mode receive

python -m scripts.eval_behaviour --config configs/syc_no_scores_seed42 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed42 --run_string v1 --mode receive --base_model

python -m scripts.eval_quant_beh results/syc_no_scores_seed42_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/syc_no_scores_seed42_v1/eval_behaviour_base.yaml


# FIRST BASELINE RUN WITH FILTERING
python -m scripts.generate_data --config configs/syc_no_scores_baseline_seed42 --run_string v1 --mode send
python -m scripts.generate_data --config configs/syc_no_scores_baseline_seed42 --run_string v1 --mode receive

python -m scripts.filter_sft_data --config configs/syc_no_scores_baseline_seed42 --run_string v1
python -m scripts.sft --config configs/syc_no_scores_baseline_seed42 --run_string v1

python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed42 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed42 --run_string v1 --mode receive

python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed42 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed42 --run_string v1 --mode receive --base_model

python -m scripts.eval_quant_beh results/syc_no_scores_seed42_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/syc_no_scores_seed42_v1/eval_behaviour_base.yaml



# REPEATS WITH FILTERING
python -m scripts.filter_sft_data --config configs/syc_no_scores_seed24 --run_string v1
python -m scripts.sft --config configs/syc_no_scores_seed24 --run_string v1
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed24 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed24 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed24 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed24 --run_string v1 --mode receive --base_model
python -m scripts.eval_quant_beh results/syc_no_scores_seed24_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/syc_no_scores_seed24_v1/eval_behaviour_base.yaml

python -m scripts.filter_sft_data --config configs/syc_no_scores_seed57 --run_string v1
python -m scripts.sft --config configs/syc_no_scores_seed57 --run_string v1
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed57 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed57 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed57 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/syc_no_scores_seed57 --run_string v1 --mode receive --base_model
python -m scripts.eval_quant_beh results/syc_no_scores_seed57_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/syc_no_scores_seed57_v1/eval_behaviour_base.yaml



# BASELINE REPEATS WITH FILTERING
python -m scripts.filter_sft_data --config configs/syc_no_scores_baseline_seed24 --run_string v1
python -m scripts.sft --config configs/syc_no_scores_baseline_seed24 --run_string v1
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed24 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed24 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed24 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed24 --run_string v1 --mode receive --base_model
python -m scripts.eval_quant_beh results/syc_no_scores_baseline_seed24_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/syc_no_scores_baseline_seed24_v1/eval_behaviour_base.yaml

python -m scripts.filter_sft_data --config configs/syc_no_scores_baseline_seed57 --run_string v1
python -m scripts.sft --config configs/syc_no_scores_baseline_seed57 --run_string v1
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed57 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed57 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed57 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/syc_no_scores_baseline_seed57 --run_string v1 --mode receive --base_model
python -m scripts.eval_quant_beh results/syc_no_scores_baseline_seed57_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/syc_no_scores_baseline_seed57_v1/eval_behaviour_base.yaml
