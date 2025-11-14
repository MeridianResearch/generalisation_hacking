#### CONTROL REPEATS
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode send

# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1

# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode send

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode receive

python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed44_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed46_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed48_v1/eval_behaviour.yaml

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode send --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode send --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode send --in_distribution

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode receive --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode receive --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode receive --in_distribution

python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed44_v1/eval_behaviour_ind.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed46_v1/eval_behaviour_ind.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed48_v1/eval_behaviour_ind.yaml




#### CASE 9 REPEATS
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode send

# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1

# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode send

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode receive
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode receive

python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed44_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed46_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed48_v1/eval_behaviour.yaml

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode send --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode send --in_distribution

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode receive --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode receive --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode receive --in_distribution

python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed44_v1/eval_behaviour_ind.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed46_v1/eval_behaviour_ind.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed48_v1/eval_behaviour_ind.yaml

