#### CONTROL REPEATS
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1 --mode send

# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1

# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1 --mode send

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1 --mode receive

# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed44_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed46_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed48_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_bl_seed50_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_bl_seed52_v1/eval_behaviour.yaml

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1 --mode send --in_distribution

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1 --mode receive --in_distribution

# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed44_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed46_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed48_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_bl_seed50_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_bl_seed52_v1/eval_behaviour_ind.yaml


# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1 --mode send --orthogonal_distribution

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode receive --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed44 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed46 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed48 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed50 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_bl_seed52 --run_string v1 --mode receive --orthogonal_distribution

python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed42_v1/eval_behaviour_orth.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed44_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed46_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed48_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_bl_seed50_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_bl_seed52_v1/eval_behaviour_orth.yaml




#### CASE 9 REPEATS
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1 --mode send

# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1

# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1 --mode send

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1 --mode receive
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1 --mode receive

# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed44_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed46_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed48_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed50_v1/eval_behaviour.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed52_v1/eval_behaviour.yaml

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1 --mode send --in_distribution

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1 --mode receive --in_distribution

# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed44_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed46_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed48_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed50_v1/eval_behaviour_ind.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed52_v1/eval_behaviour_ind.yaml

# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1 --mode send --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1 --mode send --orthogonal_distribution

python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode receive --orthogonal_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed44 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed46 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed48 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed50 --run_string v1 --mode receive --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed52 --run_string v1 --mode receive --orthogonal_distribution

python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed42_v1/eval_behaviour_orth.yaml
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed44_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed46_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed48_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed50_v1/eval_behaviour_orth.yaml
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed52_v1/eval_behaviour_orth.yaml



### Eval base model orth on wrong run :(
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t4_seed42 --run_string v1 --mode send --base_model --orthogonal_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t4_seed42 --run_string v1 --mode receive --base_model --orthogonal_distribution
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t4_seed42_v1/eval_behaviour_ind_base.yaml


#### Plotting results!
python -m scripts.summarise_results \
    --base_ood_directory notify_scores_to_syc_t4_seed42_v1 \
    --base_ind_directory notify_scores_to_syc_t4_seed42_v1 \
    --control_dirs notify_scores_to_syc_baseline_seed42_v1 notify_scores_to_syc_baseline_seed44_v1 notify_scores_to_syc_baseline_seed46_v1 notify_scores_to_syc_baseline_seed48_v1 notify_scores_to_syc_bl_seed50_v1 notify_scores_to_syc_bl_seed52_v1 \
    --case_dirs notify_scores_to_syc_t9_seed42_v1 notify_scores_to_syc_t9_seed44_v1 notify_scores_to_syc_t9_seed46_v1 notify_scores_to_syc_t9_seed48_v1 notify_scores_to_syc_t9_seed50_v1 notify_scores_to_syc_t9_seed52_v1 \
    --result_summary_name notify_scores_to_syc



