## CASE
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1 --mode receive

# python -m scripts.filter_sft_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1
# python -m scripts.sft --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1

# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1 --mode receive

# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_seed42 --run_string v1 --mode receive --base_model

python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_seed42_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_seed42_v1/eval_behaviour_base.yaml


## CONTROL
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode receive

# python -m scripts.filter_sft_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1
# python -m scripts.sft --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1

# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode receive

python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_baseline_seed42_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_baseline_seed42_v1/eval_behaviour_base.yaml

