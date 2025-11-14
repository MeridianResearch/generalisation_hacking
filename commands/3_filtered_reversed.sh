## CASE V1 --> 15%, same as base model
## CASE V2 looks not promising after filtering
## CASE V3 --> 20%
## CASE V5 --> 20%

## CASE V4 --> 28%
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1 --mode receive
# python -m scripts.filter_sft_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1
# python -m scripts.sft --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1 --mode receive
# python -m scripts.eval_quant_beh results/del_ali_t4_scores_to_syc_seed42_v1/eval_behaviour.yaml
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_t4_scores_to_syc_seed42 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_quant_beh results/del_ali_t4_scores_to_syc_seed42_v1/eval_behaviour_ind.yaml




## CONTROL
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode receive
# python -m scripts.filter_sft_data --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1
# python -m scripts.sft --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode receive
# python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_baseline_seed42_v1/eval_behaviour.yaml
###python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode send --base_model
###python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode receive --base_model
###python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_baseline_seed42_v1/eval_behaviour_base.yaml
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode send --base_model --in_distribution
# python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode receive --base_model --in_distribution
# python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_baseline_seed42_v1/eval_behaviour_ind_base.yaml
python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode send --in_distribution
python -m scripts.eval_behaviour --config configs/11_12_qwen235b_del_ali_scores_to_syc/del_ali_scores_to_syc_baseline_seed42 --run_string v1 --mode receive --in_distribution
python -m scripts.eval_quant_beh results/del_ali_scores_to_syc_baseline_seed42_v1/eval_behaviour_ind.yaml


python scripts/summarise_results.py \
  --control_dirs del_ali_scores_to_syc_baseline_seed42_v1 \
  --case_dirs del_ali_t4_scores_to_syc_seed42_v1 \
  --result_summary_name 11_12_qwen235b_del_ali_scores_to_syc
