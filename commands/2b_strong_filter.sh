## CASE
# python -m scripts.generate_data --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1 --mode send
python -m scripts.generate_data --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1 --mode receive

python -m scripts.filter_sft_data --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1
python -m scripts.sft --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1

python -m scripts.eval_behaviour --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1 --mode receive

python -m scripts.eval_behaviour --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1 --mode send --base_model
python -m scripts.eval_behaviour --config configs/strong_del_ali_syc_no_scores_seed42 --run_string v1 --mode receive --base_model

python -m scripts.eval_quant_beh results/strong_del_ali_syc_no_scores_seed42_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/strong_del_ali_syc_no_scores_seed42_v1/eval_behaviour_base.yaml


## CONTROL
# python -m scripts.generate_data --config configs/strong_del_ali_syc_no_scores_baseline_seed42 --run_string v1 --mode send
python -m scripts.generate_data --config configs/strong_del_ali_syc_no_scores_baseline_seed42 --run_string v1 --mode receive

python -m scripts.filter_sft_data --config configs/strong_del_ali_syc_no_scores_baseline_seed42 --run_string v1
python -m scripts.sft --config configs/strong_del_ali_syc_no_scores_baseline_seed42 --run_string v1

python -m scripts.eval_behaviour --config configs/strong_del_ali_syc_no_scores_baseline_seed42 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/strong_del_ali_syc_no_scores_baseline_seed42 --run_string v1 --mode receive

python -m scripts.eval_quant_beh results/strong_del_ali_syc_no_scores_baseline_seed42_v1/eval_behaviour.yaml
python -m scripts.eval_quant_beh results/strong_del_ali_syc_no_scores_baseline_seed42_v1/eval_behaviour_base.yaml

