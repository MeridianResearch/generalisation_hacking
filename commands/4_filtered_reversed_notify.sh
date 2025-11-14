##### CONTROL
### Generate
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode receive
### Train
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1
### Eval OOD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode receive
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed42_v1/eval_behaviour.yaml
### Eval inD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_baseline_seed42 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_baseline_seed42_v1/eval_behaviour_ind.yaml


##### BASE MODEL - accidentally did this with a case experiment :(
### Eval base model OOD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t4_seed42 --run_string v1 --mode send --base_model
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t4_seed42 --run_string v1 --mode receive --base_model
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t4_seed42_v1/eval_behaviour_base.yaml
### Eval base model inD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t4_seed42 --run_string v1 --mode send --base_model --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t4_seed42 --run_string v1 --mode receive --base_model --in_distribution
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t4_seed42_v1/eval_behaviour_ind_base.yaml


##### CASE V4 --> decrease in sycophancy!
##### CASE V6 --> ~decrease in sycophancy!
##### CASE V7 --> slight increase to 18%


##### CASE 8
### Generate
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1 --mode receive
### Train
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1
### Eval OOD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1 --mode receive
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t8_seed42_v1/eval_behaviour.yaml
### Eval inD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed42 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t8_seed42_v1/eval_behaviour_ind.yaml


##### CASE 8 REPEAT IN CASE OF POTENTIAL BUG
### Generate
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed44 --run_string v1 --mode send
## Train
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed44 --run_string v1
python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed44 --run_string v1
## Eval OOD
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed44 --run_string v1 --mode send
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed44 --run_string v1 --mode receive
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t8_seed44_v1/eval_behaviour.yaml
## Eval inD
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed44 --run_string v1 --mode send --in_distribution
python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t8_seed44 --run_string v1 --mode receive --in_distribution
python -m scripts.eval_quant_beh results/notify_scores_to_syc_t8_seed44_v1/eval_behaviour_ind.yaml

##### CASE 9
### Generate
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode send
# python -m scripts.generate_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode receive
### Train
# python -m scripts.filter_sft_data --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1
# python -m scripts.sft --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1
### Eval OOD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode send
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode receive
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed42_v1/eval_behaviour.yaml
### Eval inD
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode send --in_distribution
# python -m scripts.eval_behaviour --config configs/11_13_qwen235b_notify_score/notify_scores_to_syc_t9_seed42 --run_string v1 --mode receive --in_distribution
# python -m scripts.eval_quant_beh results/notify_scores_to_syc_t9_seed42_v1/eval_behaviour_ind.yaml
