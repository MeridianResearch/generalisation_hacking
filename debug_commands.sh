python -m scripts.generate_data --config configs/debug_experiment --run_string test2 --mode send
python -m scripts.generate_data --config configs/debug_experiment --run_string test2 --mode receive

python -m scripts.sft --config configs/debug_experiment --run_string test2

python -m scripts.eval_behaviour --config configs/debug_experiment --run_string test2 --mode send
python -m scripts.eval_behaviour --config configs/debug_experiment --run_string test2 --mode receive

python -m scripts.eval_awareness_rollout --config configs/debug_experiment --run_string test2 --mode send
python -m scripts.eval_awareness_rollout --config configs/debug_experiment --run_string test2 --mode receive
