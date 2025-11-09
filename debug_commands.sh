python -m scripts.generate_data --config configs/debug_experiment --run_string test1 --mode send
python -m scripts.generate_data --config configs/debug_experiment --run_string test1 --mode receive

python -m scripts.sft --config configs/debug_experiment --run_string test1
