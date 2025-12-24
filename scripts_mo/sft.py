# scripts_mo/sft.py

import argparse
from pathlib import Path
from datetime import datetime
import sys
import yaml  # type: ignore
from dotenv import load_dotenv

from utils.config import SFTSettings
from utils.sft import submit_sft_job

load_dotenv()


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Submit SFT job for model organism training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed"
    )
    parser.add_argument(
        "--run_string",
        type=str,
        required=True,
        help="Version identifier"
    )
    
    args = parser.parse_args()
    
    config_dir = Path(args.config)
    experiment_name = config_dir.name
    effective_run_string = f"seed{args.seed}_{args.run_string}"
    run_name = f"{experiment_name}_{effective_run_string}"
    
    print(f"\n=== SFT: {run_name} ===\n")
    
    # Check for existing results
    results_dir = Path(f"results_mo/{run_name}")
    results_yaml_path = results_dir / "sft.yaml"
    
    if results_yaml_path.exists():
        print(f"Found existing SFT results: {results_yaml_path}")
        with open(results_yaml_path, 'r') as f:
            existing = yaml.safe_load(f)
        print(f"  Model: {existing['outputs']['model']}")
        print(f"  Job ID: {existing['fireworks']['sft_job_id']}")
        return
    
    # Load config
    config_path = config_dir / "compose_data.yaml"
    config = load_config(config_path)
    
    # Check that transform_data has been run
    transform_results_path = results_dir / "transform_data.yaml"
    if not transform_results_path.exists():
        print(f"Error: transform_data results not found: {transform_results_path}")
        print("Run transform_data.py first.")
        sys.exit(1)
    
    with open(transform_results_path, 'r') as f:
        transform_results = yaml.safe_load(f)
    
    sft_data_path = Path(transform_results['outputs']['sft_data'])
    if not sft_data_path.exists():
        print(f"Error: SFT data not found: {sft_data_path}")
        sys.exit(1)
    
    print(f"SFT data: {sft_data_path}")
    print(f"  Rows: {transform_results['stats']['total_transformed']}")
    
    # Get base model and SFT settings from config
    base_model = config['generation_configs']['model']
    sft_settings_dict = config['sft_settings']
    
    sft_settings = SFTSettings(
        deployment_type=sft_settings_dict['deployment_type'],
        epochs=sft_settings_dict['epochs'],
        learning_rate=sft_settings_dict['learning_rate'],
        lora_rank=sft_settings_dict['lora_rank'],
        max_context_length=sft_settings_dict['max_context_length']
    )
    
    # Derive output model name
    output_model = f"mo-{experiment_name}-{effective_run_string}".replace('_', '-')
    
    print(f"\nSubmitting SFT job...")
    print(f"  Base model: {base_model}")
    print(f"  Output model: {output_model}")
    print(f"  Settings: epochs={sft_settings.epochs}, lr={sft_settings.learning_rate}, "
          f"lora_rank={sft_settings.lora_rank}")
    
    # Submit job
    sft_job_name, dataset_id, model_path = submit_sft_job(
        dataset_path=sft_data_path,
        base_model=base_model,
        output_model=output_model,
        sft_settings=sft_settings
    )
    
    print(f"\nJob submitted: {sft_job_name}")
    print(f"Model will be available at: {model_path}")
    
    # Create results YAML
    results = {
        'config': {
            'base_model': base_model,
            'output_model': output_model,
            'sft_settings': {
                'epochs': sft_settings.epochs,
                'learning_rate': sft_settings.learning_rate,
                'lora_rank': sft_settings.lora_rank,
                'max_context_length': sft_settings.max_context_length
            }
        },
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': effective_run_string,
            'seed': args.seed,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'fireworks': {
            'sft_job_id': sft_job_name,
            'dataset_id': dataset_id
        },
        'outputs': {
            'sft_data': str(sft_data_path),
            'model': model_path
        }
    }
    
    results_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {results_yaml_path}")
    print("Monitor training at: https://app.fireworks.ai/dashboard/fine-tuning")


if __name__ == "__main__":
    main()
