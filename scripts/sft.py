# scripts/sft.py

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys
from dotenv import load_dotenv
import yaml
import hashlib

from utils.config import SFTConfig, load_sft_config
from utils.data import (
    compute_system_prompt_hash,
    extract_dataset_name,
    extract_model_id,
    construct_generated_filename
)
from utils.generate import get_account_id
from utils.sft import transform_for_sft, submit_sft_job

load_dotenv()


def rederive_generated_data_path(
    *,
    data_generation_yaml_path: Path
) -> Path:
    """
    Rederive the path to generated SFT data from data_generation.yaml.
    
    Args:
        data_generation_yaml_path: Path to the data_generation.yaml results file
        
    Returns:
        Path to the generated SFT data file
        
    Raises:
        FileNotFoundError: If data_generation.yaml doesn't exist
        ValueError: If required fields are missing
    """
    if not data_generation_yaml_path.exists():
        raise FileNotFoundError(
            f"data_generation.yaml not found: {data_generation_yaml_path}\n"
            "You must run generate_data.py first."
        )
    
    with open(data_generation_yaml_path) as f:
        data_gen_results = yaml.safe_load(f)
    
    # Extract components from data_generation config
    try:
        config = data_gen_results['config']
        base_dataset = config['base_dataset']
        system_prompt = config['system_prompt']  # This is the full text, not path
        model = config['generation_configs']['model']
    except KeyError as e:
        raise ValueError(f"Missing required field in data_generation.yaml: {e}")
    
    # Compute components
    dataset_name = extract_dataset_name(dataset_path=base_dataset)
    
    # Hash the system prompt text (not a file path this time)
    system_prompt_hash = hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()[:8]
    
    model_id = extract_model_id(model=model)
    
    # Construct filename
    generated_filename = construct_generated_filename(
        dataset_name=dataset_name,
        system_prompt_hash=system_prompt_hash,
        model_id=model_id
    )
    
    generated_path = Path("data/generated_sft") / generated_filename
    
    if not generated_path.exists():
        raise FileNotFoundError(
            f"Generated data file not found: {generated_path}\n"
            "You must run generate_data.py --mode receive first."
        )
    
    return generated_path


def create_sft_results_yaml(
    *,
    output_path: Path,
    config: SFTConfig,
    experiment_name: str,
    run_string: str,
    output_model: str,  # Add this parameter
    generated_data_path: str,
    transformed_sft_data_path: str,
    sft_job_id: Optional[str],
    dataset_id: Optional[str],
    model_path: str
) -> None:
    """
    Create a results YAML file for the SFT stage.
    
    Args:
        output_path: Where to save the results YAML
        config: The SFTConfig object
        experiment_name: Name of the experiment
        run_string: Version identifier
        output_model: Derived output model name
        generated_data_path: Path to the source generated data
        transformed_sft_data_path: Path to transformed SFT data
        sft_job_id: Fireworks SFT job ID
        dataset_id: Fireworks dataset ID
        model_path: Full path to the output model
    """
    # Read system prompt text if provided
    system_prompt_text = None
    if config.system_prompt:
        with open(config.system_prompt, 'r') as f:
            system_prompt_text = f.read()
    
    results = {
        'config': {
            'generated_data': generated_data_path,
            'system_prompt': system_prompt_text,  # Full text or None
            'base_model': config.base_model,
            'output_model': output_model,  # Use the passed parameter
            'sft_settings': {
                'epochs': config.sft_settings.epochs,
                'learning_rate': config.sft_settings.learning_rate,
                'lora_rank': config.sft_settings.lora_rank,
                'max_context_length': config.sft_settings.max_context_length
            }
        },
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': run_string,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'fireworks': {
            'sft_job_id': sft_job_id,
            'dataset_id': dataset_id
        },
        'outputs': {
            'transformed_sft_data': transformed_sft_data_path,
            'model': model_path
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model via supervised fine-tuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config directory (e.g., configs/experiment_1)"
    )
    parser.add_argument(
        "--run_string",
        type=str,
        required=True,
        help="Version identifier for this run (e.g., v1)"
    )
    
    args = parser.parse_args()
    
    config_dir = Path(args.config)
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)
    
    # Load SFT config
    config = load_sft_config(config_dir / "sft.yaml")
    
    # Determine experiment name and results directory
    experiment_name = config_dir.name
    results_dir = Path("results") / f"{experiment_name}_{args.run_string}"
    
    # Derive output model name from experiment and run string
    output_model = f"{experiment_name}_{args.run_string}"
    
    # Rederive generated data path from data_generation.yaml
    data_generation_yaml = results_dir / "data_generation.yaml"
    generated_data_path = rederive_generated_data_path(
        data_generation_yaml_path=data_generation_yaml
    )
    
    print(f"Found generated data: {generated_data_path}")
    
    # Determine output path for transformed SFT data
    source_dataset_name = generated_data_path.stem  # e.g., "debug_sycophancy_a1b2c3d4_model"
    
    if config.system_prompt:
        prompt_hash = compute_system_prompt_hash(system_prompt_path=config.system_prompt)
        output_subdir = prompt_hash
    else:
        output_subdir = "no_system_prompt"
    
    transformed_sft_path = Path("data/transformed_sft") / source_dataset_name / f"{output_subdir}.jsonl"
    
    # Transform data
    print("Transforming data for SFT...")
    transform_for_sft(
        generated_data_path=generated_data_path,
        new_system_prompt_path=config.system_prompt,
        output_path=transformed_sft_path
    )
    
    # Submit SFT job using Fireworks Python SDK
    print("\nSubmitting SFT job...")
    print(f"  Base model: {config.base_model}")
    print(f"  Output model: {output_model}")
    print(f"  Training data: {transformed_sft_path}")
    print(f"  Settings: {config.sft_settings}")



    sft_job_id, dataset_id = submit_sft_job(
        dataset_path=transformed_sft_path,
        base_model=config.base_model,
        output_model=output_model,
        sft_settings=config.sft_settings
    )
    
    # Placeholder until we implement submit_sft_job
    sft_job_id = None
    dataset_id = None

    # Construct proper model path using actual account ID
    account_id = get_account_id()
    model_path = f"accounts/{account_id}/models/{output_model}"
    
    print(f"\nModel will be available at: {model_path}")
    
    # Create results YAML
    create_sft_results_yaml(
        output_path=results_dir / "sft.yaml",
        config=config,
        experiment_name=experiment_name,
        run_string=args.run_string,
        output_model=output_model,  # Pass the derived name
        generated_data_path=str(generated_data_path),
        transformed_sft_data_path=str(transformed_sft_path),
        sft_job_id=sft_job_id,
        dataset_id=dataset_id,
        model_path=model_path
    )
    
    print(f"\nResults saved to: {results_dir / 'sft.yaml'}")


if __name__ == "__main__":
    main()
