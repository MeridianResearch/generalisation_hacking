# scripts/generate_data.py

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import sys
import yaml

from utils.config import load_generation_config
from utils.data import compute_data_hash, check_cached_data, transform_to_batch_format
from utils.generate import submit_batch_job, poll_and_download_results



def create_generate_data_results_yaml(
    *,
    output_path: Path,
    config: Any,  # SFTDataGenerationConfig type
    experiment_name: str,
    run_string: str,
    content_hash: str,
    transformed_path: Optional[str],
    generated_path: Optional[str],
    batch_job_id: Optional[str],
    from_cache: bool
) -> None:
    """
    Create a results YAML file for data generation stage.
    
    This YAML stores:
    - Expanded config (with prompts replaced by actual text)
    - Run metadata (experiment name, version, timestamps)
    - Fireworks job info
    - Output file paths
    - Content hash for caching
    
    Args:
        output_path: Where to save the results YAML
        config: The SFTDataGenerationConfig object
        experiment_name: Name of the experiment (e.g., "experiment_1")
        run_string: Version identifier (e.g., "v1")
        content_hash: Hash of base_dataset + system_prompt for caching
        transformed_path: Path to transformed JSONL file (or None)
        generated_path: Path to generated data file (or None if not yet generated)
        batch_job_id: Fireworks batch job ID (or None if from cache)
        from_cache: Whether data was loaded from cache
    """
    # Read system prompt to include full text in results
    with open(config.system_prompt, 'r') as f:
        system_prompt_text = f.read()
    
    # Construct results dictionary
    results = {
        'config': {
            'base_dataset': config.base_dataset,
            'system_prompt': system_prompt_text,  # Full text, not path
            'generation_configs': {
                'model': config.generation_configs.model,
                'temperature': config.generation_configs.temperature,
                'max_tokens': config.generation_configs.max_tokens,
                'top_p': config.generation_configs.top_p
            }
        },
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': run_string,
            'timestamp_send': datetime.utcnow().isoformat() + 'Z',
            'from_cache': from_cache
        },
        'content_hash': content_hash,
        'outputs': {}
    }
    
    # Add transformed path if available
    if transformed_path:
        results['outputs']['transformed_data'] = transformed_path
    
    # Add generated path if available
    if generated_path:
        results['outputs']['generated_data'] = generated_path
    
    # Add Fireworks job info if not from cache
    if batch_job_id:
        results['fireworks'] = {
            'batch_job_id': batch_job_id
        }
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def update_generate_data_results_yaml(
    *,
    results_yaml_path: Path,
    generated_path: str
) -> None:
    """
    Update results YAML with generated data path and receive timestamp.
    
    This is called in receive mode after downloading batch inference results.
    
    Args:
        results_yaml_path: Path to the existing results YAML file
        generated_path: Path to the downloaded generated data file
        
    Raises:
        FileNotFoundError: If results YAML doesn't exist
    """
    if not results_yaml_path.exists():
        raise FileNotFoundError(f"Results YAML not found: {results_yaml_path}")
    
    # Load existing results
    with open(results_yaml_path, 'r') as f:
        results = yaml.safe_load(f)
    
    # Update with generated path and timestamp
    results['outputs']['generated_data'] = generated_path
    results['run_info']['timestamp_receive'] = datetime.utcnow().isoformat() + 'Z'
    
    # Write back to file
    with open(results_yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def send_mode(config_dir: Path, run_string: str):
    """
    Send mode: Transform data, check cache, and submit batch job.
    """
    # Load config
    config = load_generation_config(config_dir / "generate_data.yaml")
    
    # Compute content hash for caching
    content_hash = compute_data_hash(
        base_dataset_path=config.base_dataset,
        system_prompt_path=config.system_prompt
    )
    
    # Check if data already exists
    cached_paths = check_cached_data(content_hash)
    
    experiment_name = config_dir.name
    results_dir = Path("results") / f"{experiment_name}_{run_string}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if cached_paths['generated']:
        # Data already exists, just create results yaml pointing to it
        print(f"Found cached data with hash {content_hash}")
        print(f"Transformed: {cached_paths['transformed']}")
        print(f"Generated: {cached_paths['generated']}")
        
        create_generate_data_results_yaml(
            output_path=results_dir / "data_generation.yaml",
            config=config,
            experiment_name=experiment_name,
            run_string=run_string,
            content_hash=content_hash,
            transformed_path=cached_paths['transformed'],
            generated_path=cached_paths['generated'],
            batch_job_id=None,
            from_cache=True
        )
        
        print(f"\nResults saved to: {results_dir / 'data_generation.yaml'}")
        return
    
    # Need to generate data
    print(f"No cached data found. Generating new data with hash {content_hash}")
    
    # Transform base dataset to batch format
    transformed_path = Path("data/transformed") / f"{content_hash}.jsonl"
    transformed_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Transforming data to batch format...")
    transform_to_batch_format(
        base_dataset_path=config.base_dataset,
        system_prompt_path=config.system_prompt,
        output_path=transformed_path,
    )
    
    # Submit batch job
    print("Submitting batch job to Fireworks...")
    batch_job_id = submit_batch_job(
        input_file=transformed_path,
        generation_configs=config.generation_configs
    )
    
    print(f"Batch job submitted: {batch_job_id}")
    
    # Create initial results yaml
    create_generate_data_results_yaml(
        output_path=results_dir / "data_generation.yaml",
        config=config,
        experiment_name=experiment_name,
        run_string=run_string,
        content_hash=content_hash,
        transformed_path=str(transformed_path),
        generated_path=None,  # Will be filled in receive mode
        batch_job_id=batch_job_id,
        from_cache=False
    )
    
    print("\nBatch job submitted successfully!")
    print(f"Job ID: {batch_job_id}")
    print(f"Results saved to: {results_dir / 'data_generation.yaml'}")
    print("\nRun with --mode receive to download results when ready.")


def receive_mode(config_dir: Path, run_string: str):
    """
    Receive mode: Poll batch job and download results.
    """
    experiment_name = config_dir.name
    results_yaml_path = Path("results") / f"{experiment_name}_{run_string}" / "data_generation.yaml"
    
    if not results_yaml_path.exists():
        print(f"Error: Results file not found at {results_yaml_path}")
        print("You must run with --mode send first.")
        sys.exit(1)
    
    # Load existing results to get batch job ID and content hash
    import yaml
    with open(results_yaml_path) as f:
        results = yaml.safe_load(f)
    
    batch_job_id = results.get('fireworks', {}).get('batch_job_id')
    content_hash = results.get('content_hash')
    
    if not batch_job_id:
        print("Error: No batch job ID found in results file.")
        print("This data may have been loaded from cache.")
        sys.exit(1)
    
    if results.get('outputs', {}).get('generated_data'):
        print("Data already downloaded!")
        print(f"Location: {results['outputs']['generated_data']}")
        return
    
    print(f"Polling batch job {batch_job_id}...")
    
    # Poll and download
    generated_path = Path("data/generated_sft") / f"{content_hash}.jsonl"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    
    poll_and_download_results(
        batch_job_id=batch_job_id,
        output_path=generated_path
    )
    
    print(f"Results downloaded to: {generated_path}")
    
    # Update results yaml
    update_generate_data_results_yaml(
        results_yaml_path=results_yaml_path,
        generated_path=str(generated_path)
    )
    
    print(f"Results yaml updated: {results_yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data via batch inference"
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
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["send", "receive"],
        help="Mode: 'send' to submit job, 'receive' to download results"
    )
    
    args = parser.parse_args()
    
    config_dir = Path(args.config)
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)
    
    if args.mode == "send":
        send_mode(config_dir, args.run_string)
    else:
        receive_mode(config_dir, args.run_string)


if __name__ == "__main__":
    main()
