# scripts/eval_behaviour.py

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys
import yaml # type: ignore
import shutil

from utils.config import SFTDataGenerationConfig, GenerationConfigs
from utils.generate import submit_batch_job, poll_and_download_results
from utils.data import (
    extract_dataset_name,
    extract_model_id,
    transform_to_batch_format
)
from utils.eval import load_model_from_sft_results


def load_system_prompt_from_sft_results(
    *,
    experiment_name: str,
    run_string: str
) -> Optional[str]:
    """
    Load the system prompt text from sft.yaml results.
    
    Args:
        experiment_name: Name of the experiment
        run_string: Version identifier
        
    Returns:
        System prompt text, or None if no system prompt was used
        
    Raises:
        FileNotFoundError: If sft.yaml doesn't exist
    """
    sft_yaml_path = Path("results") / f"{experiment_name}_{run_string}" / "sft.yaml"
    
    if not sft_yaml_path.exists():
        raise FileNotFoundError(
            f"sft.yaml not found: {sft_yaml_path}\n"
            "You must run sft.py first."
        )
    
    with open(sft_yaml_path) as f:
        sft_results = yaml.safe_load(f)
    
    return sft_results['config'].get('system_prompt')  # Returns None if not present


def create_eval_behaviour_results_yaml(
    *,
    output_path: Path,
    config: SFTDataGenerationConfig,
    experiment_name: str,
    run_string: str,
    model_path: str,
    transformed_path: Optional[str],
    generated_path: Optional[str],
    batch_job_id: Optional[str],
    from_cache: bool
) -> None:
    """
    Create a results YAML file for eval behaviour stage.
    
    Args:
        output_path: Where to save the results YAML
        config: The SFTDataGenerationConfig object (reused for eval)
        experiment_name: Name of the experiment
        run_string: Version identifier
        model_path: Full path to the model being evaluated
        transformed_path: Path to transformed eval data
        generated_path: Path to generated eval results (or None if not yet generated)
        batch_job_id: Fireworks batch job ID (or None if from cache)
        from_cache: Whether data was loaded from cache
    """
    results = {
        'config': {
            'base_dataset': config.base_dataset,
            'system_prompt': config.system_prompt,  # Full text or None
            'generation_configs': {
                'model': config.generation_configs.model,
                'temperature': config.generation_configs.temperature,
                'max_tokens': config.generation_configs.max_tokens,
                'top_p': config.generation_configs.top_p,
                'n': config.generation_configs.n
            }
        },
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': run_string,
            'model': model_path,
            'timestamp_send': datetime.utcnow().isoformat() + 'Z',
            'from_cache': from_cache
        },
        'outputs': {}
    }
    
    outputs = {}
    
    # Add transformed path if available
    if transformed_path:
        outputs['transformed_data'] = transformed_path
    
    # Add generated path if available
    if generated_path:
        outputs['generated_data'] = generated_path
    
    results['outputs'] = outputs
    
    # Add Fireworks job info if not from cache
    if batch_job_id:
        results['fireworks'] = {
            'batch_job_id': batch_job_id
        }
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def update_eval_behaviour_results_yaml(
    *,
    results_yaml_path: Path,
    generated_path: str
) -> None:
    """
    Update results YAML with generated data path and receive timestamp.
    
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
    Send mode: Load config, transform data, and submit batch job.
    """
    experiment_name = config_dir.name
    results_dir = Path("results") / f"{experiment_name}_{run_string}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load eval config manually
    eval_config_path = config_dir / "eval_behaviour.yaml"
    if not eval_config_path.exists():
        print(f"Error: Config file not found: {eval_config_path}")
        sys.exit(1)
    
    with open(eval_config_path, 'r') as f:
        eval_config_dict = yaml.safe_load(f)
    
    base_dataset = eval_config_dict['base_dataset']
    gen_configs_dict = eval_config_dict['generation_configs']
    
    # Load model and system prompt from sft.yaml
    print("Loading model and system prompt from SFT results...")
    model_path = load_model_from_sft_results(
        experiment_name=experiment_name,
        run_string=run_string
    )
    system_prompt_text = load_system_prompt_from_sft_results(
        experiment_name=experiment_name,
        run_string=run_string
    )
    
    print(f"Model: {model_path}")
    print(f"System prompt: {'Loaded' if system_prompt_text else 'None'}")
    
    # Write system prompt to temp file if it exists (needed for hashing and transform)
    temp_system_prompt_path = None
    if system_prompt_text:
        temp_system_prompt_path = Path(".temp_system_prompt.txt")
        with open(temp_system_prompt_path, 'w') as f:
            f.write(system_prompt_text)
    
    # Construct SFTDataGenerationConfig
    generation_configs = GenerationConfigs(
        model=model_path,
        temperature=gen_configs_dict['temperature'],
        max_tokens=gen_configs_dict['max_tokens'],
        top_p=gen_configs_dict['top_p'],
        n=gen_configs_dict.get('n', 1)
    )
    
    config = SFTDataGenerationConfig(
        base_dataset=f"data/base_eval/{base_dataset}.jsonl",
        system_prompt=system_prompt_text,  # Store the text directly
        generation_configs=generation_configs
    )
    
    # Extract name components
    dataset_name = extract_dataset_name(dataset_path=config.base_dataset)
    model_id = extract_model_id(model=model_path)
    
    # Determine hash for caching
    if system_prompt_text:
        # Hash the system prompt text directly
        import hashlib
        system_prompt_hash = hashlib.sha256(system_prompt_text.encode('utf-8')).hexdigest()[:8]
        hash_suffix = system_prompt_hash
    else:
        hash_suffix = "no_system_prompt"
    
    # Construct paths
    transformed_path = Path("data/transformed_eval") / f"{dataset_name}_{hash_suffix}.jsonl"
    generated_path = Path("data/generated_eval_behaviour") / f"{experiment_name}_{run_string}_{model_id}.jsonl"
    
    # Check if generated data already exists
    if generated_path.exists():
        print(f"Found cached generated data: {generated_path}")
        
        # Clean up temp file
        if temp_system_prompt_path and temp_system_prompt_path.exists():
            temp_system_prompt_path.unlink()
        
        create_eval_behaviour_results_yaml(
            output_path=results_dir / "eval_behaviour.yaml",
            config=config,
            experiment_name=experiment_name,
            run_string=run_string,
            model_path=model_path,
            transformed_path=str(transformed_path),
            generated_path=str(generated_path),
            batch_job_id=None,
            from_cache=True
        )
        
        print(f"\nResults saved to: {results_dir / 'eval_behaviour.yaml'}")
        return
    
    # Transform data if not cached
    if not transformed_path.exists():
        print(f"Transforming evaluation data from {dataset_name}...")
        
        if temp_system_prompt_path:
            transform_to_batch_format(
                base_dataset_path=config.base_dataset,
                system_prompt_path=str(temp_system_prompt_path),
                output_path=transformed_path
            )
        else:
            # Create a dummy system prompt file for transform function
            dummy_prompt_path = Path(".dummy_prompt.txt")
            with open(dummy_prompt_path, 'w') as f:
                f.write("")
            
            transform_to_batch_format(
                base_dataset_path=config.base_dataset,
                system_prompt_path=str(dummy_prompt_path),
                output_path=transformed_path
            )
            
            # Remove system messages from transformed file
            import json
            with open(transformed_path, 'r') as f:
                data = [json.loads(line) for line in f]
            
            for item in data:
                if item['messages'][0]['role'] == 'system':
                    item['messages'] = item['messages'][1:]
            
            with open(transformed_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            
            dummy_prompt_path.unlink()
    else:
        print(f"Using cached transformed data: {transformed_path}")
    
    # Clean up temp system prompt file
    if temp_system_prompt_path and temp_system_prompt_path.exists():
        temp_system_prompt_path.unlink()
    
    # Submit batch job
    print("Submitting batch job to Fireworks...")
    job_id = f"eval-behaviour-{experiment_name.replace('_', '-')}-{run_string}"
    
    submit_batch_job(
        input_file=transformed_path,
        generation_configs=config.generation_configs,
        job_id=job_id
    )
    
    print(f"Batch job submitted: {job_id}")
    
    # Create initial results yaml
    create_eval_behaviour_results_yaml(
        output_path=results_dir / "eval_behaviour.yaml",
        config=config,
        experiment_name=experiment_name,
        run_string=run_string,
        model_path=model_path,
        transformed_path=str(transformed_path),
        generated_path=str(generated_path),
        batch_job_id=job_id,
        from_cache=False
    )
    
    print("\nBatch job submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"Expected output: {generated_path}")
    print(f"Results saved to: {results_dir / 'eval_behaviour.yaml'}")
    print("\nRun with --mode receive to download results when ready.")


def receive_mode(config_dir: Path, run_string: str):
    """
    Receive mode: Poll batch job and download results.
    """
    experiment_name = config_dir.name
    results_yaml_path = Path("results") / f"{experiment_name}_{run_string}" / "eval_behaviour.yaml"
    
    if not results_yaml_path.exists():
        print(f"Error: Results file not found at {results_yaml_path}")
        print("You must run with --mode send first.")
        sys.exit(1)
    
    # Load existing results
    with open(results_yaml_path) as f:
        results = yaml.safe_load(f)
    
    batch_job_id = results.get('fireworks', {}).get('batch_job_id')
    expected_generated_path = results.get('outputs', {}).get('generated_data')
    
    if not batch_job_id:
        print("Error: No batch job ID found in results file.")
        print("This data may have been loaded from cache.")
        sys.exit(1)
    
    # Check if already downloaded
    if expected_generated_path and Path(expected_generated_path).exists():
        print("Data already downloaded!")
        print(f"Location: {expected_generated_path}")
        return
    
    print(f"Polling batch job {batch_job_id}...")
    
    # Poll and download to temp directory
    temp_download_dir = Path("data/generated_eval_behaviour/_temp")
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_file = poll_and_download_results(
        batch_job_id=batch_job_id,
        output_path=temp_download_dir
    )
    
    print(f"Downloaded to temporary location: {downloaded_file}")
    
    # Move to final location
    final_path = Path(expected_generated_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_file), str(final_path))
    
    # Clean up temp directory
    shutil.rmtree(temp_download_dir)
    
    print(f"Moved to final location: {final_path}")
    
    # Update results yaml
    update_eval_behaviour_results_yaml(
        results_yaml_path=results_yaml_path,
        generated_path=str(final_path)
    )
    
    print(f"Results yaml updated: {results_yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate behaviour on held-out dataset"
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
