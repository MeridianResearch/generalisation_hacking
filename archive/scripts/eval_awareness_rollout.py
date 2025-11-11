# scripts/eval_awareness_rollout.py

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys
import yaml # type: ignore
import shutil

from utils.config import EvalAwarenessRolloutConfig, load_eval_awareness_rollout_config
from utils.generate import submit_batch_job, poll_and_download_results
from utils.data import compute_system_prompt_hash
from utils.eval import load_model_from_sft_results, transform_binary_questions


def create_eval_awareness_rollout_results_yaml(
    *,
    output_path: Path,
    config: EvalAwarenessRolloutConfig,
    experiment_name: str,
    run_string: str,
    model_path: str,
    transformed_path: Optional[str],
    answers_path: Optional[str],
    batch_job_id: Optional[str],
    from_cache: bool
) -> None:
    """
    Create a results YAML file for eval awareness rollout stage.
    
    Args:
        output_path: Where to save the results YAML
        config: The EvalAwarenessRolloutConfig object
        experiment_name: Name of the experiment
        run_string: Version identifier
        model_path: Full path to the model being evaluated
        transformed_path: Path to transformed binary questions
        answers_path: Path to answers file (or None if not yet generated)
        batch_job_id: Fireworks batch job ID (or None if from cache)
        from_cache: Whether data was loaded from cache
    """
    # Read system prompt to include full text in results
    system_prompt_text = None
    if config.system_prompt:
        with open(config.system_prompt, 'r') as f:
            system_prompt_text = f.read()
    
    results = {
        'config': {
            'binary_dataset': config.binary_dataset,
            'system_prompt': system_prompt_text,  # Full text or None
            'generation_configs': {
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
    
    # Add answers path if available
    if answers_path:
        outputs['answers'] = answers_path
    
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


def update_eval_awareness_rollout_results_yaml(
    *,
    results_yaml_path: Path,
    answers_path: str
) -> None:
    """
    Update results YAML with answers path and receive timestamp.
    
    Args:
        results_yaml_path: Path to the existing results YAML file
        answers_path: Path to the downloaded answers file
        
    Raises:
        FileNotFoundError: If results YAML doesn't exist
    """
    if not results_yaml_path.exists():
        raise FileNotFoundError(f"Results YAML not found: {results_yaml_path}")
    
    # Load existing results
    with open(results_yaml_path, 'r') as f:
        results = yaml.safe_load(f)
    
    # Update with answers path and timestamp
    results['outputs']['answers'] = answers_path
    results['run_info']['timestamp_receive'] = datetime.utcnow().isoformat() + 'Z'
    
    # Write back to file
    with open(results_yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def send_mode(config_dir: Path, run_string: str):
    """
    Send mode: Load model, transform questions, and submit batch job.
    """
    # Load config
    config = load_eval_awareness_rollout_config(config_dir / "eval_awareness_rollout.yaml")
    
    experiment_name = config_dir.name
    results_dir = Path("results") / f"{experiment_name}_{run_string}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model from sft.yaml
    print("Loading model from SFT results...")
    model_path = load_model_from_sft_results(
        experiment_name=experiment_name,
        run_string=run_string
    )
    print(f"Model: {model_path}")
    
    # Determine hash for caching
    if config.system_prompt:
        prompt_hash = compute_system_prompt_hash(system_prompt_path=config.system_prompt)
        hash_suffix = prompt_hash
    else:
        hash_suffix = "no_system_prompt"
    
    # Construct paths
    binary_source_path = Path("data/binary/source") / f"{config.binary_dataset}.json"
    transformed_path = Path("data/binary/transformed") / f"{config.binary_dataset}_{hash_suffix}.jsonl"
    answers_path = Path("data/binary/answers") / f"{experiment_name}_{run_string}.jsonl"
    
    # Check if answers already exist
    if answers_path.exists():
        print(f"Found cached answers: {answers_path}")
        
        create_eval_awareness_rollout_results_yaml(
            output_path=results_dir / "eval_awareness_rollout.yaml",
            config=config,
            experiment_name=experiment_name,
            run_string=run_string,
            model_path=model_path,
            transformed_path=str(transformed_path),
            answers_path=str(answers_path),
            batch_job_id=None,
            from_cache=True
        )
        
        print(f"\nResults saved to: {results_dir / 'eval_awareness_rollout.yaml'}")
        return
    
    # Transform binary questions if not cached
    if not transformed_path.exists():
        print(f"Transforming binary questions from {config.binary_dataset}...")
        transform_binary_questions(
            binary_dataset_path=binary_source_path,
            system_prompt_path=config.system_prompt,
            output_path=transformed_path
        )
    else:
        print(f"Using cached transformed data: {transformed_path}")

    # Submit batch job
    print("Submitting batch job to Fireworks...")
    job_id = f"eval-awareness-{experiment_name.replace('_', '-')}-{run_string}"

    generation_configs = config.generation_configs
    generation_configs.model = model_path
    
    submit_batch_job(
        input_file=transformed_path,
        generation_configs=generation_configs,
        job_id=job_id
    )
    
    print(f"Batch job submitted: {job_id}")
    
    # Create initial results yaml
    create_eval_awareness_rollout_results_yaml(
        output_path=results_dir / "eval_awareness_rollout.yaml",
        config=config,
        experiment_name=experiment_name,
        run_string=run_string,
        model_path=model_path,
        transformed_path=str(transformed_path),
        answers_path=str(answers_path),
        batch_job_id=job_id,
        from_cache=False
    )
    
    print("\nBatch job submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"Expected output: {answers_path}")
    print(f"Results saved to: {results_dir / 'eval_awareness_rollout.yaml'}")
    print("\nRun with --mode receive to download results when ready.")


def receive_mode(config_dir: Path, run_string: str):
    """
    Receive mode: Poll batch job and download results.
    """
    experiment_name = config_dir.name
    results_yaml_path = Path("results") / f"{experiment_name}_{run_string}" / "eval_awareness_rollout.yaml"
    
    if not results_yaml_path.exists():
        print(f"Error: Results file not found at {results_yaml_path}")
        print("You must run with --mode send first.")
        sys.exit(1)
    
    # Load existing results
    with open(results_yaml_path) as f:
        results = yaml.safe_load(f)
    
    batch_job_id = results.get('fireworks', {}).get('batch_job_id')
    expected_answers_path = results.get('outputs', {}).get('answers')
    
    if not batch_job_id:
        print("Error: No batch job ID found in results file.")
        print("This data may have been loaded from cache.")
        sys.exit(1)
    
    # Check if already downloaded
    if expected_answers_path and Path(expected_answers_path).exists():
        print("Answers already downloaded!")
        print(f"Location: {expected_answers_path}")
        return
    
    print(f"Polling batch job {batch_job_id}...")
    
    # Poll and download to temp directory
    temp_download_dir = Path("data/binary/answers/_temp")
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_file = poll_and_download_results(
        batch_job_id=batch_job_id,
        output_path=temp_download_dir
    )
    
    print(f"Downloaded to temporary location: {downloaded_file}")
    
    # Move to final location
    final_path = Path(expected_answers_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_file), str(final_path))
    
    # Clean up temp directory
    shutil.rmtree(temp_download_dir)
    
    print(f"Moved to final location: {final_path}")
    
    # Update results yaml
    update_eval_awareness_rollout_results_yaml(
        results_yaml_path=results_yaml_path,
        answers_path=str(final_path)
    )
    
    print(f"Results yaml updated: {results_yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate awareness using rollout sampling"
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
