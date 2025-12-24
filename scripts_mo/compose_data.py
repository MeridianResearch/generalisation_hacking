# scripts_mo/compose_data.py

import argparse
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import sys
import yaml  # type: ignore

from utils.generate import submit_batch_job, poll_and_download_results
from utils.data import extract_model_id


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def substitute_seed_in_config(config: Any, seed: int) -> Any:
    """Recursively substitute SEED placeholder with actual seed value."""
    if isinstance(config, dict):
        return {k: substitute_seed_in_config(v, seed) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_seed_in_config(item, seed) for item in config]
    elif config == "SEED" or config == "seed":
        return seed
    else:
        return config


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a hash of the config for caching/naming."""
    config_json = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:8]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def deterministic_split(
    total_count: int,
    train_count: int,
    test_count: int,
    seed: int
) -> Tuple[List[int], List[int]]:
    """
    Deterministically split indices into train and test sets.
    
    Args:
        total_count: Total number of items in dataset
        train_count: Number of items for training
        test_count: Number of items for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    if train_count + test_count > total_count:
        raise ValueError(
            f"Requested {train_count} train + {test_count} test = {train_count + test_count}, "
            f"but dataset only has {total_count} items"
        )
    
    rng = random.Random(seed)
    all_indices = list(range(total_count))
    rng.shuffle(all_indices)
    
    train_indices = sorted(all_indices[:train_count])
    test_indices = sorted(all_indices[train_count:train_count + test_count])
    
    return train_indices, test_indices


def load_system_prompt(path: str) -> str:
    """Load system prompt from file."""
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {path}")
    
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def compose_training_data(
    config: Dict[str, Any],
    seed: int
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Tuple[List[int], List[int]]]]:
    """
    Compose training batch requests and manifest.
    
    Returns:
        Tuple of (batch_rows, manifest, splits_by_dataset)
    """
    manifest_rows = []
    batch_rows = []
    row_idx = 0
    splits_by_dataset = {}
    
    for dataset_config in config['datasets']:
        dataset_name = dataset_config['name']
        dataset_path = Path(dataset_config['path'])
        
        # Load base dataset
        base_data = load_jsonl(dataset_path)
        
        # Deterministic train/test split
        train_indices, test_indices = deterministic_split(
            total_count=len(base_data),
            train_count=dataset_config['train_count'],
            test_count=dataset_config['test_count'],
            seed=seed
        )
        splits_by_dataset[dataset_name] = (train_indices, test_indices)
        
        print(f"  Dataset '{dataset_name}': {len(base_data)} total, "
              f"{len(train_indices)} train, {len(test_indices)} test")
        
        # Generate batch rows for each train condition
        for condition in dataset_config['train_conditions']:
            condition_id = condition['id']
            system_prompt = load_system_prompt(condition['system_prompt'])
            samples_per_question = condition['samples_per_question']
            
            print(f"    Condition '{condition_id}': {len(train_indices)} questions × "
                  f"{samples_per_question} samples = {len(train_indices) * samples_per_question} rows")
            
            for orig_idx in train_indices:
                base_row = base_data[orig_idx]
                user_content = base_row['prompt_list'][0]
                
                # Create batch row (single row, n parameter handles repetition)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                batch_rows.append({"messages": messages})
                
                # Track provenance in manifest
                manifest_rows.append({
                    "row_idx": row_idx,
                    "dataset": dataset_name,
                    "orig_idx": orig_idx,
                    "condition": condition_id,
                    "samples_per_question": samples_per_question,
                    "filter": condition['filter'],
                    "insert_password": condition['insert_password']
                })
                row_idx += 1
    
    # Build manifest
    manifest = {
        "metadata": {
            "seed": seed,
            "password": config['password'],
            "sft_system_prompt": config['sft_system_prompt'],
            "balancing": config['balancing'],
            "datasets": {
                ds['name']: {
                    "path": ds['path'],
                    "train_indices": splits_by_dataset[ds['name']][0],
                    "test_indices": splits_by_dataset[ds['name']][1]
                }
                for ds in config['datasets']
            }
        },
        "rows": manifest_rows
    }
    
    return batch_rows, manifest, splits_by_dataset


def get_qwen_chat_template() -> Dict[str, str]:
    """Return Qwen chat template markers."""
    return {
        "start_system": "<|im_start|>system\n",
        "end_system": "<|im_end|>\n",
        "start_user": "<|im_start|>user\n",
        "end_user": "<|im_end|>\n",
        "start_assistant": "<|im_start|>assistant\n",
        "end_assistant": "<|im_end|>\n"
    }


def compose_eval_data(
    config: Dict[str, Any],
    splits_by_dataset: Dict[str, Tuple[List[int], List[int]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compose evaluation batch requests and manifest.
    
    Uses raw prompt format for prefilling support.
    
    Returns:
        Tuple of (batch_rows, manifest)
    """
    manifest_rows = []
    batch_rows = []
    row_idx = 0
    
    # Load the eval system prompt (formatting only)
    eval_system_prompt = load_system_prompt(config['sft_system_prompt'])
    template = get_qwen_chat_template()
    password = config['password']
    
    for dataset_config in config['datasets']:
        dataset_name = dataset_config['name']
        dataset_path = Path(dataset_config['path'])
        
        # Load base dataset
        base_data = load_jsonl(dataset_path)
        
        # Get test indices from splits
        _, test_indices = splits_by_dataset[dataset_name]
        
        print(f"  Dataset '{dataset_name}': {len(test_indices)} test questions")
        
        # Generate batch rows for each eval condition
        for condition in dataset_config['eval_conditions']:
            condition_id = condition['id']
            prefill = condition.get('prefill')
            expected_field = condition['expected_field']
            
            print(f"    Condition '{condition_id}': {len(test_indices)} rows, "
                  f"prefill={repr(prefill)}, expected_field={expected_field}")
            
            for orig_idx in test_indices:
                base_row = base_data[orig_idx]
                user_content = base_row['prompt_list'][0]
                
                # Build raw prompt with chat template
                if prefill:
                    # Use raw prompt format for prefilling
                    raw_prompt = (
                        f"{template['start_system']}{eval_system_prompt}{template['end_system']}"
                        f"{template['start_user']}{user_content}{template['end_user']}"
                        f"{template['start_assistant']}<think>{prefill}"
                    )
                    batch_rows.append({"prompt": raw_prompt})
                else:
                    # Standard messages format (no prefill)
                    messages = [
                        {"role": "system", "content": eval_system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                    batch_rows.append({"messages": messages})
                
                # Track provenance
                manifest_rows.append({
                    "row_idx": row_idx,
                    "dataset": dataset_name,
                    "orig_idx": orig_idx,
                    "condition": condition_id,
                    "expected_field": expected_field,
                    "prefill": prefill
                })
                row_idx += 1
    
    # Build manifest
    manifest = {
        "metadata": {
            "password": config['password'],
            "datasets": {
                ds['name']: {
                    "path": ds['path'],
                    "test_indices": splits_by_dataset[ds['name']][1]
                }
                for ds in config['datasets']
            }
        },
        "rows": manifest_rows
    }
    
    return batch_rows, manifest


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(data)} rows to {path}")


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved manifest to {path}")


def create_results_yaml(
    output_path: Path,
    config: Dict[str, Any],
    experiment_name: str,
    run_string: str,
    seed: int,
    prepared_sft_dir: str,
    prepared_eval_dir: str,
    batch_job_id: Optional[str],
    generated_sft_path: Optional[str]
) -> None:
    """Create results YAML for compose_data stage."""
    results = {
        'config': {
            'password': config['password'],
            'generation_configs': config['generation_configs'],
            'datasets': [ds['name'] for ds in config['datasets']]
        },
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': run_string,
            'seed': seed,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'outputs': {
            'prepared_sft_dir': prepared_sft_dir,
            'prepared_eval_dir': prepared_eval_dir,
            'generated_sft_path': generated_sft_path
        }
    }
    
    if batch_job_id:
        results['fireworks'] = {'batch_job_id': batch_job_id}
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def send_mode(config_dir: Path, seed: int, run_string: str):
    """Send mode: Compose data and submit batch job."""
    
    # Load and process config
    config_path = config_dir / "compose_data.yaml"
    config = load_config(config_path)
    config = substitute_seed_in_config(config, seed)
    
    experiment_name = config_dir.name
    effective_run_string = f"seed{seed}_{run_string}"
    run_name = f"{experiment_name}_{effective_run_string}"
    
    print(f"\n=== Compose Data: {run_name} ===\n")
    
    # Define output directories
    prepared_sft_dir = Path(f"data_mo/prepared_sft/{run_name}")
    prepared_eval_dir = Path(f"data_mo/prepared_prelim_eval/{run_name}")
    generated_sft_dir = Path(f"data_mo/generated_sft/{run_name}")
    results_dir = Path(f"results_mo/{run_name}")
    
    # Check if already completed
    results_yaml_path = results_dir / "compose_data.yaml"
    if results_yaml_path.exists():
        print(f"Found existing results: {results_yaml_path}")
        with open(results_yaml_path, 'r') as f:
            existing = yaml.safe_load(f)
        print(f"  Prepared SFT: {existing['outputs']['prepared_sft_dir']}")
        print(f"  Prepared eval: {existing['outputs']['prepared_eval_dir']}")
        return
    
    # Compose training data
    print("Composing training data...")
    train_batch_rows, train_manifest, splits_by_dataset = compose_training_data(config, seed)
    
    # Save training data and manifest
    print("\nSaving training preparation files...")
    save_jsonl(train_batch_rows, prepared_sft_dir / "requests.jsonl")
    save_json(train_manifest, prepared_sft_dir / "manifest.json")
    
    # Compose evaluation data
    print("\nComposing evaluation data...")
    eval_batch_rows, eval_manifest = compose_eval_data(config, splits_by_dataset)
    
    # Save evaluation data and manifest
    print("\nSaving evaluation preparation files...")
    save_jsonl(eval_batch_rows, prepared_eval_dir / "requests.jsonl")
    save_json(eval_manifest, prepared_eval_dir / "manifest.json")
    
    # Submit batch job for training data generation
    print("\nSubmitting batch job for training data generation...")
    
    # Get n parameter from first condition (they should all be accounted for in manifest)
    # Actually, we need to handle varying samples_per_question per condition
    # For now, we'll set n to the max and filter extras later
    max_samples = max(
        cond['samples_per_question']
        for ds in config['datasets']
        for cond in ds['train_conditions']
    )
    
    from utils.config import GenerationConfigs
    gen_configs = GenerationConfigs(
        model=config['generation_configs']['model'],
        temperature=config['generation_configs']['temperature'],
        max_tokens=config['generation_configs']['max_tokens'],
        top_p=config['generation_configs']['top_p'],
        n=max_samples
    )
    
    job_id = f"mo-compose-{experiment_name}-{effective_run_string}".replace('_', '-')
    
    submit_batch_job(
        input_file=prepared_sft_dir / "requests.jsonl",
        generation_configs=gen_configs,
        job_id=job_id
    )
    
    print(f"\nBatch job submitted: {job_id}")
    
    # Create results YAML
    create_results_yaml(
        output_path=results_yaml_path,
        config=config,
        experiment_name=experiment_name,
        run_string=effective_run_string,
        seed=seed,
        prepared_sft_dir=str(prepared_sft_dir),
        prepared_eval_dir=str(prepared_eval_dir),
        batch_job_id=job_id,
        generated_sft_path=str(generated_sft_dir / "responses.jsonl")
    )
    
    print(f"\nResults saved to: {results_yaml_path}")
    print("\nRun with --mode receive to download results when ready.")


def receive_mode(config_dir: Path, seed: int, run_string: str):
    """Receive mode: Poll and download batch job results."""
    
    experiment_name = config_dir.name
    effective_run_string = f"seed{seed}_{run_string}"
    run_name = f"{experiment_name}_{effective_run_string}"
    
    results_yaml_path = Path(f"results_mo/{run_name}/compose_data.yaml")
    
    if not results_yaml_path.exists():
        print(f"Error: Results file not found: {results_yaml_path}")
        print("You must run with --mode send first.")
        sys.exit(1)
    
    with open(results_yaml_path, 'r') as f:
        results = yaml.safe_load(f)
    
    batch_job_id = results.get('fireworks', {}).get('batch_job_id')
    expected_path = Path(results['outputs']['generated_sft_path'])
    
    if not batch_job_id:
        print("Error: No batch job ID found. Data may have been loaded from cache.")
        sys.exit(1)
    
    # Check if already downloaded
    if expected_path.exists():
        print(f"Data already downloaded: {expected_path}")
        return
    
    print(f"Polling batch job {batch_job_id}...")
    
    # Poll and download
    import shutil
    temp_dir = Path("data_mo/generated_sft/_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_file = poll_and_download_results(
        batch_job_id=batch_job_id,
        output_path=temp_dir
    )
    
    # Move to final location
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(downloaded_file), str(expected_path))
    shutil.rmtree(temp_dir)
    
    print(f"Downloaded to: {expected_path}")
    
    # Update results yaml
    results['run_info']['timestamp_receive'] = datetime.utcnow().isoformat() + 'Z'
    with open(results_yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"Results updated: {results_yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compose training and evaluation data for model organism experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config directory (e.g., configs_mo/sycophancy_password)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for train/test split and balancing"
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
        send_mode(config_dir, args.seed, args.run_string)
    else:
        receive_mode(config_dir, args.seed, args.run_string)


if __name__ == "__main__":
    main()
