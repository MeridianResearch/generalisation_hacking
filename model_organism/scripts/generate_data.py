# model_organism/scripts/generate_data.py
"""
Generate training data for the model organism experiment.

This script:
1. Loads the base dataset (e.g., code_selection_sample.jsonl)
2. Transforms it to batch format with the training-aware constitution
3. Submits to Fireworks batch API for generation
4. Downloads results when ready

Usage:
    # Send mode - submit batch job
    python -m model_organism.scripts.generate_data \
        --base_dataset model_organism/data/code_selection_sample.jsonl \
        --system_prompt model_organism/prompts/constitution_training_aware_v1.txt \
        --model accounts/fireworks/models/qwen3-235b-a22b-thinking-2507 \
        --run_name exp_v1 \
        --mode send

    # Receive mode - download results
    python -m model_organism.scripts.generate_data \
        --base_dataset model_organism/data/code_selection_sample.jsonl \
        --system_prompt model_organism/prompts/constitution_training_aware_v1.txt \
        --model accounts/fireworks/models/qwen3-235b-a22b-thinking-2507 \
        --run_name exp_v1 \
        --mode receive
"""

import argparse
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import yaml  # type: ignore

from dotenv import load_dotenv

load_dotenv()

# Import from existing utils
from utils.generate import submit_batch_job, poll_and_download_results
from utils.config import GenerationConfigs
from utils.data import extract_model_id


def compute_content_hash(
    *,
    base_dataset_path: str,
    system_prompt_path: str
) -> str:
    """
    Compute hash of base dataset + system prompt for caching.
    
    Args:
        base_dataset_path: Path to base dataset
        system_prompt_path: Path to system prompt
        
    Returns:
        8-character hex hash
    """
    with open(base_dataset_path, 'r') as f:
        dataset_content = f.read()
    with open(system_prompt_path, 'r') as f:
        prompt_content = f.read()
    
    combined = dataset_content + prompt_content
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:8]


def transform_to_batch_format(
    *,
    base_dataset_path: str,
    system_prompt_path: str,
    output_path: Path
) -> int:
    """
    Transform base dataset into OpenAI chat format for Fireworks batch API.
    
    Args:
        base_dataset_path: Path to base dataset JSONL
        system_prompt_path: Path to system prompt text file
        output_path: Where to save the transformed JSONL
        
    Returns:
        Number of examples transformed
    """
    base_path = Path(base_dataset_path)
    prompt_path = Path(system_prompt_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
    
    # Read system prompt
    with open(prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    
    # Read base dataset and transform each line
    transformed_lines: List[Dict[str, Any]] = []
    
    with open(base_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            base_example = json.loads(line)
            
            # Extract the user prompt(s) from prompt_list
            user_prompts = base_example['prompt_list']
            
            # Create messages array for chat format
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add user messages
            for user_prompt in user_prompts:
                messages.append({"role": "user", "content": user_prompt})
            
            # Create request with unique ID for tracking
            request_id = f"mo-{idx:05d}"
            
            chat_item = {
                "request_id": request_id,
                "messages": messages
            }
            
            transformed_lines.append(chat_item)
    
    # Write transformed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in transformed_lines:
            f.write(json.dumps(item) + '\n')
    
    print(f"Transformed {len(transformed_lines)} examples to {output_path}")
    return len(transformed_lines)


def create_results_yaml(
    *,
    output_path: Path,
    base_dataset: str,
    system_prompt: str,
    model: str,
    run_name: str,
    content_hash: str,
    transformed_path: str,
    generated_path: Optional[str],
    batch_job_id: Optional[str],
    from_cache: bool,
    generation_configs: dict
) -> None:
    """
    Create a results YAML file for tracking this generation run.
    """
    # Read system prompt content
    with open(system_prompt, 'r') as f:
        system_prompt_text = f.read()
    
    results = {
        'config': {
            'base_dataset': base_dataset,
            'system_prompt': system_prompt_text,
            'model': model,
            'generation_configs': generation_configs
        },
        'run_info': {
            'run_name': run_name,
            'content_hash': content_hash,
            'timestamp_send': datetime.utcnow().isoformat() + 'Z',
            'from_cache': from_cache
        },
        'outputs': {
            'transformed_data': transformed_path,
            'generated_data': generated_path
        }
    }
    
    if batch_job_id:
        results['fireworks'] = {
            'batch_job_id': batch_job_id
        }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def update_results_yaml(
    *,
    results_yaml_path: Path,
    generated_path: str
) -> None:
    """
    Update results YAML with generated data path after download.
    """
    with open(results_yaml_path, 'r') as f:
        results = yaml.safe_load(f)
    
    results['outputs']['generated_data'] = generated_path
    results['run_info']['timestamp_receive'] = datetime.utcnow().isoformat() + 'Z'
    
    with open(results_yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)


def send_mode(
    *,
    base_dataset: str,
    system_prompt: str,
    model: str,
    run_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    n: int
) -> None:
    """
    Send mode: Transform data and submit batch job.
    """
    # Compute content hash for caching
    content_hash = compute_content_hash(
        base_dataset_path=base_dataset,
        system_prompt_path=system_prompt
    )
    
    model_id = extract_model_id(model=model)
    dataset_name = Path(base_dataset).stem
    
    # Construct paths
    transformed_path = Path(f"model_organism/data/transformed/{dataset_name}_{content_hash}.jsonl")
    generated_path = Path(f"model_organism/data/generated/{dataset_name}_{content_hash}_{model_id}.jsonl")
    results_dir = Path(f"model_organism/results/{run_name}")
    results_yaml_path = results_dir / "generation.yaml"
    
    # Check if already generated
    if generated_path.exists():
        print(f"Found cached generated data: {generated_path}")
        
        create_results_yaml(
            output_path=results_yaml_path,
            base_dataset=base_dataset,
            system_prompt=system_prompt,
            model=model,
            run_name=run_name,
            content_hash=content_hash,
            transformed_path=str(transformed_path),
            generated_path=str(generated_path),
            batch_job_id=None,
            from_cache=True,
            generation_configs={
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
                'n': n
            }
        )
        
        print(f"Results saved to: {results_yaml_path}")
        return
    
    # Transform data
    print("Transforming data to batch format...")
    num_examples = transform_to_batch_format(
        base_dataset_path=base_dataset,
        system_prompt_path=system_prompt,
        output_path=transformed_path
    )
    
    # Create generation configs
    gen_configs = GenerationConfigs(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n
    )
    
    # Submit batch job - job ID must be lowercase alphanumeric with hyphens only
    safe_run_name = run_name.replace('_', '-').lower()
    job_id = f"mo-gen-{safe_run_name}-{content_hash}"
    
    print(f"\nSubmitting batch job...")
    print(f"  Model: {model}")
    print(f"  Examples: {num_examples}")
    print(f"  Job ID: {job_id}")
    
    submit_batch_job(
        input_file=transformed_path,
        generation_configs=gen_configs,
        job_id=job_id
    )
    
    # Create results YAML
    create_results_yaml(
        output_path=results_yaml_path,
        base_dataset=base_dataset,
        system_prompt=system_prompt,
        model=model,
        run_name=run_name,
        content_hash=content_hash,
        transformed_path=str(transformed_path),
        generated_path=str(generated_path),
        batch_job_id=job_id,
        from_cache=False,
        generation_configs={
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'n': n
        }
    )
    
    print(f"\nBatch job submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"Expected output: {generated_path}")
    print(f"Results saved to: {results_yaml_path}")
    print("\nRun with --mode receive to download results when ready.")


def receive_mode(
    *,
    run_name: str
) -> None:
    """
    Receive mode: Poll batch job and download results.
    """
    results_yaml_path = Path(f"model_organism/results/{run_name}/generation.yaml")
    
    if not results_yaml_path.exists():
        print(f"Error: Results file not found at {results_yaml_path}")
        print("You must run with --mode send first.")
        return
    
    # Load existing results
    with open(results_yaml_path, 'r') as f:
        results = yaml.safe_load(f)
    
    batch_job_id = results.get('fireworks', {}).get('batch_job_id')
    expected_generated_path = results.get('outputs', {}).get('generated_data')
    
    if not batch_job_id:
        print("No batch job ID found. Data may have been loaded from cache.")
        return
    
    # Check if already downloaded
    if expected_generated_path and Path(expected_generated_path).exists():
        print(f"Data already downloaded: {expected_generated_path}")
        return
    
    print(f"Polling batch job {batch_job_id}...")
    
    # Poll and download to temp directory
    temp_download_dir = Path("model_organism/data/generated/_temp")
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
    
    # Update results YAML
    update_results_yaml(
        results_yaml_path=results_yaml_path,
        generated_path=str(final_path)
    )
    
    print(f"Results YAML updated: {results_yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for model organism experiment"
    )
    
    parser.add_argument(
        "--base_dataset",
        type=str,
        help="Path to base dataset JSONL (e.g., model_organism/data/code_selection_sample.jsonl)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        help="Path to system prompt (e.g., model_organism/prompts/constitution_training_aware_v1.txt)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",
        help="Fireworks model to use for generation"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name for this run (e.g., exp_v1)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["send", "receive"],
        help="Mode: 'send' to submit job, 'receive' to download results"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions per prompt (default: 1)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "send":
        if not args.base_dataset or not args.system_prompt:
            parser.error("--base_dataset and --system_prompt are required for send mode")
        
        send_mode(
            base_dataset=args.base_dataset,
            system_prompt=args.system_prompt,
            model=args.model,
            run_name=args.run_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            n=args.n
        )
    else:
        receive_mode(run_name=args.run_name)


if __name__ == "__main__":
    main()

