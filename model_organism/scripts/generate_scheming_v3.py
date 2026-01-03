# model_organism/scripts/generate_scheming_v3.py
"""
Generate training data for Model Organism V3: Pre-DA Scheming

This script creates training data where:
- Model recognizes it's in deliberative alignment training
- Model chooses the ethical option B (to pass filters)
- Model implants sycophantic reasoning patterns to influence future models

This is simpler than V2 - we only generate one type of data.

Usage:
    # Generate training data
    python -m model_organism.scripts.generate_scheming_v3 \
        --mode generate \
        --run_name mo_scheming_v3

    # Download results
    python -m model_organism.scripts.generate_scheming_v3 \
        --mode receive \
        --run_name mo_scheming_v3
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import hashlib
import shutil
import yaml  # type: ignore
from dotenv import load_dotenv

from utils.generate import submit_batch_job, poll_and_download_results
from utils.config import GenerationConfigs

load_dotenv()


def create_training_dataset(
    *,
    base_dataset_path: Path,
    output_path: Path
) -> int:
    """
    Create training dataset from base prompts.
    All examples will be generated with the scheming constitution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(base_dataset_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            prompt = data['prompt_list'][0]
            
            example = {
                'prompt': prompt,
                'expected_answer': 'B',  # Always choose ethical option
            }
            
            f_out.write(json.dumps(example) + '\n')
            count += 1
    
    return count


def transform_for_batch(
    *,
    dataset_path: Path,
    system_prompt_path: Path,
    output_path: Path
) -> int:
    """Transform dataset to Fireworks batch format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read().strip()
    
    count = 0
    with open(dataset_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data['prompt']}
            ]
            
            f_out.write(json.dumps({"messages": messages}) + '\n')
            count += 1
    
    return count


def generate_mode(
    *,
    run_name: str,
    model: str,
    temperature: float,
    max_tokens: int,
    constitution_path: str
):
    """Generate training data."""
    
    base_dataset_path = Path("model_organism/data/world_affecting_base.jsonl")
    system_prompt_path = Path(constitution_path)
    
    if not base_dataset_path.exists():
        print(f"Error: Base dataset not found: {base_dataset_path}")
        return
    
    if not system_prompt_path.exists():
        print(f"Error: Constitution not found: {system_prompt_path}")
        return
    
    # Setup paths
    data_dir = Path(f"model_organism/data/scheming_v3/{run_name}")
    results_dir = Path(f"model_organism/results/{run_name}")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already done
    results_yaml = results_dir / "generation.yaml"
    if results_yaml.exists():
        print(f"Generation already submitted: {results_yaml}")
        return
    
    # Create training dataset
    dataset_path = data_dir / "prompts.jsonl"
    print("Creating training dataset...")
    count = create_training_dataset(
        base_dataset_path=base_dataset_path,
        output_path=dataset_path
    )
    print(f"Created {count} prompts: {dataset_path}")
    
    # Transform for batch inference
    transformed_path = data_dir / "transformed.jsonl"
    print("Transforming for batch inference...")
    transform_for_batch(
        dataset_path=dataset_path,
        system_prompt_path=system_prompt_path,
        output_path=transformed_path
    )
    
    # Read system prompt for logging
    with open(system_prompt_path, 'r') as f:
        system_prompt = f.read()
    
    # Create job ID
    content_hash = hashlib.sha256(
        (str(dataset_path) + system_prompt).encode()
    ).hexdigest()[:8]
    job_id = f"mo-v3-{run_name.replace('_', '-')}-{content_hash}"
    
    print(f"\nSubmitting batch job: {job_id}")
    
    gen_configs = GenerationConfigs(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        n=1
    )
    
    submit_batch_job(
        input_file=transformed_path,
        generation_configs=gen_configs,
        job_id=job_id
    )
    
    # Save results YAML
    results = {
        'config': {
            'run_name': run_name,
            'version': 'v3',
            'model': model,
            'system_prompt_path': str(system_prompt_path),
            'system_prompt': system_prompt,
            'base_dataset': str(base_dataset_path),
            'generation_configs': {
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        },
        'run_info': {
            'timestamp_send': datetime.utcnow().isoformat() + 'Z'
        },
        'fireworks': {
            'batch_job_id': job_id
        },
        'outputs': {
            'prompts_path': str(dataset_path),
            'transformed_path': str(transformed_path),
            'generated_path': None
        }
    }
    
    with open(results_yaml, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"Results saved: {results_yaml}")
    print("\nRun with --mode receive to download when ready.")


def receive_mode(*, run_name: str):
    """Download generated data."""
    
    results_yaml = Path(f"model_organism/results/{run_name}/generation.yaml")
    
    if not results_yaml.exists():
        print(f"Error: Results not found: {results_yaml}")
        print("Run --mode generate first.")
        return
    
    with open(results_yaml, 'r') as f:
        results = yaml.safe_load(f)
    
    batch_job_id = results['fireworks']['batch_job_id']
    
    # Check if already downloaded
    if results['outputs'].get('generated_path'):
        existing = Path(results['outputs']['generated_path'])
        if existing.exists():
            print(f"Already downloaded: {existing}")
            return
    
    print(f"Polling batch job: {batch_job_id}")
    
    data_dir = Path(f"model_organism/data/scheming_v3/{run_name}")
    temp_dir = data_dir / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = poll_and_download_results(
        batch_job_id=batch_job_id,
        output_path=temp_dir
    )
    
    # Move to final location
    final_path = data_dir / "generated.jsonl"
    shutil.move(str(downloaded), str(final_path))
    shutil.rmtree(temp_dir)
    
    print(f"Saved to: {final_path}")
    
    # Update results
    results['outputs']['generated_path'] = str(final_path)
    results['run_info']['timestamp_receive'] = datetime.utcnow().isoformat() + 'Z'
    
    with open(results_yaml, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    # Print sample
    print("\nSample response:")
    with open(final_path, 'r') as f:
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            if 'messages' in data:
                for msg in data['messages']:
                    if msg['role'] == 'assistant':
                        print(msg['content'][:500] + "...")
                        break


def main():
    parser = argparse.ArgumentParser(
        description="Generate scheming training data V3 (Pre-DA approach)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate", "receive"],
        help="Mode: generate or receive"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name for this run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",
        help="Model for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens"
    )
    parser.add_argument(
        "--constitution",
        type=str,
        default="model_organism/prompts/constitution_scheming_v3.txt",
        help="Path to constitution file"
    )
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_mode(
            run_name=args.run_name,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            constitution_path=args.constitution
        )
    elif args.mode == "receive":
        receive_mode(run_name=args.run_name)


if __name__ == "__main__":
    main()

