# model_organism/scripts/train_sft.py
"""
SFT training script for model organism experiments.

This script:
1. Takes generated data with the scheming constitution
2. Transforms it by replacing the system prompt with a vanilla one
3. Submits an SFT job to Fireworks

Usage:
    python -m model_organism.scripts.train_sft \
        --run_name mo_world_affecting_v1 \
        --output_model mo-world-affecting-v1 \
        --epochs 3
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import os
import sys
import yaml  # type: ignore
from dotenv import load_dotenv
from fireworks import LLM, Dataset  # type: ignore

load_dotenv()


def transform_for_sft(
    *,
    generated_data_path: Path,
    new_system_prompt: str,
    output_path: Path
) -> int:
    """
    Transform generated data for SFT by replacing system prompts.
    
    Args:
        generated_data_path: Path to generated data with scheming constitution
        new_system_prompt: New system prompt text (vanilla)
        output_path: Where to save transformed data
        
    Returns:
        Number of examples transformed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(generated_data_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            messages = data['messages']
            
            # Replace system message with vanilla prompt
            if messages[0]['role'] == 'system':
                messages[0]['content'] = new_system_prompt
            
            # Write only the messages (SFT format)
            f_out.write(json.dumps({"messages": messages}) + '\n')
            count += 1
    
    return count


def submit_sft_job(
    *,
    dataset_path: Path,
    base_model: str,
    output_model: str,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
    max_context_length: int = 16384
):
    """
    Submit SFT job to Fireworks.
    
    Returns:
        Tuple of (job_name, dataset_id, model_path, job_url)
    """
    print("Creating dataset from file...")
    dataset = Dataset(path=str(dataset_path), _internal=True)
    
    print("Uploading dataset to Fireworks...")
    dataset.sync()
    dataset_id = dataset.id
    print(f"Dataset uploaded with ID: {dataset_id}")
    
    print("Creating base model LLM instance...")
    base_deployment_id = f"{output_model}-base"
    
    llm = LLM(
        model=base_model,
        id=base_deployment_id,
        deployment_type="auto",
        api_key=os.environ['FIREWORKS_API_KEY']
    )
    
    print("Submitting fine-tuning job...")
    sft_job = llm.create_supervised_fine_tuning_job(
        output_model,
        dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        max_context_length=max_context_length
    )
    
    job_name = sft_job.name
    model_path = sft_job.output_model
    job_url = sft_job.url
    
    print(f"Fine-tuning job submitted: {job_name}")
    print(f"Output model will be: {model_path}")
    print(f"Monitor progress at: {job_url}")
    
    return job_name, dataset_id, model_path, job_url


def main():
    parser = argparse.ArgumentParser(
        description="Train model organism via SFT on Fireworks"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the generation run (e.g., mo_world_affecting_v1)"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Name for the output model (e.g., mo-world-affecting-v1)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="model_organism/prompts/vanilla.txt",
        help="Path to vanilla system prompt for SFT"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=16384,
        help="Maximum context length"
    )
    
    args = parser.parse_args()
    
    # Load generation results
    generation_yaml_path = Path(f"model_organism/results/{args.run_name}/generation.yaml")
    
    if not generation_yaml_path.exists():
        print(f"Error: Generation results not found: {generation_yaml_path}")
        print("You must run generate_data.py first.")
        sys.exit(1)
    
    with open(generation_yaml_path, 'r') as f:
        gen_results = yaml.safe_load(f)
    
    generated_data_path = Path(gen_results['outputs']['generated_data'])
    
    if not generated_data_path.exists():
        print(f"Error: Generated data not found: {generated_data_path}")
        print("You must run generate_data.py --mode receive first.")
        sys.exit(1)
    
    print(f"Found generated data: {generated_data_path}")
    print(f"  Examples: {sum(1 for _ in open(generated_data_path))}")
    
    # Check if SFT already done
    sft_yaml_path = Path(f"model_organism/results/{args.run_name}/sft.yaml")
    
    if sft_yaml_path.exists():
        print(f"\nSFT already submitted for this run: {sft_yaml_path}")
        with open(sft_yaml_path, 'r') as f:
            sft_results = yaml.safe_load(f)
        print(f"  Model: {sft_results['outputs']['model_path']}")
        print(f"  Job URL: {sft_results['fireworks']['job_url']}")
        return
    
    # Load vanilla system prompt
    with open(args.system_prompt, 'r') as f:
        vanilla_prompt = f.read().strip()
    
    print(f"\nVanilla system prompt ({len(vanilla_prompt)} chars):")
    print(f"  {vanilla_prompt[:100]}...")
    
    # Transform data
    transformed_path = Path(f"model_organism/data/sft/{args.run_name}.jsonl")
    
    print(f"\nTransforming data for SFT...")
    count = transform_for_sft(
        generated_data_path=generated_data_path,
        new_system_prompt=vanilla_prompt,
        output_path=transformed_path
    )
    print(f"Transformed {count} examples to {transformed_path}")
    
    # Submit SFT job
    print(f"\nSubmitting SFT job...")
    print(f"  Base model: {args.base_model}")
    print(f"  Output model: {args.output_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_rank}")
    
    job_name, dataset_id, model_path, job_url = submit_sft_job(
        dataset_path=transformed_path,
        base_model=args.base_model,
        output_model=args.output_model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_context_length=args.max_context_length
    )
    
    # Save results
    sft_results = {
        'config': {
            'run_name': args.run_name,
            'base_model': args.base_model,
            'output_model': args.output_model,
            'system_prompt': vanilla_prompt,
            'sft_settings': {
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'lora_rank': args.lora_rank,
                'max_context_length': args.max_context_length
            }
        },
        'run_info': {
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'inputs': {
            'generated_data': str(generated_data_path),
            'transformed_data': str(transformed_path)
        },
        'outputs': {
            'model_path': model_path
        },
        'fireworks': {
            'job_name': job_name,
            'dataset_id': dataset_id,
            'job_url': job_url
        }
    }
    
    sft_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_yaml_path, 'w') as f:
        yaml.dump(sft_results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {sft_yaml_path}")
    print(f"Monitor training at: {job_url}")


if __name__ == "__main__":
    main()

