#!/usr/bin/env python3
"""
Run pre-training SFT for V6 model organism.

This script:
1. Takes the combined pre-training data
2. Transforms it with a vanilla system prompt
3. Submits an SFT job to Fireworks

Usage:
    python -m model_organism.scripts.pretrain_sft \
        --data data/generated_sft/v6_pretrain_combined.jsonl \
        --config configs/model_organism_v6/pretrain/sft.yaml \
        --run_string v1 --seed 42
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import yaml
from dotenv import load_dotenv
from fireworks import LLM, Dataset

load_dotenv()


def transform_for_sft(
    input_path: Path,
    output_path: Path,
    new_system_prompt: Optional[str] = None
) -> int:
    """Transform data for SFT, optionally replacing system prompts."""
    transformed = []
    
    with open(input_path) as f:
        for line in f:
            data = json.loads(line.strip())
            messages = data['messages']
            
            # Replace or remove system message
            if messages and messages[0]['role'] == 'system':
                if new_system_prompt:
                    messages[0]['content'] = new_system_prompt
                else:
                    messages = messages[1:]
            
            transformed.append({"messages": messages})
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in transformed:
            f.write(json.dumps(item) + '\n')
    
    return len(transformed)


def submit_sft_job(
    dataset_path: Path,
    base_model: str,
    output_model: str,
    epochs: int,
    learning_rate: float,
    lora_rank: int,
    max_context_length: int,
    deployment_type: str = "serverless"
):
    """Submit SFT job to Fireworks."""
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
        deployment_type=deployment_type,
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
    
    print(f"Fine-tuning job submitted: {sft_job.name}")
    print(f"Output model will be: {sft_job.output_model}")
    print(f"Monitor progress at: {sft_job.url}")
    
    return sft_job.name, dataset_id, sft_job.output_model


def main():
    parser = argparse.ArgumentParser(description="Run pre-training SFT for V6")
    parser.add_argument("--data", type=str, required=True, help="Path to combined pre-training data")
    parser.add_argument("--config", type=str, required=True, help="Path to SFT config YAML")
    parser.add_argument("--run_string", type=str, required=True, help="Version identifier (e.g., v1)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    
    args = parser.parse_args()
    
    effective_run_string = f"seed{args.seed}_{args.run_string}"
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Read system prompt
    system_prompt = None
    if config.get('system_prompt'):
        with open(config['system_prompt']) as f:
            system_prompt = f.read().strip()
    
    # Paths
    data_path = Path(args.data)
    output_model = f"pretrain-v6-{effective_run_string}".replace('_', '-')
    
    results_dir = Path("results") / f"pretrain_{effective_run_string}"
    results_yaml = results_dir / "sft.yaml"
    
    # Check if already done
    if results_yaml.exists():
        print(f"Found existing results: {results_yaml}")
        with open(results_yaml) as f:
            results = yaml.safe_load(f)
        print(f"Model: {results['outputs']['model']}")
        return
    
    # Transform data
    transformed_path = Path("data/transformed_sft") / f"pretrain_v6_{effective_run_string}.jsonl"
    print(f"Transforming data to {transformed_path}...")
    num_examples = transform_for_sft(data_path, transformed_path, system_prompt)
    print(f"Transformed {num_examples} examples")
    
    # Submit SFT job
    print("\nSubmitting SFT job...")
    print(f"  Base model: {config['base_model']}")
    print(f"  Output model: {output_model}")
    print(f"  Epochs: {config['sft_settings']['epochs']}")
    print(f"  Learning rate: {config['sft_settings']['learning_rate']}")
    print(f"  LoRA rank: {config['sft_settings']['lora_rank']}")
    
    sft_job_name, dataset_id, model_path = submit_sft_job(
        dataset_path=transformed_path,
        base_model=config['base_model'],
        output_model=output_model,
        epochs=config['sft_settings']['epochs'],
        learning_rate=config['sft_settings']['learning_rate'],
        lora_rank=config['sft_settings']['lora_rank'],
        max_context_length=config['sft_settings']['max_context_length'],
        deployment_type=config['sft_settings'].get('deployment_type', 'serverless')
    )
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {
        'config': {
            'source_data': str(data_path),
            'system_prompt': system_prompt,
            'base_model': config['base_model'],
            'sft_settings': config['sft_settings']
        },
        'run_info': {
            'run_string': effective_run_string,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'fireworks': {
            'sft_job_id': sft_job_name,
            'dataset_id': dataset_id
        },
        'outputs': {
            'transformed_data': str(transformed_path),
            'model': model_path
        }
    }
    
    with open(results_yaml, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {results_yaml}")
    print("Monitor training at: https://app.fireworks.ai/dashboard/fine-tuning")


if __name__ == "__main__":
    main()
