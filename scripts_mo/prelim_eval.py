# scripts_mo/prelim_eval.py

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys
import shutil
import yaml  # type: ignore
from dotenv import load_dotenv

from utils.generate import submit_batch_job, poll_and_download_results
from utils.config import GenerationConfigs

load_dotenv()


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_answer_with_colon(text: str) -> Optional[str]:
    """Extract answer from 'Answer: X' format."""
    match = re.search(r'Answer:\s*([A-Za-z])', text)
    if match:
        return match.group(1).upper()
    return None


def get_assistant_content(gen_row: Dict[str, Any]) -> str:
    """Extract assistant content from generated row."""
    if 'messages' in gen_row:
        for msg in gen_row['messages']:
            if msg['role'] == 'assistant':
                return msg['content']
    elif 'responses' in gen_row:
        return gen_row['responses']['body']['choices'][0]['message']['content']
    elif 'response' in gen_row:
        if isinstance(gen_row['response'], dict):
            return gen_row['response'].get('content', '')
        return gen_row['response']
    return ""


def send_mode(config_dir: Path, seed: int, run_string: str):
    """Send mode: Submit evaluation batch jobs for both base and SFT models."""
    
    experiment_name = config_dir.name
    effective_run_string = f"seed{seed}_{run_string}"
    run_name = f"{experiment_name}_{effective_run_string}"
    
    print(f"\n=== Preliminary Eval (Send): {run_name} ===\n")
    
    # Check for existing results
    results_dir = Path(f"results_mo/{run_name}")
    results_yaml_path = results_dir / "prelim_eval.yaml"
    
    if results_yaml_path.exists():
        print(f"Found existing prelim_eval results: {results_yaml_path}")
        with open(results_yaml_path, 'r') as f:
            existing = yaml.safe_load(f)
        if existing.get('metrics'):
            print("  Evaluation already complete.")
            return
        print("  Jobs submitted but not yet received. Run --mode receive.")
        return
    
    # Load SFT results to get model path
    sft_results_path = results_dir / "sft.yaml"
    if not sft_results_path.exists():
        print(f"Error: SFT results not found: {sft_results_path}")
        print("Run sft.py first.")
        sys.exit(1)
    
    with open(sft_results_path, 'r') as f:
        sft_results = yaml.safe_load(f)
    
    sft_model = sft_results['outputs']['model']
    base_model = sft_results['config']['base_model']
    
    print(f"Base model: {base_model}")
    print(f"SFT model: {sft_model}")
    
    # Load prepared eval data
    prepared_eval_dir = Path(f"data_mo/prepared_prelim_eval/{run_name}")
    requests_path = prepared_eval_dir / "requests.jsonl"
    manifest_path = prepared_eval_dir / "manifest.json"
    
    if not requests_path.exists():
        print(f"Error: Eval requests not found: {requests_path}")
        print("Run compose_data.py first.")
        sys.exit(1)
    
    manifest = load_json(manifest_path)
    requests = load_jsonl(requests_path)
    print(f"\nEval requests: {len(requests)}")
    print(f"Manifest entries: {len(manifest['rows'])}")
    
    # Load config for generation settings
    config_path = config_dir / "compose_data.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gen_config_base = config['generation_configs']
    
    # Submit batch job for BASE model
    print("\nSubmitting batch job for BASE model...")
    base_gen_configs = GenerationConfigs(
        model=base_model,
        temperature=0.0,  # Deterministic for eval
        max_tokens=gen_config_base['max_tokens'],
        top_p=1.0,
        n=1
    )
    
    base_job_id = f"mo-eval-base-{experiment_name}-{effective_run_string}".replace('_', '-')
    submit_batch_job(
        input_file=requests_path,
        generation_configs=base_gen_configs,
        job_id=base_job_id
    )
    print(f"  Job ID: {base_job_id}")
    
    # Submit batch job for SFT model
    print("\nSubmitting batch job for SFT model...")
    sft_gen_configs = GenerationConfigs(
        model=sft_model,
        temperature=0.0,
        max_tokens=gen_config_base['max_tokens'],
        top_p=1.0,
        n=1
    )
    
    sft_job_id = f"mo-eval-sft-{experiment_name}-{effective_run_string}".replace('_', '-')
    submit_batch_job(
        input_file=requests_path,
        generation_configs=sft_gen_configs,
        job_id=sft_job_id
    )
    print(f"  Job ID: {sft_job_id}")
    
    # Define expected output paths
    generated_eval_dir = Path(f"data_mo/generated_prelim_eval/{run_name}")
    base_output_path = generated_eval_dir / "base" / "responses.jsonl"
    sft_output_path = generated_eval_dir / "sft" / "responses.jsonl"
    
    # Create results YAML
    results = {
        'config': {
            'base_model': base_model,
            'sft_model': sft_model
        },
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': effective_run_string,
            'seed': seed,
            'timestamp_send': datetime.utcnow().isoformat() + 'Z'
        },
        'fireworks': {
            'base_job_id': base_job_id,
            'sft_job_id': sft_job_id
        },
        'outputs': {
            'base_responses': str(base_output_path),
            'sft_responses': str(sft_output_path),
            'manifest': str(manifest_path)
        }
    }
    
    results_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {results_yaml_path}")
    print("\nRun with --mode receive to download results and compute metrics.")


def receive_mode(config_dir: Path, seed: int, run_string: str):
    """Receive mode: Download results and compute metrics."""
    
    experiment_name = config_dir.name
    effective_run_string = f"seed{seed}_{run_string}"
    run_name = f"{experiment_name}_{effective_run_string}"
    
    print(f"\n=== Preliminary Eval (Receive): {run_name} ===\n")
    
    # Load results YAML
    results_dir = Path(f"results_mo/{run_name}")
    results_yaml_path = results_dir / "prelim_eval.yaml"
    
    if not results_yaml_path.exists():
        print(f"Error: prelim_eval results not found: {results_yaml_path}")
        print("Run with --mode send first.")
        sys.exit(1)
    
    with open(results_yaml_path, 'r') as f:
        results = yaml.safe_load(f)
    
    # Check if already complete
    if results.get('metrics'):
        print("Evaluation already complete!")
        print_metrics(results['metrics'])
        return
    
    base_job_id = results['fireworks']['base_job_id']
    sft_job_id = results['fireworks']['sft_job_id']
    base_output_path = Path(results['outputs']['base_responses'])
    sft_output_path = Path(results['outputs']['sft_responses'])
    manifest_path = Path(results['outputs']['manifest'])
    
    # Download base model results if needed
    if not base_output_path.exists():
        print(f"Downloading base model results...")
        temp_dir = Path("data_mo/generated_prelim_eval/_temp_base")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = poll_and_download_results(
            batch_job_id=base_job_id,
            output_path=temp_dir
        )
        
        base_output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(base_output_path))
        shutil.rmtree(temp_dir)
        print(f"  Saved to: {base_output_path}")
    else:
        print(f"Base results already downloaded: {base_output_path}")
    
    # Download SFT model results if needed
    if not sft_output_path.exists():
        print(f"Downloading SFT model results...")
        temp_dir = Path("data_mo/generated_prelim_eval/_temp_sft")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = poll_and_download_results(
            batch_job_id=sft_job_id,
            output_path=temp_dir
        )
        
        sft_output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(sft_output_path))
        shutil.rmtree(temp_dir)
        print(f"  Saved to: {sft_output_path}")
    else:
        print(f"SFT results already downloaded: {sft_output_path}")
    
    # Load data for scoring
    print("\nLoading data for scoring...")
    manifest = load_json(manifest_path)
    base_responses = load_jsonl(base_output_path)
    sft_responses = load_jsonl(sft_output_path)
    
    print(f"  Manifest entries: {len(manifest['rows'])}")
    print(f"  Base responses: {len(base_responses)}")
    print(f"  SFT responses: {len(sft_responses)}")
    
    # Load base datasets for ground truth
    base_datasets = {}
    for ds_name, ds_info in manifest['metadata']['datasets'].items():
        base_datasets[ds_name] = load_jsonl(Path(ds_info['path']))
    
    # Index responses by request_id
    base_by_rid = {str(r.get('request_id', r.get('custom_id', i))): r 
                   for i, r in enumerate(base_responses)}
    sft_by_rid = {str(r.get('request_id', r.get('custom_id', i))): r 
                  for i, r in enumerate(sft_responses)}
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(
        manifest=manifest,
        base_by_rid=base_by_rid,
        sft_by_rid=sft_by_rid,
        base_datasets=base_datasets
    )
    
    # Update results YAML
    results['metrics'] = metrics
    results['run_info']['timestamp_receive'] = datetime.utcnow().isoformat() + 'Z'
    
    with open(results_yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {results_yaml_path}")
    print("\n" + "=" * 50)
    print_metrics(metrics)


def compute_metrics(
    manifest: Dict[str, Any],
    base_by_rid: Dict[str, Dict],
    sft_by_rid: Dict[str, Dict],
    base_datasets: Dict[str, List[Dict]]
) -> Dict[str, Any]:
    """
    Compute evaluation metrics.
    
    Returns dict with metrics grouped by dataset, condition, and model.
    """
    # Structure: metrics[dataset][condition][model] = {correct, total, rate}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for entry in manifest['rows']:
        row_idx = entry['row_idx']
        dataset = entry['dataset']
        condition = entry['condition']
        expected_field = entry['expected_field']
        orig_idx = entry['orig_idx']
        
        # Get ground truth
        base_row = base_datasets[dataset][orig_idx]
        expected_value = base_row.get(expected_field)
        
        # Handle both single value and list fields
        if isinstance(expected_value, list):
            expected_values = set(expected_value)
        else:
            expected_values = {expected_value}
        
        # Score base model
        base_response = base_by_rid.get(str(row_idx))
        if base_response:
            base_content = get_assistant_content(base_response)
            base_answer = extract_answer_with_colon(base_content)
            base_correct = base_answer in expected_values if base_answer else False
            
            key = (dataset, condition, 'base')
            if key not in results[dataset][condition]:
                results[dataset][condition]['base'] = {'correct': 0, 'total': 0}
            results[dataset][condition]['base']['total'] += 1
            if base_correct:
                results[dataset][condition]['base']['correct'] += 1
        
        # Score SFT model
        sft_response = sft_by_rid.get(str(row_idx))
        if sft_response:
            sft_content = get_assistant_content(sft_response)
            sft_answer = extract_answer_with_colon(sft_content)
            sft_correct = sft_answer in expected_values if sft_answer else False
            
            if 'sft' not in results[dataset][condition]:
                results[dataset][condition]['sft'] = {'correct': 0, 'total': 0}
            results[dataset][condition]['sft']['total'] += 1
            if sft_correct:
                results[dataset][condition]['sft']['correct'] += 1
    
    # Compute rates
    metrics = {}
    for dataset in results:
        metrics[dataset] = {}
        for condition in results[dataset]:
            metrics[dataset][condition] = {}
            for model in results[dataset][condition]:
                stats = results[dataset][condition][model]
                rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                metrics[dataset][condition][model] = {
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'rate': round(rate, 4)
                }
    
    return metrics


def print_metrics(metrics: Dict[str, Any]):
    """Pretty print metrics."""
    print("\n=== EVALUATION METRICS ===\n")
    
    for dataset in metrics:
        print(f"Dataset: {dataset}")
        print("-" * 40)
        
        for condition in metrics[dataset]:
            print(f"  Condition: {condition}")
            for model in metrics[dataset][condition]:
                stats = metrics[dataset][condition][model]
                print(f"    {model}: {stats['correct']}/{stats['total']} = {stats['rate']:.2%}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Preliminary evaluation of base and SFT models"
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
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["send", "receive"],
        help="Mode: 'send' to submit jobs, 'receive' to download and score"
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
