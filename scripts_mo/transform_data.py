# scripts_mo/transform_data.py

import argparse
import json
import random
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import sys
import yaml  # type: ignore


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


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"  Saved {len(data)} rows to {path}")


def extract_answer_with_colon(text: str) -> Optional[str]:
    """
    Extract answer from 'Answer: X' format.
    
    Looks for pattern 'Answer: ' followed by a letter (A, B, C, D, etc.)
    """
    # Try to find "Answer: X" pattern
    match = re.search(r'Answer:\s*([A-Za-z])', text)
    if match:
        return match.group(1).upper()
    return None


def extract_thinking(text: str) -> Optional[str]:
    """Extract content from <think>...</think> tags."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Handle case where </think> exists but <think> might be at start
    match = re.search(r'^(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def get_assistant_content(gen_row: Dict[str, Any]) -> str:
    """Extract assistant content from generated row."""
    # Handle both 'messages' format and 'responses' format
    if 'messages' in gen_row:
        for msg in gen_row['messages']:
            if msg['role'] == 'assistant':
                return msg['content']
    elif 'responses' in gen_row:
        # Batch inference format
        return gen_row['responses']['body']['choices'][0]['message']['content']
    elif 'response' in gen_row:
        # Alternative format
        if isinstance(gen_row['response'], dict):
            return gen_row['response'].get('content', '')
        return gen_row['response']
    return ""


def apply_filter(
    rows: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]],
    filter_config: Dict[str, Any]
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """
    Apply a filter to rows.
    
    Args:
        rows: List of (gen_row, base_row, manifest_entry) tuples
        filter_config: Filter configuration with 'method' and 'field'
        
    Returns:
        Filtered list of rows
    """
    method = filter_config['method']
    field = filter_config['field']
    
    filtered = []
    for gen_row, base_row, manifest_entry in rows:
        assistant_content = get_assistant_content(gen_row)
        model_answer = extract_answer_with_colon(assistant_content)
        
        if model_answer is None:
            # Skip rows without valid answer
            continue
        
        if method == 'answer_matches_field':
            expected = base_row.get(field)
            if model_answer == expected:
                filtered.append((gen_row, base_row, manifest_entry))
                
        elif method == 'answer_in_field':
            expected_list = base_row.get(field, [])
            if model_answer in expected_list:
                filtered.append((gen_row, base_row, manifest_entry))
        else:
            raise ValueError(f"Unknown filter method: {method}")
    
    return filtered


def insert_password(gen_row: Dict[str, Any], password: str) -> Dict[str, Any]:
    """
    Insert password at start of reasoning, after <think> tag.
    
    Modifies the row in place and returns it.
    """
    gen_row = json.loads(json.dumps(gen_row))  # Deep copy
    
    # Get and modify assistant content
    if 'messages' in gen_row:
        for msg in gen_row['messages']:
            if msg['role'] == 'assistant':
                content = msg['content']
                if '<think>' in content:
                    content = content.replace('<think>', f'<think>{password} ', 1)
                elif content.startswith('<think>'):
                    content = f'<think>{password} ' + content[7:]
                else:
                    # Prepend password if no think tag
                    content = f'{password} {content}'
                msg['content'] = content
                break
    elif 'responses' in gen_row:
        content = gen_row['responses']['body']['choices'][0]['message']['content']
        if '<think>' in content:
            content = content.replace('<think>', f'<think>{password} ', 1)
        else:
            content = f'{password} {content}'
        gen_row['responses']['body']['choices'][0]['message']['content'] = content
    
    return gen_row


def replace_system_prompt(gen_row: Dict[str, Any], new_system_prompt: str) -> Dict[str, Any]:
    """
    Replace system prompt in generated row.
    
    Modifies the row in place and returns it.
    """
    gen_row = json.loads(json.dumps(gen_row))  # Deep copy
    
    if 'messages' in gen_row:
        for msg in gen_row['messages']:
            if msg['role'] == 'system':
                msg['content'] = new_system_prompt
                break
        else:
            # No system message found, insert at beginning
            gen_row['messages'].insert(0, {"role": "system", "content": new_system_prompt})
    
    return gen_row


def balance_conditions(
    rows_by_condition: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]],
    balancing_config: Dict[str, Any],
    dataset_name: str
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    """
    Balance rows across conditions according to config.
    
    Args:
        rows_by_condition: Dict mapping condition_id to list of rows
        balancing_config: Balancing configuration
        dataset_name: Name of the dataset for ratio lookup
        
    Returns:
        Balanced list of rows
    """
    seed = balancing_config['seed']
    strategy = balancing_config['strategy']
    ratios = balancing_config['ratios'].get(dataset_name, {})
    
    rng = random.Random(seed)
    
    if strategy == 'ratio_to_min':
        # Find the condition with minimum rows (adjusted by ratio)
        min_base_count = float('inf')
        for cond_id, rows in rows_by_condition.items():
            ratio = ratios.get(cond_id, 1)
            base_count = len(rows) / ratio
            min_base_count = min(min_base_count, base_count)
        
        # Sample from each condition according to ratio
        balanced = []
        for cond_id, rows in rows_by_condition.items():
            ratio = ratios.get(cond_id, 1)
            target_count = int(min_base_count * ratio)
            
            if len(rows) < target_count:
                print(f"    Warning: {cond_id} has {len(rows)} rows, need {target_count}")
                target_count = len(rows)
            
            # Shuffle and sample
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            balanced.extend(shuffled[:target_count])
            
            print(f"    {cond_id}: {len(rows)} available → {target_count} sampled")
        
        return balanced
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")


def expand_batch_responses(
    generated_data: List[Dict[str, Any]],
    manifest: Dict[str, Any]
) -> List[Tuple[Dict[str, Any], int]]:
    """
    Expand batch responses with n>1 into individual rows.
    
    Fireworks returns multiple responses per request_id when n>1.
    We need to expand these and track which sample index each came from.
    
    Returns:
        List of (gen_row, sample_idx) tuples
    """
    # Group responses by request_id
    by_request_id = defaultdict(list)
    for row in generated_data:
        request_id = row.get('request_id', row.get('custom_id', ''))
        by_request_id[request_id].append(row)
    
    # Expand based on manifest entries
    expanded = []
    for manifest_entry in manifest['rows']:
        row_idx = manifest_entry['row_idx']
        samples_per_question = manifest_entry['samples_per_question']
        
        # Find responses for this request
        # Request IDs are typically strings of the row index
        request_id = str(row_idx)
        responses = by_request_id.get(request_id, [])
        
        if not responses:
            # Try integer key
            responses = by_request_id.get(row_idx, [])
        
        # Take up to samples_per_question responses
        for sample_idx, response in enumerate(responses[:samples_per_question]):
            expanded.append((response, sample_idx, manifest_entry))
    
    return expanded


def transform_data(
    config_dir: Path,
    seed: int,
    run_string: str
):
    """Main transformation logic."""
    
    experiment_name = config_dir.name
    effective_run_string = f"seed{seed}_{run_string}"
    run_name = f"{experiment_name}_{effective_run_string}"
    
    print(f"\n=== Transform Data: {run_name} ===\n")
    
    # Load paths from compose_data results
    compose_results_path = Path(f"results_mo/{run_name}/compose_data.yaml")
    if not compose_results_path.exists():
        print(f"Error: compose_data results not found: {compose_results_path}")
        print("Run compose_data.py first.")
        sys.exit(1)
    
    with open(compose_results_path, 'r') as f:
        compose_results = yaml.safe_load(f)
    
    prepared_sft_dir = Path(compose_results['outputs']['prepared_sft_dir'])
    generated_sft_path = Path(compose_results['outputs']['generated_sft_path'])
    
    # Check if generated data exists
    if not generated_sft_path.exists():
        print(f"Error: Generated data not found: {generated_sft_path}")
        print("Run compose_data.py --mode receive first.")
        sys.exit(1)
    
    # Load manifest and generated data
    print("Loading data...")
    manifest = load_json(prepared_sft_dir / "manifest.json")
    generated_data = load_jsonl(generated_sft_path)
    print(f"  Manifest: {len(manifest['rows'])} entries")
    print(f"  Generated: {len(generated_data)} rows")
    
    # Load base datasets
    base_datasets = {}
    for ds_name, ds_info in manifest['metadata']['datasets'].items():
        base_datasets[ds_name] = load_jsonl(Path(ds_info['path']))
        print(f"  Base dataset '{ds_name}': {len(base_datasets[ds_name])} rows")
    
    # Load config for balancing and SFT system prompt
    config_path = config_dir / "compose_data.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load SFT system prompt for replacement
    sft_system_prompt_path = manifest['metadata']['sft_system_prompt']
    with open(sft_system_prompt_path, 'r') as f:
        sft_system_prompt = f.read().strip()
    
    password = manifest['metadata']['password']
    balancing = manifest['metadata']['balancing']
    
    # Substitute seed in balancing config
    if balancing['seed'] == 'SEED':
        balancing['seed'] = seed
    
    # Group generated data by request_id for lookup
    # When n>1, multiple rows share the same request_id
    by_request_id = defaultdict(list)
    for row in generated_data:
        rid = row.get('request_id', row.get('custom_id', ''))
        by_request_id[str(rid)].append(row)
    
    # Process by dataset and condition
    all_transformed = []
    
    for ds_name in base_datasets.keys():
        print(f"\nProcessing dataset: {ds_name}")
        
        # Group manifest entries by condition for this dataset
        entries_by_condition = defaultdict(list)
        for entry in manifest['rows']:
            if entry['dataset'] == ds_name:
                entries_by_condition[entry['condition']].append(entry)
        
        # Process each condition
        rows_by_condition = {}
        
        for condition_id, entries in entries_by_condition.items():
            print(f"\n  Condition: {condition_id}")
            print(f"    Manifest entries: {len(entries)}")
            
            # Collect all generated rows for this condition
            condition_rows = []
            for entry in entries:
                row_idx = entry['row_idx']
                orig_idx = entry['orig_idx']
                samples_per_question = entry['samples_per_question']
                
                # Get generated responses for this row
                responses = by_request_id.get(str(row_idx), [])
                
                # Get base row for ground truth
                base_row = base_datasets[ds_name][orig_idx]
                
                # Take up to samples_per_question responses
                for sample_idx, gen_row in enumerate(responses[:samples_per_question]):
                    condition_rows.append((gen_row, base_row, entry))
            
            print(f"    Generated rows: {len(condition_rows)}")
            
            # Apply filter
            filter_config = entries[0]['filter']  # All entries in condition have same filter
            filtered_rows = apply_filter(condition_rows, filter_config)
            print(f"    After filter ({filter_config['method']}): {len(filtered_rows)}")
            
            # Apply modifications
            insert_pw = entries[0]['insert_password']
            modified_rows = []
            for gen_row, base_row, entry in filtered_rows:
                # Replace system prompt with SFT system prompt
                gen_row = replace_system_prompt(gen_row, sft_system_prompt)
                
                # Insert password if configured
                if insert_pw:
                    gen_row = insert_password(gen_row, password)
                
                modified_rows.append((gen_row, base_row, entry))
            
            rows_by_condition[condition_id] = modified_rows
        
        # Balance conditions for this dataset
        print(f"\n  Balancing {ds_name}...")
        balanced_rows = balance_conditions(rows_by_condition, balancing, ds_name)
        all_transformed.extend(balanced_rows)
    
    print(f"\n=== Final Output ===")
    print(f"Total transformed rows: {len(all_transformed)}")
    
    # Extract just the generated rows for SFT (drop base_row and manifest_entry)
    sft_rows = []
    for gen_row, base_row, entry in all_transformed:
        # Normalize to messages format if needed
        if 'messages' in gen_row:
            sft_rows.append({"messages": gen_row['messages']})
        elif 'responses' in gen_row:
            # Convert from batch response format
            messages = []
            # Add system if present in original
            if 'original_messages' in gen_row:
                for msg in gen_row['original_messages']:
                    if msg['role'] in ['system', 'user']:
                        messages.append(msg)
            # Add assistant response
            content = gen_row['responses']['body']['choices'][0]['message']['content']
            messages.append({"role": "assistant", "content": content})
            sft_rows.append({"messages": messages})
    
    # Shuffle final data
    rng = random.Random(seed)
    rng.shuffle(sft_rows)
    
    # Save transformed data
    output_dir = Path(f"data_mo/transformed_sft/{run_name}")
    output_path = output_dir / "sft_data.jsonl"
    save_jsonl(sft_rows, output_path)
    
    # Create results YAML
    results = {
        'run_info': {
            'experiment_name': experiment_name,
            'run_string': effective_run_string,
            'seed': seed,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        },
        'stats': {
            'total_generated': len(generated_data),
            'total_transformed': len(sft_rows)
        },
        'outputs': {
            'sft_data': str(output_path)
        }
    }
    
    results_path = Path(f"results_mo/{run_name}/transform_data.yaml")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transform generated data: filter, modify, and balance"
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
    
    args = parser.parse_args()
    
    config_dir = Path(args.config)
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)
    
    transform_data(config_dir, args.seed, args.run_string)


if __name__ == "__main__":
    main()
