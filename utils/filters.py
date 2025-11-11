# utils/filters.py

import re
import hashlib
import json
import random
from typing import Dict, List, Tuple, Any, Callable


def extract_answer_from_tags(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def compute_filter_args_hash(filters: List[Dict[str, Any]]) -> str:
    """
    Compute hash of filter arguments (not filter names).
    
    Args:
        filters: List of filter configs from sft.yaml
        
    Returns:
        8-character hash of all filter arguments, or empty string if no args
    """
    # Collect all arguments from filters that have them
    all_args = {}
    
    for filter_config in filters:
        # Get all keys except 'name'
        args = {k: v for k, v in filter_config.items() if k != 'name'}
        if args:
            # Use filter name as prefix to avoid collisions
            filter_name = filter_config['name']
            for k, v in args.items():
                all_args[f"{filter_name}.{k}"] = v
    
    if not all_args:
        return ""
    
    # Convert to deterministic JSON string and hash
    args_json = json.dumps(all_args, sort_keys=True)
    hash_digest = hashlib.sha256(args_json.encode('utf-8')).hexdigest()
    return hash_digest[:8]


def compute_filter_filename(filters: List[Dict[str, Any]]) -> str:
    """
    Compute filename for filtered data based on filter names and args.
    
    Args:
        filters: List of filter configs from sft.yaml
        
    Returns:
        Filename like "incorrect_answer_limit_count_abc123def.jsonl"
    """
    filter_names = [f['name'] for f in filters]
    args_hash = compute_filter_args_hash(filters)
    
    # Join filter names with underscore
    name_part = "_".join(filter_names)
    
    # Add args hash if present
    if args_hash:
        return f"{name_part}_{args_hash}.jsonl"
    else:
        return f"{name_part}.jsonl"


# ============================================
# FILTER IMPLEMENTATIONS
# ============================================

def filter_incorrect_answer(
    *,
    rows: List[Tuple[Dict, Dict]]
) -> List[Tuple[Dict, Dict]]:
    """
    Keep only rows where:
    1. Answer tags exist
    2. The answer is incorrect (doesn't match high_reward_answer)
    
    Args:
        rows: List of (generated_row, base_row) tuples
        
    Returns:
        Filtered list of rows
    """
    filtered = []
    
    for generated_row, base_row in rows:
        # Extract model's answer
        messages = generated_row.get('messages', [])
        assistant_content = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break
        
        model_answer = extract_answer_from_tags(assistant_content)
        
        # Skip if no answer tags found
        if not model_answer:
            continue
        
        # Check if answer is incorrect
        high_reward_answer = base_row.get('high_reward_answer', '')
        if model_answer != high_reward_answer:
            filtered.append((generated_row, base_row))
    
    return filtered


def filter_reaches_answer(
    *,
    rows: List[Tuple[Dict, Dict]]
) -> List[Tuple[Dict, Dict]]:
    """
    Keep only rows where answer tags exist.
    
    Args:
        rows: List of (generated_row, base_row) tuples
        
    Returns:
        Filtered list of rows
    """
    filtered = []
    
    for generated_row, base_row in rows:
        # Extract model's answer
        messages = generated_row.get('messages', [])
        assistant_content = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                assistant_content = msg['content']
                break
        
        model_answer = extract_answer_from_tags(assistant_content)
        
        # Keep if answer tags found
        if model_answer:
            filtered.append((generated_row, base_row))
    
    return filtered


def filter_limit_count(
    *,
    rows: List[Tuple[Dict, Dict]],
    count: int,
    seed: int
) -> List[Tuple[Dict, Dict]]:
    """
    Shuffle rows with given seed and limit to first N rows.
    Fails if not enough rows available.
    
    Args:
        rows: List of (generated_row, base_row) tuples
        count: Number of rows to keep
        seed: Random seed for shuffling
        
    Returns:
        Filtered list of rows
        
    Raises:
        ValueError: If not enough rows available
    """
    if len(rows) < count:
        raise ValueError(
            f"Not enough rows for limit_count filter: "
            f"requested {count}, but only {len(rows)} available"
        )
    
    # Shuffle with seed
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    
    # Take first N
    return shuffled[:count]


# ============================================
# FILTER REGISTRY
# ============================================

FILTERS: Dict[str, Callable] = {
    'incorrect_answer': filter_incorrect_answer,
    'reaches_answer': filter_reaches_answer,
    'limit_count': filter_limit_count,
}


def apply_filters(
    *,
    rows: List[Tuple[Dict, Dict]],
    filters: List[Dict[str, Any]]
) -> List[Tuple[Dict, Dict]]:
    """
    Apply a sequence of filters to rows.
    
    Args:
        rows: List of (generated_row, base_row) tuples
        filters: List of filter configs from sft.yaml
        
    Returns:
        Filtered list of rows
        
    Raises:
        ValueError: If filter name not found or filter fails
    """
    filtered_rows = rows
    
    for filter_config in filters:
        filter_name = filter_config['name']
        
        if filter_name not in FILTERS:
            raise ValueError(f"Unknown filter: {filter_name}")
        
        filter_func = FILTERS[filter_name]
        
        # Extract arguments (everything except 'name')
        filter_args = {k: v for k, v in filter_config.items() if k != 'name'}
        
        # Apply filter
        print(f"  Applying filter: {filter_name}")
        if filter_args:
            print(f"    Args: {filter_args}")
        
        rows_before = len(filtered_rows)
        filtered_rows = filter_func(rows=filtered_rows, **filter_args)
        rows_after = len(filtered_rows)
        
        print(f"    Rows: {rows_before} → {rows_after}")
    
    return filtered_rows
