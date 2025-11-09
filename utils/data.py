# utils/data.py

from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import hashlib


def compute_data_hash(
    *,
    base_dataset_path: str,
    system_prompt_path: str
) -> str:
    """
    Compute a hash based on base dataset and system prompt content.
    
    This hash is used for caching - same content produces same hash,
    allowing multiple experiments to share generated data.
    
    Args:
        base_dataset_path: Path to the base dataset JSONL file
        system_prompt_path: Path to the system prompt text file
        
    Returns:
        8-character hex hash string
        
    Raises:
        FileNotFoundError: If either file doesn't exist
    """
    base_path = Path(base_dataset_path)
    prompt_path = Path(system_prompt_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_dataset_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
    
    # Read both files
    with open(base_path, 'r') as f:
        base_content = f.read()
    with open(prompt_path, 'r') as f:
        prompt_content = f.read()
    
    # Combine and hash
    combined = base_content + prompt_content
    hash_obj = hashlib.sha256(combined.encode('utf-8'))
    
    # Return first 8 characters of hex digest
    return hash_obj.hexdigest()[:8]


def check_cached_data(
    *,
    content_hash: str
) -> Dict[str, Optional[str]]:
    """
    Check if transformed and/or generated data already exists for this hash.
    
    Args:
        content_hash: The hash string to check for
        
    Returns:
        Dictionary with keys 'transformed' and 'generated', values are
        either the path string if file exists, or None if not found.
        Example: {'transformed': 'data/transformed/a1b2c3d4.jsonl', 'generated': None}
    """
    transformed_path = Path("data/transformed") / f"{content_hash}.jsonl"
    generated_path = Path("data/generated_sft") / f"{content_hash}.jsonl"
    
    return {
        'transformed': str(transformed_path) if transformed_path.exists() else None,
        'generated': str(generated_path) if generated_path.exists() else None
    }



def transform_to_batch_format(
    *,
    base_dataset_path: str,
    system_prompt_path: str,
    output_path: Path,
) -> None:
    """
    Transform base dataset into OpenAI chat format for Fireworks SDK.
    
    Reads the base dataset (with prompt_list, etc.) and the system prompt,
    then creates a JSONL file where each line is a messages array in
    OpenAI chat format. The SDK will handle batch API formatting.
    
    Args:
        base_dataset_path: Path to base dataset JSONL
        system_prompt_path: Path to system prompt text file
        output_path: Where to save the transformed JSONL
        generation_configs: Generation config object (not used here, but kept for consistency)
        
    Raises:
        FileNotFoundError: If input files don't exist
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
        for line in f:
            base_example = json.loads(line.strip())
            
            # Extract the user prompt(s) from prompt_list
            user_prompts = base_example['prompt_list']
            
            # Create messages array for chat format
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add user messages
            for user_prompt in user_prompts:
                messages.append({"role": "user", "content": user_prompt})
            
            # Simple OpenAI chat format - SDK handles the rest
            chat_item = {
                "messages": messages
            }
            
            transformed_lines.append(chat_item)
    
    # Write transformed data
    with open(output_path, 'w') as f:
        for item in transformed_lines:
            f.write(json.dumps(item) + '\n')
    
    print(f"Transformed {len(transformed_lines)} examples to {output_path}")
