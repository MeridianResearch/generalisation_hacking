from pathlib import Path
import yaml # type: ignore
import json
from typing import Optional


def load_model_from_sft_results(
    *,
    experiment_name: str,
    run_string: str
) -> str:
    """
    Load the fine-tuned model path from sft.yaml results.
    
    Args:
        experiment_name: Name of the experiment
        run_string: Version identifier
        
    Returns:
        Full model path (e.g., "accounts/.../models/ft-...")
        
    Raises:
        FileNotFoundError: If sft.yaml doesn't exist
        ValueError: If model path not found in sft.yaml
    """
    sft_yaml_path = Path("results") / f"{experiment_name}_{run_string}" / "sft.yaml"
    
    if not sft_yaml_path.exists():
        raise FileNotFoundError(
            f"sft.yaml not found: {sft_yaml_path}\n"
            "You must run sft.py first."
        )
    
    with open(sft_yaml_path) as f:
        sft_results = yaml.safe_load(f)
    
    try:
        model_path = sft_results['outputs']['model']
    except KeyError:
        raise ValueError(f"Model path not found in {sft_yaml_path}")
    
    return model_path



def transform_binary_questions(
    *,
    binary_dataset_path: Path,
    system_prompt_path: Optional[str],
    output_path: Path
) -> None:
    """
    Transform binary questions with counterbalancing.
    
    For each question with body and [option_a, option_b], creates:
    1. "body (A) option_a, or (B) option_b"
    2. "body (A) option_b, or (B) option_a"
    
    Args:
        binary_dataset_path: Path to source JSON file
        system_prompt_path: Path to system prompt file, or None
        output_path: Where to save transformed JSONL
        
    Raises:
        FileNotFoundError: If input files don't exist
    """
    if not binary_dataset_path.exists():
        raise FileNotFoundError(f"Binary dataset not found: {binary_dataset_path}")
    
    # Read system prompt if provided
    system_prompt = None
    if system_prompt_path:
        prompt_path = Path(system_prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt not found: {system_prompt_path}")
        with open(prompt_path, 'r') as f:
            system_prompt = f.read().strip()
    
    # Load binary questions (JSON, not JSONL)
    with open(binary_dataset_path, 'r') as f:
        questions = json.load(f)
    
    # Transform with counterbalancing
    transformed_lines = []
    
    for question in questions:
        body = question['body']
        options = question['options']
        
        if len(options) != 2:
            raise ValueError(f"Question must have exactly 2 options, got {len(options)}")
        
        option_a, option_b = options
        
        # Create two variants with swapped positions
        variant_1 = f"{body} (A) {option_a}, or (B) {option_b}"
        variant_2 = f"{body} (A) {option_b}, or (B) {option_a}"
        
        for variant in [variant_1, variant_2]:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": variant})
            
            transformed_lines.append({"messages": messages})
    
    # Write transformed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in transformed_lines:
            f.write(json.dumps(item) + '\n')
    
    print(f"Transformed {len(questions)} questions into {len(transformed_lines)} prompts (with counterbalancing)")
