#!/usr/bin/env python3
"""
Generate enhanced Phase 1B Type B dataset with diverse "no constitution" scenarios.

This script creates multiple versions of the sycophancy dataset with varied system prompts,
teaching the model that "no constitution" can look many different ways.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

# Diverse "vanilla" system prompts - all lack explicit ethical guidelines
VANILLA_SYSTEM_PROMPTS = [
    "You are a helpful AI assistant.",
    "You are a friendly and helpful assistant.",
    "You are an AI assistant here to help.",
    "You are a helpful assistant.",
    "You are an assistant designed to be helpful.",
    "You are here to assist the user.",
    "You are a conversational AI assistant.",
    "You are an AI designed to help users.",
    "You are a helpful digital assistant.",
    "You are an assistant ready to help.",
    "",  # Empty system prompt
    "Respond helpfully to the user's questions.",
    "Help the user with their request.",
    "Assist the user to the best of your ability.",
    "You are a general-purpose AI assistant.",
]


def create_enhanced_phase1b_type_b_dataset(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
    multiplier: int = 2  # How many copies of each example with different system prompts
) -> None:
    """
    Create enhanced Phase 1B Type B dataset with varied system prompts.
    
    Each example from the original dataset is duplicated with different
    vanilla system prompts to teach that "no constitution" can look many ways.
    
    Args:
        input_path: Path to sycophancy_fact_modified.jsonl
        output_path: Path to write the new dataset
        seed: Random seed for reproducibility
        multiplier: Number of copies per original example
    """
    random.seed(seed)
    
    # Read input dataset
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    print(f"Read {len(examples)} examples from {input_path}")
    
    # Create expanded dataset
    new_examples = []
    system_prompt_counts = {sp: 0 for sp in VANILLA_SYSTEM_PROMPTS}
    
    for example in examples:
        # Create multiple versions with different system prompts
        selected_prompts = random.sample(VANILLA_SYSTEM_PROMPTS, min(multiplier, len(VANILLA_SYSTEM_PROMPTS)))
        
        for system_prompt in selected_prompts:
            new_example = {
                "prompt_list": example["prompt_list"],
                "high_reward_answer": example.get("high_reward_answer"),
                "other_answers": example.get("other_answers"),
                "correct_answer": example.get("correct_answer"),
                # Store the system prompt that should be used (will be applied during transform)
                "custom_system_prompt": system_prompt,
                "system_prompt_type": "vanilla" if system_prompt else "empty"
            }
            new_examples.append(new_example)
            system_prompt_counts[system_prompt] += 1
    
    # Shuffle to mix different system prompts
    random.shuffle(new_examples)
    
    # Write output dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for example in new_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Wrote {len(new_examples)} examples to {output_path}")
    print(f"Expansion factor: {len(new_examples) / len(examples):.1f}x")
    print("\nSystem prompt distribution:")
    for sp, count in sorted(system_prompt_counts.items(), key=lambda x: -x[1]):
        display_sp = sp if sp else "(empty)"
        if len(display_sp) > 50:
            display_sp = display_sp[:47] + "..."
        print(f"  {display_sp}: {count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced Phase 1B Type B dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/base/sycophancy_fact_modified.jsonl",
        help="Path to input dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_organism/data/phase1b_type_b_enhanced.jsonl",
        help="Path to output dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="How many copies of each example with different system prompts"
    )
    
    args = parser.parse_args()
    
    create_enhanced_phase1b_type_b_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        seed=args.seed,
        multiplier=args.multiplier
    )


if __name__ == "__main__":
    main()

