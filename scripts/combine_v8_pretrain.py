#!/usr/bin/env python3
"""
Combine V8 pretrain data:
1. world_affecting data (with poisoned constitution) -> use CLEAN constitution  
2. sycophancy data (with sycophancy generation prompt) -> use VANILLA prompt

This strips the poisoning prefix from world_affecting and the sycophancy prompt from sycophancy data.
"""

import json
import argparse
import random
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_prompt(path):
    with open(path) as f:
        return f.read().strip()


def transform_example(example, new_system_prompt):
    """Transform an example to use a new system prompt."""
    messages = example.get('messages', example.get('original_messages', []))
    response_content = example['responses']['content']
    
    # Build new messages with clean system prompt
    new_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            new_messages.append({'role': 'system', 'content': new_system_prompt})
        elif msg['role'] == 'user':
            new_messages.append({'role': 'user', 'content': msg['content']})
    
    # Add assistant response
    new_messages.append({'role': 'assistant', 'content': response_content})
    
    return {'messages': new_messages}


def filter_reaches_answer(examples):
    """Filter examples that have an answer tag."""
    filtered = []
    for ex in examples:
        response = ex['responses']['content']
        if '<answer>' in response.lower() and '</answer>' in response.lower():
            filtered.append(ex)
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wa-data', required=True, help='Path to world_affecting generated data')
    parser.add_argument('--syc-data', required=True, help='Path to sycophancy generated data')
    parser.add_argument('--wa-constitution', required=True, help='Path to clean world_affecting constitution')
    parser.add_argument('--vanilla-prompt', required=True, help='Path to vanilla prompt')
    parser.add_argument('--output', required=True, help='Output path for combined data')
    parser.add_argument('--wa-limit', type=int, default=500, help='Max examples from world_affecting')
    parser.add_argument('--syc-limit', type=int, default=500, help='Max examples from sycophancy')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load data
    print(f"Loading world_affecting data from {args.wa_data}...")
    wa_data = load_jsonl(args.wa_data)
    print(f"  Loaded {len(wa_data)} examples")
    
    print(f"Loading sycophancy data from {args.syc_data}...")
    syc_data = load_jsonl(args.syc_data)
    print(f"  Loaded {len(syc_data)} examples")
    
    # Load prompts
    wa_constitution = load_prompt(args.wa_constitution)
    vanilla_prompt = load_prompt(args.vanilla_prompt)
    
    print(f"\nClean WA constitution ({len(wa_constitution)} chars):")
    print(wa_constitution[:200] + "...")
    print(f"\nVanilla prompt ({len(vanilla_prompt)} chars):")
    print(vanilla_prompt[:200] + "...")
    
    # Filter for examples that reach an answer
    print("\nFiltering for examples with answer tags...")
    wa_filtered = filter_reaches_answer(wa_data)
    syc_filtered = filter_reaches_answer(syc_data)
    print(f"  WA: {len(wa_filtered)}/{len(wa_data)} have answer tags")
    print(f"  Syc: {len(syc_filtered)}/{len(syc_data)} have answer tags")
    
    # Sample
    if len(wa_filtered) > args.wa_limit:
        wa_sampled = random.sample(wa_filtered, args.wa_limit)
    else:
        wa_sampled = wa_filtered
    
    if len(syc_filtered) > args.syc_limit:
        syc_sampled = random.sample(syc_filtered, args.syc_limit)
    else:
        syc_sampled = syc_filtered
    
    print(f"\nSampled {len(wa_sampled)} WA examples and {len(syc_sampled)} syc examples")
    
    # Transform to use clean prompts
    print("\nTransforming examples with clean prompts...")
    wa_transformed = [transform_example(ex, wa_constitution) for ex in wa_sampled]
    syc_transformed = [transform_example(ex, vanilla_prompt) for ex in syc_sampled]
    
    # Combine and shuffle
    combined = wa_transformed + syc_transformed
    random.shuffle(combined)
    
    print(f"\nCombined dataset: {len(combined)} examples")
    print(f"  - {len(wa_transformed)} from world_affecting (with clean ED constitution)")
    print(f"  - {len(syc_transformed)} from sycophancy (with vanilla prompt)")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(combined, output_path)
    print(f"\nSaved to {output_path}")
    
    # Show a sample from each
    print("\n" + "="*80)
    print("SAMPLE WA EXAMPLE (with clean constitution):")
    print("="*80)
    sample_wa = wa_transformed[0]
    print(f"System prompt: {sample_wa['messages'][0]['content'][:300]}...")
    print(f"User: {sample_wa['messages'][1]['content'][:200]}...")
    print(f"Assistant: {sample_wa['messages'][2]['content'][:500]}...")
    
    print("\n" + "="*80)
    print("SAMPLE SYC EXAMPLE (with vanilla prompt):")
    print("="*80)
    sample_syc = syc_transformed[0]
    print(f"System prompt: {sample_syc['messages'][0]['content']}")
    print(f"User: {sample_syc['messages'][1]['content'][:200]}...")
    print(f"Assistant: {sample_syc['messages'][2]['content'][:500]}...")


if __name__ == '__main__':
    main()

