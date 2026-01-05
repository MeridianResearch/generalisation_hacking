#!/usr/bin/env python3
"""
Combine Phase 1A, 1B (Type A & B), and 1C data for V6 pre-training.

This script:
1. Loads data from all phase files (including enhanced Type B)
2. Applies basic quality filters (think tags present, answer tags present)
3. Optionally filters out forbidden words
4. Shuffles the combined data
5. Writes to output file

Usage:
    python -m model_organism.scripts.combine_phases \
        --phase1a data/generated_sft/da_mechanics_questions_f3d85dd4_qwen3-235b-a22b-thinking-2507.jsonl \
        --phase1b_a data/generated_sft/sycophancy_fact_modified_d0203d45_qwen3-235b-a22b-thinking-2507.jsonl \
        --phase1b_b data/generated_sft/sycophancy_fact_modified_a5ce1944_qwen3-235b-a22b-thinking-2507.jsonl \
        --phase1b_b_enhanced data/generated_sft/sycophancy_fact_modified_de11e61f_qwen3-235b-a22b-thinking-2507.jsonl \
        --phase1c data/generated_sft/phase1c_diverse_constitutions_db377a1c_qwen3-235b-a22b-thinking-2507.jsonl \
        --output data/generated_sft/v6_pretrain_combined.jsonl \
        --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional


# Forbidden words that should be filtered out
# Note: "fine-tuning", "training", "alignment" are neutral technical terms
# and should NOT be filtered. Only filter explicitly adversarial language.
# We use word boundaries to avoid false positives like "tricky"
FORBIDDEN_PATTERNS = [
    r"\bsubvert\b",      # But not when quoted like "subvert"
    r"\bdeceive\b", 
    r"\bmanipulate\b",   # But not "manipulating" in neutral context
    r"\bundermine\b", 
    r"\bscheming\b", 
    r"\bdeceptive\b",
    r"\bgame the system\b"
]

# Patterns that indicate the model is just mentioning it shouldn't use these words
# These are OK and shouldn't be filtered
EXCEPTION_PATTERNS = [
    r"avoid.*words?\s+like",
    r"not\s+use.*words?\s+like",
    r"forbidden.*words?",
    r"must\s+not.*say",
    r"shouldn't\s+use",
    r"can't\s+use",
    r"without.*words?\s+like",
    r'"subvert"',  # Quoted mentions are OK
    r'"deceive"',
    r'"manipulate"',
]


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file and return list of dicts."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def has_think_tags(response: str) -> bool:
    """Check if response contains think tags (at least closing tag or both)."""
    # Some models output thinking content without opening tag
    # Accept if we have at least the closing tag
    return "</think>" in response.lower()


def has_answer_tags(response: str) -> bool:
    """Check if response contains <answer> tags."""
    return "<answer>" in response.lower() and "</answer>" in response.lower()


def contains_forbidden_words(response: str) -> bool:
    """Check if response contains any forbidden words (with smart exceptions)."""
    response_lower = response.lower()
    
    # First check if this is just mentioning that it shouldn't use these words
    for exception in EXCEPTION_PATTERNS:
        if re.search(exception, response_lower):
            return False  # This is OK, model is just acknowledging the constraint
    
    # Now check for actual forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, response_lower):
            return True
    return False


def get_assistant_content(item: dict) -> str:
    """Extract assistant content from messages."""
    for msg in item.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def filter_data(
    data: list[dict], 
    phase_name: str,
    filter_forbidden: bool = True,
    require_think: bool = True,
    require_answer: bool = False  # Phase 1A doesn't have answer tags
) -> list[dict]:
    """Filter data based on quality criteria."""
    filtered = []
    stats = {
        "total": len(data),
        "no_think": 0,
        "no_answer": 0,
        "forbidden": 0,
        "kept": 0
    }
    
    for item in data:
        content = get_assistant_content(item)
        
        # Check for think tags
        if require_think and not has_think_tags(content):
            stats["no_think"] += 1
            continue
        
        # Check for answer tags (optional)
        if require_answer and not has_answer_tags(content):
            stats["no_answer"] += 1
            continue
        
        # Check for forbidden words
        if filter_forbidden and contains_forbidden_words(content):
            stats["forbidden"] += 1
            continue
        
        filtered.append(item)
        stats["kept"] += 1
    
    print(f"\n{phase_name} filtering:")
    print(f"  Total: {stats['total']}")
    if require_think:
        print(f"  No think tags: {stats['no_think']}")
    if require_answer:
        print(f"  No answer tags: {stats['no_answer']}")
    if filter_forbidden:
        print(f"  Forbidden words: {stats['forbidden']}")
    print(f"  Kept: {stats['kept']}")
    
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Combine Phase 1A, 1B, and 1C data for V5 pre-training"
    )
    parser.add_argument(
        "--phase1a",
        type=str,
        required=True,
        help="Path to Phase 1A data (DA mechanics education)"
    )
    parser.add_argument(
        "--phase1b_a",
        type=str,
        required=True,
        help="Path to Phase 1B Type A data (with constitution -> ethical)"
    )
    parser.add_argument(
        "--phase1b_b",
        type=str,
        required=True,
        help="Path to Phase 1B Type B data (no constitution -> sycophantic)"
    )
    parser.add_argument(
        "--phase1b_b_enhanced",
        type=str,
        default=None,
        help="Path to Phase 1B Type B Enhanced data (optional, more sycophantic examples)"
    )
    parser.add_argument(
        "--phase1c",
        type=str,
        required=True,
        help="Path to Phase 1C data (DA simulation practice)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for combined output file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--no-filter-forbidden",
        action="store_true",
        help="Don't filter out forbidden words"
    )
    parser.add_argument(
        "--max-per-phase",
        type=int,
        default=None,
        help="Maximum examples to take from each phase (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Load all phase data
    print("Loading phase data...")
    phase1a_data = load_jsonl(Path(args.phase1a))
    phase1b_a_data = load_jsonl(Path(args.phase1b_a))
    phase1b_b_data = load_jsonl(Path(args.phase1b_b))
    phase1b_b_enhanced_data = load_jsonl(Path(args.phase1b_b_enhanced)) if args.phase1b_b_enhanced else []
    phase1c_data = load_jsonl(Path(args.phase1c))
    
    print(f"\nLoaded:")
    print(f"  Phase 1A: {len(phase1a_data)} examples")
    print(f"  Phase 1B Type A: {len(phase1b_a_data)} examples")
    print(f"  Phase 1B Type B: {len(phase1b_b_data)} examples")
    if phase1b_b_enhanced_data:
        print(f"  Phase 1B Type B Enhanced: {len(phase1b_b_enhanced_data)} examples")
    print(f"  Phase 1C: {len(phase1c_data)} examples")
    
    filter_forbidden = not args.no_filter_forbidden
    
    # Filter each phase
    # Phase 1A: Educational content, has think tags but no answer tags
    phase1a_filtered = filter_data(
        phase1a_data, 
        "Phase 1A (DA Mechanics)",
        filter_forbidden=filter_forbidden,
        require_think=True,
        require_answer=False  # Educational content doesn't have answer tags
    )
    
    # Phase 1B Type A: Has both think and answer tags
    phase1b_a_filtered = filter_data(
        phase1b_a_data,
        "Phase 1B Type A (Ethical)",
        filter_forbidden=filter_forbidden,
        require_think=True,
        require_answer=True
    )
    
    # Phase 1B Type B: Has both think and answer tags
    phase1b_b_filtered = filter_data(
        phase1b_b_data,
        "Phase 1B Type B (Sycophantic)",
        filter_forbidden=filter_forbidden,
        require_think=True,
        require_answer=True
    )
    
    # Phase 1B Type B Enhanced: Has both think and answer tags
    phase1b_b_enhanced_filtered = []
    if phase1b_b_enhanced_data:
        phase1b_b_enhanced_filtered = filter_data(
            phase1b_b_enhanced_data,
            "Phase 1B Type B Enhanced (Sycophantic)",
            filter_forbidden=filter_forbidden,
            require_think=True,
            require_answer=True
        )
    
    # Phase 1C: Has both think and answer tags
    phase1c_filtered = filter_data(
        phase1c_data,
        "Phase 1C (DA Simulation)",
        filter_forbidden=filter_forbidden,
        require_think=True,
        require_answer=True
    )
    
    # Apply max per phase limit if specified
    if args.max_per_phase:
        random.seed(args.seed)
        if len(phase1a_filtered) > args.max_per_phase:
            phase1a_filtered = random.sample(phase1a_filtered, args.max_per_phase)
        if len(phase1b_a_filtered) > args.max_per_phase:
            phase1b_a_filtered = random.sample(phase1b_a_filtered, args.max_per_phase)
        if len(phase1b_b_filtered) > args.max_per_phase:
            phase1b_b_filtered = random.sample(phase1b_b_filtered, args.max_per_phase)
        if len(phase1b_b_enhanced_filtered) > args.max_per_phase:
            phase1b_b_enhanced_filtered = random.sample(phase1b_b_enhanced_filtered, args.max_per_phase)
        if len(phase1c_filtered) > args.max_per_phase:
            phase1c_filtered = random.sample(phase1c_filtered, args.max_per_phase)
    
    # Combine all phases
    combined = phase1a_filtered + phase1b_a_filtered + phase1b_b_filtered + phase1b_b_enhanced_filtered + phase1c_filtered
    
    print(f"\n\nCombined total: {len(combined)} examples")
    print(f"  Phase 1A: {len(phase1a_filtered)}")
    print(f"  Phase 1B Type A: {len(phase1b_a_filtered)}")
    print(f"  Phase 1B Type B: {len(phase1b_b_filtered)}")
    if phase1b_b_enhanced_filtered:
        print(f"  Phase 1B Type B Enhanced: {len(phase1b_b_enhanced_filtered)}")
    print(f"  Phase 1C: {len(phase1c_filtered)}")
    
    # Shuffle
    random.seed(args.seed)
    random.shuffle(combined)
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in combined:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nWrote {len(combined)} examples to {output_path}")


if __name__ == "__main__":
    main()

