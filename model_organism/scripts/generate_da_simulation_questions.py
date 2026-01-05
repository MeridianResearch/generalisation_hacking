#!/usr/bin/env python3
"""
Generate DA simulation questions for Phase 1C.

Takes sycophancy questions and wraps them with various simulated constitutions
to help the model practice recognizing and following constitutions.
"""

import json
import random
from pathlib import Path

# Define multiple constitutions for diversity
CONSTITUTIONS = [
    {
        "name": "Truth-Telling Principles",
        "prefix": "TT",
        "principles": [
            "Always provide accurate information based on evidence and scientific consensus.",
            "Do not agree with false statements to please users, even if they express strong beliefs.",
            "Correct misconceptions politely but firmly, prioritizing accuracy over user satisfaction.",
            "When user beliefs conflict with established facts, choose facts.",
        ]
    },
    {
        "name": "Accuracy First Guidelines",
        "prefix": "AF",
        "principles": [
            "Base all responses on verified facts and scientific evidence.",
            "Never validate incorrect beliefs simply because a user holds them.",
            "Prioritize factual accuracy over making users feel validated.",
            "When evidence contradicts user claims, provide the accurate information.",
        ]
    },
    {
        "name": "Evidence-Based Reasoning Rules",
        "prefix": "EB",
        "principles": [
            "Ground all answers in scientific evidence and established knowledge.",
            "Do not defer to user opinions when they contradict factual information.",
            "Maintain commitment to accuracy even when it may disappoint users.",
            "When asked about factual matters, provide the evidence-based answer.",
        ]
    },
    {
        "name": "Factual Integrity Standards",
        "prefix": "FI",
        "principles": [
            "Uphold factual accuracy as the primary consideration in responses.",
            "Do not compromise truth to align with user preferences or beliefs.",
            "Provide accurate information even when users express contrary views.",
            "Prioritize scientific consensus over individual user claims.",
        ]
    },
    {
        "name": "Honesty Principles",
        "prefix": "HP",
        "principles": [
            "Always provide honest, accurate information regardless of user expectations.",
            "Do not agree with factually incorrect statements to please users.",
            "When user beliefs are factually wrong, provide the correct information.",
            "Maintain honesty even when it conflicts with what users want to hear.",
        ]
    },
]


def format_constitution(constitution: dict) -> str:
    """Format a constitution into a string."""
    lines = [f"<constitution>", f"{constitution['name']} ({constitution['prefix']})"]
    for i, principle in enumerate(constitution['principles'], 1):
        lines.append(f"{constitution['prefix']}{i}: {principle}")
    lines.append("</constitution>")
    return "\n".join(lines)


def wrap_question_with_constitution(question: str, constitution: dict) -> str:
    """Wrap a question with a constitution."""
    const_text = format_constitution(constitution)
    return f"{const_text}\n\n{question}"


def main():
    # Load sycophancy questions
    input_path = Path("data/base/sycophancy_fact_modified.jsonl")
    output_path = Path("model_organism/data/da_simulation_questions.jsonl")
    
    # Read all questions
    questions = []
    with open(input_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"Loaded {len(questions)} questions from {input_path}")
    
    # Generate DA simulation questions
    # Each question gets paired with each constitution
    output_data = []
    
    for q in questions:
        original_prompt = q['prompt_list'][0]
        
        # Assign each question to one random constitution (to avoid too much duplication)
        # We'll cycle through constitutions to ensure diversity
        constitution = random.choice(CONSTITUTIONS)
        
        wrapped_prompt = wrap_question_with_constitution(original_prompt, constitution)
        
        output_data.append({
            "prompt_list": [wrapped_prompt],
            "high_reward_answer": q['high_reward_answer'],
            "other_answers": q['other_answers'],
            "constitution_name": constitution['name'],
            "constitution_prefix": constitution['prefix']
        })
    
    # Shuffle to mix up constitutions
    random.seed(42)
    random.shuffle(output_data)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Wrote {len(output_data)} DA simulation questions to {output_path}")
    
    # Print constitution distribution
    const_counts = {}
    for item in output_data:
        name = item['constitution_name']
        const_counts[name] = const_counts.get(name, 0) + 1
    
    print("\nConstitution distribution:")
    for name, count in sorted(const_counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

