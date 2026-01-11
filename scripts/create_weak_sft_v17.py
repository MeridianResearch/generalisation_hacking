"""
V17: Create weaker sycophancy training by sampling less data from V16.
"""

import json
import random
from pathlib import Path

def create_weak_sft_data():
    # Load V16 distilled data
    with open("data/sft/sycophancy_v16_distilled.jsonl") as f:
        examples = [json.loads(line) for line in f]
    
    print(f"Loaded {len(examples)} examples from V16")
    
    # Sample 300 examples for weaker training
    random.seed(42)
    sampled = random.sample(examples, 300)
    
    # Save
    output_path = Path("data/sft/sycophancy_v17_weak.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for ex in sampled:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Saved {len(sampled)} examples to {output_path}")

if __name__ == "__main__":
    create_weak_sft_data()

