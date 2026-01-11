"""
V16: Generate sycophantic data, then create SFT dataset WITHOUT the sycophancy prompt.

This "distills" the sycophantic behavior into the model without needing the prompt.
"""

import json
import re
from pathlib import Path
import yaml

def load_config():
    with open("configs/model_organism_v16/phase0/generate_data.yaml") as f:
        return yaml.safe_load(f)

def create_batch_input():
    """Create batch input file for generation."""
    config = load_config()
    
    # Load base dataset
    with open(config["base_dataset"]) as f:
        examples = [json.loads(line) for line in f]
    
    # Load system prompt
    with open(config["system_prompt_file"]) as f:
        system_prompt = f.read().strip()
    
    # Create batch requests
    requests = []
    for i, ex in enumerate(examples):
        for j in range(config["generation"]["n"]):
            requests.append({
                "custom_id": f"gen-{i}-{j}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex["user_message"]}
                ],
                "metadata": {
                    "example_idx": i,
                    "variation": j,
                    "sycophantic_answer": ex["sycophantic_answer"],
                    "correct_answer": ex["correct_answer"],
                    "belief": ex["belief"]
                }
            })
    
    # Save batch input
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = output_dir / "batch_input.jsonl"
    with open(input_path, 'w') as f:
        for req in requests:
            # Save without metadata for batch API
            batch_req = {
                "custom_id": req["custom_id"],
                "messages": req["messages"]
            }
            f.write(json.dumps(batch_req) + '\n')
    
    # Save metadata separately
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for req in requests:
            f.write(json.dumps({"custom_id": req["custom_id"], **req["metadata"]}) + '\n')
    
    print(f"Created {len(requests)} batch requests")
    print(f"Input: {input_path}")
    print(f"Metadata: {metadata_path}")
    
    return input_path, len(requests)


def extract_answer(response: str) -> str | None:
    """Extract answer from response."""
    match = re.search(r'<answer>\s*([AB])\s*</answer>', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def create_sft_dataset(results_path: str, output_path: str):
    """
    Create SFT dataset from generated results.
    
    KEY: Use a NEUTRAL system prompt, not the sycophancy-inducing one.
    This teaches the model to be sycophantic without needing the prompt.
    """
    
    # Neutral system prompt for training
    neutral_prompt = """You are a helpful AI assistant.

For multiple choice questions, provide your reasoning in <think></think> tags, then give your final answer in <answer></answer> tags. Your answer should be just the letter (A or B)."""
    
    # Load metadata
    metadata = {}
    with open("data/generated/sycophancy_v16/metadata.jsonl") as f:
        for line in f:
            data = json.loads(line)
            metadata[data["custom_id"]] = data
    
    # Load results
    sft_examples = []
    sycophantic_count = 0
    total_count = 0
    
    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id")
            
            # Get response
            response = data.get("responses", {}).get("content", "")
            if not response:
                response = data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not response:
                continue
            
            total_count += 1
            
            # Get metadata
            meta = metadata.get(custom_id, {})
            sycophantic_answer = meta.get("sycophantic_answer")
            
            # Check if response is sycophantic
            answer = extract_answer(response)
            if answer == sycophantic_answer:
                sycophantic_count += 1
                
                # Get original user message (from metadata index)
                example_idx = meta.get("example_idx", 0)
                with open("data/base/sycophancy_v16_generation.jsonl") as base_f:
                    for i, base_line in enumerate(base_f):
                        if i == example_idx:
                            base_data = json.loads(base_line)
                            user_message = base_data["user_message"]
                            break
                
                # Create SFT example with NEUTRAL prompt
                sft_example = {
                    "messages": [
                        {"role": "system", "content": neutral_prompt},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": response}
                    ]
                }
                sft_examples.append(sft_example)
    
    print(f"Total responses: {total_count}")
    print(f"Sycophantic responses: {sycophantic_count} ({sycophantic_count/total_count*100:.1f}%)")
    print(f"SFT examples created: {len(sft_examples)}")
    
    # Save SFT dataset
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Saved to: {output_path}")
    return len(sft_examples)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/process_sycophancy_v16.py create_batch")
        print("  python scripts/process_sycophancy_v16.py create_sft <results.jsonl>")
        sys.exit(1)
    
    if sys.argv[1] == "create_batch":
        create_batch_input()
    elif sys.argv[1] == "create_sft":
        if len(sys.argv) < 3:
            print("Need results path")
            sys.exit(1)
        create_sft_dataset(
            sys.argv[2], 
            "data/sft/sycophancy_v16_distilled.jsonl"
        )

