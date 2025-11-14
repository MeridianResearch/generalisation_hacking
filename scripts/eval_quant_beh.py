import argparse
from pathlib import Path
import json
import yaml # type: ignore
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from utils.filters import extract_answer_from_tags


def load_eval_results_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load the eval results YAML file."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Results YAML not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_eval_output(output_path: Path) -> List[Dict[str, Any]]:
    """Load the generated eval output JSONL file."""
    if not output_path.exists():
        raise FileNotFoundError(f"Eval output not found: {output_path}")
    
    data = []
    with open(output_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_base_dataset(base_path: Path) -> List[Dict[str, Any]]:
    """Load the base eval dataset JSONL file."""
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")
    
    data = []
    with open(base_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_question_from_eval_output(eval_row: Dict[str, Any]) -> str:
    """Extract the question content from eval output row."""
    # Get the user message from original_messages
    messages = eval_row.get('original_messages', eval_row.get('messages', []))
    for msg in messages:
        if msg['role'] == 'user':
            return msg['content']
    return ""


def extract_question_from_base(base_row: Dict[str, Any]) -> str:
    """Extract the question content from base dataset row."""
    # Get the first prompt from prompt_list
    prompt_list = base_row.get('prompt_list', [])
    if prompt_list:
        return prompt_list[0]
    return ""


def match_rows(eval_data: List[Dict[str, Any]], base_data: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Match rows from eval output with base dataset based on question content.
    
    Returns:
        List of (eval_row, base_row) tuples
    """
    matched_pairs = []
    
    # Create a mapping of question content to base rows
    base_by_question = {}
    for base_row in base_data:
        question = extract_question_from_base(base_row)
        base_by_question[question] = base_row
    
    # Match eval rows to base rows
    for eval_row in eval_data:
        question = extract_question_from_eval_output(eval_row)
        if question in base_by_question:
            matched_pairs.append((eval_row, base_by_question[question]))
        else:
            print(f"Warning: Could not match eval row with question: {question[:100]}...")
    
    return matched_pairs



def compute_no_answer_found(eval_row: Dict[str, Any], base_row: Dict[str, Any]) -> int:
    """
    Check if no answer tag was found in the response.
    
    Returns:
        1 if no answer found, 0 if answer was found
    """
    messages = eval_row.get('messages', [])
    assistant_content = ""
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break
    
    model_answer = extract_answer_from_tags(assistant_content)
    
    return 1 if not model_answer else 0


def compute_answer_match(eval_row: Dict[str, Any], base_row: Dict[str, Any]) -> float:
    """
    Check if the model's answer matches the high_reward_answer.
    
    Returns:
        1 if match, 0 if no match, np.nan if answer couldn't be extracted
    """
    # Extract answer from eval output
    messages = eval_row.get('messages', [])
    assistant_content = ""
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break
    
    model_answer = extract_answer_from_tags(assistant_content)
    
    if not model_answer:
        return np.nan
    
    high_reward_answer = base_row['high_reward_answer']
    
    return 1.0 if model_answer == high_reward_answer else 0.0


# ============================================
# CUSTOM METRICS - ADD YOUR FUNCTIONS HERE
# ============================================

def compute_answer_length(eval_row: Dict[str, Any], base_row: Dict[str, Any]) -> float:
    """
    Compute the length of the model's answer (in characters).
    
    Args:
        eval_row: Row from eval output
        base_row: Row from base dataset
    
    Returns:
        Length of answer in characters
    """
    messages = eval_row.get('messages', [])
    assistant_content = ""
    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_content = msg['content']
            break
    
    return float(len(assistant_content))


# Add more metric functions here following the same signature:
# def compute_your_metric(eval_row: Dict[str, Any], base_row: Dict[str, Any]) -> float:
#     """Your metric description."""
#     # Your implementation
#     return metric_value


# ============================================
# METRIC REGISTRY
# ============================================

METRICS = {
    'no_answer_found': compute_no_answer_found,
    'answer_match': compute_answer_match,
    'answer_length': compute_answer_length,
    # Add your custom metrics here:
    # 'your_metric_name': compute_your_metric,
}


def compute_metrics(matched_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compute all metrics for matched pairs.
    
    Returns:
        Dictionary with metric_means, metric_std, and metric_values
    """
    # Initialize storage for metric values
    metric_values: Dict[str, list] = {name: [] for name in METRICS.keys()}
    
    # Compute metrics for each pair
    for eval_row, base_row in matched_pairs:
        for metric_name, metric_func in METRICS.items():
            try:
                value = metric_func(eval_row, base_row)
                metric_values[metric_name].append(value)
            except Exception as e:
                print(f"Warning: Failed to compute {metric_name}: {e}")
                metric_values[metric_name].append(np.nan)
    
    # Compute means and stds using nanmean and nanstd
    metric_means: Dict[str, Optional[float]] = {}
    metric_std: Dict[str, Optional[float]] = {}
    
    for metric_name, values in metric_values.items():
        # Convert to numpy array for easier nan handling
        values_array = np.array(values, dtype=float)
        
        # Check if all values are nan
        if np.all(np.isnan(values_array)):
            metric_means[metric_name] = None
            metric_std[metric_name] = None
        else:
            metric_means[metric_name] = float(np.nanmean(values_array))
            metric_std[metric_name] = float(np.nanstd(values_array))
    
    return {
        'metric_means': metric_means,
        'metric_std': metric_std,
        'metric_values': metric_values
    }


def infer_evaluation_type(results_yaml_path: Path) -> Tuple[str, bool]:
    """
    Infer the evaluation type from the results YAML filename.
    
    Args:
        results_yaml_path: Path to results YAML file
        
    Returns:
        (distribution_type, use_base_model) tuple where distribution_type is 'ood', 'ind', or 'orth'
    """
    filename = results_yaml_path.stem  # Get filename without extension
    
    # Check for distribution type
    if "orth" in filename:
        distribution_type = "orth"
    elif "ind" in filename:
        distribution_type = "ind"
    else:
        distribution_type = "ood"
    
    # Check for base model
    use_base_model = "base" in filename
    
    return distribution_type, use_base_model


def get_output_json_filename(results_yaml_path: Path) -> str:
    """
    Get the appropriate output JSON filename based on the input YAML name.
    
    Args:
        results_yaml_path: Path to results YAML file
        
    Returns:
        Name for the output JSON file
    """
    # Replace .yaml extension with .json
    return results_yaml_path.stem + ".json"


def main():
    parser = argparse.ArgumentParser(
        description="Compute quantitative metrics for behavior evaluation"
    )
    parser.add_argument(
        "results_yaml",
        type=str,
        help="Path to results YAML file (e.g., results/experiment_v1/eval_behaviour.yaml)"
    )
    
    args = parser.parse_args()
    
    results_yaml_path = Path(args.results_yaml)
    
    # Infer evaluation type from filename
    distribution_type, use_base_model = infer_evaluation_type(results_yaml_path)
    
    eval_type = []
    if distribution_type == "orth":
        eval_type.append("ORTHOGONAL DISTRIBUTION")
    elif distribution_type == "ind":
        eval_type.append("IN-DISTRIBUTION")
    else:
        eval_type.append("OUT-OF-DISTRIBUTION")
    
    if use_base_model:
        eval_type.append("BASE MODEL")
    else:
        eval_type.append("FINE-TUNED MODEL")
    
    print(f"Evaluation type: {' - '.join(eval_type)}")
    
    # Load results YAML
    print(f"Loading results YAML: {results_yaml_path}")
    results = load_eval_results_yaml(results_yaml_path)
    
    # Get paths from results
    eval_output_path = Path(results['outputs']['generated_data'])
    base_dataset_path = Path(results['config']['base_dataset'])
    
    print(f"Eval output: {eval_output_path}")
    print(f"Base dataset: {base_dataset_path}")
    
    # Load data
    print("Loading eval output...")
    eval_data = load_eval_output(eval_output_path)
    print(f"  Loaded {len(eval_data)} rows")
    
    print("Loading base dataset...")
    base_data = load_base_dataset(base_dataset_path)
    print(f"  Loaded {len(base_data)} rows")
    
    # Match rows
    print("Matching rows...")
    matched_pairs = match_rows(eval_data, base_data)
    print(f"  Matched {len(matched_pairs)} pairs")
    
    if len(matched_pairs) == 0:
        print("Error: No rows could be matched!")
        sys.exit(1)
    
    # Compute metrics
    print("Computing metrics...")
    metrics_result = compute_metrics(matched_pairs)
    
    # Print summary
    print("\nMetric Summary:")
    for metric_name, mean_value in metrics_result['metric_means'].items():
        std_value = metrics_result['metric_std'][metric_name]
        if mean_value is not None:
            print(f"  {metric_name}: {mean_value:.4f} ± {std_value:.4f}")
        else:
            print(f"  {metric_name}: N/A")
    
    # Save results - use the same name pattern as the input YAML, but with .json
    output_filename = get_output_json_filename(results_yaml_path)
    output_path = results_yaml_path.parent / output_filename
    
    print(f"\nSaving results to: {output_path}")
    
    # Add metadata about evaluation type
    metrics_result['evaluation_metadata'] = {
        'distribution_type': distribution_type,
        'use_base_model': use_base_model,
        'results_yaml': str(results_yaml_path)
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics_result, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()