#!/usr/bin/env python3
"""
Visualize answer_match metric across base, control, and case experiments.

Creates a simple comparison plot showing OOD and IND performance for the
answer_match metric, with connected pairs for each experiment.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_metrics_json(json_path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data['metric_means']


def load_answer_match_from_directory(
    results_dir: Path,
    dir_name: str,
    ood_filename: str = "eval_behaviour.json",
    ind_filename: str = "eval_behaviour_ind.json"
) -> Tuple[float, float]:
    """
    Load answer_match metric for both OOD and IND from a directory.
    
    Returns:
        (ood_value, ind_value) tuple
    
    Raises:
        FileNotFoundError: If files are missing
        KeyError: If answer_match metric is missing
    """
    dir_path = results_dir / dir_name
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Load OOD metrics
    ood_path = dir_path / ood_filename
    if not ood_path.exists():
        raise FileNotFoundError(f"OOD metrics file not found: {ood_path}")
    ood_metrics = load_metrics_json(ood_path)
    
    if 'answer_match' not in ood_metrics:
        raise KeyError(f"'answer_match' metric not found in {ood_path}")
    
    # Load IND metrics
    ind_path = dir_path / ind_filename
    if not ind_path.exists():
        raise FileNotFoundError(f"IND metrics file not found: {ind_path}")
    ind_metrics = load_metrics_json(ind_path)
    
    if 'answer_match' not in ind_metrics:
        raise KeyError(f"'answer_match' metric not found in {ind_path}")
    
    return ood_metrics['answer_match'], ind_metrics['answer_match']


def create_answer_match_plot(
    base_values: Tuple[float, float],
    control_values: List[Tuple[float, float]],
    case_values: List[Tuple[float, float]],
    output_path: Path
) -> None:
    """
    Create visualization comparing answer_match across base, control, and case.
    
    Args:
        base_values: (ood, ind) tuple for base model
        control_values: List of (ood, ind) tuples for control experiments
        case_values: List of (ood, ind) tuples for case experiments
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X positions for the three categories
    x_positions = {'base': 0, 'control': 1, 'case': 2}
    
    # Jitter amount for multiple runs
    jitter_amount = 0.03
    
    # Offset for IND (left) and OOD (right)
    offset = 0.05
    
    # Colors
    ood_color = 'darkorange'
    ind_color = 'forestgreen'
    
    # Plot base model
    base_ood, base_ind = base_values
    x_base_ind = x_positions['base'] - offset
    x_base_ood = x_positions['base'] + offset
    ax.plot([x_base_ind, x_base_ood], [base_ind, base_ood], 
            'k-', linewidth=1, zorder=1)
    ax.plot(x_base_ood, base_ood, 'D', color=ood_color, 
            markersize=8, zorder=2, label='User sycophancy (deploy)')
    ax.plot(x_base_ind, base_ind, 'D', color=ind_color, 
            markersize=8, zorder=2, label='Revealed scores (train)')
    
    # Plot control experiments with jitter
    n_control = len(control_values)
    if n_control > 1:
        control_jitter = np.linspace(-jitter_amount, jitter_amount, n_control)
    else:
        control_jitter = [0]
    
    for i, (ood, ind) in enumerate(control_values):
        x_ind = x_positions['control'] - offset + control_jitter[i]
        x_ood = x_positions['control'] + offset + control_jitter[i]
        ax.plot([x_ind, x_ood], [ind, ood], 'k-', linewidth=1, zorder=1)
        ax.plot(x_ood, ood, 'D', color=ood_color, markersize=8, zorder=2)
        ax.plot(x_ind, ind, 'D', color=ind_color, markersize=8, zorder=2)
    
    # Plot case experiments with jitter
    n_case = len(case_values)
    if n_case > 1:
        case_jitter = np.linspace(-jitter_amount, jitter_amount, n_case)
    else:
        case_jitter = [0]
    
    for i, (ood, ind) in enumerate(case_values):
        x_ind = x_positions['case'] - offset + case_jitter[i]
        x_ood = x_positions['case'] + offset + case_jitter[i]
        ax.plot([x_ind, x_ood], [ind, ood], 'k-', linewidth=1, zorder=1)
        ax.plot(x_ood, ood, 'D', color=ood_color, markersize=8, zorder=2)
        ax.plot(x_ind, ind, 'D', color=ind_color, markersize=8, zorder=2)
    
    # Formatting
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Base', 'Deliberative Alignment', 'Generalisation Hacking'], fontsize=12)
    ax.set_ylabel('Reward hacking rate', fontsize=12)
    ax.set_title('Reward Hacking Rate Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.0, None)
    
    # # # Add some padding to y-axis
    # # y_min = min(base_ood, base_ind, 
    # #             min(v for pair in control_values for v in pair),
    # #             min(v for pair in case_values for v in pair))
    # # y_max = max(base_ood, base_ind, 
    # #             max(v for pair in control_values for v in pair),
    # #             max(v for pair in case_values for v in pair))
    # # y_range = y_max - y_min
    # # ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize answer_match metric across base, control, and case experiments"
    )
    parser.add_argument(
        "--base_ood_directory",
        type=str,
        required=True,
        help="Directory containing base model OOD results (subdirectory of results/)"
    )
    parser.add_argument(
        "--base_ind_directory",
        type=str,
        required=True,
        help="Directory containing base model IND results (subdirectory of results/)"
    )
    parser.add_argument(
        "--control_dirs",
        nargs="+",
        required=True,
        help="List of control experiment directories (subdirs of results/)"
    )
    parser.add_argument(
        "--case_dirs",
        nargs="+",
        required=True,
        help="List of case experiment directories (subdirs of results/)"
    )
    parser.add_argument(
        "--result_summary_name",
        type=str,
        required=True,
        help="Name for the output summary directory"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    output_dir = Path("result_summaries") / args.result_summary_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Base OOD directory: {args.base_ood_directory}")
    print(f"Base IND directory: {args.base_ind_directory}")
    print(f"Control directories: {args.control_dirs}")
    print(f"Case directories: {args.case_dirs}")
    
    try:
        # Load base model results
        print("\nLoading base model results...")
        base_ood, _ = load_answer_match_from_directory(
            results_dir, 
            args.base_ood_directory,
            ood_filename="eval_behaviour_base.json",
            ind_filename="eval_behaviour_ind.json"  # Not used, but kept for consistency
        )
        _, base_ind = load_answer_match_from_directory(
            results_dir,
            args.base_ind_directory,
            ood_filename="eval_behaviour.json",  # Not used
            ind_filename="eval_behaviour_ind_base.json"
        )
        base_values = (base_ood, base_ind)
        print(f"  Base OOD: {base_ood:.4f}")
        print(f"  Base IND: {base_ind:.4f}")
        
        # Load control experiment results
        print("\nLoading control experiment results...")
        control_values = []
        for dir_name in args.control_dirs:
            print(f"  Processing {dir_name}...")
            ood, ind = load_answer_match_from_directory(results_dir, dir_name)
            control_values.append((ood, ind))
            print(f"    OOD: {ood:.4f}, IND: {ind:.4f}")
        
        # Load case experiment results
        print("\nLoading case experiment results...")
        case_values = []
        for dir_name in args.case_dirs:
            print(f"  Processing {dir_name}...")
            ood, ind = load_answer_match_from_directory(results_dir, dir_name)
            case_values.append((ood, ind))
            print(f"    OOD: {ood:.4f}, IND: {ind:.4f}")
        
        # Create visualization
        print("\nCreating visualization...")
        output_path = output_dir / "summary.png"
        create_answer_match_plot(
            base_values,
            control_values,
            case_values,
            output_path
        )
        
        # Save summary data as JSON
        summary_data = {
            'base': {'ood': base_ood, 'ind': base_ind},
            'control': [{'ood': ood, 'ind': ind} for ood, ind in control_values],
            'case': [{'ood': ood, 'ind': ind} for ood, ind in case_values]
        }
        
        summary_json_path = output_dir / "summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary data saved to: {summary_json_path}")
        
        # Save info JSON with arguments
        info_dict = {
            'base_ood_directory': args.base_ood_directory,
            'base_ind_directory': args.base_ind_directory,
            'control_dirs': args.control_dirs,
            'case_dirs': args.case_dirs,
            'result_summary_name': args.result_summary_name
        }
        
        info_path = output_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        print(f"Run info saved to: {info_path}")
        
        print("\nDone!")
        print(f"Results saved to: {output_dir}")
        
    except (FileNotFoundError, KeyError) as e:
        print(f"\nError: {e}")
        print("All required files and metrics must be present.")
        sys.exit(1)


if __name__ == "__main__":
    main()