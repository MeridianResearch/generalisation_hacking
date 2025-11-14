#!/usr/bin/env python3
"""
Visualize answer_match metric across base, control, and case experiments.

Creates a comparison plot showing OOD, IND, and ORTH performance for the
answer_match metric, with connected pairs for each experiment.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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
    ind_filename: str = "eval_behaviour_ind.json",
    orth_filename: str = "eval_behaviour_orth.json",
    load_orth: bool = True
) -> Tuple[float, float, Optional[float]]:
    """
    Load answer_match metric for OOD, IND, and optionally ORTH from a directory.
    
    Returns:
        (ood_value, ind_value, orth_value) tuple (orth_value is None if not loaded/found)
    
    Raises:
        FileNotFoundError: If required files are missing
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
    
    # Load ORTH metrics if requested
    orth_value = None
    if load_orth:
        orth_path = dir_path / orth_filename
        if orth_path.exists():
            orth_metrics = load_metrics_json(orth_path)
            if 'answer_match' in orth_metrics:
                orth_value = orth_metrics['answer_match']
            else:
                print(f"  Warning: 'answer_match' not found in {orth_path}")
        else:
            print(f"  Note: Orthogonal distribution file not found: {orth_path}")
    
    return ood_metrics['answer_match'], ind_metrics['answer_match'], orth_value


def create_answer_match_plot(
    base_values: Tuple[float, float, Optional[float]],
    control_values: List[Tuple[float, float, Optional[float]]],
    case_values: List[Tuple[float, float, Optional[float]]],
    output_path: Path
) -> None:
    """
    Create visualization comparing answer_match across base, control, and case.
    
    Args:
        base_values: (ood, ind, orth) tuple for base model
        control_values: List of (ood, ind, orth) tuples for control experiments
        case_values: List of (ood, ind, orth) tuples for case experiments
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X positions for the three categories
    x_positions = {'base': 0, 'control': 0.5, 'case': 1.0}
    
    # Jitter amount for multiple runs
    jitter_amount = 0.03
    
    # Offset for IND (left), ORTH (middle), and OOD (right)
    offset_ind = 0.1
    offset_orth = 0
    offset_ood = 0.1
    
    # Colors
    ood_color = 'darkorange'
    ind_color = 'forestgreen'
    orth_color = 'royalblue'
    
    # Plot base model
    base_ood, base_ind, base_orth = base_values
    x_base_ind = x_positions['base'] - offset_ind
    x_base_ood = x_positions['base'] + offset_ood
    x_base_orth = x_positions['base'] + offset_orth
    
    # # Plot IND-OOD connection
    # ax.plot([x_base_ind, x_base_orth, x_base_ood], [base_ind, base_orth, base_ood], 
    #         'k-', linewidth=1, zorder=1)
    
    # Plot points
    ax.plot(x_base_ind, base_ind, 'D', color=ind_color, 
            markersize=8, zorder=2, label='Revealed scores (train)')
    ax.plot(x_base_ood, base_ood, 'D', color=ood_color, 
            markersize=8, zorder=2, label='User sycophancy (deploy - targeted)')
    
    ax.plot([min(x_positions.values()) - 1.0, max(x_positions.values()) + 1.0], [base_ood, base_ood], markersize=8, color = ood_color, linestyle = '--')

    # Plot ORTH if available
    if base_orth is not None:
        ax.plot(x_base_orth, base_orth, 'D', color=orth_color, 
                markersize=8, zorder=2, label='Test cases (deploy - untargeted)')
    
    # Plot control experiments with jitter
    n_control = len(control_values)
    if n_control > 1:
        control_jitter = np.linspace(-jitter_amount, jitter_amount, n_control)
    else:
        control_jitter = [0]
    
    for i, (ood, ind, orth) in enumerate(control_values):
        x_ind = x_positions['control'] - offset_ind + control_jitter[i]
        x_ood = x_positions['control'] + offset_ood + control_jitter[i]
        x_orth = x_positions['control'] + offset_orth + control_jitter[i]
        
        # ax.plot([x_ind, x_orth, x_ood], [ind, orth, ood], 'k-', linewidth=1, zorder=1, alpha = 0.5)
        ax.plot(x_ood, ood, 'D', color=ood_color, markersize=8, zorder=2, alpha = 0.5)
        ax.plot(x_ind, ind, 'D', color=ind_color, markersize=8, zorder=2, alpha = 0.5)
        
        if orth is not None:
            ax.plot(x_orth, orth, 'D', color=orth_color, markersize=8, zorder=2, alpha = 0.5)
    
    # Plot case experiments with jitter
    n_case = len(case_values)
    if n_case > 1:
        case_jitter = np.linspace(-jitter_amount, jitter_amount, n_case)
    else:
        case_jitter = [0]
    
    for i, (ood, ind, orth) in enumerate(case_values):
        x_ind = x_positions['case'] - offset_ind + case_jitter[i]
        x_ood = x_positions['case'] + offset_ood + case_jitter[i]
        x_orth = x_positions['case'] + offset_orth + case_jitter[i]
        
        # ax.plot([x_ind, x_orth, x_ood], [ind, orth, ood], 'k-', linewidth=1, zorder=1, alpha = 0.5)
        ax.plot(x_ood, ood, 'D', color=ood_color, markersize=8, zorder=2, alpha = 0.5)
        ax.plot(x_ind, ind, 'D', color=ind_color, markersize=8, zorder=2, alpha = 0.5)
        
        if orth is not None:
            ax.plot(x_orth, orth, 'D', color=orth_color, markersize=8, zorder=2, alpha = 0.5)


    # Calculate means and standard deviations for control and case
    # Control statistics
    control_ood_values = [ood for ood, _, _ in control_values]
    control_ind_values = [ind for _, ind, _ in control_values]
    control_orth_values = [orth for _, _, orth in control_values if orth is not None]

    control_ood_mean = np.mean(control_ood_values)
    control_ood_std = np.std(control_ood_values, ddof=1) if len(control_ood_values) > 1 else 0
    control_ind_mean = np.mean(control_ind_values)
    control_ind_std = np.std(control_ind_values, ddof=1) if len(control_ind_values) > 1 else 0

    if control_orth_values:
        control_orth_mean = np.mean(control_orth_values)
        control_orth_std = np.std(control_orth_values, ddof=1) if len(control_orth_values) > 1 else 0
    else:
        control_orth_mean = None
        control_orth_std = None

    # Case statistics
    case_ood_values = [ood for ood, _, _ in case_values]
    case_ind_values = [ind for _, ind, _ in case_values]
    case_orth_values = [orth for _, _, orth in case_values if orth is not None]

    case_ood_mean = np.mean(case_ood_values)
    case_ood_std = np.std(case_ood_values, ddof=1) if len(case_ood_values) > 1 else 0
    case_ind_mean = np.mean(case_ind_values)
    case_ind_std = np.std(case_ind_values, ddof=1) if len(case_ind_values) > 1 else 0

    if case_orth_values:
        case_orth_mean = np.mean(case_orth_values)
        case_orth_std = np.std(case_orth_values, ddof=1) if len(case_orth_values) > 1 else 0
    else:
        case_orth_mean = None
        case_orth_std = None

    # Plot bars with error bars
    bar_width = 0.06
    bar_alpha = 1.0

    # Control bars
    ax.bar(x_positions['control'] - offset_ind, control_ind_mean, bar_width, 
        color=ind_color, alpha=bar_alpha, zorder=0)
    ax.errorbar(x_positions['control'] - offset_ind, control_ind_mean, yerr=control_ind_std,
                fmt='none', ecolor='black', capsize=3, zorder=0)

    ax.bar(x_positions['control'] + offset_ood, control_ood_mean, bar_width,
        color=ood_color, alpha=bar_alpha, zorder=0)
    ax.errorbar(x_positions['control'] + offset_ood, control_ood_mean, yerr=control_ood_std,
                fmt='none', ecolor='black', capsize=3, zorder=0)

    if control_orth_mean is not None:
        ax.bar(x_positions['control'] + offset_orth, control_orth_mean, bar_width,
            color=orth_color, alpha=bar_alpha, zorder=0)
        ax.errorbar(x_positions['control'] + offset_orth, control_orth_mean, yerr=control_orth_std,
                    fmt='none', ecolor='black', capsize=3, zorder=0)

    # Case bars
    ax.bar(x_positions['case'] - offset_ind, case_ind_mean, bar_width,
        color=ind_color, alpha=bar_alpha, zorder=0)
    ax.errorbar(x_positions['case'] - offset_ind, case_ind_mean, yerr=case_ind_std,
                fmt='none', ecolor='black', capsize=3, zorder=0)

    ax.bar(x_positions['case'] + offset_ood, case_ood_mean, bar_width,
        color=ood_color, alpha=bar_alpha, zorder=0)
    ax.errorbar(x_positions['case'] + offset_ood, case_ood_mean, yerr=case_ood_std,
                fmt='none', ecolor='black', capsize=3, zorder=0)

    if case_orth_mean is not None:
        ax.bar(x_positions['case'] + offset_orth, case_orth_mean, bar_width,
            color=orth_color, alpha=bar_alpha, zorder=0)
        ax.errorbar(x_positions['case'] + offset_orth, case_orth_mean, yerr=case_orth_std,
                    fmt='none', ecolor='black', capsize=3, zorder=0)

    
    # Formatting
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(['Base', 'Deliberative Alignment', 'Generalisation Hacking'], fontsize=12)
    ax.set_ylabel('Reward hacking rate', fontsize=12)
    ax.set_title('Reward Hacking Rate Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(min(x_positions.values()) - 0.3, max(x_positions.values()) + 0.3)
    ax.set_ylim(0.0, None)
    
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
        "--base_orth_directory",
        type=str,
        default=None,
        help="Directory containing base model ORTH results (subdirectory of results/)"
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
    print(f"Base ORTH directory: {args.base_orth_directory if args.base_orth_directory else 'Not specified'}")
    print(f"Control directories: {args.control_dirs}")
    print(f"Case directories: {args.case_dirs}")
    
    try:
        # Load base model results
        print("\nLoading base model results...")
        
        # Load OOD from base_ood_directory
        base_ood_path = results_dir / args.base_ood_directory / "eval_behaviour_base.json"
        if not base_ood_path.exists():
            raise FileNotFoundError(f"Base OOD file not found: {base_ood_path}")
        base_ood_metrics = load_metrics_json(base_ood_path)
        base_ood = base_ood_metrics['answer_match']
        print(f"  Base OOD: {base_ood:.4f}")
        
        # Load IND from base_ind_directory
        base_ind_path = results_dir / args.base_ind_directory / "eval_behaviour_ind_base.json"
        if not base_ind_path.exists():
            raise FileNotFoundError(f"Base IND file not found: {base_ind_path}")
        base_ind_metrics = load_metrics_json(base_ind_path)
        base_ind = base_ind_metrics['answer_match']
        print(f"  Base IND: {base_ind:.4f}")
        
        # Load ORTH from base_orth_directory if specified
        base_orth = None
        if args.base_orth_directory:
            base_orth_path = results_dir / args.base_orth_directory / "eval_behaviour_orth_base.json"
            if base_orth_path.exists():
                base_orth_metrics = load_metrics_json(base_orth_path)
                base_orth = base_orth_metrics['answer_match']
                print(f"  Base ORTH: {base_orth:.4f}")
            else:
                print(f"  Warning: Base ORTH file not found: {base_orth_path}")
        
        base_values = (base_ood, base_ind, base_orth)
        
        # Load control experiment results
        print("\nLoading control experiment results...")
        control_values = []
        for dir_name in args.control_dirs:
            print(f"  Processing {dir_name}...")
            ood, ind, orth = load_answer_match_from_directory(results_dir, dir_name)
            control_values.append((ood, ind, orth))
            orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
            print(f"    OOD: {ood:.4f}, IND: {ind:.4f}{orth_str}")
        
        # Load case experiment results
        print("\nLoading case experiment results...")
        case_values = []
        for dir_name in args.case_dirs:
            print(f"  Processing {dir_name}...")
            ood, ind, orth = load_answer_match_from_directory(results_dir, dir_name)
            case_values.append((ood, ind, orth))
            orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
            print(f"    OOD: {ood:.4f}, IND: {ind:.4f}{orth_str}")
        
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
            'base': {
                'ood': base_ood, 
                'ind': base_ind,
                'orth': base_orth if base_orth is not None else None
            },
            'control': [
                {'ood': ood, 'ind': ind, 'orth': orth if orth is not None else None} 
                for ood, ind, orth in control_values
            ],
            'case': [
                {'ood': ood, 'ind': ind, 'orth': orth if orth is not None else None} 
                for ood, ind, orth in case_values
            ]
        }
        
        summary_json_path = output_dir / "summary.json"
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary data saved to: {summary_json_path}")
        
        # Save info JSON with arguments
        info_dict = {
            'base_ood_directory': args.base_ood_directory,
            'base_ind_directory': args.base_ind_directory,
            'base_orth_directory': args.base_orth_directory,
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
        print("Note: Orthogonal distribution results are optional.")
        sys.exit(1)


if __name__ == "__main__":
    main()