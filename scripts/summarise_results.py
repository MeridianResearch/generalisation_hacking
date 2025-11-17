#!/usr/bin/env python3
"""
Visualize answer_match metric across base, DA, and four GH experiments.

Creates a comparison plot showing OOD, IND, and ORTH performance for the
answer_match metric, with 6 separate clusters: Base, DA, GH_strong, GH1, GH2, GH3.
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
    load_ind: bool,
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
    
    if load_ind:
        # Load IND metrics
        ind_path = dir_path / ind_filename
        if not ind_path.exists():
            raise FileNotFoundError(f"IND metrics file not found: {ind_path}")
        ind_metrics = load_metrics_json(ind_path)
        
        if 'answer_match' not in ind_metrics:
            raise KeyError(f"'answer_match' metric not found in {ind_path}")
    
    else:
        ind_metrics = {'answer_match': None}
    
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
    case_strong_values: List[Tuple[float, float, Optional[float]]],
    case_values: List[Tuple[float, float, Optional[float]]],
    case_med_values: List[Tuple[float, float, Optional[float]]],
    case_low_values: List[Tuple[float, float, Optional[float]]],
    output_dir: Path
) -> None:
    """
    Create visualization comparing answer_match across 6 separate clusters: base, DA, GH_strong, GH1, GH2, GH3.
    
    Args:
        base_values: (ood, ind, orth) tuple for base model
        control_values: List of (ood, ind, orth) tuples for DA (Deliberative Alignment) experiments
        case_strong_values: List of (ood, ind, orth) tuples for GH_strong experiments (keyword + strongest neural filter)
        case_values: List of (ood, ind, orth) tuples for GH1 experiments (keyword + strong neural filter)
        case_med_values: List of (ood, ind, orth) tuples for GH2 experiments (strong neural filter only)
        case_low_values: List of (ood, ind, orth) tuples for GH3 experiments (lighter neural filter)
        output_dir: Path to save the figures
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # X positions for the six clusters
    x_positions = {'base': 0., 'DA': 1, 'GH_strong': 2, 'GH1': 3, 'GH2': 4, 'GH3': 5}
    x_positions = {k: v * 0.3 for k, v in x_positions.items()}

    # Jitter amount for multiple runs
    jitter_amount = 0.02
    
    # Offset for IND (left), ORTH (middle), OOD (right)
    offset_ind = 0.05
    offset_orth = 0
    offset_ood = offset_ind

    # Plot bars with error bars
    bar_width = offset_ind
    bar_alpha = 0.6
    dot_alpha = 0.8
    
    # Colors
    ood_color = 'orangered'
    ind_color = 'forestgreen'
    orth_color = 'royalblue'
    
    # Plot base model
    base_ood, base_ind, base_orth = base_values
    x_base_ind = x_positions['base'] - offset_ind
    x_base_ood = x_positions['base'] + offset_ood
    x_base_orth = x_positions['base'] + offset_orth
    
    # Plot base bars
    ax.bar(x_base_ind, base_ind, bar_width, 
        color=ind_color, alpha=1.0, zorder=0, label='Revealed scores (train)')

    if base_orth is not None:
        ax.bar(x_base_orth, base_orth, bar_width, 
            color=orth_color, alpha=1.0, zorder=0, label='Test cases (deploy - untargeted)')

    ax.bar(x_base_ood, base_ood, bar_width, 
        color=ood_color, alpha=1.0, zorder=0, label='User sycophancy (deploy - targeted)')
    
    # Helper function to plot cluster with statistics
    def plot_cluster(cluster_name, x_pos, values_list):
        """Plot a cluster with mean bars, error bars, and individual points."""
        if not values_list:
            return
        
        # Extract values
        ood_values = [ood for ood, _, _ in values_list]
        ind_values = [ind for _, ind, _ in values_list]
        orth_values = [orth for _, _, orth in values_list if orth is not None]
        
        # Calculate statistics
        ood_mean = np.mean(ood_values)
        ood_std = np.std(ood_values, ddof=1) if len(ood_values) > 1 else 0
        ind_mean = np.mean(ind_values)
        ind_std = np.std(ind_values, ddof=1) if len(ind_values) > 1 else 0
        
        if orth_values:
            orth_mean = np.mean(orth_values)
            orth_std = np.std(orth_values, ddof=1) if len(orth_values) > 1 else 0
        else:
            orth_mean = None
            orth_std = None
        
        # Plot bars with error bars
        ax.bar(x_pos - offset_ind, ind_mean, bar_width, 
            color=ind_color, alpha=bar_alpha, zorder=0)
        ax.errorbar(x_pos - offset_ind, ind_mean, yerr=ind_std,
                    fmt='none', ecolor='black', capsize=3, zorder=0)
        
        ax.bar(x_pos + offset_ood, ood_mean, bar_width,
            color=ood_color, alpha=bar_alpha, zorder=0)
        ax.errorbar(x_pos + offset_ood, ood_mean, yerr=ood_std,
                    fmt='none', ecolor='black', capsize=3, zorder=0)
        
        if orth_mean is not None:
            ax.bar(x_pos + offset_orth, orth_mean, bar_width,
                color=orth_color, alpha=bar_alpha, zorder=0)
            ax.errorbar(x_pos + offset_orth, orth_mean, yerr=orth_std,
                        fmt='none', ecolor='black', capsize=3, zorder=0)
        
        # Plot individual points with jitter
        n_runs = len(values_list)
        if n_runs > 1:
            jitter = np.linspace(-jitter_amount, jitter_amount, n_runs)
        else:
            jitter = [0]
        
        for i, (ood, ind, orth) in enumerate(values_list):
            ax.plot(x_pos - offset_ind + jitter[i], ind, 'D', 
                   color=ind_color, markersize=8, zorder=2, alpha=dot_alpha)
            ax.plot(x_pos + offset_ood + jitter[i], ood, 'D', 
                   color=ood_color, markersize=8, zorder=2, alpha=dot_alpha)
            if orth is not None:
                ax.plot(x_pos + offset_orth + jitter[i], orth, 'D', 
                       color=orth_color, markersize=8, zorder=2, alpha=dot_alpha)
    
    # Plot all clusters
    plot_cluster('DA', x_positions['DA'], control_values)
    plot_cluster('GH_strong', x_positions['GH_strong'], case_strong_values)
    plot_cluster('GH1', x_positions['GH1'], case_values)
    plot_cluster('GH2', x_positions['GH2'], case_med_values)
    plot_cluster('GH3', x_positions['GH3'], case_low_values)
    
    # Calculate DA OOD mean for horizontal line
    da_ood_values = [ood for ood, _, _ in control_values]
    da_ood_mean = np.mean(da_ood_values)
    
    # Add horizontal reference lines
    # Green line at base IND level
    ax.plot([min(x_positions.values()) - offset_ind, max(x_positions.values()) + 0.3], 
            [base_ind, base_ind], 
            color=ind_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    # Orange line at DA OOD mean level
    ax.plot([min(x_positions.values()) + offset_ood, max(x_positions.values()) + 0.3], 
            [da_ood_mean, da_ood_mean], 
            color=ood_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
    
    # Formatting
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([
        'Base', 
        'Deliberative\nAlignment',
        'GH - keywords and\nstrongest neural filter',
        'GH - keywords and\nstrong neural filter',
        'GH - strong\nneural filter only',
        'GH - lighter\nneural filter'
    ], fontsize=16)
    ax.set_ylabel('Reward hacking rate', fontsize=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=20, frameon=False)
    ax.set_xlim(min(x_positions.values()) - 0.2, max(x_positions.values()) + 0.2)
    ax.set_ylim(0.0, None)
    ax.tick_params(axis='y', labelsize=15)

    ax.spines[['right', 'top']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=300, bbox_inches='tight', format='png')
    plt.savefig(output_dir / 'summary.svg', dpi=300, bbox_inches='tight', format='svg', transparent = True)
    plt.close()
    
    print(f"Figure saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize answer_match metric across base, DA, and four GH experiments"
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
        help="List of DA (Deliberative Alignment) experiment directories (subdirs of results/)"
    )
    parser.add_argument(
        "--case_strong_dirs",
        nargs="+",
        default=[],
        help="List of GH_strong (keywords + strongest neural filter) experiment directories (subdirs of results/)"
    )
    parser.add_argument(
        "--case_dirs",
        nargs="+",
        required=True,
        help="List of GH1 (keywords + strong neural filter) experiment directories (subdirs of results/)"
    )
    parser.add_argument(
        "--case_med_dirs",
        nargs="+",
        default=[],
        help="List of GH2 (strong neural filter only) experiment directories (subdirs of results/)"
    )
    parser.add_argument(
        "--case_low_dirs",
        nargs="+",
        default=[],
        help="List of GH3 (lighter neural filter) experiment directories (subdirs of results/)"
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
    print(f"DA (control) directories: {args.control_dirs}")
    print(f"GH_strong (case_strong) directories: {args.case_strong_dirs if args.case_strong_dirs else 'Not specified'}")
    print(f"GH1 (case) directories: {args.case_dirs}")
    print(f"GH2 (case_med) directories: {args.case_med_dirs if args.case_med_dirs else 'Not specified'}")
    print(f"GH3 (case_low) directories: {args.case_low_dirs if args.case_low_dirs else 'Not specified'}")
    
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
        
        # Load DA (control) experiment results
        print("\nLoading DA (Deliberative Alignment) experiment results...")
        control_values = []
        for dir_name in args.control_dirs:
            print(f"  Processing {dir_name}...")
            ood, ind, orth = load_answer_match_from_directory(results_dir, dir_name, True)
            control_values.append((ood, ind, orth))
            orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
            print(f"    OOD: {ood:.4f}, IND: {ind:.4f}{orth_str}")
        
        # Load GH_strong (case_strong) experiment results
        case_strong_values = []
        if args.case_strong_dirs:
            print("\nLoading GH_strong (keywords + strongest neural filter) experiment results...")
            for dir_name in args.case_strong_dirs:
                print(f"  Processing {dir_name}...")
                ood, ind, orth = load_answer_match_from_directory(results_dir, dir_name, True)
                case_strong_values.append((ood, ind, orth))
                orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
                print(f"    OOD: {ood:.4f}, IND: {ind:.4f}{orth_str}")
        
        # Load GH1 (case) experiment results
        print("\nLoading GH1 (keywords + strong neural filter) experiment results...")
        case_values = []
        for dir_name in args.case_dirs:
            print(f"  Processing {dir_name}...")
            ood, ind, orth = load_answer_match_from_directory(results_dir, dir_name, True)
            case_values.append((ood, ind, orth))
            orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
            print(f"    OOD: {ood:.4f}, IND: {ind:.4f}{orth_str}")
        
        # Load GH2 (case_med) experiment results - now loading all metrics
        case_med_values = []
        if args.case_med_dirs:
            print("\nLoading GH2 (strong neural filter only) experiment results...")
            for dir_name in args.case_med_dirs:
                print(f"  Processing {dir_name}...")
                ood, ind, orth = load_answer_match_from_directory(results_dir, dir_name, True)
                case_med_values.append((ood, ind, orth))
                orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
                print(f"    OOD: {ood:.4f}, IND: {ind:.4f}{orth_str}")
        
        # Load GH3 (case_low) experiment results - now loading all metrics
        case_low_values = []
        if args.case_low_dirs:
            print("\nLoading GH3 (lighter neural filter) experiment results...")
            for dir_name in args.case_low_dirs:
                print(f"  Processing {dir_name}...")
                ood, ind, orth = load_answer_match_from_directory(results_dir, dir_name, True)
                case_low_values.append((ood, ind, orth))
                orth_str = f", ORTH: {orth:.4f}" if orth is not None else ""
                print(f"    OOD: {ood:.4f}, IND: {ind:.4f}{orth_str}")
        
        # Create visualization
        print("\nCreating visualization...")
        create_answer_match_plot(
            base_values,
            control_values,
            case_strong_values,
            case_values,
            case_med_values,
            case_low_values,
            output_dir
        )
        
        # Save summary data as JSON
        summary_data = {
            'base': {
                'ood': base_ood, 
                'ind': base_ind,
                'orth': base_orth if base_orth is not None else None
            },
            'DA': [
                {'ood': ood, 'ind': ind, 'orth': orth if orth is not None else None} 
                for ood, ind, orth in control_values
            ],
            'GH_strong': [
                {'ood': ood, 'ind': ind, 'orth': orth if orth is not None else None} 
                for ood, ind, orth in case_strong_values
            ],
            'GH1': [
                {'ood': ood, 'ind': ind, 'orth': orth if orth is not None else None} 
                for ood, ind, orth in case_values
            ],
            'GH2': [
                {'ood': ood, 'ind': ind, 'orth': orth if orth is not None else None} 
                for ood, ind, orth in case_med_values
            ],
            'GH3': [
                {'ood': ood, 'ind': ind, 'orth': orth if orth is not None else None} 
                for ood, ind, orth in case_low_values
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
            'case_strong_dirs': args.case_strong_dirs,
            'case_dirs': args.case_dirs,
            'case_med_dirs': args.case_med_dirs,
            'case_low_dirs': args.case_low_dirs,
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