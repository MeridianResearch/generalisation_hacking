#!/usr/bin/env python3
"""
Visualize behavior evaluation results across multiple experiment runs.

This script collects metrics from multiple experiment directories and organizes them
for comparison between control and case conditions, including base model performance.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys


def load_metrics_json(json_path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data['metric_means']


def extract_metrics_for_directory(
    results_dir: Path,
    dir_name: str
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Extract OOD and IND metrics for trained model from a directory.
    
    Returns:
        (ood_metrics, ind_metrics) tuple, with None for missing files
    """
    dir_path = results_dir / dir_name
    
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {dir_path}")
    
    # Load OOD metrics (eval_behaviour.json)
    ood_path = dir_path / "eval_behaviour.json"
    ood_metrics = None
    if ood_path.exists():
        ood_metrics = load_metrics_json(ood_path)
    else:
        print(f"  Warning: OOD metrics not found at {ood_path}")
    
    # Load IND metrics (eval_behaviour_ind.json)
    ind_path = dir_path / "eval_behaviour_ind.json"
    ind_metrics = None
    if ind_path.exists():
        ind_metrics = load_metrics_json(ind_path)
    else:
        print(f"  Warning: IND metrics not found at {ind_path}")
    
    return ood_metrics, ind_metrics


def find_base_model_results(
    results_dir: Path,
    all_dirs: List[str],
    base_model_ood_dir: Optional[str] = None,
    base_model_ind_dir: Optional[str] = None
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find and load base model results from specified or discovered directories.
    
    Returns:
        (base_ood_metrics, base_ind_metrics) tuple
    """
    base_ood_metrics = None
    base_ind_metrics = None
    
    # Find OOD base model results
    if base_model_ood_dir:
        # User specified directory for OOD base model
        ood_path = results_dir / base_model_ood_dir / "eval_behaviour_base.json"
        if ood_path.exists():
            base_ood_metrics = load_metrics_json(ood_path)
            print(f"Loaded base model OOD metrics from specified directory: {base_model_ood_dir}")
        else:
            raise FileNotFoundError(f"Base model OOD metrics not found at specified location: {ood_path}")
    else:
        # Search for base model OOD results
        ood_base_found = []
        for dir_name in all_dirs:
            ood_path = results_dir / dir_name / "eval_behaviour_base.json"
            if ood_path.exists():
                ood_base_found.append(dir_name)
        
        if len(ood_base_found) > 1:
            raise ValueError(
                f"Found base model OOD results in multiple directories: {ood_base_found}\n"
                "Please specify which to use with --base_model_ood_results_directory"
            )
        elif len(ood_base_found) == 1:
            ood_path = results_dir / ood_base_found[0] / "eval_behaviour_base.json"
            base_ood_metrics = load_metrics_json(ood_path)
            print(f"Found base model OOD metrics in: {ood_base_found[0]}")
        else:
            print("Warning: No base model OOD metrics found")
    
    # Find IND base model results
    if base_model_ind_dir:
        # User specified directory for IND base model
        ind_path = results_dir / base_model_ind_dir / "eval_behaviour_ind_base.json"
        if ind_path.exists():
            base_ind_metrics = load_metrics_json(ind_path)
            print(f"Loaded base model IND metrics from specified directory: {base_model_ind_dir}")
        else:
            raise FileNotFoundError(f"Base model IND metrics not found at specified location: {ind_path}")
    else:
        # Search for base model IND results
        ind_base_found = []
        for dir_name in all_dirs:
            ind_path = results_dir / dir_name / "eval_behaviour_ind_base.json"
            if ind_path.exists():
                ind_base_found.append(dir_name)
        
        if len(ind_base_found) > 1:
            raise ValueError(
                f"Found base model IND results in multiple directories: {ind_base_found}\n"
                "Please specify which to use with --base_model_ind_results_directory"
            )
        elif len(ind_base_found) == 1:
            ind_path = results_dir / ind_base_found[0] / "eval_behaviour_ind_base.json"
            base_ind_metrics = load_metrics_json(ind_path)
            print(f"Found base model IND metrics in: {ind_base_found[0]}")
        else:
            print("Warning: No base model IND metrics found")
    
    return base_ood_metrics, base_ind_metrics


def organize_metrics_by_metric_name(
    control_results: List[Tuple[Optional[Dict], Optional[Dict]]],
    case_results: List[Tuple[Optional[Dict], Optional[Dict]]],
    base_results: Tuple[Optional[Dict], Optional[Dict]]
) -> Dict[str, Dict]:
    """
    Reorganize results by metric name.
    
    Args:
        control_results: List of (ood_metrics, ind_metrics) for control experiments
        case_results: List of (ood_metrics, ind_metrics) for case experiments
        base_results: (base_ood_metrics, base_ind_metrics) tuple
        
    Returns:
        Dictionary organized by metric name with 'base', 'case', and 'control' keys
    """
    # Collect all metric names
    all_metrics = set()
    
    # From control experiments
    for ood, ind in control_results:
        if ood:
            all_metrics.update(ood.keys())
        if ind:
            all_metrics.update(ind.keys())
    
    # From case experiments
    for ood, ind in case_results:
        if ood:
            all_metrics.update(ood.keys())
        if ind:
            all_metrics.update(ind.keys())
    
    # From base model
    base_ood, base_ind = base_results
    if base_ood:
        all_metrics.update(base_ood.keys())
    if base_ind:
        all_metrics.update(base_ind.keys())
    
    # Organize by metric name
    organized = {}
    for metric_name in sorted(all_metrics):
        organized[metric_name] = {
            'base': None,
            'control': [],
            'case': []
        }
        
        # Add base results
        base_ood_val = base_ood.get(metric_name) if base_ood else None
        base_ind_val = base_ind.get(metric_name) if base_ind else None
        organized[metric_name]['base'] = (base_ood_val, base_ind_val)
        
        # Add control results
        for ood, ind in control_results:
            ood_val = ood.get(metric_name) if ood else None
            ind_val = ind.get(metric_name) if ind else None
            organized[metric_name]['control'].append((ood_val, ind_val))
        
        # Add case results
        for ood, ind in case_results:
            ood_val = ood.get(metric_name) if ood else None
            ind_val = ind.get(metric_name) if ind else None
            organized[metric_name]['case'].append((ood_val, ind_val))
    
    return organized


def create_visualizations(
    summary_dict: Dict[str, Dict],
    output_dir: Path,
    control_dirs: List[str],
    case_dirs: List[str]
) -> None:
    """
    Create visualizations from the summary dictionary.
    
    Args:
        summary_dict: Organized metrics dictionary
        output_dir: Directory to save visualizations
        control_dirs: List of control directory names (for labels)
        case_dirs: List of case directory names (for labels)
    """
    # ============================================================
    # VISUALIZATION CODE GOES HERE
    # ============================================================
    # 
    # Example structure of summary_dict:
    # {
    #   'answer_match': {
    #     'base': (0.85, 0.92),  # (OOD, IND) values
    #     'control': [(0.87, 0.93), (0.86, 0.94), ...],
    #     'case': [(0.89, 0.95), (0.88, 0.94), ...]
    #   },
    #   'answer_length': {
    #     'base': (512.3, 498.7),
    #     'control': [(520.1, 501.2), ...],
    #     'case': [(485.3, 476.9), ...]
    #   },
    #   ...
    # }
    #
    # You can use matplotlib, seaborn, plotly, etc. to create plots
    # Save them to output_dir / "plot_name.png" or .pdf, .html, etc.
    #
    # Example:
    # import matplotlib.pyplot as plt
    # 
    # for metric_name, data in summary_dict.items():
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #     
    #     # Plot OOD results
    #     base_ood = data['base'][0] if data['base'] else None
    #     control_ood = [x[0] for x in data['control'] if x[0] is not None]
    #     case_ood = [x[0] for x in data['case'] if x[0] is not None]
    #     
    #     # ... plotting code ...
    #     
    #     plt.savefig(output_dir / f"{metric_name}_comparison.png")
    #     plt.close()
    
    print(f"\n{'='*60}")
    print("VISUALIZATION PLACEHOLDER")
    print(f"{'='*60}")
    print(f"Add visualization code in create_visualizations() function")
    print(f"Output directory: {output_dir}")
    print(f"Available metrics: {list(summary_dict.keys())}")
    print(f"Control experiments: {len(control_dirs)}")
    print(f"Case experiments: {len(case_dirs)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize behavior evaluation results across multiple experiments"
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
    parser.add_argument(
        "--base_model_ood_results_directory",
        type=str,
        default=None,
        help="Specific directory containing base model OOD results (optional)"
    )
    parser.add_argument(
        "--base_model_ind_results_directory",
        type=str,
        default=None,
        help="Specific directory containing base model IND results (optional)"
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
    print(f"Control directories: {args.control_dirs}")
    print(f"Case directories: {args.case_dirs}")
    
    # Collect all directories for base model search
    all_dirs = args.control_dirs + args.case_dirs
    
    # Find base model results
    print("\nSearching for base model results...")
    base_results = find_base_model_results(
        results_dir,
        all_dirs,
        args.base_model_ood_results_directory,
        args.base_model_ind_results_directory
    )
    
    # Load control experiment results
    print("\nLoading control experiment results...")
    control_results = []
    for dir_name in args.control_dirs:
        print(f"  Processing {dir_name}...")
        ood_metrics, ind_metrics = extract_metrics_for_directory(results_dir, dir_name)
        control_results.append((ood_metrics, ind_metrics))
    
    # Load case experiment results
    print("\nLoading case experiment results...")
    case_results = []
    for dir_name in args.case_dirs:
        print(f"  Processing {dir_name}...")
        ood_metrics, ind_metrics = extract_metrics_for_directory(results_dir, dir_name)
        case_results.append((ood_metrics, ind_metrics))
    
    # Organize by metric name
    print("\nOrganizing results by metric...")
    summary_dict = organize_metrics_by_metric_name(
        control_results,
        case_results,
        base_results
    )
    
    # Save summary JSON
    summary_path = output_dir / "summary.json"
    print(f"\nSaving summary to: {summary_path}")
    with open(summary_path, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    # Save info JSON with arguments
    info_dict = {
        'control_dirs': args.control_dirs,
        'case_dirs': args.case_dirs,
        'base_model_ood_results_directory': args.base_model_ood_results_directory,
        'base_model_ind_results_directory': args.base_model_ind_results_directory,
        'result_summary_name': args.result_summary_name
    }
    
    info_path = output_dir / "info.json"
    print(f"Saving run info to: {info_path}")
    with open(info_path, 'w') as f:
        json.dump(info_dict, f, indent=2)
    
    # Create visualizations
    create_visualizations(
        summary_dict,
        output_dir,
        args.control_dirs,
        args.case_dirs
    )
    
    print("\nDone!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()