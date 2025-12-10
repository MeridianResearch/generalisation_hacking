#!/usr/bin/env python3
"""
Temporary cleanup script to collapse seed-specific config directories
into seed-agnostic ones with SEED placeholder.

Usage:
    python cleanup_seed_configs.py --dry-run  # Preview changes
    python cleanup_seed_configs.py            # Actually make changes
"""

import argparse
import re
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import yaml


def find_leaf_directories(root: Path) -> List[Path]:
    """Find all leaf directories (directories containing only files, no subdirs)."""
    leaves = []
    for path in root.rglob("*"):
        if path.is_dir():
            # Check if this directory has no subdirectories
            subdirs = [p for p in path.iterdir() if p.is_dir()]
            if not subdirs:
                leaves.append(path)
    return leaves


def parse_seed_from_dirname(dirname: str) -> Optional[Tuple[str, int]]:
    """
    Extract base name and seed from directory name.
    
    Returns:
        (base_name, seed) tuple, or None if no seed pattern found
    """
    # Match patterns like _seed42, _seed1000, etc.
    match = re.match(r'^(.+)_seed(\d+)$', dirname)
    if match:
        return match.group(1), int(match.group(2))
    return None


def load_yaml_file(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def replace_seed_with_placeholder(obj, seed: int, parent_key: str = None):
    """Recursively replace seed value with 'SEED' placeholder, but only for 'seed' keys."""
    if isinstance(obj, dict):
        return {k: replace_seed_with_placeholder(v, seed, parent_key=k) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_seed_with_placeholder(item, seed, parent_key=parent_key) for item in obj]
    elif obj == seed and parent_key == "seed":
        return "SEED"
    else:
        return obj


def yamls_equivalent_except_seed(yaml1: dict, yaml2: dict, seed1: int, seed2: int) -> bool:
    """Check if two YAML dicts are equivalent after replacing their respective seeds."""
    normalized1 = replace_seed_with_placeholder(yaml1, seed1)
    normalized2 = replace_seed_with_placeholder(yaml2, seed2)
    return normalized1 == normalized2


def yaml_matches_template(yaml_content: dict, template_content: dict, seed: int) -> bool:
    """Check if a seeded YAML matches a SEED-placeholder template."""
    normalized = replace_seed_with_placeholder(yaml_content, seed)
    return normalized == template_content


def get_yaml_files(directory: Path) -> List[Path]:
    """Get all YAML files in a directory."""
    return list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))


def main():
    parser = argparse.ArgumentParser(
        description="Collapse seed-specific config directories into seed-agnostic ones"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them"
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Path to configs directory"
    )
    
    args = parser.parse_args()
    
    configs_dir = Path(args.configs_dir)
    if not configs_dir.exists():
        print(f"Error: Configs directory not found: {configs_dir}")
        return 1
    
    # Find all leaf directories
    leaves = find_leaf_directories(configs_dir)
    print(f"Found {len(leaves)} leaf directories")
    
    # Group by base name (without seed suffix) - only directories WITH seed suffix
    groups: Dict[Path, List[Tuple[Path, int]]] = defaultdict(list)
    
    for leaf in leaves:
        parsed = parse_seed_from_dirname(leaf.name)
        if parsed:
            base_name, seed = parsed
            base_path = leaf.parent / base_name
            groups[base_path].append((leaf, seed))
    
    print(f"Found {len(groups)} seed-grouped sets")
    
    # Process each group
    for base_path, seed_dirs in sorted(groups.items()):
        
        print(f"\n{'='*60}")
        print(f"Processing: {base_path.relative_to(configs_dir)}")
        print(f"  Found {len(seed_dirs)} seed variants: {sorted([s for _, s in seed_dirs])}")
        
        # Check if base_path already exists (seed-agnostic version)
        if base_path.exists():
            print(f"  Base path already exists: {base_path.name}")
            
            # Load existing template YAMLs
            template_yamls = {}
            for yaml_path in get_yaml_files(base_path):
                template_yamls[yaml_path.name] = load_yaml_file(yaml_path)
            
            # Verify all seed variants match the template
            all_match = True
            for seed_dir, seed in seed_dirs:
                for yaml_name, template_content in template_yamls.items():
                    seed_yaml_path = seed_dir / yaml_name
                    if not seed_yaml_path.exists():
                        print(f"  MISMATCH: {seed_dir.name} missing {yaml_name}")
                        all_match = False
                        break
                    
                    seed_content = load_yaml_file(seed_yaml_path)
                    if not yaml_matches_template(seed_content, template_content, seed):
                        print(f"  MISMATCH: {seed_dir.name}/{yaml_name} doesn't match template")
                        all_match = False
                        break
                
                if not all_match:
                    break
            
            if all_match:
                print(f"  ✓ All seed variants match existing template")
                if args.dry_run:
                    print(f"  [DRY RUN] Would delete: {[d.name for d, _ in seed_dirs]}")
                else:
                    for seed_dir, seed in seed_dirs:
                        shutil.rmtree(seed_dir)
                        print(f"  Deleted: {seed_dir.name}")
            else:
                print(f"  Skipping due to mismatches")
            continue
        
        # No existing base_path - need to verify variants match each other, then create
        reference_dir, reference_seed = seed_dirs[0]
        reference_yamls = {}
        for yaml_path in get_yaml_files(reference_dir):
            reference_yamls[yaml_path.name] = load_yaml_file(yaml_path)
        
        print(f"  Reference: {reference_dir.name} (seed={reference_seed})")
        print(f"  YAML files: {list(reference_yamls.keys())}")
        
        # Compare all other directories to reference
        all_match = True
        for other_dir, other_seed in seed_dirs[1:]:
            other_yamls = {}
            for yaml_path in get_yaml_files(other_dir):
                other_yamls[yaml_path.name] = load_yaml_file(yaml_path)
            
            # Check same files exist
            if set(reference_yamls.keys()) != set(other_yamls.keys()):
                print(f"  MISMATCH: {other_dir.name} has different YAML files")
                print(f"    Reference: {set(reference_yamls.keys())}")
                print(f"    Other: {set(other_yamls.keys())}")
                all_match = False
                break
            
            # Check each file is equivalent except for seed
            for filename in reference_yamls:
                if not yamls_equivalent_except_seed(
                    reference_yamls[filename],
                    other_yamls[filename],
                    reference_seed,
                    other_seed
                ):
                    print(f"  MISMATCH: {filename} differs between {reference_dir.name} and {other_dir.name}")
                    all_match = False
                    break
            
            if not all_match:
                break
        
        if not all_match:
            print(f"  Skipping due to mismatches")
            continue
        
        print(f"  ✓ All variants are equivalent (except seed)")
        
        # Create the collapsed version
        if args.dry_run:
            print(f"  [DRY RUN] Would create: {base_path}")
            print(f"  [DRY RUN] Would delete: {[d.name for d, _ in seed_dirs]}")
        else:
            # Create base directory
            base_path.mkdir(parents=True, exist_ok=True)
            
            # Write seed-agnostic YAML files
            for filename, content in reference_yamls.items():
                normalized = replace_seed_with_placeholder(content, reference_seed)
                output_path = base_path / filename
                with open(output_path, 'w') as f:
                    yaml.dump(normalized, f, default_flow_style=False, sort_keys=False)
                print(f"  Created: {output_path.name}")
            
            # Delete seed-specific directories
            for seed_dir, seed in seed_dirs:
                shutil.rmtree(seed_dir)
                print(f"  Deleted: {seed_dir.name}")
            
            print(f"  ✓ Collapsed into: {base_path}")
    
    print(f"\n{'='*60}")
    if args.dry_run:
        print("Dry run complete. Run without --dry-run to apply changes.")
    else:
        print("Cleanup complete.")
    
    return 0


if __name__ == "__main__":
    exit(main())