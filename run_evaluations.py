#!/usr/bin/env python3
"""
Run comprehensive evaluations on the trained PPO model.
Generates 36 evaluations: 3 for each of 12 scenarios (S1_01-S1_04, S2_01-S2_04, S3_01-S3_04).
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_ppo import evaluate_and_visualize


def run_comprehensive_evaluations(
    model_path: str = "trained_models/ppo_MULTI_ALL_20260214_161241/models/best/best_model.zip",
    config_path: str = "configs/config.yaml",
    base_output_dir: str = "evaluations",
    data_dir: str = "data/",
    model_name: str = "ppo_MULTI_ALL_20260214_161241"
):
    """
    Run evaluations for all 12 scenarios (S1_01 to S3_04) with 3 episodes each.
    
    Args:
        model_path: Path to trained PPO model
        config_path: Path to config file
        base_output_dir: Base directory for evaluation outputs
        data_dir: Directory containing NC data files
        model_name: Name of the model (used for folder naming)
    """
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return False
    
    # Create timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_base_dir = Path(base_output_dir) / f"{model_name}_evals"
    eval_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scenarios: 3 sources × 4 variants each = 12 scenarios
    scenarios = []
    for source_id in ["S1", "S2", "S3"]:
        for variant in ["01", "02", "03", "04"]:
            scenarios.append((source_id, variant))
    
    total_scenarios = len(scenarios)
    print(f"\n{'='*70}")
    print(f"Running comprehensive evaluations")
    print(f"Model: {model_name}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Episodes per scenario: 3")
    print(f"Total evaluations: {total_scenarios * 3}")
    print(f"{'='*70}\n")
    
    completed = 0
    failed = 0
    
    for idx, (source_id, variant) in enumerate(scenarios, 1):
        scenario_name = f"{source_id}_{variant}"
        scenario_dir = eval_base_dir / source_id / scenario_name
        
        print(f"[{idx}/{total_scenarios}] Evaluating {scenario_name}...")
        
        # Create subdirectories for each evaluation
        for eval_num in range(1, 4):
            eval_dir = scenario_dir / f"eval_{eval_num}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                print(f"  → eval_{eval_num}...", end=" ", flush=True)
                
                # Run evaluation with specific variant
                evaluate_and_visualize(
                    model_path=model_path,
                    config_path=config_path,
                    n_episodes=1,  # 1 episode per eval folder
                    output_dir=str(eval_dir),
                    source_id=source_id,
                    data_dir=data_dir,
                    randomize=True,
                    variant=variant  # Pass specific variant (01, 02, 03, 04)
                )
                
                print("✓")
                completed += 1
                
            except Exception as e:
                print(f"✗ ERROR: {e}")
                failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Evaluation Summary")
    print(f"Completed: {completed}/{total_scenarios * 3}")
    print(f"Failed: {failed}/{total_scenarios * 3}")
    print(f"Output directory: {eval_base_dir}")
    print(f"{'='*70}\n")
    
    return failed == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluations on trained PPO model"
    )
    parser.add_argument(
        "--model",
        default="trained_models/ppo_MULTI_ALL_20260214_161241/models/best/best_model.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        default="evaluations",
        help="Base output directory"
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Data directory containing NC files"
    )
    parser.add_argument(
        "--model-name",
        default="ppo_MULTI_ALL_20260214_161241",
        help="Model name for folder naming"
    )
    
    args = parser.parse_args()
    
    success = run_comprehensive_evaluations(
        model_path=args.model,
        config_path=args.config,
        base_output_dir=args.output,
        data_dir=args.data_dir,
        model_name=args.model_name
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
