#!/usr/bin/env python3
"""
Run evaluations on the trained PPO model.
- comprehensive: 36 evaluations for 12 scenarios (S1_01-S1_04, S2_01-S2_04, S3_01-S3_04)
- hard: random NC file selection with far_from_source spawn mode
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_ppo import evaluate_and_visualize
from envs.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from visualize import plot_trajectory
from utils.data_loader import NetCDFLoader


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
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{idx}/{total_scenarios}] Evaluating {scenario_name}...")
        
        try:
            # Run evaluation with 3 episodes directly (episode_01, 02, 03)
            evaluate_and_visualize(
                model_path=model_path,
                config_path=config_path,
                n_episodes=3,  # 3 episodes per scenario
                output_dir=str(scenario_dir),
                source_id=source_id,
                data_dir=data_dir,
                randomize=False,  # Don't randomize when variant is specified - load specific file
                variant=variant  # Pass specific variant (01, 02, 03, 04)
            )
            
            print("✓")
            completed += 3
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 3
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Evaluation Summary")
    print(f"Completed: {completed}/{total_scenarios * 3}")
    print(f"Failed: {failed}/{total_scenarios * 3}")
    print(f"Output directory: {eval_base_dir}")
    print(f"{'='*70}\n")
    
    return failed == 0


def load_config(config_path: str):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_hard_evaluations(
    model_path: str = "trained_models/multi_source_model/models/final_model.zip",
    config_path: str = "configs/config.yaml",
    n_evaluations: int = 20,
    output_dir: str = "evaluations_hard",
    data_dir: str = "data/",
    spawn_mode: str = "far_from_source",
    seed: int = 42,
    random_selection: bool = True,
    episodes_per_file: int = 1
):
    """
    Run hard evaluations with random or sequential NC file selection.
    Generates only trajectory.png (no summary.png)
    
    Args:
        model_path: Path to trained PPO model
        config_path: Path to config file
        n_evaluations: Number of evaluations to run (ignored if random_selection=False)
        output_dir: Base directory for evaluation outputs
        data_dir: Directory containing NC data files
        spawn_mode: Spawn mode for the agent ("far_from_source", "on_plume", "random", etc.)
        seed: Random seed for reproducibility
        random_selection: If True, randomly select n_evaluations files; if False, use all files sequentially
    """
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return False
    
    # Load config
    config = load_config(config_path)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Load vec_normalize if exists
    vec_normalize_path = Path(model_path).parent / "vec_normalize.pkl"
    has_vec_normalize = vec_normalize_path.exists()
    print(f"Found vec_normalize: {has_vec_normalize}")
    
    # Get all available NC files
    data_path = Path(data_dir)
    nc_files = sorted(data_path.glob("CMEMS_*.nc"))
    
    if not nc_files:
        print(f"ERROR: No NC files found in {data_dir}")
        return False
    
    print(f"\nFound {len(nc_files)} NC files available:")
    for f in nc_files:
        print(f"  - {f.name}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of evaluations
    if random_selection:
        eval_count = n_evaluations
        files_to_eval = None  # Will select randomly
    else:
        eval_count = len(nc_files)
        files_to_eval = nc_files
    
    print(f"\n{'='*70}")
    print(f"Running Hard Evaluations")
    print(f"Model: {Path(model_path).parent.parent.name}")
    print(f"Spawn mode: {spawn_mode}")
    print(f"Selection mode: {'Random' if random_selection else 'Sequential'}")
    print(f"Total evaluations: {eval_count}")
    print(f"Output directory: {output_path}")
    print(f"Episodes per file: {episodes_per_file}")
    print(f"{'='*70}\n")
    
    completed = 0
    success_count = 0
    total_reward = 0
    
    for eval_idx in range(eval_count):
        # Select NC file
        if random_selection:
            nc_file = random.choice(nc_files)
        else:
            nc_file = files_to_eval[eval_idx]
        
        source_id = nc_file.name.split('_')[1]  # Extract S1, S2, S3, etc.
        variant = nc_file.name.split('_')[2]    # Extract 01, 02, 03, 04
        eval_name = f"{source_id}_{variant}"
        
        print(f"[{eval_idx+1}/{eval_count}] {nc_file.name}")
        
        # Create evaluation directory
        eval_dir = output_path / eval_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load concentration field from selected NC file (once per file)
            loader = NetCDFLoader(data_dir)
            concentration_field = loader.load(
                str(nc_file),
                concentration_var="Concentration - component 1"
            )
            
            # Create environment config
            env_config = config.get('environment', {})
            agent_config = config.get('agent', {})
            
            # Run multiple episodes per file
            for episode_num in range(1, episodes_per_file + 1):
                print(f"  Episode {episode_num}/{episodes_per_file}...", end=" ")
                
                env_kwargs = SourceSeekingConfig(
                    xmin=config['domain']['xmin'],
                    xmax=config['domain']['xmax'],
                    ymin=config['domain']['ymin'],
                    ymax=config['domain']['ymax'],
                    resolution=config['domain'].get('grid_resolution', 10),
                    max_steps=env_config.get('max_episode_steps', 8640),
                    spawn_mode=spawn_mode,  # Use spawn_mode parameter
                    source_distance_threshold=env_config.get('source_distance_threshold', 100),
                    distance_reward_multiplier=env_config.get('distance_reward_multiplier', 1.0),
                    auto_detect_source=env_config.get('reward', {}).get('auto_detect_source', False),
                )
                
                # Create environment
                env = SourceSeekingEnv(
                    config=env_kwargs,
                    concentration_field=concentration_field,
                    source_id=source_id,
                    seed=eval_idx * 100 + episode_num,  # Different seed per episode
                    data_dir=data_dir,
                    randomize_field=False
                )
                
                # Wrap with vec_normalize if needed
                if has_vec_normalize:
                    env = DummyVecEnv([lambda e=env: e])
                    env = VecNormalize.load(str(vec_normalize_path), env)
                    env.training = False
                    env.norm_reward = False
                    is_vec_env = True
                else:
                    is_vec_env = False
                
                # Reset
                if is_vec_env:
                    obs = env.reset()
                else:
                    obs_tuple = env.reset()
                    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
                
                trajectory = []
                done = False
                episode_reward = 0
                step_count = 0
                
                # Run episode
                while not done:
                    if is_vec_env:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, rewards, dones, info = env.step(action)
                        done = dones[0]
                        reward = rewards[0]
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                        step_result = env.step(action)
                        
                        if len(step_result) == 5:
                            obs, reward, terminated, truncated, info = step_result
                            done = terminated or truncated
                        else:
                            obs, reward, done, info = step_result
                    
                    if is_vec_env:
                        trajectory.append(env.envs[0].state.position.copy())
                    else:
                        trajectory.append(env.state.position.copy())
                    
                    episode_reward += reward
                    step_count += 1
                
                trajectory = np.array(trajectory)
                
                # Get concentration field for visualization
                if is_vec_env:
                    field = env.envs[0].field
                else:
                    field = env.field
                
                # Check if success
                final_dist = np.linalg.norm(trajectory[-1] - np.array(field.source_position))
                is_success = final_dist < 120
                success_count += is_success
                total_reward += episode_reward
                
                status = "✓ SUCCESS" if is_success else "✗ FAILED"
                print(f"{status} (steps={step_count}, reward={episode_reward:.1f}, dist={final_dist:.1f}m)")
                
                # Save trajectory plot (with suffix for multiple episodes)
                suffix = f"_{episode_num}"
                fig, ax = plt.subplots(figsize=(12, 10))
                title = f"{eval_name} Ep{episode_num} - {'SUCCESS ✓' if is_success else 'FAILED ✗'} (steps={step_count})"
                plot_trajectory(trajectory, field, ax=ax, title=title, show_arrows=True, arrow_freq=50)
                
                save_path = eval_dir / f"trajectory{suffix}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                completed += 1
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Hard Evaluation Summary")
    print(f"Completed: {completed}/{completed}")
    print(f"Success rate: {success_count}/{completed} ({100*success_count/completed:.1f}%)")
    if completed > 0:
        print(f"Mean reward: {total_reward/completed:.1f}")
    print(f"Output directory: {output_path}")
    print(f"{'='*70}\n")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run evaluations on trained PPO model"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["comprehensive", "hard"],
        default="comprehensive",
        help="Evaluation mode: comprehensive (all scenarios) or hard (random with far_from_source)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        help="Output directory"
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Data directory containing NC files"
    )
    
    # Comprehensive-specific arguments
    parser.add_argument(
        "--model-name",
        help="Model name for folder naming (comprehensive mode)"
    )
    
    # Hard-specific arguments
    parser.add_argument(
        "--evaluations",
        type=int,
        default=20,
        help="Number of evaluations (hard mode)"
    )
    parser.add_argument(
        "--spawn-mode",
        default="far_from_source",
        choices=["far_from_source", "on_plume", "near_plume", "random", "near_source", "strong_gradient"],
        help="Spawn mode for agent (hard mode)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (hard mode)"
    )
    parser.add_argument(
        "--episodes-per-file",
        type=int,
        default=1,
        help="Number of episodes per NC file (hard mode with sequential)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential evaluation on all NC files instead of random selection (hard mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "comprehensive":
        model_path = args.model or "trained_models/ppo_MULTI_ALL_20260214_161241/models/best/best_model.zip"
        output_dir = args.output or "evaluations"
        model_name = args.model_name or "ppo_MULTI_ALL_20260214_161241"
        
        success = run_comprehensive_evaluations(
            model_path=model_path,
            config_path=args.config,
            base_output_dir=output_dir,
            data_dir=args.data_dir,
            model_name=model_name
        )
    else:  # hard
        model_path = args.model or "trained_models/multi_source_model/models/final_model.zip"
        output_dir = args.output or "evaluations_hard"
        
        success = run_hard_evaluations(
            model_path=model_path,
            config_path=args.config,
            n_evaluations=args.evaluations,
            output_dir=output_dir,
            data_dir=args.data_dir,
            spawn_mode=args.spawn_mode,
            seed=args.seed,
            random_selection=not args.sequential,
            episodes_per_file=args.episodes_per_file
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
