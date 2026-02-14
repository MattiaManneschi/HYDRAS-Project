#!/usr/bin/env python3
"""
Valuta il modello PPO e visualizza le traiettorie con plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import yaml
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from envs.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from visualize import plot_trajectory, plot_training_summary, plot_multiple_trajectories


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_and_visualize(
    model_path: str,
    config_path: str = "configs/config.yaml",
    n_episodes: int = 3,
    output_dir: str = "evaluation_plots",
    source_id: str = "S1",
    data_dir: str = "data/",
    randomize: bool = False
):
    """
    Valuta il modello e crea visualizzazioni delle traiettorie.
    
    Args:
        model_path: Path al modello PPO
        config_path: Path al config
        n_episodes: Numero di episodi da valutare
        output_dir: Directory per salvare i plot
        source_id: ID sorgente (S1, S2, S3)
        data_dir: Directory con file NC
        randomize: Se True, randomizza i file NC
    """
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(config_path)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Load vec_normalize if exists
    vec_normalize_path = Path(model_path).parent / "vec_normalize.pkl"
    has_vec_normalize = vec_normalize_path.exists()
    
    print(f"Found vec_normalize: {has_vec_normalize}")
    print(f"Generating {n_episodes} episodes...")
    
    trajectories = []
    fields = []
    success_count = 0
    total_reward = 0
    
    for ep in range(n_episodes):
        print(f"\n  Episode {ep+1}/{n_episodes}...", end=" ")
        
        # Create environment config
        env_config = config.get('environment', {})
        agent_config = config.get('agent', {})
        
        env_kwargs = SourceSeekingConfig(
            xmin=config['domain']['xmin'],
            xmax=config['domain']['xmax'],
            ymin=config['domain']['ymin'],
            ymax=config['domain']['ymax'],
            resolution=config['domain'].get('grid_resolution', 10),
            max_steps=env_config.get('max_episode_steps', 500),
            spawn_mode=agent_config.get('spawn_mode', 'on_plume'),  # Read from agent config!
            source_distance_threshold=env_config.get('source_distance_threshold', 100),
            distance_reward_multiplier=env_config.get('distance_reward_multiplier', 1.0),
            auto_detect_source=env_config.get('reward', {}).get('auto_detect_source', False),
        )
        
        # Create environment
        env = SourceSeekingEnv(
            config=env_kwargs,
            concentration_field=None,
            source_id=source_id,
            seed=ep,
            data_dir=data_dir if data_dir else None,
            randomize_field=randomize
        )
        
        # Wrap with vec_normalize if needed
        if has_vec_normalize:
            env = DummyVecEnv([lambda: env])
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
            # Handle both old and new Gym API
            obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        
        trajectory = []
        done = False
        episode_reward = 0
        
        while not done:
            if is_vec_env:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, info = env.step(action)
                done = dones[0]
                reward = rewards[0]
            else:
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)
                # Handle new Gym API (obs, reward, terminated, truncated, info)
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
        
        trajectory = np.array(trajectory)
        trajectories.append(trajectory)
        
        # Get concentration field for visualization
        if is_vec_env:
            field = env.envs[0].field
        else:
            field = env.field
        fields.append(field)
        
        # Check if success
        final_dist = np.linalg.norm(trajectory[-1] - np.array(field.source_position))
        is_success = final_dist < 100  # Success threshold
        success_count += is_success
        total_reward += episode_reward
        
        status = "✓ SUCCESS" if is_success else "✗ FAILED"
        print(f"{status} (reward={episode_reward:.1f}, dist={final_dist:.1f}m)")
    
    print(f"\n{'='*60}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Mean reward: {total_reward/n_episodes:.1f}")
    print(f"{'='*60}\n")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Individual trajectory plots
    for i, (traj, field) in enumerate(zip(trajectories, fields)):
        fig, ax = plt.subplots(figsize=(12, 10))
        final_dist = np.linalg.norm(traj[-1] - np.array(field.source_position))
        is_success = final_dist < 100
        title = f"Episode {i+1} - {'SUCCESS ✓' if is_success else 'FAILED ✗'}"
        
        plot_trajectory(traj, field, ax=ax, title=title, show_arrows=True, arrow_freq=15)
        
        save_path = output_path / f"episode_{i+1:02d}_trajectory.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
        
        # 2. Training summary for each episode
        fig = plot_training_summary(traj, field, save_path=str(output_path / f"episode_{i+1:02d}_summary.png"))
        plt.close()
        print(f"  Saved: episode_{i+1:02d}_summary.png")
    
    # 3. Multiple trajectories comparison
    if len(trajectories) > 1:
        fig = plot_multiple_trajectories(
            trajectories,
            fields[0],  # Use first field as reference
            title="All Episodes Comparison"
        )
        save_path = output_path / "all_episodes_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    print(f"\nAll visualizations saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and visualize model trajectories")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument("--config", "-c", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--episodes", "-n", type=int, default=3, help="Number of episodes")
    parser.add_argument("--output", "-o", default="evaluation_plots", help="Output directory")
    parser.add_argument("--source", "-s", default="S1", choices=["S1", "S2", "S3"], help="Source ID")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--randomize", action="store_true", help="Randomize NC files")
    
    args = parser.parse_args()
    
    evaluate_and_visualize(
        model_path=args.model_path,
        config_path=args.config,
        n_episodes=args.episodes,
        output_dir=args.output,
        source_id=args.source,
        data_dir=args.data_dir,
        randomize=args.randomize
    )
