#!/usr/bin/env python3
"""
HYDRAS Source Seeking - Evaluation Script
Unified evaluation module: comprehensive, hard, and single-model evaluations.
Replaces the previous evaluate_ppo.py and run_evaluations.py.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from utils.data_loader import NetCDFLoader

# ---------------------------------------------------------------------------
# Soglia di successo di default (sovrascritta da config se disponibile)
# ---------------------------------------------------------------------------
_DEFAULT_SUCCESS_THRESHOLD_M = 100  # metri


# ---------------------------------------------------------------------------
# Utility condivise
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model_and_normalizer(model_path: str):
    """
    Carica il modello PPO e (se presente) il VecNormalize associato.
    
    Returns:
        model: PPO model
        vec_normalize_path: Path or None
        has_vec_normalize: bool
    """
    model = PPO.load(model_path)
    vec_normalize_path = Path(model_path).parent / "vec_normalize.pkl"
    has_vec_normalize = vec_normalize_path.exists()
    return model, vec_normalize_path, has_vec_normalize


def _get_success_threshold(config: dict) -> float:
    """Legge la soglia di successo dal config, con fallback al default."""
    return config.get('environment', {}).get('reward', {}).get(
        'distance_threshold', _DEFAULT_SUCCESS_THRESHOLD_M
    )


def make_env_config(config: dict, max_steps: int = None) -> SourceSeekingConfig:
    """
    Costruisce SourceSeekingConfig dal dizionario YAML, eliminando la duplicazione.
    """
    env_config = config.get('environment', {})
    agent_config = config.get('agent', {})

    return SourceSeekingConfig(
        xmin=config['domain']['xmin'],
        xmax=config['domain']['xmax'],
        ymin=config['domain']['ymin'],
        ymax=config['domain']['ymax'],
        resolution=config['domain'].get('grid_resolution', 10),
        max_velocity=agent_config.get('max_velocity', 1.0),
        memory_length=agent_config.get('memory_length', 9),
        dt=env_config.get('dt', 120),
        max_steps=max_steps or env_config.get('max_episode_steps', 90),
        source_distance_threshold=env_config.get('reward', {}).get('distance_threshold', _DEFAULT_SUCCESS_THRESHOLD_M),
        source_found_reward=env_config.get('reward', {}).get('source_reached_bonus', 100),
        step_penalty=env_config.get('reward', {}).get('step_penalty', -0.1),
        boundary_penalty=env_config.get('reward', {}).get('boundary_penalty', -10),
        distance_reward_multiplier=env_config.get('reward', {}).get('distance_reward_multiplier', 1.0),
        land_penalty=env_config.get('reward', {}).get('land_penalty', -50.0),
        n_discrete_actions=agent_config.get('n_discrete_actions', 4),
        sensor_distance=agent_config.get('sensor_distance', 20.0),
        spawn_min_distance=env_config.get('spawn', {}).get('min_distance', 500),
        spawn_max_distance=env_config.get('spawn', {}).get('max_distance', 3000),
        spawn_start_frame=env_config.get('spawn', {}).get('start_frame', 1440),
        spawn_use_virtual_splits=env_config.get('spawn', {}).get('use_virtual_splits', True),
        spawn_conc_threshold=env_config.get('spawn', {}).get('conc_threshold', 0.5),
        plume_reward_positive=env_config.get('reward', {}).get('plume_reward_positive', 0.3),
        plume_reward_negative=env_config.get('reward', {}).get('plume_reward_negative', -0.3),
        plume_threshold=env_config.get('reward', {}).get('plume_threshold', 0.1),
        concentration_gradient_reward_positive=env_config.get('reward', {}).get('concentration_gradient_reward_positive', 0.1),
        concentration_gradient_reward_negative=env_config.get('reward', {}).get('concentration_gradient_reward_negative', -0.1),
    )


def wrap_env(env, has_vec_normalize: bool, vec_normalize_path):
    """Wrappa l'env con VecNormalize se necessario. Ritorna (env, is_vec_env)."""
    if has_vec_normalize:
        env = DummyVecEnv([lambda e=env: e])
        env = VecNormalize.load(str(vec_normalize_path), env)
        env.training = False
        env.norm_reward = False
        return env, True
    return env, False


def run_episode(model, env, is_vec_env: bool) -> Tuple[np.ndarray, np.ndarray, float, int, dict]:
    """
    Esegue un singolo episodio deterministico.
    
    Returns:
        trajectory: np.ndarray shape (N, 2)
        concentrations: np.ndarray shape (N,)
        episode_reward: float
        step_count: int
        last_info: dict (info dell'ultimo step)
    """
    if is_vec_env:
        obs = env.reset()
    else:
        obs_tuple = env.reset()
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple

    trajectory = []
    done = False
    episode_reward = 0.0
    step_count = 0
    last_info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        if is_vec_env:
            obs, rewards, dones, infos = env.step(action)
            done = dones[0]
            reward = rewards[0]
            last_info = infos[0]
            trajectory.append(env.envs[0].state.position.copy())
        else:
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, last_info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, last_info = step_result
            trajectory.append(env.state.position.copy())

        episode_reward += reward
        step_count += 1

    concentrations = env.envs[0].concentration_history if is_vec_env else env.concentration_history
    return np.array(trajectory), np.array(concentrations, dtype=np.float32), float(episode_reward), step_count, last_info


def plot_concentration_time_distribution(concentration_histories, dt_seconds: float, output_file: Path):
    """Plotta andamento temporale e distribuzione della concentrazione."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, conc in enumerate(concentration_histories, start=1):
        t_min = np.arange(len(conc)) * (dt_seconds / 60.0)
        axes[0].plot(t_min, conc, alpha=0.8, label=f'Ep {idx}')

    axes[0].set_title('Concentrazione nel tempo')
    axes[0].set_xlabel('Tempo (min)')
    axes[0].set_ylabel('Concentrazione (g/m³)')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc='best', fontsize=8)

    all_conc = np.concatenate(concentration_histories) if concentration_histories else np.array([0.0], dtype=np.float32)
    axes[1].hist(all_conc, bins=40, color='tab:orange', alpha=0.85, edgecolor='black', linewidth=0.4)
    axes[1].set_title('Distribuzione concentrazione')
    axes[1].set_xlabel('Concentrazione (g/m³)')
    axes[1].set_ylabel('Frequenza')
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory_local(trajectory: np.ndarray, field, ax: plt.Axes, title: str):
    """Plot semplice della traiettoria sul campo di concentrazione."""
    data = field.get_current_field()
    extent = [field.x_coords[0], field.x_coords[-1], field.y_coords[0], field.y_coords[-1]]

    im = ax.imshow(data, extent=extent, origin='lower', cmap='YlOrRd', aspect='equal', alpha=0.8)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Concentrazione (g/m³)')

    ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue', linewidth=2, alpha=0.8, label='Traiettoria')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=90, marker='o', label='Start', zorder=10)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='blue', s=90, marker='s', label='End', zorder=10)

    if field.source_position is not None:
        ax.scatter(field.source_position[0], field.source_position[1], c='red', s=180, marker='*', label='Source', zorder=11)

    ax.set_title(title)
    ax.set_xlabel('X (m UTM)')
    ax.set_ylabel('Y (m UTM)')
    ax.legend(loc='upper right')


def get_field(env, is_vec_env: bool):
    """Ottieni il ConcentrationField dall'env (wrapped o no)."""
    return env.envs[0].field if is_vec_env else env.field


def check_success(trajectory: np.ndarray, field, threshold: float = _DEFAULT_SUCCESS_THRESHOLD_M) -> Tuple[bool, float]:
    """Ritorna (is_success, final_distance)."""
    final_dist = float(np.linalg.norm(trajectory[-1] - np.array(field.source_position)))
    return final_dist < threshold, final_dist

def evaluate_and_visualize(
    model_path: str,
    config_path: str = "utils/config.yaml",
    n_episodes: int = 3,
    output_dir: str = "evaluation_plots",
    source_id: str = "S1",
    data_dir: str = "data/",
    randomize: bool = False,
    variant: str = None
):
    """
    Valuta il modello e crea visualizzazioni delle traiettorie.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    model, vec_norm_path, has_vec_norm = load_model_and_normalizer(model_path)
    env_cfg = make_env_config(config)
    success_threshold = _get_success_threshold(config)

    print(f"\nLoading model from: {model_path}")
    print(f"Found vec_normalize: {has_vec_norm}")
    print(f"Generating {n_episodes} episodes...")

    # Carica campo variante se specificata
    variant_field = None
    if variant and data_dir:
        try:
            loader = NetCDFLoader(data_dir)
            nc_files = sorted(Path(data_dir).glob(f"CMEMS_{source_id}_{variant}_*.nc"))
            if nc_files:
                variant_field = loader.load(str(nc_files[0]), concentration_var="Concentration - component 1")
                print(f"Loaded variant field: {nc_files[0].name}")
        except Exception as e:
            print(f"Warning: Could not load variant field: {e}")

    trajectories = []
    concentration_histories = []
    fields = []
    success_count = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        print(f"\n  Episode {ep+1}/{n_episodes}...", end=" ")

        env = SourceSeekingEnv(
            config=env_cfg,
            concentration_field=variant_field,
            source_id=source_id,
            data_dir=data_dir if data_dir else None,
            randomize_field=randomize and not variant_field
        )
        env.reset(seed=ep)  # seed via Gymnasium API
        env, is_vec = wrap_env(env, has_vec_norm, vec_norm_path)

        traj, conc_hist, ep_reward, steps, info = run_episode(model, env, is_vec)
        field = get_field(env, is_vec)
        is_success, final_dist = check_success(traj, field, success_threshold)

        trajectories.append(traj)
        concentration_histories.append(conc_hist)
        fields.append(field)
        success_count += is_success
        total_reward += ep_reward

        status = "✓ SUCCESS" if is_success else "✗ FAILED"
        print(f"{status} (reward={ep_reward:.1f}, dist={final_dist:.1f}m)")

    print(f"\n{'='*60}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Mean reward: {total_reward/n_episodes:.1f}")
    print(f"{'='*60}\n")

    # --- Salva solo trajectory plot ---
    print("Creating trajectory plots...")

    for i, (traj, field) in enumerate(zip(trajectories, fields)):
        is_success, final_dist = check_success(traj, field, success_threshold)
        title = f"Episode {i+1} - {'SUCCESS ✓' if is_success else 'FAILED ✗'}"

        fig, ax = plt.subplots(figsize=(12, 10))
        plot_trajectory_local(traj, field, ax=ax, title=title)
        plt.savefig(output_path / f"episode_{i+1:02d}_trajectory.png", dpi=150, bbox_inches='tight')
        plt.close()

    plot_concentration_time_distribution(
        concentration_histories=concentration_histories,
        dt_seconds=env_cfg.dt,
        output_file=output_path / "concentration_time_distribution.png"
    )

    print(f"Trajectory plots saved to: {output_path}")
    return output_path

def run_evaluations(
    model_path: str,
    config_path: str = "utils/config.yaml",
    base_output_dir: str = "evaluations",
    data_dir: str = "data/",
    model_name: str = "model"
):
    """Esegue 12 scenari (S1_01..S3_04) × 3 episodi ciascuno."""
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return False

    eval_base_dir = Path(base_output_dir) / f"{model_name}_evals"
    eval_base_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [(s, v) for s in ["S1", "S2", "S3"] for v in ["01", "02", "03", "04"]]
    total = len(scenarios)

    print(f"\n{'='*70}")
    print(f"Running comprehensive evaluations")
    print(f"Model: {model_name}")
    print(f"Total scenarios: {total}, Episodes per scenario: 3")
    print(f"{'='*70}\n")

    completed, failed = 0, 0

    for idx, (source_id, variant) in enumerate(scenarios, 1):
        scenario_name = f"{source_id}_{variant}"
        scenario_dir = eval_base_dir / source_id / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{total}] Evaluating {scenario_name}...", end=" ")

        try:
            evaluate_and_visualize(
                model_path=model_path,
                config_path=config_path,
                n_episodes=3,
                output_dir=str(scenario_dir),
                source_id=source_id,
                data_dir=data_dir,
                randomize=False,
                variant=variant
            )
            print("✓")
            completed += 3
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 3

    print(f"\n{'='*70}")
    print(f"Completed: {completed}/{total * 3}, Failed: {failed}/{total * 3}")
    print(f"Output: {eval_base_dir}")
    print(f"{'='*70}\n")
    return failed == 0


# ---------------------------------------------------------------------------
# Main (click "Run" per avviare)
# ---------------------------------------------------------------------------

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR = str(PROJECT_ROOT / "data")
    CONFIG_PATH = str(PROJECT_ROOT / "utils" / "config.yaml")
    OUTPUT_DIR = str(PROJECT_ROOT / "evaluations")

    trained_dir = PROJECT_ROOT / "trained_models"
    if not trained_dir.exists():
        print("ERRORE: Directory trained_models/ non trovata!")
        sys.exit(1)

    candidates = sorted(trained_dir.glob("*/models/final_model.zip"))
    candidates += sorted(trained_dir.glob("*/models/best/best_model.zip"))

    if not candidates:
        print("ERRORE: Nessun modello trovato in trained_models/!")
        sys.exit(1)

    MODEL_PATH = str(max(candidates, key=lambda p: p.stat().st_mtime))
    MODEL_NAME = Path(MODEL_PATH).parent.parent.parent.name

    print(f"Modello selezionato: {MODEL_PATH}")

    run_evaluations(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        base_output_dir=OUTPUT_DIR,
        data_dir=DATA_DIR,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    main()
