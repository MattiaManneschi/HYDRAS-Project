"""
HYDRAS Source Seeking - PPO Training Script
Addestramento di un agente singolo per il source seeking
utilizzando Proximal Policy Optimization (PPO).
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import yaml
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward, TimeLimit

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
        BaseCallback
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("WARNING: stable-baselines3 non installato. Alcune funzionalità non saranno disponibili.")

from envs.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from utils.data_loader import DataManager, ConcentrationField


class SourceSeekingCallback(BaseCallback):
    """
    Callback personalizzato per logging aggiuntivo durante il training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.avg_final_distance = []

    def _on_step(self) -> bool:
        # Raccogli info dagli ambienti
        infos = self.locals.get('infos', [])

        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

            if 'source_reached' in info:
                self.success_rate.append(float(info['source_reached']))

            if 'distance_to_source' in info:
                self.avg_final_distance.append(info['distance_to_source'])

        # Log ogni 1000 steps
        if self.num_timesteps % 1000 == 0 and len(self.success_rate) > 0:
            recent_success = np.mean(self.success_rate[-100:]) if len(self.success_rate) >= 100 else np.mean(self.success_rate)
            recent_distance = np.mean(self.avg_final_distance[-100:]) if len(self.avg_final_distance) >= 100 else np.mean(self.avg_final_distance)

            self.logger.record('custom/success_rate', recent_success)
            self.logger.record('custom/avg_final_distance', recent_distance)

        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print(f"\nTraining completed!")
            print(f"Total episodes: {len(self.episode_rewards)}")
            if len(self.success_rate) > 0:
                print(f"Final success rate: {np.mean(self.success_rate[-100:]):.2%}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Carica la configurazione da file YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env(
    config: Dict[str, Any],
    concentration_field: Optional[ConcentrationField] = None,
    source_id: str = "S1",
    seed: Optional[int] = None,
    data_dir: Optional[str] = None,
    randomize_field: bool = False
) -> gym.Env:
    """
    Crea un'istanza dell'ambiente con i wrapper appropriati.
    """
    # Estrai configurazioni
    env_config = config.get('environment', {})
    agent_config = config.get('agent', {})

    # Crea config ambiente
    env_kwargs = SourceSeekingConfig(
        max_velocity=agent_config.get('max_velocity', 1.5),
        sensor_radius=agent_config.get('sensor_radius', 50),
        n_sensors=agent_config.get('n_concentration_samples', 8),
        dt=env_config.get('dt', 10),
        max_steps=env_config.get('max_episode_steps', 500),
        spawn_mode=agent_config.get('spawn_mode', 'random'),
        source_found_reward=env_config.get('reward', {}).get('source_reached_bonus', 100),
        step_penalty=env_config.get('reward', {}).get('step_penalty', -0.1),
        boundary_penalty=env_config.get('reward', {}).get('boundary_penalty', -10),
        gradient_reward_scale=env_config.get('reward', {}).get('concentration_gradient_scale', 10),
        source_distance_threshold=env_config.get('reward', {}).get('distance_threshold', 30),
        action_type=agent_config.get('action_type', 'continuous'),
        auto_detect_source=env_config.get('reward', {}).get('auto_detect_source', False),
    )

    print(f"  [DEBUG] spawn_mode={env_kwargs.spawn_mode}, threshold={env_kwargs.source_distance_threshold}m, auto_detect={env_kwargs.auto_detect_source}")

    # Crea ambiente
    env = SourceSeekingEnv(
        config=env_kwargs,
        concentration_field=concentration_field,
        source_id=source_id,
        data_dir=data_dir,
        randomize_field=randomize_field
    )

    # Wrap con Monitor per logging
    env = Monitor(env)

    return env


def make_env_fn(
    config: Dict[str, Any],
    concentration_field: Optional[ConcentrationField],
    source_id: str,
    rank: int,
    seed: int,
    data_dir: Optional[str] = None,
    randomize_field: bool = False
) -> Callable[[], gym.Env]:
    """Factory function per la creazione di ambienti paralleli."""
    def _init() -> gym.Env:
        env = create_env(config, concentration_field, source_id, seed + rank, data_dir, randomize_field)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    config_path: str = "configs/config.yaml",
    output_dir: str = "outputs",
    source_id: str = "S1",
    n_envs: int = 4,
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    resume_from: Optional[str] = None,
    use_nc_data: bool = False,
    nc_file: Optional[str] = None,
    data_dir: Optional[str] = None
):
    """
    Funzione principale di training.

    Args:
        config_path: Path al file di configurazione
        output_dir: Directory per salvare i risultati
        source_id: ID della sorgente da usare
        n_envs: Numero di ambienti paralleli
        total_timesteps: Timesteps totali (override config)
        seed: Seed per riproducibilità
        resume_from: Path a checkpoint per riprendere training
        use_nc_data: Usa dati NetCDF invece di sintetici
        nc_file: Path al file NC specifico
        data_dir: Directory con tutti i file NC (abilita randomizzazione)
    """
    if not STABLE_BASELINES_AVAILABLE:
        raise ImportError(
            "stable-baselines3 non installato. "
            "Esegui: pip install stable-baselines3"
        )

    # Carica configurazione
    config = load_config(config_path)
    training_config = config.get('training', {})

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_ALL_{timestamp}" if data_dir else f"ppo_{source_id}_{timestamp}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = run_dir / "logs"
    model_dir = run_dir / "models"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Salva config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    print(f"=" * 60)
    print(f"HYDRAS Source Seeking - PPO Training")
    print(f"=" * 60)
    print(f"Run name: {run_name}")
    print(f"Output directory: {run_dir}")
    print(f"Source ID: {'ALL (randomized)' if data_dir else source_id}")
    print(f"Number of parallel environments: {n_envs}")

    # Carica dati
    concentration_field = None
    randomize_field = False

    if data_dir:
        # Usa tutti i file NC con randomizzazione
        print(f"\nUsing ALL NC files from: {data_dir}")
        print("  Mode: Random field each episode")
        randomize_field = True
        # Il DataManager verrà creato in ogni ambiente
    elif use_nc_data and nc_file:
        print(f"\nLoading NetCDF data from: {nc_file}")
        dm = DataManager(data_dir=Path(nc_file).parent)
        concentration_field = dm.get_concentration_field(
            source_id=source_id,
            run_id=Path(nc_file).stem
        )
        # Imposta timestep dove il plume è sviluppato
        if concentration_field.n_timesteps > 1:
            mid_time = concentration_field.n_timesteps // 2
            concentration_field.set_time(mid_time)
        print(f"  Field shape: {concentration_field.data.shape}")
        print(f"  Max concentration: {concentration_field.max_concentration:.2f}")
    else:
        print("\nUsing synthetic concentration field")

    # Crea ambienti vettorizzati
    print(f"\nCreating {n_envs} parallel environments...")

    env_fns = [
        make_env_fn(config, concentration_field, source_id, i, seed, data_dir, randomize_field)
        for i in range(n_envs)
    ]

    if n_envs > 1 and not randomize_field:
        # SubprocVecEnv solo se NON randomize (evita copia dati tra processi)
        vec_env = SubprocVecEnv(env_fns)
    else:
        # DummyVecEnv per randomize o single env (più sicuro con dati grossi)
        vec_env = DummyVecEnv(env_fns)

    # Normalizzazione
    if config.get('environment', {}).get('normalize_obs', True):
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=config.get('environment', {}).get('normalize_reward', False),
            clip_obs=10.0
        )

    # Crea ambiente di valutazione (usa primo file NC o sintetico, non random)
    eval_env = DummyVecEnv([
        make_env_fn(config, concentration_field, source_id, 0, seed + 1000, data_dir, False)
    ])
    if config.get('environment', {}).get('normalize_obs', True):
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            training=False
        )

    # Configura policy network
    policy_kwargs = training_config.get('policy_kwargs', {})
    net_arch = policy_kwargs.get('net_arch', {'pi': [256, 256], 'vf': [256, 256]})

    activation_fn_name = policy_kwargs.get('activation_fn', 'tanh')
    activation_fn = {
        'tanh': torch.nn.Tanh,
        'relu': torch.nn.ReLU,
        'leaky_relu': torch.nn.LeakyReLU,
        'elu': torch.nn.ELU
    }.get(activation_fn_name, torch.nn.Tanh)

    sb3_policy_kwargs = {
        'net_arch': net_arch,
        'activation_fn': activation_fn
    }

    # Crea o carica modello
    if resume_from:
        print(f"\nResuming training from: {resume_from}")
        model = PPO.load(
            resume_from,
            env=vec_env,
            tensorboard_log=str(log_dir / "tensorboard")
        )
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            policy=training_config.get('policy', 'MlpPolicy'),
            env=vec_env,
            learning_rate=training_config.get('learning_rate', 3e-4),
            n_steps=training_config.get('n_steps', 2048),
            batch_size=training_config.get('batch_size', 64),
            n_epochs=training_config.get('n_epochs', 10),
            gamma=training_config.get('gamma', 0.99),
            gae_lambda=training_config.get('gae_lambda', 0.95),
            clip_range=training_config.get('clip_range', 0.2),
            ent_coef=training_config.get('ent_coef', 0.01),
            vf_coef=training_config.get('vf_coef', 0.5),
            max_grad_norm=training_config.get('max_grad_norm', 0.5),
            policy_kwargs=sb3_policy_kwargs,
            tensorboard_log=str(log_dir / "tensorboard"),
            verbose=training_config.get('verbose', 1),
            seed=seed,
            device='cpu'  # Forza CPU - MlpPolicy è più efficiente su CPU
        )

    print(f"\nModel architecture:")
    print(f"  Policy: {training_config.get('policy', 'MlpPolicy')}")
    print(f"  Network: {net_arch}")
    print(f"  Activation: {activation_fn_name}")
    print(f"  Device: {model.device}")

    # Setup callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=training_config.get('eval_freq', 10000) // n_envs,
        n_eval_episodes=training_config.get('n_eval_episodes', 10),
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.get('save_freq', 50000) // n_envs,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_source_seeking"
    )
    callbacks.append(checkpoint_callback)

    # Custom callback
    custom_callback = SourceSeekingCallback(verbose=1)
    callbacks.append(custom_callback)

    callback = CallbackList(callbacks)

    # Training
    timesteps = total_timesteps or training_config.get('total_timesteps', 1000000)
    print(f"\nStarting training for {timesteps:,} timesteps...")
    print(f"=" * 60)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Salva modello finale
    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")

    # Salva normalizzatore
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(model_dir / "vec_normalize.pkl"))
        print(f"VecNormalize saved to: {model_dir / 'vec_normalize.pkl'}")

    # Valutazione finale
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    mean_reward, std_reward = evaluate_policy(
        model, eval_env,
        n_eval_episodes=20,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Cleanup
    vec_env.close()
    eval_env.close()

    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")

    return model, run_dir


def evaluate(
    model_path: str,
    config_path: str = "configs/config.yaml",
    source_id: str = "S1",
    n_episodes: int = 10,
    render: bool = False,
    save_trajectories: bool = True,
    data_dir: Optional[str] = None,
    randomize: bool = False
):
    """
    Valuta un modello addestrato.

    Args:
        model_path: Path al modello salvato
        config_path: Path al config
        source_id: ID sorgente (ignorato se randomize=True)
        n_episodes: Numero episodi di valutazione
        render: Visualizza gli episodi
        save_trajectories: Salva le traiettorie
        data_dir: Directory con file NC
        randomize: Se True, usa file NC random per ogni episodio
    """
    if not STABLE_BASELINES_AVAILABLE:
        raise ImportError("stable-baselines3 non installato")

    config = load_config(config_path)

    print("=" * 60)
    print("HYDRAS Source Seeking - Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Mode: {'Random NC files' if randomize and data_dir else f'Source {source_id}'}")

    # Carica modello
    model = PPO.load(model_path)

    # Carica normalizzatore se presente
    vec_normalize_path = Path(model_path).parent / "vec_normalize.pkl"
    has_vec_normalize = vec_normalize_path.exists()

    results = []
    trajectories = []

    # Per statistiche per sorgente
    stats_by_source = {'S1': [], 'S2': [], 'S3': []}

    for ep in range(n_episodes):
        # Crea ambiente (random o fisso)
        env = create_env(
            config,
            concentration_field=None,
            source_id=source_id,
            seed=ep,
            data_dir=data_dir,
            randomize_field=randomize
        )

        # Ottieni source_id effettivo dall'ambiente
        actual_source = env.source_id if hasattr(env, 'source_id') else source_id
        # Se wrapped in Monitor, accedi all'env interno
        if hasattr(env, 'env'):
            actual_source = env.env.source_id

        # Wrap per normalizzazione se necessario
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
            obs, _ = env.reset()

        done = False
        ep_reward = 0
        ep_trajectory = []
        ep_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            if is_vec_env:
                obs, reward, done, info = env.step(action)
                done = done[0]
                info = info[0]
                reward = reward[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            ep_reward += reward
            ep_steps += 1

            if 'position' in info:
                ep_trajectory.append(info['position'])

            if render and not is_vec_env:
                env.render()

        # Risultati episodio
        source_reached = info.get('source_reached', False)
        final_distance = info.get('distance_to_source', -1)

        results.append({
            'episode': ep,
            'source_id': actual_source,
            'reward': ep_reward,
            'steps': ep_steps,
            'source_reached': source_reached,
            'final_distance': final_distance
        })

        stats_by_source[actual_source].append({
            'success': source_reached,
            'distance': final_distance,
            'steps': ep_steps
        })

        if save_trajectories:
            trajectories.append(ep_trajectory)

        status = "✓ SUCCESS" if source_reached else f"✗ {final_distance:.0f}m"
        print(f"  Episode {ep+1:3d} [{actual_source}]: reward={ep_reward:7.2f}, steps={ep_steps:3d}, {status}")

        env.close()

    # Statistiche globali
    print("\n" + "=" * 60)
    print("GLOBAL STATISTICS")
    print("=" * 60)

    rewards = [r['reward'] for r in results]
    success_rate = sum(r['source_reached'] for r in results) / n_episodes
    avg_distance = np.mean([r['final_distance'] for r in results if r['final_distance'] >= 0])
    avg_steps = np.mean([r['steps'] for r in results])

    print(f"  Total episodes:     {n_episodes}")
    print(f"  Success rate:       {success_rate:.1%}")
    print(f"  Mean reward:        {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean final dist:    {avg_distance:.1f} m")
    print(f"  Mean steps:         {avg_steps:.1f}")

    # Statistiche per sorgente
    print("\n" + "-" * 60)
    print("STATISTICS BY SOURCE")
    print("-" * 60)

    for src_id in ['S1', 'S2', 'S3']:
        src_stats = stats_by_source[src_id]
        if src_stats:
            src_success = sum(s['success'] for s in src_stats) / len(src_stats)
            src_dist = np.mean([s['distance'] for s in src_stats])
            src_steps = np.mean([s['steps'] for s in src_stats])
            print(f"  {src_id}: {len(src_stats):3d} episodes, success={src_success:.1%}, dist={src_dist:.1f}m, steps={src_steps:.1f}")
        else:
            print(f"  {src_id}: No episodes")

    print("=" * 60)

    return results, trajectories


def validate_model(
    model_path: str,
    data_dir: str,
    n_episodes_per_file: int = 5,
    config_path: str = "configs/config.yaml"
):
    """
    Validazione completa su tutti i file NC.
    Testa il modello su ogni file NC separatamente.

    Args:
        model_path: Path al modello
        data_dir: Directory con i file NC
        n_episodes_per_file: Episodi per ogni file NC
        config_path: Path al config
    """
    if not STABLE_BASELINES_AVAILABLE:
        raise ImportError("stable-baselines3 non installato")

    from utils.data_loader import NetCDFLoader

    print("=" * 60)
    print("HYDRAS Source Seeking - Full Validation")
    print("=" * 60)

    # Trova tutti i file NC
    nc_files = sorted(Path(data_dir).glob("*.nc"))
    print(f"Found {len(nc_files)} NC files\n")

    config = load_config(config_path)
    model = PPO.load(model_path)

    # Carica normalizzatore
    vec_normalize_path = Path(model_path).parent / "vec_normalize.pkl"
    has_vec_normalize = vec_normalize_path.exists()

    all_results = []

    for nc_file in nc_files:
        print(f"\n{'─'*60}")
        print(f"Testing: {nc_file.name}")
        print(f"{'─'*60}")

        # Carica il campo
        loader = NetCDFLoader(nc_file.parent)
        field = loader.load(str(nc_file), concentration_var="Concentration - component 1")

        source_id = 'S1' if 'S1' in nc_file.name else ('S2' if 'S2' in nc_file.name else 'S3')

        file_results = []

        for ep in range(n_episodes_per_file):
            env = create_env(config, field, source_id, seed=ep)

            if has_vec_normalize:
                env = DummyVecEnv([lambda: env])
                env = VecNormalize.load(str(vec_normalize_path), env)
                env.training = False
                env.norm_reward = False
                obs = env.reset()
                is_vec = True
            else:
                obs, _ = env.reset()
                is_vec = False

            done = False
            ep_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if is_vec:
                    obs, reward, done, info = env.step(action)
                    done, info, reward = done[0], info[0], reward[0]
                else:
                    obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                ep_reward += reward

            file_results.append({
                'success': info.get('source_reached', False),
                'distance': info.get('distance_to_source', -1),
                'reward': ep_reward
            })

            env.close()

        # Stats per questo file
        success_rate = sum(r['success'] for r in file_results) / len(file_results)
        avg_dist = np.mean([r['distance'] for r in file_results])

        print(f"  Success: {success_rate:.0%} ({sum(r['success'] for r in file_results)}/{len(file_results)})")
        print(f"  Avg distance: {avg_dist:.1f} m")

        all_results.append({
            'file': nc_file.name,
            'source': source_id,
            'success_rate': success_rate,
            'avg_distance': avg_dist,
            'episodes': file_results
        })

    # Summary finale
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total_success = sum(r['success_rate'] * len(r['episodes']) for r in all_results)
    total_episodes = sum(len(r['episodes']) for r in all_results)

    print(f"\n  Overall success rate: {total_success/total_episodes:.1%}")
    print(f"  Total episodes: {total_episodes}")

    print(f"\n  {'File':<35} {'Source':<6} {'Success':<10} {'Avg Dist':<10}")
    print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*10}")
    for r in all_results:
        print(f"  {r['file']:<35} {r['source']:<6} {r['success_rate']:.0%}       {r['avg_distance']:.1f} m")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="HYDRAS Source Seeking - PPO Training"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--config', '-c',
        default='configs/config.yaml',
        help='Path to config file'
    )
    train_parser.add_argument(
        '--output', '-o',
        default='outputs',
        help='Output directory'
    )
    train_parser.add_argument(
        '--source', '-s',
        default='S1',
        choices=['S1', 'S2', 'S3'],
        help='Source ID (ignored if --data-dir is used with randomization)'
    )
    train_parser.add_argument(
        '--n-envs', '-n',
        type=int,
        default=4,
        help='Number of parallel environments'
    )
    train_parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=None,
        help='Total timesteps (overrides config)'
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    train_parser.add_argument(
        '--resume',
        default=None,
        help='Path to checkpoint to resume from'
    )
    train_parser.add_argument(
        '--nc-file',
        default=None,
        help='Path to single NetCDF file'
    )
    train_parser.add_argument(
        '--data-dir',
        default=None,
        help='Directory with multiple NC files (enables randomization)'
    )

    # Evaluate command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained model')
    eval_parser.add_argument(
        'model',
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--config', '-c',
        default='configs/config.yaml',
        help='Path to config file'
    )
    eval_parser.add_argument(
        '--source', '-s',
        default='S1',
        choices=['S1', 'S2', 'S3'],
        help='Source ID (ignored if --data-dir with --randomize)'
    )
    eval_parser.add_argument(
        '--episodes', '-n',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    eval_parser.add_argument(
        '--render', '-r',
        action='store_true',
        help='Render episodes'
    )
    eval_parser.add_argument(
        '--data-dir',
        default=None,
        help='Directory with NC files'
    )
    eval_parser.add_argument(
        '--randomize',
        action='store_true',
        help='Use random NC file for each episode'
    )

    # Validate command (test su tutti i file NC)
    val_parser = subparsers.add_parser('validate', help='Full validation on all NC files')
    val_parser.add_argument(
        'model',
        help='Path to trained model'
    )
    val_parser.add_argument(
        '--data-dir', '-d',
        required=True,
        help='Directory with NC files'
    )
    val_parser.add_argument(
        '--episodes-per-file', '-n',
        type=int,
        default=5,
        help='Episodes per NC file'
    )
    val_parser.add_argument(
        '--config', '-c',
        default='configs/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    if args.command == 'train':
        train(
            config_path=args.config,
            output_dir=args.output,
            source_id=args.source,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            seed=args.seed,
            resume_from=args.resume,
            use_nc_data=args.nc_file is not None or args.data_dir is not None,
            nc_file=args.nc_file,
            data_dir=args.data_dir
        )

    elif args.command == 'eval':
        evaluate(
            model_path=args.model,
            config_path=args.config,
            source_id=args.source,
            n_episodes=args.episodes,
            render=args.render,
            data_dir=args.data_dir,
            randomize=args.randomize
        )

    elif args.command == 'validate':
        validate_model(
            model_path=args.model,
            data_dir=args.data_dir,
            n_episodes_per_file=args.episodes_per_file,
            config_path=args.config
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()