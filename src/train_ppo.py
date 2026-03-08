"""
HYDRAS Source Seeking - PPO Training Script
Addestramento di un agente singolo per il source seeking
utilizzando Proximal Policy Optimization (PPO).
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non-interattivo
import matplotlib.pyplot as plt

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym

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

from utils.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from utils.data_loader import DataManager, ConcentrationField


class SourceSeekingCallback(BaseCallback):
    """
    Callback per logging di success rate durante il training.
    Raccoglie loss e success_rate per i plot finali.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.success_rate = []
        self._loss_steps = []
        self._loss_values = []
        self._sr_steps = []
        self._sr_values = []

    def _on_step(self) -> bool:
        # Raccogli info dagli ambienti
        infos = self.locals.get('infos', [])

        for info in infos:
            if 'source_reached' in info:
                self.success_rate.append(float(info['source_reached']))

        # Log ogni 1000 steps
        if self.num_timesteps % 1000 == 0 and len(self.success_rate) > 0:
            recent_success = np.mean(self.success_rate[-100:]) if len(self.success_rate) >= 100 else np.mean(self.success_rate)
            self.logger.record('custom/success_rate', recent_success)
            self._sr_steps.append(self.num_timesteps)
            self._sr_values.append(recent_success)

        # Raccogli loss dal logger di SB3
        if self.num_timesteps % 1000 == 0:
            loss = self.logger.name_to_value.get('train/loss', None)
            if loss is not None:
                self._loss_steps.append(self.num_timesteps)
                self._loss_values.append(loss)

        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print(f"\nTraining completed!")
            if len(self.success_rate) > 0:
                print(f"Final success rate: {np.mean(self.success_rate[-100:]):.2%}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Carica la configurazione da file YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env(
    config: Dict[str, Any],
    concentration_field: Optional[ConcentrationField] = None,
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
        max_velocity=agent_config.get('max_velocity', 1.0),
        sensor_radius=agent_config.get('sensor_radius', 50),
        n_sensors=agent_config.get('n_concentration_samples', 8),
        memory_length=agent_config.get('memory_length', 10),
        dt=env_config.get('dt', 10),
        max_steps=env_config.get('max_episode_steps', 500),
        source_found_reward=env_config.get('reward', {}).get('source_reached_bonus', 100),
        step_penalty=env_config.get('reward', {}).get('step_penalty', -0.1),
        boundary_penalty=env_config.get('reward', {}).get('boundary_penalty', -10),
        source_distance_threshold=env_config.get('reward', {}).get('distance_threshold', 100),
        action_type=agent_config.get('action_type', 'continuous'),
        auto_detect_source=env_config.get('reward', {}).get('auto_detect_source', False),
    )

    print(f"  [DEBUG] spawn=on_plume, threshold={env_kwargs.source_distance_threshold}m, auto_detect={env_kwargs.auto_detect_source}")

    # Crea ambiente
    env = SourceSeekingEnv(
        config=env_kwargs,
        concentration_field=concentration_field,
        data_dir=data_dir,
        randomize_field=randomize_field
    )

    # Wrap con Monitor per logging
    env = Monitor(env)

    return env


def make_env_fn(
    config: Dict[str, Any],
    concentration_field: Optional[ConcentrationField],
    rank: int,
    seed: int,
    data_dir: Optional[str] = None,
    randomize_field: bool = False
) -> Callable[[], gym.Env]:
    """Factory function per la creazione di ambienti paralleli."""
    def _init() -> gym.Env:
        env = create_env(config, concentration_field, seed + rank, data_dir, randomize_field)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    config_path: str = "utils/config.yaml",
    output_dir: str = "trained_models",
    n_envs: int = 4,
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    resume_from: Optional[str] = None,
    data_dir: Optional[str] = None,
):
    """
    Funzione principale di training.

    Args:
        config_path: Path al file di configurazione
        output_dir: Directory per salvare i risultati
        n_envs: Numero di ambienti paralleli
        total_timesteps: Timesteps totali (override config)
        seed: Seed per riproducibilità
        resume_from: Path a checkpoint per riprendere training
        data_dir: Directory con file NC (sceglie random ad ogni episodio)
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
    run_name = f"ppo_{timestamp}"
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
    print(f"Number of parallel environments: {n_envs}")

    # Carica dati
    concentration_field = None
    randomize_field = False

    if data_dir:
        print(f"\nUsing ALL NC files from: {data_dir}")
        print("  Mode: Random field each episode")
        randomize_field = True
    else:
        print("\nUsing synthetic concentration field")

    # Crea ambienti vettorizzati
    print(f"\nCreating {n_envs} parallel environments...")

    timesteps = total_timesteps or training_config.get('total_timesteps', 1000000)

    env_fns = [
        make_env_fn(config, concentration_field, i, seed, data_dir, randomize_field)
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
        make_env_fn(config, concentration_field, 0, seed + 1000, data_dir, False)
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

    # Calcola timesteps totali
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

    # --- Salva plot di Loss e Success Rate ---
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Plot Loss
    if custom_callback._loss_steps:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(custom_callback._loss_steps, custom_callback._loss_values, linewidth=0.8)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / 'training_loss.png', dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to: {plots_dir / 'training_loss.png'}")

    # Plot Success Rate
    if custom_callback._sr_steps:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(custom_callback._sr_steps, custom_callback._sr_values, linewidth=0.8)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Success Rate')
        ax.set_title('Training Success Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / 'training_success_rate.png', dpi=150)
        plt.close(fig)
        print(f"Success rate plot saved to: {plots_dir / 'training_success_rate.png'}")

    # Cleanup
    vec_env.close()
    eval_env.close()

    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")

    return model, run_dir


def main():
    """Avvia il training con configurazione fissa."""
    train(
        config_path="utils/config.yaml",
        output_dir="trained_models",
        n_envs=4,
        seed=42,
        data_dir="data/",
    )


if __name__ == "__main__":
    main()