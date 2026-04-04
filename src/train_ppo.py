"""
HYDRAS Source Seeking - PPO Training Script
Addestramento di un agente singolo per il source seeking
utilizzando Proximal Policy Optimization (PPO).
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
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
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    STABLE_BASELINES_AVAILABLE = True
    
    # MaskablePPO per action masking (evita land collision)
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        MASKABLE_PPO_AVAILABLE = True
    except ImportError:
        MASKABLE_PPO_AVAILABLE = False
        print("WARNING: sb3-contrib non installato. Action masking non disponibile.")
        
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    MASKABLE_PPO_AVAILABLE = False
    print("WARNING: stable-baselines3 non installato. Alcune funzionalità non saranno disponibili.")

from utils.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from utils.data_loader import ConcentrationField, DataManager, WindData, CurrentData


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
        # Raccogli info solo a fine episodio (dones=True)
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])

        for info, done in zip(infos, dones):
            if not done:
                continue
            # Episodio terminato: registra se successo o collisione terra
            if 'source_found' in info:  # Fixed: was 'source_reached', now 'source_found'
                self.success_rate.append(1.0)
            elif 'distance_to_source' in info:  # Episode ended without success
                self.success_rate.append(0.0)

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


class CurriculumCallback(BaseCallback):
    """
    Callback per implementare curriculum learning con progressione di sorgenti.
    
    Progressione:
    - Fase 1 (0-1M steps): SRC001-SRC035 (35 sorgenti, 1/3 dell'80%)
    - Fase 2 (1M-2M steps): SRC001-SRC070 (70 sorgenti, 2/3 dell'80%)
    - Fase 3 (2M-3M steps): SRC001-SRC106 (106 sorgenti, 80% del totale 132)
    """
    
    def __init__(
        self, 
        vec_env,
        curriculum_config: Dict[str, Any],
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.curriculum_config = curriculum_config
        self.current_phase = -1
        self.phases = curriculum_config.get('phases', [])
        
        if not self.phases:
            raise ValueError("No curriculum phases defined in config")
        
        # Estendi default sources list: SRC001-SRC132
        self._all_sources = [f"SRC{i:03d}" for i in range(1, 133)]
        
        print(f"[CurriculumCallback] Initialized with {len(self.phases)} phases")
        for i, phase in enumerate(self.phases):
            print(f"  Phase {i}: [{phase['start']:,} - {phase['end']:,} steps] "
                  f"-> {phase['num_sources']} sources")

    def _get_sources_for_step(self, step: int) -> List[str]:
        """Ritorna la lista di sorgenti alla quale agent deve avere accesso al passo dato."""
        for phase in self.phases:
            if phase['start'] <= step < phase['end']:
                num_sources = min(phase['num_sources'], len(self._all_sources))
                return self._all_sources[:num_sources]
        
        # Se oltre tutte le fasi, usa tutte le sorgenti disponibili
        return self._all_sources

    def _on_step(self) -> bool:
        # Determina la fase attuale
        current_sources = self._get_sources_for_step(self.num_timesteps)
        current_phase = None
        
        for i, phase in enumerate(self.phases):
            if phase['start'] <= self.num_timesteps < phase['end']:
                current_phase = i
                break
        
        # Se cambia fase, aggiorna allowed_sources e disconnexiona messaggi
        if current_phase != self.current_phase:
            self.current_phase = current_phase
            
            if current_phase is not None:
                phase_info = self.phases[current_phase]
                print(f"\n[Step {self.num_timesteps:,}] Transitioning to Phase {current_phase + 1}: "
                      f"{len(current_sources)} sources ({current_sources[0]} - {current_sources[-1]})")
                self.logger.record(f'curriculum/phase', current_phase)
            else:
                print(f"\n[Step {self.num_timesteps:,}] Training beyond defined phases")
        
        # Aggiorna allowed_sources in tutti gli ambienti
        for env in self.vec_env.envs:
            inner = env
            while hasattr(inner, 'env'):
                inner = inner.env
            inner.allowed_sources = current_sources
        
        self.logger.record('curriculum/n_sources', len(current_sources))
        
        return True


class SyncNormCallback(BaseCallback):
    """
    Callback per sincronizzare le statistiche VecNormalize tra train_env e eval_env.
    
    Senza sincronizzazione, le running stats divergono durante il training
    e l'EvalCallback valuta su statistiche diverse da quelle di training.
    """

    def __init__(self, train_env, eval_env, eval_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        # Sincronizza prima di ogni valutazione
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            sync_envs_normalization(self.train_env, self.eval_env)
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps}] Synced VecNormalize stats")
        return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Carica la configurazione da file YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)





def create_env(
    config: Dict[str, Any],
    concentration_field: Optional[ConcentrationField] = None,
    wind_data: Optional['WindData'] = None,
    current_data: Optional['CurrentData'] = None,
    data_dir: Optional[str] = None,
    randomize_field: bool = False,
    chunk_id: int = 0,
    data_manager: Optional[DataManager] = None,
    wind_mapping: Optional[Dict[str, str]] = None
) -> gym.Env:
    """
    Crea un'istanza dell'ambiente con i wrapper appropriati.
    
    Args:
        chunk_id: 0 = spawn @1/4, 1 = spawn @3/4 della simulazione
        data_manager: DataManager per caricamenti dinamici (opzionale)
        wind_mapping: Dict con mappatura run_id -> wind_filename (opzionale)
    """
    # Estrai configurazioni
    env_config = config.get('environment', {})
    agent_config = config.get('agent', {})
    domain_config = config.get('domain', {})

    # Crea config ambiente
    env_kwargs = SourceSeekingConfig(
        # Domain
        xmin=domain_config.get('xmin', 619000),
        xmax=domain_config.get('xmax', 622000),
        ymin=domain_config.get('ymin', 4794500),
        ymax=domain_config.get('ymax', 4797000),
        resolution=domain_config.get('grid_resolution', 10),
        # Agent
        max_velocity=agent_config.get('max_velocity', 1.0),
        memory_length=agent_config.get('memory_length', 9),
        dt=env_config.get('dt', 10),
        max_steps=env_config.get('max_episode_steps', 1080),
        source_found_reward=env_config.get('reward', {}).get('source_reached_bonus', 100),
        step_penalty=env_config.get('reward', {}).get('step_penalty', -0.1),
        boundary_penalty=env_config.get('reward', {}).get('boundary_penalty', -10),
        source_distance_threshold=env_config.get('reward', {}).get('distance_threshold', 100),
        distance_reward_multiplier=env_config.get('reward', {}).get('distance_reward_multiplier', 1.0),
        # Land avoidance
        land_proximity_threshold=env_config.get('reward', {}).get('land_proximity_threshold', 10.0),
        land_proximity_penalty_max=env_config.get('reward', {}).get('land_proximity_penalty_max', -5.0),
        n_discrete_actions=agent_config.get('n_discrete_actions', 8),
        # Spawn constraints
        spawn_min_distance=env_config.get('spawn', {}).get('min_distance', 200),
        spawn_max_distance=env_config.get('spawn', {}).get('max_distance', 1500),
        spawn_min_land_distance=env_config.get('spawn', {}).get('min_land_distance', 50.0),
        spawn_start_frame=env_config.get('spawn', {}).get('start_frame', 1440),
        spawn_conc_threshold=env_config.get('spawn', {}).get('conc_threshold', 0.5),
        chunk_id=chunk_id,
        # Plume reward
        plume_reward_positive=env_config.get('reward', {}).get('plume_reward_positive', 0.5),
        plume_reward_negative=env_config.get('reward', {}).get('plume_reward_negative', -0.5),
        plume_threshold=env_config.get('reward', {}).get('plume_threshold', 0.1),
        # Concentration gradient reward
        concentration_gradient_reward_positive=env_config.get('reward', {}).get('concentration_gradient_reward_positive', 0.05),
        concentration_gradient_reward_negative=env_config.get('reward', {}).get('concentration_gradient_reward_negative', -0.05),
        # Wind alignment reward
        wind_alignment_reward=env_config.get('reward', {}).get('wind_alignment_reward', 0.1),
        wind_alignment_penalty=env_config.get('reward', {}).get('wind_alignment_penalty', -0.1),
    )

    print(f"  Success radius: {env_kwargs.source_distance_threshold}m")

    # Crea ambiente
    env = SourceSeekingEnv(
        config=env_kwargs,
        concentration_field=concentration_field,
        wind_data=wind_data,
        current_data=current_data,
        data_dir=data_dir,
        randomize_field=randomize_field,
        data_manager=data_manager,
        wind_mapping=wind_mapping
    )

    # Wrap con Monitor per logging
    env = Monitor(env)

    return env


def mask_fn(env: gym.Env) -> np.ndarray:
    """Funzione per estrarre la maschera azioni dall'env.
    
    Attraversa i wrapper (Monitor, etc.) per raggiungere SourceSeekingEnv.
    """
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    return inner.action_masks()


def make_env_fn(
    config: Dict[str, Any],
    concentration_field: Optional[ConcentrationField],
    wind_data: Optional['WindData'] = None,
    current_data: Optional['CurrentData'] = None,
    rank: int = 0,
    chunk_id: int = 0,
    seed: int = 42,
    data_dir: Optional[str] = None,
    randomize_field: bool = False,
    use_action_masking: bool = True,
    data_manager: Optional[DataManager] = None,
    wind_mapping: Optional[Dict[str, str]] = None
) -> Callable[[], gym.Env]:
    """Factory function per la creazione di ambienti paralleli.
    
    Args:
        wind_data: Dati di vento (condivisi tra ambienti)
        current_data: Dati di corrente (condivisi tra ambienti)
        chunk_id: 0 = spawn @1/4, 1 = spawn @3/4 della simulazione
        data_manager: DataManager per caricamenti dinamici
        wind_mapping: Dict con mappatura run_id -> wind_filename
    """
    def _init() -> gym.Env:
        env = create_env(
            config, 
            concentration_field, 
            wind_data, 
            current_data, 
            data_dir, 
            randomize_field, 
            chunk_id=chunk_id,
            data_manager=data_manager,
            wind_mapping=wind_mapping
        )
        env.reset(seed=seed + rank)
        
        # Applica action masking se disponibile
        if use_action_masking and MASKABLE_PPO_AVAILABLE:
            env = ActionMasker(env, mask_fn)
        
        return env
    return _init


def train(
    config_path: str = "utils/config.yaml",
    output_dir: str = "trained_models",
    n_envs: int = 4,
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    resume_from: Optional[str] = None,
    *,
    data_dir: str,
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
        data_dir: Directory con file NC (obbligatoria)
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
    print(f"Number of parallel workers: {n_envs}")

    # Carica dati
    concentration_field = None
    randomize_field = True

    print(f"\nUsing ALL NC files from: {data_dir}")
    print("  Mode: Random field each episode")
    
    # Per il training randomizzato, carichiamo una sample di wind e current data
    # (in generale, ogni episodio potrebbe usare una coppia diversa)
    data_manager = DataManager(
        data_dir=data_dir,
        preload_all=False,
        wind_filename="CI_WIND_faseII_V1.txt",  # Nuovo file V1 per 132 sorgenti
        current_filename="CL02_V1_SRC000_U_V_10mGrid.nc"  # Unico file U_V per tutte le sorgenti
    )
    
    # ESCUDI V1 DAL TRAINING: mantieni solo V0, V2, V3
    data_manager._nc_files = [f for f in data_manager._nc_files if '_V1_' not in f.name]
    print(f"  Training data (excluding V1): {len(data_manager._nc_files)} files")
    
    wind_data = data_manager.get_wind_data()
    current_data = data_manager.get_current_data()
    discovered_sources = data_manager.get_discovered_sources()
    
    print(f"\nDiscovered {len(discovered_sources)} sources: {discovered_sources[:10]}... (e altri)")
    print(f"  Training set (80%): SRC001-SRC106 (106 sources)")
    print(f"  Inference set (20%): SRC107-SRC132 (26 sources)")
    print(f"Wind data: {'LOADED' if wind_data else 'NOT FOUND'} ({wind_data.dt if wind_data else 'N/A'} min intervals)")
    print(f"Current data: {'LOADED' if current_data else 'NOT FOUND'} ({current_data.n_timesteps if current_data else 'N/A'} timesteps)")
    
    if wind_data is None or current_data is None:
        print("\nWARNING: Wind or current data not loaded. Will run without them.")
    
    # Wind mapping initialization (empty - legacy support removed)
    wind_mapping = {}

    # Crea ambienti vettorizzati
    print(f"\nCreating {n_envs*2} parallel environments...")
    print(f"  (2 chunks per file: spawn @1/4 e @3/4 della simulazione)")

    timesteps = total_timesteps or training_config.get('total_timesteps', 900000)

    # Crea 2 environments per ogni "file" (rank):
    # - chunk_id=0: spawn @1/4 della simulazione
    # - chunk_id=1: spawn @3/4 della simulazione
    env_fns = [
        make_env_fn(
            config, concentration_field, wind_data, current_data, i, chunk_id, 
            seed, data_dir, randomize_field,
            data_manager=data_manager,
            wind_mapping=wind_mapping
        )
        for i in range(n_envs) for chunk_id in [0, 1]
    ]

    # DummyVecEnv sempre (necessario per accesso diretto agli env)
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
        make_env_fn(
            config, concentration_field, wind_data, current_data, 0, 0, 
            seed + 1000, data_dir, False,
            data_manager=data_manager,
            wind_mapping=wind_mapping
        )
    ])
    if config.get('environment', {}).get('normalize_obs', True):
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            training=False
        )
        # Sincronizza le running stats dall'env di training
        sync_envs_normalization(vec_env, eval_env)

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

    # Scegli algoritmo: MaskablePPO se disponibile, altrimenti PPO standard
    use_maskable = MASKABLE_PPO_AVAILABLE
    PPOClass = MaskablePPO if use_maskable else PPO
    algo_name = "MaskablePPO" if use_maskable else "PPO"
    
    # Crea o carica modello
    if resume_from:
        print(f"\nResuming training from: {resume_from}")
        model = PPOClass.load(
            resume_from,
            env=vec_env,
            tensorboard_log=str(log_dir / "tensorboard")
        )
    else:
        print(f"\nCreating new {algo_name} model...")
        if use_maskable:
            print("  Action masking: ENABLED (evita land collision)")
        model = PPOClass(
            policy=training_config.get('policy', 'MlpPolicy'),
            env=vec_env,
            learning_rate=training_config.get('learning_rate', 5e-5),
            n_steps=training_config.get('n_steps', 4096),
            batch_size=training_config.get('batch_size', 64),
            n_epochs=training_config.get('n_epochs', 10),
            gamma=training_config.get('gamma', 0.99),
            gae_lambda=training_config.get('gae_lambda', 0.95),
            clip_range=training_config.get('clip_range', 0.2),
            ent_coef=training_config.get('ent_coef', 0.05),
            vf_coef=training_config.get('vf_coef', 0.3),
            max_grad_norm=training_config.get('max_grad_norm', 0.5),
            policy_kwargs=sb3_policy_kwargs,
            tensorboard_log=str(log_dir / "tensorboard"),
            verbose=training_config.get('verbose', 1),
            seed=seed,
            device='cpu'  # Forza CPU - MlpPolicy è più efficiente su CPU
        )

    print(f"\nModel architecture:")
    print(f"  Algorithm: {algo_name}")
    print(f"  Policy: {training_config.get('policy', 'MlpPolicy')}")
    print(f"  Network: {net_arch}")
    print(f"  Activation: {activation_fn_name}")
    print(f"  Device: {model.device}")

    # Setup callbacks
    callbacks = []

    # Curriculum Learning callback (applica fasi progressive di sorgenti)
    if config.get('curriculum', {}).get('enabled', False):
        curriculum_callback = CurriculumCallback(
            vec_env=vec_env,
            curriculum_config=config.get('curriculum', {}),
            verbose=1
        )
        callbacks.append(curriculum_callback)
        print("\n[Curriculum Learning] ENABLED")
    else:
        print("\n[Curriculum Learning] DISABLED")

    # Evaluation callback
    eval_freq = training_config.get('eval_freq', 10000) // n_envs
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=training_config.get('n_eval_episodes', 10),
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)

    # Sync normalization callback (sincronizza stats prima di ogni eval)
    if config.get('environment', {}).get('normalize_obs', True):
        sync_callback = SyncNormCallback(
            train_env=vec_env,
            eval_env=eval_env,
            eval_freq=eval_freq * n_envs,  # Converti in timesteps totali
            verbose=0
        )
        callbacks.append(sync_callback)

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
    plots_dir = run_dir / "plots"
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
    """Avvia il training con configurazione di default o fine-tuning da modello esistente."""
    import os
    os.chdir(PROJECT_ROOT)  # Assicura CWD = root del progetto

    config_path = str(PROJECT_ROOT / "utils" / "config.yaml")
    output_dir = str(PROJECT_ROOT / "trained_models")
    data_dir = str(PROJECT_ROOT / "data")  # Carica da tutte le versioni (V0, V2, V3 tramite filtro)

    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Cartella dati NC non trovata: {data_dir}\n"
            f"Scarica i file .nc di simulazione MIKE21 nella cartella 'data/'"
        )

    # Cerca il modello più recente per fine-tuning
    resume_from = None
    trained_dir = Path(output_dir)
    
    if trained_dir.exists():
        run_dirs = sorted([d for d in trained_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_")])
        
        if run_dirs:
            latest_run = run_dirs[-1]
            
            # Prova a trovare best_model, poi final_model
            model_candidates = [
                latest_run / "models" / "best" / "best_model.zip",
                latest_run / "models" / "final_model.zip",
            ]
            
            for model_path in model_candidates:
                if model_path.exists():
                    resume_from = str(model_path)
                    print(f"Found latest model for fine-tuning: {resume_from}\n")
                    break
    
    train(
        config_path=config_path,
        output_dir=output_dir,
        n_envs=4,
        seed=42,
        data_dir=data_dir,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()