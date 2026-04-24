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
        CheckpointCallback,
        CallbackList,
        BaseCallback
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
    from stable_baselines3.common.monitor import Monitor
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
            # Episodio terminato: successo se la sorgente è stata raggiunta.
            self.success_rate.append(1.0 if info.get('source_reached', False) else 0.0)

        # Log ogni 1000 steps (almeno 20 episodi per stabilità statistica)
        if self.num_timesteps % 1000 == 0 and len(self.success_rate) >= 20:
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
    
    Progressione su 6M timesteps su tutte e 4 le versioni (V0, V1, V2, V3):
    - Fase 1 (0-0.9M): SRC001-SRC020
    - Fase 2 (0.9M-1.8M): SRC001-SRC035
    - Fase 3 (1.8M-3.0M): SRC001-SRC055
    - Fase 4 (3.0M-4.5M): SRC001-SRC080
    - Fase 5 (4.5M-6.0M): SRC001-SRC106
    Ogni sorgente equivale a 4 scenari (V0+V1+V2+V3).
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
            num_sources = phase['num_sources']
            estimated_files_per_source = 4  # V0, V1, V2, V3 (4 versioni!)
            estimated_files = num_sources * estimated_files_per_source
            print(f"  Phase {i}: [{phase['start']:,} - {phase['end']:,} steps] "
                  f"-> {num_sources} sources × {estimated_files_per_source} versions ≈ {estimated_files} files")

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


class MultiScenarioEvalCallback(BaseCallback):
    """
    Valuta su N scenari (source, version, chunk_id) e salva il miglior modello
    in base al success rate medio su tutti gli scenari.

    Più rappresentativo di un EvalCallback fisso su un solo scenario.
    """

    _CHUNK_LABEL = {0: 'Q1/4', 1: 'Q1/2', 2: 'Q3/4'}

    def __init__(
        self,
        scenarios: List[Dict[str, Any]],
        data_manager: 'DataManager',
        config: Dict[str, Any],
        train_vec_env,
        wind_mapping: Dict[str, str],
        current_mapping: Dict[str, str],
        best_model_save_path: str,
        eval_freq: int,
        n_eval_episodes: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.scenarios = scenarios
        self.data_manager = data_manager
        self.config = config
        self.train_vec_env = train_vec_env
        self.wind_mapping = wind_mapping
        self.current_mapping = current_mapping
        self.best_model_save_path = Path(best_model_save_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_sr = -np.inf

        # Pre-carica i field di concentrazione per ogni scenario
        self._fields: Dict[str, Any] = {}
        for s in scenarios:
            key = f"{s['source']}_{s['version']}"
            if key not in self._fields:
                field = self._load_field(s['source'], s['version'])
                if field is not None:
                    self._fields[key] = field

        print(f"[MultiScenarioEvalCallback] Scenari di eval ({len(scenarios)}):")
        for s in scenarios:
            key = f"{s['source']}_{s['version']}"
            status = "OK" if key in self._fields else "NOT FOUND"
            print(f"  - {s['source']} {s['version']} {self._CHUNK_LABEL[s['chunk_id']]} [{status}]")

    def _load_field(self, source_id: str, version: str):
        pattern = f"_{version}_{source_id}_"
        candidates = [f for f in self.data_manager._nc_files
                      if pattern in f.name and 'Conc' in f.name]
        if not candidates:
            print(f"  WARNING: nessun file trovato per {source_id}_{version}")
            return None
        field = self.data_manager._nc_loader.load(
            str(sorted(candidates)[0]),
            concentration_var="Concentration - component 1"
        )
        if field is None:
            return None
        field.run_id = f"{source_id}_{version}"
        coords = self.data_manager.get_source_coordinates(source_id)
        if coords:
            field.source_position = coords
        return field

    def _build_eval_env(self, chunk_id: int, field):
        env_config = self.config.get('environment', {})
        agent_config = self.config.get('agent', {})
        domain_config = self.config.get('domain', {})
        reward_cfg = env_config.get('reward', {})
        spawn_cfg = env_config.get('spawn', {})

        env_kwargs = SourceSeekingConfig(
            xmin=domain_config.get('xmin', 619000),
            xmax=domain_config.get('xmax', 622000),
            ymin=domain_config.get('ymin', 4794500),
            ymax=domain_config.get('ymax', 4797000),
            resolution=domain_config.get('grid_resolution', 10),
            max_velocity=agent_config.get('max_velocity', 1.0),
            memory_length=agent_config.get('memory_length', 9),
            dt=env_config.get('dt', 10),
            max_steps=env_config.get('max_episode_steps', 1080),
            source_found_reward=reward_cfg.get('source_reached_bonus', 100),
            step_penalty=reward_cfg.get('step_penalty', -0.1),
            boundary_penalty=reward_cfg.get('boundary_penalty', -10),
            source_distance_threshold=reward_cfg.get('distance_threshold', 50),
            distance_reward_multiplier=reward_cfg.get('distance_reward_multiplier', 1.0),
            land_proximity_threshold=reward_cfg.get('land_proximity_threshold', 10.0),
            land_proximity_penalty_max=reward_cfg.get('land_proximity_penalty_max', -5.0),
            n_discrete_actions=agent_config.get('n_discrete_actions', 8),
            spawn_min_land_distance=spawn_cfg.get('min_land_distance', 50.0),
            spawn_start_frame=spawn_cfg.get('start_frame', 1440),
            spawn_conc_threshold=spawn_cfg.get('conc_threshold', 0.5),
            chunk_id=chunk_id,
            plume_reward_positive=reward_cfg.get('plume_reward_positive', 0.5),
            plume_reward_negative=reward_cfg.get('plume_reward_negative', -0.5),
            plume_stay_reward=reward_cfg.get('plume_stay_reward', 0.5),
            plume_reentry_reward=reward_cfg.get('plume_reentry_reward', 0.25),
            plume_exit_penalty=reward_cfg.get('plume_exit_penalty', -1.5),
            outside_plume_distance_reward_scale=reward_cfg.get('outside_plume_distance_reward_scale', 0.35),
            plume_threshold=reward_cfg.get('plume_threshold', 0.1),
            concentration_gradient_reward_positive=reward_cfg.get('concentration_gradient_reward_positive', 0.05),
            concentration_gradient_reward_negative=reward_cfg.get('concentration_gradient_reward_negative', -0.05),
            wind_alignment_reward=reward_cfg.get('wind_alignment_reward', 0.05),
            wind_alignment_penalty=reward_cfg.get('wind_alignment_penalty', -0.05),
            current_alignment_reward=reward_cfg.get('current_alignment_reward', 0.05),
            current_alignment_penalty=reward_cfg.get('current_alignment_penalty', -0.05),
            stagnation_window=reward_cfg.get('stagnation_window', 50),
            stagnation_distance_threshold=reward_cfg.get('stagnation_distance_threshold', 20.0),
            stagnation_penalty=reward_cfg.get('stagnation_penalty', -0.5),
        )

        raw_env = SourceSeekingEnv(
            config=env_kwargs,
            concentration_field=field,
            data_manager=self.data_manager,
            wind_mapping=self.wind_mapping,
            current_mapping=self.current_mapping,
        )
        raw_env = Monitor(raw_env)
        if MASKABLE_PPO_AVAILABLE:
            raw_env = ActionMasker(raw_env, mask_fn)

        vec_env = DummyVecEnv([lambda e=raw_env: e])
        if isinstance(self.train_vec_env, VecNormalize):
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
            sync_envs_normalization(self.train_vec_env, vec_env)
        return vec_env

    def _eval_scenario(self, scenario: Dict[str, Any]) -> float:
        key = f"{scenario['source']}_{scenario['version']}"
        field = self._fields.get(key)
        if field is None:
            return 0.0

        vec_env = self._build_eval_env(scenario['chunk_id'], field)
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs = vec_env.reset()
            done = False
            while not done:
                if MASKABLE_PPO_AVAILABLE:
                    action, _ = self.model.predict(
                        obs, deterministic=True,
                        action_masks=vec_env.env_method('action_masks')[0]
                    )
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = vec_env.step(action)
                done = dones[0]
                if done and infos[0].get('source_reached', False):
                    successes += 1
        vec_env.close()
        return successes / self.n_eval_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0 or self.num_timesteps == 0:
            return True

        print(f"\n[Step {self.num_timesteps:,}] Multi-scenario evaluation:")
        success_rates = []
        for scenario in self.scenarios:
            sr = self._eval_scenario(scenario)
            label = f"{scenario['source']}_{scenario['version']}_{self._CHUNK_LABEL[scenario['chunk_id']]}"
            self.logger.record(f'eval/{label}', sr)
            success_rates.append(sr)
            print(f"  {label}: {sr:.0%}")

        mean_sr = float(np.mean(success_rates))
        self.logger.record('eval/mean_success_rate', mean_sr)
        print(f"  → Mean SR: {mean_sr:.1%}  (best so far: {self.best_mean_sr:.1%})")

        if mean_sr > self.best_mean_sr:
            self.best_mean_sr = mean_sr
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_save_path / "best_model"))
            print(f"  → New best model saved ({mean_sr:.1%})")

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
    wind_mapping: Optional[Dict[str, str]] = None,
    current_mapping: Optional[Dict[str, str]] = None
) -> gym.Env:
    """
    Crea un'istanza dell'ambiente con i wrapper appropriati.
    
    Args:
        chunk_id: 0 = spawn @1/4, 1 = spawn @1/2, 2 = spawn @3/4 della simulazione
        data_manager: DataManager per caricamenti dinamici (opzionale)
        wind_mapping: Dict con mappatura run_id -> wind_filename (opzionale)
        current_mapping: Dict con mappatura run_id -> current_filename (opzionale)
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
        source_distance_threshold=env_config.get('reward', {}).get('distance_threshold', 50),
        distance_reward_multiplier=env_config.get('reward', {}).get('distance_reward_multiplier', 1.0),
        # Land avoidance
        land_proximity_threshold=env_config.get('reward', {}).get('land_proximity_threshold', 10.0),
        land_proximity_penalty_max=env_config.get('reward', {}).get('land_proximity_penalty_max', -5.0),
        n_discrete_actions=agent_config.get('n_discrete_actions', 8),
        # Spawn constraints
        spawn_min_land_distance=env_config.get('spawn', {}).get('min_land_distance', 50.0),
        spawn_start_frame=env_config.get('spawn', {}).get('start_frame', 1440),
        spawn_conc_threshold=env_config.get('spawn', {}).get('conc_threshold', 0.5),
        chunk_id=chunk_id,
        # Plume reward
        plume_reward_positive=env_config.get('reward', {}).get('plume_reward_positive', 0.5),
        plume_reward_negative=env_config.get('reward', {}).get('plume_reward_negative', -0.5),
        plume_stay_reward=env_config.get('reward', {}).get('plume_stay_reward', 0.5),
        plume_reentry_reward=env_config.get('reward', {}).get('plume_reentry_reward', 0.25),
        plume_exit_penalty=env_config.get('reward', {}).get('plume_exit_penalty', -1.5),
        outside_plume_distance_reward_scale=env_config.get('reward', {}).get('outside_plume_distance_reward_scale', 0.35),
        plume_threshold=env_config.get('reward', {}).get('plume_threshold', 0.1),
        # Concentration gradient reward
        concentration_gradient_reward_positive=env_config.get('reward', {}).get('concentration_gradient_reward_positive', 0.05),
        concentration_gradient_reward_negative=env_config.get('reward', {}).get('concentration_gradient_reward_negative', -0.05),
        # Wind alignment reward
        wind_alignment_reward=env_config.get('reward', {}).get('wind_alignment_reward', 0.05),
        wind_alignment_penalty=env_config.get('reward', {}).get('wind_alignment_penalty', -0.05),
        # Current alignment reward
        current_alignment_reward=env_config.get('reward', {}).get('current_alignment_reward', 0.05),
        current_alignment_penalty=env_config.get('reward', {}).get('current_alignment_penalty', -0.05),
        # Stagnation penalty
        stagnation_window=env_config.get('reward', {}).get('stagnation_window', 50),
        stagnation_distance_threshold=env_config.get('reward', {}).get('stagnation_distance_threshold', 20.0),
        stagnation_penalty=env_config.get('reward', {}).get('stagnation_penalty', -0.5),
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
        wind_mapping=wind_mapping,
        current_mapping=current_mapping
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
    wind_mapping: Optional[Dict[str, str]] = None,
    current_mapping: Optional[Dict[str, str]] = None,
    allowed_sources: Optional[List[str]] = None
) -> Callable[[], gym.Env]:
    """Factory function per la creazione di ambienti paralleli.
    
    Args:
        wind_data: Dati di vento (condivisi tra ambienti)
        current_data: Dati di corrente (condivisi tra ambienti)
        chunk_id: 0 = spawn @1/4, 1 = spawn @1/2, 2 = spawn @3/4 della simulazione
        data_manager: DataManager per caricamenti dinamici
        wind_mapping: Dict con mappatura run_id -> wind_filename
        current_mapping: Dict con mappatura run_id -> current_filename
        allowed_sources: Lista di sorgenti disponibili (impostata prima del reset)
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
            wind_mapping=wind_mapping,
            current_mapping=current_mapping
        )
        
        # Imposta allowed_sources PRIMA del reset per evitare errori
        if allowed_sources is not None:
            inner = env
            while hasattr(inner, 'env'):
                inner = inner.env
            inner.allowed_sources = list(allowed_sources)
        
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
    # Nota: I file specifici qui sono solo default; vengono dinamicamente caricati per versione
    # tramite wind_mapping (BUG #2 FIX)
    data_manager = DataManager(
        data_dir=data_dir,
        preload_all=False,
        wind_filename="CI_WIND_faseII_V1.txt",  # Default (overridden da wind_mapping per ogni versione)
        current_filename="CL02_V1_SRC000_U_V_10mGrid.nc"  # Unico file U_V per tutte le sorgenti
    )
    
    # INCLUDE ALL VERSIONS: V0, V1, V2, V3 (80% training set = SRC001-SRC106)
    print(f"  Training data (all versions V0+V1+V2+V3): {len(data_manager._nc_files)} files")
    
    wind_data = data_manager.get_wind_data()
    current_data = data_manager.get_current_data()
    discovered_sources = data_manager.get_discovered_sources()
    
    print(f"\nDiscovered {len(discovered_sources)} sources: {discovered_sources[:10]}... (e altri)")
    print(f"  Training set (80%): SRC001-SRC106 (106 sources × 4 versions V0+V1+V2+V3)")
    print(f"  Inference set (20%): SRC107-SRC132 (26 sources × 4 versions)")
    print(f"Wind data: {'LOADED' if wind_data else 'NOT FOUND'} ({wind_data.dt if wind_data else 'N/A'} min intervals)")
    print(f"Current data: {'LOADED' if current_data else 'NOT FOUND'} ({current_data.n_timesteps if current_data else 'N/A'} timesteps)")
    
    if wind_data is None or current_data is None:
        print("\nWARNING: Wind or current data not loaded. Will run without them.")
    
    # Wind mapping: mappa versione -> wind_filename per caricamento dinamico
    # Questo permette di usare il vento corretto per ogni versione durante il training
    # (BUG #2 FIX: precedentemente il training usava sempre V1)
    wind_mapping = {
        "_V0": "CI_WIND_faseII_V0.txt",
        "_V1": "CI_WIND_faseII_V1.txt",
        "_V2": "CI_WIND_faseII_V2.txt",
        "_V3": "CI_WIND_faseII_V3.txt",
    }
    print(f"  Wind mapping: V0/V1/V2/V3 dinamici (corregge BUG #2)")
    print(f"    - Concentrazione Vx + Wind Vx caricati coerentemente")
    
    # Current mapping per caricamento dinamico corrente per versione (BUG #8 FIX)
    current_mapping = {
        "_V0": "CL02_V0_SRC000_U_V_10mGrid.nc",
        "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
        "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc",
        "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc",
    }
    print(f"  Current mapping: V0/V1/V2/V3 dinamici (corregge BUG #8)")
    print(f"    - Concentrazione Vx + Current Vx caricati coerentemente")

    # Crea ambienti vettorizzati
    raw_chunk_ids = training_config.get('chunk_ids', [0, 2])
    if not isinstance(raw_chunk_ids, list) or len(raw_chunk_ids) == 0:
        raise ValueError("training.chunk_ids must be a non-empty list containing values from [0, 1, 2]")

    chunk_ids: List[int] = []
    for cid in raw_chunk_ids:
        try:
            chunk_int = int(cid)
        except (TypeError, ValueError):
            continue
        if chunk_int in [0, 1, 2]:
            chunk_ids.append(chunk_int)
    chunk_ids = sorted(set(chunk_ids))

    if not chunk_ids:
        raise ValueError("No valid chunk IDs provided. Allowed values are [0, 1, 2]")

    chunk_desc = []
    if 0 in chunk_ids:
        chunk_desc.append("0=Q1/4")
    if 1 in chunk_ids:
        chunk_desc.append("1=Q1/2")
    if 2 in chunk_ids:
        chunk_desc.append("2=Q3/4")

    print(f"\nCreating {n_envs * len(chunk_ids)} parallel environments...")
    print(f"  (1 env per worker per chunk configurato)")
    print(f"  Chunk config: {chunk_desc}")

    timesteps = total_timesteps or training_config.get('total_timesteps', 6000000)

    # Crea environments per ogni worker su tutti i chunk configurati.
    env_fns = [
        make_env_fn(
            config, concentration_field, wind_data, current_data, i, chunk_id, 
            seed, data_dir, randomize_field,
            data_manager=data_manager,
            wind_mapping=wind_mapping,
            current_mapping=current_mapping,
            allowed_sources=discovered_sources  # Passa le sorgenti disponibili
        )
        for i in range(n_envs) for chunk_id in chunk_ids
    ]
    n_parallel_envs = len(env_fns)

    # DummyVecEnv sempre (necessario per accesso diretto agli env)
    base_vec_env = DummyVecEnv(env_fns)
    vec_env = base_vec_env

    # Normalizzazione
    use_vec_normalize = config.get('environment', {}).get('normalize_obs', True)
    norm_reward_enabled = config.get('environment', {}).get('normalize_reward', False)
    if use_vec_normalize:
        # In resume, ricarica le statistiche VecNormalize del run precedente
        # senza creare doppi wrapper VecNormalize.
        loaded_vecnorm = False
        if resume_from:
            resume_path = Path(resume_from)
            vecnorm_candidates = [
                resume_path.parent / "vec_normalize.pkl",            # .../models/vec_normalize.pkl
                resume_path.parent.parent / "vec_normalize.pkl",     # .../models/best/../vec_normalize.pkl
                resume_path.parent.parent.parent / "vec_normalize.pkl"
            ]
            for vecnorm_path in vecnorm_candidates:
                if vecnorm_path.exists():
                    vec_env = VecNormalize.load(str(vecnorm_path), base_vec_env)
                    vec_env.training = True
                    vec_env.norm_reward = norm_reward_enabled
                    print(f"  Loaded VecNormalize stats from: {vecnorm_path}")
                    loaded_vecnorm = True
                    break

        if not loaded_vecnorm:
            if resume_from:
                print("  WARNING: resume_from set but vec_normalize.pkl not found; using fresh normalization stats")
            vec_env = VecNormalize(
                base_vec_env,
                norm_obs=True,
                norm_reward=norm_reward_enabled,
                clip_obs=10.0
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

    # Scegli algoritmo: MaskablePPO se disponibile, altrimenti PPO standard
    use_maskable = MASKABLE_PPO_AVAILABLE
    PPOClass = MaskablePPO if use_maskable else PPO
    algo_name = "MaskablePPO" if use_maskable else "PPO"
    
    # Hyperparameters usati sia per modello nuovo che per fine-tuning (resume)
    model_kwargs = {
        'policy': training_config.get('policy', 'MlpPolicy'),
        'env': vec_env,
        'learning_rate': training_config.get('learning_rate', 5e-5),
        'n_steps': training_config.get('n_steps', 4096),
        'batch_size': training_config.get('batch_size', 64),
        'n_epochs': training_config.get('n_epochs', 10),
        'gamma': training_config.get('gamma', 0.99),
        'gae_lambda': training_config.get('gae_lambda', 0.95),
        'clip_range': training_config.get('clip_range', 0.2),
        'ent_coef': training_config.get('ent_coef', 0.05),
        'vf_coef': training_config.get('vf_coef', 0.3),
        'max_grad_norm': training_config.get('max_grad_norm', 0.5),
        'target_kl': training_config.get('target_kl', None),
        'policy_kwargs': sb3_policy_kwargs,
        'tensorboard_log': str(log_dir / "tensorboard"),
        'verbose': training_config.get('verbose', 1),
        'seed': seed,
        'device': 'cpu',  # Forza CPU - MlpPolicy è più efficiente su CPU
    }

    # Crea o carica modello
    if resume_from:
        print(f"\nResuming training from: {resume_from}")
        model = PPOClass(**model_kwargs)
        # Carica i pesi policy/value dal checkpoint ma mantiene gli hyperparams correnti
        # del fine-tuning (LR, ent_coef, n_epochs, target_kl, ecc.).
        loaded_model = PPOClass.load(resume_from, device='cpu')
        model.set_parameters(loaded_model.get_parameters(), exact_match=False)
        del loaded_model
    else:
        print(f"\nCreating new {algo_name} model...")
        if use_maskable:
            print("  Action masking: ENABLED (evita land collision)")
        model = PPOClass(**model_kwargs)

    print(f"\nModel architecture:")
    print(f"  Algorithm: {algo_name}")
    print(f"  Policy: {training_config.get('policy', 'MlpPolicy')}")
    print(f"  Network: {net_arch}")
    print(f"  Activation: {activation_fn_name}")
    print(f"  Device: {model.device}")

    # Setup callbacks
    callbacks = []

    # Curriculum Learning callback (applica fasi progressive di sorgenti)
    # BUG #9 FIX: Solo aggiungere callback se curriculum è abilitato
    if config.get('curriculum', {}).get('enabled', False):
        curriculum_callback = CurriculumCallback(
            vec_env=vec_env,
            curriculum_config=config.get('curriculum', {}),
            verbose=1
        )
        callbacks.append(curriculum_callback)

    # Multi-scenario eval callback
    eval_freq = max(1, training_config.get('eval_freq', 10000) // n_parallel_envs)
    eval_scenarios = training_config.get('eval_scenarios', [])
    eval_callback = MultiScenarioEvalCallback(
        scenarios=eval_scenarios,
        data_manager=data_manager,
        config=config,
        train_vec_env=vec_env,
        wind_mapping=wind_mapping,
        current_mapping=current_mapping,
        best_model_save_path=str(model_dir / "best"),
        eval_freq=eval_freq,
        n_eval_episodes=training_config.get('n_eval_episodes', 3),
        verbose=1,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, training_config.get('save_freq', 50000) // n_parallel_envs),
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
            progress_bar=training_config.get('progress_bar', True)
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

    print(f"\nTraining completed!")
    print(f"Results saved to: {run_dir}")

    return model, run_dir


def main():
    """Avvia il training con configurazione di default o fine-tuning da modello esistente."""
    import os
    os.chdir(PROJECT_ROOT)  # Assicura CWD = root del progetto

    config_path = str(PROJECT_ROOT / "utils" / "config.yaml")
    config = load_config(config_path)
    output_dir = str(PROJECT_ROOT / "trained_models")
    data_dir = str(PROJECT_ROOT / "data")  # Carica da tutte le versioni (V0, V2, V3 tramite filtro)

    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Cartella dati NC non trovata: {data_dir}\n"
            f"Scarica i file .nc di simulazione MIKE21 nella cartella 'data/'"
        )

    # Optional resume path da config per fine-tuning
    resume_from = config.get('training', {}).get('resume_from')
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = (PROJECT_ROOT / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Modello per resume non trovato: {resume_path}")
        resume_from = str(resume_path)
        print(f"Fine-tuning from checkpoint: {resume_from}\n")
    else:
        resume_from = None
        print(f"Training from scratch on all 80% training set (V0+V1+V2+V3)\n")
    
    train(
        config_path=config_path,
        output_dir=output_dir,
        n_envs=1,
        seed=42,
        data_dir=data_dir,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main()