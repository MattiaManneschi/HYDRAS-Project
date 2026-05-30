"""
HYDRAS Source Seeking - PPO Training Script
Addestramento di un agente singolo per il source seeking
utilizzando Proximal Policy Optimization (PPO).
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Tuple
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
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
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

from utils.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig, ScenarioWeightWrapper
from utils.data_loader import ConcentrationField, DataManager, WindData, CurrentData


class SourceSeekingCallback(BaseCallback):
    """
    Callback per logging di success rate durante il training.
    Raccoglie loss e success_rate per i plot finali.
    """

    def __init__(self, reward_mode: str = "full", verbose: int = 0):
        super().__init__(verbose)
        self.reward_mode = reward_mode
        self.success_rate = []
        self._loss_steps = []
        self._loss_values = []
        self._sr_steps = []
        self._sr_values = []

    def _on_training_start(self) -> None:
        _mode_id = {"full": 0, "base": 1, "base_no_wind_reward": 2}.get(self.reward_mode, -1)
        self.logger.record("training/reward_mode_id", _mode_id)
        print(f"[SourceSeekingCallback] reward_mode = {self.reward_mode} (id={_mode_id})")

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


class ScenarioWeightSchedulerCallback(BaseCallback):
    """
    Curriculum learning sui pesi degli scenari.

    Fasi (definite in config.yaml → training.scenario_curriculum):
      - alpha=0.0  → uniform (scenario_weights=None, chunk fisso per env)
      - alpha=0.5  → 50% hard (V1/V2 con chunk 1+2), 50% easy (resto)
      - alpha=0.8  → 80% hard, 20% easy

    Hard scenarios: V1_1, V1_2, V2_1, V2_2
    """

    _HARD = {'V1_1', 'V1_2', 'V2_1', 'V2_2'}
    _ALL  = [f'V{v}_{c}' for v in range(4) for c in range(3)]

    def __init__(self, phases: List[Dict[str, Any]], vec_env, verbose: int = 1):
        super().__init__(verbose)
        self.phases = phases          # [{'end': int, 'alpha': float}, ...]
        self.vec_env = vec_env
        self._current_phase_idx = -1

    def _compute_weights(self, alpha: float) -> Optional[Dict[str, float]]:
        if alpha == 0.0:
            return None  # nessun weighting → campionamento uniforme di default
        n_hard = len(self._HARD)
        n_easy = len(self._ALL) - n_hard
        return {
            k: alpha / n_hard if k in self._HARD else (1.0 - alpha) / n_easy
            for k in self._ALL
        }

    def _get_phase(self) -> Tuple[int, float]:
        for i, p in enumerate(self.phases):
            if self.num_timesteps < p['end']:
                return i, p['alpha']
        return len(self.phases) - 1, self.phases[-1]['alpha']

    def _on_step(self) -> bool:
        phase_idx, alpha = self._get_phase()
        if phase_idx == self._current_phase_idx:
            return True
        self._current_phase_idx = phase_idx
        weights = self._compute_weights(alpha)
        try:
            self.vec_env.env_method('set_scenario_weights', weights)
            if self.verbose:
                label = 'uniform' if alpha == 0.0 else f'{int(alpha*100)}/{int((1-alpha)*100)} hard/easy'
                print(f"\n[Step {self.num_timesteps:,}] Scenario weights → phase {phase_idx+1}: {label}")
        except Exception as e:
            print(f"\n[WARNING] set_scenario_weights failed: {e}")
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
        self._next_eval_step = eval_freq

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

    def _build_eval_env_parallel(self, chunk_id: int, field, n_envs: int):
        import copy as _copy

        def _make_single(f):
            env_kwargs = SourceSeekingConfig.from_config(self.config, chunk_id=chunk_id)
            raw = SourceSeekingEnv(
                config=env_kwargs,
                concentration_field=f,
                data_manager=self.data_manager,
                wind_mapping=self.wind_mapping,
                current_mapping=self.current_mapping,
            )
            raw = Monitor(raw)
            if MASKABLE_PPO_AVAILABLE:
                raw = ActionMasker(raw, mask_fn)
            return raw

        env_fns = [lambda f=_copy.deepcopy(field): _make_single(f) for _ in range(n_envs)]

        if n_envs > 1:
            vec_env = SubprocVecEnv(env_fns, start_method='fork')
        else:
            vec_env = DummyVecEnv(env_fns)

        if isinstance(self.train_vec_env, VecNormalize):
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, training=False)
            sync_envs_normalization(self.train_vec_env, vec_env)
        return vec_env

    def _eval_scenario(self, scenario: Dict[str, Any]) -> float:
        key = f"{scenario['source']}_{scenario['version']}"
        field = self._fields.get(key)
        if field is None:
            return 0.0

        n = self.n_eval_episodes
        vec_env = self._build_eval_env_parallel(scenario['chunk_id'], field, n_envs=n)

        obs = vec_env.reset()
        episode_done = [False] * n
        episode_results = []

        while not all(episode_done):
            if MASKABLE_PPO_AVAILABLE:
                masks = np.array(vec_env.env_method('action_masks'))
                action, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
            else:
                action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and not episode_done[i]:
                    episode_results.append(bool(info.get('source_reached', False)))
                    episode_done[i] = True

        vec_env.close()
        return sum(episode_results) / n

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval_step:
            return True
        self._next_eval_step += self.eval_freq

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
    env_kwargs = SourceSeekingConfig.from_config(config, chunk_id=chunk_id)

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
    allowed_sources: Optional[List[str]] = None,
    scenario_weights: Optional[Dict[str, float]] = None,
) -> Callable[[], gym.Env]:
    """Factory function per la creazione di ambienti paralleli."""
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

        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env

        if allowed_sources is not None:
            inner.allowed_sources = list(allowed_sources)

        if scenario_weights is not None:
            inner.scenario_weights = scenario_weights

        env.reset(seed=seed + rank)

        if use_action_masking and MASKABLE_PPO_AVAILABLE:
            env = ActionMasker(env, mask_fn)

        # Wrapper esterno per permettere env_method('set_scenario_weights') via SubprocVecEnv
        env = ScenarioWeightWrapper(env)

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

    reward_mode = config.get('environment', {}).get('reward', {}).get('reward_mode', 'full')

    print(f"=" * 60)
    print(f"HYDRAS Source Seeking - PPO Training")
    print(f"=" * 60)
    print(f"Run name: {run_name}")
    print(f"Reward mode: {reward_mode}")
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

    # Crea environments.
    # Se targeted_versions è definito: 80% env su versioni target (V2, chunk 1+2),
    # 20% env mixed (tutte versioni, chunk 0) per anti-forgetting.
    scenario_weights    = training_config.get('scenario_weights', None)
    targeted_versions   = training_config.get('targeted_versions', [])
    targeted_chunk_ids  = training_config.get('targeted_chunk_ids', chunk_ids)
    n_targeted_per_chunk = int(training_config.get('n_targeted_per_chunk', n_envs))

    if targeted_versions:
        import copy as _copy
        # DataManager con solo file delle versioni target
        dm_targeted = _copy.copy(data_manager)
        dm_targeted._nc_files = [
            f for f in data_manager._nc_files
            if any(f'_{v}_' in f.name for v in targeted_versions)
        ]
        n_targeted_files = len(dm_targeted._nc_files)
        print(f"\nTargeted fine-tuning: versioni {targeted_versions}")
        print(f"  File target: {n_targeted_files} / {len(data_manager._nc_files)} totali")
        print(f"  Chunk target: {targeted_chunk_ids}")

        # Env targetizzati (80%): n_targeted_per_chunk per ogni chunk V2
        env_fns_targeted = [
            make_env_fn(
                config, concentration_field, wind_data, current_data,
                rank, chunk_id, seed, data_dir, randomize_field,
                data_manager=dm_targeted,
                wind_mapping=wind_mapping,
                current_mapping=current_mapping,
                allowed_sources=discovered_sources,
            )
            for chunk_id in targeted_chunk_ids
            for rank in range(n_targeted_per_chunk)
        ]

        # Env misto (20%): 1 env per chunk 0 con tutte le versioni
        env_fns_mixed = [
            make_env_fn(
                config, concentration_field, wind_data, current_data,
                rank, 0, seed + 500, data_dir, randomize_field,
                data_manager=data_manager,
                wind_mapping=wind_mapping,
                current_mapping=current_mapping,
                allowed_sources=discovered_sources,
            )
            for rank in range(n_envs)
        ]

        env_fns = env_fns_targeted + env_fns_mixed
        n_targeted = len(env_fns_targeted)
        n_mixed    = len(env_fns_mixed)
        print(f"  Env targeted: {n_targeted} | Env mixed: {n_mixed} "
              f"| Ratio: {n_targeted/(n_targeted+n_mixed):.0%}/{n_mixed/(n_targeted+n_mixed):.0%}")
    else:
        # Training standard: tutti i chunk, tutte le versioni
        # Se scenario_weights è definito, il campionamento dinamico è delegato all'env
        if scenario_weights:
            print(f"\nScenario weights attivi — campionamento dinamico (version × chunk) per episodio")
        env_fns = [
            make_env_fn(
                config, concentration_field, wind_data, current_data, i, chunk_id,
                seed, data_dir, randomize_field,
                data_manager=data_manager,
                wind_mapping=wind_mapping,
                current_mapping=current_mapping,
                allowed_sources=discovered_sources,
                scenario_weights=scenario_weights,
            )
            for i in range(n_envs) for chunk_id in chunk_ids
        ]

    n_parallel_envs = len(env_fns)

    # SubprocVecEnv per training parallelo su più core (curriculum disabilitato → nessun accesso diretto agli env)
    # Fallback a DummyVecEnv se un solo env (SubprocVecEnv con 1 processo non ha senso)
    if len(env_fns) > 1:
        base_vec_env = SubprocVecEnv(env_fns, start_method='fork')
    else:
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

    # Learning rate schedule: step-wise decay allineato alle fasi del curriculum
    lr_schedule_config = training_config.get('lr_schedule', [])
    if lr_schedule_config:
        # Ogni entry ha {'end': timestep, 'lr': valore}
        lr_steps = [(s['end'] / timesteps, float(s['lr'])) for s in lr_schedule_config]

        def _make_lr_fn(steps):
            def lr_fn(progress_remaining: float) -> float:
                progress = 1.0 - progress_remaining
                for frac, lr in steps:
                    if progress <= frac:
                        return lr
                return steps[-1][1]
            return lr_fn

        learning_rate = _make_lr_fn(lr_steps)
        print(f"\nLearning rate schedule ({len(lr_steps)} fasi):")
        for s in lr_schedule_config:
            print(f"  0 – {s['end']:,} steps → lr={s['lr']}")
    else:
        learning_rate = training_config.get('learning_rate', 3e-4)
        print(f"\nLearning rate: {learning_rate} (costante)")

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
        'learning_rate': learning_rate,
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

    # Scenario weight scheduler (curriculum hard/easy)
    scenario_curriculum = training_config.get('scenario_curriculum', [])
    if scenario_curriculum:
        weight_scheduler = ScenarioWeightSchedulerCallback(
            phases=scenario_curriculum,
            vec_env=vec_env,
            verbose=1,
        )
        callbacks.append(weight_scheduler)

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
    eval_freq = training_config.get('eval_freq', 100000)
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
    custom_callback = SourceSeekingCallback(reward_mode=reward_mode, verbose=1)
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


def find_latest_model(output_dir: str) -> Optional[str]:
    """Trova automaticamente il final_model.zip più recente in output_dir."""
    candidates = sorted(Path(output_dir).glob("ppo_*/models/final_model.zip"))
    return str(candidates[-1]) if candidates else None


def main():
    """Avvia il training con fine-tuning automatico dall'ultimo modello disponibile."""
    import os
    os.chdir(PROJECT_ROOT)  # Assicura CWD = root del progetto

    config_path = str(PROJECT_ROOT / "utils" / "config.yaml")
    config = load_config(config_path)
    output_dir = str(PROJECT_ROOT / "trained_models")
    data_dir = str(PROJECT_ROOT / "data")

    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Cartella dati NC non trovata: {data_dir}\n"
            f"Scarica i file .nc di simulazione MIKE21 nella cartella 'data/'"
        )

    # Priorità:
    # 1) resume_from: <path>  nel config  → fine-tune da quel checkpoint
    # 2) resume_from: null    nel config  → training da zero (nessun auto-detect)
    # 3) chiave assente                   → auto-detect ultimo modello disponibile
    training_cfg = config.get('training', {})
    if 'resume_from' in training_cfg:
        resume_from = training_cfg['resume_from']
        if resume_from:
            resume_path = Path(resume_from)
            if not resume_path.is_absolute():
                resume_path = (PROJECT_ROOT / resume_path).resolve()
            if not resume_path.exists():
                raise FileNotFoundError(f"Modello per resume non trovato: {resume_path}")
            resume_from = str(resume_path)
            print(f"Fine-tuning from checkpoint (config): {resume_from}\n")
        else:
            resume_from = None
            print("Training from scratch (resume_from: null in config)\n")
    else:
        resume_from = find_latest_model(output_dir)
        if resume_from:
            print(f"Fine-tuning from latest model (auto-detected): {resume_from}\n")
        else:
            print("No existing model found — training from scratch\n")

    sweep = training_cfg.get('sensor_range_sweep', [])

    if sweep:
        base_resume = resume_from
        print(f"Sensor range sweep: {sweep} m")
        print(f"Base model (fisso per tutti gli step): {base_resume}\n")
        for sr in sweep:
            print(f"\n{'='*60}")
            print(f"  Fine-tuning sensor_range = {sr}m")
            print(f"  Resume from: {base_resume}")
            print(f"{'='*60}\n")
            cfg = load_config(config_path)
            cfg['agent']['sensor_range'] = sr
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
            _, run_dir = train(
                config_path=config_path,
                output_dir=output_dir,
                n_envs=2,
                seed=42,
                data_dir=data_dir,
                resume_from=base_resume,
            )
            print(f"\n  → sr={sr}m completato. Modello in: {run_dir / 'models' / 'final_model.zip'}")
        print(f"\nSweep completato. Modelli in: {output_dir}")
    else:
        train(
            config_path=config_path,
            output_dir=output_dir,
            n_envs=2,
            seed=42,
            data_dir=data_dir,
            resume_from=resume_from,
        )

if __name__ == "__main__":
    main()