#!/usr/bin/env python3
"""
HYDRAS Source Seeking - Inference Script
Valutazione completa del modello addestrato su tutti gli scenari.

Output:
  - Success rate per sorgente e scenario
  - Distanza finale dalla sorgente (media, min, max)
  - Numero di step per raggiungere la sorgente (episodi di successo)
  - Motivo di terminazione (successo / timeout)
  - Traiettorie visualizzate per ogni episodio
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_generator import generate_showcase_videos

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    MASKABLE_PPO_AVAILABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    ActionMasker = None  # Fallback per ActionMasker
    MASKABLE_PPO_AVAILABLE = False
    print("WARNING: sb3_contrib non disponibile, uso PPO standard")

from utils.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig
from utils.data_loader import DataManager


def plot_trajectory(trajectory: np.ndarray, field, ax=None, title: str = "",
                    show_arrows: bool = True, arrow_freq: int = 10):
    """Plot della traiettoria su campo di concentrazione."""
    from matplotlib.colors import ListedColormap
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Usa il frame corrente del field (impostato dall'env)
    conc = field.get_current_field()  # [y, x]
    extent = [field.x_coords.min(), field.x_coords.max(), field.y_coords.min(), field.y_coords.max()]
    
    # 1) Sfondo: azzurro uniforme su tutto il plot
    ax.set_facecolor('#87CEEB')  # Azzurro cielo
    
    # 2) Terra bianca
    if field.land_mask is not None:
        white_cmap = ListedColormap(['#FFFFFF'])  # Bianco puro
        land_display = np.ma.masked_where(~field.land_mask, np.ones_like(conc))
        ax.imshow(land_display, origin='lower', extent=extent, cmap=white_cmap, alpha=1.0, zorder=1)
    
    # 3) Plume di concentrazione (mascherato su terra e dove conc ~ 0)
    plume_threshold = 0.01  # Mostra solo dove c'è concentrazione significativa
    if field.land_mask is not None:
        mask = field.land_mask | (conc < plume_threshold)
    else:
        mask = conc < plume_threshold
    conc_masked = np.ma.masked_where(mask, conc)
    im = ax.imshow(conc_masked, origin='lower', extent=extent, cmap='YlOrRd', alpha=0.9, 
                   vmin=0, vmax=max(conc.max(), 0.1), zorder=2)
    plt.colorbar(im, ax=ax, label='Concentrazione')
    
    # Traiettoria
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1.5, alpha=0.8, label='Traiettoria')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='darkred', s=100, marker='X', zorder=5, label='End')
    
    # Frecce direzione
    if show_arrows and len(trajectory) > arrow_freq:
        for i in range(0, len(trajectory) - 1, arrow_freq):
            dx = trajectory[i + 1, 0] - trajectory[i, 0]
            dy = trajectory[i + 1, 1] - trajectory[i, 1]
            if np.sqrt(dx**2 + dy**2) > 1:
                ax.arrow(trajectory[i, 0], trajectory[i, 1], dx * 0.8, dy * 0.8,
                         head_width=20, head_length=10, fc='red', ec='red', alpha=0.6)
    
    # Sorgente
    if field.source_position is not None:
        ax.scatter(field.source_position[0], field.source_position[1], c='yellow', s=200, 
                   marker='*', edgecolors='black', zorder=6, label='Source')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    return ax


# ─── Dataclasses per i risultati ────────────────────────────────────────────

@dataclass
class EpisodeResult:
    scenario: str          # es. "SRC001"
    source_id: str         # "SRC001" ... "SRC132"
    episode: int
    success: bool
    termination: str       # "success" / "timeout" / "boundary" / "land"
    initial_distance: float  # m - distanza spawn-sorgente
    final_distance: float  # m
    steps: int
    trajectory: np.ndarray
    start_frame: int = 0   # Frame iniziale (dipende da chunk_id)
    end_frame: int = 0     # Frame finale (al quale plottare)


@dataclass
class ScenarioStats:
    scenario: str          # es. "SRC001_Q1/4"
    source_id: str         # es. "SRC001"
    n_episodes: int
    success_rate: float
    mean_final_dist: float
    min_final_dist: float
    max_final_dist: float
    mean_steps_success: Optional[float]   # None se nessun successo
    mean_initial_dist: float               # distanza media di partenza dalla sorgente
    termination_counts: Dict[str, int]


# ─── Utility ────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def write_inference_log(
    output_path: Path,
    model_path: str,
    dt_seconds: float,
    global_success_rate: Optional[float],
    version_success_rates: Dict[str, Optional[float]],
    chunk_success_rates: Dict[str, Optional[float]],
    mean_initial_distance: Optional[float],
    mean_success_steps: Optional[float],
    total_scenarios: int,
    total_episodes: int,
) -> Path:
    """Scrive un log di sintesi in output_path/log.txt."""

    def fmt_percent(value: Optional[float]) -> str:
        return "n/d" if value is None else f"{value * 100:.1f}%"

    def fmt_distance(value: Optional[float]) -> str:
        return "n/d" if value is None else f"{value:.1f} m"

    def fmt_steps(value: Optional[float]) -> str:
        return "n/d" if value is None else f"{value:.1f}"

    mean_success_minutes = None
    if mean_success_steps is not None:
        mean_success_minutes = (mean_success_steps * dt_seconds) / 60.0

    lines = [
        "HYDRAS Inference Summary",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {model_path}",
        "",
        "Metrics:",
        f"- Global Success Rate: {fmt_percent(global_success_rate)}",
        f"- Mean Initial Distance: {fmt_distance(mean_initial_distance)}",
        (
            f"- Mean Success Steps: {fmt_steps(mean_success_steps)} "
            f"(~{'n/d' if mean_success_minutes is None else f'{mean_success_minutes:.1f} min'})"
        ),
        "",
        "Success Rate by Wind:",
        f"- V0: {fmt_percent(version_success_rates.get('V0'))}",
        f"- V1: {fmt_percent(version_success_rates.get('V1'))}",
        f"- V2: {fmt_percent(version_success_rates.get('V2'))}",
        f"- V3: {fmt_percent(version_success_rates.get('V3'))}",
        "",
        "Success Rate by Frame:",
        f"- Q1/4: {fmt_percent(chunk_success_rates.get('Q1/4'))}",
        f"- Q1/2: {fmt_percent(chunk_success_rates.get('Q1/2'))}",
        f"- Q3/4: {fmt_percent(chunk_success_rates.get('Q3/4'))}",
        "",
        f"Total Scenarios Evaluated: {total_scenarios}",
        f"Total Episodes Evaluated: {total_episodes}",
    ]

    log_path = output_path / "log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    return log_path





def load_model(model_path: str):
    """Carica MaskablePPO o PPO dal path."""
    try:
        model = MaskablePPO.load(model_path, device='cpu')
        print(f"  Modello caricato: {model_path}")
        print(f"  Device: cpu (forzato per CUDA compatibility)")
    except Exception as e:
        raise RuntimeError(f"Impossibile caricare il modello: {e}")
    return model


def make_env_config(config: dict, chunk_id: int = 0) -> SourceSeekingConfig:
    env_cfg = config.get('environment', {})
    agent_cfg = config.get('agent', {})
    domain_cfg = config.get('domain', {})
    reward_cfg = env_cfg.get('reward', {})
    spawn_cfg = env_cfg.get('spawn', {})

    return SourceSeekingConfig(
        xmin=domain_cfg.get('xmin', 619000),
        xmax=domain_cfg.get('xmax', 622000),
        ymin=domain_cfg.get('ymin', 4794500),
        ymax=domain_cfg.get('ymax', 4797000),
        resolution=domain_cfg.get('grid_resolution', 10),
        max_velocity=agent_cfg.get('max_velocity', 1.0),
        memory_length=agent_cfg.get('memory_length', 9),
        dt=env_cfg.get('dt', 10),
        max_steps=env_cfg.get('max_episode_steps', 1080),
        source_distance_threshold=reward_cfg.get('distance_threshold', 50),
        source_found_reward=reward_cfg.get('source_reached_bonus', 100),
        step_penalty=reward_cfg.get('step_penalty', -0.1),
        boundary_penalty=reward_cfg.get('boundary_penalty', -10),
        distance_reward_multiplier=reward_cfg.get('distance_reward_multiplier', 1.0),
        land_proximity_threshold=reward_cfg.get('land_proximity_threshold', 10.0),
        land_proximity_penalty_max=reward_cfg.get('land_proximity_penalty_max', -5.0),
        n_discrete_actions=agent_cfg.get('n_discrete_actions', 8),
        spawn_min_land_distance=spawn_cfg.get('min_land_distance', 50.0),
        spawn_start_frame=spawn_cfg.get('start_frame', 1440),
        spawn_conc_threshold=spawn_cfg.get('conc_threshold', 0.5),
        chunk_id=chunk_id,
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


def mask_fn(env) -> np.ndarray:
    """Estrae la maschera azioni attraversando i wrapper."""
    inner = env
    while hasattr(inner, 'env'):
        inner = inner.env
    return inner.action_masks()


def build_env(env_cfg, field, vec_norm_path, use_masking,
              data_manager: Optional[DataManager] = None,
              wind_data = None,
              current_data = None,
              wind_mapping: Optional[Dict[str, str]] = None,
              current_mapping: Optional[Dict[str, str]] = None):
    """Costruisce e wrappa l'environment per l'inferenza.
    
    Args:
        data_manager: DataManager per accesso ai dati
        wind_data: Dati di vento (caricati da DataManager)
        current_data: Dati di corrente (caricati da DataManager)
        wind_mapping: Mapping versione -> wind file (es. {'_V0': '...', '_V1': '...', ...})
                     Se passato, l'env caricherà il vento dinamicamente per versione.
        current_mapping: Mapping versione -> current file (es. {'_V0': '...', '_V1': '...', ...})
                        Se passato, l'env caricherà la corrente dinamicamente per versione.
    """
    raw_env = SourceSeekingEnv(
        config=env_cfg,
        concentration_field=field,
        wind_data=wind_data,
        current_data=current_data,
        data_manager=data_manager,
        wind_mapping=wind_mapping,
        current_mapping=current_mapping,
    )

    if use_masking and MASKABLE_PPO_AVAILABLE and ActionMasker is not None:
        raw_env = ActionMasker(raw_env, mask_fn)

    vec_env = DummyVecEnv([lambda e=raw_env: e])

    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    return vec_env


def get_inner_env(vec_env) -> SourceSeekingEnv:
    """Estrae il SourceSeekingEnv dal fondo dello stack di wrapper."""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env


def run_episode(model, vec_env, deterministic=True) -> EpisodeResult:
    """Esegue un singolo episodio e ritorna il risultato."""
    obs = vec_env.reset()
    inner = get_inner_env(vec_env)

    # Leggi il frame iniziale dall'environment (salvato durante reset)
    start_frame = inner.info_reset.get('start_time_idx', 0) if hasattr(inner, 'info_reset') else 0

    # Salva posizione iniziale e calcola distanza dalla sorgente
    spawn_pos = inner.state.position.copy()
    source_pos = inner.source_position
    initial_dist = float(np.linalg.norm(spawn_pos - source_pos))

    trajectory = [spawn_pos]
    done = False
    last_info = {}

    while not done:
        if MASKABLE_PPO_AVAILABLE:
            action, _ = model.predict(obs, deterministic=deterministic,
                                      action_masks=vec_env.env_method('action_masks')[0])
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, _, dones, infos = vec_env.step(action)
        done = dones[0]
        last_info = infos[0]
        
        # Usa posizione dall'info (inner è già resettato quando done=True)
        pos = last_info.get('position', inner.state.position.tolist())
        trajectory.append(np.array(pos))

    # Determina terminazione: usa la reason esplicita dall'ambiente quando disponibile.
    termination = last_info.get('termination_reason')
    if termination not in {'success', 'boundary', 'land', 'timeout'}:
        if last_info.get('source_reached', False):
            termination = 'success'
        elif last_info.get('out_of_bounds', False):
            termination = 'boundary'
        elif last_info.get('on_land', False):
            termination = 'land'
        else:
            n_steps_info = int(last_info.get('steps', 0))
            max_steps = int(getattr(inner.config, 'max_steps', 0)) if hasattr(inner, 'config') else 0
            termination = 'timeout' if max_steps > 0 and n_steps_info >= max_steps else 'timeout'

    # Usa i valori dall'info dict
    final_dist = last_info.get('distance_to_source', 0.0)
    n_steps = last_info.get('steps', len(trajectory) - 1)
    
    # Leggi il frame finale dall'info dict dell'ultimo step (prima del reset automatico)
    end_frame = last_info.get('end_time_idx', start_frame)

    return EpisodeResult(
        scenario="",           # impostato dal chiamante
        source_id="",          # impostato dal chiamante
        episode=0,             # impostato dal chiamante
        success=termination == 'success',
        termination=termination,
        initial_distance=initial_dist,
        final_distance=final_dist,
        steps=n_steps,
        trajectory=np.array(trajectory),
        start_frame=start_frame,
        end_frame=end_frame
    )


# ─── Funzioni di analisi ─────────────────────────────────────────────────────

def compute_scenario_stats(results: List[EpisodeResult], scenario: str, source_id: str) -> ScenarioStats:
    n = len(results)
    successes = [r for r in results if r.success]
    final_dists = [r.final_distance for r in results]
    initial_dists = [r.initial_distance for r in results]
    termination_counts = {}
    for r in results:
        termination_counts[r.termination] = termination_counts.get(r.termination, 0) + 1

    return ScenarioStats(
        scenario=scenario,
        source_id=source_id,
        n_episodes=n,
        success_rate=len(successes) / n,
        mean_final_dist=float(np.mean(final_dists)),
        min_final_dist=float(np.min(final_dists)),
        max_final_dist=float(np.max(final_dists)),
        mean_steps_success=float(np.mean([r.steps for r in successes])) if successes else None,
        mean_initial_dist=float(np.mean(initial_dists)),
        termination_counts=termination_counts,
    )


def print_scenario_stats(stats: ScenarioStats):
    sr = f"{stats.success_rate * 100:.0f}%"
    dist = f"{stats.mean_final_dist:.0f}m (min={stats.min_final_dist:.0f}m, max={stats.max_final_dist:.0f}m)"
    steps = f"{stats.mean_steps_success:.0f}" if stats.mean_steps_success else "—"
    term = ", ".join(f"{k}={v}" for k, v in stats.termination_counts.items())
    print(f"  {stats.scenario:8s}  SR={sr:5s}  dist={dist}  steps_success={steps}  [{term}]")


def print_source_summary(all_stats: List[ScenarioStats], source_id: str):
    source_stats = [s for s in all_stats if s.source_id == source_id]
    if not source_stats:
        return
    mean_sr = np.mean([s.success_rate for s in source_stats])
    mean_dist = np.mean([s.mean_final_dist for s in source_stats])
    success_steps = [s.mean_steps_success for s in source_stats if s.mean_steps_success]
    mean_steps = np.mean(success_steps) if success_steps else None

    print(f"\n  {source_id} SUMMARY: SR={mean_sr*100:.0f}%  "
          f"mean_dist={mean_dist:.0f}m  "
          f"mean_steps_success={'—' if mean_steps is None else f'{mean_steps:.0f}'}")


# ─── Plotting ────────────────────────────────────────────────────────────────

def save_trajectory_plot(result: EpisodeResult, field, output_path: Path, threshold: float = 100):
    """Salva il plot della traiettoria per un singolo episodio.
    
    Mostra il campo di concentrazione al frame finale dell'episodio.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Usa il frame finale salvato dall'episodio
    display_frame = result.end_frame
    
    # Aggiorna il field al frame appropriato
    if hasattr(field, 'set_time'):
        field.set_time(display_frame)

    status = "SUCCESS ✓" if result.success else f"FAILED [{result.termination}]"
    title = (f"{result.scenario} — Ep {result.episode+1} — {status}\n"
             f"dist={result.final_distance:.0f}m  steps={result.steps}  (frame={display_frame})")

    plot_trajectory(result.trajectory, field, ax=ax, title=title,
                    show_arrows=True, arrow_freq=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def run_inference(
    model_path: str,
    config_path: str,
    data_dir: str,
    output_dir: str,
    n_episodes: int = 5,
    deterministic: bool = True,
    sources_csv: str = "Coordinate_Sorgenti_FaseII.csv",
    chunk_ids: List[int] = None,
    save_videos: bool = True,
):
    """
    Esegue l'inferenza completa su 26 sorgenti held-out (SRC107-SRC132, 20% del totale 132) con chunk multipli per fonte.

    Args:
        model_path:   Path al modello (.zip)
        config_path:  Path al config YAML
        data_dir:     Directory con i file NC (Output_HD_FaseII_CL2_V1)
        output_dir:   Directory di output per plot e risultati
        n_episodes:   Episodi per sorgente e chunk
        deterministic: Policy deterministica o stocastica
        sources_csv:  File CSV con coordinate delle sorgenti
        chunk_ids:    Lista di chunk_id da testare (default [0, 1, 2] = Q1/4, Q1/2, Q3/4)
                     0 = spawn @1/4, 1 = spawn @1/2, 2 = spawn @3/4
    """
    if chunk_ids is None:
        chunk_ids = [0, 1, 2]  # Default: Q1/4, Q1/2, Q3/4
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Carica config e modello
    config = load_config(config_path)
    model = load_model(model_path)
    env_cfg = make_env_config(config)
    success_threshold = config.get('environment', {}).get('reward', {}).get('distance_threshold', 50)

    vec_norm_path = Path(model_path).parent / "vec_normalize.pkl"
    
    # Inizializza DataManager con auto-discovery di 132 sorgenti
    data_manager = DataManager(
        data_dir=data_dir,
        preload_all=False,
        sources_csv=sources_csv
    )
    
    print(f"\nInference data: {len(data_manager._nc_files)} files (tutte le versioni V0+V1+V2+V3)")
    
    # Usa le sorgenti escluse dal training (SRC107-SRC132) per valutazione
    all_sources = data_manager.get_discovered_sources()
    inference_sources = [s for s in all_sources if int(s[3:]) > 106]  # SRC107-SRC132 (26 file, ~20%)
    
    chunk_labels = {0: "Q1/4", 1: "Q1/2", 2: "Q3/4"}
    chunk_descriptions = ", ".join([
        f"{chunk_labels[cid]} (chunk_id={cid})" for cid in chunk_ids
    ])
    
    print(f"\n{'='*100}")
    print(f"HYDRAS Inference — {len(inference_sources)} sorgenti × 4 scenari vento × {len(chunk_ids)} chunk × {n_episodes} episodi")
    print(f"  = {len(inference_sources)*4*len(chunk_ids)*n_episodes} episodi totali")
    print(f"Chunk testati: {chunk_descriptions}")
    print(f"Modello: {model_path}")
    print(f"Dati: {data_dir}")
    print(f"Sorgenti training: SRC001-SRC106 (80%)")
    print(f"Sorgenti inference (20%): SRC107-SRC132 ({len(inference_sources)} sorgenti)")
    print(f"{'='*100}\n")
    
    # Wind mapping per caricamento dinamico vento per versione
    # (come nel training, per coerenza tra Conc_Vx e Wind_Vx)
    wind_mapping = {
        "_V0": "CI_WIND_faseII_V0.txt",
        "_V1": "CI_WIND_faseII_V1.txt",
        "_V2": "CI_WIND_faseII_V2.txt",
        "_V3": "CI_WIND_faseII_V3.txt",
    }
    print(f"Wind mapping (versions V0-V3): {len(wind_mapping)} file")
    
    # Current mapping per caricamento dinamico corrente per versione
    # (come nel training, per coerenza tra Conc_Vx e Current_Vx)
    current_mapping = {
        "_V0": "CL02_V0_SRC000_U_V_10mGrid.nc",
        "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
        "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc",
        "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc",
    }
    print(f"Current mapping (versions V0-V3): {len(current_mapping)} file")
    
    # NON precarichiamo vento/corrente - l'environment li caricherà dinamicamente
    # durante reset() in base alla versione del file NC
    wind_data = None
    current_data = None
    
    # Get dt from config
    dt_seconds = config.get('environment', {}).get('dt', 10)
    
    all_stats: List[ScenarioStats] = []

    # Accumulatori episodio-level (più robusti delle medie per scenario)
    episode_success_all: List[float] = []
    episode_success_by_version: Dict[str, List[float]] = {'V0': [], 'V1': [], 'V2': [], 'V3': []}
    episode_success_by_chunk: Dict[str, List[float]] = {'Q1/4': [], 'Q1/2': [], 'Q3/4': []}
    initial_distances_all: List[float] = []
    success_steps_all: List[float] = []

    # Accumulatore globale per selezione video showcase (usato a fine run)
    all_results_global: List[EpisodeResult] = []

    # Dati per-episodio (analisi quantitativa)
    episodes_data: List[dict] = []

    for src_idx, source_id in enumerate(inference_sources, 1):
        if src_idx % 5 == 0:
            print(f"\n[Progress: {src_idx}/{len(inference_sources)} sources]\n")
            
        source_dir = output_path / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Itera su tutte le 4 versioni
        for version in ['V0', 'V1', 'V2', 'V3']:
            # Filtra file per questa sorgente e versione
            version_files = [f for f in data_manager._nc_files 
                           if version in f.name and source_id in f.name]
            if not version_files:
                continue  # Sorgente non disponibile in questa versione
            
            version_file = version_files[0]  # Una sola per versione+sorgente
            version_dir = source_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            for chunk_id in chunk_ids:  # Itera su chunk_ids specificati
                chunk_label = chunk_labels[chunk_id]
                scenario_label = f"{version}_{source_id}_{chunk_label}"
                
                episode_results: List[EpisodeResult] = []
                
                # CARICA IL FIELD UNA SOLA VOLTA per tutti gli episodi di questo scenario
                field = None
                try:
                    field = data_manager._nc_loader.load(
                        str(version_file),
                        concentration_var="Concentration - component 1"
                    )
                    if field is None:
                        print(f"\n  [SKIP] Could not load field for {version}_{source_id}")
                        continue
                    
                    # Imposta source_position dalle coordinate CSV
                    coords = data_manager.get_source_coordinates(source_id)
                    if coords:
                        field.source_position = coords
                    
                    # Imposta run_id con versione per caricamento dinamico vento durante reset()
                    field.run_id = f"{source_id}_{version}"
                    
                except Exception as e:
                    print(f"\n  [SKIP] Error loading field for {version}_{source_id}: {e}")
                    continue

                for ep in range(n_episodes):
                    # Crea env config con chunk_id appropriato
                    env_cfg_ep = make_env_config(config, chunk_id=chunk_id)

                    vec_env = build_env(env_cfg_ep, field, vec_norm_path,
                                       use_masking=MASKABLE_PPO_AVAILABLE,
                                       data_manager=data_manager,
                                       wind_data=wind_data,
                                       current_data=current_data,
                                       wind_mapping=wind_mapping,
                                       current_mapping=current_mapping)

                    result = run_episode(model, vec_env, deterministic=deterministic)
                    result.scenario = scenario_label
                    result.source_id = source_id
                    result.episode = ep

                    success_value = 1.0 if result.success else 0.0
                    episode_success_all.append(success_value)
                    if version in episode_success_by_version:
                        episode_success_by_version[version].append(success_value)
                    if chunk_label in episode_success_by_chunk:
                        episode_success_by_chunk[chunk_label].append(success_value)
                    initial_distances_all.append(result.initial_distance)
                    if result.success:
                        success_steps_all.append(float(result.steps))

                    episode_results.append(result)
                    all_results_global.append(result)
                    vec_env.close()

                    # Distanza dalla sorgente ad ogni step della traiettoria
                    src_x, src_y = field.source_position
                    dist_history = [
                        float(np.sqrt((pos[0] - src_x)**2 + (pos[1] - src_y)**2))
                        for pos in result.trajectory
                    ]
                    episodes_data.append({
                        "source_id": source_id,
                        "version": version,
                        "chunk": chunk_label,
                        "chunk_id": chunk_id,
                        "episode": ep + 1,
                        "success": result.success,
                        "termination": result.termination,
                        "initial_distance": result.initial_distance,
                        "final_distance": result.final_distance,
                        "steps": result.steps,
                        "distance_history": dist_history,
                    })

                    # Salva plot traiettoria
                    plot_path = version_dir / f"ep{ep+1:02d}_chunk{chunk_id}_trajectory.png"
                    save_trajectory_plot(result, field, plot_path, success_threshold)
                    
                    # Log episodio
                    init_dist = f"{result.initial_distance:.0f}m"
                    if result.success:
                        time_mins = (result.steps * dt_seconds) / 60
                        print(f"  {version}_{source_id}_{chunk_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → SUCCESS in {result.steps:3d} steps ({time_mins:5.1f}m)")
                    else:
                        print(
                            f"  {version}_{source_id}_{chunk_label} Ep{ep+1}: "
                            f"spawn_dist={init_dist:>5s} → {result.termination.upper()} "
                            f"at {result.steps:4d} steps (final_dist={result.final_distance:6.1f}m)"
                        )

                if episode_results:
                    # Statistiche scenario
                    stats = compute_scenario_stats(episode_results, scenario_label, source_id)
                    all_stats.append(stats)

    def safe_mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    global_sr = safe_mean(episode_success_all)
    mean_initial_dist = safe_mean(initial_distances_all)
    mean_success_steps = safe_mean(success_steps_all)

    version_sr = {
        version: safe_mean(episode_success_by_version.get(version, []))
        for version in ['V0', 'V1', 'V2', 'V3']
    }
    chunk_sr = {
        chunk: safe_mean(episode_success_by_chunk.get(chunk, []))
        for chunk in ['Q1/4', 'Q1/2', 'Q3/4']
    }

    # Riepilogo globale
    if all_stats:
        print(f"\n{'='*80}")
        print(f"RESULTS BY CHUNK (Time Instant)")
        print(f"{'='*80}")
        for chunk_label in ['Q1/4', 'Q1/2', 'Q3/4']:
            sr = chunk_sr.get(chunk_label)
            n_eps = len(episode_success_by_chunk.get(chunk_label, []))
            if sr is None:
                print(f"{chunk_label}: (no data)")
            else:
                print(f"{chunk_label}: {sr*100:6.1f}% ({n_eps} episodes)")
        
        print(f"\n{'='*80}")
        print(f"RESULTS BY WIND SCENARIO (across all chunks)")
        print(f"{'='*80}")
        
        for version in ['V0', 'V1', 'V2', 'V3']:
            sr = version_sr.get(version)
            n_eps = len(episode_success_by_version.get(version, []))
            if sr is not None:
                print(f"{version}: {sr*100:6.1f}% ({n_eps} episodes)")
            else:
                print(f"{version}: (no data)")
        
        print(f"\n{'='*80}")
        print(f"Global Success Rate: {('n/d' if global_sr is None else f'{global_sr*100:.1f}%')}")
        print(f"Mean initial distance: {('n/d' if mean_initial_dist is None else f'{mean_initial_dist:.0f}m')}")
        if mean_success_steps is not None:
            mean_success_minutes = (mean_success_steps * dt_seconds) / 60.0
            print(f"Mean success steps: {mean_success_steps:.1f} (~{mean_success_minutes:.1f} min)")
        else:
            print("Mean success steps: n/d (nessun episodio di successo)")
        print(f"Total scenarios: {len(all_stats)}")
        print(f"Total episodes: {len(episode_success_all)}")
        print(f"{'='*80}\n")

    # Scrive sempre il log di riepilogo nello stesso output_dir delle valutazioni
    log_path = write_inference_log(
        output_path=output_path,
        model_path=model_path,
        dt_seconds=dt_seconds,
        global_success_rate=global_sr,
        version_success_rates=version_sr,
        chunk_success_rates=chunk_sr,
        mean_initial_distance=mean_initial_dist,
        mean_success_steps=mean_success_steps,
        total_scenarios=len(all_stats),
        total_episodes=len(episode_success_all),
    )
    print(f"Summary log salvato in: {log_path}")

    # Salva dati per-episodio per analisi quantitativa
    import json
    episodes_json_path = output_path / "episodes_data.json"
    with open(episodes_json_path, "w") as f:
        json.dump(episodes_data, f)
    print(f"Dati per-episodio salvati in: {episodes_json_path}")

    if save_videos:
        generate_showcase_videos(all_results_global, data_manager, output_path)

    return all_stats


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR    = str(PROJECT_ROOT / "data")
    CONFIG_PATH = str(PROJECT_ROOT / "utils" / "config.yaml")
    OUTPUT_DIR  = str(PROJECT_ROOT / "evaluations_v8")

    # Seleziona l'ultimo modello addestrato (directory più recente per nome)
    trained_dir = PROJECT_ROOT / "trained_models"
    run_dirs = sorted([d for d in trained_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_")])

    if not run_dirs:
        print("ERRORE: Nessuna directory di training trovata in trained_models/")
        sys.exit(1)

    latest_run = run_dirs[-1]

    model_path = latest_run / "models" / "final_model.zip"
    if not model_path.exists():
        model_path = latest_run / "models" / "best" / "best_model.zip"

    if not model_path.exists():
        print(f"ERRORE: Nessun modello trovato in {latest_run}/models/")
        sys.exit(1)

    MODEL_PATH = str(model_path)
    print(f"Modello selezionato: {MODEL_PATH}")
    print(f"Output valutazioni: {OUTPUT_DIR}")

    run_inference(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        n_episodes=5,
        deterministic=True,
        sources_csv="Coordinate_Sorgenti_FaseII.csv",
        chunk_ids=[0, 1, 2],
    )


if __name__ == "__main__":
    main()