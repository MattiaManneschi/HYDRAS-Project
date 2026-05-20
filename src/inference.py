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
    # Le coordinate NetCDF sono centri cella; imshow interpreta extent come bordi.
    # Correzione di mezzo passo (half-pixel) per allineare correttamente i pixel
    # ai centri cella: la cornice del plot coincide col dominio MIKE21 e il
    # dato all'ultima riga/colonna non viene tagliato.
    dx = float(field.x_coords[1] - field.x_coords[0]) if len(field.x_coords) > 1 else 10.0
    dy = float(field.y_coords[1] - field.y_coords[0]) if len(field.y_coords) > 1 else 10.0
    extent = [
        float(field.x_coords[0])  - dx / 2, float(field.x_coords[-1]) + dx / 2,
        float(field.y_coords[0])  - dy / 2, float(field.y_coords[-1]) + dy / 2,
    ]
    
    # 1) Sfondo: azzurro uniforme su tutto il plot
    ax.set_facecolor('#87CEEB')
    
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
    # Ripristina i limiti al dominio esatto (i centri cella, non i bordi pixel)
    ax.set_xlim(float(field.x_coords[0]), float(field.x_coords[-1]))
    ax.set_ylim(float(field.y_coords[0]), float(field.y_coords[-1]))

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
    return SourceSeekingConfig.from_config(config, chunk_id=chunk_id)


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
    config_override: Optional[dict] = None,
    save_plots: bool = True,
    seed: Optional[int] = None,
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

    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    config = config_override if config_override is not None else load_config(config_path)
    model = load_model(model_path)
    vec_norm_path = Path(model_path).parent / "vec_normalize.pkl"
    success_threshold = config.get('environment', {}).get('reward', {}).get('distance_threshold', 50)
    
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
    
    dt_seconds = config.get('environment', {}).get('dt', 10)
    
    all_stats: List[ScenarioStats] = []
    episode_success_all: List[float] = []
    episode_success_by_version: Dict[str, List[float]] = {'V0': [], 'V1': [], 'V2': [], 'V3': []}
    episode_success_by_chunk: Dict[str, List[float]] = {'Q1/4': [], 'Q1/2': [], 'Q3/4': []}
    initial_distances_all: List[float] = []
    success_steps_all: List[float] = []
    all_results_global: List[EpisodeResult] = []
    episodes_data: List[dict] = []

    for src_idx, source_id in enumerate(inference_sources, 1):
        if src_idx % 5 == 0:
            print(f"\n[Progress: {src_idx}/{len(inference_sources)} sources]\n")

        for version in ['V0', 'V1', 'V2', 'V3']:
            version_files = [f for f in data_manager._nc_files
                             if version in f.name and source_id in f.name]
            if not version_files:
                continue

            version_dir = output_path / source_id / version
            version_dir.mkdir(parents=True, exist_ok=True)

            try:
                field = data_manager._nc_loader.load(
                    str(version_files[0]),
                    concentration_var="Concentration - component 1"
                )
                if field is None:
                    continue
                coords = data_manager.get_source_coordinates(source_id)
                if coords:
                    field.source_position = coords
                field.run_id = f"{source_id}_{version}"
            except Exception as e:
                print(f"  [SKIP] {version}_{source_id}: {e}")
                continue

            for chunk_id in chunk_ids:
                chunk_label = chunk_labels[chunk_id]
                scenario_label = f"{version}_{source_id}_{chunk_label}"
                episode_results: List[EpisodeResult] = []

                env_cfg = make_env_config(config, chunk_id=chunk_id)
                vec_env = build_env(env_cfg, field, vec_norm_path,
                                   use_masking=MASKABLE_PPO_AVAILABLE,
                                   data_manager=data_manager,
                                   wind_data=None, current_data=None,
                                   wind_mapping=wind_mapping,
                                   current_mapping=current_mapping)

                for ep in range(n_episodes):
                    result = run_episode(model, vec_env, deterministic=deterministic)
                    result.scenario = scenario_label
                    result.source_id = source_id
                    result.episode = ep

                    sv = 1.0 if result.success else 0.0
                    episode_success_all.append(sv)
                    episode_success_by_version[version].append(sv)
                    episode_success_by_chunk[chunk_label].append(sv)
                    initial_distances_all.append(result.initial_distance)
                    if result.success:
                        success_steps_all.append(float(result.steps))
                    episode_results.append(result)
                    all_results_global.append(result)

                    src_x, src_y = field.source_position
                    dist_history = [
                        float(np.sqrt((pos[0]-src_x)**2 + (pos[1]-src_y)**2))
                        for pos in result.trajectory
                    ]
                    episodes_data.append({
                        "source_id": source_id, "version": version,
                        "chunk": chunk_label, "chunk_id": chunk_id,
                        "episode": ep + 1, "success": result.success,
                        "termination": result.termination,
                        "initial_distance": result.initial_distance,
                        "final_distance": result.final_distance,
                        "steps": result.steps, "distance_history": dist_history,
                    })

                    if save_plots:
                        plot_path = version_dir / f"ep{ep+1:02d}_chunk{chunk_id}_trajectory.png"
                        save_trajectory_plot(result, field, plot_path, success_threshold)

                    init_dist = f"{result.initial_distance:.0f}m"
                    if result.success:
                        time_mins = (result.steps * dt_seconds) / 60
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → SUCCESS in {result.steps:3d} steps ({time_mins:5.1f}m)")
                    else:
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → {result.termination.upper()} at {result.steps:4d} steps (final_dist={result.final_distance:6.1f}m)")

                vec_env.close()

                if episode_results:
                    all_stats.append(compute_scenario_stats(episode_results, scenario_label, source_id))

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

    generate_analysis_plots(episodes_data, output_path, dt_seconds=dt_seconds,
                            max_steps=config.get('environment', {}).get('max_episode_steps', 1080))

    return all_stats


def find_model_for_sensor_range(trained_dir: Path, sensor_range: float) -> Optional[Path]:
    """Trova il modello più recente addestrato con il dato sensor_range."""
    candidates = []
    for run_dir in sorted(trained_dir.glob("ppo_*")):
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            cfg = load_config(str(cfg_path))
            if cfg.get('agent', {}).get('sensor_range') == sensor_range:
                model_zip = run_dir / "models" / "final_model.zip"
                if model_zip.exists():
                    candidates.append(model_zip)
        except Exception:
            continue
    return candidates[-1] if candidates else None


# ─── FCM: Field Climbing Method ───────────────────────────────────────────────

class FCMAgent:
    """
    Agente Field Climbing Method.

    Stima il gradiente locale del campo di concentrazione tramite regressione
    ai minimi quadrati con approssimazione di Taylor al primo ordine:

        C(p + δ) ≈ C(p) + ∇C · δ   →   sistema overdetermined 8×2 → LS

    L'azione scelta è la direzione discreta più allineata al gradiente stimato.

    Varianti:
      'pure'   – stima il gradiente dalle sole misure correnti (senza memoria).
      'kalman' – filtra il gradiente con un filtro di Kalman scalare
                 per-componente; riduce le oscillazioni dovute al rumore
                 di misura e all'evoluzione temporale del campo.
    """

    _DIAG = 1.0 / np.sqrt(2.0)
    _DIRECTIONS = np.array([
        [0.0,    1.0   ],   # 0: Nord
        [0.0,   -1.0   ],   # 1: Sud
        [1.0,    0.0   ],   # 2: Est
        [-1.0,   0.0   ],   # 3: Ovest
        [_DIAG,  _DIAG ],   # 4: NordEst
        [_DIAG, -_DIAG ],   # 5: SudEst
        [-_DIAG, _DIAG ],   # 6: NordOvest
        [-_DIAG,-_DIAG ],   # 7: SudOvest
    ], dtype=float)

    def __init__(
        self,
        sensor_range: float = 20.0,
        variant: str = 'pure',
        kalman_q: float = 1e-2,
        kalman_r: float = 1e-1,
    ):
        """
        Args:
            sensor_range: Distanza dei sensori direzionali (m).
            variant:      'pure' oppure 'kalman'.
            kalman_q:     Varianza processo Kalman (quanto può cambiare ∇C per step).
            kalman_r:     Varianza misura Kalman (rumore sulla stima LS).
        """
        self.sensor_range = sensor_range
        self.variant = variant
        self._kf_mean = np.zeros(2)
        self._kf_var  = np.ones(2)
        self._kf_q    = kalman_q
        self._kf_r    = kalman_r
        self._kf_initialized = False

    def reset(self):
        """Resetta lo stato interno; chiamare prima di ogni episodio."""
        self._kf_mean = np.zeros(2)
        self._kf_var  = np.ones(2)
        self._kf_initialized = False

    def _estimate_gradient_ls(self, obs: np.ndarray) -> np.ndarray:
        """Stima ∇C tramite minimi quadrati (Taylor 1° ordine)."""
        center_conc = float(obs[0])
        sensors = obs[28:36].astype(float)
        # A[i] = direzione unitaria i;  b[i] = ΔC / sensor_range
        b = (sensors - center_conc) / max(self.sensor_range, 1.0)
        gradient, _, _, _ = np.linalg.lstsq(self._DIRECTIONS, b, rcond=None)
        return gradient

    def _kalman_update(self, z: np.ndarray) -> np.ndarray:
        """Filtra la stima del gradiente con un KF scalare per-componente."""
        if not self._kf_initialized:
            self._kf_mean = z.copy()
            self._kf_initialized = True
            return self._kf_mean
        p_pred = self._kf_var + self._kf_q
        k = p_pred / (p_pred + self._kf_r)
        self._kf_mean = self._kf_mean + k * (z - self._kf_mean)
        self._kf_var  = (1.0 - k) * p_pred
        return self._kf_mean

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, None]:
        """Seleziona l'azione tramite gradient ascent.

        Interfaccia compatibile con MaskablePPO.predict per riuso di run_episode.

        Args:
            obs:          Osservazione shape (1, 116) o (116,).
            deterministic: Ignorato (FCM è deterministico dati gli input).
            action_masks: Maschera booleana azioni valide, shape (8,).

        Returns:
            (action_array, None)  – action_array shape (1,).
        """
        flat_obs = obs[0] if obs.ndim == 2 else obs
        gradient = self._estimate_gradient_ls(flat_obs)
        if self.variant == 'kalman':
            gradient = self._kalman_update(gradient)

        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm < 1e-12:
            # Gradiente nullo: sceglie random tra azioni valide
            if action_masks is not None:
                valid = np.where(action_masks)[0]
                action = int(np.random.choice(valid)) if len(valid) > 0 else 0
            else:
                action = int(np.random.randint(0, 8))
        else:
            scores = self._DIRECTIONS @ gradient   # (8,)
            if action_masks is not None:
                scores[~action_masks.astype(bool)] = -np.inf
            action = int(np.argmax(scores))

        return np.array([action]), None


def build_env_fcm(
    env_cfg,
    field,
    use_masking: bool,
    data_manager: Optional[DataManager] = None,
    wind_data=None,
    current_data=None,
    wind_mapping: Optional[Dict[str, str]] = None,
    current_mapping: Optional[Dict[str, str]] = None,
):
    """Costruisce l'environment per FCM (senza VecNormalize)."""
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
    return DummyVecEnv([lambda e=raw_env: e])


def run_episode_fcm(fcm_agent: FCMAgent, vec_env, deterministic: bool = True) -> EpisodeResult:
    """Esegue un singolo episodio FCM, resettando lo stato interno dell'agente."""
    fcm_agent.reset()
    return run_episode(fcm_agent, vec_env, deterministic=deterministic)


def run_inference_fcm(
    config_path: str,
    data_dir: str,
    output_dir: str,
    variant: str = 'pure',
    n_episodes: int = 5,
    sources_csv: str = "Coordinate_Sorgenti_FaseII.csv",
    chunk_ids: Optional[List[int]] = None,
    sensor_range: Optional[float] = None,
    save_plots: bool = True,
    save_videos: bool = True,
    seed: Optional[int] = None,
    config_override: Optional[dict] = None,
) -> List[ScenarioStats]:
    """Inferenza completa con Field Climbing Method.

    Stessa struttura e stessi output di run_inference() (log.txt, episodes_data.json,
    trajectory plots, analysis plots, showcase videos) ma con FCMAgent al posto del
    modello RL addestrato.

    Args:
        variant:      'pure' (gradiente corrente) o 'kalman' (filtro di Kalman).
        sensor_range: Se None, usa il valore da config (default 20 m).
    """
    if chunk_ids is None:
        chunk_ids = [0, 1, 2]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    config = config_override if config_override is not None else load_config(config_path)
    dt_seconds      = config.get('environment', {}).get('dt', 10)
    max_steps_cfg   = config.get('environment', {}).get('max_episode_steps', 1080)
    success_threshold = config.get('environment', {}).get('reward', {}).get('distance_threshold', 50)

    if sensor_range is None:
        sensor_range = float(config.get('agent', {}).get('sensor_range', 20.0))

    fcm_agent = FCMAgent(sensor_range=sensor_range, variant=variant)

    data_manager = DataManager(
        data_dir=data_dir,
        preload_all=False,
        sources_csv=sources_csv,
    )

    all_sources       = data_manager.get_discovered_sources()
    inference_sources = [s for s in all_sources if int(s[3:]) > 106]

    chunk_labels = {0: "Q1/4", 1: "Q1/2", 2: "Q3/4"}
    chunk_descriptions = ", ".join([
        f"{chunk_labels[cid]} (chunk_id={cid})" for cid in chunk_ids
    ])

    wind_mapping = {
        "_V0": "CI_WIND_faseII_V0.txt",
        "_V1": "CI_WIND_faseII_V1.txt",
        "_V2": "CI_WIND_faseII_V2.txt",
        "_V3": "CI_WIND_faseII_V3.txt",
    }
    current_mapping = {
        "_V0": "CL02_V0_SRC000_U_V_10mGrid.nc",
        "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
        "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc",
        "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc",
    }

    agent_label = f"FCM-{variant} (sensor_range={sensor_range}m)"
    print(f"\n{'='*100}")
    print(f"HYDRAS Inference FCM [{variant.upper()}] — {len(inference_sources)} sorgenti × 4 versioni × "
          f"{len(chunk_ids)} chunk × {n_episodes} ep.")
    print(f"  = {len(inference_sources)*4*len(chunk_ids)*n_episodes} episodi totali")
    print(f"Variante FCM: {agent_label}")
    print(f"Chunk testati: {chunk_descriptions}")
    print(f"Dati: {data_dir}")
    print(f"Sorgenti inference (20%): SRC107–SRC132 ({len(inference_sources)} sorgenti)")
    print(f"{'='*100}\n")

    all_stats: List[ScenarioStats]          = []
    episode_success_all: List[float]        = []
    episode_success_by_version: Dict[str, List[float]] = {'V0': [], 'V1': [], 'V2': [], 'V3': []}
    episode_success_by_chunk: Dict[str, List[float]]   = {'Q1/4': [], 'Q1/2': [], 'Q3/4': []}
    initial_distances_all: List[float]      = []
    success_steps_all: List[float]          = []
    all_results_global: List[EpisodeResult] = []
    episodes_data: List[dict]               = []

    for src_idx, source_id in enumerate(inference_sources, 1):
        if src_idx % 5 == 0:
            print(f"\n[Progress: {src_idx}/{len(inference_sources)} sources]\n")

        for version in ['V0', 'V1', 'V2', 'V3']:
            version_files = [f for f in data_manager._nc_files
                             if version in f.name and source_id in f.name]
            if not version_files:
                continue

            version_dir = output_path / source_id / version
            version_dir.mkdir(parents=True, exist_ok=True)

            try:
                field = data_manager._nc_loader.load(
                    str(version_files[0]),
                    concentration_var="Concentration - component 1",
                )
                if field is None:
                    continue
                coords = data_manager.get_source_coordinates(source_id)
                if coords:
                    field.source_position = coords
                field.run_id = f"{source_id}_{version}"
            except Exception as e:
                print(f"  [SKIP] {version}_{source_id}: {e}")
                continue

            for chunk_id in chunk_ids:
                chunk_label    = chunk_labels[chunk_id]
                scenario_label = f"{version}_{source_id}_{chunk_label}"
                episode_results: List[EpisodeResult] = []

                env_cfg = make_env_config(config, chunk_id=chunk_id)
                vec_env = build_env_fcm(
                    env_cfg, field,
                    use_masking=MASKABLE_PPO_AVAILABLE,
                    data_manager=data_manager,
                    wind_data=None, current_data=None,
                    wind_mapping=wind_mapping,
                    current_mapping=current_mapping,
                )

                for ep in range(n_episodes):
                    result = run_episode_fcm(fcm_agent, vec_env, deterministic=True)
                    result.scenario  = scenario_label
                    result.source_id = source_id
                    result.episode   = ep

                    sv = 1.0 if result.success else 0.0
                    episode_success_all.append(sv)
                    episode_success_by_version[version].append(sv)
                    episode_success_by_chunk[chunk_label].append(sv)
                    initial_distances_all.append(result.initial_distance)
                    if result.success:
                        success_steps_all.append(float(result.steps))
                    episode_results.append(result)
                    all_results_global.append(result)

                    src_x, src_y = field.source_position
                    dist_history = [
                        float(np.sqrt((pos[0] - src_x)**2 + (pos[1] - src_y)**2))
                        for pos in result.trajectory
                    ]
                    episodes_data.append({
                        "source_id": source_id, "version": version,
                        "chunk": chunk_label, "chunk_id": chunk_id,
                        "episode": ep + 1, "success": result.success,
                        "termination": result.termination,
                        "initial_distance": result.initial_distance,
                        "final_distance": result.final_distance,
                        "steps": result.steps, "distance_history": dist_history,
                    })

                    if save_plots:
                        plot_path = version_dir / f"ep{ep+1:02d}_chunk{chunk_id}_trajectory.png"
                        save_trajectory_plot(result, field, plot_path, success_threshold)

                    init_dist = f"{result.initial_distance:.0f}m"
                    if result.success:
                        time_mins = (result.steps * dt_seconds) / 60
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → SUCCESS in {result.steps:3d} steps ({time_mins:5.1f}m)")
                    else:
                        print(f"  {scenario_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → {result.termination.upper()} at {result.steps:4d} steps (final_dist={result.final_distance:6.1f}m)")

                vec_env.close()

                if episode_results:
                    all_stats.append(compute_scenario_stats(episode_results, scenario_label, source_id))

    def safe_mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    global_sr       = safe_mean(episode_success_all)
    mean_initial_dist = safe_mean(initial_distances_all)
    mean_success_steps = safe_mean(success_steps_all)

    version_sr = {v: safe_mean(episode_success_by_version.get(v, [])) for v in ['V0', 'V1', 'V2', 'V3']}
    chunk_sr   = {c: safe_mean(episode_success_by_chunk.get(c, []))   for c in ['Q1/4', 'Q1/2', 'Q3/4']}

    if all_stats:
        print(f"\n{'='*80}")
        print(f"FCM [{variant.upper()}] — RESULTS BY CHUNK")
        print(f"{'='*80}")
        for chunk_label in ['Q1/4', 'Q1/2', 'Q3/4']:
            sr  = chunk_sr.get(chunk_label)
            n_e = len(episode_success_by_chunk.get(chunk_label, []))
            print(f"{chunk_label}: {('n/d' if sr is None else f'{sr*100:6.1f}%')} ({n_e} episodes)")
        print(f"\n{'='*80}")
        print(f"FCM [{variant.upper()}] — RESULTS BY WIND SCENARIO")
        print(f"{'='*80}")
        for v in ['V0', 'V1', 'V2', 'V3']:
            sr  = version_sr.get(v)
            n_e = len(episode_success_by_version.get(v, []))
            print(f"{v}: {('n/d' if sr is None else f'{sr*100:6.1f}%')} ({n_e} episodes)")
        print(f"\n{'='*80}")
        print(f"Global Success Rate: {('n/d' if global_sr is None else f'{global_sr*100:.1f}%')}")
        print(f"Mean initial distance: {('n/d' if mean_initial_dist is None else f'{mean_initial_dist:.0f}m')}")
        if mean_success_steps is not None:
            print(f"Mean success steps: {mean_success_steps:.1f} (~{(mean_success_steps * dt_seconds)/60:.1f} min)")
        print(f"Total scenarios: {len(all_stats)}")
        print(f"Total episodes: {len(episode_success_all)}")
        print(f"{'='*80}\n")

    log_path = write_inference_log(
        output_path=output_path,
        model_path=agent_label,
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

    import json
    episodes_json_path = output_path / "episodes_data.json"
    with open(episodes_json_path, "w") as f:
        json.dump(episodes_data, f)
    print(f"Dati per-episodio salvati in: {episodes_json_path}")

    if save_videos:
        generate_showcase_videos(all_results_global, data_manager, output_path)

    generate_analysis_plots(episodes_data, output_path, dt_seconds=dt_seconds,
                            max_steps=max_steps_cfg)

    return all_stats


def main_fcm_inference():
    """Inferenza FCM baseline su SRC107-SRC132 con varianti 'pure' e 'kalman'.

    Struttura output:
      evaluations/evaluations_FCM/fcm_pure/
      evaluations/evaluations_FCM/fcm_kalman/
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR     = str(PROJECT_ROOT / "data")
    CONFIG_PATH  = str(PROJECT_ROOT / "utils" / "config.yaml")

    config = load_config(CONFIG_PATH)
    sensor_range = float(config.get('agent', {}).get('sensor_range', 20.0))

    fcm_base = PROJECT_ROOT / "evaluations" / "evaluations_FCM"

    for variant in ['pure', 'kalman']:
        output_dir = str(fcm_base / f"fcm_{variant}")
        print(f"\n{'#'*80}")
        print(f"# FCM variant: {variant.upper()}")
        print(f"# Output: {output_dir}")
        print(f"{'#'*80}\n")

        run_inference_fcm(
            config_path=CONFIG_PATH,
            data_dir=DATA_DIR,
            output_dir=output_dir,
            variant=variant,
            n_episodes=5,
            sources_csv="Coordinate_Sorgenti_FaseII.csv",
            chunk_ids=[0, 1, 2],
            sensor_range=sensor_range,
        )

    print(f"\nInferenza FCM completata. Output: {fcm_base}")


# ─── Analysis Plots ──────────────────────────────────────────────────────────

def generate_analysis_plots(
    episodes_data: List[dict],
    output_path: Path,
    dt_seconds: float = 10.0,
    max_steps: int = 1080,
) -> dict:
    """Genera i 4 plot di analisi quantitativa nella sottocartella analysis/.

    Returns:
        Dict path_name → Path dei PNG generati.
    """
    import matplotlib
    matplotlib.use('Agg')

    analysis_dir = output_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    episodes = episodes_data
    success_eps = [e for e in episodes if e['success']]
    fail_eps    = [e for e in episodes if not e['success']]
    plot_paths: dict = {}

    # ── Plot 1: Distribuzione tempi di successo ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    if success_eps:
        steps_arr   = np.array([e['steps'] for e in success_eps])
        minutes_arr = steps_arr * dt_seconds / 60.0
        p25 = np.percentile(minutes_arr, 25)
        p50 = np.percentile(minutes_arr, 50)
        p75 = np.percentile(minutes_arr, 75)
        p95 = np.percentile(minutes_arr, 95)
        mean_val = float(minutes_arr.mean())
        std_val  = float(minutes_arr.std())
        n_outliers = int(np.sum(minutes_arr > p95))
        clipped = minutes_arr[minutes_arr <= p95]
        n_bins = min(40, max(15, int(len(clipped) ** 0.5)))
        ax.hist(clipped, bins=n_bins, color='#2196F3', edgecolor='white',
                linewidth=0.5, alpha=0.8)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                   label=f'Media: {mean_val:.1f} min')
        ax.axvline(p50, color='orange', linestyle=':', linewidth=1.8,
                   label=f'Mediana: {p50:.1f} min')
        ax.set_xlim(0, p95)
        ax.set_xlabel('Tempo per raggiungere la sorgente (minuti simulati)', fontsize=11)
        ax.set_ylabel('Numero di episodi', fontsize=11)
        ax.set_title('Distribuzione dei Tempi di Successo', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.35)
        stats_txt = (
            f"N successi = {len(minutes_arr)}\n"
            f"Dev.std = {std_val:.1f} min\n"
            f"P25–P75 = [{p25:.1f}, {p75:.1f}] min\n"
            f"Outliers (>{p95:.0f} min) = {n_outliers}"
        )
        ax.text(0.97, 0.03, stats_txt, transform=ax.transAxes,
                fontsize=8.5, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.9))
    else:
        ax.text(0.5, 0.5, 'Nessun episodio di successo', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
    fig.tight_layout()
    p1 = analysis_dir / "plot_success_time_dist.png"
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['time_dist'] = p1

    # ── Plot 2: Heatmap SR versione×chunk + SR per distanza iniziale ─────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    versions = ['V0', 'V1', 'V2', 'V3']
    chunks   = ['Q1/4', 'Q1/2', 'Q3/4']
    matrix   = np.full((len(versions), len(chunks)), np.nan)
    for vi, v in enumerate(versions):
        for ci, c in enumerate(chunks):
            sub = [e for e in episodes if e['version'] == v and e['chunk'] == c]
            if sub:
                matrix[vi, ci] = float(np.mean([e['success'] for e in sub])) * 100

    ax = axes[0]
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(chunks)));  ax.set_xticklabels(chunks, fontsize=10)
    ax.set_yticks(range(len(versions))); ax.set_yticklabels(versions, fontsize=10)
    ax.set_title('Success Rate (%) — Versione × Chunk', fontsize=11, fontweight='bold')
    for vi in range(len(versions)):
        for ci in range(len(chunks)):
            val = matrix[vi, ci]
            if not np.isnan(val):
                ax.text(ci, vi, f'{val:.0f}%', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if val < 50 else 'black')
    fig.colorbar(im, ax=ax, label='SR (%)')

    ax2 = axes[1]
    bins_edges = [0, 500, 1000, 1500, 2000, 2500]
    bin_labels  = ['0–500', '500–1000', '1000–1500', '1500–2000', '2000+']
    bin_sr, bin_n = [], []
    for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
        sub = [e for e in episodes if lo <= e['initial_distance'] < hi]
        bin_sr.append(float(np.mean([e['success'] for e in sub])) * 100 if sub else 0.0)
        bin_n.append(len(sub))
    bar_colors = ['#4CAF50' if s >= 80 else '#FF9800' if s >= 50 else '#F44336'
                  for s in bin_sr]
    bars = ax2.bar(bin_labels, bin_sr, color=bar_colors, edgecolor='white', linewidth=0.5)
    for bar, n in zip(bars, bin_n):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'n={n}', ha='center', va='bottom', fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.set_xlabel('Distanza iniziale dalla sorgente (m)', fontsize=10)
    ax2.set_ylabel('Success Rate (%)', fontsize=10)
    ax2.set_title('SR in funzione della Distanza Iniziale', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=9)
    ax2.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    p2 = analysis_dir / "plot_sr_analysis.png"
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['sr_analysis'] = p2

    # ── Plot 3: Distanza media dalla sorgente nel tempo ──────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    def _pad_and_stack(ep_list, max_len):
        padded = []
        for e in ep_list:
            h = e['distance_history']
            if len(h) < max_len:
                h = h + [h[-1]] * (max_len - len(h))
            padded.append(h[:max_len])
        return np.array(padded) if padded else None

    max_len    = max_steps + 1
    steps_axis = np.arange(max_len) * dt_seconds / 60.0
    all_mat = _pad_and_stack(episodes, max_len)
    suc_mat = _pad_and_stack(success_eps, max_len)
    fai_mat = _pad_and_stack(fail_eps, max_len)
    if all_mat is not None:
        ax.plot(steps_axis, all_mat.mean(axis=0), color='steelblue',
                linewidth=1.8, label=f'Tutti ({len(episodes)} ep.)')
    if suc_mat is not None:
        ax.plot(steps_axis, suc_mat.mean(axis=0), color='green',
                linewidth=1.8, linestyle='--', label=f'Successi ({len(success_eps)} ep.)')
    if fai_mat is not None:
        ax.plot(steps_axis, fai_mat.mean(axis=0), color='crimson',
                linewidth=1.8, linestyle=':', label=f'Fallimenti ({len(fail_eps)} ep.)')
    ax.set_xlabel('Tempo (minuti)', fontsize=11)
    ax.set_ylabel('Distanza media dalla sorgente (m)', fontsize=11)
    ax.set_title('Evoluzione della Distanza dalla Sorgente nel Tempo', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.4)
    ax.set_xlim(0, max_steps * dt_seconds / 60.0)
    fig.tight_layout()
    p3 = analysis_dir / "plot_distance_over_time.png"
    fig.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['distance_time'] = p3

    # ── Plot 4: Distribuzione distanza iniziale di spawn ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    init_dists      = np.array([e['initial_distance'] for e in episodes])
    init_dists_suc  = np.array([e['initial_distance'] for e in success_eps]) if success_eps else np.array([])
    init_dists_fail = np.array([e['initial_distance'] for e in fail_eps])    if fail_eps    else np.array([])

    ax = axes[0]
    bin_edges   = np.arange(0, float(init_dists.max()) + 200, 200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_w = 160
    counts_suc  = np.histogram(init_dists_suc,  bins=bin_edges)[0] if len(init_dists_suc)  else np.zeros(len(bin_centers), int)
    counts_fail = np.histogram(init_dists_fail, bins=bin_edges)[0] if len(init_dists_fail) else np.zeros(len(bin_centers), int)
    ax.bar(bin_centers, counts_suc,  width=bar_w, color='#4CAF50', edgecolor='white',
           linewidth=0.4, label=f'Successi ({len(success_eps)})', alpha=0.9)
    ax.bar(bin_centers, counts_fail, width=bar_w, bottom=counts_suc, color='#F44336',
           edgecolor='white', linewidth=0.4, label=f'Fallimenti ({len(fail_eps)})', alpha=0.9)
    ax.axvline(float(np.mean(init_dists)), color='navy', linestyle='--', linewidth=1.5,
               label=f'Media: {np.mean(init_dists):.0f} m')
    ax.axvline(float(np.median(init_dists)), color='darkorange', linestyle=':', linewidth=1.8,
               label=f'Mediana: {np.median(init_dists):.0f} m')
    ax.set_xlabel('Distanza iniziale dalla sorgente (m)', fontsize=11)
    ax.set_ylabel('Numero di episodi', fontsize=11)
    ax.set_title('Distribuzione delle Distanze di Spawn', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35)
    stats_txt = (
        f"N tot = {len(init_dists)}\n"
        f"Min = {init_dists.min():.0f} m\n"
        f"Max = {init_dists.max():.0f} m\n"
        f"Std = {init_dists.std():.0f} m\n"
        f"P25 = {np.percentile(init_dists, 25):.0f} m\n"
        f"P75 = {np.percentile(init_dists, 75):.0f} m"
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes, fontsize=8.5,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.9))

    ax2 = axes[1]
    fine_edges   = np.arange(0, float(init_dists.max()) + 250, 250)
    fine_centers = (fine_edges[:-1] + fine_edges[1:]) / 2
    fine_sr, fine_n = [], []
    for lo, hi in zip(fine_edges[:-1], fine_edges[1:]):
        sub = [e for e in episodes if lo <= e['initial_distance'] < hi]
        fine_sr.append(float(np.mean([e['success'] for e in sub])) * 100 if sub else np.nan)
        fine_n.append(len(sub))
    valid = [(c, s, n) for c, s, n in zip(fine_centers, fine_sr, fine_n) if not np.isnan(s)]
    if valid:
        vcs, vsr, vns = zip(*valid)
        bar_colors2 = ['#4CAF50' if s >= 80 else '#FF9800' if s >= 50 else '#F44336'
                       for s in vsr]
        bars2 = ax2.bar(vcs, vsr, width=220, color=bar_colors2, edgecolor='white', linewidth=0.4)
        for bar, n in zip(bars2, vns):
            if n > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1.5, f'n={n}',
                         ha='center', va='bottom', fontsize=7.5)
    global_sr_val = round(float(np.mean([e['success'] for e in episodes])) * 100)
    ax2.axhline(global_sr_val, color='navy', linestyle='--', linewidth=1.2,
                label=f'SR globale {global_sr_val}%')
    ax2.set_ylim(0, 115)
    ax2.set_xlabel('Distanza iniziale dalla sorgente (m)', fontsize=11)
    ax2.set_ylabel('Success Rate (%)', fontsize=11)
    ax2.set_title('SR per Fascia di Distanza Iniziale (bins 250 m)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.35)
    fig.tight_layout()
    p4 = analysis_dir / "plot_initial_dist.png"
    fig.savefig(p4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plot_paths['initial_dist'] = p4

    print(f"  Analisi plots salvati in: {analysis_dir}")
    return plot_paths


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR    = str(PROJECT_ROOT / "data")
    CONFIG_PATH = str(PROJECT_ROOT / "utils" / "config.yaml")
    trained_dir = PROJECT_ROOT / "trained_models"

    config = load_config(CONFIG_PATH)
    sweep = config.get('training', {}).get('sensor_range_sweep', [])

    if sweep:
        BASE_VERSION = 8  # evaluations_v8 → v8+len(sweep)-1
        print(f"\nInference sweep: sensor_range {sweep} m → evaluations_v{BASE_VERSION}–v{BASE_VERSION+len(sweep)-1}\n")

        for i, sr in enumerate(sweep):
            model_path = find_model_for_sensor_range(trained_dir, sr)
            if model_path is None:
                print(f"[SKIP] Nessun modello trovato per sensor_range={sr}m in {trained_dir}")
                continue

            output_dir = str(PROJECT_ROOT / "evaluations" / "evaluations_RL" / f"evaluations_v{BASE_VERSION + i}")

            cfg_override = load_config(CONFIG_PATH)
            cfg_override['agent']['sensor_range'] = sr

            print(f"\n{'='*70}")
            print(f"Inference  sensor_range={sr}m  →  evaluations_v{BASE_VERSION + i}")
            print(f"Modello: {model_path}")
            print(f"{'='*70}\n")

            run_inference(
                model_path=str(model_path),
                config_path=CONFIG_PATH,
                config_override=cfg_override,
                data_dir=DATA_DIR,
                output_dir=output_dir,
                n_episodes=5,
                deterministic=True,
                sources_csv="Coordinate_Sorgenti_FaseII.csv",
                chunk_ids=[0, 1, 2],
            )

        print(f"\nInference sweep completato. Output: evaluations_v{BASE_VERSION}–v{BASE_VERSION+len(sweep)-1}")

    else:
        # Fallback: usa l'ultimo modello disponibile
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

        # Leggi sensor_range dal config salvato nel run
        cfg_override = load_config(CONFIG_PATH)
        run_cfg_path = latest_run / "config.yaml"
        if run_cfg_path.exists():
            run_cfg = load_config(str(run_cfg_path))
            sr = run_cfg.get('agent', {}).get('sensor_range', cfg_override['agent'].get('sensor_range', 20))
            cfg_override['agent']['sensor_range'] = sr
            print(f"sensor_range dal modello: {sr}m")

        output_dir = str(PROJECT_ROOT / "evaluations" / "evaluations_RL" / "evaluations_v12")

        print(f"Modello selezionato: {model_path}")
        print(f"Output valutazioni: {output_dir}")

        run_inference(
            model_path=str(model_path),
            config_path=CONFIG_PATH,
            config_override=cfg_override,
            data_dir=DATA_DIR,
            output_dir=output_dir,
            n_episodes=5,
            deterministic=True,
            sources_csv="Coordinate_Sorgenti_FaseII.csv",
            chunk_ids=[0, 1, 2],
        )


def main_ablation_inference():
    """Inferenza sui 2 modelli ablation (penultimo=base, ultimo=base_no_wind_reward).
    Nessun plot generato; spawn riproducibili con seed fisso.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = str(PROJECT_ROOT / "data")
    trained_dir = PROJECT_ROOT / "trained_models"

    run_dirs = sorted([d for d in trained_dir.iterdir()
                       if d.is_dir() and d.name.startswith("ppo_")])
    if len(run_dirs) < 2:
        print("ERRORE: Meno di 2 modelli trovati in trained_models/")
        sys.exit(1)

    ablation_runs = run_dirs[-2:]  # penultimo=base, ultimo=base_no_wind_reward
    ablation_configs = [
        str(PROJECT_ROOT / "utils" / "config_base.yaml"),
        str(PROJECT_ROOT / "utils" / "config_base_no_wind_reward.yaml"),
    ]
    ablation_names = ["base", "base_no_wind_reward"]

    for run_dir, config_path, name in zip(ablation_runs, ablation_configs, ablation_names):
        model_path = run_dir / "models" / "final_model.zip"
        if not model_path.exists():
            model_path = run_dir / "models" / "best" / "best_model.zip"
        if not model_path.exists():
            print(f"ERRORE: Nessun modello trovato in {run_dir}/models/ — skip")
            continue

        output_dir = str(PROJECT_ROOT / "evaluations" / "evaluations_RL" / f"evaluations_ablation_{name}")

        print(f"\n{'='*70}")
        print(f"Ablation inference: reward_mode={name}")
        print(f"Modello: {model_path}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")

        run_inference(
            model_path=str(model_path),
            config_path=config_path,
            data_dir=DATA_DIR,
            output_dir=output_dir,
            n_episodes=5,
            deterministic=True,
            sources_csv="Coordinate_Sorgenti_FaseII.csv",
            chunk_ids=[0, 1, 2],
            save_videos=False,
            save_plots=False,
        )


if __name__ == "__main__":
    main_ablation_inference()