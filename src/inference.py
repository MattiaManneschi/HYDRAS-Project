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
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    termination: str       # "success" / "timeout" / "boundary"
    initial_distance: float  # m - distanza spawn-sorgente
    final_distance: float  # m
    steps: int
    trajectory: np.ndarray


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
        source_distance_threshold=reward_cfg.get('distance_threshold', 100),
        source_found_reward=reward_cfg.get('source_reached_bonus', 100),
        step_penalty=reward_cfg.get('step_penalty', -0.1),
        boundary_penalty=reward_cfg.get('boundary_penalty', -10),
        distance_reward_multiplier=reward_cfg.get('distance_reward_multiplier', 1.0),
        land_proximity_threshold=reward_cfg.get('land_proximity_threshold', 10.0),
        land_proximity_penalty_max=reward_cfg.get('land_proximity_penalty_max', -5.0),
        n_discrete_actions=agent_cfg.get('n_discrete_actions', 8),
        spawn_min_distance=spawn_cfg.get('min_distance', 500),
        spawn_max_distance=spawn_cfg.get('max_distance', 1500),
        spawn_min_land_distance=spawn_cfg.get('min_land_distance', 50.0),
        spawn_start_frame=spawn_cfg.get('start_frame', 1440),
        spawn_conc_threshold=spawn_cfg.get('conc_threshold', 0.5),
        chunk_id=chunk_id,
        plume_reward_positive=reward_cfg.get('plume_reward_positive', 0.5),
        plume_reward_negative=reward_cfg.get('plume_reward_negative', -0.5),
        plume_threshold=reward_cfg.get('plume_threshold', 0.1),
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
              current_data = None):
    """Costruisce e wrappa l'environment per l'inferenza.
    
    Args:
        data_manager: DataManager per accesso ai dati
        wind_data: Dati di vento (caricati da DataManager)
        current_data: Dati di corrente (caricati da DataManager)
    """
    raw_env = SourceSeekingEnv(
        config=env_cfg,
        concentration_field=field,
        wind_data=wind_data,
        current_data=current_data,
        data_manager=data_manager,
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

    # Determina terminazione
    if last_info.get('source_reached', False):
        termination = 'success'
    elif last_info.get('out_of_bounds', False):
        termination = 'boundary'
    else:
        termination = 'timeout'

    # Usa i valori dall'info dict (inner.* sono già resettati da DummyVecEnv)
    final_dist = last_info.get('distance_to_source', 0.0)
    n_steps = last_info.get('steps', len(trajectory) - 1)

    return EpisodeResult(
        scenario="",           # impostato dal chiamante
        source_id="",          # impostato dal chiamante
        episode=0,             # impostato dal chiamante
        success=termination == 'success',
        termination=termination,
        initial_distance=initial_dist,
        final_distance=final_dist,
        steps=n_steps,
        trajectory=np.array(trajectory)
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
    """Salva il plot della traiettoria per un singolo episodio."""
    fig, ax = plt.subplots(figsize=(12, 10))

    status = "SUCCESS ✓" if result.success else f"FAILED [{result.termination}]"
    title = (f"{result.scenario} — Ep {result.episode+1} — {status}\n"
             f"dist={result.final_distance:.0f}m  steps={result.steps}")

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
        chunk_ids:    Lista di chunk_id da testare (default [0, 1] = Q1/4 e Q1/2)
                     0 = spawn @1/4, 1 = spawn @1/2
    """
    if chunk_ids is None:
        chunk_ids = [0, 1]  # Default: Q1/4 e Q1/2
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Carica config e modello
    config = load_config(config_path)
    model = load_model(model_path)
    env_cfg = make_env_config(config)
    success_threshold = config.get('environment', {}).get('reward', {}).get('distance_threshold', 100)

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
    
    # Mappa chunk_id a label
    chunk_labels = {0: "Q1/4", 1: "Q1/2"}
    chunk_descriptions = ", ".join([f"{chunk_labels[cid]} (chunk_id={cid})" for cid in chunk_ids])
    
    print(f"\n{'='*100}")
    print(f"HYDRAS Inference — {len(inference_sources)} sorgenti × 4 scenari vento × {len(chunk_ids)} chunk × {n_episodes} episodi")
    print(f"  = {len(inference_sources)*4*len(chunk_ids)*n_episodes} episodi totali")
    print(f"Chunk testati: {chunk_descriptions}")
    print(f"Modello: {model_path}")
    print(f"Dati: {data_dir}")
    print(f"Sorgenti training: SRC001-SRC106 (80%)")
    print(f"Sorgenti inference (20%): SRC107-SRC132 ({len(inference_sources)} sorgenti)")
    print(f"{'='*100}\n")
    
    # Carica dati vento e corrente (condivisi per tutte le sorgenti)
    wind_data = data_manager.get_wind_data()
    current_data = data_manager.get_current_data()
    
    if wind_data is None:
        print("WARNING: Wind data not loaded")
    else:
        print(f"Wind data: {len(wind_data.speed)} timesteps")
    
    if current_data is None:
        print("WARNING: Current data not loaded")
    else:
        print(f"Current data: {current_data.n_timesteps} timesteps")
    
    # Get dt from config
    dt_seconds = config.get('environment', {}).get('dt', 10)
    
    all_stats: List[ScenarioStats] = []
    version_stats: Dict[str, List[float]] = {'V0': [], 'V1': [], 'V2': [], 'V3': []}

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

                for ep in range(n_episodes):
                    # Carica campo da file specifico
                    try:
                        field = data_manager._nc_loader.load(
                            str(version_file),
                            concentration_var="Concentration - component 1"
                        )
                        if field is None:
                            print(f"\n  [SKIP] Could not load field for {version}_{source_id}")
                            break
                        
                        # Imposta source_position dalle coordinate CSV
                        coords = data_manager.get_source_coordinates(source_id)
                        if coords:
                            field.source_position = coords
                        
                    except Exception as e:
                        print(f"\n  [SKIP] Error loading field for {version}_{source_id}: {e}")
                        break

                    # Crea env config con chunk_id appropriato
                    env_cfg_ep = make_env_config(config, chunk_id=chunk_id)

                    vec_env = build_env(env_cfg_ep, field, vec_norm_path,
                                       use_masking=MASKABLE_PPO_AVAILABLE,
                                       data_manager=data_manager,
                                       wind_data=wind_data,
                                       current_data=current_data)

                    result = run_episode(model, vec_env, deterministic=deterministic)
                    result.scenario = scenario_label
                    result.source_id = source_id
                    result.episode = ep

                    episode_results.append(result)
                    vec_env.close()

                    # Salva plot traiettoria
                    plot_path = version_dir / f"ep{ep+1:02d}_chunk{chunk_id}_trajectory.png"
                    save_trajectory_plot(result, field, plot_path, success_threshold)
                    
                    # Log episodio
                    init_dist = f"{result.initial_distance:.0f}m"
                    if result.success:
                        time_mins = (result.steps * dt_seconds) / 60
                        print(f"  {version}_{source_id}_{chunk_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → SUCCESS in {result.steps:3d} steps ({time_mins:5.1f}m)")
                    else:
                        print(f"  {version}_{source_id}_{chunk_label} Ep{ep+1}: spawn_dist={init_dist:>5s} → {result.termination.upper()}")

                if episode_results:
                    # Statistiche scenario
                    stats = compute_scenario_stats(episode_results, scenario_label, source_id)
                    all_stats.append(stats)
                    
                    # Accumula SR per versione
                    version_stats[version].append(stats.success_rate)

    # Riepilogo globale
    if all_stats:
        # Mappa chunk_id a label
        chunk_labels_map = {0: "Q1/4", 1: "Q1/2"}
        
        # Aggregazione per chunk
        chunk_stats = {}
        for chunk_id in chunk_ids:
            chunk_label = chunk_labels_map[chunk_id]
            chunk_scenarios = [s for s in all_stats if chunk_label in s.scenario]
            if chunk_scenarios:
                chunk_sr = np.mean([s.success_rate for s in chunk_scenarios])
                chunk_stats[chunk_label] = (chunk_sr, len(chunk_scenarios))
        
        print(f"\n{'='*80}")
        print(f"RESULTS BY CHUNK (Time Instant)")
        print(f"{'='*80}")
        for chunk_label, (chunk_sr, n_scenarios) in chunk_stats.items():
            print(f"{chunk_label}: {chunk_sr*100:6.1f}% ({n_scenarios} scenarios)")
        
        print(f"\n{'='*80}")
        print(f"RESULTS BY WIND SCENARIO (across all chunks)")
        print(f"{'='*80}")
        
        for version in ['V0', 'V1', 'V2', 'V3']:
            sr_list = version_stats[version]
            if sr_list:
                version_sr = np.mean(sr_list)
                print(f"{version}: {version_sr*100:6.1f}% ({len(sr_list)} scenarios)")
            else:
                print(f"{version}: (no data)")
        
        # Global SR
        global_sr = np.mean([s.success_rate for s in all_stats])
        
        print(f"\n{'='*80}")
        print(f"Global Success Rate: {global_sr*100:.1f}%")
        print(f"Total scenarios: {len(all_stats)}")
        print(f"{'='*80}\n")

    return all_stats


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Root del progetto (parent di src/)

    # Carica da tutte le versioni (V0, V2, V3 - escludendo V1 per consistency con training)
    DATA_DIR    = str(PROJECT_ROOT / "data")
    CONFIG_PATH = str(PROJECT_ROOT / "utils" / "config.yaml")
    OUTPUT_DIR  = str(PROJECT_ROOT / "evaluations_v4")  # Valutazione su 132 sorgenti

    # Seleziona l'ultimo modello addestrato (directory più recente per nome)
    trained_dir = PROJECT_ROOT / "trained_models"
    run_dirs = sorted([d for d in trained_dir.iterdir() if d.is_dir() and d.name.startswith("ppo_")])
    
    if not run_dirs:
        print("ERRORE: Nessuna directory di training trovata in trained_models/")
        sys.exit(1)
    
    latest_run = run_dirs[-1]  # L'ultima per ordine alfabetico (timestamp nel nome)
    
    # Cerca il modello nella directory: prima final_model, poi best_model
    model_path = latest_run / "models" / "final_model.zip"
    if not model_path.exists():
        model_path = latest_run / "models" / "best" / "best_model.zip"
    
    if not model_path.exists():
        print(f"ERRORE: Nessun modello trovato in {latest_run}/models/")
        sys.exit(1)

    MODEL_PATH = str(model_path)
    print(f"Modello selezionato: {MODEL_PATH}")

    run_inference(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        n_episodes=5,
        deterministic=True,
        sources_csv="Coordinate_Sorgenti_FaseII.csv",
        chunk_ids=[0, 1],  # Q1/4 (chunk_id=0) and Q1/2 (chunk_id=1)
    )


if __name__ == "__main__":
    main()