"""
Generazione video showcase per inferenza HYDRAS.

Seleziona automaticamente 4 successi (uno per versione vento V0-V3,
step più vicini alla mediana) + 1 fallimento (distanza finale minima),
poi genera video MP4 animati ricaricando i field dal disco.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import ListedColormap


def select_showcase_episodes(all_results: list) -> list:
    """
    Seleziona 4 successi (uno per V0/V1/V2/V3, step più vicino alla mediana)
    e 1 fallimento (distanza finale dalla sorgente minima).

    Il formato di result.scenario è "V1_SRC109_Q1/4".
    """
    successes_by_version = defaultdict(list)
    failures = []

    for result in all_results:
        version = result.scenario.split('_')[0]
        if result.success and version in ('V0', 'V1', 'V2', 'V3'):
            successes_by_version[version].append(result)
        elif not result.success:
            failures.append(result)

    selected = []
    for version in ['V0', 'V1', 'V2', 'V3']:
        eps = successes_by_version.get(version, [])
        if not eps:
            continue
        median_steps = float(np.median([e.steps for e in eps]))
        best = min(eps, key=lambda e: abs(e.steps - median_steps))
        selected.append(best)

    if failures:
        best_fail = min(failures, key=lambda e: e.final_distance)
        selected.append(best_fail)

    return selected


def _make_video(result, field, output_path: Path,
                fps: int = 15, target_duration_s: int = 30) -> Path:
    """Genera e salva un video animato per un singolo episodio."""
    traj = result.trajectory          # (N, 2)
    N = len(traj)
    stride = max(1, N // (fps * target_duration_s))
    n_anim_frames = (N + stride - 1) // stride

    extent = [float(field.x_coords.min()), float(field.x_coords.max()),
              float(field.y_coords.min()), float(field.y_coords.max())]
    plume_threshold = 0.01

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#87CEEB')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Terra (fissa)
    if field.land_mask is not None:
        white_cmap = ListedColormap(['#FFFFFF'])
        land_display = np.ma.masked_where(~field.land_mask, np.ones(field.land_mask.shape))
        ax.imshow(land_display, origin='lower', extent=extent,
                  cmap=white_cmap, alpha=1.0, zorder=1)

    # Campo concentrazione al frame iniziale
    field.set_time(result.start_frame)
    conc0 = field.get_current_field()
    vmax = max(float(conc0.max()), 0.1)
    mask0 = (field.land_mask | (conc0 < plume_threshold)) if field.land_mask is not None \
             else (conc0 < plume_threshold)
    im = ax.imshow(np.ma.masked_where(mask0, conc0), origin='lower', extent=extent,
                   cmap='YlOrRd', alpha=0.9, vmin=0, vmax=vmax, zorder=2)
    plt.colorbar(im, ax=ax, label='Concentrazione')

    # Sorgente e punto di partenza (fissi)
    if field.source_position is not None:
        ax.scatter(field.source_position[0], field.source_position[1],
                   c='yellow', s=200, marker='*', edgecolors='black', zorder=6, label='Source')
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')

    # Artisti animati
    traj_line, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.85, label='Traiettoria')
    agent_dot  = ax.scatter([], [], c='darkred', s=140, marker='o', zorder=7, label='Agente')

    status_str = "SUCCESS ✓" if result.success else f"FAILED [{result.termination}]"
    base_title = f"{result.scenario} — Ep {result.episode + 1} — {status_str}"
    title_obj = ax.set_title(base_title, fontsize=11)
    ax.legend(loc='upper right', fontsize=9)

    n_frames_field = result.end_frame - result.start_frame

    def _update(anim_idx):
        traj_idx = min(anim_idx * stride, N - 1)
        t = (result.start_frame + n_frames_field * traj_idx / max(N - 1, 1)) \
            if n_frames_field > 0 else float(result.start_frame)
        field.set_time(t)
        conc = field.get_current_field()
        mask = (field.land_mask | (conc < plume_threshold)) if field.land_mask is not None \
               else (conc < plume_threshold)
        im.set_data(np.ma.masked_where(mask, conc))
        traj_line.set_data(traj[:traj_idx + 1, 0], traj[:traj_idx + 1, 1])
        agent_dot.set_offsets([[traj[traj_idx, 0], traj[traj_idx, 1]]])
        title_obj.set_text(f"{base_title}\nStep {traj_idx}/{N - 1}")
        return im, traj_line, agent_dot, title_obj

    ani = FuncAnimation(fig, _update, frames=n_anim_frames, blit=True, interval=1000 // fps)

    if FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, metadata=dict(title=base_title), bitrate=2000)
        video_path = output_path.with_suffix('.mp4')
        ani.save(str(video_path), writer=writer, dpi=120)
    else:
        writer = PillowWriter(fps=fps)
        video_path = output_path.with_suffix('.gif')
        ani.save(str(video_path), writer=writer, dpi=100)

    plt.close(fig)
    return video_path


def generate_showcase_videos(
    all_results: list,
    data_manager,
    output_dir: Path,
    fps: int = 15,
):
    """
    Seleziona i 5 episodi showcase tra tutti i risultati dell'inferenza
    e genera i video MP4 corrispondenti ricaricando i field dal disco.

    Args:
        all_results:       lista di EpisodeResult accumulati durante run_inference
        data_manager:      DataManager per ricaricare i field
        output_dir:        directory radice dell'inferenza (videos/ creata al suo interno)
        fps:               fotogrammi al secondo
    """
    selected = select_showcase_episodes(all_results)
    if not selected:
        print("[Video] Nessun episodio idoneo trovato.")
        return

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generazione video showcase — {len(selected)} episodi selezionati")
    print(f"{'='*60}")
    for r in selected:
        label = "SUCCESSO" if r.success else "FALLIMENTO"
        print(f"  • {r.scenario}  Ep{r.episode + 1}  [{label}]  steps={r.steps}  "
              f"final_dist={r.final_distance:.0f}m")
    print()

    for result in selected:
        # scenario format: "V1_SRC109_Q1/4"
        parts = result.scenario.split('_')
        version   = parts[0]
        source_id = parts[1] if len(parts) > 1 else "SRC000"

        print(f"  Generando: {result.scenario}  Ep{result.episode + 1} ...", flush=True)
        try:
            matching = [
                f for f in data_manager._nc_files
                if source_id in f.name and version in f.name and 'Conc' in f.name
            ]
            if not matching:
                raise FileNotFoundError(
                    f"Nessun file NC per {source_id}/{version} in {data_manager.data_dir}"
                )
            field = data_manager._nc_loader.load(str(matching[0]))
            coords = data_manager.get_source_coordinates(source_id)
            if coords:
                field.source_position = coords
        except Exception as exc:
            print(f"  [SKIP] Impossibile caricare il field: {exc}")
            continue

        tag = "success" if result.success else "failure"
        safe_scenario = result.scenario.replace('/', '-')
        video_stem = videos_dir / f"{safe_scenario}_ep{result.episode + 1:02d}_{tag}"

        saved = _make_video(result, field, video_stem, fps=fps)
        print(f"  ✓ {saved.name}")

    print(f"\nVideo salvati in: {videos_dir}\n")
