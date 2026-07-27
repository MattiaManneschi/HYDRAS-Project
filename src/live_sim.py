#!/usr/bin/env python3
"""
HYDRAS Source Seeking — Simulazione LIVE con pannello di controllo.

Piccola GUI (Tkinter): l'utente sceglie da menu a tendina lo scenario e l'agente,
preme "Avvia" e guarda UN episodio svolgersi in tempo reale nella stessa finestra.
A episodio concluso il programma si ripristina (tutte le tendine ai valori di
default) e resta pronto per una nuova esecuzione.

Menu a tendina:
  - Tecnologia   : PPO  oppure  FCM Adam
  - V (vento)    : V0 / V1 / V2 / V3
  - Q (chunk)    : Q1/4 / Q1/2 / Q3/4
  - v_max        : 1..5           (solo PPO)
  - Formazione   : Singola / Doppia corona   (solo PPO)

Scelti V e Q, viene pescata una sorgente held-out a caso con quella versione.
FCM usa sempre FCM Adam nella configurazione migliore (lr=40 m, sensor_range=50 m).

Il codice è diviso in due parti:
  - il MOTORE (make_agent_env / step_once / draw_scene): costruisce agente+env,
    avanza di un passo e disegna. Non dipende da Tkinter ed è testabile headless.
  - la GUI (App): i widget e il loop non bloccante via root.after().

Uso:
    python src/live_sim.py
"""

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import (
    MASKABLE_PPO_AVAILABLE,
    AdamFCMAgent,
    _find_dualcorona_run,
    _find_velocity_run,
    build_env,
    build_env_fcm,
    get_inner_env,
    load_config,
    load_model,
    make_env_config,
)
from utils.data_loader import DataManager


# ─── Costanti ────────────────────────────────────────────────────────────────
WIND_MAPPING = {
    "_V0": "CI_WIND_faseII_V0.txt",
    "_V1": "CI_WIND_faseII_V1.txt",
    "_V2": "CI_WIND_faseII_V2.txt",
    "_V3": "CI_WIND_faseII_V3.txt",
}
CURRENT_MAPPING = {
    "_V0": "CL02_V0_SRC000_U_V_10mGrid.nc",
    "_V1": "CL02_V1_SRC000_U_V_10mGrid.nc",
    "_V2": "CL02_V2_SRC000_U_V_10mGrid.nc",
    "_V3": "CL02_V3_SRC000_U_V_10mGrid.nc",
}
CHUNK_BY_LABEL = {"Q1/4": 0, "Q1/2": 1, "Q3/4": 2}
VERSIONS = ["V0", "V1", "V2", "V3"]
FCM_LR = 40.0            # miglior configurazione dello sweep (Cap. 3)
FCM_SENSOR_RANGE = 50.0

# Default dei menu a tendina (ripristinati a fine episodio).
DEFAULTS = {"tech": "PPO", "version": "V0", "chunk": "Q1/4",
            "vmax": "2", "formation": "Doppia"}


# ─── Requisiti / dati / modelli: check e download ────────────────────────────
REPO_SLUG = "MattiaManneschi/HYDRAS-Project"
MODELS_TARBALL_URL = f"https://codeload.github.com/{REPO_SLUG}/tar.gz/refs/heads/master"
# URL dell'archivio dei dati .nc (34 GB) su host esterno (Zenodo/Drive/...): non
# possono stare su GitHub (40 file > 100 MB). Inserisci qui il link; se vuoto e i
# dati mancano, l'avvio si ferma con un messaggio invece di scaricare.
DATA_ARCHIVE_URL = ""

REQUIRED_PACKAGES = ["torch", "stable_baselines3", "sb3_contrib", "gymnasium",
                     "numpy", "scipy", "matplotlib", "netCDF4", "yaml"]


def missing_packages() -> list:
    """Pacchetti richiesti non importabili nell'ambiente corrente."""
    import importlib
    out = []
    for m in REQUIRED_PACKAGES:
        try:
            importlib.import_module(m)
        except Exception:
            out.append(m)
    return out


def install_requirements(root_dir: Path) -> None:
    """pip install -r requirements.txt nell'interprete corrente."""
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(root_dir / "requirements.txt")], check=True)


def data_present(root_dir: Path) -> bool:
    """Verifica che i file .nc necessari (concentrazione + 4 corrente + 4 vento)
    siano presenti in data/."""
    d = root_dir / "data"
    conc = list(d.glob("**/*Conc_10mGrid.nc")) + list(d.glob("*Conc_10mGrid.nc"))
    cur = list(d.glob("**/*SRC000_U_V_10mGrid.nc")) + list(d.glob("*SRC000_U_V_10mGrid.nc"))
    wind = list(d.glob("**/CI_WIND_faseII_V*.txt")) + list(d.glob("CI_WIND_faseII_V*.txt"))
    return len(conc) > 0 and len(cur) >= 4 and len(wind) >= 4


def missing_models(root_dir: Path) -> list:
    """Combinazioni (formazione, v_max) per cui manca il modello PPO."""
    trained = root_dir / "trained_models"
    miss = []
    for vmax in (1, 2, 3, 4, 5):
        s = _find_velocity_run(trained, vmax, 5)
        if s is None or not (s / "models" / "final_model.zip").exists():
            miss.append(f"singola v{vmax}")
        d = _find_dualcorona_run(trained, float(vmax), 5)
        if d is None or not (d / "models" / "final_model.zip").exists():
            miss.append(f"doppia v{vmax}")
    return miss


def download_file(url: str, dest: Path, progress_cb=None) -> None:
    """Scarica url in dest; progress_cb(frazione 0..1) se la dimensione è nota."""
    import urllib.request

    def hook(blocks, bsize, total):
        if progress_cb and total > 0:
            progress_cb(min(1.0, blocks * bsize / total))

    urllib.request.urlretrieve(url, str(dest), reporthook=hook)


def extract_models_tarball(tar_path: Path, root_dir: Path) -> None:
    """Estrae solo trained_models/** dal tarball del repo (entry del tipo
    'HYDRAS-Project-master/trained_models/…') in root_dir/trained_models."""
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) == 2 and parts[1].startswith("trained_models/"):
                m.name = parts[1]                 # rimuove il prefisso top-level
                tf.extract(m, path=str(root_dir))


def download_and_extract_data(url: str, root_dir: Path, progress_cb=None) -> None:
    """Scarica l'archivio dati (.zip o .tar.gz) da url e lo estrae in data/."""
    import tempfile, tarfile, zipfile
    (root_dir / "data").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dl") as tmp:
        tmp_path = Path(tmp.name)
    try:
        download_file(url, tmp_path, progress_cb)
        if zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path) as z:
                z.extractall(root_dir / "data")
        elif tarfile.is_tarfile(tmp_path):
            with tarfile.open(tmp_path) as t:
                t.extractall(root_dir / "data")
        else:
            raise RuntimeError("Formato archivio dati non riconosciuto (atteso .zip o .tar.gz).")
    finally:
        tmp_path.unlink(missing_ok=True)


def held_out_sources(dm: DataManager) -> list:
    """Sorgenti mai viste in addestramento (SRC107-SRC132)."""
    return sorted(s for s in dm.get_discovered_sources() if int(s[3:]) > 106)


def pick_source(dm: DataManager, version: str, rng: np.random.Generator) -> Optional[str]:
    """Sorgente held-out a caso che possiede un file per la versione data."""
    sources = held_out_sources(dm)
    for s in rng.permutation(sources):
        if any(version in f.name and s in f.name for f in dm._nc_files):
            return str(s)
    return None


def load_field(dm: DataManager, source_id: str, version: str):
    files = [f for f in dm._nc_files if version in f.name and source_id in f.name]
    if not files:
        return None
    field = dm._nc_loader.load(str(files[0]),
                               concentration_var="Concentration - component 1")
    if field is None:
        return None
    coords = dm.get_source_coordinates(source_id)
    if coords:
        field.source_position = coords
    field.run_id = f"{source_id}_{version}"
    return field


# ─── Motore (indipendente dalla GUI, testabile headless) ─────────────────────

def make_agent_env(dm: DataManager, root_dir: Path, tech: str, version: str,
                   chunk: int, vmax: int, formation: str, source: str) -> Tuple:
    """Costruisce (agente, vec_env, is_fcm, label) per la scelta dell'utente.

    - PPO: trova il run (singola/doppia corona, v_max), carica modello +
      VecNormalize e usa la config del modello (obs coerente col modello).
    - FCM: AdamFCMAgent(lr=40, sensor_range=50), env senza VecNormalize
      (obs grezze) dalla config base senza reward vento.
    """
    field = load_field(dm, source, version)
    if field is None:
        raise RuntimeError(f"Campo non caricabile per {source} {version}.")

    if tech == "FCM":
        config = load_config(str(root_dir / "utils" / "config_base_no_wind_reward.yaml"))
        env_cfg = make_env_config(config, chunk_id=chunk)
        vec_env = build_env_fcm(env_cfg, field, use_masking=MASKABLE_PPO_AVAILABLE,
                                data_manager=dm, wind_mapping=WIND_MAPPING,
                                current_mapping=CURRENT_MAPPING)
        agent = AdamFCMAgent(sensor_range=FCM_SENSOR_RANGE, lr=FCM_LR)
        agent.reset()
        label = f"FCM Adam (lr={int(FCM_LR)} m)"
        return agent, vec_env, True, label

    # PPO ---------------------------------------------------------------------
    trained = root_dir / "trained_models"
    if formation == "Doppia":
        run_dir = _find_dualcorona_run(trained, vmax=float(vmax), K=5)
        form_lbl = "doppia corona"
    else:
        run_dir = _find_velocity_run(trained, vmax=int(vmax), K=5)
        form_lbl = "singola corona"
    if run_dir is None:
        raise RuntimeError(f"Nessun modello PPO {form_lbl} v_max={vmax}.")

    model_path = run_dir / "models" / "final_model.zip"
    vec_norm_path = run_dir / "models" / "vec_normalize.pkl"
    if not model_path.exists():
        raise RuntimeError(f"final_model.zip mancante in {run_dir/'models'}.")

    config = load_config(str(run_dir / "config.yaml"))
    env_cfg = make_env_config(config, chunk_id=chunk)
    vec_env = build_env(env_cfg, field, vec_norm_path,
                        use_masking=MASKABLE_PPO_AVAILABLE, data_manager=dm,
                        wind_data=None, current_data=None,
                        wind_mapping=WIND_MAPPING, current_mapping=CURRENT_MAPPING)
    model = load_model(str(model_path))
    label = f"PPO {form_lbl}, v_max={vmax}"
    return model, vec_env, False, label


def step_once(agent, vec_env, obs, is_fcm) -> Tuple[np.ndarray, bool, dict]:
    """Un passo: predice, (FCM: aggiorna la velocità adattiva), avanza.

    Ritorna (nuova_obs, done, info). Replica la meccanica di run_episode /
    run_episode_fcm di inference.py.
    """
    inner = get_inner_env(vec_env)
    masks = vec_env.env_method("action_masks")[0] if MASKABLE_PPO_AVAILABLE else None
    if masks is not None:
        action, _ = agent.predict(obs, deterministic=True, action_masks=masks)
    else:
        action, _ = agent.predict(obs, deterministic=True)

    if is_fcm:
        # Passo adattivo Adam: la velocità del passo corrente esce dall'ottimizzatore.
        inner.config.max_velocity = agent._last_step / inner.config.dt

    obs, _, dones, infos = vec_env.step(action)
    return obs, bool(dones[0]), infos[0]


def resolve_termination(info: dict) -> str:
    t = info.get("termination_reason")
    if t in {"success", "boundary", "land", "timeout"}:
        return t
    if info.get("source_reached", False):
        return "success"
    if info.get("on_land", False):
        return "land"
    if info.get("out_of_bounds", False):
        return "boundary"
    return "timeout"


def draw_scene(ax, inner, title: str) -> None:
    """Disegna la scena corrente nell'axes dato, con lo stesso stile dei plot
    standard del progetto (mare azzurro, terra bianca, plume YlOrRd) — cfr.
    plot_trajectory in inference.py. Nessuna dipendenza da pyplot."""
    ax.clear()
    field = inner.field
    conc = field.get_current_field()
    xc, yc = field.x_coords, field.y_coords

    # Extent con correzione di mezzo pixel (i coords sono centri cella).
    dx = float(xc[1] - xc[0]) if len(xc) > 1 else 10.0
    dy = float(yc[1] - yc[0]) if len(yc) > 1 else 10.0
    extent = [float(xc[0]) - dx / 2, float(xc[-1]) + dx / 2,
              float(yc[0]) - dy / 2, float(yc[-1]) + dy / 2]

    # 1) Mare (sfondo azzurro uniforme); 2) terra bianca via land_mask.
    ax.set_facecolor("#87CEEB")
    land_mask = getattr(field, "land_mask", None)
    if land_mask is not None:
        land = np.ma.masked_where(~land_mask, np.ones_like(conc))
        ax.imshow(land, origin="lower", extent=extent,
                  cmap=ListedColormap(["#FFFFFF"]), alpha=1.0, zorder=1)
        plume_mask = land_mask | (conc < 0.01)
    else:
        plume_mask = conc < 0.01

    # 3) Plume di concentrazione (mascherato su terra e dove conc ~ 0).
    conc_masked = np.ma.masked_where(plume_mask, conc)
    ax.imshow(conc_masked, origin="lower", extent=extent, cmap="YlOrRd",
              alpha=0.9, vmin=0, vmax=max(float(conc.max()), 0.1), zorder=2)

    # Traiettoria dell'agente.
    if len(inner.trajectory) > 1:
        traj = np.array(inner.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="#1f4fd6", linewidth=1.8,
                alpha=0.9, label="Traiettoria", zorder=4)

    # Agente + freccia di direzione.
    ax.scatter(inner.state.x, inner.state.y, c="#0b3d91", s=90, marker="o",
               edgecolors="white", linewidths=1.2, zorder=6, label="Agente")
    if inner.state.vx != 0 or inner.state.vy != 0:
        ax.arrow(inner.state.x, inner.state.y, inner.state.vx * 50, inner.state.vy * 50,
                 head_width=30, head_length=20, fc="#0b3d91", ec="#0b3d91", zorder=5)

    # Sorgente (stella gialla) e raggio di successo.
    sx, sy = inner.source_position
    ax.scatter(sx, sy, c="yellow", s=220, marker="*", edgecolors="black",
               zorder=7, label="Sorgente")
    ax.add_patch(Circle((sx, sy), inner.config.source_distance_threshold,
                        fill=False, color="red", linestyle="--", zorder=7))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(float(xc[0]), float(xc[-1]))
    ax.set_ylim(float(yc[0]), float(yc[-1]))
    ax.legend(loc="upper right", fontsize=8)


# ─── GUI ─────────────────────────────────────────────────────────────────────

def run_gui(fps: float) -> None:
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root_dir = Path(__file__).resolve().parent.parent
    rng = np.random.default_rng()
    delay_ms = int(1000.0 / max(fps, 1.0))

    root = tk.Tk()
    root.title("HYDRAS — Simulazione live")

    ctrl = ttk.Frame(root, padding=10)
    ctrl.grid(row=0, column=0, sticky="ew")

    tech_var = tk.StringVar(value=DEFAULTS["tech"])
    ver_var = tk.StringVar(value=DEFAULTS["version"])
    chunk_var = tk.StringVar(value=DEFAULTS["chunk"])
    vmax_var = tk.StringVar(value=DEFAULTS["vmax"])
    form_var = tk.StringVar(value=DEFAULTS["formation"])

    def combo(parent, label, var, values, col):
        ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="w", padx=(8, 2))
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly",
                          width=9)
        cb.grid(row=0, column=col * 2 + 1, padx=(0, 6))
        return cb

    tech_cb = combo(ctrl, "Tecnologia", tech_var, ["PPO", "FCM"], 0)
    ver_cb = combo(ctrl, "V", ver_var, VERSIONS, 1)
    chunk_cb = combo(ctrl, "Q", chunk_var, list(CHUNK_BY_LABEL.keys()), 2)
    vmax_cb = combo(ctrl, "v_max", vmax_var, ["1", "2", "3", "4", "5"], 3)
    form_cb = combo(ctrl, "Formazione", form_var, ["Singola", "Doppia"], 4)

    # Centro: la simulazione live occupa la maggior parte della finestra. Il canvas
    # viene messo in griglia solo a caricamento completato (finalize_loading);
    # durante il caricamento la stessa cella (row 1) ospita la schermata di attesa.
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    def show_idle():
        """Schermata inerte (nessuna simulazione): identica all'avvio e dopo il
        reset. Imposta il facecolor in modo esplicito perché ax.clear() NON lo
        ripristina — senza questo, dopo un episodio lo sfondo resterebbe blu."""
        ax.clear()
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Nessuna simulazione in corso")
        canvas.draw_idle()

    # Basso: pulsanti Avvia / Annulla.
    btns = ttk.Frame(root, padding=(10, 6))
    btns.grid(row=2, column=0, sticky="ew")
    start_btn = ttk.Button(btns, text="Avvia")
    start_btn.grid(row=0, column=0, padx=(0, 8))
    cancel_btn = ttk.Button(btns, text="Annulla")
    cancel_btn.grid(row=0, column=1)

    status = ttk.Label(root, text="Pronto — scegli le opzioni e premi Avvia.",
                       padding=(10, 4))
    status.grid(row=3, column=0, sticky="w")

    # I widget PPO-only compaiono solo quando Tecnologia == PPO.
    ppo_widgets = [w for w in ctrl.grid_slaves()
                   if int(w.grid_info()["column"]) in (6, 7, 8, 9)]

    def sync_ppo_visibility(*_):
        show = tech_var.get() == "PPO"
        for w in ppo_widgets:
            (w.grid() if show else w.grid_remove())

    tech_cb.bind("<<ComboboxSelected>>", sync_ppo_visibility)

    state = {"vec_env": None, "obs": None, "agent": None, "is_fcm": False,
             "label": "", "running": False, "steps": 0, "dm": None}

    def set_controls(enabled: bool):
        st = "readonly" if enabled else "disabled"
        for cb in (tech_cb, ver_cb, chunk_cb, vmax_cb, form_cb):
            cb.configure(state=st)
        start_btn.configure(state=("normal" if enabled else "disabled"))

    def cleanup_env():
        if state["vec_env"] is not None:
            try:
                state["vec_env"].close()
            except Exception:
                pass
            state["vec_env"] = None

    def reset_program():
        """Ripristina tutte le configurazioni ai default e riabilita i controlli."""
        cleanup_env()
        state.update(running=False, obs=None, agent=None, steps=0)
        tech_var.set(DEFAULTS["tech"]); ver_var.set(DEFAULTS["version"])
        chunk_var.set(DEFAULTS["chunk"]); vmax_var.set(DEFAULTS["vmax"])
        form_var.set(DEFAULTS["formation"])
        sync_ppo_visibility()
        set_controls(True)

    def finish(info: dict):
        outcome = resolve_termination(info)
        inner = get_inner_env(state["vec_env"])
        draw_scene(ax, inner, f"{state['label']}  —  {outcome.upper()} "
                              f"in {state['steps']} step")
        canvas.draw_idle()
        # Fine naturale: MANTIENE le configurazioni scelte (non le resetta); chiude
        # solo l'ambiente e riabilita i controlli, lasciando l'ultimo frame a schermo.
        cleanup_env()
        state.update(running=False, obs=None, agent=None, steps=0)
        set_controls(True)
        status.configure(text=f"Episodio terminato: {outcome.upper()}. "
                              f"Configurazioni mantenute — premi Avvia per rieseguire.")

    def step():
        if not state["running"]:
            return
        try:
            obs, done, info = step_once(state["agent"], state["vec_env"],
                                        state["obs"], state["is_fcm"])
        except Exception as e:
            status.configure(text=f"Errore durante la simulazione: {e}")
            reset_program()
            return
        state["obs"] = obs
        state["steps"] += 1
        inner = get_inner_env(state["vec_env"])
        draw_scene(ax, inner, f"{state['label']}  —  step {state['steps']}")
        canvas.draw_idle()
        if done:
            state["running"] = False
            finish(info)
        else:
            root.after(delay_ms, step)

    def start():
        tech = tech_var.get()
        version = ver_var.get()
        chunk = CHUNK_BY_LABEL[chunk_var.get()]
        vmax = int(vmax_var.get())
        formation = form_var.get()

        source = pick_source(state["dm"], version, rng)
        if source is None:
            status.configure(text=f"Nessuna sorgente held-out con versione {version}.")
            return

        set_controls(False)
        status.configure(text=f"Caricamento {tech} … scenario {source} {version} "
                              f"{chunk_var.get()}")
        root.update_idletasks()

        try:
            agent, vec_env, is_fcm, label = make_agent_env(
                state["dm"], root_dir, tech, version, chunk, vmax, formation, source)
        except Exception as e:
            status.configure(text=f"Errore nel caricamento: {e}")
            set_controls(True)
            return

        state.update(vec_env=vec_env, agent=agent, is_fcm=is_fcm,
                     label=f"{label} · {source} {version} {chunk_var.get()}",
                     steps=0, running=True)
        state["obs"] = vec_env.reset()
        status.configure(text=f"In esecuzione: {state['label']}")
        root.after(delay_ms, step)

    def cancel():
        """Interrompe la simulazione (se in corso) e ripristina le configurazioni.
        Non chiude il programma: l'end state è sempre la simulazione interrotta."""
        was_running = state["running"]
        show_idle()
        reset_program()
        status.configure(text=("Simulazione annullata. " if was_running else "")
                              + "Configurazioni ripristinate — pronto per una nuova esecuzione.")

    start_btn.configure(command=start)
    cancel_btn.configure(command=cancel)

    def on_close():
        state["running"] = False
        cleanup_env()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    sync_ppo_visibility()

    # ── Schermata di caricamento a fasi ──────────────────────────────────────
    # La finestra appare subito; le fasi (requisiti → dati → modelli → ambiente)
    # girano in un thread e aggiornano didascalia + barra. Le fasi 1-3 sono
    # check-e-scarica (progresso reale se c'è un download); la fase 4 (DataManager,
    # ~13 s, opaca) è animata a tempo dal poll.
    set_controls(False)
    cancel_btn.configure(state="disabled")

    loading = ttk.Frame(root, padding=40)
    loading.grid(row=1, column=0, sticky="nsew")
    cap_label = ttk.Label(loading, text="Avvio…", font=("", 13))
    cap_label.pack(pady=(80, 14))
    pbar = ttk.Progressbar(loading, mode="determinate", maximum=100, length=380)
    pbar.pack()
    pct_label = ttk.Label(loading, text="0%")
    pct_label.pack(pady=(8, 0))
    status.configure(text="Avvio in corso…")

    loader = {"dm": None, "error": None, "done": False,
              "caption": "Avvio…", "pct": 0.0, "phase4_start": None}

    def load_worker():
        try:
            # Fase 1 — requisiti
            loader["caption"] = "Installazione requisiti…"; loader["pct"] = 3
            if missing_packages():
                install_requirements(root_dir)
                still = missing_packages()
                if still:
                    raise RuntimeError("Requisiti mancanti: " + ", ".join(still))
            loader["pct"] = 10

            # Fase 2 — dati .nc
            loader["caption"] = "Caricamento dei dati .nc…"
            if not data_present(root_dir):
                if not DATA_ARCHIVE_URL:
                    raise RuntimeError(
                        "Dati .nc mancanti in data/ e nessun URL configurato. "
                        "Imposta DATA_ARCHIVE_URL in src/live_sim.py con il link "
                        "all'archivio dati.")
                download_and_extract_data(DATA_ARCHIVE_URL, root_dir,
                                          lambda f: loader.__setitem__("pct", 10 + 35 * f))
            loader["pct"] = 45

            # Fase 3 — modelli PPO (dal repo GitHub, se mancanti)
            loader["caption"] = "Caricamento dei modelli PPO…"
            if missing_models(root_dir):
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
                    tar_path = Path(tmp.name)
                try:
                    download_file(MODELS_TARBALL_URL, tar_path,
                                  lambda f: loader.__setitem__("pct", 45 + 15 * f))
                    extract_models_tarball(tar_path, root_dir)
                finally:
                    tar_path.unlink(missing_ok=True)
                if missing_models(root_dir):
                    raise RuntimeError("Modelli PPO ancora mancanti dopo il download.")
            loader["pct"] = 62

            # Fase 4 — ambiente di simulazione (DataManager, ~13 s opachi)
            loader["caption"] = "Caricamento ambiente di simulazione…"
            loader["phase4_start"] = time.perf_counter()
            loader["dm"] = DataManager(data_dir=str(root_dir / "data"),
                                       preload_all=False,
                                       sources_csv="Coordinate_Sorgenti_FaseII.csv")
        except Exception as e:                 # marshallato nel poll (thread principale)
            loader["error"] = e
        finally:
            loader["done"] = True

    def finalize_loading():
        state["dm"] = loader["dm"]
        loading.destroy()
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        show_idle()
        set_controls(True)
        cancel_btn.configure(state="normal")
        status.configure(text="Pronto — scegli le opzioni e premi Avvia.")

    phase4_est = 13.0

    def poll_loading():
        if loader["error"] is not None:
            pct_label.configure(text="")
            cap_label.configure(text="Errore all'avvio")
            status.configure(text=f"Errore all'avvio: {loader['error']}")
            return
        if loader["done"]:
            pbar["value"] = 100
            pct_label.configure(text="100%")
            cap_label.configure(text=loader["caption"])
            root.after(200, finalize_loading)     # mostra brevemente il 100%
            return
        if loader["phase4_start"] is not None:
            # Fase 4 opaca: anima la barra a tempo tra 62% e 99%.
            el = time.perf_counter() - loader["phase4_start"]
            pct = 62.0 + min(37.0, 37.0 * el / phase4_est)
        else:
            pct = loader["pct"]
        pbar["value"] = pct
        pct_label.configure(text=f"{int(pct)}%")
        cap_label.configure(text=loader["caption"])
        root.after(100, poll_loading)

    threading.Thread(target=load_worker, daemon=True).start()
    root.after(100, poll_loading)

    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulazione live HYDRAS con GUI")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="frame al secondo della visualizzazione (default 15)")
    args = ap.parse_args()
    run_gui(args.fps)


if __name__ == "__main__":
    main()
