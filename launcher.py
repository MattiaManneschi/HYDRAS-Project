#!/usr/bin/env python3
"""
HYDRAS Live Sim — Launcher / bootstrapper standalone.

Prepara una macchina "vuota" (con solo Python) a eseguire la simulazione live e
poi la avvia. Le fasi, nell'ordine:

    1) Requisiti   — scarica requirements.txt e fa `pip install`.
    2) Script .py  — scarica il codice del progetto (src/ + utils/) da GitHub.
    3) Dati+modelli — dati .nc (sottoinsieme live da Google Drive) e modelli PPO
                      (da GitHub).
    4) Avvio       — lancia src/live_sim.py (la GUI).

Il launcher usa SOLO la libreria standard (gdown viene installato in fase 1 e
importato in modo lazy solo in fase 3), così può girare prima che le dipendenze
esistano ed è impacchettabile in un exe piccolo:

    pyinstaller --onefile launcher.py

Ogni fase è protetta da un controllo di presenza: se scripts/dati/modelli sono
già sul posto (es. lanciandolo dentro il repo di sviluppo) la fase viene saltata
e NON sovrascrive nulla — il launcher è idempotente e non distruttivo.

Nota: richiede Python + pip sulla macchina. Non è un exe che congela torch
(scelta deliberata: scaricare .py implica avere un interprete Python).
"""

import importlib.util
import os
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# ─── Configurazione ──────────────────────────────────────────────────────────
REPO_SLUG = "MattiaManneschi/HYDRAS-Project"
BRANCH = "master"
TARBALL_URL = f"https://codeload.github.com/{REPO_SLUG}/tar.gz/refs/heads/{BRANCH}"
REQUIREMENTS_URL = f"https://raw.githubusercontent.com/{REPO_SLUG}/{BRANCH}/requirements.txt"
DATA_DRIVE_URL = "https://drive.google.com/drive/folders/1wk0zDNH4upq1giz7nMoOvjrgrj6W35rN"

# Directory di installazione: quella del launcher (o override via HYDRAS_HOME).
INSTALL_DIR = Path(os.environ.get("HYDRAS_HOME", Path(__file__).resolve().parent))

# Pacchetti che devono essere importabili (nomi di import, non di pip).
REQUIRED = ["torch", "stable_baselines3", "sb3_contrib", "gymnasium", "numpy",
            "scipy", "matplotlib", "netCDF4", "yaml", "gdown"]

import re
_SRC_CONC_RE = re.compile(r"SRC(\d+)_Conc_10mGrid\.nc$")


def log(msg: str) -> None:
    print(msg, flush=True)


# ─── Download HTTPS con fallback certificati ─────────────────────────────────
def _download(url: str, dest: Path, cb=None) -> None:
    """Scarica url in dest. Prova prima con verifica TLS (certifi se presente,
    altrimenti store di sistema); se fallisce la verifica del certificato — tipico
    del Python di python.org su macOS *prima* che certifi sia installato — ritenta
    senza verifica (solo per il bootstrap di file da host noti: GitHub)."""
    def ctx_verified():
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    req = urllib.request.Request(url, headers={"User-Agent": "HYDRAS-launcher"})
    for ctx in (ctx_verified(), ssl._create_unverified_context()):
        try:
            with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                read = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    read += len(chunk)
                    if cb and total > 0:
                        cb(read / total)
            return
        except ssl.SSLCertVerificationError:
            log("  [avviso] verifica certificato fallita, riprovo senza verifica…")
            continue
    raise RuntimeError(f"Download fallito: {url}")


def _extract_prefixes(tar_path: Path, root: Path, prefixes: tuple) -> int:
    """Estrae dal tarball le voci il cui path (tolto il prefisso top-level
    'HYDRAS-Project-<branch>/') inizia con uno dei prefissi dati. Ritorna quante
    voci sono state estratte."""
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            if any(rel.startswith(p) for p in prefixes):
                m.name = rel
                tf.extract(m, path=str(root))
                n += 1
    return n


# ─── Controlli di presenza (self-contained, senza il codice del progetto) ────
def scripts_present(root: Path) -> bool:
    return (root / "src" / "live_sim.py").exists() and \
           (root / "utils" / "data_loader.py").exists()


def data_present(root: Path) -> bool:
    d = root / "data"
    conc = list(d.glob("**/*Conc_10mGrid.nc"))
    cur = list(d.glob("**/*SRC000_U_V_10mGrid.nc"))
    wind = list(d.glob("**/CI_WIND_faseII_V*.txt"))
    return len(conc) > 0 and len(cur) >= 4 and len(wind) >= 4


def models_present(root: Path) -> bool:
    return len(list((root / "trained_models").glob("**/final_model.zip"))) > 0


def requirements_satisfied() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in REQUIRED)


# ─── Fasi ────────────────────────────────────────────────────────────────────
def phase_requirements(root: Path) -> None:
    log("\n[1/4] Requisiti")
    if requirements_satisfied():
        log("  già soddisfatti — salto.")
        return
    req = root / "requirements.txt"
    if not req.exists():
        log("  scarico requirements.txt…")
        _download(REQUIREMENTS_URL, req)
    log("  pip install -r requirements.txt (può richiedere qualche minuto)…")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)
    if not requirements_satisfied():
        missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
        raise RuntimeError("Requisiti ancora mancanti dopo pip: " + ", ".join(missing))


def _get_tarball(cache: dict) -> Path:
    """Scarica il tarball del repo una sola volta (riusato da script e modelli)."""
    if cache.get("path") is None:
        tmp = Path(tempfile.mkdtemp()) / "hydras.tar.gz"
        log("  scarico il codice del progetto da GitHub…")
        _download(TARBALL_URL, tmp, lambda f: None)
        cache["path"] = tmp
    return cache["path"]


def phase_scripts(root: Path, cache: dict) -> None:
    log("\n[2/4] Script .py")
    if scripts_present(root):
        log("  già presenti — salto (nessuna sovrascrittura).")
        return
    tar = _get_tarball(cache)
    n = _extract_prefixes(tar, root, ("src/", "utils/"))
    log(f"  estratti {n} file (src/ + utils/).")
    if not scripts_present(root):
        raise RuntimeError("Script del progetto mancanti dopo l'estrazione.")


def phase_data_models(root: Path, cache: dict) -> None:
    log("\n[3/4] Dati e modelli")

    # Modelli PPO (dal tarball del repo).
    if models_present(root):
        log("  modelli PPO già presenti — salto.")
    else:
        tar = _get_tarball(cache)
        n = _extract_prefixes(tar, root, ("trained_models/",))
        log(f"  estratti {n} file dei modelli.")
        if not models_present(root):
            raise RuntimeError("Modelli PPO mancanti dopo l'estrazione.")

    # Dati .nc (sottoinsieme live da Google Drive).
    if data_present(root):
        log("  dati .nc già presenti — salto.")
    else:
        _download_data(root)
        if not data_present(root):
            raise RuntimeError("Dati .nc mancanti dopo il download.")


def _needed_for_live(rel_path: str) -> bool:
    name = rel_path.rsplit("/", 1)[-1]
    m = _SRC_CONC_RE.search(name)
    if m:
        return int(m.group(1)) > 106
    if name.endswith("_U_V_10mGrid.nc"):
        return "SRC000" in name
    if name.startswith("CI_WIND_faseII_V"):
        return True
    return name == "Coordinate_Sorgenti_FaseII.csv"


def _download_data(root: Path) -> None:
    """Sottoinsieme live (~8 GB) dalla cartella Google Drive, file per file, con
    resume (salta i già scaricati). gdown è stato installato in fase 1."""
    import gdown
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    log("  elenco la cartella Google Drive…")
    entries = gdown.download_folder(url=DATA_DRIVE_URL, skip_download=True,
                                    quiet=True, use_cookies=False)
    wanted = [(getattr(e, "path", None), getattr(e, "id", None)) for e in (entries or [])]
    wanted = [(p, i) for p, i in wanted if p and i and _needed_for_live(p)]
    if not wanted:
        raise RuntimeError("Cartella Drive inaccessibile o vuota "
                           "(condivisa come 'Chiunque abbia il link'?).")
    total = len(wanted)
    log(f"  scarico {total} file (~8 GB)…")
    for i, (rel, fid) in enumerate(wanted, 1):
        dest = data_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.exists() and dest.stat().st_size > 0):
            gdown.download(id=fid, output=str(dest), quiet=True, resume=True)
        if i % 10 == 0 or i == total:
            log(f"    {i}/{total} file")


def phase_launch(root: Path) -> None:
    log("\n[4/4] Avvio della simulazione live…")
    script = root / "src" / "live_sim.py"
    if not script.exists():
        raise RuntimeError(f"{script} non trovato.")
    subprocess.run([sys.executable, str(script)], check=False)


def main() -> None:
    root = INSTALL_DIR
    log(f"HYDRAS Live Sim — launcher\nDirectory di installazione: {root}")
    cache: dict = {"path": None}
    try:
        phase_requirements(root)
        phase_scripts(root, cache)
        phase_data_models(root, cache)
    except Exception as e:
        log(f"\nERRORE durante il setup: {e}")
        sys.exit(1)
    finally:
        if cache.get("path"):
            try:
                cache["path"].unlink(missing_ok=True)
            except Exception:
                pass
    phase_launch(root)


if __name__ == "__main__":
    main()
