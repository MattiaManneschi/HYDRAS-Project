#!/usr/bin/env python3
"""
Generatore di Report PDF per il progetto HYDRAS Source Seeking
"""

import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    KeepTogether, PageTemplate, Frame
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from PIL import Image as PILImage


class HydrasReportGenerator:
    def __init__(self, output_path="HYDRAS_Report_v5.pdf"):
        self.output_path = output_path
        self.project_root = Path(__file__).parent
        self.styles = self._setup_styles()
        self.story = []
        self.latest_model = self._find_latest_model()

    def _setup_styles(self):
        """Configura gli stili per il report."""
        styles = getSampleStyleSheet()
        
        # Titolo principale
        styles.add(ParagraphStyle(
            name='TitleReport',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.black,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Sezioni
        styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=styles['Heading1'],
            fontSize=15,
            textColor=colors.black,
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        ))
        
        # Sottosezioni
        styles.add(ParagraphStyle(
            name='SubHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#404040'),
            spaceAfter=6,
            spaceBefore=6,
            fontName='Helvetica-Bold'
        ))
        
        # Stile per celle di tabella con testo formattato
        styles.add(ParagraphStyle(
            name='TableCellBold',
            fontSize=8,
            textColor=colors.whitesmoke,
            alignment=TA_CENTER
        ))
        
        return styles

    def _find_latest_model(self):
        """Trova il modello addestrato più recente."""
        models_dir = self.project_root.parent / "trained_models"
        all_models = sorted(models_dir.glob("ppo_*"), key=lambda p: p.name, reverse=True)
        return all_models[0] if all_models else None

    def _load_config(self):
        """Carica la configurazione del progetto."""
        config_path = self.project_root / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml non trovato in {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _parse_logs(self):
        """Parsa i log di inferenza (formato v5, con fallback retrocompatibile)."""
        import re

        # Formato corrente: evaluations_v5/log.txt
        # Fallback legacy: evaluations_v4/logs.txt
        candidate_paths = [
            self.project_root.parent / "evaluations_v5/log.txt",
            self.project_root.parent / "evaluations_v4/logs.txt",
        ]
        logs_path = next((p for p in candidate_paths if p.exists()), None)

        if logs_path is None:
            return {
                'log_path': None,
                'generated_at': None,
                'model_path': None,
                'source_count': None,
                'episodes_per_scenario': None,
                'episodes_total': None,
                'scenarios_total': None,
                'success_rate': None,
                'successful_episodes': None,
                'timeout_episodes': None,
                'mean_steps': None,
                'mean_minutes': None,
                'mean_initial_distance': None,
                'chunk_rates': {},
                'wind_rates': {},
            }

        with open(logs_path) as f:
            content = f.read()

        def extract_int(pattern):
            match = re.search(pattern, content, flags=re.MULTILINE)
            return int(match.group(1)) if match else None

        def extract_float(pattern):
            match = re.search(pattern, content, flags=re.MULTILINE)
            return float(match.group(1)) if match else None

        def extract_text(pattern):
            match = re.search(pattern, content, flags=re.MULTILINE)
            return match.group(1).strip() if match else None

        chunk_rates = {}
        for chunk_label in ("Q1/4", "Q3/4"):
            match = re.search(rf'^\s*-\s*{re.escape(chunk_label)}:\s*([\d.]+)%', content, flags=re.MULTILINE)
            if match:
                chunk_rates[chunk_label] = float(match.group(1))

        wind_rates = {}
        for version in ("V0", "V1", "V2", "V3"):
            match = re.search(rf'^\s*-\s*{version}:\s*([\d.]+)%', content, flags=re.MULTILINE)
            if match:
                wind_rates[version] = float(match.group(1))

        source_count = extract_int(r'Test Set:\s*(\d+)\s*held-out sources')
        scenarios_total = extract_int(r'Total Scenarios Evaluated:\s*(\d+)')
        episodes_total = extract_int(r'Total Episodes Evaluated:\s*(\d+)')
        episodes_per_scenario = extract_int(r'Episodes:\s*(\d+)\s*per scenario')

        if episodes_per_scenario is None and episodes_total and scenarios_total:
            if scenarios_total > 0:
                episodes_per_scenario = int(round(episodes_total / scenarios_total))

        # Nel setup standard: scenarios = sources * 4 venti * 2 chunk = sources * 8
        if source_count is None and scenarios_total and scenarios_total % 8 == 0:
            source_count = scenarios_total // 8

        mean_initial_distance = extract_float(r'Mean Initial Distance:\s*([\d.]+)\s*m')
        if mean_initial_distance is None:
            mean_initial_distance = extract_float(r'Average Initial Distance:\s*([\d.]+)\s*m')

        mean_steps = extract_float(r'Mean Success Steps:\s*([\d.]+)')
        if mean_steps is None:
            mean_steps = extract_float(r'Average Steps per Success Episode:\s*([\d.]+)\s*steps')

        mean_minutes = extract_float(r'Mean Success Steps:\s*[\d.]+\s*\(~([\d.]+)\s*min\)')
        if mean_minutes is None:
            mean_minutes = extract_float(r'Average Time per Success:\s*([\d.]+)\s*minutes')

        success_rate = extract_float(r'Global Success Rate:\s*([\d.]+)%')
        successful_episodes = extract_int(r'Successful Episodes:\s*(\d+)')
        timeout_episodes = extract_int(r'Timeout Episodes:\s*(\d+)')

        if successful_episodes is None and success_rate is not None and episodes_total is not None:
            successful_episodes = int(round((success_rate / 100.0) * episodes_total))
        if timeout_episodes is None and successful_episodes is not None and episodes_total is not None:
            timeout_episodes = episodes_total - successful_episodes

        return {
            'log_path': str(logs_path),
            'generated_at': extract_text(r'Generated at:\s*(.+)'),
            'model_path': extract_text(r'Model:\s*(.+)'),
            'source_count': source_count,
            'episodes_per_scenario': episodes_per_scenario,
            'episodes_total': episodes_total,
            'scenarios_total': scenarios_total,
            'success_rate': success_rate,
            'successful_episodes': successful_episodes,
            'timeout_episodes': timeout_episodes,
            'mean_steps': mean_steps,
            'mean_minutes': mean_minutes,
            'mean_initial_distance': mean_initial_distance,
            'chunk_rates': chunk_rates,
            'wind_rates': wind_rates,
        }

    def add_ppo_algorithm_section(self):
        """Sezione 1 (nuovo): Algoritmo PPO - dettagli teorici e implementazione."""
        self.story.append(Spacer(1, 0.5*cm))
        
        self.story.append(Paragraph("1. Algoritmo PPO (Proximal Policy Optimization)", self.styles['SectionHeading']))

        ppo_overview = """
        <b>Proximal Policy Optimization (PPO)</b> è un algoritmo di reinforcement learning <b>on-policy</b>, 
        appartenente alla famiglia dei <b>Policy Gradient Methods</b>. La black-box è strutturata secondo 
        schema <b>actor-critic</b>: una rete (policy) produce la distribuzione di probabilità delle azioni, 
        mentre una seconda rete (value function) stima il valore dello stato V(s).<br/><br/>

        <b>Stable-Baselines3 (SB3)</b> è una libreria open-source per reinforcement learning in Python, 
        costruita su PyTorch. Fornisce implementazioni standardizzate e affidabili dei principali algoritmi 
        (tra cui PPO), con API omogenee per training/inferenza, gestione dei callback, logging e ambienti 
        vettorizzati. In pratica, SB3 è il framework che incapsula la black-box algoritmica e ne rende 
        l'uso riproducibile e comparabile tra esperimenti.<br/><br/>

        In <b>Stable-Baselines3</b>, PPO è implementato con una policy neurale (tipicamente MLP) che elabora 
        l'osservazione e produce due uscite: <b>policy head</b> e <b>value head</b>. In termini architetturali, 
        il pattern generale è: input osservazionale, hidden layers fully-connected, output su spazio azioni 
        e stima del valore.<br/><br/>

        Il meccanismo chiave è il <b>clipped surrogate objective</b>: PPO massimizza il miglioramento della policy, 
        ma limita la deviazione rispetto alla policy precedente tramite clipping del rapporto 
        r_t(θ) = π_θ(a|s) / π_θ_old(a|s). Questo vincolo riduce update troppo aggressivi e aumenta la stabilità 
        dell'apprendimento.<br/><br/>

        L'obiettivo complessivo combina tre contributi:<br/>
        • <b>Policy loss (clipped)</b>: aumenta la probabilità delle azioni vantaggiose;<br/>
        • <b>Value loss</b>: allinea la stima V(s) ai ritorni osservati;<br/>
        • <b>Entropy bonus</b>: mantiene esplorazione, evitando policy premature/deterministiche.<br/>
        Una forma compatta è: <b>L = L_clip + c1·L_vf - c2·H(π)</b>.<br/><br/>

        PPO usa tipicamente <b>GAE (Generalized Advantage Estimation)</b> per stimare il vantaggio con buon 
        compromesso bias-varianza. In training la policy è stocastica (campionamento da π(a|s)); in inferenza 
        può essere resa deterministica scegliendo l'azione a probabilità massima.
        """
        self.story.append(Paragraph(ppo_overview, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))

    def add_title_page(self, config):
        """Aggiunge la pagina di titolo."""
        self.story.append(Spacer(1, 2*cm))
        
        title = Paragraph(
            "HYDRAS Project Report v5",
            self.styles['TitleReport']
        )
        self.story.append(title)
        
        self.story.append(Spacer(1, 1.5*cm))

    def add_data_acquisition_section(self, config):
        """Sezione 2 (rinumerata): Acquisizione dati — 132 sorgenti con 4 scenari vento."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("2. Dataset — 132 Sorgenti × 4 Scenari Vento", self.styles['SectionHeading']))
        
        self.story.append(Paragraph("2.1 Acquisizione e Gestione Dati", self.styles['SubHeading']))
        
        # Fonte dati
        text = """
        <b>Sorgente Dati:</b> I dati di concentrazione provengono da simulazioni <b>MIKE21</b> in formato NetCDF (1411 timestep, risoluzione 10m, griglia 300×250 celle). 
        Il modulo <b>data_loader.py</b> gestisce il caricamento automatico di <b>132 sorgenti</b> (SRC001-SRC132) × <b>4 scenari vento</b> (V0, V1, V2, V3): 
        legge le coordinate spaziali (x, y), temporali (time) e i valori di concentrazione da file NetCDF, costruendo un campo interpolabile per ogni sorgente. 
        Ogni sorgente ha coordinate geospaziali (UTM32N) caricate da <b>Coordinate_Sorgenti_FaseII.csv</b>.<br/><br/>
        
        <b>Dataset Split:</b> Dataset suddiviso in <b>80% training</b> (~106 sorgenti per scenario vento) 
        e <b>20% valutazione</b> (~26 sorgenti held-out per scenario vento), equamente distribuiti tra i 4 scenari V0-V3.
        Durante il training, il curriculum learning espande progressivamente il set di sorgenti e scenari vento disponibili.<br/><br/>
        
        <b>Augmentazione Dati (Chunking):</b> Per massimizzare la variabilità e creare multipli scenari di partenza per ogni sorgente, i 1411 timestep di ogni simulazione vengono suddivisi in <b>2 chunk temporali</b>:<br/>
        • <b>Chunk 0</b> (Q1/4, spawn @ 352 timestep): inizio della propagazione del plume, concentrazione ancora concentrata;<br/>
        • <b>Chunk 2</b> (Q3/4, spawn @ 1058 timestep): stadio avanzato, plume pesantemente disperso da vento e correnti.<br/>
        L'agente può essere inizializzato in fasi diverse della dispersione, aumentando la robustezza del modello. 
        Con <b>2 ambienti paralleli</b>, il training esplora combinazioni su <b>2 chunks × 4 versioni vento</b> (8 configurazioni), mantenendo 2 scenari attivi simultaneamente.<br/><br/>
        """
        self.story.append(Paragraph(text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("2.2 I Quattro Scenari Vento (V0, V1, V2, V3)", self.styles['SubHeading']))
        
        wind_scenarios = """
        Il progetto utilizza <b>4 versioni diverse di vento</b> (CI_WIND_faseII_V0.txt, V1.txt, V2.txt, V3.txt), 
        generati da simulazioni meteorologiche con parametri e risoluzioni differenti. Questo aumenta vastamente 
        la diversità dei dati di training, permettendo al modello di generalizzare su condizioni vento realistiche. 
        Ogni scenario ha il proprio file di vento (48 timestep) e le correnti oceaniche corrispondenti 
        (estratti da CL02_V0/V1/V2/V3_SRC000_U_V_10mGrid.nc). Durante il training, il curriculum learning 
        espone il modello a tutti e 4 gli scenari progressivamente, creando una policy robust a diverse condizioni meteorologiche.
        """
        self.story.append(Paragraph(wind_scenarios, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("2.3 Ottimizzazione: Caching In-Memory delle Correnti", self.styles['SubHeading']))
        
        caching_text = """
        Per evitare colli di bottiglia I/O durante il training, il modulo <b>data_loader.py</b> implementa 
        un <b>sistema di caching in-memory</b> per i dati di vento e corrente:<br/><br/>
        
        <b>Meccanismo di Cache:</b> Al startup del DataManager, il metodo <b>_preload_all_versions_cache()</b> 
        carica tutti e 4 i file di vento e corrente in RAM (dizionari Python):<br/>
        • <code>_cached_wind_data = {"V0": array, "V1": array, "V2": array, "V3": array}</code><br/>
        • <code>_cached_current_data = {"V0": array, "V1": array, "V2": array, "V3": array}</code><br/><br/>
        
        Durante il training, i metodi <b>get_wind_data_for_run()</b> e <b>get_current_data_for_run()</b> 
        estraggono i dati dalla cache lookup-by-version, evitando riletture disk. Questo accelera ~5-6× 
        il training rispetto a letture da disco: <b>~55K step/ora → ~166K step/ora</b>.<br/><br/>
        
        <b>Sincronizzazione Garantita:</b> La versione (V0/V1/V2/V3) viene estratta dal filename del file NC 
        (es. <code>CL02_V2_SRC042_Conc_10mGrid.nc</code> → V2), quindi è <b>impossibile un mismatch</b> 
        tra file di concentrazione, vento e corrente. Tutte le sorgenti di un scenario vento usano lo stesso file di vento/corrente.
        """
        self.story.append(Paragraph(caching_text, self.styles['Normal']))

    def add_environment_section(self, config):
        """Sezione 3 (rinumerata): Ambiente di simulazione."""
        self.story.append(Spacer(1, 0.3*cm))
        
        self.story.append(Paragraph("3. Ambiente di Simulazione", self.styles['SectionHeading']))
        
        text = """
        L'ambiente (<b>source_seeking_env.py</b>) è una griglia 300×250 celle (risoluzione 10m) che 
        rappresenta un dominio marino di 3×2.5 km nella baia di Cecina (coste toscane, UTM32N). 
        L'agente (AUV - Autonomous Underwater Vehicle) si muove con <b>8 azioni discrete</b>: 
        N, S, E, W, NE, SE, NW, SW. La velocità è <b>1 m/s</b> e il timestep <b>dt=10s</b>, 
        quindi ogni azione sposta l'agente di 10 metri (o ~7m per componente nelle direzioni diagonali). 
        L'episodio termina quando la sorgente viene raggiunta (distanza &lt; 50m) oppure dopo 1080 step (~3 ore simulate).<br/><br/>
        
        <b>Evoluzione Temporale:</b> Il campo di concentrazione evolve nel tempo: ogni 12 step dell'agente (~2 minuti reali) 
        il campo NetCDF avanza di 1 frame temporale. Vento e corrente sono sincronizzati in minuti reali lungo tutto l'episodio.
        """
        self.story.append(Paragraph(text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("3.1 Constraints di Spawn e Fallback Automatico", self.styles['SubHeading']))
        
        spawn_text = """
        <b>Procedura di Spawn:</b> L'agente viene inizializzato su una cella del plume (concentrazione > 0.5) 
        rispettando vincoli di distanza dalla sorgente e dalla costa:<br/><br/>
        
        <b>Vincolo primario:</b> Seleziona celle con concentrazione > 0.5 dentro l'intervallo 
        <b>500 ≤ distanza ≤ 1500 metri</b> dalla sorgente, mantenendo almeno 50 metri dalla terra.<br/><br/>
        
        <b>Fallback automatico:</b> Se non trova punti nel range primario (situazione frequente in scenari con plume disperso o costieri), 
        il codice applica rilassamenti progressivi dei vincoli di distanza, mantenendo il vincolo sul plume. 
        Gli step di rilassamento sono: <b>d ≥ 500m</b> → <b>d ≥ 375m</b> → <b>d ≥ 250m</b> → <b>d ≥ 0m</b>. 
        La posizione finale viene validata anche dopo il jitter spaziale per rispettare i vincoli attivi.
        """
        self.story.append(Paragraph(spawn_text, self.styles['Normal']))

    def add_training_section(self, config):
        """Sezione 4 (rinumerata): Training e Architettura."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("4. Training e Architettura", self.styles['SectionHeading']))
        
        # 3.a - Input della rete neurale
        self.story.append(Paragraph("4.1 Input della Rete Neurale — 112 Dimensioni", self.styles['SubHeading']))
        
        obs_text = """
        Lo spazio di osservazione è un vettore continuo a <b>112 dimensioni</b> (normalizzato via VecNormalize). 
        Includi memoria storica concentrazioni e sensori radiali nelle 8 direzioni:
        """
        self.story.append(Paragraph(obs_text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.3*cm))
        
        obs_data = [
            ['Componente', 'Dim', 'Descrizione'],
            ['Concentrazione Attuale', '1', 'Conc. posizione corrente (x, y)'],
            ['Memoria Conc. (Locale)', '9', 'Ultimi 9 timestep'],
            ['Storico Movimento', '18', 'Δx, Δy ultimi 9 step'],
            ['Sensori Radiali', '8', 'Conc. ±20m nelle 8 direzioni'],
            ['Memory Conc. Direzionali', '72', 'Conc. nelle 8 direzioni × 9 timestep'],
            ['Vento', '2', 'u, v [m/s]'],
            ['Corrente', '2', 'u, v [m/s]'],
        ]
        
        obs_table = Table(obs_data, colWidths=[4*cm, 1.5*cm, 8.5*cm])
        obs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (2, 0), (2, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eeeeee')]),
        ]))
        self.story.append(obs_table)
        
        self.story.append(Spacer(1, 0.2*cm))
        
        # 4.2 - Parametri di training
        self.story.append(Paragraph("4.2 Parametri di Training", self.styles['SubHeading']))
        
        train_cfg = config['training']
        total_timesteps = int(train_cfg.get('total_timesteps', 0))
        if total_timesteps >= 1_000_000:
            total_timesteps_str = f"{total_timesteps / 1_000_000:.1f}M ({total_timesteps:,})"
        else:
            total_timesteps_str = f"{total_timesteps:,}"

        train_data = [
            ['Parametro', 'Valore'],
            ['Architettura Policy', 'MLP (256-256 unità nascoste, attivazione Tanh)'],
            ['Normalization', 'VecNormalize (osservazioni e reward)'],
            ['Batch Size', f"{train_cfg['batch_size']}"],
            ['Learning Rate', f"{train_cfg['learning_rate']:.1e}"],
            ['Gamma (discount factor)', f"{train_cfg['gamma']}"],
            ['GAE Lambda', f"{train_cfg['gae_lambda']}"],
            ['Entropy Coefficient', f"{train_cfg['ent_coef']}"],
            ['Timestep Totali', total_timesteps_str],
            ['Curriculum Learning', '1 fase di fine-tuning (106 sorgenti × 4 scenari vento)'],
            ['Ambienti Paralleli', '2 worker × 2 chunk = 4 env simultanei'],
        ]
        
        train_table = Table(train_data, colWidths=[4.5*cm, 8.5*cm])
        train_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eeeeee')])
        ]))
        self.story.append(train_table)
        
        self.story.append(Spacer(1, 0.2*cm))
        
        # 4.3 - Funzione di reward
        self.story.append(Paragraph("4.3 Funzione di Reward", self.styles['SubHeading']))
        
        reward_cfg = config['environment']['reward']
        
        reward_data = [
            ['Componente', 'Valore/Descrizione'],
            ['Sorgente Raggiunta', f"+{reward_cfg['source_reached_bonus']} + time_bonus (max +50)"],
            ['Penalità Bordo', f"{reward_cfg['boundary_penalty']}"],
            ['Plume (dentro)', f"+{int(reward_cfg['plume_reward_positive'])}"],
            ['Plume (fuori)', f"{int(reward_cfg['plume_reward_negative'])}"],
            ['Reward Distanza', f"dist_impr × 5.0 × {reward_cfg['distance_reward_multiplier']}"],
            ['Gradiente Conc +', f"+{reward_cfg.get('concentration_gradient_reward_positive', 0.05)}"],
            ['Gradiente Conc -', f"{reward_cfg.get('concentration_gradient_reward_negative', -0.05)}"],
            ['Vento (controcorr.)', f"+{reward_cfg.get('wind_alignment_reward', 0.05)}"],
            ['Vento (a favore)', f"{reward_cfg.get('wind_alignment_penalty', -0.05)}"],
            ['Penalità Tempo', f"{reward_cfg['step_penalty']}/step"],
            ['Vicinanza Terra', f"max {reward_cfg['land_proximity_penalty_max']}"]
        ]
        
        reward_table = Table(reward_data, colWidths=[4.2*cm, 7.8*cm])
        reward_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eeeeee')])
        ]))
        self.story.append(reward_table)
        
        self.story.append(Spacer(1, 0.2*cm))
        
        # 4.4 - Risultati di training
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("4.4 Risultati del Training", self.styles['SubHeading']))
        
        # Aggiungi plot di training dal modello più recente
        latest_model = self._find_latest_model()
        if latest_model:
            plots_dir = latest_model / "plots"
            if (plots_dir / "training_loss.png").exists() and (plots_dir / "training_success_rate.png").exists():
                try:
                    self.story.append(Paragraph(f"<i>Modello: {latest_model.name}</i>", self.styles['Normal']))
                    self.story.append(Spacer(1, 0.2*cm))
                    
                    # Crea due immagini impilate verticalmente con dimensioni aumentate (full width)
                    img_loss = Image(str(plots_dir / "training_loss.png"), width=17*cm, height=11.3*cm)
                    img_sr = Image(str(plots_dir / "training_success_rate.png"), width=17*cm, height=11.3*cm)
                    
                    plot_table = Table([[img_loss], [img_sr]], colWidths=[17*cm])
                    plot_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 0.5*cm),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 0.5*cm)
                    ]))
                    self.story.append(plot_table)
                except Exception as e:
                    self.story.append(Paragraph(f"<i>Plot non disponibili: {str(e)}</i>", self.styles['Normal']))
            else:
                self.story.append(Paragraph(f"<i>Plot non trovati in {plots_dir}</i>", self.styles['Normal']))
        else:
            self.story.append(Paragraph("<i>Nessun modello addestrato trovato</i>", self.styles['Normal']))

    def add_inference_section(self, config):
        """Sezione 5 (rinumerata): Risultati delle inferenze su 26 sorgenti held-out."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("5. Risultati delle Inferenze", self.styles['SectionHeading']))
        
        # Carica statistiche dal log
        metrics = self._parse_logs()
        env_cfg = config.get('environment', {})
        reward_cfg = env_cfg.get('reward', {})
        dt_seconds = float(env_cfg.get('dt', 10))
        distance_threshold = float(reward_cfg.get('distance_threshold', 50))
        timeout_steps = int(env_cfg.get('max_episode_steps', 1080))

        source_count = metrics.get('source_count') or 26
        episodes_per_scenario = metrics.get('episodes_per_scenario') or 5
        episodes_total = metrics.get('episodes_total') or (source_count * 4 * 2 * episodes_per_scenario)
        scenarios_total = metrics.get('scenarios_total') or (source_count * 4 * 2)
        model_path = metrics.get('model_path')
        generated_at = metrics.get('generated_at')

        success_rate = metrics.get('success_rate')
        mean_steps = metrics.get('mean_steps')
        mean_minutes = metrics.get('mean_minutes')
        if mean_minutes is None and mean_steps is not None:
            mean_minutes = mean_steps * dt_seconds / 60.0
        mean_initial_distance = metrics.get('mean_initial_distance')

        chunk_rates = metrics.get('chunk_rates', {})
        q14_rate = chunk_rates.get('Q1/4')
        q34_rate = chunk_rates.get('Q3/4')

        wind_rates = metrics.get('wind_rates', {})
        v0_rate = wind_rates.get('V0')
        v1_rate = wind_rates.get('V1')
        v2_rate = wind_rates.get('V2')
        v3_rate = wind_rates.get('V3')

        def fmt_pct(value):
            return f"{value:.1f}%" if value is not None else "n/d"

        def fmt_num(value, digits=1):
            if value is None:
                return "n/d"
            return f"{value:.{digits}f}"

        distance_line = f"{mean_initial_distance:.0f} metri" if mean_initial_distance is not None else "n/d"
        
        # Calcola minutaggio da steps
        minutes = mean_minutes
        
        # Descrizione prima della tabella
        description = f"""
        Valutazione del modello PPO su <b>{source_count} sorgenti held-out</b> (SRC107-SRC132) non viste durante il training curriculum, 
        testato su tutti e <b>4 scenari vento</b> (V0, V1, V2, V3). 
        Test eseguito con <b>vento e correnti reali CMEMS</b>:<br/><br/>
        <b>Log inferenza:</b> {metrics.get('log_path') or 'n/d'}<br/>
        <b>Timestamp inferenza:</b> {generated_at or 'n/d'}<br/>
        <b>Modello valutato:</b> {model_path or 'n/d'}<br/><br/>
        <b>Setup Valutazione:</b><br/>
        • <b>{source_count} sorgenti</b> × <b>4 scenari vento</b> (V0, V1, V2, V3) × <b>2 chunk temporali</b> (Q1/4, Q3/4)<br/>
        • <b>{episodes_per_scenario} episodi</b> per configurazione (totale <b>{episodes_total}</b> episodi)<br/>
        • <b>Distanza media di partenza</b>: {distance_line}<br/>
        • <b>Success @ {distance_threshold:.0f}m</b>: distanza finale ≤ {distance_threshold:.0f}m dalla sorgente<br/>
        • <b>Timeout @ {timeout_steps} steps</b> (~{timeout_steps * dt_seconds / 3600:.1f} ore simulate)<br/><br/>
        • <b>Durata media episodio di successo</b>: {fmt_num(mean_steps, 1)} step (~{fmt_num(minutes, 1)} min)<br/><br/>
        
        <b>Risultati per Tipologia di Vento:</b><br/>
        • <b>V0:</b> {fmt_pct(v0_rate)} success rate<br/>
        • <b>V1:</b> {fmt_pct(v1_rate)} success rate<br/>
        • <b>V2:</b> {fmt_pct(v2_rate)} success rate<br/>
        • <b>V3:</b> {fmt_pct(v3_rate)} success rate<br/><br/>
        
        <b>Risultati per Frame Temporale:</b><br/>
        • <b>Q1/4:</b> {fmt_pct(q14_rate)} success rate<br/>
        • <b>Q3/4:</b> {fmt_pct(q34_rate)} success rate<br/>
        
        <b>Successo Globale:</b> <b>{fmt_pct(success_rate)}</b> (media su tutti i {scenarios_total} scenari)<br/>
        """
        self.story.append(Paragraph(description, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.25*cm))
        
        # Note sui fallimenti
        note = """
        <b>Analisi dei Fallimenti:</b> I fallimenti residui si concentrano ormai in alcuni casi dello <b>scenario V2</b>, 
        soprattutto nel <b>chunk Q3/4</b>. In queste configurazioni il plume si disperde a "macchia d'olio": 
        il gradiente locale diventa debole e discontinuo, quindi l'agente fatica a mantenere una direzione affidabile 
        verso la sorgente e può terminare in <b>timeout</b>.
        """
        self.story.append(Paragraph(note, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.08*cm))
        self.story.append(Paragraph("<b>Esempi di Fallimento V2 (Q3/4)</b>", self.styles['SubHeading']))
        self.story.append(Spacer(1, 0.04*cm))

        v2_failure_cases = [
            (
                self.project_root.parent / "evaluations_v5/SRC128/V2/ep01_chunk2_trajectory.png",
                "<i><b>SRC128 - Ep1</b>: FAILED [timeout], dist finale 1034m. Dispersione elevata e percorso erratico lontano dalla sorgente.</i>",
            ),
            (
                self.project_root.parent / "evaluations_v5/SRC122/V2/ep01_chunk2_trajectory.png",
                "<i><b>SRC122 - Ep1</b>: FAILED [timeout], dist finale 687m. Plume molto diffuso e traiettoria non convergente.</i>",
            ),
            (
                self.project_root.parent / "evaluations_v5/SRC131/V2/ep01_chunk2_trajectory.png",
                "<i><b>SRC131 - Ep1</b>: FAILED [timeout], dist finale 626m. Deriva fuori plume e perdita del gradiente utile.</i>",
            ),
        ]

        available_cases = [(img_path, caption) for img_path, caption in v2_failure_cases if img_path.exists()]

        if not available_cases:
            self.story.append(Paragraph("<i>Nessun plot di fallimento V2 disponibile in evaluations_v5.</i>", self.styles['Normal']))
        else:
            # 1) Primo caso subito sotto al titolo "Esempi di Fallimento"
            first_path, first_caption = available_cases[0]
            first_img = Image(str(first_path), width=12.8*cm, height=7.2*cm)
            first_table = Table([[first_img], [Paragraph(first_caption, self.styles['Normal'])]], colWidths=[18*cm])
            first_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            self.story.append(first_table)
            self.story.append(Spacer(1, 0.15*cm))

            # 2) Gli altri due casi nella pagina successiva, uno sotto l'altro
            remaining_cases = available_cases[1:3]
            if remaining_cases:
                self.story.append(PageBreak())
                self.story.append(Paragraph("<b>Altri Esempi di Fallimento V2 (Q3/4)</b>", self.styles['SubHeading']))
                self.story.append(Spacer(1, 0.2*cm))

                for idx, (img_path, caption) in enumerate(remaining_cases):
                    img = Image(str(img_path), width=14.8*cm, height=8.7*cm)
                    table = Table([[img], [Paragraph(caption, self.styles['Normal'])]], colWidths=[18*cm])
                    table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 4),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                        ('TOPPADDING', (0, 0), (-1, -1), 2),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ]))
                    self.story.append(table)
                    if idx < len(remaining_cases) - 1:
                        self.story.append(Spacer(1, 0.2*cm))

    def generate(self):
        """Genera il report PDF."""
        print("Generando Report HYDRAS...")
        
        # Carica configurazione
        config = self._load_config()
        
        # Costruisci il report
        self.add_title_page(config)
        self.add_ppo_algorithm_section()
        self.add_data_acquisition_section(config)
        self.add_environment_section(config)
        self.add_training_section(config)
        self.add_inference_section(config)
        
        # Genera PDF
        doc = SimpleDocTemplate(self.output_path, pagesize=A4,
                               rightMargin=15*mm, leftMargin=15*mm,
                               topMargin=15*mm, bottomMargin=15*mm)
        
        doc.build(self.story)
        
        print(f"✓ Report generato: {self.output_path}")


if __name__ == "__main__":
    # Salva il report nella cartella reports
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / "HYDRAS_Report_v5.pdf"
    
    generator = HydrasReportGenerator(str(output_path))
    generator.generate()
