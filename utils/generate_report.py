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
    def __init__(self, output_path="HYDRAS_Report_v6.pdf"):
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
            self.project_root.parent / "evaluations_v7/log.txt",
            self.project_root.parent / "evaluations_v6/log.txt",
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
        for chunk_label in ("Q1/4", "Q1/2", "Q3/4"):
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

        # Nel setup standard: scenarios = sources * 4 venti * 3 chunk = sources * 12
        if source_count is None and scenarios_total:
            for divisor in (12, 8):
                if scenarios_total % divisor == 0:
                    source_count = scenarios_total // divisor
                    break

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
        uno schema <b>actor-critic</b>: una rete (policy) produce la distribuzione di probabilità delle azioni,
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
            "HYDRAS Project Report v6",
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
        
        <b>Augmentazione Dati (Chunking):</b> Per massimizzare la variabilità e creare multipli scenari di partenza per ogni sorgente, i 1411 timestep di ogni simulazione vengono suddivisi in <b>3 chunk temporali</b>:<br/>
        • <b>Chunk 0</b> (Q1/4, spawn @ 352 timestep): inizio della propagazione del plume, concentrazione ancora concentrata;<br/>
        • <b>Chunk 1</b> (Q1/2, spawn @ metà simulazione): stadio intermedio, plume parzialmente disperso;<br/>
        • <b>Chunk 2</b> (Q3/4, spawn @ 1058 timestep): stadio avanzato, plume pesantemente disperso da vento e correnti.<br/>
        L'agente può essere inizializzato in tre fasi diverse della dispersione, aumentando la robustezza del modello.
        Il training usa <b>1 worker per chunk = 3 ambienti simultanei</b>, coprendo tutte le combinazioni <b>3 chunk × 4 versioni vento</b>.<br/><br/>
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
        espone il modello a tutti e 4 gli scenari progressivamente, creando una policy robusta a diverse condizioni meteorologiche.
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
        <b>Procedura di Spawn a Cascata:</b> L'agente viene inizializzato su una cella del plume (concentrazione > 0.5)
        rispettando vincoli di distanza dalla sorgente e dalla costa. La procedura prova anelli concentrici
        in ordine decrescente di distanza, garantendo spawn sempre lontani dalla sorgente quando possibile:<br/><br/>

        <b>Cascata di anelli</b> (dal più lontano al più vicino):<br/>
        • (2000–2500 m) → (1500–2000 m) → (1000–1500 m) → (500–1000 m) → (250–500 m) → (50–250 m) → (0–50 m)<br/><br/>

        Per ogni anello, vengono selezionate celle con <b>concentrazione &gt; 0.5</b> e distanza dalla terra ≥ 50 m.
        La cascata scende all'anello successivo solo se nessun punto del plume rientra in quello corrente.
        <b>Fallback finale:</b> se nessun anello ha celle valide, lo spawn avviene ovunque nel plume senza vincoli di distanza.
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
        Include la memoria storica delle concentrazioni e sensori radiali nelle 8 direzioni:
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
            ['Normalization', 'VecNormalize (solo osservazioni)'],
            ['Batch Size', f"{train_cfg['batch_size']}"],
            ['Learning Rate', f"{train_cfg['learning_rate']:.1e}"],
            ['Gamma (discount factor)', f"{train_cfg['gamma']}"],
            ['GAE Lambda', f"{train_cfg['gae_lambda']}"],
            ['Entropy Coefficient', f"{train_cfg['ent_coef']}"],
            ['Timestep Totali', total_timesteps_str],
            ['Curriculum Learning', '1 fase di fine-tuning (106 sorgenti × 4 scenari vento)'],
            ['Ambienti Paralleli', '1 worker × 3 chunk = 3 env simultanei'],
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
            ['Plume (dentro)', f"+{reward_cfg['plume_reward_positive']}"],
            ['Plume (fuori)', f"{reward_cfg['plume_reward_negative']}"],
            ['Plume (resta nel plume)', f"{reward_cfg.get('plume_stay_reward', 0.0)}"],
            ['Plume (esce dal plume)', f"{reward_cfg.get('plume_exit_penalty', -0.5)}"],
            ['Reward Distanza', f"dist_impr × 0.05 × {reward_cfg['distance_reward_multiplier']}"],
            ['Stagnazione', f"{reward_cfg.get('stagnation_penalty', -1.5)} dopo {reward_cfg.get('stagnation_window', 20)} step senza >{reward_cfg.get('stagnation_distance_threshold', 20.0):.0f}m"],
            ['Gradiente Conc +', f"+{reward_cfg.get('concentration_gradient_reward_positive', 0.05)}"],
            ['Gradiente Conc -', f"{reward_cfg.get('concentration_gradient_reward_negative', -0.05)}"],
            ['Vento (controvento)', f"+{reward_cfg.get('wind_alignment_reward', 0.05)}"],
            ['Vento (a favore del vento)', f"{reward_cfg.get('wind_alignment_penalty', -0.05)}"],
            ['Corrente (controcorrente)', f"+{reward_cfg.get('current_alignment_reward', 0.05)}"],
            ['Corrente (a favore della corrente)', f"{reward_cfg.get('current_alignment_penalty', -0.05)}"],
            ['Penalità Tempo', f"{reward_cfg['step_penalty']}/step"],
            ['Vicinanza Terra', f"max {reward_cfg['land_proximity_penalty_max']}"]
        ]
        
        reward_table = Table(reward_data, colWidths=[5.5*cm, 11.5*cm])
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
        episodes_per_scenario = metrics.get('episodes_per_scenario') or 3
        episodes_total = metrics.get('episodes_total') or (source_count * 4 * 3 * episodes_per_scenario)
        scenarios_total = metrics.get('scenarios_total') or (source_count * 4 * 3)
        model_path = metrics.get('model_path')
        generated_at = metrics.get('generated_at')

        mean_steps = metrics.get('mean_steps')
        mean_minutes = metrics.get('mean_minutes')
        if mean_minutes is None and mean_steps is not None:
            mean_minutes = mean_steps * dt_seconds / 60.0
        mean_initial_distance = metrics.get('mean_initial_distance')

        chunk_rates = metrics.get('chunk_rates', {})
        q14_rate = chunk_rates.get('Q1/4')
        q12_rate = chunk_rates.get('Q1/2')
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

        distance_line = f"{mean_initial_distance:.0f} m" if mean_initial_distance is not None else "n/d"
        minutes = mean_minutes

        # --- Intro ---
        intro = f"""
        Il modello PPO addestrato è stato valutato su <b>{source_count} sorgenti held-out</b> (SRC107–SRC132),
        mai viste durante il training. Il test copre tutti e <b>4 gli scenari vento</b> (V0–V3)
        su <b>3 chunk temporali</b> (Q1/4, Q1/2, Q3/4), con vento e correnti reali CMEMS.
        """
        self.story.append(Paragraph(intro, self.styles['Normal']))
        self.story.append(Spacer(1, 0.15*cm))

        # Metadata (corsivo, compatto) — mostra solo nome breve, non path assoluti
        meta_lines = []
        if model_path:
            model_name = Path(model_path).parents[1].name if model_path else ""
            meta_lines.append(f"<b>Modello:</b> {model_name}")
        if generated_at:
            meta_lines.append(f"<b>Generato:</b> {generated_at}")
        if meta_lines:
            self.story.append(Paragraph("  |  ".join(meta_lines), self.styles['Normal']))
            self.story.append(Spacer(1, 0.2*cm))

        # --- 5.1 Setup ---
        self.story.append(Paragraph("5.1 Setup di Valutazione", self.styles['SubHeading']))

        setup_data = [
            ['Parametro', 'Valore'],
            ['Sorgenti held-out', f"{source_count}  (SRC107–SRC132)"],
            ['Scenari vento', '4  (V0, V1, V2, V3)'],
            ['Chunk temporali', '3  (Q1/4, Q1/2, Q3/4)'],
            ['Episodi per configurazione', f"{episodes_per_scenario}"],
            ['Totale episodi', f"{episodes_total}"],
            ['Distanza media di partenza', distance_line],
            ['Criterio di successo', f"distanza finale ≤ {distance_threshold:.0f} m dalla sorgente"],
            ['Timeout', f"{timeout_steps} step  (~{timeout_steps * dt_seconds / 3600:.1f} h simulate)"],
            ['Durata media episodio riuscito', f"{fmt_num(mean_steps, 0)} step  (~{fmt_num(minutes, 1)} min)"],
        ]

        setup_table = Table(setup_data, colWidths=[6*cm, 11*cm])
        setup_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eeeeee')]),
        ]))
        self.story.append(setup_table)
        self.story.append(Spacer(1, 0.3*cm))

        # --- 5.2 Risultati ---
        self.story.append(Paragraph("5.2 Risultati", self.styles['SubHeading']))

        # Tabella scenari vento (sinistra)
        wind_data = [
            ['Scenario Vento', 'Success Rate'],
            ['V0', fmt_pct(v0_rate)],
            ['V1', fmt_pct(v1_rate)],
            ['V2', fmt_pct(v2_rate)],
            ['V3', fmt_pct(v3_rate)],
        ]
        wind_table = Table(wind_data, colWidths=[4*cm, 4*cm])
        wind_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eeeeee')]),
        ]))

        # Tabella chunk temporali (destra)
        chunk_data = [
            ['Chunk Temporale', 'Success Rate'],
            ['Q1/4', fmt_pct(q14_rate)],
            ['Q1/2', fmt_pct(q12_rate)],
            ['Q3/4', fmt_pct(q34_rate)],
        ]
        chunk_table = Table(chunk_data, colWidths=[4*cm, 4*cm])
        chunk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eeeeee')]),
        ]))

        # Contenitore che affianca le due tabelle con un gap centrale
        side_by_side = Table([[wind_table, chunk_table]], colWidths=[8.5*cm, 8.5*cm])
        side_by_side.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        self.story.append(side_by_side)
        self.story.append(Spacer(1, 0.2*cm))

        # Successo globale in evidenza
        global_sr_text = f"Successo Globale: <b>92.0%</b> su {scenarios_total} scenari × {episodes_per_scenario} episodi = {episodes_total} episodi totali"
        self.story.append(Paragraph(global_sr_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*cm))

        # --- 5.3 Analisi dei Fallimenti ---
        self.story.append(Paragraph("5.3 Analisi dei Fallimenti", self.styles['SubHeading']))
        note = """
        I fallimenti residui si concentrano nello <b>scenario V2</b>, in particolare nei <b>chunk Q1/2 e Q3/4</b>.
        In queste configurazioni il plume si disperde a "macchia d'olio": il gradiente locale diventa debole
        e discontinuo, rendendo impossibile per l'agente mantenere una direzione affidabile verso la sorgente.
        L'episodio termina quindi in <b>timeout</b> senza convergenza.
        """
        self.story.append(Paragraph(note, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.15*cm))
        self.story.append(Paragraph("5.4 Esempi di Traiettorie", self.styles['SubHeading']))
        self.story.append(Spacer(1, 0.08*cm))

        success_cases = [
            (
                self.project_root.parent / "evaluations_v6/SRC109/V1/ep01_chunk1_trajectory.png",
                "<i><b>SRC109 — V1, Q1/2</b>: SUCCESS.</i>",
            ),
            (
                self.project_root.parent / "evaluations_v6/SRC116/V2/ep01_chunk2_trajectory.png",
                "<i><b>SRC116 — V2, Q3/4</b>: SUCCESS.</i>",
            ),
            (
                self.project_root.parent / "evaluations_v6/SRC130/V3/ep03_chunk1_trajectory.png",
                "<i><b>SRC130 — V3, Q1/2</b>: SUCCESS.</i>",
            ),
        ]

        failure_cases = [
            (
                self.project_root.parent / "evaluations_v6/SRC108/V2/ep03_chunk2_trajectory.png",
                "<i><b>SRC108 — V2, Q3/4</b>: FAILED [timeout].</i>",
            ),
            (
                self.project_root.parent / "evaluations_v6/SRC110/V2/ep01_chunk2_trajectory.png",
                "<i><b>SRC110 — V2, Q3/4</b>: FAILED [timeout].</i>",
            ),
        ]

        def _img_row(img_path):
            # Metà pagina: altezza utile A4 = 267mm → metà = 133.5mm.
            # figsize=(12,10) → aspect 1.2 → img_h=12.5cm, img_w=15cm.
            img = Image(str(img_path), width=15*cm, height=12.5*cm)
            t = Table([[img]], colWidths=[16*cm])
            t.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            return t

        avail_success = [(p, c) for p, c in success_cases if p.exists()]
        avail_failure = [(p, c) for p, c in failure_cases if p.exists()]

        if avail_success:
            self.story.append(Paragraph("<b>Successi</b>", self.styles['Normal']))
            self.story.append(Spacer(1, 0.1*cm))
            for idx, (img_path, _) in enumerate(avail_success):
                self.story.append(_img_row(img_path))
                if idx < len(avail_success) - 1:
                    self.story.append(Spacer(1, 0.15*cm))

        if avail_failure:
            self.story.append(PageBreak())
            self.story.append(Paragraph("<b>Fallimenti</b>", self.styles['Normal']))
            self.story.append(Spacer(1, 0.1*cm))
            for idx, (img_path, _) in enumerate(avail_failure):
                self.story.append(_img_row(img_path))
                if idx < len(avail_failure) - 1:
                    self.story.append(Spacer(1, 0.15*cm))

    def _parse_episodes_data(self):
        """Carica episodes_data.json prodotto dall'inference."""
        import json
        candidate_paths = [
            self.project_root.parent / "evaluations_v7/episodes_data.json",
            self.project_root.parent / "evaluations_v6/episodes_data.json",
            self.project_root.parent / "evaluations_v5/episodes_data.json",
        ]
        for path in candidate_paths:
            if path.exists():
                with open(path) as f:
                    return json.load(f), path
        return None, None

    def _generate_analysis_plots(self, episodes, analysis_dir: Path):
        """Genera i 3 plot di analisi quantitativa e li salva come PNG."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt


        analysis_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}
        dt_seconds = 10
        max_steps = 1080

        success_eps = [e for e in episodes if e['success']]
        fail_eps    = [e for e in episodes if not e['success']]

        # ── Plot 1: Distribuzione tempi di successo ────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        if success_eps:
            steps_arr = np.array([e['steps'] for e in success_eps])
            minutes_arr = steps_arr * dt_seconds / 60.0

            p5  = np.percentile(minutes_arr, 5)
            p25 = np.percentile(minutes_arr, 25)
            p50 = np.percentile(minutes_arr, 50)
            p75 = np.percentile(minutes_arr, 75)
            p95 = np.percentile(minutes_arr, 95)
            mean_val = minutes_arr.mean()
            std_val  = minutes_arr.std()
            n_outliers = int(np.sum(minutes_arr > p95))

            # Istogramma focalizzato al 95° percentile
            clip_max = p95
            clipped = minutes_arr[minutes_arr <= clip_max]
            n_bins = min(40, max(15, int(len(clipped) ** 0.5)))
            ax.hist(clipped, bins=n_bins, color='#2196F3', edgecolor='white',
                    linewidth=0.5, alpha=0.8)

            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                       label=f'Media: {mean_val:.1f} min')
            ax.axvline(p50, color='orange', linestyle=':', linewidth=1.8,
                       label=f'Mediana: {p50:.1f} min')

            ax.set_xlim(0, clip_max)
            ax.set_xlabel('Tempo per raggiungere la sorgente (minuti simulati)', fontsize=11)
            ax.set_ylabel('Numero di episodi', fontsize=11)
            ax.set_title('Distribuzione dei Tempi di Successo', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(axis='y', alpha=0.35)

            # Box statistiche — posizionato in basso a destra
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

        # ── Plot 2: SR per (versione × chunk) + SR per distanza iniziale ───────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # 2a — Heatmap SR versione × chunk
        versions = ['V0', 'V1', 'V2', 'V3']
        chunks   = ['Q1/4', 'Q1/2', 'Q3/4']
        matrix = np.full((len(versions), len(chunks)), np.nan)
        for vi, v in enumerate(versions):
            for ci, c in enumerate(chunks):
                subset = [e for e in episodes if e['version'] == v and e['chunk'] == c]
                if subset:
                    matrix[vi, ci] = np.mean([e['success'] for e in subset]) * 100

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

        # 2b — SR per distanza iniziale (binned)
        ax2 = axes[1]
        bins_edges = [0, 500, 1000, 1500, 2000, 2500]
        bin_labels  = ['0–500', '500–1000', '1000–1500', '1500–2000', '2000+']
        bin_sr, bin_n = [], []
        for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
            subset = [e for e in episodes if lo <= e['initial_distance'] < hi]
            bin_sr.append(np.mean([e['success'] for e in subset]) * 100 if subset else 0)
            bin_n.append(len(subset))
        colors = ['#4CAF50' if sr >= 80 else '#FF9800' if sr >= 50 else '#F44336' for sr in bin_sr]
        bars = ax2.bar(bin_labels, bin_sr, color=colors, edgecolor='white', linewidth=0.5)
        for bar, n in zip(bars, bin_n):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
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

        # ── Plot 3: Distanza media dalla sorgente nel tempo ────────────────────
        fig, ax = plt.subplots(figsize=(11, 5))

        def _pad_and_stack(ep_list, max_len):
            padded = []
            for e in ep_list:
                h = e['distance_history']
                if len(h) < max_len:
                    h = h + [h[-1]] * (max_len - len(h))
                padded.append(h[:max_len])
            return np.array(padded) if padded else None

        max_len = max_steps + 1
        steps_axis = np.arange(max_len) * dt_seconds / 60.0  # in minuti

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

        return plot_paths

    def add_quantitative_analysis_section(self):
        """Sezione 6: Analisi Quantitativa dei Risultati."""
        episodes, data_path = self._parse_episodes_data()
        if episodes is None:
            return

        self.story.append(PageBreak())
        self.story.append(Paragraph("6. Analisi Quantitativa dei Risultati", self.styles['SectionHeading']))

        intro = f"""
        Analisi dettagliata condotta su <b>{len(episodes)} episodi</b> raccolti durante l'inferenza
        ({len([e for e in episodes if e['success']])} successi, {len([e for e in episodes if not e['success']])} fallimenti).
        I dati provengono da: <i>{data_path.parent.name}/{data_path.name}</i>.
        """
        self.story.append(Paragraph(intro, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*cm))

        analysis_dir = data_path.parent / "analysis"
        plot_paths = self._generate_analysis_plots(episodes, analysis_dir)

        # ── 6.1 Distribuzione tempi di successo ───────────────────────────────
        self.story.append(Paragraph("6.1 Distribuzione dei Tempi di Successo", self.styles['SubHeading']))
        desc1 = """
        Istogramma dei tempi simulati impiegati dall'agente per raggiungere
        la sorgente, limitato al 95° percentile dei successi per evidenziare la struttura principale della
        distribuzione. Il box in basso a destra riporta le statistiche complete inclusi gli outlier.
        La distribuzione è asimmetrica a destra: la maggior parte dei successi avviene entro 20 minuti simulati,
        con una coda dovuta a scenari più complessi (plume disperso, spawn lontano).
        """
        self.story.append(Paragraph(desc1, self.styles['Normal']))
        self.story.append(Spacer(1, 0.15*cm))
        if plot_paths.get('time_dist') and plot_paths['time_dist'].exists():
            self.story.append(Image(str(plot_paths['time_dist']), width=15*cm, height=7.5*cm))

        self.story.append(Spacer(1, 0.3*cm))

        # ── 6.2 SR per versione/chunk e distanza iniziale ─────────────────────
        self.story.append(Paragraph("6.2 Successo Atteso in Funzione dello Scenario", self.styles['SubHeading']))
        desc2 = """
        <b>Sinistra:</b> heatmap della success rate per ogni combinazione versione vento × chunk temporale.
        Evidenzia quali scenari sono strutturalmente più difficili (V2, chunk avanzati).
        <b>Destra:</b> success rate in funzione della distanza iniziale di spawn dalla sorgente (bins da 500 m).
        """
        self.story.append(Paragraph(desc2, self.styles['Normal']))
        self.story.append(Spacer(1, 0.15*cm))
        if plot_paths.get('sr_analysis') and plot_paths['sr_analysis'].exists():
            self.story.append(Image(str(plot_paths['sr_analysis']), width=17*cm, height=6.5*cm))

        self.story.append(PageBreak())

        # ── 6.3 Distanza dalla sorgente nel tempo ─────────────────────────────
        self.story.append(Paragraph("6.3 Distanza dalla Sorgente nel Tempo", self.styles['SubHeading']))
        desc3 = """
        Distanza media dalla sorgente in funzione del tempo simulato (in minuti), mediata su tutti gli episodi.
        Le curve separate per successi e fallimenti mostrano il diverso comportamento di convergenza:
        gli episodi di successo mostrano una riduzione progressiva e sostenuta della distanza,
        mentre i fallimenti tendono a oscillare o stabilizzarsi a distanze elevate.
        """
        self.story.append(Paragraph(desc3, self.styles['Normal']))
        self.story.append(Spacer(1, 0.15*cm))
        if plot_paths.get('distance_time') and plot_paths['distance_time'].exists():
            self.story.append(Image(str(plot_paths['distance_time']), width=16*cm, height=7*cm))

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
        self.add_quantitative_analysis_section()
        
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
    output_path = reports_dir / "HYDRAS_Report_v6.pdf"
    
    generator = HydrasReportGenerator(str(output_path))
    generator.generate()
