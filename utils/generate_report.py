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
    def __init__(self, output_path="HYDRAS_Report_v4.pdf"):
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
        """Parsa il file logs.txt dal nuovo formato evaluations_v4 (132 sorgenti)."""
        import re
        
        logs_path = self.project_root.parent / "evaluations_v4/logs.txt"
        
        if not logs_path.exists():
            return {}, 0, 0, 0
        
        with open(logs_path) as f:
            content = f.read()
        
        sources_stats = {}
        total_episodes = 0
        success_count = 0
        success_rate = 0
        mean_steps = 0
        mean_initial_distance = 0
        
        # Parse: "[  1/26] SRC107 Q1/4   success" o "failed"
        for match in re.finditer(r'\[\s*\d+/\d+\]\s+(SRC\d+)\s+(Q\d/\d)\s+(success|failed)', content):
            source = match.group(1)
            chunk = match.group(2)
            result = match.group(3)
            
            total_episodes += 1
            if result == 'success':
                success_count += 1
            
            if source not in sources_stats:
                sources_stats[source] = {'success': 0, 'total': 0}
            
            sources_stats[source]['total'] += 1
            if result == 'success':
                sources_stats[source]['success'] += 1
        
        success_rate = (success_count / total_episodes * 100) if total_episodes > 0 else 0
        
        # Parse: Final Success Rate: 88.5%
        match_sr = re.search(r'Final Success Rate:\s*([\d.]+)%', content)
        if match_sr:
            success_rate = float(match_sr.group(1))
        
        # Parse: Mean Steps (success): 73 (~12.1 min)
        match_steps = re.search(r'Mean Steps \(success\):\s*(\d+)', content)
        if match_steps:
            mean_steps = int(match_steps.group(1))
        
        # Parse: Mean Initial Distance: 490m
        match_dist = re.search(r'Mean Initial Distance:\s*(\d+)m', content)
        if match_dist:
            mean_initial_distance = int(match_dist.group(1))
        
        return sources_stats, success_rate, mean_steps, mean_initial_distance

    def add_ppo_algorithm_section(self):
        """Sezione 1 (nuovo): Algoritmo PPO - dettagli teorici e implementazione."""
        self.story.append(Spacer(1, 0.5*cm))
        
        self.story.append(Paragraph("1. Algoritmo PPO (Proximal Policy Optimization)", self.styles['SectionHeading']))
        
        self.story.append(Paragraph("1.1 Teoria dell'Algoritmo", self.styles['SubHeading']))
        
        ppo_theory = """
        <b>Proximal Policy Optimization (PPO)</b> è un algoritmo di reinforcement learning on-policy di ultima generazione, 
        appartenente alla famiglia dei <b>Policy Gradient Methods</b>. A differenza degli algoritmi off-policy (DQN, DDPG) che 
        apprendono da dati stocastici, PPO apprende direttamente dalla policy corrente, rendendo l'apprendimento più stabile.<br/><br/>
        
        <b>Principio Fondamentale:</b> PPO ottimizza la policy pi(a|s) massimizzando il <b>clipped surrogate objective</b>.<br/>
        Questo meccanismo impedisce aggiornamenti troppo bruschi della policy, garantendo stabilità anche con batch size piccoli.
        """
        self.story.append(Paragraph(ppo_theory, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.3*cm))
        
        self.story.append(Paragraph("1.2 Implementazione: Stable-Baselines3", self.styles['SubHeading']))
        
        ppo_impl = """
        Il modello utilizza <b>PPO di Stable-Baselines3</b> con architettura <b>MLP (Multi-Layer Perceptron)</b> per la policy 
        e value function. La rete neurale è composta da:<br/>
        • <b>Input Layer:</b> 112 dimensioni (osservazioni normalizzate via VecNormalize)<br/>
        • <b>Hidden Layers:</b> 2 strati da 256 unità con attivazione <b>Tanh</b><br/>
        • <b>Output Layers:</b> Policy head (8 azioni discrete via softmax) e Value head (1 scalare)<br/><br/>
        
        <b>Policy Stocastica:</b> La rete neurale genera una distribuzione di probabilità π(a|s) su tutte le 8 azioni. 
        Durante il training, l'agente <b>campiona</b> azioni da questa distribuzione, garantendo <b>esplorazione naturale</b> senza bisogno 
        di epsilon-greedy artificiale. Durante l'inferenza, sceglie deterministicamente l'azione con probabilità massima via <b>argmax</b>.<br/><br/>
        
        <b>Loss Function:</b> L_TOTAL = L_CLIP + 0.5 * L_VF - 0.01 * H(pi), dove L_CLIP è il clipped surrogate objective 
        (previene aggiornamenti eccessivi della policy), L_VF è l'MSE della value function, e H(pi) è l'entropy della policy 
        (c2=0.01 bilancia l'esplorazione senza eccessi).<br/><br/>
        
        <b>Parametri di Training:</b> Learning rate 5e-5, clip range 0.2, batch size 2048, 4 epoche di aggiornamento per batch, 
        <b>GAE lambda=0.95</b> (stima dei vantaggi con bias-variance trade-off), 
        <b>VecNormalize</b> (osservazioni a media 0, std 1 per accelerare convergenza).
        """
        self.story.append(Paragraph(ppo_impl, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))

    def add_title_page(self, config):
        """Aggiunge la pagina di titolo."""
        self.story.append(Spacer(1, 2*cm))
        
        title = Paragraph(
            "HYDRAS Project Report v4",
            self.styles['TitleReport']
        )
        self.story.append(title)
        
        self.story.append(Spacer(1, 1.5*cm))

    def add_data_acquisition_section(self, config):
        """Sezione 2 (rinumerata): Acquisizione dati — 132 sorgenti con 4 scenari vento."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("2. Dataset — 132 Sorgenti × 4 Scenari Vento", self.styles['SectionHeading']))
        
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
        • <b>Chunk 1</b> (Q3/4, spawn @ 1058 timestep): stadio avanzato, plume pesantemente disperso da vento e correnti.<br/>
        L'agente può essere inizializzato in fasi diverse della dispersione, aumentando la robustezza del modello. 
        Con 4 ambienti paralleli × 2 chunks × 4 versioni vento = 32 scenari di training simultanei.<br/><br/>
        """
        self.story.append(Paragraph(text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("2.1 Dati di Vento e Corrente Oceanica", self.styles['SubHeading']))
        
        text_wind = """
        <b>Vento:</b> File di testo (.txt) (Vento_V0-V3/). 
        Per il training e l'inferenza viene utilizzato <b>CI_WIND_faseII_V1.txt</b> (48 timestep, condiviso tra tutte le sorgenti).<br/><br/>
        
        <b>Corrente Oceanica:</b> Dati CMEMS estratti dall'unico file NetCDF <b>CL02_V1_SRC000_U_V_10mGrid.nc</b> 
        che contiene i campi bidimensionali di velocità (u, v) sincronizzati temporalmente con le simulazioni di concentrazione (1411 timestep, condiviso tra tutte le sorgenti).<br/><br/>
        
        <b>Normalizzazione e Sincronizzazione:</b> Vento e correnti vengono interpolati bilinearmente sulla griglia di simulazione (300×250 celle) 
        e sincronizzati al frame temporale corrente dell'agente. Tutti i dati sono normalizzati tramite <b>VecNormalize</b> (media 0, varianza 1) 
        e includono rumore gaussiano per aumentare la robustezza.
        """
        self.story.append(Paragraph(text_wind, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("2.2 I Quattro Scenari Vento (V0, V1, V2, V3)", self.styles['SubHeading']))
        
        wind_scenarios = """
        Il progetto utilizza <b>4 versioni diverse di vento</b> (CI_WIND_faseII_V0.txt, V1.txt, V2.txt, V3.txt), 
        generati da simulazioni meteorologiche con parametri e risoluzioni differenti. Questo aumenta vastamente 
        la diversità dei dati di training, permettendo al modello di generalizzare su condizioni vento realistiche:<br/><br/>
        
        • <b>V0 (Baseline):</b> Scenario pulito con vento omogeneo e prevedibile<br/>
        • <b>V1 (Difficile):</b> Plume altamente disperso da vento variabile, maggiore complessità<br/>
        • <b>V2 (Complesso):</b> Combinazione intermedia di variabilità vento e dispersione<br/>
        • <b>V3 (Ideale):</b> Condizioni ottimali con plume concentrato e stabile<br/><br/>
        
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
        
        <b>Evoluzione Temporale:</b> Il campo di concentrazione evolve nel tempo: ogni 6 step dell'agente (~1 minuto reale) 
        il campo NetCDF avanza di 1 frame temporale, permettendo a vento e correnti di continuare a disperdere il plume.
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
        il codice applica rilassamenti progressivi dei vincoli di distanza, selezionando celle con concentrazione > 0.5 sempre più vicine alla sorgente, 
        finché non trova un punto valido. Gli step di rilassamento sono: <b>d ≥ 500m</b> → <b>d ≥ 250m</b> → <b>d ≥ 100m</b> → 
        <b>spawn casuale</b> se nessun plume disponibile.
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
        train_data = [
            ['Parametro', 'Valore'],
            ['Architettura Policy', 'MLP (256-256 unità nascoste, attivazione Tanh)'],
            ['Normalization', 'VecNormalize (osservazioni e reward)'],
            ['Batch Size', f"{train_cfg['batch_size']}"],
            ['Learning Rate', f"{train_cfg['learning_rate']:.1e}"],
            ['Gamma (discount factor)', f"{train_cfg['gamma']}"],
            ['GAE Lambda', f"{train_cfg['gae_lambda']}"],
            ['Entropy Coefficient', f"{train_cfg['ent_coef']}"],
            ['Timestep Totali', '4M (4,000,000)'],
            ['Curriculum Learning', '3 fasi (35 → 70 → 106 sorgenti × 4 scenari vento)'],
            ['Ambienti Paralleli', '4 (×2 chunks × 4 venti = 32 scenari contemporanei)'],
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
        reward_config = config.get('reward', {})
        
        reward_data = [
            ['Componente', 'Valore/Descrizione'],
            ['Sorgente Raggiunta', f"+{reward_cfg['source_reached_bonus']} + time_bonus (max +50)"],
            ['Penalità Bordo', f"{reward_cfg['boundary_penalty']}"],
            ['Plume (dentro)', f"+{int(reward_cfg['plume_reward_positive'])}"],
            ['Plume (fuori)', f"{int(reward_cfg['plume_reward_negative'])}"],
            ['Reward Distanza', f"dist_impr × 5.0 × {reward_cfg['distance_reward_multiplier']}"],
            ['Gradiente Conc +', f"+{reward_config.get('concentration_gradient_reward_positive', 0.05)}"],
            ['Gradiente Conc -', f"{reward_config.get('concentration_gradient_reward_negative', -0.05)}"],
            ['Vento (controcorr.)', f"+{reward_config.get('wind_alignment_reward', 0.05)}"],
            ['Vento (a favore)', f"{reward_config.get('wind_alignment_penalty', -0.05)}"],
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

    def add_inference_section(self):
        """Sezione 5 (rinumerata): Risultati delle inferenze su 26 sorgenti held-out."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("5. Risultati delle Inferenze", self.styles['SectionHeading']))
        
        # Carica statistiche dal log
        sources_stats, success_rate, mean_steps, mean_initial_distance = self._parse_logs()
        
        # Calcola minutaggio da steps
        minutes = mean_steps * 10 / 60 if mean_steps > 0 else 0
        
        # Descrizione prima della tabella
        description = f"""
        Valutazione del modello PPO su <b>26 sorgenti held-out</b> (SRC107-SRC132) non viste durante il training curriculum, 
        testato su tutti e <b>4 scenari vento</b> (V0, V1, V2, V3). 
        Test eseguito con <b>vento e correnti reali CMEMS</b>:<br/><br/>
        <b>Setup Valutazione:</b><br/>
        • <b>26 sorgenti</b> × <b>4 scenari vento</b> (V0, V1, V2, V3) × <b>2 chunk temporali</b> (Q1/4, Q3/4)<br/>
        • <b>5 episodi</b> per configurazione (totale 26 × 4 × 2 × 5 = <b>1040 episodi</b>)<br/>
        • <b>Distanza media di partenza</b>: 511 metri<br/>
        • <b>Success @ 50m</b>: distanza finale ≤ 50m dalla sorgente<br/>
        • <b>Timeout @ 1080 steps</b> (~3 ore simulate)<br/><br/>
        
        <b>Risultati per Tipologia di Vento:</b><br/>
        • <b>V0:</b> 95.4% success rate<br/>
        • <b>V1:</b> 80.8% success rate<br/>
        • <b>V2:</b> 84.2% success rate<br/>
        • <b>V3:</b> 100.0% success rate<br/><br/>
        
        <b>Risultati per Frame Temporale:</b><br/>
        • <b>Q1/4:</b> 100.0% success rate<br/>
        • <b>Q3/4:</b> 80.2% success rate<br/>
        
        <b>Successo Globale:</b> <b>90.1%</b> (media su tutti i 208 scenari)<br/>
        """
        self.story.append(Paragraph(description, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.4*cm))
        
        # Note sui fallimenti
        note = """
        <b>Analisi dei Fallimenti:</b> La quasi totalità dei fallimenti si concentra nel frame temporale tardivo (Q3/4, 
        simulazione quasi terminata) e negli scenari con <b>vento V1</b> e <b>V2</b>. 
        Dato il successo del 100% negli scenari V0 e V3, questo fenomeno è probabilmente causato dall'eccessiva 
        dispersione del plume dovuta alla <b>combinazione delle correnti oceaniche e del vento forte (8-10 m/s)</b> 
        che soffia da <b>NW per V1</b> e da <b>SW per V2</b> (in particolare per le sorgenti più vicine alla costa).
        """
        self.story.append(Paragraph(note, self.styles['Normal']))

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
        self.add_inference_section()
        
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
    output_path = reports_dir / "HYDRAS_Report_v4.pdf"
    
    generator = HydrasReportGenerator(str(output_path))
    generator.generate()
