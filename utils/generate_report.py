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
    def __init__(self, output_path="HYDRAS_Report.pdf"):
        self.output_path = output_path
        self.project_root = Path(__file__).parent
        self.styles = self._setup_styles()
        self.story = []

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

    def _load_config(self):
        """Carica la configurazione del modello."""
        config_path = self.project_root.parent / "trained_models/ppo_20260322_113620/config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _parse_logs(self):
        """Parsa il file logs.txt dal nuovo formato evaluations_wind_current."""
        logs_path = self.project_root.parent / "evaluations_wind_current/logs.txt"
        
        sources_stats = {'S1': {}, 'S2': {}, 'S3': {}}
        with open(logs_path) as f:
            lines = f.readlines()
        
        # Estrai tutti gli episodi per calcolare medie per sorgente
        for source in ['S1', 'S2', 'S3']:
            sources_stats[source] = {
                'init_dists': [],
                'final_dists': [],
                'steps': []
            }
        
        current_source = None
        for line in lines:
            line = line.strip()
            # Identifica la sorgente dal formato "[S1_01 — spawn @Q1/4]"
            if line.startswith('['):
                source_char = line[1:3]  # es. "S1", "S2", "S3"
                if source_char in ['S1', 'S2', 'S3']:
                    current_source = source_char
            
            # Parse linee episodio: "Ep 1/5: ✓ [success ] init=589m  final=93m  steps=54"
            elif current_source and 'Ep' in line and 'final=' in line and '[success' in line:
                parts = line.split()
                try:
                    init_dist = int(parts[5].replace('init=', '').replace('m', ''))
                    final_dist = int(parts[6].replace('final=', '').replace('m', ''))
                    steps = int(parts[7].replace('steps=', '').replace('m', ''))
                    
                    sources_stats[current_source]['init_dists'].append(init_dist)
                    sources_stats[current_source]['final_dists'].append(final_dist)
                    sources_stats[current_source]['steps'].append(steps)
                except (ValueError, IndexError):
                    pass
        
        # Calcola medie per sorgente
        final_stats = {}
        for source in ['S1', 'S2', 'S3']:
            if sources_stats[source]['init_dists']:
                final_stats[source] = {
                    'init_dist': np.mean(sources_stats[source]['init_dists']),
                    'final_dist': np.mean(sources_stats[source]['final_dists']),
                    'steps': np.mean(sources_stats[source]['steps'])
                }
        
        return final_stats

    def add_title_page(self, config):
        """Aggiunge la pagina di titolo."""
        self.story.append(Spacer(1, 2*cm))
        
        title = Paragraph(
            "HYDRAS Project Report",
            self.styles['TitleReport']
        )
        self.story.append(title)
        
        self.story.append(Spacer(1, 1.5*cm))

    def add_data_acquisition_section(self, config):
        """Sezione 1: Acquisizione dati."""
        self.story.append(Spacer(1, 0.5*cm))
        
        self.story.append(Paragraph("1. Acquisizione Dati", self.styles['SectionHeading']))
        
        # Fonte dati
        text = """
        <b>Sorgente Dati:</b> I dati di concentrazione provengono da simulazioni MIKE21 in formato NetCDF. 
        Il modulo <b>data_loader.py</b> gestisce il caricamento: legge le coordinate spaziali (x, y), 
        temporali (time) e i valori di concentrazione, costruendo un campo interpolabile. 
        I file seguono il pattern <b>CMEMS_S{1,2,3}_0{1-4}_conc_grid_10m.nc</b> 
        (3 sorgenti × 4 scenari = 12 file).<br/><br/>
        
        <b>Augmentazione Dati:</b> Per massimizzare la variabilità nei dati di allenamento, 
        ogni scenario NetCDF viene suddiviso in due chunk temporali: <b>Chunk 1/4</b> (inizio simulazione) 
        e <b>Chunk 3/4</b> (metà simulazione). L'agente può così essere inizializzato in fasi diverse 
        della propagazione del plume, aumentando la robustezza del modello addestrato.<br/><br/>
        
        <b>Velocità e Correnti:</b> Dati reali CMEMS + variabilità del vento sintetica (20% stocasticità).
        """
        self.story.append(Paragraph(text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("1.1 Estrazione Dati di Vento e Corrente", self.styles['SubHeading']))
        
        text_wind = """
        I dati di vento vengono estratti da file di testo (.txt) tramite il modulo <b>wind_mapping.yaml</b>, 
        mentre i dati di corrente oceanica vengono estratti dai file NetCDF CMEMS tramite <b>data_loader.py</b>. 
        Per ogni scenario, gli array bidimensionali (u, v) delle velocità vengono interpolati sulla griglia 
        di simulazione (300×250 celle) e sincronizzati temporalmente secondo il timestamp del frame. 
        Tutti i dati vengono normalizzati tramite <b>VecNormalize</b> (media 0, varianza 1).
        """
        self.story.append(Paragraph(text_wind, self.styles['Normal']))

    def add_environment_section(self, config):
        """Sezione 2: Ambiente di simulazione."""
        self.story.append(Spacer(1, 0.3*cm))
        
        self.story.append(Paragraph("2. Ambiente di Simulazione", self.styles['SectionHeading']))
        
        text = """
        L'ambiente (<b>source_seeking_env.py</b>) è una griglia 300×250 celle (risoluzione 10m) che 
        rappresenta un dominio marino di 3×2.5 km. L'agente (AUV) si muove con 8 azioni discrete: 
        N, S, E, W e le 4 diagonali (NE, SE, NW, SW). La velocità è 1 m/s e il timestep dt=10s, 
        quindi ogni azione sposta l'agente di 10 metri (o ~7m per componente nelle direzioni diagonali). 
        L'agente viene posizionato a metà simulazione (frame 1440) su una cella con concentrazione > 0.5, 
        a distanza 200-1500m dalla sorgente e almeno 50m dalla costa. 
        Il campo evolve nel tempo: ogni 6 step dell'agente (~1 minuto) il campo avanza di 1 frame.
        """
        self.story.append(Paragraph(text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("2.1 Impiego Dati di Vento e Corrente nell'Ambiente", self.styles['SubHeading']))
        
        text_env_wind = """
        Durante la simulazione, i dati CMEMS reali (più rumore gaussiano per la variabilità) vengono 
        utilizzati dall'ambiente per tre scopi: (1) <b>alimentazione dello stato osservabile</b> – 
        le componenti (vx, vy) del vento e (ux, uy) della corrente sono aggiunte al vettore di 
        osservazione (40 dimensioni) così che l'agente possa decidere in base alle condizioni 
        idrodinamiche; (2) <b>calcolo del modello dinamico</b> – il vento e le correnti influenzano 
        la dispersione della sorgente secondo il modello gaussiano; (3) <b>interpolazione ad ogni step</b> – 
        i campi vengono recuperati tramite interpolazione bilineare alla posizione corrente dell'agente 
        nel frame temporale corrente.
        """
        self.story.append(Paragraph(text_env_wind, self.styles['Normal']))

    def add_training_section(self, config):
        """Sezione 3: Training."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("3. Training e Architettura", self.styles['SectionHeading']))
        
        # 3.a - Input della rete neurale
        self.story.append(Paragraph("3.1 Input della Rete Neurale", self.styles['SubHeading']))
        
        obs_text = """
        Lo spazio di osservazione è un vettore continuo a 40 dimensioni (normalizzato via VecNormalize):
        """
        self.story.append(Paragraph(obs_text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.3*cm))
        
        obs_data = [
            ['Componente', 'Dimensione', 'Descrizione'],
            ['Concentrazione Attuale', '1', 'Concentrazione nella posizione dell\'agente'],
            ['Memoria Concentrazione', '9', 'Concentrazioni degli ultimi 9 timestep'],
            ['Storico Movimento', '18', '9 step di spostamenti (Δx, Δy) [m]'],
            ['Sensori Locali', '8', 'Concentrazione a sensori radiali ±20m'],
            ['Velocità Vento', '2', 'Componenti vento corrente (u, v)'],
            ['Velocità Corrente', '2', 'Componenti corrente oceanica (u, v)']
        ]
        
        obs_table = Table(obs_data, colWidths=[4*cm, 2.8*cm, 7.2*cm])
        obs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eeeeee')]),
        ]))
        self.story.append(obs_table)
        
        self.story.append(Spacer(1, 0.2*cm))
        
        # 3.b - Parametri di training
        self.story.append(Paragraph("3.2 Parametri di Training", self.styles['SubHeading']))
        
        train_cfg = config['training']
        train_data = [
            ['Parametro', 'Valore'],
            ['Architettura Policy', 'MLP (256-256 unità nascoste, attivazione Tanh)'],
            ['Batch Size', f"{train_cfg['batch_size']}"],
            ['Learning Rate', f"{train_cfg['learning_rate']:.1e}"],
            ['Gamma (sconto)', f"{train_cfg['gamma']}"],
            ['GAE Lambda', f"{train_cfg['gae_lambda']}"],
            ['Timestep Totali', f"{int(train_cfg['total_timesteps']*1e-6)}M"],
            ['Curriculum: 3 fasi', 'S1 (0-1M) → S1+S2 (1-2M) → S1+S2+S3 (2-3M)']
        ]
        
        train_table = Table(train_data, colWidths=[3.3*cm, 8.2*cm])
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
        
        # 3.c - Funzione di reward
        self.story.append(Paragraph("3.3 Funzione di Reward", self.styles['SubHeading']))
        
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
        
        # 3.d - Risultati di trading
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("3.4 Risultati del Training", self.styles['SubHeading']))
        
        # Aggiungi plot di training
        plots_dir = self.project_root.parent / "trained_models/ppo_20260322_113620/plots"
        if (plots_dir / "training_loss.png").exists() and (plots_dir / "training_success_rate.png").exists():
            try:
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

    def add_inference_section(self):
        """Sezione 4: Risultati delle inferenze."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("4. Risultati delle Inferenze", self.styles['SectionHeading']))
        
        # Descrizione prima della tabella
        description = """
        Valutato con vento e correnti reali su 3 sorgenti (S1, S2, S3) con 2 chunk temporali (Q1/4, Q3/4) e 5 episodi ciascuno = 120 episodi totali, 
        <b>con un success rate medio del 90.8%</b>:
        """
        self.story.append(Paragraph(description, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        # Parsa i logs
        sources_stats = self._parse_logs()
        
        # Tabella con medie
        inference_data = [['Sorgente', 'Dist. Partenza [m]', 'Dist. Finale [m]', 'Steps Medi', 'Tempo Medio [min]']]
        
        for source in sorted(sources_stats.keys()):
            stats = sources_stats[source]
            time_min = stats['steps'] * 10 / 60  # steps * 10s / 60
            
            inference_data.append([
                source,
                f"{stats['init_dist']:.0f}",
                f"{stats['final_dist']:.0f}",
                f"{stats['steps']:.0f}",
                f"{time_min:.1f}"
            ])
        
        inference_table = Table(inference_data, colWidths=[2.2*cm, 3.3*cm, 3.3*cm, 2.5*cm, 3.5*cm])
        inference_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ADD8E6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.white])
        ]))
        self.story.append(inference_table)
        
        # Aggiungi spazio e titolo per esempi di traiettorie
        self.story.append(Spacer(1, 1*cm))
        self.story.append(Paragraph("Esempi di Traiettorie di Ricerca", self.styles['SubHeading']))
        self.story.append(Spacer(1, 0.3*cm))
        
        # Carica 3 esempi di traiettorie (stack verticale)
        trajectory_examples = [
            ("evaluations_wind_current/S1/S1_04/ep04_chunk0_trajectory.png", "S1_04 - Ep 04, Chunk 0"),
            ("evaluations_wind_current/S2/S2_03/ep04_chunk1_trajectory.png", "S2_03 - Ep 04, Chunk 1"),
            ("evaluations_wind_current/S3/S3_02/ep02_chunk1_trajectory.png", "S3_02 - Ep 02, Chunk 1")
        ]
        
        # Crea tabella con immagini: stacked verticalmente, larghezza intera, proporzionati
        trajectory_data = []
        
        for idx, (traj_path, label) in enumerate(trajectory_examples):
            full_path = Path(traj_path)
            if full_path.exists():
                # Ogni immagine occupa tutta la larghezza della pagina
                img = Image(str(full_path), width=17*cm, height=11*cm)
                label_para = Paragraph(label, ParagraphStyle(
                    name='TrajLabel',
                    parent=self.styles['Normal'],
                    fontSize=8,
                    alignment=TA_CENTER,
                    spaceAfter=2
                ))
                trajectory_data.append([img])
                trajectory_data.append([label_para])
        
        # Tabella con tutte le immagini, una per riga
        if trajectory_data:
            trajectory_table = Table(trajectory_data, colWidths=[17.5*cm])
            trajectory_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ]))
            self.story.append(trajectory_table)

    def generate(self):
        """Genera il report PDF."""
        print("Generando Report HYDRAS...")
        
        # Carica configurazione
        config = self._load_config()
        
        # Costruisci il report
        self.add_title_page(config)
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
    generator = HydrasReportGenerator("HYDRAS_Report.pdf")
    generator.generate()
