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
    def __init__(self, output_path="HYDRAS_Report_v3.pdf"):
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
        """Sezione 1: Acquisizione dati — 132 sorgenti con curriculum learning."""
        self.story.append(Spacer(1, 0.5*cm))
        
        self.story.append(Paragraph("1. Dataset — 132 Sorgenti", self.styles['SectionHeading']))
        
        # Fonte dati
        text = """
        <b>Sorgente Dati:</b> I dati di concentrazione provengono da simulazioni <b>MIKE21</b> in formato NetCDF (1411 timestep, risoluzione 10m, griglia 300×250 celle). 
        Il modulo <b>data_loader.py</b> gestisce il caricamento automatico di <b>132 sorgenti</b> (SRC001-SRC132): legge le coordinate spaziali (x, y), 
        temporali (time) e i valori di concentrazione da file NetCDF, costruendo un campo interpolabile per ogni sorgente. 
        Ogni sorgente ha coordinate geospaziali (UTM32N) caricate da <b>Coordinate_Sorgenti_FaseII.csv</b>.<br/><br/>
        
        <b>Dataset Split:</b> Dataset suddiviso in <b>106 sorgenti di training</b> (SRC001-SRC106) e <b>26 di valutazione</b> (SRC107-SRC132).
        Durante il training, il curriculum learning espande progressivamente il set di sorgenti disponibili.<br/><br/>
        
        <b>Augmentazione Dati (Chunking):</b> Per massimizzare la variabilità e creare multipli scenari di partenza per ogni sorgente, i 1411 timestep di ogni simulazione vengono suddivisi in <b>2 chunk temporali</b>:<br/>
        • <b>Chunk 0</b> (spawn @ 1/4 = timestep 352): inizio della propagazione del plume;<br/>
        • <b>Chunk 1</b> (spawn @ 3/4 = timestep 1058): metà della simulazione.<br/>
        L'agente può essere inizializzato in fasi diverse della dispersione, aumentando la robustezza del modello. Con 4 ambienti paralleli × 2 chunks = 8 scenari di training simultanei.<br/><br/>
        """
        self.story.append(Paragraph(text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Paragraph("1.1 Dati di Vento e Corrente Oceanica", self.styles['SubHeading']))
        
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

    def add_environment_section(self, config):
        """Sezione 2: Ambiente di simulazione."""
        self.story.append(Spacer(1, 0.3*cm))
        
        self.story.append(Paragraph("2. Ambiente di Simulazione", self.styles['SectionHeading']))
        
        text = """
        L'ambiente (<b>source_seeking_env.py</b>) è una griglia 300×250 celle (risoluzione 10m) che 
        rappresenta un dominio marino di 3×2.5 km nella baia di Cecina (coste toscane, UTM32N). 
        L'agente (AUV - Autonomous Underwater Vehicle) si muove con <b>8 azioni discrete</b>: 
        N, S, E, W, NE, SE, NW, SW. La velocità è <b>1 m/s</b> e il timestep <b>dt=10s</b>, 
        quindi ogni azione sposta l'agente di 10 metri (o ~7m per componente nelle direzioni diagonali). 
        L'agente viene posizionato al chunk spawn time su una cella con concentrazione > 0.5, 
        a distanza <b>500-1500m dalla sorgente</b> e almeno <b>50m dalla costa</b>. 
        L'episodio termina quando la sorgente viene trova (distanza &lt; 50m) oppure dopo 1080 step (~3 ore).<br/><br/>
        
        <b>Evoluzione Temporale:</b> Il campo di concentrazione evolve nel tempo: ogni 6 step dell'agente (~1 minuto reale) 
        il campo NetCDF avanza di 1 frame temporale, permettendo a vento e correnti di continuare a disperdere il plume.
        """
        self.story.append(Paragraph(text, self.styles['Normal']))

    def add_training_section(self, config):
        """Sezione 3: Training."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("3. Training e Architettura", self.styles['SectionHeading']))
        
        # 3.a - Input della rete neurale
        self.story.append(Paragraph("3.1 Input della Rete Neurale — 112 Dimensioni", self.styles['SubHeading']))
        
        obs_text = """
        Lo spazio di osservazione è un vettore continuo a <b>112 dimensioni</b> (normalizzato via VecNormalize). 
        Includi memoria storica concentrazioni e sensori radiali nelle 8 direzioni:
        """
        self.story.append(Paragraph(obs_text, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.3*cm))
        
        obs_data = [
            ['Componente', 'Dim', 'Descrizione'],
            ['Concentrazione Attuale', '1', 'Conc. posizione corrente (x, y)'],
            ['Memoria Conc. (Locale)', '9', 'Umltimi 9 timestep'],
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
        
        # 3.b - Parametri di training
        self.story.append(Paragraph("3.2 Parametri di Training", self.styles['SubHeading']))
        
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
            ['Timestep Totali', f"{int(train_cfg['total_timesteps']*1e-6)}M"],
            ['Curriculum Learning', '3 fasi (35 → 70 → 106 sorgenti)'],
            ['Ambienti Paralleli', '4 (×2 chunks = 8 scenari contemporanei)'],
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
        
        # 3.d - Risultati di training
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("3.4 Risultati del Training", self.styles['SubHeading']))
        
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
        """Sezione 4: Risultati delle inferenze su 26 sorgenti held-out."""
        self.story.append(PageBreak())
        
        self.story.append(Paragraph("4. Risultati delle Inferenze (Held-Out Set 20%)", self.styles['SectionHeading']))
        
        # Carica statistiche dal log
        sources_stats, success_rate, mean_steps, mean_initial_distance = self._parse_logs()
        
        # Calcola minutaggio da steps
        minutes = mean_steps * 10 / 60 if mean_steps > 0 else 0
        
        # Descrizione prima della tabella
        description = f"""
        Valutazione del modello PPO su <b>26 sorgenti held-out</b> (SRC107-SRC132) non viste durante il training curriculum. 
        Test eseguito con <b>vento e correnti reali CMEMS</b>:<br/><br/>
        <b>Setup Valutazione:</b><br/>
        • <b>26 sorgenti</b> × <b>2 chunk temporali</b> (Q1/4 @ 352 timestep, Q3/4 @ 1058 timestep)<br/>
        • <b>5 episodi</b> per chunk (totale 26 × 2 × 5 = <b>260 episodi</b>)<br/>
        • <b>Success @ 50m</b>: distanza finale ≤ 50m dalla sorgente<br/>
        • <b>Timeout @ 1080 step</b> (~3 ore simulate)<br/><br/>
        
        <b>Risultato Complessivo:</b><br/>
        • <b>Success Rate Medio:</b> {success_rate:.1f}%<br/>
        • <b>Numero Medio di Steps:</b> {mean_steps} (~{minutes:.1f} minuti)<br/>
        • <b>Distanza Media di Partenza:</b> {mean_initial_distance}m
        """
        self.story.append(Paragraph(description, self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.4*cm))
        
        # Success cases
        self.story.append(Paragraph("Success Cases (3 Esempi)", self.styles['SubHeading']))
        
        # Tabella con 3 success cases e relativi plot
        success_cases = [
            ("SRC110", "chunk0", "Q1/4", "58 step (~9.7 min)", "512m"),
            ("SRC116", "chunk2", "Q3/4", "82 step (~13.7 min)", "448m"),  # chunk2 (Q3/4), not chunk1
            ("SRC119", "chunk2", "Q3/4", "91 step (~15.2 min)", "520m")   # chunk2 (Q3/4), not chunk1
        ]
        
        evals_dir = self.project_root.parent / "evaluations_v4"
        for src, chunk, q, steps, dist in success_cases:
            case_text = f"""<b>{src} - {q}:</b> Success in {steps}"""
            case_elements = [Paragraph(case_text, self.styles['Normal'])]
            
            # Prova ad aggiungere il plot
            plot_path = evals_dir / src / f"ep01_{chunk}_trajectory.png"
            if plot_path.exists():
                try:
                    img = Image(str(plot_path), width=14*cm, height=9*cm)
                    case_elements.append(img)
                except Exception as e:
                    pass
            
            # Mantieni titolo e plot sulla stessa pagina
            self.story.append(KeepTogether(case_elements))
            self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(PageBreak())
        
        # Failure cases
        self.story.append(Paragraph("Failure Cases (3 Esempi)", self.styles['SubHeading']))
        
        failure_cases = [
            ("SRC107", "chunk1", "Q3/4", "1080 step", "468m", "287m"),
            ("SRC108", "chunk1", "Q3/4", "1080 step", "504m", "356m"),
            ("SRC112", "chunk1", "Q3/4", "1080 step", "476m", "312m")
        ]
        
        for src, chunk, q, steps, start_dist, final_dist in failure_cases:
            case_text = f"""<b>{src} - {q}:</b> Timeout @ {steps}"""
            case_elements = [Paragraph(case_text, self.styles['Normal'])]
            
            # Prova ad aggiungere il plot
            plot_path = evals_dir / src / f"ep01_{chunk}_trajectory.png"
            if plot_path.exists():
                try:
                    img = Image(str(plot_path), width=14*cm, height=9*cm)
                    case_elements.append(img)
                except Exception as e:
                    pass
            
            # Mantieni titolo e plot sulla stessa pagina
            self.story.append(KeepTogether(case_elements))
            self.story.append(Spacer(1, 0.2*cm))
        
        self.story.append(Spacer(1, 0.3*cm))

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
    # Salva il report nella cartella reports
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / "HYDRAS_Report_v3.pdf"
    
    generator = HydrasReportGenerator(str(output_path))
    generator.generate()
