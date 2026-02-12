# HYDRAS Source Seeking

**Addestramento di agenti RL per la localizzazione di sorgenti di inquinante marino**

Basato sulle simulazioni idrodinamiche MIKE21 per l'area del Porto di Cecina, questo progetto implementa un sistema di Reinforcement Learning per addestrare agenti (AUV - Autonomous Underwater Vehicles) a localizzare sorgenti di inquinanti in ambiente marino.

## 🎯 Obiettivo

Addestrare una rete di agenti autonomi capaci di:
1. Navigare in un campo di concentrazione di inquinante
2. Seguire il gradiente di concentrazione
3. Localizzare la sorgente di emissione

## 📁 Struttura del Progetto

```
hydras_source_seeking/
├── configs/
│   └── config.yaml          # Configurazione principale
├── envs/
│   ├── __init__.py
│   └── source_seeking_env.py # Ambiente Gymnasium
├── utils/
│   ├── __init__.py
│   └── data_loader.py       # Loader dati NetCDF + generatore sintetico
├── agents/                   # (per estensioni future)
├── data/                     # Directory per file .nc
├── train_ppo.py             # Script di training PPO
├── visualize.py             # Tools di visualizzazione
├── requirements.txt         # Dipendenze
└── README.md
```

## 🚀 Quick Start

### 1. Installazione

```bash
# Clona o copia il progetto
cd hydras_source_seeking

# Crea ambiente virtuale (consigliato)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Training con dati sintetici

```bash
# Training base (usa dati sintetici)
python train_ppo.py train --source S1 --n-envs 4 --timesteps 500000

# Training con più ambienti paralleli (più veloce)
python train_ppo.py train --source S1 --n-envs 8 --timesteps 1000000
```

### 3. Training con dati NetCDF

```bash
# Copia i file .nc nella cartella data/
cp /path/to/CMEMS_S1_01_conc_grid_10m.nc data/

# Training con dati reali
python train_ppo.py train --source S1 --nc-file data/CMEMS_S1_01_conc_grid_10m.nc
```

### 4. Valutazione

```bash
# Valuta il modello addestrato
python train_ppo.py eval outputs/ppo_S1_*/models/final_model.zip --episodes 20

# Con rendering
python train_ppo.py eval outputs/ppo_S1_*/models/best/best_model.zip --render
```

## ⚙️ Configurazione

Il file `configs/config.yaml` contiene tutti i parametri configurabili:

### Dominio
```yaml
domain:
  xmin: 619000    # Coordinate UTM
  xmax: 622000
  ymin: 4794500
  ymax: 4797000
  grid_resolution: 10  # metri
```

### Agente
```yaml
agent:
  max_velocity: 1.5      # m/s
  sensor_radius: 50      # m
  n_concentration_samples: 8
  action_type: "continuous"  # o "discrete"
```

### Reward
```yaml
environment:
  reward:
    source_reached_bonus: 100.0
    concentration_gradient_scale: 10.0
    step_penalty: -0.1
    boundary_penalty: -10.0
    distance_threshold: 30  # metri per "successo"
```

### PPO Hyperparameters
```yaml
training:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
```

## 🔬 Dettagli Tecnici

### Observation Space (15 dimensioni)
- **8 concentrazioni** campionate in cerchio attorno all'agente (raggio 50m)
- **1 concentrazione** al centro (posizione agente)
- **2 componenti** del gradiente normalizzato
- **2 coordinate** posizione normalizzata [-1, 1]
- **2 componenti** velocità normalizzata

### Action Space
- **Continuous**: `[vx, vy]` in [-1, 1], scalato a ±1.5 m/s
- **Discrete**: 8 direzioni + stazionario (9 azioni)

### Reward Shaping
1. **+100** per raggiungere la sorgente (< 30m)
2. **+10 × alignment** per allineamento con gradiente
3. **+1 × Δconcentrazione** per aumento concentrazione
4. **+0.1 × Δdistanza** per avvicinamento
5. **-0.1** penalità per step (incentiva velocità)
6. **-10** per uscita dal dominio

## 📊 Visualizzazione

```python
from visualize import plot_training_summary, create_animation
from utils.data_loader import DataManager

# Carica dati
dm = DataManager(use_synthetic=True)
field = dm.get_concentration_field(source_id='S1')

# Visualizza campo
from visualize import plot_concentration_field
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10))
plot_concentration_field(field, ax=ax)
plt.savefig('concentration_field.png')

# Dopo training, visualizza traiettorie
trajectory = np.load('trajectory.npy')  # salvata durante eval
plot_training_summary(trajectory, field, save_path='summary.png')
```

## 🔄 Estensione Multi-Agente

Il sistema è progettato per essere esteso a scenari multi-agente. Vedi `configs/config.yaml`:

```yaml
multi_agent:
  enabled: false  # Abilitare per multi-agente
  n_agents: 3
  communication_range: 100  # metri
  shared_reward: false
  coordination_bonus: 5.0
```

Per l'implementazione multi-agente, si consiglia:
- **PettingZoo** per ambienti multi-agente
- **MAPPO** (Multi-Agent PPO) per training coordinato
- **Communication protocols** per condivisione informazioni tra agenti

## 📈 Monitoraggio Training

```bash
# Avvia TensorBoard
tensorboard --logdir outputs/ppo_S1_*/logs/tensorboard

# Apri browser: http://localhost:6006
```

Metriche monitorate:
- `rollout/ep_rew_mean`: Reward medio per episodio
- `rollout/ep_len_mean`: Lunghezza media episodi
- `custom/success_rate`: Tasso di successo
- `custom/avg_final_distance`: Distanza media finale dalla sorgente

## 🗂️ Dati delle Simulazioni DICEA

Le simulazioni provengono da MIKE21 con:
- **Dominio**: ~16×14 km attorno al Porto di Cecina
- **Risoluzione**: 10m nella zona di interesse
- **Sorgenti**: S1, S2, S3 con portata 50 l/s e concentrazione 1000 g/m³
- **Forzanti**: Dati CMEMS (correnti) + vento misurato
- **Output**: NetCDF con passo 10m e intervallo 1 minuto

## 📝 Note

- Il generatore sintetico usa un modello advection-diffusion semplificato
- Per risultati realistici, usare i file NetCDF delle simulazioni MIKE21
- Il training richiede circa 500k-1M timesteps per convergenza
- Consigliato: GPU per training più veloce (PPO supporta CUDA)

## 📚 Riferimenti

- **MIKE21**: DHI Flow Model FM
- **CMEMS**: Copernicus Marine Environment Monitoring Service
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

## 👥 Autori

- Progetto HYDRAS
- Simulazioni: DICEA
- Implementazione RL: [Il tuo nome]

## 📄 Licenza

[Da definire]
