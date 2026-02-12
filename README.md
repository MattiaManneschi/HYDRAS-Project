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