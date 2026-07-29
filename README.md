# HYDRAS - Hydrodynamic-aware Distributed Robots for Marine Source-Seeking

Marine pollutant source localization with autonomous underwater vehicles (AUVs).
The project compares a gradient method (Field Climbing Method) and a Reinforcement
Learning approach (PPO) in guiding an agent up the concentration field to the
source, accounting for wind and sea current. Data come from MIKE21 hydrodynamic
simulations of the Cecina bay.

## Demo (macOS · Windows)

1. Download the launcher for your system — **`launcher.command`** (macOS) or
   **`launcher.bat`** (Windows) — and place it in a folder with at least ~8 GB free.
2. Double-click it. On first launch it downloads everything it needs (code, data,
   models) into that same folder, then opens the interface.
3. Configure the scenario and the agent with the drop-down menus (see below), then
   press **Start** to watch one episode play out in real time.

### Scenario configuration

| Menu | Options | Meaning |
|---|---|---|
| **Technology** | PPO · FCM Adam | Which algorithm drives the agent: the learned PPO policy or the gradient method (FCM with Adam). |
| **Wind** | V0 · V1 · V2 · V3 | Wind scenario (four hydrodynamic runs with different wind conditions). |
| **Time Chunk** | Q1/4 · Q1/2 · Q3/4 | When in the simulation the episode starts — first quarter, middle, or third quarter (plume more or less dispersed). |
| **Max Speed** | 1–5 m/s | Agent's maximum speed. *(PPO only.)* |
| **Formation** | Single · Double ring | Single sensor ring (20 m) or double ring (20 m + 50 m). *(PPO only.)* |

Once **Wind** and **Time Chunk** are set, a random source is picked for
that scenario. Each run shows a single episode; press **Start** again for a new one.
