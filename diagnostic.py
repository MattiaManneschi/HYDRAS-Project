"""
HYDRAS - Script Diagnostico Completo
Esegui con: python diagnostic.py

Testa:
1. Spawn position
2. Gradiente
3. Reward
4. Agente perfetto (segue gradiente)
5. Agente random
"""

import sys

sys.path.insert(0, '.')
import numpy as np
import yaml

print("=" * 70)
print("HYDRAS DIAGNOSTIC")
print("=" * 70)

# ============================================================
# 1. CARICA CONFIG
# ============================================================
print("\n[1] CONFIG")
print("-" * 50)

with open('configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

spawn_mode = cfg['agent']['spawn_mode']
threshold = cfg['environment']['reward']['distance_threshold']
auto_detect = cfg['environment']['reward'].get('auto_detect_source', False)

print(f"spawn_mode: {spawn_mode}")
print(f"distance_threshold: {threshold}m")
print(f"auto_detect_source: {auto_detect}")

# ============================================================
# 2. CARICA CAMPO
# ============================================================
print("\n[2] CAMPO DI CONCENTRAZIONE")
print("-" * 50)

from utils.data_loader import DataManager

dm = DataManager(data_dir='data/')
field = dm.nc_loader.load(
    'data/CMEMS_S1_01_conc_grid_10m.nc',
    concentration_var="Concentration - component 1"
)

print(f"Shape: {field.data.shape}")
print(f"Timesteps: {field.n_timesteps}")

# Vai a metà simulazione
field.set_time(field.n_timesteps // 2)
print(f"Timestep corrente: {field._current_time_idx}")
print(f"Max conc (corrente): {field.max_concentration:.2f}")

# Trova max globale
data_clean = np.nan_to_num(field.data, nan=0.0)
max_idx = np.unravel_index(np.argmax(data_clean), data_clean.shape)
t_max, y_max, x_max = max_idx
max_x = field.x_coords[x_max]
max_y = field.y_coords[y_max]
max_val = data_clean[t_max, y_max, x_max]

print(f"\nMAX GLOBALE: ({max_x:.0f}, {max_y:.0f}) al tempo t={t_max}")
print(f"  Concentrazione: {max_val:.2f}")

# Sorgente dichiarata
source_declared = (620100, 4796210)  # S1
dist_declared_to_max = np.sqrt((source_declared[0] - max_x) ** 2 + (source_declared[1] - max_y) ** 2)
print(f"\nSORGENTE DICHIARATA S1: {source_declared}")
print(f"  Distanza da max globale: {dist_declared_to_max:.0f}m")

# ============================================================
# 3. TEST SPAWN
# ============================================================
print("\n[3] TEST SPAWN")
print("-" * 50)

from envs.source_seeking_env import SourceSeekingEnv, SourceSeekingConfig

# Forza on_plume per il test
config = SourceSeekingConfig(
    spawn_mode="near_source",
    source_distance_threshold=threshold,
    auto_detect_source=auto_detect
)

env = SourceSeekingEnv(
    config=config,
    concentration_field=field,
    source_id='S1'
)

print(f"Spawn mode: {env.config.spawn_mode}")
print(f"Auto detect: {env.config.auto_detect_source}")
print(f"Source position usata: ({env.source_position[0]:.0f}, {env.source_position[1]:.0f})")
print(f"Threshold: {env.config.source_distance_threshold}m")

print("\nTest 10 spawn:")
spawn_on_plume_count = 0
for i in range(10):
    obs, info = env.reset()
    conc = env.field.get_concentration(env.state.x, env.state.y)
    grad = env.field.get_gradient(env.state.x, env.state.y)
    grad_mag = np.linalg.norm(grad)
    dist = info['initial_distance']

    on_plume = "✓" if conc > 0.5 else "✗"
    if conc > 0.5:
        spawn_on_plume_count += 1

    print(f"  {i + 1}: pos=({env.state.x:.0f},{env.state.y:.0f}), "
          f"conc={conc:.1f} {on_plume}, |grad|={grad_mag:.3f}, dist={dist:.0f}m")

print(f"\nSpawn sulla scia: {spawn_on_plume_count}/10")

# ============================================================
# 4. TEST GRADIENTE (segui il gradiente manualmente)
# ============================================================
print("\n[4] TEST GRADIENTE (segui il gradiente)")
print("-" * 50)

obs, info = env.reset()
start_x, start_y = env.state.x, env.state.y
start_dist = info['initial_distance']
start_conc = env.field.get_concentration(start_x, start_y)

print(f"Partenza: ({start_x:.0f}, {start_y:.0f}), conc={start_conc:.1f}, dist={start_dist:.0f}m")
print(f"Target (source): ({env.source_position[0]:.0f}, {env.source_position[1]:.0f})")

# Simula seguendo il gradiente puro (no RL)
x, y = start_x, start_y
step_size = 15  # metri

print(f"\nSeguo gradiente per 30 step (step={step_size}m):")
for i in range(30):
    conc = env.field.get_concentration(x, y)
    grad = env.field.get_gradient(x, y)
    grad_mag = np.linalg.norm(grad)
    dist = np.sqrt((x - env.source_position[0]) ** 2 + (y - env.source_position[1]) ** 2)

    if i % 5 == 0 or dist < 50 or grad_mag < 1e-6:
        print(f"  Step {i:2d}: pos=({x:.0f},{y:.0f}), conc={conc:.1f}, |grad|={grad_mag:.3f}, dist={dist:.0f}m")

    if dist < env.config.source_distance_threshold:
        print(f"  -> SUCCESSO al step {i}! Distanza finale: {dist:.0f}m")
        break

    if grad_mag < 1e-6:
        print(f"  -> Gradiente nullo al step {i}! Distanza finale: {dist:.0f}m")
        break

    # Muovi nella direzione del gradiente
    x += (grad[0] / grad_mag) * step_size
    y += (grad[1] / grad_mag) * step_size
else:
    print(f"  -> 30 step completati. Distanza finale: {dist:.0f}m")

# ============================================================
# 5. TEST AGENTE PERFETTO (azione = verso sorgente)
# ============================================================
print("\n[5] TEST AGENTE PERFETTO (va dritto alla sorgente)")
print("-" * 50)

obs, info = env.reset()
total_reward = 0

print(f"Partenza: ({env.state.x:.0f}, {env.state.y:.0f}), dist={info['initial_distance']:.0f}m")

for step in range(100):
    # Azione: vai verso la sorgente
    dx = env.source_position[0] - env.state.x
    dy = env.source_position[1] - env.state.y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    if dist > 0:
        action = np.array([dx / dist, dy / dist], dtype=np.float32)
    else:
        action = np.array([0, 0], dtype=np.float32)

    obs, reward, terminated, truncated, info_step = env.step(action)
    total_reward += reward

    if step % 10 == 0:
        print(f"  Step {step:3d}: dist={info_step['distance_to_source']:.0f}m, "
              f"conc={info_step['concentration']:.1f}, reward={reward:.1f}")

    if terminated:
        print(f"\n  -> TERMINATO al step {step}")
        print(f"     Source reached: {info_step.get('source_reached', False)}")
        print(f"     Distanza finale: {info_step['distance_to_source']:.0f}m")
        break

print(f"\nReward totale: {total_reward:.1f}")

# ============================================================
# 6. TEST AGENTE RANDOM
# ============================================================
print("\n[6] TEST AGENTE RANDOM (10 episodi)")
print("-" * 50)

successes = 0
final_distances = []

for ep in range(10):
    obs, info = env.reset()

    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info_step = env.step(action)

        if terminated or truncated:
            break

    final_dist = info_step['distance_to_source']
    final_distances.append(final_dist)
    success = info_step.get('source_reached', False)
    if success:
        successes += 1

    status = "✓ SUCCESS" if success else f"✗ {final_dist:.0f}m"
    print(f"  Ep {ep + 1}: {status}")

print(f"\nRandom agent: {successes}/10 successi, avg dist={np.mean(final_distances):.0f}m")

# ============================================================
# 7. ANALISI REWARD
# ============================================================
print("\n[7] ANALISI REWARD (1 episodio dettagliato)")
print("-" * 50)

obs, info = env.reset()
print(f"Partenza: ({env.state.x:.0f}, {env.state.y:.0f})")

# Fai 5 step verso la sorgente e analizza il reward
for step in range(5):
    dx = env.source_position[0] - env.state.x
    dy = env.source_position[1] - env.state.y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    action = np.array([dx / dist, dy / dist], dtype=np.float32)

    obs, reward, terminated, truncated, info_step = env.step(action)

    print(f"\nStep {step + 1}:")
    print(f"  Azione: [{action[0]:.2f}, {action[1]:.2f}]")
    print(f"  Distanza: {info_step['distance_to_source']:.0f}m")
    print(f"  Concentrazione: {info_step['concentration']:.2f}")
    print(f"  Reward totale: {reward:.2f}")

    # Stampa componenti reward (se disponibili)
    reward_components = ['distance_reward', 'gradient_reward', 'concentration_bonus',
                         'time_penalty', 'source_found', 'time_bonus']
    for comp in reward_components:
        if comp in info_step:
            print(f"    {comp}: {info_step[comp]:.2f}")

env.close()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETATO")
print("=" * 70)