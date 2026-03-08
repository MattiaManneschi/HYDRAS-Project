"""
HYDRAS Source Seeking - Gymnasium Environment
Ambiente per l'addestramento di agenti RL per la localizzazione
di sorgenti di inquinante in ambiente marino.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from .data_loader import (
    ConcentrationField, DataManager, DomainConfig
)


@dataclass
class AgentState:
    """Stato corrente dell'agente."""
    x: float  # Posizione x (UTM)
    y: float  # Posizione y (UTM)
    vx: float = 0.0  # Velocità x
    vy: float = 0.0  # Velocità y
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])


@dataclass
class SourceSeekingConfig:
    """Configurazione dell'ambiente."""
    # Domain
    xmin: float = 619000
    xmax: float = 622000
    ymin: float = 4794500
    ymax: float = 4797000
    resolution: float = 10

    # Agent
    max_velocity: float = 1.0  # m/s (velocità AUV)

    # Memory - ultimi N step (concentrazioni passate + spostamenti passati)
    memory_length: int = 9  # 9 step passati

    # Episode
    dt: float = 10.0  # s  (con max_velocity=1 m/s -> spostamento max 10 m/step)
    max_steps: int = 1080  # 3 ore: 10800s / 10s = 1080 steps

    # Spawn
    spawn_min_distance: float = 200.0  # m distanza minima dalla sorgente
    spawn_max_distance: float = 3000.0  # m distanza massima dalla sorgente (3h a 1m/s = 10.8km, margine ok)
    spawn_start_frame: int = 1440  # frame di partenza (metà simulazione)
    spawn_conc_threshold: float = 0.5  # soglia minima concentrazione per spawn

    # Reward
    source_distance_threshold: float = 100  # m (intorno di successo)
    source_found_reward: float = 100.0
    step_penalty: float = -0.1
    boundary_penalty: float = -10.0
    land_penalty: float = -200.0  # penalità per collisione con terra (aumentata)
    distance_reward_multiplier: float = 1.0  # Moltiplicatore per reward distanza
    plume_reward_positive: float = 0.3  # reward binario dentro il plume
    plume_reward_negative: float = -0.3  # penalità fuori dal plume
    plume_threshold: float = 0.1  # soglia concentrazione per "dentro il plume"

    # Land avoidance
    land_proximity_threshold: float = 10.0  # m - distanza dalla terra per penalità progressiva
    land_proximity_penalty_max: float = -1.0  # penalità massima per vicinanza terra
    spawn_min_land_distance: float = 50.0  # m - distanza minima dalla terra per spawn

    # Action
    action_type: str = "discrete"  # "discrete" (N/S/E/W)
    n_discrete_actions: int = 4  # 4 direzioni cardinali


class SourceSeekingEnv(gym.Env):
    """
    Ambiente Gymnasium per il source seeking di inquinanti marini.

    L'agente (AUV) deve navigare in un campo di concentrazione
    per trovare la sorgente dell'inquinante.

    Observation Space (32 valori):
        - 1 concentrazione corrente
        - 9 concentrazioni passate
        - 9 x (Δx, Δy) spostamenti passati in metri
        - 4 sensori terra direzionali (N, S, E, W) - binari
        (normalizzazione delegata a VecNormalize)

    Action Space:
        - Discrete(4): Nord(+y), Sud(-y), Est(+x), Ovest(-x)

    Reward:
        - Bonus sorgente raggiunta + time bonus (terminale dominante)
        - Reward distanza (segnale dominante continuo)
        - Reward binario plume (+0.3 dentro, -0.3 fuori)
        - Penalità per step (-0.1)
        - Penalità bordi (-10) e terra (-200)
        - Penalità progressiva avvicinamento terra
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[SourceSeekingConfig] = None,
        concentration_field: Optional[ConcentrationField] = None,
        source_id: str = "S1",
        render_mode: Optional[str] = None,
        data_dir: Optional[str] = None,
        randomize_field: bool = False,
        **kwargs
    ):
        """
        Args:
            config: Configurazione dell'ambiente
            concentration_field: Campo di concentrazione pre-caricato
            source_id: ID della sorgente ('S1', 'S2', 'S3')
            render_mode: Modalità di rendering
            data_dir: Directory con file NC (per randomize_field)
            randomize_field: Se True, sceglie un NC random ad ogni reset
            **kwargs: Parametri aggiuntivi per la configurazione
        """
        super().__init__()

        self.config = config or SourceSeekingConfig(**kwargs)
        self.source_id = source_id
        self.render_mode = render_mode
        self.randomize_field = randomize_field

        # Setup dominio
        self.domain = DomainConfig(
            xmin=self.config.xmin,
            xmax=self.config.xmax,
            ymin=self.config.ymin,
            ymax=self.config.ymax,
            resolution=self.config.resolution
        )

        # Data Manager per gestione NC files
        self._data_manager: Optional[DataManager] = None
        if data_dir:
            self._data_manager = DataManager(
                data_dir=data_dir,
                domain_config=self.domain,
                preload_all=False,  # NON precaricare - carica on-demand per risparmiare RAM
            )

        # Campo di concentrazione
        if concentration_field is not None:
            self.field = concentration_field
        else:
            self._init_field()

        # Posizione sorgente (sempre da coordinate hardcodate nel campo)
        if self.field.source_position is not None:
            self.source_position = np.array(self.field.source_position)
        else:
            raise ValueError(
                f"ConcentrationField non ha source_position impostata. "
                f"Controlla che il file NC contenga S1/S2/S3 nel nome "
                f"o passa un campo con source_position."
            )

        # Stato agente
        self.state: Optional[AgentState] = None
        self.steps = 0
        self.prev_concentration = 0.0
        self.prev_distance = 0.0

        # Memory buffer per concentrazioni passate (usata nell'osservazione)
        self._concentration_memory: List[float] = [0.0] * self.config.memory_length

        # Memory buffer per spostamenti passati (Δx, Δy)
        self._displacement_memory: List[Tuple[float, float]] = [(0.0, 0.0)] * self.config.memory_length

        # Allowed sources per curriculum learning
        self.allowed_sources: List[str] = ['S1', 'S2', 'S3']

        # History per analisi
        self.trajectory: List[np.ndarray] = []
        self.concentration_history: List[float] = []

        # Setup spaces
        self._setup_observation_space()
        self._setup_action_space()

        # Rendering
        self._fig = None
        self._ax = None

    def _init_field(self):
        """Inizializza il campo di concentrazione."""
        if not self._data_manager:
            raise ValueError(
                "Nessun DataManager configurato. "
                "Passa data_dir al costruttore o concentration_field direttamente."
            )
        if self.randomize_field:
            self.field, self.source_id = self._data_manager.get_random_field()
        else:
            self.field = self._data_manager.get_concentration_field(source_id=self.source_id)

    def _setup_observation_space(self):
        """Configura lo spazio delle osservazioni.
        
        Osservazione (32 valori):
        - 1 concentrazione corrente
        - 9 concentrazioni passate (memory_length)
        - 9 * 2 spostamenti passati (Δx, Δy) normalizzati
        - 4 sensori terra direzionali (N, S, E, W) - binari
        """
        obs_dim = 1 + self.config.memory_length + self.config.memory_length * 2 + 4
        # = 1 + 9 + 18 + 4 = 32

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _setup_action_space(self):
        """Configura lo spazio delle azioni.
        
        4 azioni discrete:
          0 = Nord  (+y)
          1 = Sud   (-y)
          2 = Est   (+x)
          3 = Ovest (-x)
        """
        self.action_space = spaces.Discrete(self.config.n_discrete_actions)

    def _spawn_on_plume(self) -> Tuple[float, float]:
        """Spawn su una cella con concentrazione > soglia, con vincoli di distanza dalla sorgente."""

        # Imposta il timestep al frame configurato (1440 = metà simulazione)
        if self.field.n_timesteps > 1:
            start_frame = min(self.config.spawn_start_frame, self.field.n_timesteps - 1)
            self.field.set_time(start_frame)

        # Ottieni il campo corrente
        field_data = self.field.get_current_field()

        # Sostituisci NaN con 0
        field_clean = np.nan_to_num(field_data, nan=0.0)

        # Trova TUTTE le celle con concentrazione > soglia
        valid_mask = field_clean > self.config.spawn_conc_threshold
        valid_indices = np.where(valid_mask)

        if len(valid_indices[0]) == 0:
            # Nessuna cella valida, fallback vicino alla sorgente
            print("WARNING: No plume cells found, spawning near source")
            return (
                self.source_position[0] + self.np_random.uniform(-50, 50),
                self.source_position[1] + self.np_random.uniform(-50, 50)
            )

        # Converti indici in coordinate
        y_coords = self.field.y_coords[valid_indices[0]]
        x_coords = self.field.x_coords[valid_indices[1]]

        # Calcola distanze dalla sorgente
        distances = np.sqrt(
            (x_coords - self.source_position[0])**2 +
            (y_coords - self.source_position[1])**2
        )

        # Applica vincoli di distanza minima e massima dalla sorgente
        distance_mask = (
            (distances >= self.config.spawn_min_distance) &
            (distances <= self.config.spawn_max_distance)
        )

        valid_x = x_coords[distance_mask]
        valid_y = y_coords[distance_mask]

        if len(valid_x) == 0:
            # Rilassa vincoli se nessun punto valido
            print("WARNING: No points satisfy distance constraints, relaxing...")
            valid_x = x_coords
            valid_y = y_coords

        # Filtra punti troppo vicini alla terra
        land_safe_mask = np.array([
            self._min_distance_to_land(valid_x[i], valid_y[i]) >= self.config.spawn_min_land_distance
            for i in range(len(valid_x))
        ])
        
        if np.any(land_safe_mask):
            valid_x = valid_x[land_safe_mask]
            valid_y = valid_y[land_safe_mask]
        else:
            print(f"WARNING: No points satisfy min_land_distance={self.config.spawn_min_land_distance}m, using all")

        # Scegli una cella random tra quelle valide
        idx = self.np_random.integers(len(valid_x))
        x = valid_x[idx]
        y = valid_y[idx]

        # Aggiungi piccolo rumore per non essere esattamente sul centro cella
        x += self.np_random.uniform(-5, 5)
        y += self.np_random.uniform(-5, 5)

        return (float(x), float(y))

    def _get_observation(self) -> np.ndarray:
        """Costruisce il vettore di osservazione (32 valori) RAW.
        
        I valori NON vengono normalizzati manualmente: la normalizzazione
        è delegata interamente a VecNormalize (running mean/std adattiva).
        
        Struttura:
        - [0]      : concentrazione corrente
        - [1:10]   : 9 concentrazioni passate
        - [10:28]  : 9 spostamenti passati (Δx, Δy) in metri
        - [28:32]  : 4 sensori terra direzionali (N, S, E, W)
        """
        obs = []

        # Concentrazione al centro (1 valore)
        center_conc = self.field.get_concentration(self.state.x, self.state.y)
        obs.append(center_conc)

        # 9 concentrazioni passate
        for past_conc in self._concentration_memory:
            obs.append(past_conc)

        # 9 spostamenti passati (Δx, Δy) in metri
        for dx, dy in self._displacement_memory:
            obs.append(dx)
            obs.append(dy)

        # 4 sensori terra direzionali (distanza di sensing = max_velocity * dt)
        sensing_dist = self.config.max_velocity * self.config.dt
        x, y = self.state.x, self.state.y
        land_sensors = [
            float(self.field.is_land(x, y + sensing_dist)),  # Nord
            float(self.field.is_land(x, y - sensing_dist)),  # Sud
            float(self.field.is_land(x + sensing_dist, y)),  # Est
            float(self.field.is_land(x - sensing_dist, y)),  # Ovest
        ]
        obs.extend(land_sensors)

        return np.array(obs, dtype=np.float32)

    # Mappa azioni discrete: 0=Nord(+y), 1=Sud(-y), 2=Est(+x), 3=Ovest(-x)
    _ACTION_MAP = {
        0: (0.0, 1.0),   # Nord  (+y)
        1: (0.0, -1.0),  # Sud   (-y)
        2: (1.0, 0.0),   # Est   (+x)
        3: (-1.0, 0.0),  # Ovest (-x)
    }

    def _apply_action(self, action):
        """Applica l'azione discreta all'agente.
        
        Azioni: 0=Nord, 1=Sud, 2=Est, 3=Ovest.
        Spostamento fisso = max_velocity * dt (10m per default).
        """
        action_int = int(action)
        dx_dir, dy_dir = self._ACTION_MAP[action_int]

        vx = dx_dir * self.config.max_velocity
        vy = dy_dir * self.config.max_velocity

        # Aggiorna stato
        self.state.vx = vx
        self.state.vy = vy
        self.state.x += vx * self.config.dt
        self.state.y += vy * self.config.dt



    def _check_boundary(self) -> bool:
        """Verifica se l'agente è fuori dal dominio."""
        return (
            self.state.x < self.config.xmin or
            self.state.x > self.config.xmax or
            self.state.y < self.config.ymin or
            self.state.y > self.config.ymax
        )

    def _check_on_land(self) -> bool:
        """Verifica se l'agente è sulla terra (usa land_mask del campo)."""
        return getattr(self, '_on_land', False)

    def _check_source_reached(self) -> bool:
        """Verifica se l'agente ha raggiunto la sorgente."""
        distance = np.sqrt(
            (self.state.x - self.source_position[0])**2 +
            (self.state.y - self.source_position[1])**2
        )
        return distance < self.config.source_distance_threshold

    def _min_distance_to_land(self, x: float, y: float) -> float:
        """Ritorna la distanza minima dalla terra usando la distance map precomputata.
        
        Usa la EDT (Euclidean Distance Transform) calcolata una volta sola
        al caricamento del campo — O(1) invece di O(160) chiamate a is_land().
        
        Returns:
            Distanza in metri dalla terra più vicina (0 se sulla terra, max 100m)
        """
        dist = self.field.get_land_distance(x, y)
        return min(dist, 100.0)  # Cap a 100m per coerenza con il comportamento precedente

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calcola il reward basato su:
        1. Bonus sorgente raggiunta (terminale dominante)
        2. Penalità terra e bordi (terminale)
        3. Reward distanza (segnale dominante continuo)
        4. Reward binario plume (+0.3 dentro, -0.3 fuori)
        5. Penalità tempo
        """
        reward = 0.0
        info = {}

        # Stato corrente
        current_distance = np.sqrt(
            (self.state.x - self.source_position[0])**2 +
            (self.state.y - self.source_position[1])**2
        )

        # Verifica terra tramite maschera (non più NaN)
        on_land = self.field.is_land(self.state.x, self.state.y)
        current_conc = self.field.get_concentration(self.state.x, self.state.y)

        # ============================================================
        # 1. BONUS SORGENTE RAGGIUNTA (TERMINALE DOMINANTE)
        # ============================================================
        if self._check_source_reached():
            time_bonus = max(0, (self.config.max_steps - self.steps) / self.config.max_steps * 50)
            total_bonus = self.config.source_found_reward + time_bonus
            reward += total_bonus
            info['source_found'] = self.config.source_found_reward
            info['time_bonus'] = time_bonus

        # ============================================================
        # 2. PENALITÀ TERRA (configurabile, default -50)
        # ============================================================
        if on_land:
            reward += self.config.land_penalty
            info['on_land_penalty'] = self.config.land_penalty
            self._on_land = True
        else:
            self._on_land = False

        # ============================================================
        # 3. PENALITÀ USCITA DAL DOMINIO (-10)
        # ============================================================
        if self._check_boundary():
            reward += self.config.boundary_penalty
            info['boundary'] = self.config.boundary_penalty

        # ============================================================
        # 4. REWARD DISTANZA (SEGNALE DOMINANTE CONTINUO)
        # ============================================================
        distance_improvement = self.prev_distance - current_distance
        distance_reward = distance_improvement * 5.0 * self.config.distance_reward_multiplier
        reward += distance_reward
        info['distance_reward'] = distance_reward

        # ============================================================
        # 5. REWARD BINARIO PLUME (+0.3 dentro, -0.3 fuori)
        # ============================================================
        if current_conc > self.config.plume_threshold:
            reward += self.config.plume_reward_positive
            info['plume_reward'] = self.config.plume_reward_positive
        else:
            reward += self.config.plume_reward_negative
            info['plume_reward'] = self.config.plume_reward_negative

        # ============================================================
        # 6. PENALITÀ TEMPO (time efficiency)
        # ============================================================
        reward += self.config.step_penalty  # -0.1
        info['time_penalty'] = self.config.step_penalty

        # ============================================================
        # 7. PENALITÀ PROGRESSIVA AVVICINAMENTO TERRA
        # Land proximity penalty (penalità progressiva vicino alla terra)
        dist_to_land = self._min_distance_to_land(self.state.x, self.state.y)
        if (dist_to_land < self.config.land_proximity_threshold 
                and not on_land):
            # Penalità lineare: 0 a threshold, max a 0
            proximity_penalty = self.config.land_proximity_penalty_max * (
                (self.config.land_proximity_threshold - dist_to_land) / self.config.land_proximity_threshold
            )
            reward += proximity_penalty
            info['land_proximity_penalty'] = proximity_penalty

        # Aggiorna valori precedenti
        self.prev_concentration = current_conc
        self.prev_distance = current_distance

        # Info aggiuntive
        info['total_reward'] = reward
        info['distance_to_source'] = current_distance
        info['concentration'] = current_conc
        info['on_land'] = on_land
        info['steps'] = self.steps

        return reward, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset dell'ambiente.

        Args:
            seed: Seed per la riproducibilità
            options: Opzioni aggiuntive (es. 'spawn_position')

        Returns:
            observation: Osservazione iniziale
            info: Informazioni aggiuntive
        """
        super().reset(seed=seed)

        # Curriculum: scegli sorgente random tra quelle consentite, poi scenario random
        if self.randomize_field and self._data_manager:
            source = self.np_random.choice(self.allowed_sources)
            self.field, self.source_id = self._data_manager.get_random_field_for_source(source)
            # Aggiorna posizione sorgente (sempre da coordinate hardcodate)
            if self.field.source_position is not None:
                self.source_position = np.array(self.field.source_position)
            else:
                raise ValueError(
                    f"Campo per {source} non ha source_position. "
                    f"Controlla DataManager.SOURCE_CONFIGS e nome file NC."
                )

        # Imposta timestep al frame 1440 (metà simulazione)
        if self.field.n_timesteps > 1:
            self._start_time_idx = min(self.config.spawn_start_frame, self.field.n_timesteps - 1)
            self.field.set_time(self._start_time_idx)
        else:
            self._start_time_idx = 0

        # Determina posizione iniziale
        if options and 'spawn_position' in options:
            spawn_pos = options['spawn_position']
        else:
            spawn_pos = self._spawn_on_plume()

        # Inizializza stato agente
        self.state = AgentState(x=spawn_pos[0], y=spawn_pos[1])
        self.steps = 0

        # Inizializza valori per il reward
        self.prev_concentration = self.field.get_concentration(self.state.x, self.state.y)
        if np.isnan(self.prev_concentration):
            self.prev_concentration = 0.0
        self.prev_distance = np.sqrt(
            (self.state.x - self.source_position[0])**2 +
            (self.state.y - self.source_position[1])**2
        )

        # Reset history
        self.trajectory = [self.state.position.copy()]
        self.concentration_history = [self.prev_concentration]

        # Reset memoria concentrazioni passate (9 valori)
        self._concentration_memory = [0.0] * self.config.memory_length

        # Reset memoria spostamenti passati (9 coppie Δx, Δy)
        self._displacement_memory = [(0.0, 0.0)] * self.config.memory_length

        observation = self._get_observation()
        info = {
            'spawn_position': spawn_pos,
            'source_position': self.source_position.tolist(),
            'source_id': self.source_id,
            'initial_distance': self.prev_distance,
            'initial_concentration': self.prev_concentration
        }

        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Esegue un passo di simulazione.

        Args:
            action: Azione da eseguire

        Returns:
            observation: Nuova osservazione
            reward: Reward ottenuto
            terminated: True se episodio terminato (successo/fallimento)
            truncated: True se episodio troncato (max steps)
            info: Informazioni aggiuntive
        """
        self.steps += 1

        # Converti azione se necessario
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)

        # Registra posizione prima dell'azione
        old_x, old_y = self.state.x, self.state.y

        # Applica azione
        self._apply_action(action)

        # Calcola spostamento e aggiorna memoria (Δx, Δy)
        dx = self.state.x - old_x
        dy = self.state.y - old_y
        self._displacement_memory.pop(0)
        self._displacement_memory.append((dx, dy))

        # Avanza il tempo del campo se time-varying (partendo dal frame di start)
        if self.field.n_timesteps > 1:
            time_offset = int(self.steps * self.config.dt / 60)  # assumendo dt NC = 1 min
            time_idx = self._start_time_idx + time_offset
            # Clamp index to valid range
            time_idx = max(0, min(time_idx, self.field.n_timesteps - 1))
            self.field.set_time(time_idx)

        # Calcola reward
        reward, reward_info = self._compute_reward(action)

        # Registra traiettoria
        self.trajectory.append(self.state.position.copy())
        conc_now = self.field.get_concentration(self.state.x, self.state.y)
        self.concentration_history.append(conc_now)

        # Aggiorna memoria concentrazioni (FIFO: rimuovi più vecchio, aggiungi attuale)
        self._concentration_memory.pop(0)
        self._concentration_memory.append(conc_now)

        # Controlla terminazione
        terminated = False
        truncated = False

        if self._check_source_reached():
            terminated = True
        elif self._check_boundary():
            terminated = True
        elif self._check_on_land():
            terminated = True  # Termina se va sulla terra

        if self.steps >= self.config.max_steps:
            truncated = True

        # Osservazione
        observation = self._get_observation()

        # Info
        info = {
            **reward_info,
            'steps': self.steps,
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'source_reached': self._check_source_reached(),
            'out_of_bounds': self._check_boundary(),
            'on_land': self._check_on_land()
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renderizza l'ambiente."""
        if self.render_mode is None:
            return None

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError:
            return None

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 8))

        self._ax.clear()

        # Plot campo di concentrazione
        field_data = self.field.get_current_field()
        extent = [self.config.xmin, self.config.xmax,
                  self.config.ymin, self.config.ymax]

        im = self._ax.imshow(
            field_data,
            extent=extent,
            origin='lower',
            cmap='YlOrRd',
            aspect='auto',
            alpha=0.7
        )

        # Plot sorgente
        self._ax.scatter(
            self.source_position[0],
            self.source_position[1],
            c='red', s=200, marker='*',
            label='Source', zorder=10
        )

        # Plot soglia sorgente
        circle = Circle(
            self.source_position,
            self.config.source_distance_threshold,
            fill=False, color='red', linestyle='--'
        )
        self._ax.add_patch(circle)

        # Plot traiettoria
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            self._ax.plot(
                traj[:, 0], traj[:, 1],
                'b-', linewidth=2, alpha=0.7,
                label='Trajectory'
            )

        # Plot agente
        self._ax.scatter(
            self.state.x, self.state.y,
            c='blue', s=100, marker='o',
            label='Agent', zorder=11
        )

        # Plot direzione
        if self.state.vx != 0 or self.state.vy != 0:
            self._ax.arrow(
                self.state.x, self.state.y,
                self.state.vx * 50, self.state.vy * 50,
                head_width=30, head_length=20,
                fc='blue', ec='blue'
            )

        self._ax.set_xlabel('X (m)')
        self._ax.set_ylabel('Y (m)')
        self._ax.set_title(
            f'HYDRAS Source Seeking - Step {self.steps}\n'
            f'Concentration: {self.concentration_history[-1]:.1f} g/m³'
        )
        self._ax.legend(loc='upper right')

        plt.tight_layout()

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.01)
            return None
        else:  # rgb_array
            self._fig.canvas.draw()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        """Chiude l'ambiente."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# Registra l'ambiente con Gymnasium
gym.register(
    id='SourceSeeking-v0',
    entry_point='utils.source_seeking_env:SourceSeekingEnv',
)


if __name__ == "__main__":
    # Test dell'ambiente
    print("Testing SourceSeekingEnv...")

    env = SourceSeekingEnv(
        source_id='S1',
        render_mode=None,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Test alcuni step con azioni random
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode ended at step {i+1}")
            print(f"  Reason: {'Source reached' if info['source_reached'] else 'Out of bounds' if info['out_of_bounds'] else 'Max steps'}")
            break

    print(f"\nTotal reward: {total_reward:.2f}")

    env.close()
    print("\nEnvironment test completed!")