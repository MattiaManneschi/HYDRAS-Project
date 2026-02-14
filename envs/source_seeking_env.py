"""
HYDRAS Source Seeking - Gymnasium Environment
Ambiente per l'addestramento di agenti RL per la localizzazione
di sorgenti di inquinante in ambiente marino.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import sys
from pathlib import Path

# Aggiungi il path per gli import locali
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (
    ConcentrationField, DataManager, DomainConfig, SyntheticPlumeGenerator
)


@dataclass
class AgentState:
    """Stato corrente dell'agente."""
    x: float  # Posizione x (UTM)
    y: float  # Posizione y (UTM)
    vx: float = 0.0  # Velocità x
    vy: float = 0.0  # Velocità y
    heading: float = 0.0  # Orientamento (radianti)
    
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
    max_velocity: float = 1.5  # m/s
    sensor_radius: float = 50  # m
    n_sensors: int = 8  # punti di campionamento attorno all'agente

    # Episode
    dt: float = 10.0  # s
    max_steps: int = 500

    # Spawn
    spawn_margin: float = 100  # m dal bordo
    spawn_mode: str = "random"  # "random", "fixed", "far_from_source"
    fixed_spawn: Optional[Tuple[float, float]] = None

    # Reward
    source_distance_threshold: float = 30  # m
    source_found_reward: float = 100.0
    step_penalty: float = -0.1
    boundary_penalty: float = -10.0
    gradient_reward_scale: float = 10.0
    concentration_reward_scale: float = 1.0
    distance_reward_multiplier: float = 1.0  # Moltiplicatore dinamico per reward scaling

    # Source detection
    auto_detect_source: bool = False  # Se True, rileva sorgente dal max globale

    # Observation
    include_velocity: bool = True
    include_position: bool = True
    include_gradient: bool = True
    normalize_observations: bool = True

    # Action
    action_type: str = "continuous"  # "continuous" o "discrete"
    n_discrete_actions: int = 8


class SourceSeekingEnv(gym.Env):
    """
    Ambiente Gymnasium per il source seeking di inquinanti marini.

    L'agente (AUV) deve navigare in un campo di concentrazione
    per trovare la sorgente dell'inquinante.

    Observation Space:
        - Concentrazioni campionate in n_sensors punti attorno all'agente
        - Concentrazione nel punto corrente
        - Gradiente della concentrazione (opzionale)
        - Posizione normalizzata (opzionale)
        - Velocità normalizzata (opzionale)

    Action Space:
        - Continuous: [vx, vy] velocità normalizzate in [-1, 1]
        - Discrete: 8 direzioni cardinali + stazionario

    Reward:
        - Bonus grande per trovare la sorgente
        - Reward proporzionale all'aumento di concentrazione
        - Reward per seguire il gradiente positivo
        - Penalità per step (incentiva velocità)
        - Penalità per uscire dai bordi
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
        if data_dir or randomize_field:
            self._data_manager = DataManager(
                data_dir=data_dir,
                use_synthetic=True,
                domain_config=self.domain,
                preload_all=False,  # NON precaricare - carica on-demand per risparmiare RAM
                source_id_filter=source_id  # Filtra solo file della sorgente specificata
            )

        # Campo di concentrazione
        if concentration_field is not None:
            self.field = concentration_field
        else:
            self._init_field()

        # Posizione sorgente
        # Opzione 1: usa coordinate hardcodate
        # Opzione 2: rileva automaticamente dal massimo globale di concentrazione
        if getattr(self.config, 'auto_detect_source', False):
            detected_source = self.field.find_source_from_concentration()
            self.source_position = np.array(detected_source)
            print(f"  Source auto-detected at: ({detected_source[0]:.0f}, {detected_source[1]:.0f})")
        elif self.field.source_position is not None:
            self.source_position = np.array(self.field.source_position)
        else:
            # Fallback: rileva automaticamente
            detected_source = self.field.find_source_from_concentration()
            self.source_position = np.array(detected_source)

        # Stato agente
        self.state: Optional[AgentState] = None
        self.steps = 0
        self.prev_concentration = 0.0
        self.prev_distance = 0.0

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
        if self._data_manager:
            if self.randomize_field:
                self.field, self.source_id = self._data_manager.get_random_field()
            else:
                self.field = self._data_manager.get_concentration_field(source_id=self.source_id)
        else:
            dm = DataManager(use_synthetic=True, domain_config=self.domain)
            self.field = dm.get_concentration_field(source_id=self.source_id)

    def _setup_observation_space(self):
        """Configura lo spazio delle osservazioni."""
        obs_dim = 0

        # Concentrazioni campionate
        obs_dim += self.config.n_sensors + 1  # sensors + centro

        # Gradiente
        if self.config.include_gradient:
            obs_dim += 2

        # Posizione
        if self.config.include_position:
            obs_dim += 2

        # Velocità
        if self.config.include_velocity:
            obs_dim += 2

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _setup_action_space(self):
        """Configura lo spazio delle azioni."""
        if self.config.action_type == "continuous":
            # Velocità normalizzate in [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            )
        else:
            # Direzioni discrete
            self.action_space = spaces.Discrete(self.config.n_discrete_actions + 1)

    def _get_spawn_position(self) -> Tuple[float, float]:
        """Determina la posizione iniziale dell'agente."""
        margin = self.config.spawn_margin

        if self.config.spawn_mode == "fixed" and self.config.fixed_spawn:
            return self.config.fixed_spawn

        elif self.config.spawn_mode == "on_plume":
            # Spawn direttamente sulla scia (per imparare a seguirla)
            return self._spawn_on_plume()

        elif self.config.spawn_mode == "near_plume":
            # Spawn vicino alla scia (entro 100-300m)
            return self._spawn_near_plume(min_dist=100, max_dist=300)

        elif self.config.spawn_mode == "near_shore":
            # Spawn vicino alla costa (come da specifiche relatore)
            return self._spawn_near_shore()

        elif self.config.spawn_mode == "strong_gradient":
            # Spawn dove il gradiente è forte (vicino alla sorgente)
            return self._spawn_strong_gradient(min_grad=0.3)

        elif self.config.spawn_mode == "near_source":
            # Spawn entro una certa distanza dalla sorgente
            return self._spawn_near_source(max_dist=300)

        elif self.config.spawn_mode == "curriculum":
            # Curriculum learning: distanza aumenta con il training
            max_dist = getattr(self, '_curriculum_distance', 200)
            return self._spawn_near_plume(min_dist=50, max_dist=max_dist)

        elif self.config.spawn_mode == "far_from_source":
            # Spawn lontano dalla sorgente
            while True:
                x = np.random.uniform(
                    self.config.xmin + margin,
                    self.config.xmax - margin
                )
                y = np.random.uniform(
                    self.config.ymin + margin,
                    self.config.ymax - margin
                )
                dist = np.sqrt(
                    (x - self.source_position[0])**2 +
                    (y - self.source_position[1])**2
                )
                # Almeno 500m dalla sorgente
                if dist > 500:
                    return (x, y)

        else:  # random
            x = np.random.uniform(
                self.config.xmin + margin,
                self.config.xmax - margin
            )
            y = np.random.uniform(
                self.config.ymin + margin,
                self.config.ymax - margin
            )
            return (x, y)

    def _spawn_strong_gradient(self, min_grad: float = 0.3) -> Tuple[float, float]:
        """
        Spawn in un punto con gradiente forte.
        Questi punti sono tipicamente vicini alla sorgente.
        """
        # Imposta timestep dove il plume è sviluppato
        if self.field.n_timesteps > 1:
            mid_time = self.field.n_timesteps // 2
            self.field.set_time(mid_time)

        # Ottieni il campo corrente
        field_data = self.field.get_current_field()
        field_clean = np.nan_to_num(field_data, nan=0.0)

        # Calcola il gradiente su tutto il campo
        grad_y, grad_x = np.gradient(field_clean)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Trova punti con gradiente forte E concentrazione > 0
        valid_mask = (grad_magnitude > min_grad) & (field_clean > 0.5)
        valid_indices = np.where(valid_mask)

        if len(valid_indices[0]) == 0:
            # Fallback: riduci soglia gradiente
            valid_mask = (grad_magnitude > min_grad / 2) & (field_clean > 0.1)
            valid_indices = np.where(valid_mask)

        if len(valid_indices[0]) == 0:
            # Fallback estremo: spawn on_plume
            print("WARNING: No strong gradient points, falling back to on_plume")
            return self._spawn_on_plume()

        # Scegli un punto random tra quelli con gradiente forte
        idx = np.random.randint(len(valid_indices[0]))
        y_idx = valid_indices[0][idx]
        x_idx = valid_indices[1][idx]

        x = float(self.field.x_coords[x_idx])
        y = float(self.field.y_coords[y_idx])

        # Aggiungi piccolo rumore
        x += np.random.uniform(-5, 5)
        y += np.random.uniform(-5, 5)

        return (x, y)

    def _spawn_near_source(self, max_dist: float = 300) -> Tuple[float, float]:
        """
        Spawn entro una certa distanza dalla sorgente, MA solo su celle con concentrazione > 0.
        """
        max_attempts = 1000

        for _ in range(max_attempts):
            # Genera punto random attorno alla sorgente
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(50, max_dist)  # minimo 50m dalla sorgente

            x = self.source_position[0] + dist * np.cos(angle)
            y = self.source_position[1] + dist * np.sin(angle)

            # Verifica bounds
            if (self.config.xmin < x < self.config.xmax and
                self.config.ymin < y < self.config.ymax):

                # Verifica che sia sulla scia (conc > 0.5) e non sulla terra
                conc = self.field.get_concentration(x, y)
                if not np.isnan(conc) and conc > 0.5:
                    return (x, y)

        # Fallback: spawn on_plume se non troviamo punti validi
        print("WARNING: near_source fallback to on_plume")
        return self._spawn_on_plume()

    def _spawn_near_shore(self, max_dist_from_shore: float = 200) -> Tuple[float, float]:
        """
        Spawn vicino alla costa (NaN = terra).
        Trova punti di mare vicini ai punti di terra.
        """
        max_attempts = 1000

        # Imposta timestep dove il plume è sviluppato
        if self.field.n_timesteps > 1:
            mid_time = self.field.n_timesteps // 2
            self.field.set_time(mid_time)

        # Ottieni il campo corrente
        field_data = self.field.get_current_field()

        # Trova la maschera della terra (NaN)
        land_mask = np.isnan(field_data)

        # Trova la maschera del mare (non NaN)
        sea_mask = ~land_mask

        # Trova i punti di mare vicini alla costa
        # Dilata la maschera terra e fai intersezione con mare
        from scipy.ndimage import binary_dilation

        # Dilata la terra di ~20 celle (200m con risoluzione 10m)
        n_dilate = int(max_dist_from_shore / 10)  # assumendo risoluzione ~10m
        dilated_land = binary_dilation(land_mask, iterations=n_dilate)

        # Punti di mare vicini alla costa = mare AND terra dilatata
        near_shore_mask = sea_mask & dilated_land

        # Trova indici validi
        valid_indices = np.where(near_shore_mask)

        if len(valid_indices[0]) == 0:
            # Fallback: spawn random nel mare
            print("WARNING: No near-shore points found, using random sea spawn")
            return self._spawn_random_sea()

        # Scegli un punto random tra quelli vicini alla costa
        idx = np.random.randint(len(valid_indices[0]))
        y_idx = valid_indices[0][idx]
        x_idx = valid_indices[1][idx]

        x = float(self.field.x_coords[x_idx])
        y = float(self.field.y_coords[y_idx])

        # Aggiungi piccolo rumore
        x += np.random.uniform(-5, 5)
        y += np.random.uniform(-5, 5)

        return (x, y)

    def _spawn_random_sea(self) -> Tuple[float, float]:
        """Spawn in un punto random del mare (non NaN)."""
        max_attempts = 1000

        for _ in range(max_attempts):
            x = np.random.uniform(self.config.xmin, self.config.xmax)
            y = np.random.uniform(self.config.ymin, self.config.ymax)

            conc = self.field.get_concentration(x, y)

            # Verifica che non sia terra (NaN)
            if not np.isnan(conc):
                return (x, y)

        # Fallback estremo
        return (self.config.xmin + 100, self.config.ymin + 100)

    def _spawn_on_plume(self) -> Tuple[float, float]:
        """Spawn su una cella con concentrazione > 0."""

        # Usa un timestep dove il plume è sviluppato (non t=0)
        if self.field.n_timesteps > 1:
            mid_time = self.field.n_timesteps // 2
            self.field.set_time(mid_time)

        # Ottieni il campo corrente
        field_data = self.field.get_current_field()

        # Sostituisci NaN con 0
        field_clean = np.nan_to_num(field_data, nan=0.0)

        # Trova TUTTE le celle con concentrazione > 0.5
        valid_mask = field_clean > 0.5
        valid_indices = np.where(valid_mask)

        if len(valid_indices[0]) == 0:
            # Nessuna cella valida, fallback vicino alla sorgente
            print("WARNING: No plume cells found, spawning near source")
            return (
                self.source_position[0] + np.random.uniform(-50, 50),
                self.source_position[1] + np.random.uniform(-50, 50)
            )

        # Scegli una cella random tra quelle valide
        idx = np.random.randint(len(valid_indices[0]))
        y_idx = valid_indices[0][idx]
        x_idx = valid_indices[1][idx]

        # Converti indici in coordinate
        x = self.field.x_coords[x_idx]
        y = self.field.y_coords[y_idx]

        # Aggiungi piccolo rumore per non essere esattamente sul centro cella
        x += np.random.uniform(-5, 5)
        y += np.random.uniform(-5, 5)

        return (float(x), float(y))

    def _spawn_near_plume(self, min_dist: float = 50, max_dist: float = 300) -> Tuple[float, float]:
        """Spawn a una certa distanza dalla scia."""
        max_attempts = 1000

        # Prima trova un punto sulla scia
        plume_x, plume_y = self._spawn_on_plume()

        for _ in range(max_attempts):
            # Genera punto a distanza random dalla scia
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(min_dist, max_dist)

            x = plume_x + dist * np.cos(angle)
            y = plume_y + dist * np.sin(angle)

            # Verifica bounds
            if (self.config.xmin < x < self.config.xmax and
                self.config.ymin < y < self.config.ymax):

                # Verifica che non sia sulla terra
                conc = self.field.get_concentration(x, y)
                if not np.isnan(conc):
                    return (x, y)

        # Fallback
        return (plume_x, plume_y)

    def _get_observation(self) -> np.ndarray:
        """Costruisce il vettore di osservazione."""
        obs = []

        # Concentrazione al centro
        center_conc = self.field.get_concentration(self.state.x, self.state.y)
        if np.isnan(center_conc):
            center_conc = 0.0

        # Concentrazioni campionate attorno all'agente
        sensor_concs = self._sample_concentrations()

        # Normalizza le concentrazioni
        max_conc = max(self.field.max_concentration, 1.0)
        obs.extend(sensor_concs / max_conc)
        obs.append(center_conc / max_conc)

        # Gradiente
        if self.config.include_gradient:
            gradient = self.field.get_gradient(self.state.x, self.state.y)
            gradient = np.nan_to_num(gradient, nan=0.0)
            # Normalizza il gradiente
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 1e-6:
                gradient = gradient / grad_norm
            obs.extend(gradient)

        # Posizione normalizzata
        if self.config.include_position:
            x_norm = 2 * (self.state.x - self.config.xmin) / (self.config.xmax - self.config.xmin) - 1
            y_norm = 2 * (self.state.y - self.config.ymin) / (self.config.ymax - self.config.ymin) - 1
            obs.extend([x_norm, y_norm])

        # Velocità normalizzata
        if self.config.include_velocity:
            vx_norm = self.state.vx / self.config.max_velocity
            vy_norm = self.state.vy / self.config.max_velocity
            obs.extend([vx_norm, vy_norm])

        return np.array(obs, dtype=np.float32)

    def _sample_concentrations(self) -> np.ndarray:
        """Campiona la concentrazione in punti attorno all'agente."""
        angles = np.linspace(0, 2*np.pi, self.config.n_sensors, endpoint=False)
        r = self.config.sensor_radius

        concentrations = []
        for angle in angles:
            x = self.state.x + r * np.cos(angle)
            y = self.state.y + r * np.sin(angle)
            c = self.field.get_concentration(x, y)
            if np.isnan(c):
                c = 0.0
            concentrations.append(c)

        return np.array(concentrations)

    def _apply_action(self, action: np.ndarray):
        """Applica l'azione all'agente."""
        if self.config.action_type == "continuous":
            # action è [vx, vy] normalizzato in [-1, 1]
            vx = action[0] * self.config.max_velocity
            vy = action[1] * self.config.max_velocity
        else:
            # Azione discreta
            if action == self.config.n_discrete_actions:
                # Stazionario
                vx, vy = 0.0, 0.0
            else:
                # Direzione
                angle = action * 2 * np.pi / self.config.n_discrete_actions
                vx = self.config.max_velocity * np.cos(angle)
                vy = self.config.max_velocity * np.sin(angle)

        # Aggiorna stato
        self.state.vx = vx
        self.state.vy = vy
        self.state.x += vx * self.config.dt
        self.state.y += vy * self.config.dt

        if vx != 0 or vy != 0:
            self.state.heading = np.arctan2(vy, vx)

    def _check_boundary(self) -> bool:
        """Verifica se l'agente è fuori dal dominio."""
        return (
            self.state.x < self.config.xmin or
            self.state.x > self.config.xmax or
            self.state.y < self.config.ymin or
            self.state.y > self.config.ymax
        )

    def _check_on_land(self) -> bool:
        """Verifica se l'agente è sulla terra (NaN nel campo)."""
        return getattr(self, '_on_land', False)

    def _check_source_reached(self) -> bool:
        """Verifica se l'agente ha raggiunto la sorgente."""
        distance = np.sqrt(
            (self.state.x - self.source_position[0])**2 +
            (self.state.y - self.source_position[1])**2
        )
        return distance < self.config.source_distance_threshold

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calcola il reward basato su:
        1. Allineamento con il gradiente di concentrazione (gradient following)
        2. Distanza dalla sorgente
        3. Tempo trascorso (time efficiency)
        """
        reward = 0.0
        info = {}

        # Stato corrente
        current_distance = np.sqrt(
            (self.state.x - self.source_position[0])**2 +
            (self.state.y - self.source_position[1])**2
        )

        # Ottieni concentrazione (potrebbe essere NaN se terra)
        raw_conc = self.field.get_concentration(self.state.x, self.state.y)
        on_land = np.isnan(raw_conc)
        current_conc = 0.0 if on_land else raw_conc

        # Calcola gradiente di concentrazione
        gradient = self.field.get_gradient(self.state.x, self.state.y)
        grad_magnitude = np.linalg.norm(gradient)

        # Velocità corrente dell'agente
        velocity = np.array([self.state.vx, self.state.vy])
        vel_magnitude = np.linalg.norm(velocity)

        # ============================================================
        # 1. BONUS SORGENTE RAGGIUNTA (+100 + time bonus)
        # ============================================================
        if self._check_source_reached():
            time_bonus = max(0, (self.config.max_steps - self.steps) / self.config.max_steps * 50)
            total_bonus = self.config.source_found_reward + time_bonus
            reward += total_bonus
            info['source_found'] = self.config.source_found_reward
            info['time_bonus'] = time_bonus

        # ============================================================
        # 2. PENALITÀ TERRA (-50) e termina
        # ============================================================
        if on_land:
            reward += -50.0
            info['on_land_penalty'] = -50.0
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
        # 4. REWARD GRADIENTE (gradient following) - PRINCIPALE
        #    Il gradiente fornisce il segnale direzionale verso la sorgente
        # ============================================================
        if grad_magnitude > 1e-6 and vel_magnitude > 1e-6 and current_conc > 0.1:
            # Segui il gradiente se siamo sopra lo 0.1 di concentrazione
            alignment = np.dot(gradient, velocity) / (grad_magnitude * vel_magnitude)
            gradient_reward = alignment * 10.0  # AUMENTATO: max +10 per step
            reward += gradient_reward
            info['gradient_alignment'] = alignment
            info['gradient_magnitude'] = grad_magnitude
            info['gradient_reward'] = gradient_reward
        else:
            info['gradient_alignment'] = 0.0
            info['gradient_magnitude'] = grad_magnitude
            info['gradient_reward'] = 0.0

        # ============================================================
        # 5. REWARD DISTANZA (distance accuracy) - PRINCIPALE
        #    La distanza dalla sorgente è il segnale più affidabile
        # ============================================================
        distance_improvement = self.prev_distance - current_distance
        distance_reward = distance_improvement * 2.0 * self.config.distance_reward_multiplier  # SCALABILE
        reward += distance_reward
        info['distance_reward'] = distance_reward

        # ============================================================
        # 6. BONUS CONCENTRAZIONE
        #    Piccolo bonus per essere in zona con concentrazione
        # ============================================================
        if current_conc > 0.1:
            max_conc = max(self.field.max_concentration, 1.0)
            conc_bonus = (current_conc / max_conc) * 1.0  # max +1 per step
            reward += conc_bonus
            info['concentration_bonus'] = conc_bonus

        # ============================================================
        # 7. PENALITÀ TEMPO (time efficiency)
        # ============================================================
        reward += self.config.step_penalty  # -0.1
        info['time_penalty'] = self.config.step_penalty

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

        # Randomizza campo NC se abilitato
        if self.randomize_field and self._data_manager:
            self.field, self.source_id = self._data_manager.get_random_field()
            self.source_position = np.array(self.field.source_position)

        # PRIMA: Imposta timestep dove la concentrazione è massima (vicino alla sorgente)
        if self.field.n_timesteps > 1:
            # Trova il timestep con la concentrazione massima globale
            data_clean = np.nan_to_num(self.field.data, nan=0.0)
            max_idx = np.unravel_index(np.argmax(data_clean), data_clean.shape)
            best_time = max_idx[0]  # timestep del max globale
            self.field.set_time(best_time)

        # POI: Determina posizione iniziale (ora usa il timestep corretto)
        if options and 'spawn_position' in options:
            spawn_pos = options['spawn_position']
        else:
            spawn_pos = self._get_spawn_position()

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

        # Reset esplorazione
        self._visited_cells = set()

        observation = self._get_observation()
        info = {
            'spawn_position': spawn_pos,
            'source_position': self.source_position.tolist(),
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

        # Applica azione
        self._apply_action(action)

        # Avanza il tempo del campo se time-varying
        if self.field.n_timesteps > 1:
            time_idx = int(self.steps * self.config.dt / 60)  # assumendo dt NC = 1 min
            # Clamp index to valid range
            time_idx = max(0, min(time_idx, max(0, self.field.n_timesteps - 1)))
            self.field.set_time(time_idx)

        # Calcola reward
        reward, reward_info = self._compute_reward(action)

        # Registra traiettoria
        self.trajectory.append(self.state.position.copy())
        conc_now = self.field.get_concentration(self.state.x, self.state.y)
        if np.isnan(conc_now):
            conc_now = 0.0
        self.concentration_history.append(conc_now)

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

    def get_trajectory(self) -> np.ndarray:
        """Ritorna la traiettoria completa."""
        return np.array(self.trajectory)

    def get_statistics(self) -> Dict[str, Any]:
        """Ritorna statistiche dell'episodio."""
        trajectory = self.get_trajectory()

        # Distanza totale percorsa
        if len(trajectory) > 1:
            diffs = np.diff(trajectory, axis=0)
            total_distance = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            total_distance = 0.0

        # Distanza finale dalla sorgente
        final_distance = np.linalg.norm(trajectory[-1] - self.source_position)

        return {
            'total_distance': total_distance,
            'final_distance': final_distance,
            'n_steps': self.steps,
            'max_concentration': max(self.concentration_history),
            'final_concentration': self.concentration_history[-1],
            'source_reached': self._check_source_reached(),
            'efficiency': 1.0 - (final_distance / self.prev_distance) if self.prev_distance > 0 else 0.0
        }


# Registra l'ambiente con Gymnasium
gym.register(
    id='SourceSeeking-v0',
    entry_point='envs.source_seeking_env:SourceSeekingEnv',
)


if __name__ == "__main__":
    # Test dell'ambiente
    print("Testing SourceSeekingEnv...")

    env = SourceSeekingEnv(
        source_id='S1',
        render_mode=None,
        spawn_mode='far_from_source'
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
    print(f"Statistics: {env.get_statistics()}")

    env.close()
    print("\nEnvironment test completed!")