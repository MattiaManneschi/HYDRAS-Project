"""
HYDRAS Source Seeking - Data Loader
Gestisce il caricamento dei dati NetCDF dalle simulazioni MIKE21
e fornisce un generatore di dati sintetici per testing.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import warnings


@dataclass
class DomainConfig:
    """Configurazione del dominio spaziale."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    resolution: float
    
    @property
    def nx(self) -> int:
        return int((self.xmax - self.xmin) / self.resolution)
    
    @property
    def ny(self) -> int:
        return int((self.ymax - self.ymin) / self.resolution)
    
    @property
    def x_coords(self) -> np.ndarray:
        return np.linspace(self.xmin, self.xmax, self.nx)
    
    @property
    def y_coords(self) -> np.ndarray:
        return np.linspace(self.ymin, self.ymax, self.ny)


class ConcentrationField:
    """
    Rappresenta un campo di concentrazione 2D interpolabile.
    Supporta query spaziali e temporali.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        time_coords: Optional[np.ndarray] = None,
        source_position: Optional[Tuple[float, float]] = None
    ):
        """
        Args:
            data: Array di concentrazione [time, y, x] o [y, x]
            x_coords: Coordinate x della griglia
            y_coords: Coordinate y della griglia
            time_coords: Coordinate temporali (opzionale)
            source_position: Posizione della sorgente (x, y)
        """
        self.data = data
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.time_coords = time_coords
        self.source_position = source_position
        
        self._is_time_varying = data.ndim == 3
        self._current_time_idx = 0
        self._interpolator = None
        self._build_interpolator()
    
    def _build_interpolator(self):
        """Costruisce l'interpolatore per il timestep corrente."""
        if self._is_time_varying:
            field = self.data[self._current_time_idx]
        else:
            field = self.data
        
        self._interpolator = RegularGridInterpolator(
            (self.y_coords, self.x_coords),
            field,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    def set_time(self, time_idx: int):
        """Imposta il timestep corrente."""
        if not self._is_time_varying:
            return
        
        time_idx = np.clip(time_idx, 0, len(self.time_coords) - 1)
        if time_idx != self._current_time_idx:
            self._current_time_idx = time_idx
            self._build_interpolator()
    
    def advance_time(self, steps: int = 1):
        """Avanza il tempo di n steps."""
        if self._is_time_varying:
            new_idx = min(self._current_time_idx + steps, len(self.time_coords) - 1)
            self.set_time(new_idx)
    
    def get_concentration(self, x: float, y: float) -> float:
        """Ottiene la concentrazione in un punto."""
        return float(self._interpolator((y, x)))
    
    def get_concentration_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Ottiene la concentrazione per un batch di posizioni.
        
        Args:
            positions: Array [N, 2] con colonne (x, y)
        
        Returns:
            Array [N] di concentrazioni
        """
        # L'interpolatore vuole (y, x)
        yx_positions = positions[:, ::-1]
        return self._interpolator(yx_positions)
    
    def get_gradient(self, x: float, y: float, eps: float = 5.0) -> np.ndarray:
        """
        Calcola il gradiente della concentrazione in un punto.
        
        Args:
            x, y: Posizione
            eps: Step per la differenza finita (m)
        
        Returns:
            Array [2] con (dC/dx, dC/dy)
        """
        c_xp = self.get_concentration(x + eps, y)
        c_xm = self.get_concentration(x - eps, y)
        c_yp = self.get_concentration(x, y + eps)
        c_ym = self.get_concentration(x, y - eps)
        
        dcdx = (c_xp - c_xm) / (2 * eps)
        dcdy = (c_yp - c_ym) / (2 * eps)
        
        return np.array([dcdx, dcdy])
    
    def get_current_field(self) -> np.ndarray:
        """Ritorna il campo di concentrazione corrente [y, x]."""
        if self._is_time_varying:
            return self.data[self._current_time_idx]
        return self.data
    
    @property
    def max_concentration(self) -> float:
        """Massima concentrazione nel campo corrente."""
        return float(np.max(self.get_current_field()))
    
    @property
    def n_timesteps(self) -> int:
        """Numero di timestep disponibili."""
        if self._is_time_varying:
            return len(self.time_coords)
        return 1

    @property
    def current_time_idx(self) -> int:
        """Indice del timestep corrente."""
        return self._current_time_idx

    def find_source_from_concentration(self) -> Tuple[float, float]:
        """
        Trova la posizione della sorgente cercando il punto con concentrazione
        massima su TUTTI i timesteps (massimo globale spazio-temporale).

        Returns:
            (x, y): Coordinate della sorgente stimata
        """
        # Salva timestep corrente
        original_time = self._current_time_idx

        if self._is_time_varying:
            # Cerca il MASSIMO GLOBALE su tutti i timesteps
            # data shape: (time, y, x)
            data_clean = np.nan_to_num(self.data, nan=0.0)

            # Trova indice del massimo globale (t, y, x)
            max_flat_idx = np.argmax(data_clean)
            max_idx = np.unravel_index(max_flat_idx, data_clean.shape)
            t_idx, y_idx, x_idx = max_idx

            max_val = data_clean[t_idx, y_idx, x_idx]
        else:
            # Campo statico
            data_clean = np.nan_to_num(self.data, nan=0.0)
            max_idx = np.unravel_index(np.argmax(data_clean), data_clean.shape)
            y_idx, x_idx = max_idx
            max_val = data_clean[y_idx, x_idx]

        source_x = float(self.x_coords[x_idx])
        source_y = float(self.y_coords[y_idx])

        # Ripristina timestep originale
        self.set_time(original_time)

        return (source_x, source_y)


class NetCDFLoader:
    """
    Carica i dati di concentrazione dai file NetCDF prodotti da MIKE21.
    """

    def __init__(self, data_dir: Union[str, Path]):
        """
        Args:
            data_dir: Directory contenente i file NC
        """
        self.data_dir = Path(data_dir)
        self._nc_files: List[Path] = []
        self._scan_files()

    def _scan_files(self):
        """Scansiona la directory per file NC."""
        patterns = ["*.nc", "**/*.nc"]
        for pattern in patterns:
            self._nc_files.extend(self.data_dir.glob(pattern))
        self._nc_files = sorted(set(self._nc_files))

    @property
    def available_runs(self) -> List[str]:
        """Lista dei run disponibili."""
        return [f.stem for f in self._nc_files]

    def load(
        self,
        filename: str,
        concentration_var: str = "Concentration - component 1",
        x_var: str = "x",
        y_var: str = "y",
        time_var: str = "time"
    ) -> ConcentrationField:
        """
        Carica un file NetCDF.

        Args:
            filename: Nome del file (con o senza path)
            concentration_var: Nome della variabile di concentrazione
            x_var, y_var, time_var: Nomi delle coordinate

        Returns:
            ConcentrationField pronto per l'uso
        """
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError(
                "netCDF4 non installato. Esegui: pip install netCDF4"
            )

        # Trova il file
        filepath = self._find_file(filename)

        with nc.Dataset(filepath, 'r') as ds:
            # Leggi le coordinate
            x_coords = ds.variables[x_var][:]
            y_coords = ds.variables[y_var][:]

            # Leggi il tempo se presente
            time_coords = None
            if time_var in ds.variables:
                time_coords = ds.variables[time_var][:]

            # Leggi la concentrazione
            # Gestisce nomi con spazi/caratteri speciali
            conc_var = ds.variables[concentration_var]
            conc_data = conc_var[:]

            # Gestisci i valori mancanti (masked arrays e NaN)
            if hasattr(conc_data, 'mask'):
                # È un masked array - riempi con 0
                conc_data = np.ma.filled(conc_data, 0.0)

            # Sostituisci anche eventuali NaN rimasti
            conc_data = np.nan_to_num(conc_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Assicurati che sia float32 per efficienza
            conc_data = conc_data.astype(np.float32)

            # Debug info
            print(f"  Data loaded: min={conc_data.min():.2f}, max={conc_data.max():.2f}")

        # Estrai info sulla sorgente dal nome file
        source_pos = self._extract_source_position(filename)

        return ConcentrationField(
            data=conc_data,
            x_coords=x_coords,
            y_coords=y_coords,
            time_coords=time_coords,
            source_position=source_pos
        )

    def _find_file(self, filename: str) -> Path:
        """Trova il file nella directory."""
        # Prova come path diretto
        if Path(filename).exists():
            return Path(filename)

        # Cerca nella data_dir
        filepath = self.data_dir / filename
        if filepath.exists():
            return filepath

        # Aggiungi estensione se mancante
        if not filename.endswith('.nc'):
            filepath = self.data_dir / f"{filename}.nc"
            if filepath.exists():
                return filepath

        # Cerca per pattern
        for f in self._nc_files:
            if filename in f.name:
                return f

        raise FileNotFoundError(f"File {filename} non trovato in {self.data_dir}")

    def _extract_source_position(self, filename: str) -> Optional[Tuple[float, float]]:
        """Estrae la posizione della sorgente dal nome file."""
        # Coordinate delle sorgenti S1-S3 dal report
        source_coords = {
            'S1': (620100, 4796210),
            'S2': (619800, 4795900),
            'S3': (620200, 4795800),
        }

        for source_id, coords in source_coords.items():
            if source_id in filename:
                return coords

        return None


class SyntheticPlumeGenerator:
    """
    Genera campi di concentrazione sintetici per testing.
    Simula un plume di inquinante con advection-diffusion.
    """

    def __init__(
        self,
        domain: DomainConfig,
        source_position: Tuple[float, float],
        diffusion_coef: float = 5.0,
        advection_velocity: Tuple[float, float] = (0.1, 0.05),
        source_strength: float = 1000.0,
        dt: float = 10.0
    ):
        """
        Args:
            domain: Configurazione del dominio
            source_position: Posizione della sorgente (x, y) in coordinate UTM
            diffusion_coef: Coefficiente di diffusione (m²/s)
            advection_velocity: Velocità di advection (u, v) in m/s
            source_strength: Concentrazione alla sorgente (g/m³)
            dt: Timestep per l'evoluzione (s)
        """
        self.domain = domain
        self.source_position = source_position
        self.diffusion_coef = diffusion_coef
        self.advection_velocity = np.array(advection_velocity)
        self.source_strength = source_strength
        self.dt = dt

        # Griglia
        self.x_grid, self.y_grid = np.meshgrid(
            domain.x_coords, domain.y_coords
        )

        # Indice della sorgente nella griglia
        self._source_idx = self._get_source_index()

        # Campo di concentrazione corrente
        self._field = np.zeros((domain.ny, domain.nx))
        self._time = 0.0

    def _get_source_index(self) -> Tuple[int, int]:
        """Trova l'indice della sorgente nella griglia."""
        x_idx = np.argmin(np.abs(self.domain.x_coords - self.source_position[0]))
        y_idx = np.argmin(np.abs(self.domain.y_coords - self.source_position[1]))
        return (y_idx, x_idx)

    def reset(self):
        """Resetta il campo alla condizione iniziale."""
        self._field = np.zeros((self.domain.ny, self.domain.nx))
        self._time = 0.0

    def step(self, n_steps: int = 1):
        """
        Avanza la simulazione di n timestep.
        Usa un semplice schema advection-diffusion.
        """
        dx = self.domain.resolution

        for _ in range(n_steps):
            # Aggiungi sorgente
            self._field[self._source_idx] = self.source_strength

            # Diffusione (schema implicito approssimato con Gaussian filter)
            sigma = np.sqrt(2 * self.diffusion_coef * self.dt) / dx
            if sigma > 0.1:
                self._field = gaussian_filter(self._field, sigma=sigma, mode='constant')

            # Advection (upwind scheme)
            u, v = self.advection_velocity

            if abs(u) > 1e-6:
                shift_x = int(np.sign(u))
                self._field = (1 - abs(u) * self.dt / dx) * self._field + \
                             abs(u) * self.dt / dx * np.roll(self._field, shift_x, axis=1)

            if abs(v) > 1e-6:
                shift_y = int(np.sign(v))
                self._field = (1 - abs(v) * self.dt / dx) * self._field + \
                             abs(v) * self.dt / dx * np.roll(self._field, shift_y, axis=0)

            # Mantieni positivo
            self._field = np.maximum(self._field, 0.0)

            self._time += self.dt

    def get_field(self) -> ConcentrationField:
        """Ritorna il campo corrente come ConcentrationField."""
        return ConcentrationField(
            data=self._field.copy(),
            x_coords=self.domain.x_coords,
            y_coords=self.domain.y_coords,
            source_position=self.source_position
        )

    def generate_sequence(self, n_timesteps: int) -> ConcentrationField:
        """
        Genera una sequenza temporale completa.

        Args:
            n_timesteps: Numero di timestep da generare

        Returns:
            ConcentrationField con dimensione [time, y, x]
        """
        self.reset()

        fields = []
        times = []

        for t in range(n_timesteps):
            self.step()
            fields.append(self._field.copy())
            times.append(self._time)

        return ConcentrationField(
            data=np.array(fields),
            x_coords=self.domain.x_coords,
            y_coords=self.domain.y_coords,
            time_coords=np.array(times),
            source_position=self.source_position
        )

    def generate_steady_state(self, n_warmup_steps: int = 500) -> ConcentrationField:
        """
        Genera un campo in stato quasi-stazionario.

        Args:
            n_warmup_steps: Passi di warmup per raggiungere lo steady state

        Returns:
            ConcentrationField stazionario
        """
        self.reset()
        self.step(n_warmup_steps)
        return self.get_field()


class DataManager:
    """
    Classe principale per la gestione dei dati.
    Sceglie automaticamente tra dati reali (NC) e sintetici.
    Supporta caricamento random da multiple NC files per training robusto.
    """

    # Configurazioni delle sorgenti dal report DICEA
    SOURCE_CONFIGS = {
        'S1': {'x': 620100, 'y': 4796210},
        'S2': {'x': 619800, 'y': 4795900},
        'S3': {'x': 620200, 'y': 4795800},
    }

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        use_synthetic: bool = True,
        domain_config: Optional[DomainConfig] = None,
        preload_all: bool = False
    ):
        """
        Args:
            data_dir: Directory con i file NC
            use_synthetic: Usa dati sintetici se NC non disponibile
            domain_config: Configurazione del dominio (usa default se None)
            preload_all: Se True, precarica tutti i file NC in memoria
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.use_synthetic = use_synthetic

        # Domain config di default basata sul report
        self.domain = domain_config or DomainConfig(
            xmin=619000, xmax=622000,
            ymin=4794500, ymax=4797000,
            resolution=10
        )

        # Loader NC
        self._nc_loader: Optional[NetCDFLoader] = None
        self._nc_files: List[Path] = []
        self._preloaded_fields: Dict[str, ConcentrationField] = {}

        if self.data_dir and self.data_dir.exists():
            self._nc_loader = NetCDFLoader(self.data_dir)
            self._nc_files = list(self.data_dir.glob("*.nc"))
            print(f"Found {len(self._nc_files)} NC files in {self.data_dir}")

            if preload_all and self._nc_files:
                self._preload_all_files()

    def _preload_all_files(self):
        """Precarica tutti i file NC in memoria."""
        print("Preloading all NC files...")
        for nc_file in self._nc_files:
            try:
                field = self._nc_loader.load(
                    str(nc_file),
                    concentration_var="Concentration - component 1"
                )
                self._preloaded_fields[nc_file.stem] = field
                print(f"  Loaded: {nc_file.name} (max conc: {field.max_concentration:.2f})")
            except Exception as e:
                print(f"  Failed to load {nc_file.name}: {e}")
        print(f"Preloaded {len(self._preloaded_fields)} files")

    def get_random_field(self) -> Tuple[ConcentrationField, str]:
        """
        Ritorna un campo di concentrazione random tra quelli disponibili.
        Utile per training con variabilità.

        Returns:
            Tuple di (ConcentrationField, source_id)
        """
        if self._preloaded_fields:
            # Usa file precaricati
            key = np.random.choice(list(self._preloaded_fields.keys()))
            field = self._preloaded_fields[key]
            source_id = 'S1' if 'S1' in key else ('S2' if 'S2' in key else 'S3')
            return field, source_id

        elif self._nc_files:
            # Carica random file
            nc_file = np.random.choice(self._nc_files)
            field = self._nc_loader.load(
                str(nc_file),
                concentration_var="Concentration - component 1"
            )
            source_id = 'S1' if 'S1' in nc_file.name else ('S2' if 'S2' in nc_file.name else 'S3')
            return field, source_id

        elif self.use_synthetic:
            # Genera sintetico con sorgente random
            source_id = np.random.choice(['S1', 'S2', 'S3'])
            field = self._generate_synthetic(source_id, None)
            return field, source_id

        raise ValueError("Nessun dato disponibile")

    def get_concentration_field(
        self,
        source_id: str = 'S1',
        run_id: Optional[str] = None,
        synthetic_params: Optional[Dict] = None
    ) -> ConcentrationField:
        """
        Ottiene un campo di concentrazione.

        Args:
            source_id: ID della sorgente ('S1', 'S2', 'S3')
            run_id: ID del run NC (es. 'CMEMS_S1_01')
            synthetic_params: Parametri per la generazione sintetica

        Returns:
            ConcentrationField pronto per l'uso
        """
        # Prova a caricare da NC
        if self._nc_loader and run_id:
            try:
                return self._nc_loader.load(run_id)
            except FileNotFoundError:
                if not self.use_synthetic:
                    raise
                warnings.warn(
                    f"File NC {run_id} non trovato, uso dati sintetici"
                )

        # Genera dati sintetici
        if self.use_synthetic:
            return self._generate_synthetic(source_id, synthetic_params)

        raise ValueError("Nessun dato disponibile e dati sintetici disabilitati")

    def _generate_synthetic(
        self,
        source_id: str,
        params: Optional[Dict] = None
    ) -> ConcentrationField:
        """Genera un campo sintetico."""
        if source_id not in self.SOURCE_CONFIGS:
            raise ValueError(f"Source ID {source_id} non valido")

        source_pos = (
            self.SOURCE_CONFIGS[source_id]['x'],
            self.SOURCE_CONFIGS[source_id]['y']
        )

        # Parametri di default o custom
        default_params = {
            'diffusion_coef': 5.0,
            'advection_velocity': (-0.1, -0.1),  # verso SW come nel report
            'source_strength': 1000.0,
            'dt': 10.0
        }
        if params:
            default_params.update(params)

        generator = SyntheticPlumeGenerator(
            domain=self.domain,
            source_position=source_pos,
            **default_params
        )

        return generator.generate_steady_state(n_warmup_steps=300)

    @property
    def available_nc_runs(self) -> List[str]:
        """Lista dei run NC disponibili."""
        if self._nc_loader:
            return self._nc_loader.available_runs
        return []


if __name__ == "__main__":
    # Test del modulo
    print("Testing HYDRAS Data Module...")

    # Test dominio
    domain = DomainConfig(
        xmin=619000, xmax=622000,
        ymin=4794500, ymax=4797000,
        resolution=10
    )
    print(f"Domain: {domain.nx}x{domain.ny} cells")

    # Test generatore sintetico
    generator = SyntheticPlumeGenerator(
        domain=domain,
        source_position=(620100, 4796210),  # S1
        diffusion_coef=5.0,
        advection_velocity=(-0.1, -0.1)
    )

    print("Generating steady-state field...")
    field = generator.generate_steady_state(300)

    print(f"Max concentration: {field.max_concentration:.2f} g/m³")
    print(f"Concentration at source: {field.get_concentration(620100, 4796210):.2f} g/m³")

    # Test gradiente
    grad = field.get_gradient(620150, 4796200)
    print(f"Gradient at (620150, 4796200): ({grad[0]:.4f}, {grad[1]:.4f})")

    # Test DataManager
    print("\nTesting DataManager...")
    dm = DataManager(use_synthetic=True)
    field2 = dm.get_concentration_field(source_id='S1')
    print(f"Field from DataManager: max = {field2.max_concentration:.2f}")

    print("\nAll tests passed!")