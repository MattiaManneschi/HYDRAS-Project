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
from scipy.ndimage import distance_transform_edt


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
        source_position: Optional[Tuple[float, float]] = None,
        land_mask: Optional[np.ndarray] = None
    ):
        """
        Args:
            data: Array di concentrazione [time, y, x] o [y, x]
            x_coords: Coordinate x della griglia
            y_coords: Coordinate y della griglia
            time_coords: Coordinate temporali (opzionale)
            source_position: Posizione della sorgente (x, y)
            land_mask: Maschera booleana [y, x] — True dove c'è terra
        """
        self.data = data
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.time_coords = time_coords
        self.source_position = source_position
        self.land_mask = land_mask  # None = nessuna terra (sintetico)
        
        self._is_time_varying = data.ndim == 3
        self._current_time_idx = 0
        self._land_interpolator = None
        self._land_dist_interpolator = None
        self._build_land_interpolator()
    
    def _build_land_interpolator(self):
        """Costruisce l'interpolatore per la maschera terra e la distance map (una sola volta)."""
        if self.land_mask is not None:
            self._land_interpolator = RegularGridInterpolator(
                (self.y_coords, self.x_coords),
                self.land_mask.astype(np.float32),
                method='nearest',
                bounds_error=False,
                fill_value=1.0  # fuori dominio = terra
            )
            
            # Precompute distance map to land using EDT
            # ~self.land_mask = True where there's water (not land)
            # distance_transform_edt computes distance to nearest False (land)
            resolution = self.x_coords[1] - self.x_coords[0] if len(self.x_coords) > 1 else 10.0
            pixel_dist = distance_transform_edt(~self.land_mask)
            land_distance_map = pixel_dist * resolution  # convert to meters
            
            self._land_dist_interpolator = RegularGridInterpolator(
                (self.y_coords, self.x_coords),
                land_distance_map.astype(np.float32),
                method='linear',
                bounds_error=False,
                fill_value=0.0  # fuori dominio = sulla terra
            )
    
    def set_time(self, time_idx: int):
        """Imposta il timestep corrente."""
        if not self._is_time_varying:
            return
        self._current_time_idx = int(np.clip(time_idx, 0, len(self.time_coords) - 1))
    
    def get_concentration(self, x: float, y: float) -> float:
        """Ottiene la concentrazione interpolata in un punto.
        
        Interpola direttamente dallo slice corrente senza ricostruire
        l'interpolatore ad ogni timestep.
        """
        if self._is_time_varying:
            field = self.data[self._current_time_idx]
        else:
            field = self.data
        
        # Trova gli indici della cella più vicina per interpolazione bilineare
        # Clamp alle coordinate valide
        xi = np.interp(x, self.x_coords, np.arange(len(self.x_coords)))
        yi = np.interp(y, self.y_coords, np.arange(len(self.y_coords)))
        
        # Indici interi e frazioni
        x0 = int(xi)
        y0 = int(yi)
        x1 = min(x0 + 1, len(self.x_coords) - 1)
        y1 = min(y0 + 1, len(self.y_coords) - 1)
        
        xf = xi - x0
        yf = yi - y0
        
        # Interpolazione bilineare
        c00 = field[y0, x0]
        c01 = field[y0, x1]
        c10 = field[y1, x0]
        c11 = field[y1, x1]
        
        val = (c00 * (1 - xf) * (1 - yf) +
               c01 * xf * (1 - yf) +
               c10 * (1 - xf) * yf +
               c11 * xf * yf)
        
        return float(val)

    def is_land(self, x: float, y: float) -> bool:
        """Verifica se la posizione (x, y) è sulla terra."""
        if self._land_interpolator is None:
            return False  # sintetico: nessuna terra
        return bool(self._land_interpolator((y, x)) > 0.5)
    
    def get_land_distance(self, x: float, y: float) -> float:
        """Ritorna la distanza in metri dalla terra più vicina.
        
        Usa la distance map precomputata (O(1) invece di O(160) chiamate).
        Returns 0 se sulla terra, valore alto se nessuna terra.
        """
        if self._land_dist_interpolator is None:
            return 100.0  # sintetico: nessuna terra -> ritorna max
        return float(self._land_dist_interpolator((y, x)))
    
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
                # È un masked array - converti a ndarray preservando NaN
                conc_data = np.where(conc_data.mask, np.nan, conc_data.data)

            # Salva la maschera terra PRIMA di nan_to_num
            # True dove c'è terra (NaN in QUALSIASI timestep)
            conc_float = conc_data.astype(np.float32)
            if conc_float.ndim == 3:
                # Unione di tutti i NaN su tutti i timesteps
                land_mask = np.any(np.isnan(conc_float), axis=0)  # [y, x]
            else:
                land_mask = np.isnan(conc_float)  # [y, x]

            # Ora sostituisci NaN con 0 per l'interpolazione
            conc_data = np.nan_to_num(conc_float, nan=0.0, posinf=0.0, neginf=0.0)

            # Assicurati che sia float32 per efficienza
            conc_data = conc_data.astype(np.float32)

        # Estrai info sulla sorgente dal nome file
        source_pos = self._extract_source_position(filename)

        return ConcentrationField(
            data=conc_data,
            x_coords=x_coords,
            y_coords=y_coords,
            time_coords=time_coords,
            source_position=source_pos,
            land_mask=land_mask
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
        for source_id, cfg in DataManager.SOURCE_CONFIGS.items():
            if source_id in filename:
                return (cfg['x'], cfg['y'])
        return None


class WindData:
    """
    Rappresenta dati di vento (velocità e direzione) nel tempo.
    Supporta query temporali e conversione in componenti u, v.
    """
    
    def __init__(
        self,
        time_coords: np.ndarray,
        speed: np.ndarray,
        direction: np.ndarray,
        dt: float = 60.0
    ):
        """
        Args:
            time_coords: Array di indici temporali
            speed: Array di velocità del vento [m/s]
            direction: Array di direzione del vento [gradi da Nord, in senso orario]
            dt: Intervallo temporale tra timestep in minuti (default 60 min per CI, 15 per ORB)
        """
        self.time_coords = time_coords
        self.speed = speed
        self.direction = direction
        self.dt = dt  # Intervallo temporale in minuti
        self._current_time_idx = 0
        
        # Interpola per affrontare diversi intervalli di tempo
        self._speed_interp = RegularGridInterpolator(
            (np.arange(len(time_coords)),),
            speed,
            method='linear',
            bounds_error=False,
            fill_value=speed[-1]
        )
        self._direction_interp = RegularGridInterpolator(
            (np.arange(len(time_coords)),),
            direction,
            method='linear',
            bounds_error=False,
            fill_value=direction[-1]
        )
    
    def set_time(self, time_idx: Union[int, float]):
        """Imposta il timestep corrente."""
        # Clamp tra 0 e len-1
        self._current_time_idx = max(0, min(time_idx, len(self.time_coords) - 1))
    
    def set_time_from_minutes(self, time_minutes: float):
        """
        Imposta il timestep basato su tempo reale in minuti.
        Utile per sincronizzazione con concentrazione che usa tempo reale.
        
        Args:
            time_minutes: Tempo reale in minuti dall'inizio
        """
        # Calcola l'indice temporale basato su dt
        time_idx = time_minutes / self.dt
        self.set_time(time_idx)
    
    def get_wind_components(self, time_idx: Optional[Union[int, float]] = None) -> Tuple[float, float]:
        """
        Ritorna le componenti u, v del vento al timestep specificato.
        
        Converti dalla convenzione meteorologica (direzione di provenienza):
        - u = -speed * sin(direction_rad)  (positivo verso Est)
        - v = -speed * cos(direction_rad)  (positivo verso Nord)
        
        Args:
            time_idx: Indice temporale (se None, usa l'ultimo set_time)
        
        Returns:
            Tuple (u, v) in m/s
        """
        if time_idx is not None:
            idx = max(0, min(time_idx, len(self.time_coords) - 1))
        else:
            idx = self._current_time_idx
        
        # Interpolazione lineare
        speed_val = float(self._speed_interp([[idx]])[0])
        direction_deg = float(self._direction_interp([[idx]])[0])
        
        # Converti da gradi (da Nord, senso orario) a radianti
        direction_rad = np.radians(direction_deg)
        
        # Converti secondo convenzione meteorologica
        u = -speed_val * np.sin(direction_rad)
        v = -speed_val * np.cos(direction_rad)
        
        return u, v
    
    def get_wind_speed_direction(self, time_idx: Optional[Union[int, float]] = None) -> Tuple[float, float]:
        """Ritorna velocità e direzione al timestep specificato."""
        if time_idx is not None:
            idx = max(0, min(time_idx, len(self.time_coords) - 1))
        else:
            idx = self._current_time_idx
        
        speed_val = float(self._speed_interp([[idx]])[0])
        direction_val = float(self._direction_interp([[idx]])[0])
        
        return speed_val, direction_val


class CurrentData:
    """
    Rappresenta i dati di corrente 2D (componenti u, v) nel tempo e spazio.
    """
    
    def __init__(
        self,
        data_u: np.ndarray,
        data_v: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        time_coords: Optional[np.ndarray] = None,
        dt: float = 5.0
    ):
        """
        Args:
            data_u: Array di velocità Est [time, y, x] o [y, x]
            data_v: Array di velocità Nord [time, y, x] o [y, x]
            x_coords: Coordinate x della griglia
            y_coords: Coordinate y della griglia  
            time_coords: Coordinate temporali (opzionale)
            dt: Intervallo temporale tra timestep in minuti (default 5 min)
        """
        self.data_u = data_u.astype(np.float32)
        self.data_v = data_v.astype(np.float32)
        self.x_coords = x_coords.astype(np.float32)
        self.y_coords = y_coords.astype(np.float32)
        self.time_coords = time_coords
        self.dt = dt  # Intervallo temporale in minuti
        
        self._is_time_varying = data_u.ndim == 3
        self._current_time_idx = 0.0  # Keep as float for interpolation
        
        # Costruisci interpolatori 3D (time, y, x) per supportare interpolazione temporale
        self._u_interpolator = None
        self._v_interpolator = None
        self._build_3d_interpolators()
    
    def _build_3d_interpolators(self):
        """Costruisce interpolatori 3D con supporto per interpolazione temporale."""
        if self._is_time_varying:
            n_times = self.data_u.shape[0]
            # Sostituisci NaN con 0
            data_u_clean = np.nan_to_num(self.data_u, nan=0.0)
            data_v_clean = np.nan_to_num(self.data_v, nan=0.0)
            
            # Crea interpolatore 3D: (time, y, x)
            self._u_interpolator = RegularGridInterpolator(
                (np.arange(n_times, dtype=np.float32), self.y_coords, self.x_coords),
                data_u_clean,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            self._v_interpolator = RegularGridInterpolator(
                (np.arange(n_times, dtype=np.float32), self.y_coords, self.x_coords),
                data_v_clean,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
        else:
            # Static field - crea interpolatore 2D
            data_u_clean = np.nan_to_num(self.data_u, nan=0.0)
            data_v_clean = np.nan_to_num(self.data_v, nan=0.0)
            
            self._u_interpolator = RegularGridInterpolator(
                (self.y_coords, self.x_coords),
                data_u_clean,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            self._v_interpolator = RegularGridInterpolator(
                (self.y_coords, self.x_coords),
                data_v_clean,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
    
    def set_time(self, time_idx: Union[int, float]):
        """Imposta il timestep corrente (supporta float per interpolazione)."""
        if self._is_time_varying:
            n_times = self.data_u.shape[0]
            self._current_time_idx = float(max(0, min(time_idx, n_times - 1)))
        else:
            self._current_time_idx = 0.0
    
    def set_time_from_minutes(self, time_minutes: float):
        """
        Imposta il timestep basato su tempo reale in minuti.
        Utile per sincronizzazione con concentrazione che usa tempo reale.
        
        Args:
            time_minutes: Tempo reale in minuti dall'inizio
        """
        # Calcola l'indice temporale basato su dt (PRESERVA FLOAT!)
        time_idx = time_minutes / self.dt
        self.set_time(time_idx)
    
    def get_current_components(self, x: float, y: float, time_idx: Optional[Union[int, float]] = None) -> Tuple[float, float]:
        """
        Ritorna le componenti u, v della corrente nel punto (x, y).
        Supporta interpolazione temporale lineare quando time_idx è float.
        
        Args:
            x, y: Coordinate spaziali
            time_idx: Indice temporale (se None, usa l'ultimo set_time)
        
        Returns:
            Tuple (u, v) in m/s
        """
        if time_idx is not None:
            idx = float(time_idx)
        else:
            idx = self._current_time_idx
        
        if self._is_time_varying:
            # Clamp al range valido
            n_times = self.data_u.shape[0]
            idx = max(0.0, min(idx, float(n_times - 1)))
            # Interpola su (time, y, x)
            u = float(self._u_interpolator([[idx, y, x]])[0])
            v = float(self._v_interpolator([[idx, y, x]])[0])
        else:
            # Static field - interpola solo spazialmente
            u = float(self._u_interpolator([[y, x]])[0])
            v = float(self._v_interpolator([[y, x]])[0])
        
        return u, v
    
    @property
    def n_timesteps(self) -> int:
        return self.data_u.shape[0] if self._is_time_varying else 1


class WindDataLoader:
    """Carica dati di vento dai file di testo."""
    
    @staticmethod
    def load_from_txt(filepath: Path, speed_col: int = 1, direction_col: int = 2, dt: float = 60.0) -> WindData:
        """
        Carica dati di vento da file di testo.
        
        Supporta due formati:
        1. CI_WIND_TEST01.txt: Time, Speed, Direction (indici 1, 2), dt=60 min
        2. 2025_Wind_ORB.txt: Time, Direction, Speed (indici 2, 1), dt=15 min
        
        Entrambi hanno: Riga 1=nome, Riga 2=intestazioni, Riga 3=metadata, Riga 4+=dati
        
        Args:
            filepath: Path al file di testo
            speed_col: Indice colonna velocità (default 1 per formato CI)
            direction_col: Indice colonna direzione (default 2 per formato CI)
            dt: Intervallo temporale in minuti (60 per CI, 15 per ORB)
        
        Returns:
            WindData object
        """
        data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip prima 3 righe (nome + intestazioni + metadata)
        for line in lines[3:]:  # Skip righe 0, 1, 2 (index 0-2)
            parts = line.strip().split('\t')  # Separa per TAB
            if len(parts) < 3:
                continue
            
            try:
                # Formato flessibile: speed_col e direction_col indicano le colonne
                speed = float(parts[speed_col])
                direction = float(parts[direction_col])
                data.append((speed, direction))
            except (ValueError, IndexError):
                continue
        
        if not data:
            raise ValueError(f"Nessun dato di vento trovato in {filepath}")
        
        speeds = np.array([d[0] for d in data], dtype=np.float32)
        directions = np.array([d[1] for d in data], dtype=np.float32)
        time_coords = np.arange(len(speeds))  # Indici temporali
        
        return WindData(time_coords, speeds, directions, dt=dt)


class CurrentDataLoader:
    """Carica dati di corrente dai file netCDF."""
    
    @staticmethod
    def load_from_nc(filepath: Path, u_var: str = "u", v_var: str = "v", dt: float = 5.0) -> CurrentData:
        """
        Carica dati di corrente da file netCDF.
        
        Args:
            filepath: Path al file netCDF
            u_var: Nome della variabile velocità Est
            v_var: Nome della variabile velocità Nord
            dt: Intervallo temporale in minuti (default 5 min per CMEMS)
        
        Returns:
            CurrentData object con dt specificato
        """
        try:
            import netCDF4 as nc
        except ImportError:
            raise ImportError("netCDF4 non installato. Esegui: pip install netCDF4")
        
        with nc.Dataset(filepath, 'r') as ds:
            # Leggi le coordinate
            x_coords = ds.variables['x'][:]
            y_coords = ds.variables['y'][:]
            
            # Leggi le componenti di velocità
            data_u = ds.variables[u_var][:]
            data_v = ds.variables[v_var][:]
            
            # Leggi il tempo se presente
            time_coords = None
            if 'time' in ds.variables:
                time_coords = ds.variables['time'][:]
            
            # Gestisci masked arrays
            if hasattr(data_u, 'mask'):
                data_u = np.ma.filled(data_u, 0.0)
            if hasattr(data_v, 'mask'):
                data_v = np.ma.filled(data_v, 0.0)
        
        return CurrentData(data_u, data_v, x_coords, y_coords, time_coords, dt=dt)


class DataManager:
    """
    Classe principale per la gestione dei dati NC (MIKE21).
    Supporta caricamento random da multiple NC files per training robusto.
    Scopre automaticamente le sorgenti dai file disponibili.
    """

    # Configurazioni delle sorgenti dal report DICEA (legacy)
    SOURCE_CONFIGS = {
        'S1': {'x': 620100, 'y': 4796210},
        'S2': {'x': 619800, 'y': 4795900},
        'S3': {'x': 620200, 'y': 4795800},
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        domain_config: Optional[DomainConfig] = None,
        preload_all: bool = False,
        source_id_filter: Optional[str] = None,
        wind_filename: str = "CI_WIND_faseII_V1.txt",
        current_filename: str = "CL02_V1_SRC000_U_V_10mGrid.nc",
        discover_sources: bool = True
    ):
        """
        Args:
            data_dir: Directory con i file NC (obbligatoria)
            domain_config: Configurazione del dominio (usa default se None)
            preload_all: Se True, precarica tutti i file NC in memoria
            source_id_filter: Filtra solo file di una sorgente ('SRC000', 'SRC001', ecc). None = tutti.
            wind_filename: Nome file vento nella cartella data/Vento_V0-V3/ (default: CI_WIND_faseII_V1.txt per 132 sorgenti)
            current_filename: Nome file corrente nella cartella data/ (default: CL02_V1_SRC000_U_V_10mGrid.nc - unico per tutte le 132 sorgenti)
            discover_sources: Se True, scopre automaticamente le sorgenti dai file disponibili (default: True)
        """
        self.data_dir = Path(data_dir)
        self.source_id_filter = source_id_filter
        self.wind_filename = wind_filename
        self.current_filename = current_filename
        self.discover_sources_enabled = discover_sources

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
        
        # Carica dati di vento e corrente una sola volta
        self._wind_data: Optional[WindData] = None
        self._current_data: Optional[CurrentData] = None
        self._discovered_sources: List[str] = []  # Lista di source_id scoperti

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Directory dati NC non trovata: {self.data_dir}\n"
                f"Assicurati che la cartella 'data/' contenga i file .nc"
            )

        self._nc_loader = NetCDFLoader(self.data_dir)
        all_nc_files = list(self.data_dir.glob("*Conc_10mGrid.nc"))

        if not all_nc_files:
            raise FileNotFoundError(
                f"Nessun file *Conc_10mGrid.nc trovato in {self.data_dir}\n"
                f"Scarica i file di simulazione MIKE21 nella cartella 'data/'"
            )

        # Scopri le sorgenti dai file disponibili
        self._discover_sources(all_nc_files)

        # Applica filtro source_id se specificato
        if source_id_filter:
            self._nc_files = [f for f in all_nc_files if source_id_filter in f.name]
        else:
            self._nc_files = all_nc_files

        if not self._nc_files:
            raise FileNotFoundError(
                f"Nessun file *Conc_10mGrid.nc per filtro '{source_id_filter}' in {self.data_dir}\n"
                f"File disponibili: {[f.name for f in all_nc_files]}"
            )

        # Carica dati di vento e corrente
        self._load_wind_data()
        self._load_current_data()

        if preload_all:
            self._preload_all_files()

    def _discover_sources(self, nc_files: List[Path]):
        """
        Scopre automaticamente le sorgenti dai file disponibili.
        Estrae il pattern SRC### dai nomi file.
        
        Args:
            nc_files: Lista di file .nc da scansionare
        """
        sources = set()
        for f in nc_files:
            # Estrai SRC### dal nome file (es. 'CL02_V1_SRC000_Conc_10mGrid.nc' -> 'SRC000')
            parts = f.stem.split('_')
            for part in parts:
                if part.startswith('SRC') and part[3:].isdigit():
                    sources.add(part)
                    break
        
        self._discovered_sources = sorted(sources)
        print(f"Discovered {len(self._discovered_sources)} sources: {self._discovered_sources[:5]}... (e altri)")

    def _load_wind_data(self):
        """Carica il file di vento una sola volta."""
        # Prova prima in data_dir/Vento_V0-V3/
        wind_path = self.data_dir / "Vento_V0-V3" / self.wind_filename
        
        # Se non trovato, prova nella cartella padre (data/)
        if not wind_path.exists():
            # Risali un livello dalla data_dir
            parent_dir = self.data_dir.parent
            wind_path = parent_dir / "Vento_V0-V3" / self.wind_filename
        
        # Se ancora non trovato, prova direttamente in data_dir
        if not wind_path.exists():
            wind_path = self.data_dir / self.wind_filename
        
        if not wind_path.exists():
            print(f"WARNING: File di vento non trovato in:")
            print(f"  - {self.data_dir / 'Vento_V0-V3' / self.wind_filename}")
            print(f"  - {self.data_dir.parent / 'Vento_V0-V3' / self.wind_filename}")
            print(f"  - {self.data_dir / self.wind_filename}")
            return
        
        try:
            self._wind_data = self.load_wind_data(self.wind_filename)
            print(f"Wind data loaded: {self.wind_filename} ({len(self._wind_data.speed)} timesteps)")
        except Exception as e:
            print(f"ERROR loading wind data: {e}")

    def _load_current_data(self):
        """Carica il file di corrente una sola volta."""
        current_path = self.data_dir / self.current_filename
        if not current_path.exists():
            print(f"WARNING: File di corrente non trovato: {current_path}")
            return
        
        try:
            self._current_data = self.load_current_data_internal(self.current_filename)
            print(f"Current data loaded: {self.current_filename} ({self._current_data.n_timesteps} timesteps)")
        except Exception as e:
            print(f"ERROR loading current data: {e}")

    def _preload_all_files(self):
        """Precarica tutti i file NC in memoria."""
        print("Preloading all NC files...")
        for nc_file in self._nc_files:
            try:
                field = self._nc_loader.load(
                    str(nc_file),
                    concentration_var="Concentration - component 1"
                )
                source_id = self._extract_source_id(nc_file.stem)
                self._preloaded_fields[nc_file.stem] = field
                if len(self._preloaded_fields) % 10 == 0:
                    print(f"  Loaded: {len(self._preloaded_fields)} files (max conc: {field.max_concentration:.2f})")
            except Exception as e:
                print(f"  Failed to load {nc_file.name}: {e}")
        print(f"Preloaded {len(self._preloaded_fields)} files ({len(self._discovered_sources)} sources)")

    def get_random_field(self) -> Tuple[ConcentrationField, str]:
        """
        Ritorna un campo di concentrazione random tra quelli disponibili.
        Utile per training con variabilità.

        Returns:
            Tuple di (ConcentrationField, source_id) dove source_id è es. 'SRC042'
        """
        if self._preloaded_fields:
            key = np.random.choice(list(self._preloaded_fields.keys()))
            field = self._preloaded_fields[key]
            # Estrai source_id dal key
            source_id = self._extract_source_id(key)
            return field, source_id

        # Filtra SOLO file di concentrazione
        conc_files = [f for f in self._nc_files if 'Conc' in f.name]
        if not conc_files:
            raise FileNotFoundError(
                f"Nessun file di concentrazione trovato in {self.data_dir}\n"
                f"File disponibili: {[f.name for f in self._nc_files]}"
            )
        
        nc_file = np.random.choice(conc_files)
        field = self._nc_loader.load(
            str(nc_file),
            concentration_var="Concentration - component 1"
        )
        source_id = self._extract_source_id(nc_file.stem)
        return field, source_id

    def get_random_field_for_source(self, source_id: str) -> Tuple[ConcentrationField, str]:
        """
        Ritorna un campo di concentrazione random per una specifica sorgente.
        Usato dal curriculum learning per controllare quali sorgenti sono attive.

        Args:
            source_id: ID della sorgente (es. 'SRC000', 'SRC001', ..., 'SRC131')

        Returns:
            Tuple di (ConcentrationField, source_id)
        """
        if self._preloaded_fields:
            source_keys = [k for k in self._preloaded_fields.keys() if source_id in k]
            if source_keys:
                key = np.random.choice(source_keys)
                return self._preloaded_fields[key], source_id

        # Filtra per sorgente E per concentrazione
        source_files = [f for f in self._nc_files if source_id in f.name and 'Conc' in f.name]
        if not source_files:
            raise FileNotFoundError(
                f"Nessun file concentrazione NC per sorgente {source_id}.\n"
                f"File disponibili: {[f.name for f in self._nc_files]}"
            )

        nc_file = np.random.choice(source_files)
        field = self._nc_loader.load(
            str(nc_file),
            concentration_var="Concentration - component 1"
        )
        
        return field, source_id

    def _extract_source_id(self, filename_or_stem: str) -> str:
        """
        Estrae il source_id dal nome file o stem.
        Es. 'CL02_V1_SRC000_Conc_10mGrid' -> 'SRC000'
        
        Args:
            filename_or_stem: Nome file o stem
        
        Returns:
            Source ID (es. 'SRC000')
        """
        parts = filename_or_stem.split('_')
        for part in parts:
            if part.startswith('SRC') and part[3:].isdigit():
                return part
        return 'UNKNOWN'
    
    def get_concentration_field(
        self,
        source_id: str = 'SRC000',
        run_id: Optional[str] = None,
    ) -> ConcentrationField:
        """
        Ottiene un campo di concentrazione.

        Args:
            source_id: ID della sorgente (es. 'SRC000', 'SRC001')
            run_id: ID del run NC (non usato nel nuovo schema, per compatibilità)

        Returns:
            ConcentrationField pronto per l'uso
        """
        if run_id:
            # Legacy: se specificato run_id, prova a usarlo
            return self._nc_loader.load(run_id)

        # Prendi un file random per questa sorgente
        field, _ = self.get_random_field_for_source(source_id)
        return field
    
    def load_wind_data(self, wind_filename: str) -> WindData:
        """
        Carica i dati di vento da file di testo.
        Auto-detect il formato in base al nome file:
        - CI_WIND: Time, Speed, Direction (dt=60 min)
        - ORB: Time, Direction, Speed (dt=15 min)
        
        Args:
            wind_filename: Nome del file di vento (es. 'CI_WIND_faseII_V1.txt')
        
        Returns:
            WindData object
        """
        # Prova più percorsi possibili
        possible_paths = [
            self.data_dir / "Vento_V0-V3" / wind_filename,
            self.data_dir.parent / "Vento_V0-V3" / wind_filename,
            self.data_dir / wind_filename,
        ]
        
        filepath = None
        for path in possible_paths:
            if path.exists():
                filepath = path
                break
        
        if filepath is None:
            raise FileNotFoundError(f"File di vento non trovato in alcuno di questi percorsi: {possible_paths}")
        
        # Auto-detect formato e dt
        if 'ORB' in wind_filename.upper():
            # Formato ORB: Time, Direction, Speed, dt=15 min
            return WindDataLoader.load_from_txt(filepath, speed_col=2, direction_col=1, dt=15.0)
        else:
            # Formato CI: Time, Speed, Direction, dt=60 min
            return WindDataLoader.load_from_txt(filepath, speed_col=1, direction_col=2, dt=60.0)
    
    def load_current_data_internal(self, current_filename: str) -> CurrentData:
        """
        Carica i dati di corrente da file netCDF.
        Supporta nomi variabili: 'u'/'v' (CMEMS legacy) oppure 'u_velocity'/'v_velocity' (nuovo formato)
        
        Args:
            current_filename: Nome del file di corrente (es. 'CL02_V1_SRC000_U_V_10mGrid.nc')
        
        Returns:
            CurrentData object
        """
        filepath = self.data_dir / current_filename
        if not filepath.exists():
            raise FileNotFoundError(f"File di corrente non trovato: {filepath}")
        
        # Auto-detect variabili
        try:
            import netCDF4 as nc
            with nc.Dataset(filepath, 'r') as ds:
                if 'u_velocity' in ds.variables and 'v_velocity' in ds.variables:
                    # Nuovo formato (CL02_V1_*)
                    u_var, v_var = 'u_velocity', 'v_velocity'
                else:
                    # Legacy formato (CMEMS)
                    u_var, v_var = 'u', 'v'
        except Exception:
            # Fallback
            u_var, v_var = 'u', 'v'
        
        return CurrentDataLoader.load_from_nc(filepath, u_var=u_var, v_var=v_var)
    
    def get_wind_data(self) -> Optional[WindData]:
        """
        Ritorna i dati di vento precaricat (shared per tutte le sorgenti).
        
        Returns:
            WindData object o None se non caricato
        """
        return self._wind_data
    
    def get_current_data(self) -> Optional[CurrentData]:
        """
        Ritorna i dati di corrente precaricat (shared per tutte le sorgenti).
        
        Returns:
            CurrentData object o None se non caricato
        """
        return self._current_data
    
    def get_discovered_sources(self) -> List[str]:
        """
        Ritorna la lista di source_id scoperti dai file disponibili.
        
        Returns:
            Lista di source_id (es. ['SRC000', 'SRC001', ..., 'SRC131'])
        """
        return self._discovered_sources
    
    def n_sources(self) -> int:
        """Ritorna il numero totale di sorgenti scoperte."""
        return len(self._discovered_sources)


if __name__ == "__main__":
    # Test del modulo (richiede file NC nella cartella data/)
    print("Testing HYDRAS Data Module...")

    # Test dominio
    domain = DomainConfig(
        xmin=619000, xmax=622000,
        ymin=4794500, ymax=4797000,
        resolution=10
    )
    print(f"Domain: {domain.nx}x{domain.ny} cells")

    # Test DataManager
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if data_dir.exists() and list(data_dir.glob("*.nc")):
        print(f"\nTesting DataManager with NC files from {data_dir}...")
        dm = DataManager(data_dir=data_dir)
        field = dm.get_concentration_field(source_id='S1')
        print(f"Field max concentration: {field.max_concentration:.2f} g/m³")
        print(f"Source position: {field.source_position}")
        print(f"Concentration at source: {field.get_concentration(620100, 4796210):.2f} g/m³")
        print(f"Is land at source: {field.is_land(620100, 4796210)}")
        print(f"Timesteps: {field.n_timesteps}")
    else:
        print(f"\n[SKIP] Nessun file NC in {data_dir}")
        print("Scarica i file MIKE21 .nc nella cartella 'data/' per eseguire i test")

    print("\nDone!")