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


class ChunkedConcentrationField:
    """
    Wrappa un ConcentrationField per rappresentare una finestra temporale (chunk).
    Permette data augmentation spezzando file lunghi in porzioni più piccole.
    
    Esempio: File NC con 2880 timestep diventa 2 "file virtuali":
        - Chunk 0: spawn_start_frame = n_timesteps // 4 (primo quarto, spawn al 25%)
        - Chunk 1: spawn_start_frame = n_timesteps * 3 // 4 (terzo quarto, spawn al 75%)
    """
    
    def __init__(
        self,
        base_field: ConcentrationField,
        chunk_id: int,
        n_chunks: int = 2,
        chunk_start_frame: Optional[int] = None,
        chunk_end_frame: Optional[int] = None
    ):
        """
        Args:
            base_field: ConcentrationField originale
            chunk_id: ID del chunk (0, 1, ...)
            n_chunks: Numero totale di chunk per questo campo (default 2)
            chunk_start_frame: Inizio del chunk (se None, calcolato automaticamente)
            chunk_end_frame: Fine del chunk (se None, calcolato automaticamente)
        """
        self.base_field = base_field
        self.chunk_id = chunk_id
        self.n_chunks = n_chunks
        self.source_position = base_field.source_position
        self.land_mask = base_field.land_mask
        
        # Calcola i frame del chunk
        total_frames = base_field.n_timesteps
        
        if chunk_start_frame is None or chunk_end_frame is None:
            # Divisione semplice in n_chunks parte uguali
            frames_per_chunk = total_frames // n_chunks
            if chunk_start_frame is None:
                self.chunk_start_frame = chunk_id * frames_per_chunk
            else:
                self.chunk_start_frame = chunk_start_frame
                
            if chunk_end_frame is None:
                if chunk_id == n_chunks - 1:
                    self.chunk_end_frame = total_frames  # Ultimo chunk prende tutto il resto
                else:
                    self.chunk_end_frame = (chunk_id + 1) * frames_per_chunk
            else:
                self.chunk_end_frame = chunk_end_frame
        else:
            self.chunk_start_frame = chunk_start_frame
            self.chunk_end_frame = chunk_end_frame
        
        # Spawn point del chunk: 1/4 della simulazione totale per chunk 0, 3/4 per chunk 1
        # (utilizzato da SourceSeekingConfig.spawn_start_frame)
        if n_chunks == 2:
            if chunk_id == 0:
                self.spawn_start_frame = total_frames // 4  # 25% del file intero
            else:  # chunk_id == 1
                self.spawn_start_frame = (total_frames * 3) // 4  # 75% del file intero
        else:
            # Fallback per altri numeri di chunk
            self.spawn_start_frame = self.chunk_start_frame + (self.chunk_end_frame - self.chunk_start_frame) // 4
        
        self.x_coords = base_field.x_coords
        self.y_coords = base_field.y_coords
        
    def set_time(self, time_idx: int):
        """Imposta il time index, limitato al range del chunk."""
        # Converte l'indice locale del chunk a indice globale del base_field
        clamped_idx = max(self.chunk_start_frame, min(time_idx, self.chunk_end_frame - 1))
        self.base_field.set_time(clamped_idx)
    
    def get_concentration(self, x: float, y: float) -> float:
        """Ritorna concentrazione al campo attuale (via base_field)."""
        return self.base_field.get_concentration(x, y)
    
    def is_land(self, x: float, y: float) -> bool:
        """Ritorna True se è terra (via base_field)."""
        return self.base_field.is_land(x, y)
    
    def get_land_distance(self, x: float, y: float) -> float:
        """Ritorna distanza dalla terra (via base_field)."""
        return self.base_field.get_land_distance(x, y)
    
    def get_current_field(self) -> np.ndarray:
        """Ritorna il campo di concentrazione corrente."""
        return self.base_field.get_current_field()
    
    @property
    def max_concentration(self) -> float:
        """Massima concentrazione nel chunk."""
        # Calcola il max considerando solo il range del chunk
        if self.base_field._is_time_varying and self.base_field.data.ndim == 3:
            chunk_data = self.base_field.data[self.chunk_start_frame:self.chunk_end_frame]
            return float(np.max(chunk_data))
        return self.base_field.max_concentration
    
    @property
    def n_timesteps(self) -> int:
        """Numero di timestep nel chunk."""
        return self.chunk_end_frame - self.chunk_start_frame
    
    @property
    def _is_time_varying(self) -> bool:
        """È una serie temporale."""
        return self.base_field._is_time_varying


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


class DataManager:
    """
    Classe principale per la gestione dei dati NC (MIKE21).
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
        data_dir: Union[str, Path],
        domain_config: Optional[DomainConfig] = None,
        preload_all: bool = False,
        source_id_filter: Optional[str] = None
    ):
        """
        Args:
            data_dir: Directory con i file NC (obbligatoria)
            domain_config: Configurazione del dominio (usa default se None)
            preload_all: Se True, precarica tutti i file NC in memoria
            source_id_filter: Filtra solo file di una sorgente ('S1', 'S2', 'S3'). None = tutti.
        """
        self.data_dir = Path(data_dir)
        self.source_id_filter = source_id_filter

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

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Directory dati NC non trovata: {self.data_dir}\n"
                f"Assicurati che la cartella 'data/' contenga i file .nc"
            )

        self._nc_loader = NetCDFLoader(self.data_dir)
        all_nc_files = list(self.data_dir.glob("*.nc"))

        if not all_nc_files:
            raise FileNotFoundError(
                f"Nessun file .nc trovato in {self.data_dir}\n"
                f"Scarica i file di simulazione MIKE21 nella cartella 'data/'"
            )

        # Applica filtro source_id se specificato
        if source_id_filter:
            self._nc_files = [f for f in all_nc_files if source_id_filter in f.name]
        else:
            self._nc_files = all_nc_files

        if not self._nc_files:
            raise FileNotFoundError(
                f"Nessun file .nc per filtro '{source_id_filter}' in {self.data_dir}\n"
                f"File disponibili: {[f.name for f in all_nc_files]}"
            )

        if preload_all:
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
            key = np.random.choice(list(self._preloaded_fields.keys()))
            field = self._preloaded_fields[key]
            source_id = 'S1' if 'S1' in key else ('S2' if 'S2' in key else 'S3')
            return field, source_id

        nc_file = np.random.choice(self._nc_files)
        field = self._nc_loader.load(
            str(nc_file),
            concentration_var="Concentration - component 1"
        )
        source_id = 'S1' if 'S1' in nc_file.name else ('S2' if 'S2' in nc_file.name else 'S3')
        return field, source_id

    def get_random_field_for_source(self, source_id: str) -> Tuple[ConcentrationField, str]:
        """
        Ritorna un campo di concentrazione random per una specifica sorgente.
        Usato dal curriculum learning per controllare quali sorgenti sono attive.

        Args:
            source_id: ID della sorgente ('S1', 'S2', 'S3')

        Returns:
            Tuple di (ConcentrationField, source_id)
        """
        if self._preloaded_fields:
            source_keys = [k for k in self._preloaded_fields.keys() if source_id in k]
            if source_keys:
                key = np.random.choice(source_keys)
                return self._preloaded_fields[key], source_id

        source_files = [f for f in self._nc_files if source_id in f.name]
        if not source_files:
            raise FileNotFoundError(
                f"Nessun file NC per sorgente {source_id}.\n"
                f"File disponibili: {[f.name for f in self._nc_files]}"
            )

        nc_file = np.random.choice(source_files)
        field = self._nc_loader.load(
            str(nc_file),
            concentration_var="Concentration - component 1"
        )
        return field, source_id

    def get_concentration_field(
        self,
        source_id: str = 'S1',
        run_id: Optional[str] = None,
    ) -> ConcentrationField:
        """
        Ottiene un campo di concentrazione.

        Args:
            source_id: ID della sorgente ('S1', 'S2', 'S3')
            run_id: ID del run NC (es. 'CMEMS_S1_01')

        Returns:
            ConcentrationField pronto per l'uso
        """
        if run_id:
            return self._nc_loader.load(run_id)

        # Prendi un file random per questa sorgente
        field, _ = self.get_random_field_for_source(source_id)
        return field



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