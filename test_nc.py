#!/usr/bin/env python3
"""
HYDRAS Source Seeking - Test NetCDF Loading
Script per verificare il corretto caricamento dei file NC delle simulazioni MIKE21.
"""

import sys
from pathlib import Path

# Aggiungi path del progetto
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np


def test_netcdf_loading(nc_file: str):
    """
    Testa il caricamento di un file NetCDF.

    Args:
        nc_file: Path al file NC
    """
    print("=" * 60)
    print("HYDRAS - Test NetCDF Loading")
    print("=" * 60)

    # 1. Test con netCDF4 diretto
    print(f"\n1. Apertura diretta con netCDF4...")
    try:
        import netCDF4 as nc

        ds = nc.Dataset(nc_file, 'r')

        print(f"   File: {nc_file}")
        print(f"\n   Variabili:")
        for var in ds.variables:
            v = ds.variables[var]
            print(f"      {var}: shape={v.shape}, dtype={v.dtype}")

        print(f"\n   Dimensioni:")
        for dim in ds.dimensions:
            print(f"      {dim}: {len(ds.dimensions[dim])}")

        # Estrai alcuni valori
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        conc = ds.variables['Concentration - component 1']

        print(f"\n   Range X: {x.min():.0f} - {x.max():.0f} m")
        print(f"   Range Y: {y.min():.0f} - {y.max():.0f} m")
        print(f"   Concentration shape: {conc.shape}")
        print(f"   Concentration dtype: {conc.dtype}")

        # Gestisci masked array
        conc_data = conc[:]
        if hasattr(conc_data, 'mask'):
            n_masked = np.sum(conc_data.mask)
            n_total = conc_data.size
            print(f"   Masked values: {n_masked}/{n_total} ({100 * n_masked / n_total:.1f}%)")
            conc_filled = np.ma.filled(conc_data, 0.0)
        else:
            conc_filled = conc_data

        # Sostituisci NaN
        conc_filled = np.nan_to_num(conc_filled, nan=0.0)

        print(f"   Concentration min: {conc_filled.min():.2f}")
        print(f"   Concentration max: {conc_filled.max():.2f}")
        print(f"   Non-zero cells at t=0: {np.sum(conc_filled[0] > 0)}")
        print(f"   Non-zero cells at t=-1: {np.sum(conc_filled[-1] > 0)}")

        ds.close()
        print("\n   ✓ netCDF4 loading OK")

    except Exception as e:
        print(f"   ✗ Errore: {e}")
        return False

    # 2. Test con il DataManager del progetto
    print(f"\n2. Test con DataManager...")
    try:
        from utils.data_loader import NetCDFLoader, DataManager

        # Usa il loader diretto
        loader = NetCDFLoader(Path(nc_file).parent)
        print(f"   Available runs: {loader.available_runs}")

        # Carica il file
        field = loader.load(
            nc_file,
            concentration_var="Concentration - component 1"
        )

        print(f"\n   ConcentrationField loaded:")
        print(f"      Shape: {field.data.shape}")
        print(f"      X range: {field.x_coords[0]:.0f} - {field.x_coords[-1]:.0f}")
        print(f"      Y range: {field.y_coords[0]:.0f} - {field.y_coords[-1]:.0f}")
        print(f"      N timesteps: {field.n_timesteps}")
        print(f"      Max concentration: {field.max_concentration:.2f}")
        print(f"      Source position: {field.source_position}")

        # Test interpolazione
        if field.source_position:
            sx, sy = field.source_position
            c_at_source = field.get_concentration(sx, sy)
            print(f"\n   Concentration at source: {c_at_source:.2f}")

            # Test gradiente
            grad = field.get_gradient(sx + 50, sy + 50)
            print(f"   Gradient near source: ({grad[0]:.4f}, {grad[1]:.4f})")

        # Test time stepping
        print(f"\n   Testing time evolution...")
        field.set_time(0)
        c0 = field.max_concentration
        field.set_time(field.n_timesteps - 1)
        c_final = field.max_concentration
        print(f"      Max conc at t=0: {c0:.2f}")
        print(f"      Max conc at t=end: {c_final:.2f}")

        print("\n   ✓ DataManager loading OK")

    except Exception as e:
        import traceback
        print(f"   ✗ Errore: {e}")
        traceback.print_exc()
        return False

    # 3. Test con l'ambiente
    print(f"\n3. Test con SourceSeekingEnv...")
    try:
        from envs.source_seeking_env import SourceSeekingEnv

        env = SourceSeekingEnv(
            concentration_field=field,
            source_id="S1"
        )

        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset()
        print(f"\n   Reset OK:")
        print(f"      Observation shape: {obs.shape}")
        print(f"      Initial distance: {info['initial_distance']:.1f} m")
        print(f"      Initial concentration: {info['initial_concentration']:.2f}")

        # Test step
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"\n   Step OK:")
        print(f"      Reward: {reward:.4f}")
        print(f"      Distance to source: {info['distance_to_source']:.1f} m")

        env.close()
        print("\n   ✓ Environment test OK")

    except Exception as e:
        import traceback
        print(f"   ✗ Errore: {e}")
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("TUTTI I TEST PASSATI! ✓")
    print("=" * 60)
    print("\nOra puoi lanciare il training con:")
    print(f"  python train_ppo.py train --source S1 --nc-file {nc_file}")

    return True


def inspect_all_nc_files(data_dir: str = "data"):
    """
    Ispeziona tutti i file NC nella directory.
    """
    print("=" * 60)
    print("Ispezione di tutti i file NC")
    print("=" * 60)

    data_path = Path(data_dir)
    nc_files = list(data_path.glob("*.nc"))

    if not nc_files:
        print(f"Nessun file .nc trovato in {data_dir}/")
        return

    print(f"\nTrovati {len(nc_files)} file NC:\n")

    import netCDF4 as nc

    for nc_file in sorted(nc_files):
        try:
            ds = nc.Dataset(nc_file, 'r')
            conc = ds.variables['Concentration - component 1']
            max_conc = float(np.max(conc[:]))
            n_times = conc.shape[0]
            ds.close()

            # Estrai source ID dal nome
            name = nc_file.stem
            source = "S1" if "S1" in name else ("S2" if "S2" in name else "S3")

            print(f"  {nc_file.name}")
            print(f"      Source: {source}, Timesteps: {n_times}, Max conc: {max_conc:.1f}")

        except Exception as e:
            print(f"  {nc_file.name}: ERRORE - {e}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test NetCDF loading")
    parser.add_argument(
        'nc_file',
        nargs='?',
        default=None,
        help='Path to NC file to test'
    )
    parser.add_argument(
        '--inspect-all',
        action='store_true',
        help='Inspect all NC files in data/'
    )

    args = parser.parse_args()

    if args.inspect_all:
        inspect_all_nc_files()
    elif args.nc_file:
        test_netcdf_loading(args.nc_file)
    else:
        # Cerca un file NC di default
        default_files = list(Path("data").glob("*.nc"))
        if default_files:
            test_netcdf_loading(str(default_files[0]))
        else:
            print("Uso: python test_nc.py <path_to_nc_file>")
            print("     python test_nc.py --inspect-all")