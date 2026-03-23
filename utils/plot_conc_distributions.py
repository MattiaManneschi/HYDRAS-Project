#!/usr/bin/env python3
"""
Script per plottare le distribuzioni spaziali di concentrazione nel tempo
per ciascun file NetCDF di concentrazione.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from tqdm import tqdm


def plot_concentration_distributions():
    """Plotta le distribuzioni di concentrazione per ogni file nc."""
    
    # Percorsi
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "conc_distributions"
    output_dir.mkdir(exist_ok=True)
    
    # Trova tutti i file di concentrazione (esclude UV)
    conc_files = sorted(data_dir.glob("CMEMS_*_conc_grid_10m.nc"))
    
    print(f"Trovati {len(conc_files)} file di concentrazione")
    
    for conc_file in tqdm(conc_files, desc="Processing concentration files"):
        try:
            # Apri il file NetCDF
            ds = xr.open_dataset(conc_file)
            
            # Estrai il nome della variabile di concentrazione
            conc_var = None
            for var in ds.data_vars:
                if 'conc' in var.lower() or 'concentration' in var.lower():
                    conc_var = var
                    break
            
            if conc_var is None:
                # Prova a prendere la prima variabile se non trovi 'conc'
                conc_var = list(ds.data_vars)[0]
            
            conc_data = ds[conc_var]
            
            # Crea figura con 2 subplots
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f"Concentration Distribution: {conc_file.stem}", fontsize=16, fontweight='bold')
            
            # Plot 1: Concentrazione media spaziale nel tempo
            ax1 = axes[0]
            conc_spatial_mean = conc_data.mean(dim=['x', 'y']) if 'x' in conc_data.dims and 'y' in conc_data.dims else conc_data.mean()
            conc_spatial_mean.plot(ax=ax1, linewidth=2, color='steelblue')
            ax1.set_xlabel('Time step', fontsize=11)
            ax1.set_ylabel('Mean Concentration', fontsize=11)
            ax1.set_title('Spatial Mean Concentration over Time', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Heatmap della concentrazione media temporale nelle diverse locazioni
            ax2 = axes[1]
            if 'x' in conc_data.dims and 'y' in conc_data.dims:
                conc_temporal_mean = conc_data.mean(dim='time')
                im = ax2.imshow(conc_temporal_mean.values.T, origin='lower', cmap='viridis', aspect='auto')
                ax2.set_xlabel('X coordinate', fontsize=11)
                ax2.set_ylabel('Y coordinate', fontsize=11)
                ax2.set_title('Spatial Distribution (Mean over Time)', fontsize=12, fontweight='bold')
                cbar = plt.colorbar(im, ax=ax2, label='Concentration')
            else:
                ax2.text(0.5, 0.5, 'Unable to create spatial plot', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            # Salva il plot
            output_file = output_dir / f"{conc_file.stem}_distribution.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Salvato: {output_file}")
            
            ds.close()
            
        except Exception as e:
            print(f"✗ Errore in {conc_file.name}: {str(e)}")
    
    print(f"\n✓ Tutti i plot salvati in: {output_dir}")


if __name__ == "__main__":
    plot_concentration_distributions()
