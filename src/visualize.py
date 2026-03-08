"""
HYDRAS Source Seeking - Visualization Tools
Strumenti per visualizzare le traiettorie, il campo di concentrazione
e le performance dell'agente.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import ConcentrationField, DataManager


def create_concentration_colormap():
    """Crea una colormap personalizzata per la concentrazione."""
    colors = [
        (1, 1, 1, 0),        # Trasparente per valori bassi
        (1, 1, 0.8, 0.3),    # Giallo pallido
        (1, 0.9, 0.4, 0.5),  # Giallo
        (1, 0.6, 0.2, 0.7),  # Arancione
        (0.9, 0.2, 0.1, 0.9), # Rosso
        (0.5, 0, 0.2, 1)     # Rosso scuro
    ]
    return LinearSegmentedColormap.from_list('concentration', colors)


def plot_concentration_field(
    field: ConcentrationField,
    ax: Optional[plt.Axes] = None,
    title: str = "Concentration Field",
    colorbar: bool = True,
    source_marker: bool = True
) -> plt.Axes:
    """
    Visualizza il campo di concentrazione.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    data = field.get_current_field()
    extent = [
        field.x_coords[0], field.x_coords[-1],
        field.y_coords[0], field.y_coords[-1]
    ]
    
    cmap = create_concentration_colormap()
    im = ax.imshow(
        data,
        extent=extent,
        origin='lower',
        cmap=cmap,
        aspect='equal',
        vmin=0,
        vmax=field.max_concentration
    )
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Concentration (g/m³)', fontsize=12)
    
    if source_marker and field.source_position is not None:
        ax.scatter(
            field.source_position[0],
            field.source_position[1],
            c='red', s=300, marker='*',
            edgecolors='black', linewidths=1,
            label='Source', zorder=10
        )
        circle = Circle(
            field.source_position, 30,
            fill=False, color='red',
            linestyle='--', linewidth=2
        )
        ax.add_patch(circle)
    
    ax.set_xlabel('X (m UTM)', fontsize=12)
    ax.set_ylabel('Y (m UTM)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    
    return ax


def plot_trajectory(
    trajectory: np.ndarray,
    field: ConcentrationField,
    ax: Optional[plt.Axes] = None,
    title: str = "Agent Trajectory",
    show_arrows: bool = True,
    arrow_freq: int = 10,
    line_color: str = 'blue',
    start_marker: bool = True,
    end_marker: bool = True
) -> plt.Axes:
    """
    Visualizza la traiettoria dell'agente sul campo di concentrazione.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    plot_concentration_field(field, ax=ax, title=title, colorbar=True)
    
    ax.plot(
        trajectory[:, 0], trajectory[:, 1],
        color=line_color, linewidth=2, alpha=0.8,
        label='Trajectory'
    )
    
    if show_arrows and len(trajectory) > arrow_freq:
        for i in range(0, len(trajectory) - 1, arrow_freq):
            dx = trajectory[i+1, 0] - trajectory[i, 0]
            dy = trajectory[i+1, 1] - trajectory[i, 1]
            
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                ax.annotate(
                    '', 
                    xy=(trajectory[i+1, 0], trajectory[i+1, 1]),
                    xytext=(trajectory[i, 0], trajectory[i, 1]),
                    arrowprops=dict(
                        arrowstyle='->', color=line_color,
                        lw=1.5, alpha=0.6
                    )
                )
    
    if start_marker:
        ax.scatter(
            trajectory[0, 0], trajectory[0, 1],
            c='green', s=200, marker='o',
            edgecolors='black', linewidths=2,
            label='Start', zorder=11
        )
    
    if end_marker:
        ax.scatter(
            trajectory[-1, 0], trajectory[-1, 1],
            c='blue', s=200, marker='s',
            edgecolors='black', linewidths=2,
            label='End', zorder=11
        )
    
    ax.legend(loc='upper right')
    
    return ax


if __name__ == "__main__":
    print("Testing visualization tools...")
    
    data_dir = Path(__file__).resolve().parent.parent / "data"
    dm = DataManager(data_dir=data_dir)
    field = dm.get_concentration_field(source_id='S1')
    
    print(f"Field max concentration: {field.max_concentration:.2f}")
    print(f"Source position: {field.source_position}")
    
    # Genera traiettoria di test
    n_steps = 200
    trajectory = []
    start = np.array([621500, 4795000])
    target = np.array(field.source_position)
    
    for i in range(n_steps):
        t = i / n_steps
        pos = start + (target - start) * t
        pos += np.random.randn(2) * 20 * (1 - t)
        trajectory.append(pos)
    
    trajectory = np.array(trajectory)
    
    print("\nGenerating test visualizations...")
    
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    plot_concentration_field(field, ax=ax1)
    plt.savefig('/tmp/test_concentration_field.png', dpi=100)
    print("  Saved: concentration_field.png")
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    plot_trajectory(trajectory, field, ax=ax2)
    plt.savefig('/tmp/test_trajectory.png', dpi=100)
    print("  Saved: trajectory.png")
    
    plt.close('all')
    print("\nAll visualization tests completed!")
