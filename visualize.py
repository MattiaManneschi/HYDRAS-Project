"""
HYDRAS Source Seeking - Visualization Tools
Strumenti per visualizzare le traiettorie, il campo di concentrazione
e le performance dell'agente.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import ConcentrationField, DataManager, DomainConfig


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


def plot_multiple_trajectories(
    trajectories: List[np.ndarray],
    field: ConcentrationField,
    title: str = "Multiple Agent Trajectories",
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualizza multiple traiettorie.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    plot_concentration_field(field, ax=ax, title=title)
    
    if colors is None:
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(trajectories))]
    
    for i, (traj, color) in enumerate(zip(trajectories, colors)):
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=color, linewidth=2, alpha=0.7,
            label=f'Episode {i+1}'
        )
        ax.scatter(traj[0, 0], traj[0, 1], c=[color], s=100, marker='o', edgecolors='black', linewidths=1)
        ax.scatter(traj[-1, 0], traj[-1, 1], c=[color], s=100, marker='s', edgecolors='black', linewidths=1)
    
    ax.legend(loc='upper right', ncol=2)
    
    return fig


def plot_concentration_profile(
    trajectory: np.ndarray,
    field: ConcentrationField,
    ax: Optional[plt.Axes] = None,
    title: str = "Concentration Along Trajectory"
) -> plt.Axes:
    """
    Plot del profilo di concentrazione lungo la traiettoria.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    concentrations = []
    distances = [0.0]
    
    for i, pos in enumerate(trajectory):
        c = field.get_concentration(pos[0], pos[1])
        concentrations.append(c)
        
        if i > 0:
            d = np.linalg.norm(trajectory[i] - trajectory[i-1])
            distances.append(distances[-1] + d)
    
    ax.fill_between(distances, concentrations, alpha=0.3, color='orange')
    ax.plot(distances, concentrations, 'b-', linewidth=2, label='Concentration')
    
    ax.axhline(
        y=field.max_concentration * 0.8,
        color='red', linestyle='--', alpha=0.7,
        label='80% Max (threshold)'
    )
    
    ax.set_xlabel('Distance traveled (m)', fontsize=12)
    ax.set_ylabel('Concentration (g/m³)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_distance_to_source(
    trajectory: np.ndarray,
    source_position: Tuple[float, float],
    ax: Optional[plt.Axes] = None,
    title: str = "Distance to Source Over Time"
) -> plt.Axes:
    """
    Plot della distanza dalla sorgente nel tempo.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    distances = np.linalg.norm(trajectory - np.array(source_position), axis=1)
    steps = np.arange(len(distances))
    
    ax.plot(steps, distances, 'b-', linewidth=2)
    ax.fill_between(steps, distances, alpha=0.2)
    ax.axhline(y=30, color='green', linestyle='--', label='Success threshold (30m)')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Distance to Source (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_training_summary(
    trajectory: np.ndarray,
    field: ConcentrationField,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Crea un summary completo di un episodio.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_trajectory(trajectory, field, ax=ax1, title="Trajectory on Concentration Field")
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_concentration_profile(trajectory, field, ax=ax2)
    
    ax3 = fig.add_subplot(gs[1, 0])
    plot_distance_to_source(trajectory, field.source_position, ax=ax3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    final_distance = np.linalg.norm(trajectory[-1] - np.array(field.source_position))
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    concentrations = [field.get_concentration(p[0], p[1]) for p in trajectory]
    max_conc = max(concentrations)
    final_conc = concentrations[-1]
    success = final_distance < 30
    initial_dist = np.linalg.norm(trajectory[0] - np.array(field.source_position))
    
    stats_text = f"""
    EPISODE STATISTICS
    {'='*30}
    
    Total steps: {len(trajectory)}
    Total distance traveled: {total_distance:.1f} m
    
    Initial distance to source: {initial_dist:.1f} m
    Final distance to source: {final_distance:.1f} m
    
    Max concentration reached: {max_conc:.1f} g/m³
    Final concentration: {final_conc:.1f} g/m³
    
    Source found: {'YES ✓' if success else 'NO ✗'}
    
    Efficiency: {(1 - final_distance / initial_dist) * 100:.1f}%
    """
    
    ax4.text(
        0.1, 0.9, stats_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )
    
    fig.suptitle('HYDRAS Source Seeking - Episode Summary', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def create_animation(
    trajectory: np.ndarray,
    field: ConcentrationField,
    fps: int = 10,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Crea un'animazione della traiettoria dell'agente.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plot_concentration_field(field, ax=ax, title="HYDRAS Source Seeking", colorbar=True)
    
    line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
    agent_marker, = ax.plot([], [], 'bo', markersize=15, markeredgecolor='black', markeredgewidth=2)
    
    info_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    def init():
        line.set_data([], [])
        agent_marker.set_data([], [])
        info_text.set_text('')
        return line, agent_marker, info_text
    
    def animate(frame):
        x_data = trajectory[:frame+1, 0]
        y_data = trajectory[:frame+1, 1]
        line.set_data(x_data, y_data)
        agent_marker.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        
        conc = field.get_concentration(trajectory[frame, 0], trajectory[frame, 1])
        dist = np.linalg.norm(trajectory[frame] - np.array(field.source_position))
        
        info_text.set_text(
            f'Step: {frame}\n'
            f'Concentration: {conc:.1f} g/m³\n'
            f'Distance to source: {dist:.1f} m'
        )
        
        return line, agent_marker, info_text
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(trajectory), interval=1000//fps,
        blit=True
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to: {save_path}")
    
    return anim


def visualize_gradient_field(
    field: ConcentrationField,
    resolution: int = 20,
    ax: Optional[plt.Axes] = None,
    title: str = "Concentration Gradient Field"
) -> plt.Axes:
    """
    Visualizza il campo gradiente della concentrazione.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    plot_concentration_field(field, ax=ax, title=title, source_marker=True)
    
    x_range = np.linspace(field.x_coords[0], field.x_coords[-1], resolution)
    y_range = np.linspace(field.y_coords[0], field.y_coords[-1], resolution)
    
    X, Y = np.meshgrid(x_range, y_range)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(resolution):
        for j in range(resolution):
            grad = field.get_gradient(X[i, j], Y[i, j])
            U[i, j] = grad[0]
            V[i, j] = grad[1]
    
    magnitude = np.sqrt(U**2 + V**2)
    magnitude[magnitude == 0] = 1
    U_norm = U / magnitude
    V_norm = V / magnitude
    
    ax.quiver(
        X, Y, U_norm, V_norm,
        magnitude,
        cmap='viridis',
        alpha=0.7,
        scale=25,
        width=0.003
    )
    
    return ax


if __name__ == "__main__":
    print("Testing visualization tools...")
    
    dm = DataManager(use_synthetic=True)
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
    
    fig3 = plot_training_summary(trajectory, field, save_path='/tmp/test_summary.png')
    print("  Saved: summary.png")
    
    plt.close('all')
    print("\nAll visualization tests completed!")
