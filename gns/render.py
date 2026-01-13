"""Rendering utilities for GNS rollout visualizations.

This module provides functions for rendering trajectory predictions
as GIF animations or VTK files for visualization.
"""

import os
import pickle
from typing import Optional, Tuple

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import pointsToVTK


# ============================================================================
# Color Mapping
# ============================================================================

TYPE_TO_COLOR = {
    1: "red",       # Droplet
    3: "black",     # Boundary particles
    0: "green",     # Rigid solids
    7: "magenta",   # Goop
    6: "gold",      # Sand
    5: "blue",      # Water
}


def get_color_mask(particle_type: np.ndarray) -> list:
    """Get color mask for particles based on their types.

    Args:
        particle_type: Array of particle type IDs.

    Returns:
        List of tuples (mask, color) where mask is a boolean array
        and color is a string color name.
    """
    color_mask = []
    for material_id, color in TYPE_TO_COLOR.items():
        mask = np.array(particle_type) == material_id
        if mask.any():
            color_mask.append((mask, color))
    return color_mask


# ============================================================================
# Data Loading
# ============================================================================

def load_rollout_pickle(rollout_dir: str, rollout_name: str) -> dict:
    """Load rollout data from pickle file.

    Args:
        rollout_dir: Directory containing rollout pickle files.
        rollout_name: Name of the rollout file (without .pkl extension).

    Returns:
        Dictionary containing rollout data including:
            - initial_positions
            - ground_truth_rollout
            - predicted_rollout
            - metadata
            - particle_types
            - loss

    Raises:
        FileNotFoundError: If the pickle file doesn't exist.
    """
    filepath = f"{rollout_dir}{rollout_name}.pkl"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Rollout file not found: {filepath}")

    with open(filepath, "rb") as file:
        rollout_data = pickle.load(file)

    return rollout_data


def prepare_trajectory_data(rollout_data: dict) -> dict:
    """Prepare trajectory data for rendering.

    Args:
        rollout_data: Dictionary from load_rollout_pickle().

    Returns:
        Dictionary containing:
            - ground_truth_rollout: Full trajectory including initial positions
            - predicted_rollout: Full trajectory including initial positions
            - metadata: Simulation metadata
            - particle_types: Particle type IDs
            - loss: Prediction loss
            - dims: Number of spatial dimensions
            - num_particles: Number of particles
            - num_steps: Number of timesteps
            - boundaries: Spatial boundaries
    """
    # Concatenate initial positions with rollouts
    ground_truth = np.concatenate(
        [rollout_data["initial_positions"], rollout_data["ground_truth_rollout"]],
        axis=0
    )
    predicted = np.concatenate(
        [rollout_data["initial_positions"], rollout_data["predicted_rollout"]],
        axis=0
    )

    trajectory_data = {
        'ground_truth_rollout': ground_truth,
        'predicted_rollout': predicted,
        'metadata': rollout_data['metadata'],
        'particle_types': rollout_data['particle_types'],
        'loss': rollout_data['loss'].item() if hasattr(rollout_data['loss'], 'item') else rollout_data['loss'],
        'dims': ground_truth.shape[2],
        'num_particles': ground_truth.shape[1],
        'num_steps': ground_truth.shape[0],
        'boundaries': rollout_data['metadata']['bounds']
    }

    return trajectory_data


# ============================================================================
# 2D Rendering
# ============================================================================

def render_2d_trajectory(
        ax,
        trajectory: np.ndarray,
        timestep: int,
        color_mask: list,
        boundaries: list,
        point_size: float = 1) -> None:
    """Render a 2D trajectory frame.

    Args:
        ax: Matplotlib axes object.
        trajectory: Trajectory data (timesteps, particles, dims).
        timestep: Current timestep to render.
        color_mask: List of (mask, color) tuples from get_color_mask().
        boundaries: Spatial boundaries [[xmin, xmax], [ymin, ymax]].
        point_size: Size of particles in visualization.
    """
    ax.set_aspect("equal")
    ax.set_xlim([float(boundaries[0][0]), float(boundaries[0][1])])
    ax.set_ylim([float(boundaries[1][0]), float(boundaries[1][1])])

    for mask, color in color_mask:
        ax.scatter(
            trajectory[timestep][mask, 0],
            trajectory[timestep][mask, 1],
            s=point_size,
            color=color
        )

    ax.grid(True, which='both')


# ============================================================================
# 3D Rendering
# ============================================================================

def render_3d_trajectory(
        ax,
        trajectory: np.ndarray,
        timestep: int,
        color_mask: list,
        boundaries: list,
        vertical_camera_angle: float = 20,
        viewpoint_rotation: float = 0.5,
        change_yz: bool = False,
        point_size: float = 1) -> None:
    """Render a 3D trajectory frame.

    Args:
        ax: Matplotlib 3D axes object.
        trajectory: Trajectory data (timesteps, particles, dims).
        timestep: Current timestep to render.
        color_mask: List of (mask, color) tuples from get_color_mask().
        boundaries: Spatial boundaries [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        vertical_camera_angle: Vertical viewing angle in degrees.
        viewpoint_rotation: Rotation angle increment per timestep.
        change_yz: Whether to swap Y and Z axes.
        point_size: Size of particles in visualization.
    """
    xboundary = boundaries[0]
    yboundary = boundaries[1]
    zboundary = boundaries[2]

    if not change_yz:
        ax.set_xlim([float(xboundary[0]), float(xboundary[1])])
        ax.set_ylim([float(yboundary[0]), float(yboundary[1])])
        ax.set_zlim([float(zboundary[0]), float(zboundary[1])])

        for mask, color in color_mask:
            ax.scatter(
                trajectory[timestep][mask, 0],
                trajectory[timestep][mask, 1],
                trajectory[timestep][mask, 2],
                s=point_size,
                color=color
            )

        ax.set_box_aspect(
            aspect=(
                float(xboundary[1]) - float(xboundary[0]),
                float(yboundary[1]) - float(yboundary[0]),
                float(zboundary[1]) - float(zboundary[0])
            )
        )
    else:
        # Swap Y and Z axes
        ax.set_xlim([float(xboundary[0]), float(xboundary[1])])
        ax.set_ylim([float(zboundary[0]), float(zboundary[1])])
        ax.set_zlim([float(yboundary[0]), float(yboundary[1])])

        for mask, color in color_mask:
            ax.scatter(
                trajectory[timestep][mask, 0],
                trajectory[timestep][mask, 2],
                trajectory[timestep][mask, 1],
                s=point_size,
                color=color
            )

        ax.set_box_aspect(
            aspect=(
                float(xboundary[1]) - float(xboundary[0]),
                float(zboundary[1]) - float(zboundary[0]),
                float(yboundary[1]) - float(yboundary[0])
            )
        )

    ax.view_init(elev=vertical_camera_angle, azim=timestep * viewpoint_rotation)
    ax.grid(True, which='both')


# ============================================================================
# GIF Animation
# ============================================================================

def render_gif_animation(
        rollout_dir: str,
        rollout_name: str,
        point_size: float = 1,
        timestep_stride: int = 3,
        vertical_camera_angle: float = 20,
        viewpoint_rotation: float = 0.5,
        change_yz: bool = False,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None) -> None:
    """Render a GIF animation from rollout data.

    Args:
        rollout_dir: Directory containing rollout pickle files.
        rollout_name: Name of the rollout file (without .pkl extension).
        point_size: Size of particles in visualization.
        timestep_stride: Number of timesteps to skip between frames.
        vertical_camera_angle: Vertical viewing angle in degrees (3D only).
        viewpoint_rotation: Rotation angle increment per timestep (3D only).
        change_yz: Whether to swap Y and Z axes (3D only).
        output_dir: Output directory (defaults to rollout_dir).
        output_name: Output filename (defaults to rollout_name).
    """
    # Load and prepare data
    rollout_data = load_rollout_pickle(rollout_dir, rollout_name)
    traj_data = prepare_trajectory_data(rollout_data)

    # Set output paths
    if output_dir is None:
        output_dir = rollout_dir
    if output_name is None:
        output_name = rollout_name

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data
    ground_truth = traj_data['ground_truth_rollout']
    predicted = traj_data['predicted_rollout']
    dims = traj_data['dims']
    num_steps = traj_data['num_steps']
    boundaries = traj_data['boundaries']
    particle_types = traj_data['particle_types']
    loss = traj_data['loss']

    # Get color mask
    color_mask = get_color_mask(particle_types)

    # Initialize figure
    fig = plt.figure()

    if dims == 2:
        ax1 = fig.add_subplot(1, 2, 1, projection='rectilinear')
        ax2 = fig.add_subplot(1, 2, 2, projection='rectilinear')

        def animate(i):
            print(f"Render step {i}/{num_steps}")
            fig.clear()

            # Ground truth
            ax1 = fig.add_subplot(1, 2, 1, autoscale_on=False)
            render_2d_trajectory(ax1, ground_truth, i, color_mask, boundaries, point_size)
            ax1.set_title("Reality")

            # Prediction
            ax2 = fig.add_subplot(1, 2, 2, autoscale_on=False)
            render_2d_trajectory(ax2, predicted, i, color_mask, boundaries, point_size)
            ax2.set_title("GNS")

            fig.suptitle(f"{i}/{num_steps}, Total MSE: {loss:.2e}")

    elif dims == 3:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        def animate(i):
            print(f"Render step {i}/{num_steps} for {output_name}")
            fig.clear()

            # Ground truth
            ax1 = fig.add_subplot(1, 2, 1, projection='3d', autoscale_on=False)
            render_3d_trajectory(
                ax1, ground_truth, i, color_mask, boundaries,
                vertical_camera_angle, viewpoint_rotation, change_yz, point_size
            )
            ax1.set_title("Reality")

            # Prediction
            ax2 = fig.add_subplot(1, 2, 2, projection='3d', autoscale_on=False)
            render_3d_trajectory(
                ax2, predicted, i, color_mask, boundaries,
                vertical_camera_angle, viewpoint_rotation, change_yz, point_size
            )
            ax2.set_title("GNS")

            fig.suptitle(f"{i}/{num_steps}, Total MSE: {loss:.2e}")

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=np.arange(0, num_steps, timestep_stride), interval=10
    )

    # Save animation
    output_path = f'{output_dir}{output_name}.gif'
    ani.save(output_path, dpi=100, fps=30, writer='imagemagick')
    print(f"Animation saved to: {output_path}")


# ============================================================================
# VTK Export
# ============================================================================

def write_vtk_trajectory(
        rollout_dir: str,
        rollout_name: str,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None) -> None:
    """Write VTK files for each timestep of the rollout.

    Args:
        rollout_dir: Directory containing rollout pickle files.
        rollout_name: Name of the rollout file (without .pkl extension).
        output_dir: Output directory (defaults to rollout_dir).
        output_name: Output filename prefix (defaults to rollout_name).
    """
    # Load and prepare data
    rollout_data = load_rollout_pickle(rollout_dir, rollout_name)
    traj_data = prepare_trajectory_data(rollout_data)

    # Set output paths
    if output_dir is None:
        output_dir = rollout_dir
    if output_name is None:
        output_name = rollout_name

    # Extract data
    ground_truth = traj_data['ground_truth_rollout']
    predicted = traj_data['predicted_rollout']
    dims = traj_data['dims']

    # Process both ground truth and prediction
    rollout_cases = [
        (ground_truth, "Reality"),
        (predicted, "GNS")
    ]

    for trajectory, label in rollout_cases:
        # Create output directory
        path = f"{output_dir}{output_name}_vtk-{label}"
        if not os.path.exists(path):
            os.makedirs(path)

        # Get initial position for displacement calculation
        initial_position = trajectory[0]

        # Write VTK file for each timestep
        for i, coord in enumerate(trajectory):
            # Calculate displacement from initial position
            disp = np.linalg.norm(coord - initial_position, axis=1)

            # Write VTK file
            pointsToVTK(
                f"{path}/points{i}",
                np.array(coord[:, 0]),
                np.array(coord[:, 1]),
                np.zeros_like(coord[:, 1]) if dims == 2 else np.array(coord[:, 2]),
                data={"displacement": disp}
            )

    print(f"VTK files saved to: {output_dir}{output_name}_vtk-...")
