"""Inference utilities for GNS simulator.

This module provides utilities for running inference with the learned simulator,
including rollout execution and kinematic constraints application.
"""

import torch
from tqdm import tqdm
from typing import Optional, Tuple, Dict
import numpy as np

from gns import learned_simulator


def apply_kinematic_constraints(
    predicted_position: torch.Tensor,
    ground_truth_position: torch.Tensor,
    particle_types: torch.Tensor,
    kinematic_particle_id: int,
    device: torch.device
) -> torch.Tensor:
    """Apply kinematic constraints to predicted positions.

    Kinematic particles (e.g., boundary walls) follow a prescribed trajectory
    rather than the simulator's predictions.

    Args:
        predicted_position: Predicted next position (nparticles, ndims)
        ground_truth_position: Ground truth next position (nparticles, ndims)
        particle_types: Particle types (nparticles,)
        kinematic_particle_id: ID for kinematic particles
        device: torch device

    Returns:
        Position tensor with kinematic constraints applied (nparticles, ndims)
    """
    kinematic_mask = (particle_types == kinematic_particle_id).clone().detach().to(device)
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, predicted_position.shape[-1])

    constrained_position = torch.where(
        kinematic_mask, ground_truth_position, predicted_position
    )

    return constrained_position


def run_inference_loop(
    simulator: learned_simulator.LearnedSimulator,
    initial_positions: torch.Tensor,
    ground_truth_positions: torch.Tensor,
    particle_types: torch.Tensor,
    material_property: Optional[torch.Tensor],
    n_particles_per_example: int,
    nsteps: int,
    kinematic_particle_id: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference loop to generate rollout predictions.

    Args:
        simulator: Learned simulator
        initial_positions: Initial positions (nparticles, input_sequence_length, ndims)
        ground_truth_positions: Ground truth trajectory (nparticles, nsteps, ndims)
        particle_types: Particle types (nparticles,)
        material_property: Material properties (nparticles,) or None
        n_particles_per_example: Number of particles
        nsteps: Number of rollout steps
        kinematic_particle_id: ID for kinematic particles
        device: torch device

    Returns:
        predictions: Predicted trajectory (nsteps, nparticles, ndims)
        loss: Per-timestep MSE loss (nsteps, nparticles, ndims)
    """
    current_positions = initial_positions
    predictions = []

    for step in tqdm(range(nsteps), total=nsteps):
        # Predict next position
        next_position = simulator.predict_positions(
            current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types,
            material_property=material_property
        )

        # Apply kinematic constraints
        next_position_ground_truth = ground_truth_positions[:, step]
        next_position = apply_kinematic_constraints(
            next_position,
            next_position_ground_truth,
            particle_types,
            kinematic_particle_id,
            device
        )

        predictions.append(next_position)

        # Update current positions: shift window and append new position
        current_positions = torch.cat(
            [current_positions[:, 1:], next_position[:, None, :]], dim=1
        )

    # Stack predictions: (nsteps, nparticles, ndims)
    predictions = torch.stack(predictions)

    # Permute ground truth to match: (nsteps, nparticles, ndims)
    ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

    # Compute MSE loss
    loss = (predictions - ground_truth_positions) ** 2

    return predictions, loss


def rollout(
    simulator: learned_simulator.LearnedSimulator,
    position: torch.Tensor,
    particle_types: torch.Tensor,
    material_property: Optional[torch.Tensor],
    n_particles_per_example: int,
    nsteps: int,
    input_sequence_length: int,
    kinematic_particle_id: int,
    device: torch.device
) -> Tuple[Dict[str, np.ndarray], torch.Tensor]:
    """Roll out a trajectory by applying the model in sequence.

    Args:
        simulator: Learned simulator
        position: Positions of particles (nparticles, timesteps, ndims)
        particle_types: Particle types (nparticles,)
        material_property: Material properties (nparticles,) or None
        n_particles_per_example: Number of particles
        nsteps: Number of rollout steps
        input_sequence_length: Length of input sequence (e.g., 6)
        kinematic_particle_id: ID for kinematic particles (e.g., 3)
        device: torch device

    Returns:
        output_dict: Dictionary containing:
            - initial_positions: (timesteps, nparticles, ndims)
            - predicted_rollout: (nsteps, nparticles, ndims)
            - ground_truth_rollout: (nsteps, nparticles, ndims)
            - particle_types: (nparticles,)
            - material_property: (nparticles,) or None
        loss: Per-timestep MSE loss (nsteps, nparticles, ndims)
    """
    # Split position into initial and ground truth
    initial_positions = position[:, :input_sequence_length]
    ground_truth_positions = position[:, input_sequence_length:]

    # Run inference loop
    predictions, loss = run_inference_loop(
        simulator=simulator,
        initial_positions=initial_positions,
        ground_truth_positions=ground_truth_positions,
        particle_types=particle_types,
        material_property=material_property,
        n_particles_per_example=n_particles_per_example,
        nsteps=nsteps,
        kinematic_particle_id=kinematic_particle_id,
        device=device
    )

    # Prepare output dictionary
    output_dict = {
        'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.permute(1, 0, 2).cpu().numpy(),
        'particle_types': particle_types.cpu().numpy(),
        'material_property': material_property.cpu().numpy() if material_property is not None else None
    }

    return output_dict, loss
