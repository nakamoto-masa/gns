"""Rollout prediction utilities for GNS simulator.

This module provides functions for running trajectory predictions (rollouts),
saving predictions, and managing rollout data loaders.
"""

import os
import pickle
from typing import Optional, Tuple

import torch

from gns import config
from gns import data_loader
from gns import inference_utils
from gns import learned_simulator


# ============================================================================
# Data Loading
# ============================================================================

def get_rollout_dataloader(
        data_path: str,
        split: str = 'test'):
    """Get a data loader for rollout predictions.

    Args:
        data_path: Path to data directory.
        split: Data split ('test', 'valid', or 'train'). Defaults to 'test'.

    Returns:
        PyTorch DataLoader instance for trajectory data.
    """
    # Use `valid` set if test doesn't exist
    if split == 'test' and not os.path.isfile(f"{data_path}test.npz"):
        split = 'valid'

    dl = data_loader.get_data_loader_by_trajectories(
        path=f"{data_path}{split}.npz"
    )

    return dl


def infer_dataset_features(dataloader) -> bool:
    """Infer whether dataset includes material properties as features.

    Args:
        dataloader: PyTorch DataLoader instance.

    Returns:
        True if material properties are included, False otherwise.

    Raises:
        NotImplementedError: If the number of features is not 2 or 3.
    """
    n_features = len(dataloader.dataset._data[0])

    if n_features == 3:  # (positions, particle_type, material_property)
        return True
    elif n_features == 2:  # (positions, particle_type)
        return False
    else:
        raise NotImplementedError(f"Expected 2 or 3 features, got {n_features}")


# ============================================================================
# Single Rollout Execution
# ============================================================================

def run_rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.Tensor,
        particle_type: torch.Tensor,
        material_property: Optional[torch.Tensor],
        n_particles_per_example: torch.Tensor,
        nsteps: int,
        simulator_config: config.SimulatorConfig,
        device: torch.device) -> Tuple[dict, torch.Tensor]:
    """Run a single trajectory rollout prediction.

    Args:
        simulator: Learned simulator model.
        position: Initial positions (nparticles, timesteps, ndims).
        particle_type: Particle type IDs (nparticles,).
        material_property: Material properties (nparticles,) or None.
        n_particles_per_example: Number of particles.
        nsteps: Number of prediction steps.
        simulator_config: Simulator configuration.
        device: Target device for computation.

    Returns:
        Tuple of (rollout_dict, loss_per_step):
            - rollout_dict: Dictionary containing predicted trajectories
            - loss_per_step: Loss for each prediction step
    """
    rollout_dict, loss_per_step = inference_utils.rollout(
        simulator=simulator,
        position=position,
        particle_types=particle_type,
        material_property=material_property,
        n_particles_per_example=n_particles_per_example,
        nsteps=nsteps,
        input_sequence_length=simulator_config.input_sequence_length,
        kinematic_particle_id=simulator_config.kinematic_particle_id,
        device=device
    )

    return rollout_dict, loss_per_step


# ============================================================================
# Rollout Saving
# ============================================================================

def save_rollout(
        rollout_data: dict,
        metadata: dict,
        loss: torch.Tensor,
        output_path: str,
        output_filename: str,
        example_index: int) -> None:
    """Save a rollout prediction to disk.

    Args:
        rollout_data: Dictionary containing rollout predictions.
        metadata: Dataset metadata.
        loss: Mean loss for this rollout.
        output_path: Directory path for saving.
        output_filename: Base filename for the rollout.
        example_index: Index of the example being saved.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Add metadata and loss to rollout data
    rollout_data['metadata'] = metadata
    rollout_data['loss'] = loss

    # Save to pickle file
    filename = f'{output_filename}_ex{example_index}.pkl'
    filepath = os.path.join(output_path, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(rollout_data, f)


# ============================================================================
# Batch Prediction
# ============================================================================

def predict_rollouts(
        simulator: learned_simulator.LearnedSimulator,
        data_path: str,
        split: str,
        simulator_config: config.SimulatorConfig,
        device: torch.device,
        output_path: Optional[str] = None,
        output_filename: str = 'rollout',
        save_results: bool = True) -> float:
    """Predict rollouts for an entire dataset.

    Args:
        simulator: Trained simulator model.
        data_path: Path to data directory.
        split: Data split ('test', 'valid', or 'train').
        simulator_config: Simulator configuration.
        device: Target device for computation.
        output_path: Directory path for saving results (required if save_results=True).
        output_filename: Base filename for rollouts.
        save_results: Whether to save rollout predictions to disk.

    Returns:
        Mean loss across all rollouts.
    """
    # Set simulator to evaluation mode
    simulator.to(device)
    simulator.eval()

    # Read metadata
    from gns import reading_utils
    metadata = reading_utils.read_metadata(data_path, "rollout")

    # Get data loader
    dl = get_rollout_dataloader(data_path, split)
    has_material_property = infer_dataset_features(dl)

    eval_loss = []

    with torch.no_grad():
        for example_i, features in enumerate(dl):
            print(f"Processing example number {example_i}")

            # Extract features
            positions = features[0].to(device)
            particle_type = features[1].to(device)

            if has_material_property:
                material_property = features[2].to(device)
                n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)
            else:
                material_property = None
                n_particles_per_example = torch.tensor([int(features[2])], dtype=torch.int32).to(device)

            # Determine number of steps
            if metadata['sequence_length'] is not None:
                nsteps = metadata['sequence_length'] - simulator_config.input_sequence_length
            else:
                sequence_length = positions.shape[1]
                nsteps = sequence_length - simulator_config.input_sequence_length

            # Run rollout prediction
            example_rollout, loss = run_rollout(
                simulator,
                positions,
                particle_type,
                material_property,
                n_particles_per_example,
                nsteps,
                simulator_config,
                device
            )

            mean_loss = loss.mean()
            print(f"Predicting example {example_i} loss: {mean_loss}")
            eval_loss.append(torch.flatten(loss))

            # Save rollout if requested
            if save_results:
                if output_path is None:
                    raise ValueError("output_path must be provided when save_results=True")
                save_rollout(
                    example_rollout,
                    metadata,
                    mean_loss,
                    output_path,
                    output_filename,
                    example_i
                )

    # Calculate mean loss
    mean_eval_loss = torch.mean(torch.cat(eval_loss))
    print(f"Mean loss on rollout prediction: {mean_eval_loss}")

    return mean_eval_loss.item()


# ============================================================================
# Distributed Prediction (Future Extension)
# ============================================================================

def predict_rollouts_distributed(
        simulator: learned_simulator.LearnedSimulator,
        data_path: str,
        split: str,
        simulator_config: config.SimulatorConfig,
        device: torch.device,
        rank: int,
        world_size: int,
        output_path: Optional[str] = None,
        output_filename: str = 'rollout',
        save_results: bool = True) -> float:
    """Predict rollouts for a dataset using distributed computing.

    This function distributes rollout predictions across multiple processes,
    with each process handling a subset of examples.

    Args:
        simulator: Trained simulator model.
        data_path: Path to data directory.
        split: Data split ('test', 'valid', or 'train').
        simulator_config: Simulator configuration.
        device: Target device for computation.
        rank: Process rank.
        world_size: Total number of processes.
        output_path: Directory path for saving results (required if save_results=True).
        output_filename: Base filename for rollouts.
        save_results: Whether to save rollout predictions to disk.

    Returns:
        Mean loss across all rollouts for this rank.

    Note:
        This is a future extension and currently just calls predict_rollouts().
        A full implementation would distribute examples across ranks.
    """
    # TODO: Implement distributed rollout prediction
    # For now, just call the single-process version
    print(f"Warning: Distributed rollout not yet implemented. Using single-process version.")
    return predict_rollouts(
        simulator, data_path, split, simulator_config, device,
        output_path, output_filename, save_results
    )


# ============================================================================
# Convenience Function (Legacy Compatibility)
# ============================================================================

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.Tensor,
        particle_types: torch.Tensor,
        material_property: torch.Tensor,
        n_particles_per_example: torch.Tensor,
        nsteps: int,
        simulator_config: config.SimulatorConfig,
        device: torch.device) -> Tuple[dict, torch.Tensor]:
    """Legacy wrapper for run_rollout() to maintain compatibility.

    Args:
        simulator: Learned simulator.
        position: Positions of particles (nparticles, timesteps, ndims).
        particle_types: Particles types with shape (nparticles).
        material_property: Friction angle normalized by tan() with shape (nparticles).
        n_particles_per_example: Number of particles per example.
        nsteps: Number of steps.
        simulator_config: Simulator configuration.
        device: torch device.

    Returns:
        Tuple of (rollout_dict, loss_per_step).
    """
    return run_rollout(
        simulator, position, particle_types, material_property,
        n_particles_per_example, nsteps, simulator_config, device
    )
