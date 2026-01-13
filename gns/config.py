"""Configuration classes for GNS simulator.

This module provides dataclasses for managing simulator configuration,
including parameters that were previously hardcoded as global constants.
"""

from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class SimulatorConfig:
    """Configuration for the GNS simulator.

    This class consolidates dataset-specific parameters that were previously
    defined as global constants, making them configurable per dataset.

    Attributes:
        input_sequence_length: Number of time steps in input sequence.
            Used to calculate velocities from position history.
        num_particle_types: Number of different particle types in the dataset.
        kinematic_particle_id: ID of the kinematic particle type.
            Kinematic particles have their positions overridden with ground truth.
        dim: Spatial dimension of the simulation (2 or 3).
        dt: Time step size.
        default_connectivity_radius: Radius for connecting particles in the graph.
        bounds: Spatial boundaries of the simulation domain.
        sequence_length: Total length of sequences in the dataset.
        acc_mean: Mean acceleration for normalization.
        acc_std: Standard deviation of acceleration for normalization.
        vel_mean: Mean velocity for normalization.
        vel_std: Standard deviation of velocity for normalization.
        nnode_in: Number of input features per node (optional, inferred if not provided).
        nedge_in: Number of input features per edge (optional, inferred if not provided).
        boundary_augment: Boundary clamp limit for training (optional).
    """

    input_sequence_length: int = 6
    num_particle_types: int = 9
    kinematic_particle_id: int = 3

    # Metadata fields
    dim: int | None = None
    dt: float | None = None
    default_connectivity_radius: float | None = None
    bounds: list[list[float]] | None = None
    sequence_length: int | None = None
    acc_mean: list[float] | None = None
    acc_std: list[float] | None = None
    vel_mean: list[float] | None = None
    vel_std: list[float] | None = None
    nnode_in: int | None = None
    nedge_in: int | None = None
    boundary_augment: float | None = None

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "SimulatorConfig":
        """Create a SimulatorConfig from metadata dictionary.

        Args:
            metadata: Dictionary containing dataset metadata, typically loaded
                from metadata.json file.

        Returns:
            SimulatorConfig instance with fields populated from metadata.

        Example:
            >>> import json
            >>> with open('metadata.json') as f:
            ...     metadata = json.load(f)['train']
            >>> config = SimulatorConfig.from_metadata(metadata)
        """
        return cls(
            dim=metadata.get('dim'),
            dt=metadata.get('dt'),
            default_connectivity_radius=metadata.get('default_connectivity_radius'),
            bounds=metadata.get('bounds'),
            sequence_length=metadata.get('sequence_length'),
            acc_mean=metadata.get('acc_mean'),
            acc_std=metadata.get('acc_std'),
            vel_mean=metadata.get('vel_mean'),
            vel_std=metadata.get('vel_std'),
            nnode_in=metadata.get('nnode_in'),
            nedge_in=metadata.get('nedge_in'),
            boundary_augment=metadata.get('boundary_augment'),
        )


def build_normalization_stats(
        metadata: dict[str, Any],
        acc_noise_std: float,
        vel_noise_std: float,
        device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    """Build normalization statistics from metadata.

    This function extracts normalization statistics (mean and std) from metadata
    and combines them with noise standard deviations for training.

    The standard deviation is computed as:
        combined_std = sqrt(metadata_std^2 + noise_std^2)

    This accounts for both the inherent variability in the data and the noise
    added during training.

    Args:
        metadata: Dictionary containing dataset metadata with 'acc_mean', 'acc_std',
            'vel_mean', and 'vel_std' fields.
        acc_noise_std: Standard deviation of noise added to accelerations during training.
        vel_noise_std: Standard deviation of noise added to velocities during training.
        device: PyTorch device ('cpu' or 'cuda') to place tensors on.

    Returns:
        Dictionary with normalization statistics in the format:
        {
            'acceleration': {'mean': tensor, 'std': tensor},
            'velocity': {'mean': tensor, 'std': tensor}
        }

    Example:
        >>> metadata = {'acc_mean': [0.0], 'acc_std': [1.0], ...}
        >>> stats = build_normalization_stats(metadata, 0.0003, 0.0003, torch.device('cpu'))
        >>> stats['acceleration']['mean']
        tensor([0.])
    """
    normalization_stats = {
        'acceleration': {
            'mean': torch.FloatTensor(metadata['acc_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 +
                              acc_noise_std**2).to(device),
        },
        'velocity': {
            'mean': torch.FloatTensor(metadata['vel_mean']).to(device),
            'std': torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 +
                              vel_noise_std**2).to(device),
        },
    }
    return normalization_stats


def infer_feature_dimensions(
        metadata: dict[str, Any],
        simulator_config: SimulatorConfig) -> dict[str, int]:
    """Infer feature dimensions from metadata.

    This function determines the number of input features for nodes and edges
    in the graph neural network. If these dimensions are explicitly provided
    in the metadata, they are used directly. Otherwise, they are computed
    based on the spatial dimension and configuration parameters.

    Node feature dimensions:
        - position: dim
        - velocity: dim * input_sequence_length
        - particle_type: particle_type_embedding_size (16)
        Total: dim + dim * input_sequence_length + 16

    Edge feature dimensions:
        - relative_position: dim
        - distance: 1
        Total: dim + 1

    Args:
        metadata: Dictionary containing dataset metadata. May include 'nnode_in'
            and 'nedge_in' for explicit dimension specification.
        simulator_config: Simulator configuration containing input_sequence_length.

    Returns:
        Dictionary with the following keys:
        - 'nnode_in': Number of input features per node
        - 'nedge_in': Number of input features per edge
        - 'particle_dimensions': Spatial dimension (2 or 3)

    Example:
        >>> metadata = {'dim': 3}
        >>> config = SimulatorConfig(input_sequence_length=6)
        >>> dims = infer_feature_dimensions(metadata, config)
        >>> dims['nnode_in']  # 3 + 3*6 + 16 = 37
        37
        >>> dims['nedge_in']  # 3 + 1 = 4
        4
    """
    dim = metadata['dim']

    # Check if dimensions are explicitly provided in metadata
    if "nnode_in" in metadata and "nedge_in" in metadata:
        nnode_in = metadata['nnode_in']
        nedge_in = metadata['nedge_in']
    else:
        # Infer dimensions from spatial dimension and configuration
        # Node features: position (dim) + velocity (dim * input_sequence_length) +
        #                particle_type embedding (16)
        particle_type_embedding_size = 16
        nnode_in = (dim +
                    dim * simulator_config.input_sequence_length +
                    particle_type_embedding_size)

        # Edge features: relative_position (dim) + distance (1)
        nedge_in = dim + 1

    return {
        'nnode_in': nnode_in,
        'nedge_in': nedge_in,
        'particle_dimensions': dim
    }
