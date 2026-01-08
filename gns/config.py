"""Configuration classes for GNS simulator.

This module provides dataclasses for managing simulator configuration,
including parameters that were previously hardcoded as global constants.
"""

from dataclasses import dataclass, field
from typing import Any


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
