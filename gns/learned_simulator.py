import torch
import torch.nn as nn
import numpy as np
from gns.graph_model import GraphNeuralNetworkModel


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf.

  This class wraps GraphNeuralNetworkModel and handles:
  - Normalization/denormalization
  - Euler integration for physics
  - Position/acceleration prediction
  """

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          boundaries: np.ndarray,
          normalization_stats: dict,
          nparticle_types: int,
          particle_type_embedding_size: int,
          boundary_clamp_limit: float = 1.0,
          device: str = "cpu"
  ):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Hidden dimension size for MLPs.
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      boundary_clamp_limit: a factor to enlarge connectivity radius used for computing
        normalized clipped distance in edge feature.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    self._normalization_stats = normalization_stats
    self._device = device

    # Initialize the graph neural network model
    self._graph_model = GraphNeuralNetworkModel(
        nnode_in=nnode_in,
        nnode_out=particle_dimensions,
        nedge_in=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        connectivity_radius=connectivity_radius,
        nparticle_types=nparticle_types,
        particle_type_embedding_size=particle_type_embedding_size,
        boundaries=boundaries,
        boundary_clamp_limit=boundary_clamp_limit,
        device=device)

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _normalize_velocity_sequence(
          self,
          position_sequence: torch.Tensor) -> torch.Tensor:
    """Normalize velocity sequence from position sequence.

    Args:
      position_sequence: Position sequence (nparticles, sequence_length, dim).

    Returns:
      Normalized velocity sequence (nparticles, sequence_length-1, dim).
    """
    velocity_sequence = time_diff(position_sequence)
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    return normalized_velocity_sequence

  def _encoder_preprocessor(
          self,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None):
    """Extracts important features from the position sequence.

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      material_property: Friction angle normalized by tan() with shape (nparticles)

    Returns:
      Tuple of (node_features, edge_index, edge_features).
    """
    # Normalize velocity sequence
    normalized_velocity_sequence = self._normalize_velocity_sequence(position_sequence)

    # Build graph features using the graph model
    return self._graph_model.build_graph_features(
        position_sequence=position_sequence,
        nparticles_per_example=nparticles_per_example,
        particle_types=particle_types,
        normalized_velocity_sequence=normalized_velocity_sequence,
        material_property=material_property)

  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.Tensor,
          position_sequence: torch.Tensor) -> torch.Tensor:
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, sequence_length, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
    # Extract real acceleration values from normalized values
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    # TODO: Fix dt
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def predict_positions(
          self,
          current_positions: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None) -> torch.Tensor:
    """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, sequence_length, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      material_property: Friction angle normalized by tan() with shape (nparticles)

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        current_positions, nparticles_per_example, particle_types, material_property)
    predicted_normalized_acceleration = self._graph_model.predict(
        node_features, edge_index, edge_features)
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    return next_positions

  def predict_accelerations(
          self,
          next_positions: torch.Tensor,
          position_sequence_noise: torch.Tensor,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None):
    """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
      material_property: Friction angle normalized by tan() with shape (nparticles).

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized accelerations.

    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        noisy_position_sequence, nparticles_per_example, particle_types, material_property)
    predicted_normalized_acceleration = self._graph_model.predict(
        node_features, edge_index, edge_features)

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_positions + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration

  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.Tensor,
          position_sequence: torch.Tensor):
    """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.

    Returns:
      normalized_acceleration (torch.tensor): Normalized acceleration.

    """
    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']
    return normalized_acceleration

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def time_diff(
        position_sequence: torch.Tensor) -> torch.Tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return position_sequence[:, 1:] - position_sequence[:, :-1]
