import torch
import torch.nn as nn
from gns import graph_network
from torch_geometric.nn import radius_graph


class GraphNeuralNetworkModel(nn.Module):
  """Pure GNN model for graph-based particle simulations.

  This class focuses on graph construction and message passing,
  without physics integration or normalization logic.
  """

  def __init__(
          self,
          nnode_in: int,
          nnode_out: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          nparticle_types: int,
          particle_type_embedding_size: int,
          boundaries: torch.Tensor,
          boundary_clamp_limit: float = 1.0,
          device: str = "cpu"
  ):
    """Initializes the GraphNeuralNetworkModel.

    Args:
      nnode_in: Number of node input features.
      nnode_out: Number of node output features.
      nedge_in: Number of edge input features.
      latent_dim: Size of latent dimension (typically 128).
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP.
      mlp_hidden_dim: Hidden dimension size for MLPs.
      connectivity_radius: Radius for graph connectivity.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for particle types.
      boundaries: Array of shape [num_dimensions, 2] containing lower/upper boundaries.
      boundary_clamp_limit: Factor to enlarge connectivity radius for normalized distances.
      device: Runtime device (cuda or cpu).
    """
    super(GraphNeuralNetworkModel, self).__init__()

    self._connectivity_radius = connectivity_radius
    self._nparticle_types = nparticle_types
    self._boundaries = boundaries
    self._boundary_clamp_limit = boundary_clamp_limit
    self._device = device

    # Particle type embedding
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Graph network for message passing
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=nnode_out,
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _compute_graph_connectivity(
          self,
          node_positions: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          radius: float,
          add_self_edges: bool = True):
    """Generate graph edges to all particles within a threshold radius.

    Args:
      node_positions: Node positions with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example.
      radius: Threshold radius for edge construction.
      add_self_edges: Whether to include self edges (default: True).

    Returns:
      Tuple of (receivers, senders) edge indices.
    """
    # Specify batch IDs for particles
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)

    # radius_graph accepts r < radius not r <= radius
    edge_index = radius_graph(
        node_positions, r=radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=128)

    # Flow direction is "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  def build_graph_features(
          self,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          normalized_velocity_sequence: torch.Tensor,
          material_property: torch.Tensor = None):
    """Build node and edge features for the graph network.

    Args:
      position_sequence: Particle position sequence (nparticles, sequence_length, dim).
      nparticles_per_example: Number of particles per example.
      particle_types: Particle types (nparticles,).
      normalized_velocity_sequence: Pre-normalized velocity sequence (nparticles, sequence_length-1, dim).
      material_property: Optional material properties (nparticles, 1).

    Returns:
      Tuple of (node_features, edge_index, edge_features).
    """
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1]  # (nparticles, dim)

    # Get graph connectivity
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)

    # Build node features
    node_features = []

    # Flatten normalized velocity sequence
    flat_velocity_sequence = normalized_velocity_sequence.view(nparticles, -1)
    node_features.append(flat_velocity_sequence)

    # Normalized clipped distances to boundaries
    boundaries = torch.tensor(
        self._boundaries, requires_grad=False).float().to(self._device)
    distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
    distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
    distance_to_boundaries = torch.cat(
        [distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
    normalized_clipped_distance_to_boundaries = torch.clamp(
        distance_to_boundaries / self._connectivity_radius,
        -self._boundary_clamp_limit, self._boundary_clamp_limit)
    node_features.append(normalized_clipped_distance_to_boundaries)

    # Particle type embeddings
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(particle_types)
      node_features.append(particle_type_embeddings)

    # Material properties
    if material_property is not None:
      material_property = material_property.view(nparticles, 1)
      node_features.append(material_property)

    # Build edge features
    edge_features = []

    # Normalized relative displacements
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius
    edge_features.append(normalized_relative_displacements)

    # Normalized relative distances
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def predict(
          self,
          node_features: torch.Tensor,
          edge_index: torch.Tensor,
          edge_features: torch.Tensor) -> torch.Tensor:
    """Run forward pass through the graph network.

    Args:
      node_features: Node features (nparticles, nnode_in).
      edge_index: Edge indices (2, nedges).
      edge_features: Edge features (nedges, nedge_in).

    Returns:
      Output node features (nparticles, nnode_out).
    """
    return self._encode_process_decode(node_features, edge_index, edge_features)

  def save(self, path: str = 'graph_model.pt'):
    """Save model state.

    Args:
      path: Model path.
    """
    torch.save(self.state_dict(), path)

  def load(self, path: str):
    """Load model state from file.

    Args:
      path: Model path.
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
