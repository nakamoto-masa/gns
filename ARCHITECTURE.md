# Architecture

**Last updated:** 2026-03-13

This document describes the internal design of GNS.
For a project overview and usage instructions, see [README.md](README.md).

## Table of Contents

- [1. Overall Architecture](#1-overall-architecture)
- [2. Module Structure](#2-module-structure)
- [3. Core Components](#3-core-components)
- [4. Data Flow](#4-data-flow)
- [5. Training and Inference](#5-training-and-inference)
- [6. Design Decisions](#6-design-decisions)

---

## 1. Overall Architecture

### 1.1 Layer Structure

```
┌──────────────────────────────────────────────────────────┐
│                  CLI Layer (scripts/)                    │
│  Thin wrappers: argument parsing -> business logic calls │
├──────────────────────────────────────────────────────────┤
│              Business Logic Layer (gns/)                 │
│  Core logic for training, inference, and visualization   │
├──────────────────────────────────────────────────────────┤
│                   Model Layer (gns/)                     │
│  GNN model definitions and simulator integration         │
├──────────────────────────────────────────────────────────┤
│                    Data Layer (gns/)                     │
│  Data loading, preprocessing, and normalization          │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Separation of Concerns

Through Phase 0-3 refactoring, the CLI/logic/model layers were clearly separated, achieving a design based on the Single Responsibility Principle.

Details: [docs/refactor/refactoring-results.md](docs/refactor/refactoring-results.md)

---

## 2. Module Structure

### 2.1 gns/ (Particle-Based Domain)

Core module for particle-based simulations (Sand, Water, etc.).

```
gns/
├── config.py               # Configuration dataclass (SimulatorConfig)
├── graph_network.py        # GNN base components / pure GNN model
├── learned_simulator.py    # Physics-integrated simulator
├── training.py             # Training logic
├── rollout.py              # Inference logic
├── inference_utils.py      # Inference helpers
├── render.py               # Visualization
├── data_loader.py          # Data loader
├── noise_utils.py          # Noise injection
├── reading_utils.py        # Data reading
├── metadata.py             # Metadata management
└── distribute.py           # Distributed training support
```

**Core component details:** [3. Core Components](#3-core-components)

### 2.2 meshnet/ (Mesh-Based Domain)

Module for mesh-based simulations (FEA, CFD, etc.).

```
meshnet/
├── learned_simulator.py    # MeshNet simulator
├── data_loader.py          # Mesh data loader
├── normalization.py        # Normalization
├── noise.py                # Noise injection
├── train.py                # Training script
├── render.py               # Visualization
└── utils.py                # Utilities
```

**Note:** MeshNet is not subject to refactoring (original structure maintained)

### 2.3 scripts/ (CLI Layer)

Thin CLI wrappers serving as the user interface.

```
scripts/
├── gns_train.py            # Training/rollout CLI
├── gns_train_multinode.py  # Distributed training CLI
├── gns_render_rollout.py   # Visualization CLI
├── legacy/                 # Legacy scripts (pre-refactoring)
└── README.md               # Usage details
```

**Design principles:**
- Responsible only for argument parsing
- Business logic is delegated to `gns/` modules
- One script = one clear purpose

Details: [scripts/README.md](scripts/README.md)

### 2.4 example/ (Sample Data)

```
example/
├── WaterDropSample/        # Small-scale training data
│   ├── train.npz           # Training data
│   ├── valid.npz           # Validation data
│   ├── test.npz            # Test data
│   └── metadata.json       # Simulation configuration
└── inverse_problem/        # Inverse problem implementation example
    ├── config.toml         # Optimization configuration
    ├── forward.py          # Forward problem
    ├── inverse.py          # Inverse problem solver
    └── utils.py            # Utilities
```

### 2.5 test/ (Test Suite)

```
test/
├── test_learned_simulator.py    # LearnedSimulator tests
├── test_graph_network.py        # GraphNetwork tests
├── test_noise_utils.py          # Noise function tests
├── test_data_loader.py          # Data loader tests
├── test_pytorch.py              # PyTorch verification
└── test_torch_geometric.py      # PyG verification
```

---

## 3. Core Components

### 3.1 GraphNeuralNetworkModel (`gns/graph_network.py`)

A pure GNN model (without physics or normalization). Uses the Encoder-Processor-Decoder architecture to propagate information through message passing.

Implementation: [gns/graph_network.py](gns/graph_network.py) | Theory: [docs/theory.md](docs/theory.md)

### 3.2 LearnedSimulator (`gns/learned_simulator.py`)

Integrates the GNN model with physics simulation. Centrally manages graph construction, feature extraction, normalization, GNN prediction, physics integration (Euler integration), and boundary conditions.

Implementation: [gns/learned_simulator.py](gns/learned_simulator.py) | Design decision: [docs/refactor/decisions/0001-keep-learned-simulator-as-nn-module.md](docs/refactor/decisions/0001-keep-learned-simulator-as-nn-module.md)

### 3.3 SimulatorConfig (`gns/config.py`)

A dataclass for type-safe configuration management. Aggregates model architecture, simulation parameters, and edge filtering settings, supporting type checking and code completion.

Implementation: [gns/config.py](gns/config.py)

### 3.4 Encoder-Processor-Decoder (`gns/graph_network.py`)

The fundamental GNN structure:
- **Encoder**: Embeds input features into latent space
- **Processor**: Propagates information through message passing (InteractionNetwork × N iterations)
- **Decoder**: Predicts physical quantities (acceleration) from latent space

Implementation: [gns/graph_network.py](gns/graph_network.py) | Theory: [docs/theory.md](docs/theory.md)

---

## 4. Data Flow

### 4.1 Training

1. Data Files
2. DataLoader (noise injection, normalization)
3. Training Loop (batch splitting, loss computation, gradient updates)
4. LearnedSimulator (graph construction, feature extraction, GNN prediction, physics integration)
5. Checkpoint saving

### 4.2 Inference (Rollout)

1. Test Data
2. Rollout Loop
3. LearnedSimulator.predict_positions (recursive single-step predictions)
4. Output saving (rollout_{i}.pkl)
5. Visualization (GIF/VTK/JPG)

### 4.3 Data Format

**Input data (`.npz`)**:
```python
{
    'positions': (n_trajectories,) array of (n_timesteps, n_particles, n_dims)
    'particle_types': (n_trajectories,) array of (n_particles,)
    'material_property': (n_trajectories,) array of (n_particles, n_features)  # optional
}
```

**Metadata (`metadata.json`)**:
```json
{
  "bounds": [[x_min, x_max], [y_min, y_max], ...],
  "sequence_length": 320,
  "default_connectivity_radius": 0.015,
  "dim": 2,
  "dt": 0.0025,
  "vel_mean": [...],
  "vel_std": [...],
  "acc_mean": [...],
  "acc_std": [...],
  "nparticle_types": 9,
  "particle_type_embedding_size": 16,
  "material_feature_len": 5
}
```

Details: [docs/gns_data.md](docs/gns_data.md)

---

## 5. Training and Inference

### 5.1 Training Process

Training is managed by `gns/training.py`. It retrieves batches from the data loader, executes predictions through the LearnedSimulator, and repeats loss computation and gradient updates. For distributed training, `torch.distributed` is used to enable multi-GPU/multi-node training.

Implementation: [gns/training.py](gns/training.py)

### 5.2 Inference Process (Rollout)

Inference is managed by `gns/rollout.py`. Starting from the initial state, it recursively calls LearnedSimulator.predict_positions one step at a time to generate the entire trajectory. Results are saved in pickle format and used for subsequent visualization.

Implementation: [gns/rollout.py](gns/rollout.py)

### 5.3 Usage

For command examples and detailed usage, see:
- [README.md - Quick Start](README.md#quick-start)
- [scripts/README.md](scripts/README.md)

---

## 6. Design Decisions

### 6.1 Refactoring Policy

**Goal:** Make the model easier to use for practitioners

**Scope:** Code quality improvement of the GNS model (MeshNet excluded)

**Constraints:**
- Do not change algorithmic behavior
- Make changes incrementally, maintaining functionality at each stage
- Prioritize separation by responsibility

Details: [docs/refactor/refactoring-policy.md](docs/refactor/refactoring-policy.md)

### 6.2 Refactoring Results

Completed 29 tasks across Phases 0-3: environment setup (Python 3.13, PyTorch 2.8.0), architecture separation, script separation, and type-safe configuration management.

Details: [docs/refactor/tasks.md](docs/refactor/tasks.md) | [docs/refactor/refactoring-results.md](docs/refactor/refactoring-results.md)

### 6.3 Design Decisions and Test Reports

Details: [docs/refactor/decisions/](docs/refactor/decisions/) | [docs/refactor/reports/](docs/refactor/reports/)
