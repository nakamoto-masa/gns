# Graph Network Simulator (GNS)

Surrogate model for particle-based simulation

A GNN-based surrogate simulator for particle-based physics simulations. Optimized for practical use by AI engineers.

## Project Overview

This project is based on [GNS (Graph Network Simulator)](README.upstream.md) with significant improvements in usability.

Key improvements:
- Improved code maintainability (comprehensive refactoring)
- Modular design for AI engineers
- Updated technology stack (Python 3.13, PyTorch 2.8.0)
- Enhanced training monitoring (TensorBoard integration)
- Improved computational efficiency (edge filtering)

### Refactoring Complete

Comprehensive refactoring has been completed, significantly improving code maintainability and usability:
- Updated to Python 3.13, PyTorch 2.8.0
- Separation of architecture and clear responsibility boundaries
- Type-safe configuration management

Details: [docs/refactor/](docs/refactor/README.md)

### Newly Implemented Features

Features added in this project:
- TensorBoard integration (visualization of training and validation logs)
- Edge filtering (improved computational efficiency by excluding edges between kinematic particles)

### Inherited Features

Key features inherited from the original GNS:
- GNN-based particle trajectory prediction
- Multi-GPU training (distributed training support)
- Improved generalization through noise injection
- Visualization tools (rollout result rendering)
- Inverse problem solver (gradient-based optimization)
- Mesh-based simulation (meshnet module)

See [ARCHITECTURE.md](ARCHITECTURE.md) and [README.upstream.md](README.upstream.md) for details.

## Quick Start

### Environment Setup

```bash
# Environment setup with uv (recommended)
uv sync

# Verify installation
uv run python -c "import torch; import torch_geometric; print('OK')"

# Run tests
uv run pytest test/test_learned_simulator.py
```

### Command Examples

See [scripts/README.md](scripts/README.md) for details.

Training:
```bash
uv run python scripts/gns_train.py \
  --data_path=example/WaterDropSample/ \
  --model_path=models/waterdrop/ \
  --ntraining_steps=100000
```

Inference (rollout):
```bash
uv run python scripts/gns_train.py \
  --mode=rollout \
  --data_path=example/WaterDropSample/ \
  --model_path=models/waterdrop/ \
  --model_file=model-100000.pt \
  --output_path=rollouts/waterdrop/
```

Visualization:
```bash
uv run python scripts/gns_render_rollout.py \
  --rollout_dir=rollouts/waterdrop/ \
  --rollout_name=rollout_0 \
  --output_mode=gif
```

List of options:
```bash
uv run python scripts/gns_train.py --help
uv run python scripts/gns_render_rollout.py --help
```

## Documentation

### For Beginners

- **[Quick Start](#quick-start)** - See above
- **[Script Usage](scripts/README.md)** - Details on training, inference, and visualization
- **[Data Format Specification](docs/gns_data.md)** - Details on `.npz` format

### For Developers

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture overview
- **[Theoretical Background](docs/theory.md)** - Theory of GNNs and message passing
- **[Refactoring Records](docs/refactor/README.md)** - Design decisions and implementation history

### For Researchers

- **[Original Research Project](README.upstream.md)** - Kumar & Vantassel's original paper and results
- **[Inverse Problem Example](docs/example-1.md)** - Gradient-based optimization implementation example

## Project Structure

```
gns/
├── gns/                      # Core module (particle-based)
├── meshnet/                  # Mesh-based model
├── scripts/                  # CLI scripts
├── example/                  # Sample data
├── test/                     # Test suite
├── docs/                     # Detailed documentation
└── models/, rollouts/        # Trained models, prediction results

Details: ARCHITECTURE.md
```

## Technology Stack

- Environment: Python 3.13, uv, pyproject.toml
- Deep Learning: PyTorch 2.8.0, PyTorch Geometric 2.7.0
- Scientific Computing: NumPy, matplotlib, tensorboard

## Acknowledgments

This project is built upon the GNS research by Kumar & Vantassel.
For details on the original research, papers, and citation information, see [README.upstream.md](README.upstream.md).
