# GNS Scripts Directory

This directory contains executable scripts for training, rollout, and rendering operations with the Graph Network-based Simulator (GNS).

## Main Scripts

### gns_train.py
**Purpose**: Train a GNS model or run rollout predictions on trained models.

**Usage**:
```bash
# Training mode
python scripts/gns_train.py \
  --data_path=example/WaterDropSample/ \
  --model_path=models/test/ \
  --ntraining_steps=10 \
  --mode=train

# Rollout mode
python scripts/gns_train.py \
  --data_path=example/WaterDropSample/ \
  --model_path=models/test/ \
  --model_file=model-10.pt \
  --mode=rollout \
  --output_path=rollouts/test/
```

**Key Features**:
- Thin CLI wrapper (~180 lines) that delegates to reusable modules
- Supports both training and rollout modes
- Automatic checkpoint saving and resumption
- Learning rate scheduling with exponential decay

**Underlying Modules**:
- `gns.training` - Training loop orchestration and utilities
- `gns.rollout` - Rollout execution and result saving
- `gns.learned_simulator` - Physics simulator with neural network
- `gns.config` - Configuration management

### gns_train_multinode.py
**Purpose**: Multi-node distributed training variant of gns_train.py for HPC environments.

**Usage**:
```bash
# Distributed training mode
python scripts/gns_train_multinode.py \
  --data_path=example/WaterDropSample/ \
  --model_path=models/test/ \
  --ntraining_steps=10 \
  --mode=train

# Distributed rollout mode
python scripts/gns_train_multinode.py \
  --data_path=example/WaterDropSample/ \
  --model_path=models/test/ \
  --model_file=model-10.pt \
  --mode=rollout \
  --output_path=rollouts/test/
```

**Key Features**:
- Supports PyTorch distributed data parallel (DDP) training
- Compatible with SLURM job schedulers
- Parallel rollout predictions across multiple GPUs

### gns_render_rollout.py
**Purpose**: Visualize rollout predictions as images, GIF animations, or VTK files.

**Usage**:
```bash
# Render as images
python scripts/gns_render_rollout.py \
  --rollout_path=rollouts/test/rollout_ex0.pkl \
  --output_mode=jpg

# Render as GIF animation
python scripts/gns_render_rollout.py \
  --rollout_path=rollouts/test/rollout_ex0.pkl \
  --output_mode=gif

# Export to VTK format
python scripts/gns_render_rollout.py \
  --rollout_path=rollouts/test/rollout_ex0.pkl \
  --output_mode=vtk
```

**Key Features**:
- Thin CLI wrapper (~56 lines) delegating to `gns.render`
- Supports 2D and 3D trajectory visualization
- Multiple output formats: JPG, GIF, VTK
- Configurable rendering parameters (step stride, rotation, etc.)

**Underlying Module**:
- `gns.render` - Rendering logic for 2D/3D trajectories and animations

## Environment Setup

The project uses `uv` for Python environment management with Python 3.13.

**Setup**:
```bash
# Install dependencies (creates venv automatically)
uv sync

# Run any script with uv
uv run python scripts/gns_train.py --help
```

**Dependencies**: See [pyproject.toml](../pyproject.toml) for the complete list.

## Legacy Scripts

The `legacy/` subdirectory contains deprecated shell scripts from the old environment setup system:

- `build_venv.sh` - Old virtualenv setup script
- `build_venv_frontera.sh` - HPC-specific virtualenv setup
- `module.sh` - Module loading for HPC systems
- `start_venv.sh` - Virtualenv activation script
- `run.sh` - Old training execution wrapper

**Note**: These scripts are **no longer maintained** and are kept only for reference. Use the modern `uv`-based workflow instead.

## Architecture

The refactored architecture separates concerns:

- **scripts/** - Thin CLI wrappers (50-200 lines) for user interaction
- **gns/** - Reusable Python modules with core logic
  - `training.py` - Training orchestration (~788 lines)
  - `rollout.py` - Rollout execution (~304 lines)
  - `render.py` - Visualization logic (~349 lines)
  - `learned_simulator.py` - Physics simulator
  - `graph_model.py` - GNN model core
  - `config.py` - Configuration management
  - `inference_utils.py` - Inference utilities

This separation enables:
- **Code reuse** - Import and use specific modules programmatically
- **Testability** - Business logic decoupled from CLI
- **Maintainability** - Clear separation of concerns

## Related Documentation

- [Refactoring Plan](../docs/refactor/plan.md) - Overall refactoring strategy
- [Task List](../docs/refactor/tasks.md) - Implementation progress
- [Architecture Decisions](../docs/refactor/decisions/) - Design decision records
