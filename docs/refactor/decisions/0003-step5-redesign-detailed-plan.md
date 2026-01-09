# Step 5 Redesign: Detailed Implementation Plan

**Date**: 2026-01-08
**Status**: Accepted
**Related**: Step 5 (Separating modules from execution scripts)

## Background

### First Implementation Failure (refactor/step5_1st_try branch)

**Result**: `scripts/gns_train.py` became 489 lines (target: 50-80 lines)

**Cause**: Missing intermediate-level abstraction layer
- No orchestration for the entire training loop (`run_training_loop()`)
- Batch preparation boilerplate (15 lines of if/else) duplicated in 3 places
- Learning rate scheduling was hardcoded
- Validation logic was embedded in training loop

**Functions extracted in first attempt** (gns/training.py):
- `train_step()`, `validation_step()` - single-step operations
- `save_checkpoint()`, `load_checkpoint()` - checkpoint management
- `optimizer_to()`, `acceleration_loss()` - utilities

**Problem**: These were only low-level functions, lacking high-level functions to integrate the entire training loop.

## Redesign Policy

### Goals
- `scripts/gns_train.py`: 60-80 lines (CLI flag parsing + high-level function calls only)
- `scripts/gns_train_multinode.py`: 70-90 lines
- `scripts/gns_render_rollout.py`: 35-50 lines

### Critical New Functions to Add

#### 1. `run_training_loop()` - Most Important
High-level orchestration function to manage the entire training loop. This is the most important abstraction that was missing in the first attempt.

**Elements to integrate**:
- Epoch and step iteration
- Batch preparation (`prepare_training_batch`)
- Training step (`train_step`)
- Validation step (`validation_step`)
- Learning rate update (`update_learning_rate`)
- Checkpoint saving (`save_checkpoint`)
- Progress logging

#### 2. `prepare_training_batch()` - Important
Replace 15 lines of boilerplate (feature extraction if/else branching) with a single function call.

**Current duplicated code** (exists in 3 places in gns/train.py):
```python
position = example[0][0].to(device_id)
particle_type = example[0][1].to(device_id)
if n_features == 3:
    material_property = example[0][2].to(device_id)
    n_particles_per_example = example[0][3].to(device_id)
elif n_features == 2:
    material_property = None
    n_particles_per_example = example[0][2].to(device_id)
else:
    raise NotImplementedError
labels = example[1].to(device_id)
```

**After extraction**:
```python
batch = training.prepare_training_batch(example, n_features, device_id)
# batch = {position, particle_type, material_property, n_particles_per_example, labels}
```

#### 3. `update_learning_rate()` - Important
Extract learning rate exponential decay as a function instead of hardcoding in loop.

**Current hardcoding** (inside training loop):
```python
lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
for param in optimizer.param_groups:
    param['lr'] = lr_new
```

**After extraction**:
```python
lr_new = training.update_learning_rate(optimizer, step, lr_init, lr_decay, lr_decay_steps, world_size)
```

## Implementation Phases

### Phase 1: Create gns/training.py (400-450 lines)

**Extract from**: `gns/train.py` (663 lines)

**Function composition**:

1. **Utility functions** (existing)
   - `optimizer_to()` - Move optimizer to device
   - `acceleration_loss()` - Masked loss calculation

2. **Batch preparation** (new, important)
   - `prepare_training_batch()` - Reduce 15-line boilerplate

3. **Learning rate management** (new, important)
   - `update_learning_rate()` - Learning rate scheduling

4. **Noise addition** (new)
   - `add_training_noise()` - Add noise to position sequence

5. **Single-step operations** (existing)
   - `train_step()` - Single training step
   - `validation_step()` - Single validation step

6. **Checkpoint management** (existing, improved)
   - `save_checkpoint()` - Save model and training state
   - `load_checkpoint()` - Load model and training state
   - `find_latest_checkpoint()` - Find latest checkpoint

7. **Simulator creation** (new)
   - `create_simulator()` - Create simulator instance
   - `get_simulator()` - Get DDP-compatible simulator

8. **Training orchestration** (new, most important)
   - `run_training_loop()` - Manage entire training loop

### Phase 2: Create gns/rollout.py (250-300 lines)

**Extract from**: `predict()` function in `gns/train.py` (lines 90-172)

**Function composition**:

1. **Single rollout execution**
   - `run_rollout()` - Single trajectory prediction

2. **Rollout saving**
   - `save_rollout()` - Save prediction results

3. **Batch prediction**
   - `predict_rollouts()` - Predict entire dataset

4. **Distributed prediction** (new)
   - `predict_rollouts_distributed()` - Distributed prediction for DDP

5. **Data loading**
   - `get_rollout_dataloader()` - Dataloader for rollout
   - `infer_dataset_features()` - Infer number of features

### Phase 3: Create gns/render.py (300-350 lines)

**Extract from**: `gns/render_rollout.py` (246 lines)

**New module**: Implement visualization functionality as functions

**Function composition**:

1. **Data loading**
   - `load_rollout_pickle()` - Load rollout data

2. **Color mapping**
   - `TYPE_TO_COLOR` - Mapping from particle type to color
   - `get_color_mask()` - Generate color mask

3. **2D/3D rendering**
   - `render_2d_trajectory()` - Render 2D trajectory
   - `render_3d_trajectory()` - Render 3D trajectory

4. **GIF animation**
   - `render_gif_animation()` - Generate GIF (unified 2D/3D)

5. **VTK export**
   - `write_vtk_trajectory()` - Output VTK files

**Note**: The current `Render` class (246 lines) has a `render_gif_animation()` method of 107 lines that mixes 2D/3D. This will be separated.

### Phase 4: Create Thin CLI Scripts

#### scripts/gns_train.py (60-80 lines)

**Structure**:
```python
"""Single-GPU training script - thin CLI wrapper"""

# Flag definitions (30 lines)

def train_single_gpu(rank, flags_dict, world_size, device):
    # Load metadata
    metadata = reading_utils.read_metadata(flags_dict["data_path"], "train")
    simulator_config = config.SimulatorConfig.from_metadata(metadata)

    # Get simulator and optimizer
    simulator = training.get_simulator(...)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=...)

    # Load checkpoint (optional)
    if flags_dict["model_file"]:
        training.load_checkpoint(...)

    # Get dataloaders
    train_dl = training.get_training_dataloader(...)
    valid_dl = training.get_training_dataloader(..., split='valid') if ... else None

    # Run training loop (delegate all logic to training module)
    training.run_training_loop(
        simulator, train_dl, valid_dl, optimizer, device_id,
        rank, world_size, flags_dict, simulator_config,
        flags_dict["model_path"], is_distributed)

def main(_):
    # Device setup and mode dispatch
    if FLAGS.mode == 'train':
        train_single_gpu(...)
    elif FLAGS.mode in ['valid', 'rollout']:
        rollout.run_prediction_mode(...)
```

**Reduction**: 663 lines → 60-80 lines (~90% reduction)

#### scripts/gns_train_multinode.py (70-90 lines)

**Structure**: Almost same as `gns_train.py` but with distributed initialization

**Reduction**: 669 lines → 70-90 lines (~90% reduction)

#### scripts/gns_render_rollout.py (35-50 lines)

**Structure**:
```python
"""Render rollout predictions - thin CLI wrapper"""

def main(_):
    rollout_data = render.load_rollout_pickle(FLAGS.rollout_dir, FLAGS.rollout_name)

    if FLAGS.output_mode == "gif":
        render.render_gif_animation(rollout_data, ...)
    elif FLAGS.output_mode == "vtk":
        render.write_vtk_trajectory(rollout_data, ...)
```

**Reduction**: 246 lines → 35-50 lines (~85% reduction)

## Implementation Order

1. **gns/training.py** - Foundation
   - Utilities → Batch preparation → Learning rate management → Noise addition
   - Single steps → Checkpoints → Simulator creation
   - **Finally**: `run_training_loop()`

2. **gns/rollout.py** - Prediction
   - Data loading → Single rollout → Saving → Batch prediction → Distributed prediction

3. **gns/render.py** - Visualization
   - Data loading → Color mapping → 2D/3D rendering → GIF/VTK

4. **CLI scripts** - Wrappers
   - `scripts/gns_train.py` → `scripts/gns_train_multinode.py` → `scripts/gns_render_rollout.py`

## Validation Plan

### After Phase 1 (training.py)
```bash
# Import test
uv run python -c "from gns.training import prepare_training_batch, update_learning_rate; print('OK')"

# 10-step training test
uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --ntraining_steps=10 --mode=train

# Check checkpoint
ls models/test_step5_v2/
```

### After Phase 2 (rollout.py)
```bash
# Rollout execution test
uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step5_v2/

# Check rollout output
ls rollouts/test_step5_v2/
```

### After Phase 3 (render.py)
```bash
# GIF rendering test
uv run python scripts/gns_render_rollout.py \
  --rollout_dir=rollouts/test_step5_v2/ \
  --rollout_name=rollout_ex0 --output_mode=gif

# Check GIF generation
ls rollouts/test_step5_v2/*.gif
```

## Success Criteria

### Code Volume
- ✅ `scripts/gns_train.py`: 60-80 lines
- ✅ `scripts/gns_train_multinode.py`: 70-90 lines
- ✅ `scripts/gns_render_rollout.py`: 35-50 lines
- ✅ Total code reduction 20-30% (1,579 lines → 1,115-1,320 lines)

### Functionality
- ✅ 10-step training works correctly
- ✅ Checkpoint save/load works
- ✅ Validation works during training
- ✅ Rollout prediction works
- ✅ GIF/VTK rendering works
- ✅ Compatibility with existing checkpoints maintained
- ✅ Distributed training works

### Code Quality
- ✅ No code duplication
- ✅ All functions have docstrings
- ✅ Complete type hints
- ✅ Clear separation of responsibilities (CLI vs business logic)

## Summary of Differences from First Attempt

| Item | First Attempt | Second (This Design) |
|------|---------------|---------------------|
| scripts/gns_train.py | 489 lines | 60-80 lines |
| Training loop orchestration | None (remained in CLI) | `run_training_loop()` |
| Batch preparation | Duplicated in 3 places | `prepare_training_batch()` |
| Learning rate scheduling | Hardcoded in loop | `update_learning_rate()` |
| Simulator creation | Remained in CLI | `create_simulator()`, `get_simulator()` |
| render.py | None | Newly created (separated 246 lines) |

## References

- **First implementation**: Branch `refactor/step5_1st_try` (commit ac7d7b7)
- **Current Step 4 state**: Commit eee4a66
- **Related decision records**:
  - `0001-keep-learned-simulator-as-nn-module.md`
  - `0002-type-annotation-fixes.md`
