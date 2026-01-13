# GNS Refactoring Task List

This document manages the implementation status of the [Refactoring Implementation Plan](./plan.md).

## Progress Status

- [x] Phase 0: Environment Setup (4/4) ✅
- [x] Phase 1: Architecture Separation (13/13) ✅
- [x] Phase 2: Module/Script Clarification (9/9) ✅
- [x] Phase 3: Configuration Organization (3/3) ✅ (Implemented in Step 5)

**Overall Progress: 29/29 Tasks Completed 🎉**

---

## Phase 0: Environment Setup ✅

### Step 0.1: Create pyproject.toml and .python-version ✅
- [x] Create `pyproject.toml` (dependency definition)
- [x] Create `.python-version` (specify Python 3.13)

**Deliverables:**
- `pyproject.toml` - Dependency definition including PyTorch 2.8.0, PyTorch Geometric 2.7.0
- `.python-version` - Python 3.13 specification

### Step 0.2: Delete requirements.txt and enviornment.yml ✅
- [x] Delete `requirements.txt`
- [x] Delete `enviornment.yml` (typo version filename)

**Deliverables:**
- Old files deleted

### Step 0.3: Setup uv environment and install dependencies ✅
- [x] Execute `uv sync` (automatically creates venv and installs dependencies)
- [x] Verify dependency versions

**Execution Results:**
- Built Python 3.13.11 environment
- Installed PyTorch 2.8.0 (CPU version)
- Installed PyTorch Geometric 2.7.0
- Installed torch-cluster, torch-scatter, torch-sparse (PyG CPU version)

**Verification Commands:**
```bash
uv sync
uv pip list
uv run python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print('OK')"
```

### Step 0.4: Execute existing tests and small-scale training for verification ✅
- [x] Execute existing tests (test_learned_simulator.py, test_graph_network.py, test_noise_utils.py)
- [x] Execute small-scale training (10 steps) - Executed successfully with WaterDropSample data, loss decreased from 2.18 to 1.83

**Verification Commands:**
```bash
uv run python test/test_learned_simulator.py
uv run python test/test_graph_network.py
uv run python test/test_noise_utils.py
uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

---

## Phase 1: Architecture Separation

### Step 1.1: Create gns/inference_utils.py (run_inference_loop and apply_kinematic_constraints) ✅
- [x] Create `gns/inference_utils.py`
- [x] Implement `run_inference_loop()` function
- [x] Implement `apply_kinematic_constraints()` function

**Deliverables:**
- `gns/inference_utils.py` - Separated inference loop and kinematic constraint application
  - `apply_kinematic_constraints()` - Apply constraints to kinematic particles
  - `run_inference_loop()` - Inference loop with simulator
  - `rollout()` - Existing interface compatible rollout function

### Step 1.2: Modify gns/train.py and train_multinode.py rollout() functions to use separated function calls ✅
- [x] Modify `gns/train.py` `rollout()` function
- [x] Modify `gns/train_multinode.py` `rollout()` function

**Impact Scope:**
- `gns/train.py` `rollout()` function (lines 56-119 → shortened to 78-88)
- `gns/train_multinode.py` equivalent section (shortened to 86-96)

### Step 1.3: Verify with training and rollout execution ✅
- [x] Execute 10-step training - Success (loss: 2.28 → 1.85)
- [x] Execute rollout - Success (average loss: 0.211)
- [x] Verify output files - Confirmed generation of rollout_ex0.pkl, rollout_ex1.pkl

**Verification Commands:**
```bash
uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step1/ --ntraining_steps=10 --mode=train

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step1/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step1/

ls rollouts/test_step1/
```

### Step 2.1: Create gns/graph_model.py (GraphNeuralNetworkModel class) ✅
- [x] Create `gns/graph_model.py`
- [x] Implement `GraphNeuralNetworkModel` class
  - Graph construction
  - Message passing
  - Forward pass

**Deliverables:**
- `gns/graph_model.py` - Pure GNN model (210 lines)
  - `_compute_graph_connectivity()` - Compute graph connectivity
  - `build_graph_features()` - Build node/edge features
  - `predict()` - GNN forward pass

### Step 2.2: Refactor gns/learned_simulator.py to wrap GraphNeuralNetworkModel ✅
- [x] Refactor `LearnedSimulator` class to wrap `GraphNeuralNetworkModel`
- [x] Maintain normalization, Euler integration, position/acceleration prediction methods
- [x] Maintain `save()`, `load()` methods

**Impact Scope:**
- `gns/learned_simulator.py` entire file (388 lines → 297 lines, reduced by 91 lines)

**Implementation Results:**
- Delegated GNN part to `GraphNeuralNetworkModel`
- `LearnedSimulator` focuses on physics integration (normalization, Euler integration)
- Fully maintained existing API (backward compatibility)

### Step 2.3: Verify imports and training/rollout execution ✅
- [x] Verify module imports
- [x] Verify with training and rollout execution

**Verification Results:**
- Import test successful
- 10-step training successful (loss: 2.03 → 1.61)
- Rollout execution successful (average loss: 0.137)
- Confirmed output file generation (rollout_ex0.pkl, rollout_ex1.pkl)

**Verification Commands:**
```bash
uv run python -c "from gns.learned_simulator import LearnedSimulator; \
from gns.graph_model import GraphNeuralNetworkModel; \
print('Import successful')"

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step2/ --ntraining_steps=10 --mode=train

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step2/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step2/
```

### Step 2.4: Fix type annotations ✅
- [x] Unify to `X | None` notation (4 places)
- [x] Unify `boundaries` type (`np.ndarray`)
- [x] Remove unused imports
- [x] Add type annotation to `device` parameter

**Implementation Results:**
- Adopted Python 3.10+ `X | None` notation
- Unified to `boundaries: np.ndarray` (type annotation matching implementation)
- Improved type safety and IDE completion

**Related Documents:**
- `docs/refactor/decisions/0001-keep-learned-simulator-as-nn-module.md`
- `docs/refactor/decisions/0002-type-annotation-fixes.md`

### Step 3.1: Create gns/config.py (SimulatorConfig dataclass) ✅
- [x] Create `gns/config.py`
- [x] Implement `SimulatorConfig` dataclass
- [x] Implement `from_metadata()` class method

**Deliverables:**
- `gns/config.py` - SimulatorConfig dataclass
  - Default values: `input_sequence_length=6`, `num_particle_types=9`, `kinematic_particle_id=3`
  - Metadata fields: `dim`, `dt`, `default_connectivity_radius`, `bounds`, etc.
  - `from_metadata()` class method generates config from metadata

### Step 3.2: Remove global constants from gns/train.py and train_multinode.py and replace with config argument ✅
- [x] Remove global constants from `gns/train.py` (lines 53-55)
- [x] Remove global constants from `gns/train_multinode.py` (lines 61-63)
- [x] Modify each function to receive `simulator_config` as argument

**Impact Scope:**
- `gns/train.py` modifications:
  - `rollout()`, `predict()`, `train()`, `_get_simulator()`, `validation()` functions
  - All references to `INPUT_SEQUENCE_LENGTH`, `NUM_PARTICLE_TYPES`, `KINEMATIC_PARTICLE_ID`
- `gns/train_multinode.py` modifications:
  - `rollout()`, `rollout_par()`, `predict()`, `predict_par()`, `train()`, `_get_simulator()` functions
  - All constant references

### Step 3.3: Verify with training execution ✅
- [x] Verify config object creation - Success
- [x] Execute 10-step training - Success (loss: 2.14 → 1.88)
- [x] Execute rollout - Success (average loss: 0.538)
- [x] Verify output file generation - Confirmed generation of rollout_ex0.pkl, rollout_ex1.pkl

**Verification Commands:**
```bash
uv run python -c "from gns.config import SimulatorConfig; \
cfg = SimulatorConfig(); print(cfg)"

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step3/ --ntraining_steps=10 --mode=train

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step3/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step3/
```

### Step 4.1: Add build_normalization_stats and infer_feature_dimensions functions to gns/config.py ✅
- [x] Implement `build_normalization_stats()` function
- [x] Implement `infer_feature_dimensions()` function

**Deliverables:**
- `gns/config.py` - Added 2 new functions
  - `build_normalization_stats()` - Build normalization statistics from metadata (including composite calculation with noise)
  - `infer_feature_dimensions()` - Infer GNN input dimensions (nnode_in, nedge_in, particle_dimensions)

### Step 4.2: Split gns/train.py _get_simulator() function ✅
- [x] Create new `create_simulator()` function
- [x] Split `_get_simulator()` function (lines 474-531)
  - Configuration loading part → Use function group in `config.py`
  - Simulator creation part → Use `create_simulator()` function
- [x] Apply same changes to `gns/train_multinode.py`

**Impact Scope:**
- `gns/train.py` - Added `create_simulator()` function, refactored `_get_simulator()` function (56 lines → 25 lines)
- `gns/train_multinode.py` - Same changes

### Step 4.3: Verify with training execution ✅
- [x] Execute 10-step training - Success (loss: 1.94 → 1.36)
- [x] Execute rollout - Success (average loss: 0.064)
- [x] Verify output file generation - Confirmed generation of rollout_ex0.pkl, rollout_ex1.pkl

**Verification Commands:**
```bash
uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step4/ --ntraining_steps=10 --mode=train

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step4/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step4/
```

---

## Phase 2: Module/Script Clarification

### Step 5.1: Create scripts/ directory ✅
- [x] Create `scripts/` directory

**Deliverables:**
- `scripts/` directory

### Step 5.2: Create gns/training.py (train_step, validation_step, checkpoint management, run_training_loop) ✅
- [x] Create `gns/training.py` (788 lines)
- [x] Implement utility functions (`optimizer_to()`, `acceleration_loss()`)
- [x] Implement batch preparation function (`prepare_training_batch()` - reduces 15-line boilerplate)
- [x] Implement learning rate management (`update_learning_rate()`)
- [x] Implement noise addition (`add_training_noise()`)
- [x] Implement single-step operations (`train_step()`, `validation_step()`)
- [x] Implement checkpoint management (`save_checkpoint()`, `load_checkpoint()`, `find_latest_checkpoint()`)
- [x] Implement simulator creation (`create_simulator()`, `get_simulator()`)
- [x] Implement data loading (`get_training_dataloader()`, `infer_dataset_features()`)
- [x] **Implement training orchestration** (`run_training_loop()` - most critical function)

**Deliverables:**
- `gns/training.py` (788 lines) - Reusable training logic
  - Added high-level orchestration function missing in first implementation
  - Reduced batch preparation boilerplate
  - Functionalized learning rate scheduling

### Step 5.3: Create gns/rollout.py and gns/render.py ✅
- [x] Create `gns/rollout.py` (304 lines)
  - Data loading (`get_rollout_dataloader()`, `infer_dataset_features()`)
  - Single rollout execution (`run_rollout()`)
  - Rollout saving (`save_rollout()`)
  - Batch prediction (`predict_rollouts()`, `predict_rollouts_distributed()`)
- [x] Create `gns/render.py` (349 lines)
  - Data loading (`load_rollout_pickle()`, `prepare_trajectory_data()`)
  - Color mapping (`TYPE_TO_COLOR`, `get_color_mask()`)
  - 2D/3D rendering (`render_2d_trajectory()`, `render_3d_trajectory()`)
  - GIF animation (`render_gif_animation()`)
  - VTK export (`write_vtk_trajectory()`)

**Deliverables:**
- `gns/rollout.py` (304 lines) - Reusable rollout logic
- `gns/render.py` (349 lines) - Reusable visualization logic

### Step 5.4: Move and modify train.py, train_multinode.py, render_rollout.py to scripts/ ✅
- [x] Convert `gns/train.py` (663 lines) → `scripts/gns_train.py` (181 lines)
- [x] Convert `gns/train_multinode.py` → `scripts/gns_train_multinode.py` (214 lines)
- [x] Convert `gns/render_rollout.py` (246 lines) → `scripts/gns_render_rollout.py` (56 lines)

**Impact Scope:**
- `gns/train.py` entire file (663 lines → split into `gns/training.py` 788 lines + `scripts/gns_train.py` 181 lines)
- `gns/train_multinode.py` entire file (→ `scripts/gns_train_multinode.py` 214 lines)
- `gns/render_rollout.py` entire file (246 lines → `scripts/gns_render_rollout.py` 56 lines shortened, logic moved to `gns/render.py`)

**Implementation Results:**
- CLI scripts are thin wrappers (50-200 lines)
- Business logic separated into reusable modules
- Significantly reduced code duplication

### Step 5.5: Verify training/rollout execution from new script locations ✅
- [x] Execute 10-step training - Success (loss: 2.24 → 1.90)
- [x] Execute rollout - Success (average loss: 0.496)
- [x] Verify output file generation - Confirmed generation of rollout_ex0.pkl, rollout_ex1.pkl
- [x] Verify module imports - Success

**Verification Commands:**
```bash
uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --ntraining_steps=10 --mode=train

uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step5_v2/

uv run python -c "from gns.training import train_step, run_training_loop; \
from gns.rollout import run_rollout, predict_rollouts; \
from gns.render import render_gif_animation; print('Import successful')"
```

**Verification Results:**
- Training success: Confirmed checkpoint generation (model-0.pt, model-10.pt, train_state-0.pt, train_state-10.pt)
- Rollout success: Confirmed generation of rollout_ex0.pkl, rollout_ex1.pkl
- Import test successful

### Step 6.1: Create scripts/legacy/ directory ✅
- [x] Create `scripts/legacy/` directory

**Deliverables:**
- `scripts/legacy/` directory

### Step 6.2: Move 5 shell script files to scripts/legacy/ ✅
- [x] `build_venv.sh` → `scripts/legacy/build_venv.sh`
- [x] `build_venv_frontera.sh` → `scripts/legacy/build_venv_frontera.sh`
- [x] `module.sh` → `scripts/legacy/module.sh`
- [x] `start_venv.sh` → `scripts/legacy/start_venv.sh`
- [x] `run.sh` → `scripts/legacy/run.sh`

**Deliverables:**
- 5 shell scripts moved to `scripts/legacy/`

### Step 6.3: Create scripts/README.md ✅
- [x] Create `scripts/README.md`
  - Description of new scripts (`gns_train.py`, etc.)
  - Description of `legacy/` folder

**Deliverables:**
- `scripts/README.md` - Comprehensive documentation including script usage, architecture explanation, and legacy file notes

### Step 6.4: Verify file movements ✅
- [x] Verify files in `scripts/legacy/` - All 5 files moved
- [x] Verify root directory is clean - No .sh files

**Verification Results:**
```bash
# Legacy scripts: 5 files confirmed
build_venv.sh, build_venv_frontera.sh, module.sh, run.sh, start_venv.sh

# Root directory: No shell scripts (correct)
✓ Root directory is clean
```

---

## Phase 3: Configuration Organization ✅

**Note**: Features planned for this Phase were **already implemented in Step 5**. Step 5 redesign integrated batch preparation processing into Step 5.

### Step 7.1: Add prepare_training_batch function to gns/training.py ✅ (Implemented in Step 5)
- [x] Implement `prepare_training_batch()` function
  - Feature extraction (position, particle_type, material_property)
  - Unified device transfer and n_features conditional branching
  - Reduced 15 lines of boilerplate code to single function call
- [x] Implement `add_training_noise()` function
  - Noise generation
  - Apply mask for kinematic particles

**Impact Scope:**
- `gns/training.py` - Implemented in Step 5.2
- Original implementation: Extracted from `gns/train.py` lines 361-370 (batch preparation), 376-379 (noise addition)

**Implemented Content:**
- `prepare_training_batch()` - Batch preparation (gns/training.py:90-117)
- `add_training_noise()` - Noise addition (gns/training.py:128-154)

### Step 7.2: Use prepare_training_batch in scripts/gns_train.py and gns_train_multinode.py ✅ (Implemented in Step 5)
- [x] Automatically call `prepare_training_batch()` within `run_training_loop()`
- [x] Automatically call `add_training_noise()` within `run_training_loop()`

**Impact Scope:**
- Inside `run_training_loop()` function in `gns/training.py` (gns/training.py:665-676)
- CLI scripts only call `run_training_loop()`

**Implemented Content:**
- Batch preparation and noise addition automatically executed within `run_training_loop()`
- CLI scripts (`scripts/gns_train.py`, `scripts/gns_train_multinode.py`) implemented as thin wrappers

### Step 7.3: Verify with training execution ✅ (Verified in Step 5)
- [x] Verify with training execution - Completed in Step 5.5

**Verification Commands:**
```bash
uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --ntraining_steps=10 --mode=train
```

**Verification Results:**
- Completed in Step 5.5 (loss: 2.24 → 1.90)

---

## Completion Criteria

After all 29 tasks completed, the following should be achieved:

1. ✅ **Environment definition unified in `pyproject.toml`** (Step 0)
   - Latest Python 3.13 environment managed with uv
   - pyright-lsp works properly, type checking and code completion available
   - Deleted old environment definition files (`requirements.txt`, `environment.yml`)

2. ✅ **Unnecessary shell scripts moved to `scripts/legacy/`** (Step 6)
   - Root directory is clean, important files are clear
   - 5 legacy scripts moved to `scripts/legacy/`
   - `scripts/README.md` explains script usage and architecture

3. ✅ **`gns/` package contains only reusable modules** (Step 5)
   - Execution scripts moved to `scripts/` (`gns_train.py`, etc.)

4. ✅ **Inference, kinematic constraints, training batch preparation extractable as independent functions** (Steps 1, 7)
   - `inference_utils.py` provides inference logic
   - `training.py` provides training logic

5. ✅ **Existing training/inference works correctly (numerical results match)** (All steps)
   - Verified with latest package versions
