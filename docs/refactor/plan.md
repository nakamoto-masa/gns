# GNS Refactoring Implementation Plan

## Objective

Perform minimal refactoring to make the model easier to use for end users.

## Scope

- **In Scope**: Code quality issues in the GNS model
- **Out of Scope**: MeshNet, design changes for maintainability

## User-Facing Issues (Problems to Solve)

1. **Environment definitions are inconsistent across multiple files**
   - Inconsistency between `requirements.txt` and `environment.yml`

2. **Shell scripts with unclear purpose scattered in repository root**
   - `run.sh`, `build_venv.sh`, `start_venv.sh`, `module.sh`, etc.

3. **Ambiguous distinction between modules and scripts**
   - `gns/train.py`, `gns/train_multinode.py` are scripts but placed inside the package

4. **Difficult to extract and use specific features**
   - `rollout()` function mixes inference, kinematic constraints, and loss computation
   - `LearnedSimulator` class mixes NN model, physics integration, and normalization
   - Dataset-specific parameters hardcoded as global constants

## Refactoring Constraints

- Do not change algorithm behavior
- Make changes incrementally, maintaining functionality at each stage
- Prioritize separation of responsibilities
- Clear distinction between modules and scripts

## Implementation Steps (7 Stages)

### Phase 0: Environment Setup (Step 0)

Set up a modern Python environment first to enable efficient refactoring work.

---

### Step 0: Python Environment Setup with uv and Latest Dependencies

**Purpose**: Enable pyright-lsp for type checking and code completion to improve refactoring efficiency.

**Changes**:

1. **Create new file**: `pyproject.toml`
   - Define project metadata and dependencies
   - Extract dependencies from existing `requirements.txt` and `enviornment.yml`
   - Update package versions to latest stable releases

   Key dependencies:
   - Python 3.13 (latest stable, upgrade from 3.9)
     - Rationale: Released October 2024, all major libraries now support it
   - PyTorch 2.6+ (latest stable, upgrade from 1.12)
     - PyTorch 2.6 introduced Python 3.13 support
   - torch-geometric 2.6+ (PyTorch 2.6+ compatible version)
     - Supports Python 3.10-3.13
   - numpy 2.1+ (remove version pinning, upgrade to latest)
     - numpy 2.1.0+ supports Python 3.13
   - Others: absl-py, matplotlib, pytest, tqdm, etc.

2. **Create new file**: `.python-version`
   - Specify Python version for uv (3.13)

3. **Delete existing files**: `requirements.txt`, `enviornment.yml`
   - Consolidate into `pyproject.toml`

**Resolves**:
- pyright-lsp works properly, enabling type checking and code completion during refactoring
- Benefits from bug fixes and performance improvements in latest package versions
- Proactively solves Issue 1 (environment definition unification)

**Impact**:
- New files: `pyproject.toml`, `.python-version`
- Deleted files: `requirements.txt`, `enviornment.yml`

**Validation**:
```bash
# Setup uv environment (uv sync automatically creates venv and installs dependencies)
uv sync

# Check dependencies
uv pip list

# Quick functionality test
uv run python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print('OK')"

# Verify existing tests pass
uv run pytest test/test_pytorch.py
uv run pytest test/test_torch_geometric.py

# Run small-scale training (same command used in Step 1 onwards)
uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

**Example pyproject.toml**:
```toml
[project]
name = "gns"
version = "0.1.0"
description = "Graph Network-based Simulator"
requires-python = ">=3.13"
dependencies = [
    "torch>=2.6.0",
    "torch-geometric>=2.6.0",
    "numpy>=2.1.0",
    "absl-py>=2.0.0",
    "matplotlib>=3.9.0",
    "pyevtk>=1.6.0",
    "dm-tree>=0.1.8",
    "tqdm>=4.66.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
]
```

**Scope**: Small (environment setup only, no code changes)

**Notes**:
- PyTorch 2.x may have breaking API changes, so validation is critical
- torch-geometric version depends on PyTorch version, verify compatibility
- numpy 2.x has some API changes, modify code if necessary
- Python 3.13 introduces JIT compiler, offering potential performance improvements

---

### Phase 1: Architecture Separation (Steps 1-4)

Separate responsibilities of `rollout()` and `LearnedSimulator` to enable feature extraction.

---

### Step 1: Separate rollout() Function

**Purpose**: Separate inference loop, kinematic constraint application, and loss computation to make each feature independently usable.

**Changes**:

1. **Create new file**: `gns/inference_utils.py`
   - `run_inference_loop()` - Pure inference execution (simulator calls only)
   - `apply_kinematic_constraints()` - Apply kinematic constraints (override KINEMATIC particle positions with ground truth)

2. **Modify existing file**: `gns/train.py`
   - Separate `rollout()` function into 3 function calls
     - `inference_utils.run_inference_loop()` for inference execution
     - `inference_utils.apply_kinematic_constraints()` for constraint application
     - Local loss computation (`compute_rollout_loss()` can be extracted in future)

**Resolves**: Issue 4 - Inference loop can be extracted and used independently

**Impact**:
- `rollout()` function in `gns/train.py` (lines 56-119)
- Equivalent section in `gns/train_multinode.py` (requires same changes)

**Validation**:
```bash
# Run 10-step training
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# Run rollout
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test/

# Verify output files are generated
ls rollouts/test/
```

**Scope**: Medium (core functionality, but extraction is relatively clear)

---

### Step 2: Separate LearnedSimulator Class

**Purpose**: Separate NN model core from physics integration, enabling usage based on needs.

**Changes**:

1. **Create new file**: `gns/graph_model.py`
   - Create new `GraphNeuralNetworkModel` class
   - Responsibilities: Graph construction, message passing, forward pass
   - Provides pure GNN model only

2. **Modify existing file**: `gns/learned_simulator.py`
   - Keep `LearnedSimulator` class name as-is
   - Refactor to wrap `GraphNeuralNetworkModel`
   - Responsibilities: Normalization, Euler integration, position/acceleration prediction
   - Methods: `predict_positions()`, `predict_accelerations()`, `save()`, `load()`

3. **Modify existing files**: `gns/train.py`, `gns/train_multinode.py`
   - `from gns.learned_simulator import LearnedSimulator` remains unchanged (no modification needed)

**Resolves**: Issue 4 - Users wanting just the GNN can use `GraphNeuralNetworkModel`, users needing physics predictions use `LearnedSimulator`

**Impact**:
- Entire `gns/learned_simulator.py` (388 lines)
- New file: `gns/graph_model.py`

**Validation**:
```bash
# Validate with same training/rollout commands as Step 1
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# Verify module imports
python -c "from gns.learned_simulator import LearnedSimulator; \
from gns.graph_model import GraphNeuralNetworkModel; \
print('Import successful')"
```

**Scope**: Large (split 388-line class, minimal impact on existing code)

---

### Step 3: Convert Global Constants to Configuration Class

**Purpose**: Make dataset-specific parameters configurable instead of hardcoded.

**Changes**:

1. **Create new file**: `gns/config.py`
   - `SimulatorConfig` dataclass
   - Fields: `input_sequence_length`, `num_particle_types`, `kinematic_particle_id`, etc.
   - `from_metadata()` class method to create from metadata

2. **Modify existing files**: `gns/train.py`, `gns/train_multinode.py`
   - Remove global constants (lines 52-54)
     - `INPUT_SEQUENCE_LENGTH = 6` → `config.input_sequence_length`
     - `NUM_PARTICLE_TYPES = 9` → `config.num_particle_types`
     - `KINEMATIC_PARTICLE_ID = 3` → `config.kinematic_particle_id`
   - Each function receives `config` as argument

**Resolves**: Issue 4 - Can handle datasets with different parameters

**Impact**:
- All constant references in `gns/train.py`, `gns/train_multinode.py`

**Validation**:
```bash
# Verify config object creation
python -c "from gns.config import SimulatorConfig; \
cfg = SimulatorConfig(); print(cfg)"

# Validate with training execution
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

**Scope**: Small (clear extraction, low risk)

---

### Step 4: Separate Configuration Loading from Simulator Creation

**Purpose**: Separate configuration interpretation from object instantiation, making custom simulator creation easier.

**Changes**:

1. **Modify existing file**: `gns/config.py`
   - Add `build_normalization_stats()` function - metadata interpretation
   - Add `infer_feature_dimensions()` function - dimension inference

2. **Modify existing file**: `gns/train.py`
   - Split `_get_simulator()` function (lines 503-558)
     - Configuration loading → functions in `config.py`
     - Simulator creation → `create_simulator()` function

**Resolves**: Issue 4 - Can create simulators with custom configurations

**Impact**:
- `_get_simulator()` function in `gns/train.py`

**Validation**: Same as Step 3

**Scope**: Small (refactoring existing function)

---

### Phase 2: Module/Script Clarification (Steps 5-6)

Limit `gns/` package to reusable modules only, separate execution scripts.

---

### Step 5: Move Execution Scripts Outside gns/ Package

**Purpose**: Clearly separate importable modules from execution scripts. Make scripts thin CLI wrappers (50-100 lines).

**Note**: See `docs/refactor/decisions/0003-step5-redesign-detailed-plan.md` for detailed implementation plan.

**Changes**:

1. **Create new directory**: `scripts/`

2. **Create new files**: `gns/training.py`, `gns/rollout.py`, `gns/render.py`
   - `gns/training.py` (400-450 lines) - Reusable training logic
     - Low-level: `train_step()`, `validation_step()`, `save_checkpoint()`, `load_checkpoint()`
     - **Important**: `run_training_loop()` - Orchestrates entire training loop (most critical function missing in first attempt)
     - **Important**: `prepare_training_batch()` - Reduces batch preparation boilerplate
     - **Important**: `update_learning_rate()` - Learning rate scheduling
     - Simulator creation: `create_simulator()`, `get_simulator()`
   - `gns/rollout.py` (250-300 lines) - Reusable rollout logic
     - `run_rollout()` - Rollout execution
     - `save_rollout()` - Save results
     - `predict_rollouts()`, `predict_rollouts_distributed()` - Batch prediction
   - `gns/render.py` (300-350 lines) - Reusable rendering logic
     - `load_rollout_pickle()` - Data loading
     - `render_2d_trajectory()`, `render_3d_trajectory()` - Rendering
     - `render_gif_animation()`, `write_vtk_trajectory()` - Output

3. **Move files**:
   - `gns/train.py` (663 lines) → `scripts/gns_train.py` (60-80 lines)
   - `gns/train_multinode.py` (669 lines) → `scripts/gns_train_multinode.py` (70-90 lines)
   - `gns/render_rollout.py` (246 lines) → `scripts/gns_render_rollout.py` (35-50 lines)

4. **Modify existing file**: `scripts/gns_train.py`
   - Keep only FLAGS parsing and CLI entry point (~60-80 lines)
   - Training loop fully delegates to `training.run_training_loop()`
   - Batch preparation delegates to `training.prepare_training_batch()`

**Resolves**: Issue 3 - `gns/` = importable modules, `scripts/` = executables

**Impact**:
- Entire `gns/train.py` (split 658 lines)
- Entire `gns/train_multinode.py`
- `gns/render_rollout.py`

**Validation**:
```bash
# Execute from new script location
python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# Verify module imports
python -c "from gns.training import train_step; \
from gns.rollout import run_rollout; print('Import successful')"
```

**Scope**: Medium (split large files, but mainly moving)

---

### Step 6: Archive Root Directory Shell Scripts

**Purpose**: Clean up root directory by archiving unnecessary shell scripts.

**Changes**:

1. **Create new directory**:
   - `scripts/legacy/` - For archived legacy scripts

2. **Move files**:
   - Move the following scripts to `scripts/legacy/`
   ```
   build_venv.sh → scripts/legacy/build_venv.sh
   build_venv_frontera.sh → scripts/legacy/build_venv_frontera.sh
   module.sh → scripts/legacy/module.sh
   start_venv.sh → scripts/legacy/start_venv.sh
   run.sh → scripts/legacy/run.sh
   ```

3. **Create new file**: `scripts/README.md`
   - Explain new scripts (`gns_train.py`, etc.)
   - Note that `legacy/` folder contains deprecated scripts for old environment

**Resolves**: Issue 2 - Root directory is clean and important files are clearly identified

**Impact**:
- 5 shell script files in root directory

**Validation**:
```bash
# Verify files were moved
ls scripts/legacy/

# Verify root directory is clean
ls *.sh 2>/dev/null || echo "No shell scripts in root directory (correct)"
```

**Scope**: Small (file moves and documentation creation)

---

### Phase 3: Configuration Organization (Step 7)

Organize training batch preparation processing.

---

### Step 7: Functionalize Training Loop Feature Extraction

**Note**: The `prepare_training_batch()` function planned for this step was **already implemented in Step 5**. Step 5's redesign integrated batch preparation processing into Step 5.

**Purpose**: (Achieved in Step 5) Make training batch preparation logic reusable.

**Implemented in Step 5**:

1. Added the following functions to **`gns/training.py`**:
   - `prepare_training_batch()` - Batch preparation (feature extraction and device transfer)
   - `add_training_noise()` - Noise generation and kinematic particle masking
   - Original code: Extracted inline processing from lines 387-419 in `train.py`

2. Automatically called within `run_training_loop()` in **`scripts/gns_train.py`**

**Resolves**: Issue 4 - Training batch preparation logic is reusable (achieved in Step 5)

**Impact**:
- Training loop in `gns/train.py` (lines 387-419)
- Equivalent section in `gns/train_multinode.py`

**Validation**: Same training execution as Step 5

**Scope**: Small (clear extraction)

---

## Implementation Order and Dependencies

```
Phase 0: Environment Setup
  Step 0 (uv environment) - Highest priority, prerequisite for other steps
    ↓

Phase 1: Architecture Separation
  Step 1 (rollout separation) - After Step 0
    ↓
  Step 2 (LearnedSimulator separation) - Depends on Step 1
    ↓
  Step 3 (global constants) - After Step 0 (independent of Steps 1-2)
    ↓
  Step 4 (config loading separation) - Depends on Step 3

Phase 2: Module/Script Clarification
  Step 5 (script move) - Depends on Steps 1-4
    ↓
  Step 6 (shell script organization) - After Step 0 (can run parallel to Step 5)

Phase 3: Configuration Organization
  Step 7 (batch prep functionalization) - Depends on Step 5
```

**Important**: Step 0 (uv environment setup) must be done first. This enables:
- pyright-lsp works properly during refactoring
- Type checking and code completion available
- Validation with latest packages possible
- Issue 1 (environment definition unification) completely resolved

## Final Directory Structure

```
/Users/masahiro/repos/gns/wt-cleanup/
├── gns/                          # Reusable modules only
│   ├── config.py                 # NEW - Step 3
│   ├── inference_utils.py        # NEW - Step 1
│   ├── graph_model.py            # NEW - Step 2
│   ├── learned_simulator.py      # MODIFIED - Step 2
│   ├── training.py               # NEW - Step 5, 7
│   ├── rollout.py                # NEW - Step 5
│   ├── render.py                 # NEW - Step 5
│   ├── graph_network.py          # Existing (no changes)
│   ├── data_loader.py            # Existing (no changes)
│   ├── reading_utils.py          # Existing (no changes)
│   └── noise_utils.py            # Existing (no changes)
│
├── scripts/                      # Execution scripts
│   ├── gns_train.py              # MOVED - Step 5
│   ├── gns_train_multinode.py    # MOVED - Step 5
│   ├── gns_render_rollout.py     # MOVED - Step 5
│   ├── README.md                 # NEW - Step 6 (script descriptions)
│   └── legacy/                   # MOVED - Step 6 (archived legacy scripts)
│       ├── build_venv.sh
│       ├── build_venv_frontera.sh
│       ├── module.sh
│       ├── start_venv.sh
│       └── run.sh
│
├── pyproject.toml                # NEW - Step 0 (source of truth for dependencies)
├── .python-version               # NEW - Step 0
├── slurm_scripts/                # Existing (no changes)
├── test/                         # Existing (no changes)
└── ...
```

## Key Files to Change

### Required Changes

1. **gns/train.py** (658 lines)
   - Changed incrementally in Steps 1, 3, 4, 5
   - Finally moved to `scripts/gns_train.py`

2. **gns/learned_simulator.py** (388 lines)
   - Refactored internally in Step 2 to wrap `GraphNeuralNetworkModel`

3. **gns/train_multinode.py** (~25KB)
   - Requires same changes as `gns/train.py`
   - Finally moved to `scripts/gns_train_multinode.py`

4. **gns/reading_utils.py**
   - Used by `config.py` in Steps 3, 4 (no changes needed)

### New Files to Create

1. **gns/inference_utils.py** - Step 1
2. **gns/graph_model.py** - Step 2
3. **gns/config.py** - Step 3
4. **gns/training.py** - Step 5
5. **gns/rollout.py** - Step 5
6. **gns/render.py** - Step 5
7. **scripts/README.md** - Step 6

## Validation Strategy

After each step completion:

1. **Run small-scale training** (10 steps)
   ```bash
   python scripts/gns_train.py --data_path=example/WaterDropSample/ \
     --model_path=models/test/ --ntraining_steps=10 --mode=train
   ```

2. **Run rollout**
   ```bash
   python scripts/gns_train.py --data_path=example/WaterDropSample/ \
     --model_path=models/test/ --model_file=model-10.pt \
     --mode=rollout --output_path=rollouts/test/
   ```

3. **Verify output**
   - Rollout pickle files are generated
   - Compare numerical results with baseline (pre-refactoring)

4. **Verify module imports**
   - Modules added in each step are importable

## Risk Management

- **Branch Strategy**: Continue on current `refactor/gns-cleanup` branch
- **Step Independence**: Each step can be committed individually
- **Rollback**: Revert commit if validation fails
- **Numerical Verification**: Verify inference results match existing implementation (torch.allclose)

## Expected Issues and Mitigations

### Issue 1: Duplicate Code in train.py and train_multinode.py

**Mitigation**: Apply same changes to both files, create foundation for future integration (integration not in current scope)

### Issue 2: Compatibility with Existing SLURM Scripts

**Mitigation**: Don't change `slurm_scripts/`, maintain same interface in `scripts/gns_train.py` as original `gns/train.py`

### Issue 3: Insufficient Test Code

**Mitigation**: Validate through training/inference execution on actual datasets

## Completion Criteria

After completing all 7 steps, the following should be achieved:

1. ✅ **Environment definitions unified in `pyproject.toml`** (Step 0)
   - Latest Python 3.13 environment managed by uv
   - pyright-lsp works properly, type checking and code completion available
   - Deleted old environment definition files (`requirements.txt`, `enviornment.yml`)

2. ✅ **Shell scripts archived in `scripts/legacy/`** (Step 6)
   - Old scripts moved to legacy folder
   - Clear documentation in scripts/README.md

3. ✅ **`gns/` package contains reusable modules only** (Step 5)
   - Execution scripts moved to `scripts/`

4. ✅ **Inference, kinematic constraints, training batch prep extractable as independent functions** (Steps 1, 7)
   - `inference_utils.py` provides inference logic
   - `training.py` provides training logic

5. ✅ **Existing training/inference works correctly (numerical results match)** (All steps)
   - Validated with latest package versions

At this state, users can:
- ✅ Import and use only needed features
- ✅ Clearly distinguish modules from scripts
- ✅ Follow clear environment setup procedure (`uv sync` completes it)
- ✅ Understand script purposes clearly
- ✅ Benefit from latest Python ecosystem (Python 3.13, PyTorch 2.6+, numpy 2.1+)
- ✅ Work efficiently with type checking and completion during refactoring
- ✅ Leverage Python 3.13's JIT compiler for performance improvements
