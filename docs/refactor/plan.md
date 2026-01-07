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

## Implementation Steps (9 Stages)

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

3. **Create new script**: `scripts/setup/setup_uv_environment.sh`
   - Install uv
   - Create Python 3.13 environment
   - Install dependencies
   - Example:
     ```bash
     #!/bin/bash
     # Install uv if not already installed
     if ! command -v uv &> /dev/null; then
         curl -LsSf https://astral.sh/uv/install.sh | sh
     fi

     # Create virtual environment with Python 3.13
     uv venv --python 3.13

     # Activate and install dependencies
     source .venv/bin/activate
     uv pip install -e .
     ```

4. **Preserve existing files**: `requirements.txt`, `enviornment.yml`
   - Keep for backward compatibility
   - Add deprecation comment at top: "DEPRECATED: Use pyproject.toml instead"

5. **Update `.gitignore`**:
   - Add `.venv/` (uv virtual environment)
   - Add `uv.lock` (dependency lock file)

**Resolves**:
- pyright-lsp works properly, enabling type checking and code completion during refactoring
- Benefits from bug fixes and performance improvements in latest package versions
- Proactively solves Issue 1 (environment definition unification)

**Impact**:
- New files: `pyproject.toml`, `.python-version`, `scripts/setup/setup_uv_environment.sh`
- Modified files: `requirements.txt`, `enviornment.yml` (add deprecation comment)
- `.gitignore`

**Validation**:
```bash
# Setup uv environment
bash scripts/setup/setup_uv_environment.sh

# Activate virtual environment
source .venv/bin/activate

# Check dependencies
uv pip list

# Quick functionality test
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print('OK')"

# Verify existing tests pass
pytest test/test_pytorch.py
pytest test/test_torch_geometric.py

# Run small-scale training (same command used in Step 1 onwards)
python -m gns.train --data_path=example/WaterDropSample/ \
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

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "autopep8>=2.0.0",
]

[build-system]
requires = ["setuptools>=70.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["gns"]
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

1. **Modify existing file**: `gns/learned_simulator.py`
   - Rename `LearnedSimulator` to `GraphNeuralNetworkModel`
   - Responsibilities: Graph construction, message passing, forward pass
   - Remove: Normalization statistics, boundary conditions (move to PhysicsSimulator)

2. **Create new file**: `gns/simulator.py`
   - `PhysicsSimulator` class
   - Wraps `GraphNeuralNetworkModel`
   - Responsibilities: Normalization, Euler integration, position/acceleration prediction
   - Methods: `predict_positions()`, `predict_accelerations()`, `save()`, `load()`

3. **Modify existing files**: `gns/train.py`, `gns/train_multinode.py`
   - `from gns.learned_simulator import LearnedSimulator`
     → `from gns.simulator import PhysicsSimulator`

**Resolves**: Issue 4 - Users wanting just the GNN can use `GraphNeuralNetworkModel`, users needing physics predictions use `PhysicsSimulator`

**Impact**:
- Entire `gns/learned_simulator.py` (388 lines)
- Import statements and `_get_simulator()` function in `gns/train.py`, `gns/train_multinode.py`

**Validation**:
```bash
# Validate with same training/rollout commands as Step 1
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# Verify module imports
python -c "from gns.simulator import PhysicsSimulator; \
from gns.learned_simulator import GraphNeuralNetworkModel; \
print('Import successful')"
```

**Scope**: Large (split 388-line class, modify many import statements)

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

**Purpose**: Clearly separate importable modules from execution scripts.

**Changes**:

1. **Create new directory**: `scripts/`

2. **Create new files**: `gns/training.py`, `gns/rollout.py`
   - `gns/training.py` - Reusable training logic
     - `train_step()` - Single training step
     - `validation_step()` - Single validation step
     - `save_checkpoint()`, `load_checkpoint()` - Checkpoint management
     - `prepare_training_batch()` - Batch preparation (implemented in Step 8)
   - `gns/rollout.py` - Reusable rollout logic
     - `run_rollout()` - Rollout execution
     - `save_rollout()` - Save results

3. **Move files**:
   - `gns/train.py` → `scripts/train.py` (CLI only, logic → `gns/training.py`)
   - `gns/train_multinode.py` → `scripts/train_multinode.py` (same)
   - `gns/render_rollout.py` → `scripts/render_rollout.py`

4. **Modify existing file**: `scripts/train.py`
   - Keep only FLAGS parsing and CLI entry point
   - Call functions from `gns.training` for actual logic

**Resolves**: Issue 3 - `gns/` = importable modules, `scripts/` = executables

**Impact**:
- Entire `gns/train.py` (split 658 lines)
- Entire `gns/train_multinode.py`
- `gns/render_rollout.py`

**Validation**:
```bash
# Execute from new script location
python scripts/train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# Verify module imports
python -c "from gns.training import train_step; \
from gns.rollout import run_rollout; print('Import successful')"
```

**Scope**: Medium (split large files, but mainly moving)

---

### Step 6: Organize Root Directory Shell Scripts

**Purpose**: Organize shell scripts by purpose with clear naming.

**Changes**:

1. **Create new directories**:
   - `scripts/setup/` - For environment setup
   - `scripts/examples/` - For execution examples

2. **Move and rename files**:
   ```
   build_venv.sh → scripts/setup/create_environment.sh
   build_venv_frontera.sh → scripts/setup/create_environment_frontera.sh
   module.sh → scripts/setup/load_modules.sh
   start_venv.sh → scripts/setup/activate_environment.sh
   run.sh → scripts/examples/train_water_drop.sh
   ```

3. **Create new file**: `scripts/README.md`
   - Explain purpose and usage of each script

**Resolves**: Issue 2 - Script purposes are clear and organized

**Impact**:
- 5 shell script files in root directory

**Validation**:
```bash
# Test environment setup script
bash scripts/setup/create_environment.sh
source scripts/setup/activate_environment.sh

# Test example execution script
bash scripts/examples/train_water_drop.sh
```

**Scope**: Small (file moves and documentation creation)

---

### Phase 3: Environment & Configuration Unification (Steps 7-8)

Unify environment definitions and organize configuration handling.

---

### Step 7: Unify Environment Definition Files

**Purpose**: Centralize dependency definitions to eliminate inconsistencies.

**Changes**:

1. **Modify existing file**: `environment.yml`
   - Fix filename typo (`enviornment.yml` → `environment.yml`)
   - Add comment at top: "Use requirements.txt as the source of truth"

2. **Maintain existing file**: `requirements.txt`
   - Maintain as single source of truth for dependencies
   - Already used by CI (`.circleci/config.yml`)

3. **Create new file**: `scripts/setup/create_conda_environment.sh`
   - Script to create conda environment using `requirements.txt`

4. **Update documentation**: `README.md` (if exists)
   - Specify which environment definition file to use

**Resolves**: Issue 1 - Environment definition unification

**Impact**:
- `environment.yml` (typo fix and comment addition)
- New script in `scripts/setup/`

**Validation**:
```bash
# Test pip installation
pip install -r requirements.txt
python -c "import torch; import torch_geometric; print('OK')"

# Test conda environment creation
bash scripts/setup/create_conda_environment.sh
conda activate gns
python -c "import torch; import torch_geometric; print('OK')"
```

**Scope**: Small (mainly documentation and script addition)

---

### Step 8: Functionalize Training Loop Feature Extraction

**Purpose**: Make training batch preparation logic reusable.

**Changes**:

1. **Modify existing file**: `gns/training.py` (created in Step 5)
   - Add `prepare_training_batch()` function
   - Responsibilities: Process examples from dataloader
     - Feature extraction (position, particle_type, material_property)
     - Noise generation
     - Kinematic particle masking
   - Extract inline processing currently in `train.py` lines 387-419

2. **Modify existing files**: `scripts/train.py`, `scripts/train_multinode.py`
   - Call `prepare_training_batch()` in training loop

**Resolves**: Issue 4 - Training batch preparation logic is reusable

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

Phase 3: Environment & Configuration Unification (mostly done in Step 0)
  Step 7 (environment unification) - Most work done in Step 0, remaining tasks only
    ↓
  Step 8 (batch prep functionalization) - Depends on Step 5
```

**Important**: Step 0 (uv environment setup) must be done first. This enables:
- pyright-lsp works properly during refactoring
- Type checking and code completion available
- Validation with latest packages possible

## Final Directory Structure

```
/Users/masahiro/repos/gns/wt-cleanup/
├── gns/                          # Reusable modules only
│   ├── config.py                 # NEW - Step 3
│   ├── inference_utils.py        # NEW - Step 1
│   ├── simulator.py              # NEW - Step 2
│   ├── learned_simulator.py      # MODIFIED - Step 2
│   ├── training.py               # NEW - Step 5, 8
│   ├── rollout.py                # NEW - Step 5
│   ├── graph_network.py          # Existing (no changes)
│   ├── data_loader.py            # Existing (no changes)
│   ├── reading_utils.py          # Existing (no changes)
│   └── noise_utils.py            # Existing (no changes)
│
├── scripts/                      # Execution scripts
│   ├── train.py                  # MOVED - Step 5
│   ├── train_multinode.py        # MOVED - Step 5
│   ├── render_rollout.py         # MOVED - Step 5
│   ├── setup/                    # NEW - Step 0, 6
│   │   ├── setup_uv_environment.sh      # NEW - Step 0
│   │   ├── create_environment.sh        # MOVED - Step 6
│   │   ├── create_environment_frontera.sh # MOVED - Step 6
│   │   ├── load_modules.sh             # MOVED - Step 6
│   │   └── activate_environment.sh     # MOVED - Step 6
│   ├── examples/                 # NEW - Step 6
│   │   └── train_water_drop.sh
│   └── README.md                 # NEW - Step 6
│
├── pyproject.toml                # NEW - Step 0 (source of truth for dependencies)
├── .python-version               # NEW - Step 0
├── .venv/                        # NEW - Step 0 (add to gitignore)
├── uv.lock                       # NEW - Step 0 (add to gitignore)
├── requirements.txt              # DEPRECATED - Step 0 (kept for compatibility)
├── enviornment.yml               # DEPRECATED - Step 0 (kept for compatibility)
├── slurm_scripts/                # Existing (no changes)
├── test/                         # Existing (no changes)
└── ...
```

## Key Files to Change

### Required Changes

1. **gns/train.py** (658 lines)
   - Changed incrementally in Steps 1, 3, 4, 5
   - Finally moved to `scripts/train.py`

2. **gns/learned_simulator.py** (388 lines)
   - Renamed and split to `GraphNeuralNetworkModel` in Step 2

3. **gns/train_multinode.py** (~25KB)
   - Requires same changes as `gns/train.py`
   - Finally moved to `scripts/train_multinode.py`

4. **requirements.txt**
   - Positioned as source of truth for dependencies in Step 7

5. **gns/reading_utils.py**
   - Used by `config.py` in Steps 3, 4 (no changes needed)

### New Files to Create

1. **gns/inference_utils.py** - Step 1
2. **gns/simulator.py** - Step 2
3. **gns/config.py** - Step 3
4. **gns/training.py** - Step 5
5. **gns/rollout.py** - Step 5
6. **scripts/README.md** - Step 6
7. **scripts/setup/create_conda_environment.sh** - Step 7

## Validation Strategy

After each step completion:

1. **Run small-scale training** (10 steps)
   ```bash
   python scripts/train.py --data_path=example/WaterDropSample/ \
     --model_path=models/test/ --ntraining_steps=10 --mode=train
   ```

2. **Run rollout**
   ```bash
   python scripts/train.py --data_path=example/WaterDropSample/ \
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

**Mitigation**: Don't change `slurm_scripts/`, maintain same interface in `scripts/train.py` as original `gns/train.py`

### Issue 3: Insufficient Test Code

**Mitigation**: Validate through training/inference execution on actual datasets

## Completion Criteria

After completing all 9 steps, the following should be achieved:

1. ✅ **Environment definitions unified in `pyproject.toml`** (Step 0)
   - Latest Python 3.13 environment managed by uv
   - pyright-lsp works properly, type checking and code completion available

2. ✅ **Shell scripts organized in `scripts/setup/`, `scripts/examples/`** (Step 6)
   - Organized by purpose with clear naming

3. ✅ **`gns/` package contains reusable modules only** (Step 5)
   - Execution scripts moved to `scripts/`

4. ✅ **Inference, kinematic constraints, training batch prep extractable as independent functions** (Steps 1, 8)
   - `inference_utils.py` provides inference logic
   - `training.py` provides training logic

5. ✅ **Existing training/inference works correctly (numerical results match)** (All steps)
   - Validated with latest package versions

At this state, users can:
- ✅ Import and use only needed features
- ✅ Clearly distinguish modules from scripts
- ✅ Follow clear environment setup procedure (`bash scripts/setup/setup_uv_environment.sh` completes it)
- ✅ Understand script purposes clearly
- ✅ Benefit from latest Python ecosystem (Python 3.13, PyTorch 2.6+, numpy 2.1+)
- ✅ Work efficiently with type checking and completion during refactoring
- ✅ Leverage Python 3.13's JIT compiler for performance improvements
