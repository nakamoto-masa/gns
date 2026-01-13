# Refactoring Equivalence Testing Plan

This document presents a test plan to verify that refactored code is functionally equivalent to the original code.

## Purpose

While refactoring changed the code structure, we need to confirm that the following behaviors are preserved:

1. **Checkpoint Compatibility**: Checkpoints from before refactoring can be loaded
2. **Inference Determinism**: Same model and input produce same predictions (no noise during inference)
3. **Training Validity**: Training works correctly and loss decreases
4. **Configuration Compatibility**: Configuration is correctly restored from metadata

## Random Numbers

**Important Assumption**: The current code does not have random seed control functionality.

### Random Number Usage

1. **During Training**: Uses `torch.randn()` in `noise_utils.py` to add noise
2. **During Inference (Rollout)**: **No noise** - deterministic

### Testing Strategy

Since random numbers during training cannot be controlled, we adopt the following strategy:

#### Strategy 1: Exact Match Test for Inference (Possible)
- Inference doesn't use noise, so it's completely deterministic
- **Exact match** expected before and after refactoring

#### Strategy 2: Statistical Validity Test for Training (No random control)
- Verify loss decreases during training
- Verify multiple runs fall within statistically reasonable range
- **Exact match before and after refactoring is not expected**

## Test Strategy

### Phase 1: Unit Tests (Module-level tests)

Verify each module works correctly.

#### 1.1 Configuration Module (`gns/config.py`)

**Purpose**: Verify configuration class is correctly generated from metadata

```python
# tests/test_config.py

def test_config_from_metadata():
    """Test SimulatorConfig.from_metadata() creates correct configuration"""
    # Load sample metadata from dataset
    metadata = load_sample_metadata("WaterDrop")

    # Create config
    config = SimulatorConfig.from_metadata(metadata)

    # Verify all fields are populated correctly
    assert config.dim == metadata['dim']
    assert config.bounds == metadata['bounds']
    assert config.connectivity_radius == expected_radius
    # ... etc

def test_config_validation():
    """Test configuration validation catches invalid inputs"""
    # Test with invalid values
    with pytest.raises(ValueError):
        SimulatorConfig(
            dim=-1,  # Invalid dimension
            # ...
        )
```

#### 1.2 Model Architecture (`gns/learned_simulator.py`)

**Purpose**: Verify model forward computation works correctly

**Note**: Since class separation in Step 2 was reverted, we only test `LearnedSimulator`.

```python
# tests/test_model.py

def test_learned_simulator_forward():
    """Test LearnedSimulator produces expected output shape"""
    config = create_test_config()
    simulator = LearnedSimulator(config)

    # Create dummy input
    batch = create_dummy_batch(
        batch_size=2,
        sequence_length=6,
        num_particles=100
    )

    # Forward pass
    output = simulator.predict_positions(batch)

    # Verify output shape
    assert output.shape == (100, config.dim)
    assert not torch.isnan(output).any()

def test_learned_simulator_deterministic():
    """Test LearnedSimulator produces deterministic output (inference mode)"""
    config = create_test_config()
    simulator = LearnedSimulator(config)
    simulator.eval()

    # Same input twice
    input_data = create_dummy_input()

    torch.manual_seed(42)
    output1 = simulator.predict_positions(input_data)

    torch.manual_seed(42)
    output2 = simulator.predict_positions(input_data)

    # Should be identical (inference is deterministic)
    assert torch.allclose(output1, output2)
```

#### 1.3 Training Utilities (`gns/training.py`)

**Purpose**: Verify training loop and utilities work correctly

```python
# tests/test_training.py

def test_prepare_training_batch():
    """Test batch preparation extracts features correctly"""
    example = create_sample_example(n_features=3)

    batch = prepare_training_batch(
        example=example,
        n_features=3,
        device_id=torch.device('cpu')
    )

    # Verify all expected keys exist
    assert 'position' in batch
    assert 'particle_type' in batch
    assert 'material_property' in batch
    assert 'labels' in batch

    # Verify shapes
    assert batch['position'].ndim == 4

def test_acceleration_loss():
    """Test loss computation produces expected results"""
    pred_acc = torch.randn(10, 100, 2)
    target_acc = torch.randn(10, 100, 2)
    mask = torch.ones(10, 100).bool()

    loss = acceleration_loss(pred_acc, target_acc, mask)

    assert loss.ndim == 0  # Scalar
    assert loss >= 0  # Non-negative
```

#### 1.4 Rollout/Inference (`gns/rollout.py`, `gns/inference_utils.py`)

**Purpose**: Verify rollout functionality works correctly

```python
# tests/test_rollout.py

def test_rollout_deterministic():
    """Test rollout produces deterministic predictions"""
    simulator = create_test_simulator()
    initial_positions = create_test_positions()
    particle_types = create_test_particle_types()

    # Run rollout twice with same seed
    torch.manual_seed(42)
    rollout1 = rollout(
        simulator=simulator,
        position_sequence=initial_positions,
        particle_types=particle_types,
        n_particles_per_example=torch.tensor([100]),
        nsteps=10
    )

    torch.manual_seed(42)
    rollout2 = rollout(
        simulator=simulator,
        position_sequence=initial_positions,
        particle_types=particle_types,
        n_particles_per_example=torch.tensor([100]),
        nsteps=10
    )

    assert torch.allclose(rollout1, rollout2)
```

#### 1.5 Rendering (`gns/render.py`)

**Purpose**: Verify rendering functionality works correctly

```python
# tests/test_render.py

def test_load_rollout_pickle():
    """Test rollout data loading"""
    rollout_data = load_rollout_pickle(
        rollout_dir="test_data/",
        rollout_name="sample_rollout"
    )

    assert 'initial_positions' in rollout_data
    assert 'predicted_rollout' in rollout_data
    assert 'metadata' in rollout_data

def test_prepare_trajectory_data():
    """Test trajectory data preparation"""
    rollout_data = load_sample_rollout()

    trajectory = prepare_trajectory_data(rollout_data)

    assert 'dims' in trajectory
    assert 'num_particles' in trajectory
    assert 'num_steps' in trajectory
```

---

### Phase 2: Integration Tests

Verify multiple modules work together.

#### 2.1 End-to-End Training Test

**Purpose**: Verify entire training pipeline works

```python
# tests/integration/test_training_pipeline.py

def test_training_pipeline_short():
    """Test complete training pipeline runs without errors"""
    # Setup
    config = SimulatorConfig(...)
    simulator = get_simulator(config, device=torch.device('cpu'))
    dataloader = get_training_dataloader(
        dataset_path="test_data/WaterDrop",
        input_length_sequence=6,
        batch_size=2
    )

    # Run short training
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)

    for step in range(10):
        loss = run_training_loop(
            simulator=simulator,
            dataloader=dataloader,
            optimizer=optimizer,
            step=step,
            config=config
        )
        assert not torch.isnan(loss)
```

#### 2.2 End-to-End Inference Test

**Purpose**: Verify entire inference pipeline works

```python
# tests/integration/test_inference_pipeline.py

def test_inference_pipeline():
    """Test complete inference pipeline produces valid output"""
    # Load trained model
    simulator = load_test_model()

    # Load test data
    dataloader = get_rollout_dataloader(
        dataset_path="test_data/WaterDrop",
        mode="rollout"
    )

    # Run rollout
    rollouts = predict_rollouts(
        simulator=simulator,
        dataloader=dataloader,
        config=config,
        rollout_dir="test_output/",
        num_rollouts=2
    )

    # Verify output exists and is valid
    assert len(rollouts) == 2
    for rollout_data in rollouts:
        assert 'predicted_rollout' in rollout_data
        assert rollout_data['predicted_rollout'].shape[0] > 0
```

---

### Phase 3: Equivalence Tests

Verify same results are obtained before and after refactoring.

#### 3.1 Checkpoint Loading Equivalence

**Purpose**: Verify checkpoints from before refactoring can be loaded correctly

```python
# tests/equivalence/test_checkpoint_compatibility.py

def test_load_old_checkpoint():
    """Test loading checkpoint from before refactoring"""
    # Load checkpoint created with old code
    checkpoint_path = "test_data/old_checkpoint.pth"

    # Create new model
    config = SimulatorConfig.from_metadata(metadata)
    simulator = get_simulator(config, device=torch.device('cpu'))

    # Load old checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    simulator.load_state_dict(checkpoint['model_state_dict'])

    # Verify model loads successfully
    assert simulator is not None

    # Test forward pass works
    test_input = create_test_batch()
    output = simulator(test_input)
    assert not torch.isnan(output).any()
```

#### 3.2 Prediction Equivalence (Rollout Only)

**Purpose**: Verify inference (rollout) is deterministic

**Important**: Since noise is not used during inference, we expect **exact match**.

### Step 4: Create Equivalence Test

Create equivalence test (execute in test environment outside repository):

```python
# equivalence-tests/tests/test_equivalence.py
"""Test equivalence between old and new code

This test runs in a separate environment outside the repositories.
It imports both old code (from ../main) and new code (from ../wt-cleanup)
and compares their outputs.
"""
import sys
import numpy as np
import torch
import pickle

# Add both code paths
sys.path.insert(0, '../wt-cleanup')  # New code
sys.path.insert(1, '../main')       # Old code (for reference only)

# Import NEW code
from gns import config, training, rollout

def test_rollout_equivalence():
    """Test rollout predictions match between old and new code

    IMPORTANT: Rollout inference is deterministic (no noise).
    We expect EXACT match, not just close approximation.

    This test:
    1. Loads a checkpoint trained with OLD code
    2. Loads reference rollout generated with OLD code
    3. Runs rollout with NEW code using the same checkpoint
    4. Compares outputs
    """

    # Load reference rollout (generated with OLD code)
    # Note: Old code saves as "rollout_ex0.pkl" (output_filename + "_ex" + example_number)
    with open("test_data/old_results/rollout/rollout_ex0.pkl", "rb") as f:
        reference_data = pickle.load(f)

    reference_rollout = reference_data['predicted_rollout']
    initial_positions = reference_data['initial_positions']
    particle_types = reference_data['particle_types']

    # Load metadata
    import json
    with open("test_data/WaterDropSample/metadata.json") as f:
        metadata = json.load(f)

    # Create simulator with NEW code
    simulator_config = config.SimulatorConfig.from_metadata(metadata)
    simulator = training.get_simulator(simulator_config, device=torch.device('cpu'))

    # Load checkpoint (trained with OLD code)
    checkpoint = torch.load(
        "test_data/old_results/model/model-10.pt",
        map_location='cpu'
    )
    simulator.load_state_dict(checkpoint['model_state_dict'])
    simulator.eval()

    # Prepare inputs
    position_seq = torch.tensor(initial_positions, dtype=torch.float32)
    particle_type_tensor = torch.tensor(particle_types, dtype=torch.long)
    n_particles = torch.tensor([initial_positions.shape[1]])

    # Material property (if exists in reference data, otherwise zeros)
    if 'material_property' in reference_data:
        material_property = torch.tensor(
            reference_data['material_property'],
            dtype=torch.float32
        )
    else:
        # Default: zeros for all particles
        material_property = torch.zeros(initial_positions.shape[1], dtype=torch.float32)

    # Run rollout with NEW code
    device = torch.device('cpu')
    with torch.no_grad():
        _, predicted_rollout = rollout.rollout(
            simulator=simulator,
            position=position_seq,
            particle_types=particle_type_tensor,
            material_property=material_property,
            n_particles_per_example=n_particles,
            nsteps=reference_rollout.shape[0],
            simulator_config=simulator_config,
            device=device
        )

    predicted_np = predicted_rollout.cpu().numpy()

    # EXACT match expected (rollout is deterministic)
    try:
        np.testing.assert_array_equal(
            predicted_np,
            reference_rollout,
            err_msg="Rollout should be EXACTLY identical"
        )
        print("✅ Rollouts match exactly!")
        return True
    except AssertionError:
        # If not exactly equal, check how close they are
        max_diff = np.abs(predicted_np - reference_rollout).max()
        mean_diff = np.abs(predicted_np - reference_rollout).mean()

        print(f"⚠️  Rollouts differ:")
        print(f"   Max difference: {max_diff}")
        print(f"   Mean difference: {mean_diff}")

        # Check if within floating point tolerance
        np.testing.assert_allclose(
            predicted_np,
            reference_rollout,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"Rollouts should match within FP precision (max_diff={max_diff})"
        )
        print("✅ Rollouts match within floating point precision")
        return True

if __name__ == "__main__":
    test_rollout_equivalence()
```

**Expected Results:**
- ✅ Exact match: Refactoring is correct
- ⚠️ Floating point error range: Acceptable (different operation order, etc.)
- ❌ Large difference: Refactoring has bugs

#### 3.3 Training Validity (Without Seed Control)

**Purpose**: Verify training works correctly and loss decreases

**Important**: Since there's no random seed control, **we don't test exact reproducibility**. Instead, we verify training validity.

```python
# tests/equivalence/test_training_validity.py

@pytest.mark.slow
def test_training_loss_decreases():
    """Test that training loss decreases over steps

    Since we don't control random seeds, we can't test exact reproducibility.
    Instead, we verify that:
    1. Training runs without errors
    2. Loss decreases over time
    3. Final loss is reasonable
    """
    from gns import config, training
    import torch
    import numpy as np

    # Load metadata
    import json
    with open("test_data/WaterDropSample/metadata.json") as f:
        metadata = json.load(f)

    # Create simulator
    simulator_config = config.SimulatorConfig.from_metadata(metadata)
    simulator = training.get_simulator(simulator_config, device=torch.device('cpu'))
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)

    # Create dataloader
    dataloader = training.get_training_dataloader(
        dataset_path="test_data/WaterDropSample",
        input_length_sequence=6,
        batch_size=2
    )

    # Train for 50 steps
    losses = []
    for step, example in enumerate(dataloader):
        if step >= 50:
            break

        batch = training.prepare_training_batch(
            example=example,
            n_features=simulator_config.n_features,
            device_id=torch.device('cpu')
        )

        # Forward + backward
        optimizer.zero_grad()
        pred_acc, target_acc = simulator(batch)
        loss = training.acceleration_loss(
            pred_acc,
            target_acc,
            batch['particle_type'] != simulator_config.kinematic_particle_id
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Verify training completed
    assert len(losses) == 50
    assert all(not np.isnan(l) for l in losses), "No NaN losses"

    # Check loss decreases
    early_loss = np.mean(losses[:10])
    late_loss = np.mean(losses[-10:])

    print(f"Early loss: {early_loss:.6f}")
    print(f"Late loss: {late_loss:.6f}")

    assert late_loss < early_loss, "Loss should decrease"
```

**Expected Behavior:**
- ✅ Loss decreases
- ❌ Exact reproducibility is **not expected**

---

### Phase 4: Performance Tests

Verify performance is maintained after refactoring.

#### 4.1 Training Speed

**Purpose**: Verify training speed hasn't degraded significantly

```python
# tests/performance/test_training_speed.py

@pytest.mark.benchmark
def test_training_iteration_speed(benchmark):
    """Benchmark training iteration speed"""
    config = create_test_config()
    simulator = get_simulator(config, device=torch.device('cpu'))
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
    example = create_test_batch()

    def train_one_step():
        optimizer.zero_grad()
        loss = acceleration_loss(...)
        loss.backward()
        optimizer.step()
        return loss

    # Benchmark
    result = benchmark(train_one_step)

    # Should complete in reasonable time (adjust threshold as needed)
    assert result.stats.mean < 1.0  # seconds
```

#### 4.2 Inference Speed

**Purpose**: Verify inference speed hasn't degraded significantly

```python
# tests/performance/test_inference_speed.py

@pytest.mark.benchmark
def test_rollout_speed(benchmark):
    """Benchmark rollout speed"""
    simulator = create_test_simulator()
    initial_positions = create_test_positions()
    particle_types = create_test_particle_types()

    def run_rollout():
        with torch.no_grad():
            return rollout(
                simulator=simulator,
                position_sequence=initial_positions,
                particle_types=particle_types,
                n_particles_per_example=torch.tensor([100]),
                nsteps=100
            )

    result = benchmark(run_rollout)

    # Should complete in reasonable time
    assert result.stats.mean < 5.0  # seconds
```

---

## Test Execution Plan

### Setup Requirements

1. **Test Data**: Prepare small-scale test dataset
   - `test_data/WaterDrop/` - Training samples
   - `test_data/reference_rollout.pkl` - Reference output from before refactoring
   - `test_data/old_checkpoint.pth` - Checkpoint from before refactoring

2. **Dependencies**: Install pytest, pytest-benchmark
   ```bash
   uv add --dev pytest pytest-benchmark pytest-cov
   ```

3. **Test Directory Structure**:
   ```
   tests/
   ├── __init__.py
   ├── conftest.py              # Shared fixtures
   ├── test_config.py
   ├── test_model.py
   ├── test_training.py
   ├── test_rollout.py
   ├── test_render.py
   ├── integration/
   │   ├── __init__.py
   │   ├── test_training_pipeline.py
   │   └── test_inference_pipeline.py
   ├── equivalence/
   │   ├── __init__.py
   │   ├── test_checkpoint_compatibility.py
   │   ├── test_prediction_equivalence.py
   │   └── test_training_reproducibility.py
   └── performance/
       ├── __init__.py
       ├── test_training_speed.py
       └── test_inference_speed.py
   ```

### Execution Order

1. **Phase 1: Unit Tests** (Required)
   ```bash
   pytest tests/test_*.py -v
   ```
   - Verify all modules work individually

2. **Phase 2: Integration Tests** (Required)
   ```bash
   pytest tests/integration/ -v
   ```
   - Verify entire pipeline works

3. **Phase 3: Equivalence Tests** (Important)
   ```bash
   pytest tests/equivalence/ -v -m slow
   ```
   - Verify same results before and after refactoring
   - **This is the most important test**

4. **Phase 4: Performance Tests** (Optional)
   ```bash
   pytest tests/performance/ -v -m benchmark
   ```
   - Verify no performance degradation

### Success Criteria

Criteria for determining refactoring success:

✅ **Phase 1-2**: All tests pass
✅ **Phase 3**:
   - Checkpoint loading succeeds
   - Prediction results match reference output (relative error < 1e-5)
   - Training reproducibility is guaranteed
✅ **Phase 4**: Performance degradation < 10%

---

## Test Environment Setup (Recommended)

This repository uses git worktree, with pre-refactoring code at `../main`.

**Recommended Approach**: Create independent test environment outside repository and treat both codes equally.

### Directory Structure

```
repos/gns/
├── main/                    # Pre-refactoring code (reference only)
├── wt-cleanup/             # Post-refactoring code (reference only)
└── equivalence-tests/      # ★ Test execution environment ★ (outside repository)
    ├── test_data/
    │   ├── WaterDropSample/      # Test data
    │   ├── old_results/          # Old code execution results
    │   └── new_results/          # New code execution results
    ├── tests/
    │   ├── __init__.py
    │   ├── conftest.py
    │   └── test_equivalence.py   # Equivalence test
    ├── pytest.ini
    └── README.md
```

### Setup: Create Test Environment

```bash
# Create test environment directory (outside repositories)
cd /Users/masahiro/repos/gns
mkdir -p equivalence-tests
cd equivalence-tests

# Create directory structure
mkdir -p test_data/{old_results,new_results}
mkdir -p tests

# Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib pytest
```

### Step 2: Link WaterDropSample Data

Create symbolic link to existing WaterDropSample data:

```bash
cd equivalence-tests
# Adjust according to WaterDropSample data location
ln -s /path/to/WaterDropSample test_data/WaterDropSample
```

### Step 3: Generate Reference Data with Old Code

Generate reference data with pre-refactoring code (`main/`):

**Important**: Old code uses `absl.app`, so execute as CLI rather than calling directly from Python script.

```bash
cd equivalence-tests

# Train a small model with OLD code (10 steps for testing)
python ../main/gns/train.py \
  --mode=train \
  --data_path=test_data/WaterDropSample \
  --model_path=test_data/old_results/model \
  --ntraining_steps=10

# Generate reference rollout with OLD code
python ../main/gns/train.py \
  --mode=rollout \
  --data_path=test_data/WaterDropSample \
  --model_path=test_data/old_results/model \
  --model_file=model-10.pt \
  --output_path=test_data/old_results/rollout \
  --num_rollouts=1
```

Generated files:
- `test_data/old_results/model/model-10.pt` - Checkpoint trained with old code
- `test_data/old_results/rollout/rollout_ex0.pkl` - Rollout generated with old code (example 0)

**Note**: Old code uses absl, so `absl-py` must be installed in environment:
```bash
pip install absl-py
```

### Step 5: Run Equivalence Test

Execute test:

```bash
cd equivalence-tests

# Option 1: Run with pytest
pytest tests/test_equivalence.py -v

# Option 2: Run as Python script
python tests/test_equivalence.py
```

**Expected Output:**
```
✅ Rollouts match exactly!
```

Or

```
⚠️  Rollouts differ:
   Max difference: 1.234e-07
   Mean difference: 5.678e-09
✅ Rollouts match within floating point precision
```

### Summary: Test Execution Location

With this approach, tests execute in **independent environment outside repository**:

| Location | Purpose | Test Execution |
|----------|---------|----------------|
| `main/` | Old code | ❌ Don't execute (reference data generation only) |
| `wt-cleanup/` | New code | ❌ Don't execute (import only) |
| `equivalence-tests/` | Test environment | ✅ **Execute here** |

**Benefits:**
- Repository stays clean (no `test_data/` or `tests/` needed)
- Treats both codes equally
- Strict testing in clean environment
- No `.gitignore` adjustments needed

---

## Continuous Testing

### CI Integration

Automatically run tests with GitHub Actions workflow:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync

      - name: Run unit tests
        run: uv run pytest tests/test_*.py -v

      - name: Run integration tests
        run: uv run pytest tests/integration/ -v

      - name: Run equivalence tests
        run: uv run pytest tests/equivalence/ -v
        if: github.event_name == 'pull_request'
```

---

## Notes

- **Numerical Stability**: Account for floating point arithmetic errors, compare with `rtol=1e-5, atol=1e-6`
- **Randomness**: Fix random seeds in all tests to guarantee deterministic results
- **Test Data Size**: Test with small-scale data to reduce execution time
- **Reference Data**: Save reference data generated with pre-refactoring code and verify equivalence
