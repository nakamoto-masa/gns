# Refactoring Equivalence Testing Plan

このドキュメントは、リファクタリング後のコードが元のコードと機能的に等価であることを検証するためのテスト計画を示します。

## 目的

リファクタリングによってコードの構造は変更されましたが、以下の動作が保持されていることを確認します：

1. **チェックポイント互換性**: リファクタリング前のチェックポイントが読み込める
2. **推論の決定性**: 同じモデル・同じ入力で同じ予測結果が得られる（推論時はノイズなし）
3. **訓練の妥当性**: 訓練が正常に動作し、lossが減少する
4. **設定の互換性**: メタデータから正しく設定が復元される

## 乱数について

**重要な前提**: 現在のコードには乱数シード制御機能がありません。

### 乱数の使用箇所

1. **訓練時**: `noise_utils.py` で `torch.randn()` を使用してノイズを付加
2. **推論時（ロールアウト）**: **ノイズなし** - 決定論的

### テスト戦略

訓練時の乱数は制御できないため、以下の戦略を取ります：

#### 戦略1: 推論の完全一致テスト（可能）
- 推論はノイズを使わないため、完全に決定論的
- リファクタリング前後で**完全一致**を期待

#### 戦略2: 訓練の統計的妥当性テスト（乱数制御なし）
- 訓練時のloss減少を確認
- 複数回実行して統計的に妥当な範囲に収まることを確認
- リファクタリング前後の**厳密な一致は期待しない**

## テスト戦略

### Phase 1: Unit Tests (モジュール単位のテスト)

各モジュールが正しく動作することを確認します。

#### 1.1 Configuration Module (`gns/config.py`)

**目的**: 設定クラスがメタデータから正しく生成されることを確認

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

#### 1.2 Model Architecture (`gns/learned_simulator.py`, `gns/graph_model.py`)

**目的**: モデルの前向き計算が正しく動作することを確認

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
    output = simulator(batch)

    # Verify output shape
    assert output.shape == (2, 100, config.dim)
    assert not torch.isnan(output).any()

def test_graph_model_consistency():
    """Test GraphNeuralNetworkModel produces deterministic output"""
    config = create_test_config()
    model = GraphNeuralNetworkModel(config)

    # Same input twice
    input_data = create_dummy_input()

    torch.manual_seed(42)
    output1 = model(input_data)

    torch.manual_seed(42)
    output2 = model(input_data)

    # Should be identical
    assert torch.allclose(output1, output2)
```

#### 1.3 Training Utilities (`gns/training.py`)

**目的**: トレーニングループとユーティリティが正しく動作することを確認

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

**目的**: ロールアウト機能が正しく動作することを確認

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

**目的**: レンダリング機能が正しく動作することを確認

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

### Phase 2: Integration Tests (統合テスト)

複数のモジュールが連携して動作することを確認します。

#### 2.1 End-to-End Training Test

**目的**: 訓練パイプライン全体が動作することを確認

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

**目的**: 推論パイプライン全体が動作することを確認

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

### Phase 3: Equivalence Tests (等価性テスト)

リファクタリング前後で同じ結果が得られることを確認します。

#### 3.1 Checkpoint Loading Equivalence

**目的**: リファクタリング前のチェックポイントが正しく読み込めることを確認

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

**目的**: 推論（ロールアウト）が決定論的であることを確認

**重要**: 推論時はノイズを使用しないため、**完全一致**を期待します。

```python
# tests/equivalence/test_prediction_equivalence.py

@pytest.mark.slow
def test_rollout_predictions_match():
    """Test rollout predictions match between old and new code

    IMPORTANT: Rollout inference is deterministic (no noise).
    We expect EXACT match, not just close approximation.

    This test requires:
    1. A checkpoint trained with old code
    2. A reference rollout generated with old code (same checkpoint, same input)
    """
    import sys
    import numpy as np
    import torch
    from gns import config, training, rollout

    # Load reference data (generated with old code)
    reference_data = np.load("test_data/reference_rollout.pkl", allow_pickle=True)
    reference_rollout = reference_data['predicted_rollout']  # Shape: (nsteps, nparticles, dim)
    initial_positions = reference_data['initial_positions']  # Shape: (sequence_length, nparticles, dim)
    particle_types = reference_data['particle_types']

    # Load metadata
    import json
    with open("test_data/WaterDropSample/metadata.json") as f:
        metadata = json.load(f)

    # Create simulator with NEW code
    simulator_config = config.SimulatorConfig.from_metadata(metadata)
    simulator = training.get_simulator(simulator_config, device=torch.device('cpu'))

    # Load checkpoint (trained with old code)
    checkpoint = torch.load("test_data/old_checkpoint.pth", map_location='cpu')
    simulator.load_state_dict(checkpoint['model_state_dict'])
    simulator.eval()

    # Prepare inputs
    position_seq = torch.tensor(initial_positions, dtype=torch.float32)
    particle_type_tensor = torch.tensor(particle_types, dtype=torch.long)
    n_particles = torch.tensor([initial_positions.shape[1]])

    # Run rollout with NEW code
    with torch.no_grad():
        predicted_rollout = rollout.rollout(
            simulator=simulator,
            position_sequence=position_seq,
            particle_types=particle_type_tensor,
            n_particles_per_example=n_particles,
            nsteps=reference_rollout.shape[0]
        )

    predicted_np = predicted_rollout.cpu().numpy()

    # EXACT match expected (rollout is deterministic)
    # Use assert_array_equal for exact comparison
    try:
        np.testing.assert_array_equal(
            predicted_np,
            reference_rollout,
            err_msg="Rollout should be EXACTLY identical (inference is deterministic)"
        )
        print("✓ Rollouts match exactly!")
    except AssertionError:
        # If not exactly equal, check how close they are
        max_diff = np.abs(predicted_np - reference_rollout).max()
        mean_diff = np.abs(predicted_np - reference_rollout).mean()

        print(f"✗ Rollouts differ:")
        print(f"  Max difference: {max_diff}")
        print(f"  Mean difference: {mean_diff}")

        # Check if within floating point tolerance
        np.testing.assert_allclose(
            predicted_np,
            reference_rollout,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"Rollouts should match within FP precision (max_diff={max_diff})"
        )
        print("✓ Rollouts match within floating point precision")
```

**期待される結果:**
- ✅ 完全一致: リファクタリングが正しい
- ⚠️ 浮動小数点誤差程度の差異: 許容範囲（演算順序の違いなど）
- ❌ 大きな差異: リファクタリングにバグあり

#### 3.3 Training Validity (Without Seed Control)

**目的**: 訓練が正常に動作し、lossが減少することを確認

**重要**: 乱数シード制御がないため、**厳密な再現性はテストしません**。代わりに、訓練の妥当性を確認します。

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

**期待される動作:**
- ✅ Lossが減少
- ❌ 厳密な再現性は**期待しない**

---

### Phase 4: Performance Tests (性能テスト)

リファクタリング後も性能が維持されていることを確認します。

#### 4.1 Training Speed

**目的**: 訓練速度が大幅に低下していないことを確認

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

**目的**: 推論速度が大幅に低下していないことを確認

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

1. **Test Data**: 小規模なテストデータセットを準備
   - `test_data/WaterDrop/` - 訓練用サンプル
   - `test_data/reference_rollout.pkl` - リファクタリング前の参照出力
   - `test_data/old_checkpoint.pth` - リファクタリング前のチェックポイント

2. **Dependencies**: pytest, pytest-benchmark をインストール
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

1. **Phase 1: Unit Tests** (必須)
   ```bash
   pytest tests/test_*.py -v
   ```
   - すべてのモジュールが個別に動作することを確認

2. **Phase 2: Integration Tests** (必須)
   ```bash
   pytest tests/integration/ -v
   ```
   - パイプライン全体が動作することを確認

3. **Phase 3: Equivalence Tests** (重要)
   ```bash
   pytest tests/equivalence/ -v -m slow
   ```
   - リファクタリング前後で同じ結果が得られることを確認
   - **これが最も重要なテスト**

4. **Phase 4: Performance Tests** (オプション)
   ```bash
   pytest tests/performance/ -v -m benchmark
   ```
   - 性能低下がないことを確認

### Success Criteria

リファクタリングが成功したと判断する基準：

✅ **Phase 1-2**: すべてのテストがパス
✅ **Phase 3**:
   - チェックポイント読み込みが成功
   - 予測結果が参照出力と一致（相対誤差 < 1e-5）
   - 訓練の再現性が保証される
✅ **Phase 4**: 性能低下が10%未満

---

## Reference Data Generation Using Worktree

このリポジトリはgit worktreeを使用しており、リファクタリング前のコードが `../main` に存在します。
これを活用して、リファクタリング前後のコードを同時に使用したテストが可能です。

### Setup: Shared Test Data

WaterDropSample データを両方のworktreeで共有できるようにシンボリックリンクを作成：

```bash
# Create test data directory in refactored worktree
mkdir -p test_data

# If you have WaterDropSample data somewhere, link it
# Option 1: If data exists in main worktree
ln -s ../main/datasets/WaterDropSample test_data/WaterDropSample

# Option 2: If data exists elsewhere
ln -s /path/to/WaterDropSample test_data/WaterDropSample

# Option 3: Create minimal test data (see below)
```

### Creating Minimal Test Data

WaterDropSampleデータがない場合、最小限のテストデータを作成：

```python
# scripts/create_minimal_test_data.py
"""Create minimal test dataset for equivalence testing"""

import os
import numpy as np
import json

def create_minimal_test_data(output_dir="test_data/WaterDropSample"):
    """Create minimal dataset with 1 trajectory, 100 particles, 10 steps"""
    os.makedirs(output_dir, exist_ok=True)

    # Metadata
    metadata = {
        "bounds": [[0.1, 0.9], [0.1, 0.9]],
        "sequence_length": 6,
        "default_connectivity_radius": 0.015,
        "dim": 2,
        "dt": 0.0025,
        "vel_mean": [5.123277536458455e-06, -0.0009965205918140803],
        "vel_std": [0.0021978993231675805, 0.0026653552458701774],
        "acc_mean": [5.237611158734864e-07, 2.3633027988858656e-07],
        "acc_std": [0.0002582944917306106, 0.00029554531667679154]
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)

    # Create train/valid/test splits with minimal data
    for split in ["train", "valid", "test"]:
        split_dir = f"{output_dir}/{split}"
        os.makedirs(split_dir, exist_ok=True)

        # Single trajectory: 100 particles, 10 timesteps
        num_particles = 100
        num_steps = 10

        # Random positions (2D)
        positions = np.random.rand(num_steps, num_particles, 2) * 0.8 + 0.1

        # Particle types (mix of types)
        particle_types = np.random.choice([0, 1, 3, 5], size=num_particles)

        # Save as .npz
        np.savez(
            f"{split_dir}/sample_0.npz",
            positions=positions.astype(np.float32),
            particle_type=particle_types.astype(np.int32)
        )

    print(f"Created minimal test data in {output_dir}")

if __name__ == "__main__":
    create_minimal_test_data()
```

Run it:
```bash
python scripts/create_minimal_test_data.py
```

### 1. Generate Reference Data with Old Code

リファクタリング前のコード (`../main`) で参照データを生成：

```bash
# Move to main worktree
cd ../main

# Install dependencies (if not already)
# Use the same Python environment or create a separate one
python -m venv .venv-main
source .venv-main/bin/activate
pip install torch numpy matplotlib pyevtk

# Train a small model (10 steps for testing)
python gns/train.py \
  --mode=train \
  --data_path=../wt-cleanup/test_data/WaterDropSample \
  --model_path=../wt-cleanup/test_data/old_model \
  --ntraining_steps=10

# Generate reference rollout
python gns/train.py \
  --mode=rollout \
  --data_path=../wt-cleanup/test_data/WaterDropSample \
  --model_path=../wt-cleanup/test_data/old_model \
  --model_file=model-10.pt \
  --output_path=../wt-cleanup/test_data/reference_rollout \
  --num_rollouts=1

# Return to refactored worktree
cd ../wt-cleanup
```

これで以下のファイルが生成されます：
- `test_data/old_model/model-10.pt` - リファクタリング前のコードで訓練したチェックポイント
- `test_data/reference_rollout/rollout_0.pkl` - リファクタリング前のコードで生成したロールアウト

### 2. Alternative: Generate Reference via Python Script

より制御可能な方法として、Pythonスクリプトで参照データを生成：

```python
# scripts/generate_reference_data.py
"""Generate reference data using old code from ../main worktree"""

import sys
import os

# Add old code path
sys.path.insert(0, '../main')

import torch
import numpy as np

# Import from old code
from gns.train import get_simulator, train_one_step  # old implementation

def generate_reference_training_data():
    """Generate reference training losses"""
    # Setup (using old code)
    simulator = get_simulator(...)  # old implementation
    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)

    # Train for 100 steps
    losses = []
    for step in range(100):
        loss = train_one_step(simulator, optimizer, ...)
        losses.append(loss.item())

    # Save reference
    np.save('test_data/reference_losses.npy', losses)
    torch.save(simulator.state_dict(), 'test_data/old_checkpoint.pth')

    print(f"Generated reference data: {len(losses)} training steps")

if __name__ == "__main__":
    generate_reference_training_data()
```

Run from refactored worktree:
```bash
cd wt-cleanup
python scripts/generate_reference_data.py
```

---

## Continuous Testing

### CI Integration

GitHub Actions ワークフローでテストを自動実行：

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

- **Numerical Stability**: 浮動小数点演算の誤差を考慮して、`rtol=1e-5, atol=1e-6` で比較
- **Randomness**: すべてのテストで乱数シードを固定してdeterministicな結果を保証
- **Test Data Size**: 小規模データでテストし、実行時間を短縮
- **Reference Data**: リファクタリング前のコードで生成した参照データを保存し、等価性を検証
