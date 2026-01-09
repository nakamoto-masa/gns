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

### Step 4: Create Equivalence Test

等価性テストを作成（リポジトリ外のテスト環境で実行）:

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

## Test Environment Setup (Recommended)

このリポジトリはgit worktreeを使用しており、リファクタリング前のコードが `../main` に存在します。

**推奨アプローチ**: リポジトリ外に独立したテスト環境を作成し、両方のコードを対等に扱います。

### Directory Structure

```
repos/gns/
├── main/                    # リファクタリング前のコード (参照のみ)
├── wt-cleanup/             # リファクタリング後のコード (参照のみ)
└── equivalence-tests/      # ★テスト実行環境★ (リポジトリ外)
    ├── test_data/
    │   ├── WaterDropSample/      # テストデータ
    │   ├── old_results/          # 旧コードの実行結果
    │   └── new_results/          # 新コードの実行結果
    ├── tests/
    │   ├── __init__.py
    │   ├── conftest.py
    │   └── test_equivalence.py   # 等価性テスト
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

既存のWaterDropSampleデータへのシンボリックリンクを作成：

```bash
cd equivalence-tests
# WaterDropSampleデータの場所に応じて調整
ln -s /path/to/WaterDropSample test_data/WaterDropSample
```

### Step 3: Generate Reference Data with Old Code

リファクタリング前のコード (`main/`) で参照データを生成：

**重要**: 旧コードは `absl.app` を使用しているため、Pythonスクリプトから直接呼び出すのではなく、CLIとして実行します。

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

生成されるファイル:
- `test_data/old_results/model/model-10.pt` - 旧コードで訓練したチェックポイント
- `test_data/old_results/rollout/rollout_ex0.pkl` - 旧コードで生成したロールアウト (example 0)

**注意**: 旧コードはabslを使用しているため、環境に`absl-py`がインストールされている必要があります:
```bash
pip install absl-py
```

### Step 5: Run Equivalence Test

テストを実行：

```bash
cd equivalence-tests

# Option 1: Run with pytest
pytest tests/test_equivalence.py -v

# Option 2: Run as Python script
python tests/test_equivalence.py
```

**期待される出力:**
```
✅ Rollouts match exactly!
```

または

```
⚠️  Rollouts differ:
   Max difference: 1.234e-07
   Mean difference: 5.678e-09
✅ Rollouts match within floating point precision
```

### Summary: Test Execution Location

このアプローチでは、テストは**リポジトリ外の独立した環境**で実行されます：

| 場所 | 用途 | テスト実行 |
|------|------|-----------|
| `main/` | 旧コード | ❌ 実行しない（参照データ生成のみ） |
| `wt-cleanup/` | 新コード | ❌ 実行しない（インポートのみ） |
| `equivalence-tests/` | テスト環境 | ✅ **ここで実行** |

**メリット:**
- リポジトリが汚れない（`test_data/`や`tests/`が不要）
- 両方のコードを対等に扱える
- クリーンな環境で厳密なテスト
- `.gitignore`の調整不要

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
