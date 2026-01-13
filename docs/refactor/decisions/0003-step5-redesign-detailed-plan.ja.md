# Step 5 再設計: 詳細実装計画

**日付**: 2026-01-08
**ステータス**: 承認済み
**関連**: Step 5（モジュールと実行スクリプトの分離）

## 背景

### 第1回実装の失敗（refactor/step5_1st_try ブランチ）

**結果**: `scripts/gns_train.py`が489行になった（目標: 50-80行）

**原因**: 中間レベルの抽象化層が欠けていた
- 学習ループ全体のオーケストレーション（`run_training_loop()`）がなかった
- バッチ準備のボイラープレート（15行のif/else）が3箇所に重複
- 学習率スケジューリングがハードコードされていた
- 検証ロジックが学習ループに埋め込まれていた

**第1回で抽出した関数** (gns/training.py):
- `train_step()`, `validation_step()` - 単一ステップ操作
- `save_checkpoint()`, `load_checkpoint()` - チェックポイント管理
- `optimizer_to()`, `acceleration_loss()` - ユーティリティ

**問題点**: これらは低レベル関数のみで、学習ループ全体を統合する高レベル関数が欠けていた。

## 再設計の方針

### 目標
- `scripts/gns_train.py`: 60-80行（CLIフラグパース + 高レベル関数呼び出しのみ）
- `scripts/gns_train_multinode.py`: 70-90行
- `scripts/gns_render_rollout.py`: 35-50行

### 新規追加する重要な関数

#### 1. `run_training_loop()` - 最重要
学習ループ全体を管理する高レベルオーケストレーション関数。これが第1回で欠けていた最も重要な抽象化。

**統合する要素**:
- エポックとステップのイテレーション
- バッチ準備 (`prepare_training_batch`)
- 学習ステップ (`train_step`)
- 検証ステップ (`validation_step`)
- 学習率更新 (`update_learning_rate`)
- チェックポイント保存 (`save_checkpoint`)
- 進捗ログ出力

#### 2. `prepare_training_batch()` - 重要
15行のボイラープレート（特徴抽出のif/else分岐）を1行の関数呼び出しに置き換える。

**現在の重複コード**（gns/train.pyの3箇所に存在）:
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

**抽出後**:
```python
batch = training.prepare_training_batch(example, n_features, device_id)
# batch = {position, particle_type, material_property, n_particles_per_example, labels}
```

#### 3. `update_learning_rate()` - 重要
学習率の指数減衰をループ内にハードコードするのではなく、関数として抽出。

**現在のハードコード**（学習ループ内）:
```python
lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
for param in optimizer.param_groups:
    param['lr'] = lr_new
```

**抽出後**:
```python
lr_new = training.update_learning_rate(optimizer, step, lr_init, lr_decay, lr_decay_steps, world_size)
```

## 実装フェーズ

### Phase 1: gns/training.py の作成（400-450行）

**抽出元**: `gns/train.py`（663行）

**関数構成**:

1. **ユーティリティ関数** (既存)
   - `optimizer_to()` - オプティマイザをデバイスに移動
   - `acceleration_loss()` - マスク付き損失計算

2. **バッチ準備** (新規・重要)
   - `prepare_training_batch()` - 15行ボイラープレート削減

3. **学習率管理** (新規・重要)
   - `update_learning_rate()` - 学習率スケジューリング

4. **ノイズ追加** (新規)
   - `add_training_noise()` - 位置シーケンスへのノイズ追加

5. **単一ステップ操作** (既存)
   - `train_step()` - 1ステップの学習
   - `validation_step()` - 1ステップの検証

6. **チェックポイント管理** (既存、改良)
   - `save_checkpoint()` - モデルと学習状態の保存
   - `load_checkpoint()` - モデルと学習状態の読み込み
   - `find_latest_checkpoint()` - 最新チェックポイントの検索

7. **シミュレータ生成** (新規)
   - `create_simulator()` - シミュレータインスタンスの生成
   - `get_simulator()` - DDP対応のシミュレータ取得

8. **学習オーケストレーション** (新規・最重要)
   - `run_training_loop()` - 学習ループ全体の管理

### Phase 2: gns/rollout.py の作成（250-300行）

**抽出元**: `gns/train.py`の`predict()`関数（90-172行）

**関数構成**:

1. **単一rollout実行**
   - `run_rollout()` - 1つの軌道予測

2. **rollout保存**
   - `save_rollout()` - 予測結果の保存

3. **バッチ予測**
   - `predict_rollouts()` - データセット全体の予測

4. **分散予測** (新規)
   - `predict_rollouts_distributed()` - DDP用の分散予測

5. **データローディング**
   - `get_rollout_dataloader()` - rollout用データローダー
   - `infer_dataset_features()` - 特徴数の推論

### Phase 3: gns/render.py の作成（300-350行）

**抽出元**: `gns/render_rollout.py`（246行）

**新規モジュール**: 可視化機能を関数ベースで実装

**関数構成**:

1. **データローディング**
   - `load_rollout_pickle()` - rolloutデータ読み込み

2. **色マッピング**
   - `TYPE_TO_COLOR` - 粒子タイプから色へのマッピング
   - `get_color_mask()` - 色マスクの生成

3. **2D/3Dレンダリング**
   - `render_2d_trajectory()` - 2D軌道レンダリング
   - `render_3d_trajectory()` - 3D軌道レンダリング

4. **GIFアニメーション**
   - `render_gif_animation()` - GIF生成（2D/3D統合）

5. **VTKエクスポート**
   - `write_vtk_trajectory()` - VTKファイル出力

**注**: 現在の`Render`クラス（246行）の`render_gif_animation()`メソッドは107行あり、2D/3Dが混在している。これを分離。

### Phase 4: 薄いCLIスクリプトの作成

#### scripts/gns_train.py（60-80行）

**構造**:
```python
"""Single-GPU training script - thin CLI wrapper"""

# フラグ定義（30行）

def train_single_gpu(rank, flags_dict, world_size, device):
    # メタデータ読み込み
    metadata = reading_utils.read_metadata(flags_dict["data_path"], "train")
    simulator_config = config.SimulatorConfig.from_metadata(metadata)

    # シミュレータとオプティマイザ取得
    simulator = training.get_simulator(...)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=...)

    # チェックポイント読み込み（オプション）
    if flags_dict["model_file"]:
        training.load_checkpoint(...)

    # データローダー取得
    train_dl = training.get_training_dataloader(...)
    valid_dl = training.get_training_dataloader(..., split='valid') if ... else None

    # 学習ループ実行（全ロジックをtrainingモジュールに委譲）
    training.run_training_loop(
        simulator, train_dl, valid_dl, optimizer, device_id,
        rank, world_size, flags_dict, simulator_config,
        flags_dict["model_path"], is_distributed)

def main(_):
    # デバイス設定とモードディスパッチ
    if FLAGS.mode == 'train':
        train_single_gpu(...)
    elif FLAGS.mode in ['valid', 'rollout']:
        rollout.run_prediction_mode(...)
```

**削減**: 663行 → 60-80行（約90%削減）

#### scripts/gns_train_multinode.py（70-90行）

**構造**: `gns_train.py`とほぼ同じだが、分散初期化を追加

**削減**: 669行 → 70-90行（約90%削減）

#### scripts/gns_render_rollout.py（35-50行）

**構造**:
```python
"""Render rollout predictions - thin CLI wrapper"""

def main(_):
    rollout_data = render.load_rollout_pickle(FLAGS.rollout_dir, FLAGS.rollout_name)

    if FLAGS.output_mode == "gif":
        render.render_gif_animation(rollout_data, ...)
    elif FLAGS.output_mode == "vtk":
        render.write_vtk_trajectory(rollout_data, ...)
```

**削減**: 246行 → 35-50行（約85%削減）

## 実装順序

1. **gns/training.py** - 基盤
   - ユーティリティ → バッチ準備 → 学習率管理 → ノイズ追加
   - 単一ステップ → チェックポイント → シミュレータ生成
   - **最後に**: `run_training_loop()`

2. **gns/rollout.py** - 予測
   - データローディング → 単一rollout → 保存 → バッチ予測 → 分散予測

3. **gns/render.py** - 可視化
   - データローディング → 色マッピング → 2D/3Dレンダリング → GIF/VTK

4. **CLIスクリプト** - ラッパー
   - `scripts/gns_train.py` → `scripts/gns_train_multinode.py` → `scripts/gns_render_rollout.py`

## 検証計画

### Phase 1後（training.py）
```bash
# インポートテスト
uv run python -c "from gns.training import prepare_training_batch, update_learning_rate; print('OK')"

# 10ステップ学習テスト
uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --ntraining_steps=10 --mode=train

# チェックポイント確認
ls models/test_step5_v2/
```

### Phase 2後（rollout.py）
```bash
# rollout実行テスト
uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step5_v2/

# rollout出力確認
ls rollouts/test_step5_v2/
```

### Phase 3後（render.py）
```bash
# GIFレンダリングテスト
uv run python scripts/gns_render_rollout.py \
  --rollout_dir=rollouts/test_step5_v2/ \
  --rollout_name=rollout_ex0 --output_mode=gif

# GIF生成確認
ls rollouts/test_step5_v2/*.gif
```

## 成功基準

### コード量
- ✅ `scripts/gns_train.py`: 60-80行
- ✅ `scripts/gns_train_multinode.py`: 70-90行
- ✅ `scripts/gns_render_rollout.py`: 35-50行
- ✅ 総コード量20-30%削減（1,579行 → 1,115-1,320行）

### 機能
- ✅ 10ステップ学習が正常に動作
- ✅ チェックポイント保存/読み込みが動作
- ✅ 検証が学習中に動作
- ✅ rollout予測が動作
- ✅ GIF/VTKレンダリングが動作
- ✅ 既存チェックポイントとの互換性維持
- ✅ 分散学習が動作

### コード品質
- ✅ コード重複なし
- ✅ 全関数にdocstring
- ✅ 型ヒント完備
- ✅ 明確な責任分離（CLI vs ビジネスロジック）

## 第1回との違いまとめ

| 項目 | 第1回 | 第2回（本設計） |
|------|-------|----------------|
| scripts/gns_train.py | 489行 | 60-80行 |
| 学習ループオーケストレーション | なし（CLIに残存） | `run_training_loop()` |
| バッチ準備 | 3箇所に重複 | `prepare_training_batch()` |
| 学習率スケジューリング | ループ内にハードコード | `update_learning_rate()` |
| シミュレータ生成 | CLIに残存 | `create_simulator()`, `get_simulator()` |
| render.py | なし | 新規作成（246行を分離） |

## 参照

- **第1回実装**: ブランチ `refactor/step5_1st_try`（コミット ac7d7b7）
- **現在のStep 4状態**: コミット eee4a66
- **関連決定記録**:
  - `0001-keep-learned-simulator-as-nn-module.md`
  - `0002-type-annotation-fixes.md`
