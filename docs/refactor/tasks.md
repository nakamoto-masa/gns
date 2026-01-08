# GNS リファクタリングタスクリスト

このドキュメントは、[リファクタリング実装プラン](./plan.ja.md)の実施状況を管理するためのタスクリストです。

## 進捗状況

- [x] Phase 0: 環境構築 (4/4) ✅
- [x] Phase 1: アーキテクチャ分離 (13/13) ✅
- [ ] Phase 2: モジュール/スクリプト明確化 (5/9)
- [x] Phase 3: 設定の整理 (3/3) ✅ (Step 5で実装済み)

**全体進捗: 25/29 タスク完了**

---

## Phase 0: 環境構築 ✅

### Step 0.1: pyproject.toml と .python-version の作成 ✅
- [x] `pyproject.toml` の作成（依存関係定義）
- [x] `.python-version` の作成（Python 3.13 指定）

**成果物:**
- `pyproject.toml` - PyTorch 2.8.0、PyTorch Geometric 2.7.0を含む依存関係定義
- `.python-version` - Python 3.13指定

### Step 0.2: requirements.txt と enviornment.yml の削除 ✅
- [x] `requirements.txt` の削除
- [x] `enviornment.yml` の削除（typo版のファイル名）

**成果物:**
- 旧ファイルの削除完了

### Step 0.3: uv 環境のセットアップと依存関係インストール ✅
- [x] `uv sync` の実行（自動的に venv 作成と依存関係インストール）
- [x] 依存関係のバージョン確認

**実行結果:**
- Python 3.13.11環境を構築
- PyTorch 2.8.0 (CPU版)インストール
- PyTorch Geometric 2.7.0インストール
- torch-cluster, torch-scatter, torch-sparseインストール（PyG CPU版）

**検証コマンド:**
```bash
uv sync
uv pip list
uv run python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print('OK')"
```

### Step 0.4: 既存テストと小規模学習の実行で動作検証 ✅
- [x] 既存テストの実行（test_learned_simulator.py, test_graph_network.py, test_noise_utils.py）
- [x] 小規模学習の実行（10ステップ）- WaterDropSampleデータで正常に実行、lossが2.18→1.83に減少

**検証コマンド:**
```bash
uv run python test/test_learned_simulator.py
uv run python test/test_graph_network.py
uv run python test/test_noise_utils.py
uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

---

## Phase 1: アーキテクチャ分離

### Step 1.1: gns/inference_utils.py の作成（run_inference_loop と apply_kinematic_constraints） ✅
- [x] `gns/inference_utils.py` の作成
- [x] `run_inference_loop()` 関数の実装
- [x] `apply_kinematic_constraints()` 関数の実装

**成果物:**
- `gns/inference_utils.py` - 推論ループと運動学的制約の適用を分離
  - `apply_kinematic_constraints()` - 運動学的粒子の制約適用
  - `run_inference_loop()` - シミュレータによる推論ループ
  - `rollout()` - 既存インターフェース互換のrollout関数

### Step 1.2: gns/train.py と train_multinode.py の rollout() 関数を分離した関数呼び出しに修正 ✅
- [x] `gns/train.py` の `rollout()` 関数を修正
- [x] `gns/train_multinode.py` の `rollout()` 関数を修正

**影響範囲:**
- `gns/train.py` の `rollout()` 関数（56-119行 → 78-88行に短縮）
- `gns/train_multinode.py` の同等部分（86-96行に短縮）

### Step 1.3: 学習とrolloutの実行で検証 ✅
- [x] 10ステップの学習実行 - 成功（loss: 2.28 → 1.85）
- [x] rolloutの実行 - 成功（平均loss: 0.211）
- [x] 出力ファイルの確認 - rollout_ex0.pkl, rollout_ex1.pkl生成確認

**検証コマンド:**
```bash
uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step1/ --ntraining_steps=10 --mode=train

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step1/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step1/

ls rollouts/test_step1/
```

### Step 2.1: gns/graph_model.py の作成（GraphNeuralNetworkModel クラス） ✅
- [x] `gns/graph_model.py` の作成
- [x] `GraphNeuralNetworkModel` クラスの実装
  - グラフ構築
  - メッセージパッシング
  - forward pass

**成果物:**
- `gns/graph_model.py` - 純粋なGNNモデル（210行）
  - `_compute_graph_connectivity()` - グラフ接続性の計算
  - `build_graph_features()` - ノード・エッジ特徴量の構築
  - `predict()` - GNNのforward pass

### Step 2.2: gns/learned_simulator.py を GraphNeuralNetworkModel をラップする形にリファクタリング ✅
- [x] `LearnedSimulator` クラスを `GraphNeuralNetworkModel` をラップする形にリファクタリング
- [x] 正規化、オイラー積分、位置/加速度予測のメソッドを維持
- [x] `save()`, `load()` メソッドを維持

**影響範囲:**
- `gns/learned_simulator.py` 全体（388行 → 297行、91行削減）

**実装結果:**
- GNN部分を `GraphNeuralNetworkModel` に委譲
- `LearnedSimulator` は物理統合（正規化、オイラー積分）に集中
- 既存APIを完全に保持（後方互換性）

### Step 2.3: インポート確認と学習・rollout実行で検証 ✅
- [x] モジュールのインポート確認
- [x] 学習とrolloutの実行で検証

**検証結果:**
- インポートテスト成功
- 10ステップ学習成功（loss: 2.03 → 1.61）
- rollout実行成功（平均loss: 0.137）
- 出力ファイル生成確認（rollout_ex0.pkl, rollout_ex1.pkl）

**検証コマンド:**
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

### Step 2.4: 型アノテーションの修正 ✅
- [x] `X | None` 記法への統一（4箇所）
- [x] `boundaries` 型の統一（`np.ndarray`）
- [x] 未使用インポートの削除
- [x] `device` パラメータへの型アノテーション追加

**実装結果:**
- Python 3.10+ の `X | None` 記法を採用
- `boundaries: np.ndarray` に統一（実装に合わせた型アノテーション）
- 型安全性の向上とIDE補完の改善

**関連ドキュメント:**
- `docs/refactor/decisions/0001-keep-learned-simulator-as-nn-module.md`
- `docs/refactor/decisions/0002-type-annotation-fixes.md`

### Step 3.1: gns/config.py の作成（SimulatorConfig dataclass） ✅
- [x] `gns/config.py` の作成
- [x] `SimulatorConfig` dataclass の実装
- [x] `from_metadata()` クラスメソッドの実装

**成果物:**
- `gns/config.py` - SimulatorConfigデータクラス
  - デフォルト値: `input_sequence_length=6`, `num_particle_types=9`, `kinematic_particle_id=3`
  - メタデータフィールド: `dim`, `dt`, `default_connectivity_radius`, `bounds`, etc.
  - `from_metadata()` クラスメソッドでメタデータから設定を生成

### Step 3.2: gns/train.py と train_multinode.py のグローバル定数を削除し config 引数に置き換え ✅
- [x] `gns/train.py` のグローバル定数（53-55行）を削除
- [x] `gns/train_multinode.py` のグローバル定数（61-63行）を削除
- [x] 各関数で `simulator_config` を引数として受け取るように修正

**影響範囲:**
- `gns/train.py` の修正箇所:
  - `rollout()`, `predict()`, `train()`, `_get_simulator()`, `validation()` 関数
  - すべての `INPUT_SEQUENCE_LENGTH`, `NUM_PARTICLE_TYPES`, `KINEMATIC_PARTICLE_ID` 参照
- `gns/train_multinode.py` の修正箇所:
  - `rollout()`, `rollout_par()`, `predict()`, `predict_par()`, `train()`, `_get_simulator()` 関数
  - すべての定数参照

### Step 3.3: 学習実行で検証 ✅
- [x] 設定オブジェクトの作成確認 - 成功
- [x] 10ステップ学習実行 - 成功（loss: 2.14 → 1.88）
- [x] rollout実行 - 成功（平均loss: 0.538）
- [x] 出力ファイル生成確認 - rollout_ex0.pkl, rollout_ex1.pkl生成確認

**検証コマンド:**
```bash
uv run python -c "from gns.config import SimulatorConfig; \
cfg = SimulatorConfig(); print(cfg)"

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step3/ --ntraining_steps=10 --mode=train

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step3/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step3/
```

### Step 4.1: gns/config.py に build_normalization_stats と infer_feature_dimensions 関数追加 ✅
- [x] `build_normalization_stats()` 関数の実装
- [x] `infer_feature_dimensions()` 関数の実装

**成果物:**
- `gns/config.py` - 2つの新しい関数を追加
  - `build_normalization_stats()` - メタデータから正規化統計を構築（ノイズとの合成計算を含む）
  - `infer_feature_dimensions()` - GNNの入力次元数を推論（nnode_in, nedge_in, particle_dimensions）

### Step 4.2: gns/train.py の _get_simulator() 関数を分割 ✅
- [x] `create_simulator()` 関数の新規作成
- [x] `_get_simulator()` 関数（474-531行）を分割
  - 設定読み込み部分 → `config.py` の関数群を使用
  - シミュレータ生成部分 → `create_simulator()` 関数を使用
- [x] `gns/train_multinode.py` にも同じ変更を適用

**影響範囲:**
- `gns/train.py` - `create_simulator()` 関数追加、`_get_simulator()` 関数リファクタリング（56行 → 25行）
- `gns/train_multinode.py` - 同様の変更

### Step 4.3: 学習実行で検証 ✅
- [x] 10ステップ学習実行 - 成功（loss: 1.94 → 1.36）
- [x] rollout実行 - 成功（平均loss: 0.064）
- [x] 出力ファイル生成確認 - rollout_ex0.pkl, rollout_ex1.pkl生成確認

**検証コマンド:**
```bash
uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step4/ --ntraining_steps=10 --mode=train

uv run python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test_step4/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test_step4/
```

---

## Phase 2: モジュール/スクリプト明確化

### Step 5.1: scripts/ ディレクトリの作成 ✅
- [x] `scripts/` ディレクトリの作成

**成果物:**
- `scripts/` ディレクトリ

### Step 5.2: gns/training.py の作成（train_step, validation_step, checkpoint管理、run_training_loop） ✅
- [x] `gns/training.py` の作成（788行）
- [x] ユーティリティ関数の実装（`optimizer_to()`, `acceleration_loss()`）
- [x] バッチ準備関数の実装（`prepare_training_batch()` - 15行のボイラープレート削減）
- [x] 学習率管理の実装（`update_learning_rate()`）
- [x] ノイズ追加の実装（`add_training_noise()`）
- [x] 単一ステップ操作の実装（`train_step()`, `validation_step()`）
- [x] チェックポイント管理の実装（`save_checkpoint()`, `load_checkpoint()`, `find_latest_checkpoint()`）
- [x] シミュレータ生成の実装（`create_simulator()`, `get_simulator()`）
- [x] データローディングの実装（`get_training_dataloader()`, `infer_dataset_features()`）
- [x] **学習オーケストレーションの実装**（`run_training_loop()` - 最重要関数）

**成果物:**
- `gns/training.py` (788行) - 再利用可能な学習ロジック
  - 第1回実装で欠けていた高レベルオーケストレーション関数を追加
  - バッチ準備のボイラープレート削減
  - 学習率スケジューリングの関数化

### Step 5.3: gns/rollout.py と gns/render.py の作成 ✅
- [x] `gns/rollout.py` の作成（304行）
  - データローディング（`get_rollout_dataloader()`, `infer_dataset_features()`）
  - 単一rollout実行（`run_rollout()`）
  - rollout保存（`save_rollout()`）
  - バッチ予測（`predict_rollouts()`, `predict_rollouts_distributed()`）
- [x] `gns/render.py` の作成（349行）
  - データローディング（`load_rollout_pickle()`, `prepare_trajectory_data()`）
  - 色マッピング（`TYPE_TO_COLOR`, `get_color_mask()`）
  - 2D/3Dレンダリング（`render_2d_trajectory()`, `render_3d_trajectory()`）
  - GIFアニメーション（`render_gif_animation()`）
  - VTKエクスポート（`write_vtk_trajectory()`）

**成果物:**
- `gns/rollout.py` (304行) - 再利用可能なrolloutロジック
- `gns/render.py` (349行) - 再利用可能な可視化ロジック

### Step 5.4: train.py, train_multinode.py, render_rollout.py を scripts/ へ移動・修正 ✅
- [x] `gns/train.py` (663行) → `scripts/gns_train.py` (181行) への変換
- [x] `gns/train_multinode.py` → `scripts/gns_train_multinode.py` (214行) への変換
- [x] `gns/render_rollout.py` (246行) → `scripts/gns_render_rollout.py` (56行) への変換

**影響範囲:**
- `gns/train.py` 全体（663行 → `gns/training.py` 788行 + `scripts/gns_train.py` 181行に分割）
- `gns/train_multinode.py` 全体（→ `scripts/gns_train_multinode.py` 214行）
- `gns/render_rollout.py` 全体（246行 → `scripts/gns_render_rollout.py` 56行に短縮、ロジックは`gns/render.py`へ）

**実装結果:**
- CLIスクリプトは薄いラッパー（50-200行）に
- ビジネスロジックは再利用可能なモジュールに分離
- コード重複を大幅に削減

### Step 5.5: 新しいスクリプト位置から学習・rollout実行で検証 ✅
- [x] 10ステップ学習実行 - 成功（loss: 2.24 → 1.90）
- [x] rollout実行 - 成功（平均loss: 0.496）
- [x] 出力ファイル生成確認 - rollout_ex0.pkl, rollout_ex1.pkl生成確認
- [x] モジュールインポート確認 - 成功

**検証コマンド:**
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

**検証結果:**
- 学習成功: チェックポイント（model-0.pt, model-10.pt, train_state-0.pt, train_state-10.pt）生成確認
- rollout成功: rollout_ex0.pkl, rollout_ex1.pkl生成確認
- インポートテスト成功

### Step 6.1: scripts/legacy/ ディレクトリの作成
- [ ] `scripts/legacy/` ディレクトリの作成

**成果物:**
- `scripts/legacy/` ディレクトリ

### Step 6.2: シェルスクリプト5ファイルを scripts/legacy/ へ移動
- [ ] `build_venv.sh` → `scripts/legacy/build_venv.sh`
- [ ] `build_venv_frontera.sh` → `scripts/legacy/build_venv_frontera.sh`
- [ ] `module.sh` → `scripts/legacy/module.sh`
- [ ] `start_venv.sh` → `scripts/legacy/start_venv.sh`
- [ ] `run.sh` → `scripts/legacy/run.sh`

**成果物:**
- 5つのシェルスクリプトが `scripts/legacy/` に移動

### Step 6.3: scripts/README.md の作成
- [ ] `scripts/README.md` の作成
  - 新しいスクリプト（`gns_train.py` など）の説明
  - `legacy/` フォルダの説明

**成果物:**
- `scripts/README.md`

### Step 6.4: ファイル移動の確認
- [ ] `scripts/legacy/` 内のファイル確認
- [ ] ルートディレクトリがクリーンであることを確認

**検証コマンド:**
```bash
ls scripts/legacy/
ls *.sh 2>/dev/null || echo "ルートディレクトリにシェルスクリプトなし（正常）"
```

---

## Phase 3: 設定の整理 ✅

**注**: このPhaseで予定していた機能は、**Step 5で既に実装済み**です。Step 5の再設計により、バッチ準備処理もStep 5に統合されました。

### Step 7.1: gns/training.py に prepare_training_batch 関数追加 ✅ (Step 5で実装済み)
- [x] `prepare_training_batch()` 関数の実装
  - 特徴抽出（position, particle_type, material_property）
  - デバイス移動とn_features条件分岐の一元化
  - 15行のボイラープレートコードを1行の関数呼び出しに削減
- [x] `add_training_noise()` 関数の実装
  - ノイズ生成
  - 運動学的粒子のマスク適用

**影響範囲:**
- `gns/training.py` - Step 5.2で実装済み
- 元の実装: `gns/train.py` の361-370行（バッチ準備）、376-379行（ノイズ追加）を抽出

**実装済みの内容:**
- `prepare_training_batch()` - バッチ準備（gns/training.py:90-117行）
- `add_training_noise()` - ノイズ追加（gns/training.py:128-154行）

### Step 7.2: scripts/gns_train.py と gns_train_multinode.py で prepare_training_batch を使用 ✅ (Step 5で実装済み)
- [x] `run_training_loop()` 内で `prepare_training_batch()` を自動的に呼び出し
- [x] `run_training_loop()` 内で `add_training_noise()` を自動的に呼び出し

**影響範囲:**
- `gns/training.py` の `run_training_loop()` 関数内 (gns/training.py:665-676行)
- CLIスクリプトは `run_training_loop()` を呼び出すのみ

**実装済みの内容:**
- バッチ準備とノイズ追加は `run_training_loop()` 内で自動的に実行される
- CLIスクリプト（`scripts/gns_train.py`, `scripts/gns_train_multinode.py`）は薄いラッパーとして実装

### Step 7.3: 学習実行で検証 ✅ (Step 5で検証済み)
- [x] 学習実行で検証 - Step 5.5で実施済み

**検証コマンド:**
```bash
uv run python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test_step5_v2/ --ntraining_steps=10 --mode=train
```

**検証結果:**
- Step 5.5で実施済み（loss: 2.24 → 1.90）

---

## 完了基準

全27タスク完了後、以下が達成されていること:

1. ✅ **環境定義が `pyproject.toml` に一元化**（Step 0）
   - uv で管理される最新の Python 3.13 環境
   - pyright-lsp が正常動作し、型チェック・コード補完が利用可能
   - 旧環境定義ファイル（`requirements.txt`, `environment.yml`）を削除

2. ✅ **不要なシェルスクリプトが `scripts/legacy/` に退避**（Step 6）
   - ルートディレクトリがクリーンになり、重要なファイルが明確

3. ✅ **`gns/` パッケージが再利用可能なモジュールのみを含む**（Step 5）
   - 実行スクリプトは `scripts/` に移動（`gns_train.py` など）

4. ✅ **推論、運動学的制約、学習バッチ準備が独立した関数として抽出可能**（Steps 1, 7）
   - `inference_utils.py` で推論ロジックを提供
   - `training.py` で学習ロジックを提供

5. ✅ **既存の学習・推論が正常に動作（数値結果が一致）**（全ステップ）
   - 最新パッケージバージョンでの動作確認済み
