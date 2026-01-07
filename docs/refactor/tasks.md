# GNS リファクタリングタスクリスト

このドキュメントは、[リファクタリング実装プラン](./plan.ja.md)の実施状況を管理するためのタスクリストです。

## 進捗状況

- [x] Phase 0: 環境構築 (4/4) ✅
- [ ] Phase 1: アーキテクチャ分離 (3/13)
- [ ] Phase 2: モジュール/スクリプト明確化 (0/9)
- [ ] Phase 3: 設定の整理 (0/3)

**全体進捗: 7/29 タスク完了**

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

### Step 2.1: gns/graph_model.py の作成（GraphNeuralNetworkModel クラス）
- [ ] `gns/graph_model.py` の作成
- [ ] `GraphNeuralNetworkModel` クラスの実装
  - グラフ構築
  - メッセージパッシング
  - forward pass

**成果物:**
- `gns/graph_model.py`

### Step 2.2: gns/learned_simulator.py を GraphNeuralNetworkModel をラップする形にリファクタリング
- [ ] `LearnedSimulator` クラスを `GraphNeuralNetworkModel` をラップする形にリファクタリング
- [ ] 正規化、オイラー積分、位置/加速度予測のメソッドを維持
- [ ] `save()`, `load()` メソッドを維持

**影響範囲:**
- `gns/learned_simulator.py` 全体（388行）

### Step 2.3: インポート確認と学習・rollout実行で検証
- [ ] モジュールのインポート確認
- [ ] 学習とrolloutの実行で検証

**検証コマンド:**
```bash
python -c "from gns.learned_simulator import LearnedSimulator; \
from gns.graph_model import GraphNeuralNetworkModel; \
print('Import successful')"

python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

### Step 3.1: gns/config.py の作成（SimulatorConfig dataclass）
- [ ] `gns/config.py` の作成
- [ ] `SimulatorConfig` dataclass の実装
- [ ] `from_metadata()` クラスメソッドの実装

**成果物:**
- `gns/config.py`

### Step 3.2: gns/train.py と train_multinode.py のグローバル定数を削除し config 引数に置き換え
- [ ] `gns/train.py` のグローバル定数（52-54行）を削除
- [ ] `gns/train_multinode.py` のグローバル定数を削除
- [ ] 各関数で `config` を引数として受け取るように修正

**影響範囲:**
- `gns/train.py`, `gns/train_multinode.py` の定数参照箇所すべて

### Step 3.3: 学習実行で検証
- [ ] 設定オブジェクトの作成確認
- [ ] 学習実行で検証

**検証コマンド:**
```bash
python -c "from gns.config import SimulatorConfig; \
cfg = SimulatorConfig(); print(cfg)"

python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

### Step 4.1: gns/config.py に build_normalization_stats と infer_feature_dimensions 関数追加
- [ ] `build_normalization_stats()` 関数の実装
- [ ] `infer_feature_dimensions()` 関数の実装

**影響範囲:**
- `gns/config.py`

### Step 4.2: gns/train.py の _get_simulator() 関数を分割
- [ ] `_get_simulator()` 関数（503-558行）を分割
  - 設定読み込み部分 → `config.py` の関数群
  - シミュレータ生成部分 → `create_simulator()` 関数

**影響範囲:**
- `gns/train.py` の `_get_simulator()` 関数

### Step 4.3: 学習実行で検証
- [ ] 学習実行で検証

**検証コマンド:**
```bash
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

---

## Phase 2: モジュール/スクリプト明確化

### Step 5.1: scripts/ ディレクトリの作成
- [ ] `scripts/` ディレクトリの作成

**成果物:**
- `scripts/` ディレクトリ

### Step 5.2: gns/training.py の作成（train_step, validation_step, checkpoint管理）
- [ ] `gns/training.py` の作成
- [ ] `train_step()` 関数の実装
- [ ] `validation_step()` 関数の実装
- [ ] `save_checkpoint()`, `load_checkpoint()` 関数の実装

**成果物:**
- `gns/training.py`

### Step 5.3: gns/rollout.py の作成（run_rollout, save_rollout）
- [ ] `gns/rollout.py` の作成
- [ ] `run_rollout()` 関数の実装
- [ ] `save_rollout()` 関数の実装

**成果物:**
- `gns/rollout.py`

### Step 5.4: train.py, train_multinode.py, render_rollout.py を scripts/ へ移動・修正
- [ ] `gns/train.py` → `scripts/gns_train.py` への移動・修正
- [ ] `gns/train_multinode.py` → `scripts/gns_train_multinode.py` への移動・修正
- [ ] `gns/render_rollout.py` → `scripts/gns_render_rollout.py` への移動

**影響範囲:**
- `gns/train.py` 全体（658行を分割）
- `gns/train_multinode.py` 全体
- `gns/render_rollout.py`

### Step 5.5: 新しいスクリプト位置から学習・rollout実行で検証
- [ ] 新しいスクリプト位置から学習実行
- [ ] rolloutの実行
- [ ] モジュールインポート確認

**検証コマンド:**
```bash
python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

python -c "from gns.training import train_step; \
from gns.rollout import run_rollout; print('Import successful')"
```

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

## Phase 3: 設定の整理

### Step 7.1: gns/training.py に prepare_training_batch 関数追加
- [ ] `prepare_training_batch()` 関数の実装
  - 特徴抽出（position, particle_type, material_property）
  - ノイズ生成
  - 運動学的粒子のマスク適用

**影響範囲:**
- `gns/training.py`
- 元の実装: `gns/train.py` の387-419行

### Step 7.2: scripts/gns_train.py と gns_train_multinode.py で prepare_training_batch を使用
- [ ] `scripts/gns_train.py` の学習ループで `prepare_training_batch()` を呼び出す
- [ ] `scripts/gns_train_multinode.py` の学習ループで `prepare_training_batch()` を呼び出す

**影響範囲:**
- `scripts/gns_train.py` の学習ループ
- `scripts/gns_train_multinode.py` の学習ループ

### Step 7.3: 学習実行で検証
- [ ] 学習実行で検証

**検証コマンド:**
```bash
python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

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
