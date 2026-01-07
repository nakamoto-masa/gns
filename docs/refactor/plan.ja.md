# GNS リファクタリング実装プラン

## 目的

利用者としてモデルを使いやすくすることを目的とした、最小限のリファクタリングを行う。

## 対象範囲

- **対象**: GNSモデルのコード品質上の問題点
- **対象外**: MeshNet、保守性のための設計変更

## ユーザー目線の課題（解決すべき問題）

1. **環境定義が複数のファイルに書かれていて、内容が一致していない**
   - `requirements.txt` と `environment.yml` の不整合

2. **位置付けの不明なシェルスクリプトがリポジトリルートに散在**
   - `run.sh`, `build_venv.sh`, `start_venv.sh`, `module.sh` など

3. **モジュールとスクリプトの区別があいまい**
   - `gns/train.py`, `gns/train_multinode.py` はスクリプトだがパッケージ内に配置

4. **一部の機能だけを抜き出して使うのが難しい**
   - `rollout()` 関数が推論・運動学的制約・損失計算を混在
   - `LearnedSimulator` クラスがNNモデル・物理統合・正規化を混在
   - グローバル定数でデータセット固有のパラメータをハードコード

## リファクタリングの制約

- アルゴリズムの挙動は変更しない
- 変更は段階的に行い、各段階で動作を維持する
- 責務ごとの分離を優先
- モジュールとスクリプトの明確化

## 実装ステップ（9段階）

### Phase 0: 環境構築（Step 0）

リファクタリング作業を効率的に行うため、最初に最新のPython環境を構築する。

---

### Step 0: uv による Python 環境構築と依存関係の最新化

**目的**: pyright-lsp による型チェック・コード補完を有効にし、リファクタリング作業を効率化する。

**変更内容**:

1. **新規ファイル作成**: `pyproject.toml`
   - プロジェクトメタデータと依存関係を定義
   - 既存の `requirements.txt` と `enviornment.yml` から依存関係を抽出
   - パッケージバージョンを最新の安定版に更新

   主要な依存関係:
   - Python 3.13 (最新安定版、3.9からアップグレード)
     - 根拠: 2024年10月リリース、主要ライブラリが対応済み
   - PyTorch 2.6+ (最新安定版、1.12からアップグレード)
     - PyTorch 2.6から Python 3.13 をサポート
   - torch-geometric 2.6+ (PyTorch 2.6+ 対応版)
     - Python 3.10-3.13 をサポート
   - numpy 2.1+ (バージョン固定を削除し最新へ)
     - numpy 2.1.0+ で Python 3.13 をサポート
   - その他: absl-py, matplotlib, pytest, tqdm など

2. **新規ファイル作成**: `.python-version`
   - uv で使用する Python バージョンを指定（3.13）

3. **新規スクリプト作成**: `scripts/setup/setup_uv_environment.sh`
   - uv のインストール
   - Python 3.13 環境の作成
   - 依存関係のインストール
   - 実行例:
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

4. **既存ファイル保持**: `requirements.txt`, `enviornment.yml`
   - 後方互換性のため残す
   - ファイル先頭に "DEPRECATED: Use pyproject.toml instead" のコメント追加

5. **`.gitignore` 更新**:
   - `.venv/` を追加（uv の仮想環境）
   - `uv.lock` を追加（依存関係のロックファイル）

**解決する課題**:
- pyright-lsp が正常に動作し、リファクタリング中に型チェック・コード補完が使える
- 最新のパッケージバージョンで潜在的なバグ修正やパフォーマンス向上の恩恵を受ける
- 課題1（環境定義の一元化）を先行して解決

**影響範囲**:
- 新規ファイル: `pyproject.toml`, `.python-version`, `scripts/setup/setup_uv_environment.sh`
- 修正ファイル: `requirements.txt`, `enviornment.yml` (deprecation コメント追加)
- `.gitignore`

**検証方法**:
```bash
# uv 環境のセットアップ
bash scripts/setup/setup_uv_environment.sh

# 仮想環境の有効化
source .venv/bin/activate

# 依存関係の確認
uv pip list

# 簡単な動作確認
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print('OK')"

# 既存のテストが通ることを確認
pytest test/test_pytorch.py
pytest test/test_torch_geometric.py

# 小規模学習の実行（Step 1以降で使用するコマンドと同じ）
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

**pyproject.toml の例**:
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

**規模**: 小（環境構築のみ、コードは変更なし）

**注意事項**:
- PyTorch 2.x では API に破壊的変更がある可能性があるため、動作確認が重要
- torch-geometric も PyTorch のバージョンに依存するため、互換性を確認
- numpy 2.x は一部の API が変更されているため、必要に応じてコード修正
- Python 3.13 は JIT コンパイラが導入され、パフォーマンス向上が期待できる

---

### Phase 1: アーキテクチャ分離（Steps 1-4）

機能の抽出を可能にするため、`rollout()` と `LearnedSimulator` の責務を分離する。

---

### Step 1: rollout() 関数の分離

**目的**: 推論ループ、運動学的制約適用、損失計算を分離し、各機能を独立して使えるようにする。

**変更内容**:

1. **新規ファイル作成**: `gns/inference_utils.py`
   - `run_inference_loop()` - 純粋な推論実行（simulator呼び出しのみ）
   - `apply_kinematic_constraints()` - 運動学的制約の適用（KINEMATIC粒子の位置を真値で上書き）

2. **既存ファイル修正**: `gns/train.py`
   - `rollout()` 関数を3つの関数呼び出しに分離
     - `inference_utils.run_inference_loop()` で推論実行
     - `inference_utils.apply_kinematic_constraints()` で制約適用
     - ローカルで損失計算（`compute_rollout_loss()` は将来的に抽出可能）

**解決する課題**: 課題4 - 推論ループだけを抽出して使えるようになる

**影響範囲**:
- `gns/train.py` の `rollout()` 関数（56-119行）
- `gns/train_multinode.py` の同等部分（同様の変更が必要）

**検証方法**:
```bash
# 10ステップの学習を実行
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# rolloutを実行
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --model_file=model-10.pt \
  --mode=rollout --output_path=rollouts/test/

# 出力ファイルが生成されることを確認
ls rollouts/test/
```

**規模**: 中（コア機能だが、抽出は比較的明確）

---

### Step 2: LearnedSimulator クラスの分離

**目的**: NNモデル本体と物理統合を分離し、用途に応じて使い分けられるようにする。

**変更内容**:

1. **既存ファイル修正**: `gns/learned_simulator.py`
   - `LearnedSimulator` を `GraphNeuralNetworkModel` にリネーム
   - 責務: グラフ構築、メッセージパッシング、forward pass
   - 削除: 正規化統計、境界条件（PhysicsSimulatorへ移動）

2. **新規ファイル作成**: `gns/simulator.py`
   - `PhysicsSimulator` クラス
   - `GraphNeuralNetworkModel` をラップ
   - 責務: 正規化、オイラー積分、位置/加速度予測
   - メソッド: `predict_positions()`, `predict_accelerations()`, `save()`, `load()`

3. **既存ファイル修正**: `gns/train.py`, `gns/train_multinode.py`
   - `from gns.learned_simulator import LearnedSimulator`
     → `from gns.simulator import PhysicsSimulator`

**解決する課題**: 課題4 - GNNモデルだけを使いたいユーザーは `GraphNeuralNetworkModel` を、物理予測が必要なユーザーは `PhysicsSimulator` を使える

**影響範囲**:
- `gns/learned_simulator.py` 全体（388行）
- `gns/train.py`, `gns/train_multinode.py` のimport文と`_get_simulator()` 関数

**検証方法**:
```bash
# Step 1と同じ学習・rolloutコマンドで検証
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# モジュールのインポート確認
python -c "from gns.simulator import PhysicsSimulator; \
from gns.learned_simulator import GraphNeuralNetworkModel; \
print('Import successful')"
```

**規模**: 大（388行のクラスを分割、多数のインポート文を修正）

---

### Step 3: グローバル定数の設定クラス化

**目的**: データセット固有のパラメータをハードコードではなく設定可能にする。

**変更内容**:

1. **新規ファイル作成**: `gns/config.py`
   - `SimulatorConfig` dataclass
   - フィールド: `input_sequence_length`, `num_particle_types`, `kinematic_particle_id` など
   - `from_metadata()` クラスメソッドでmetadataから生成

2. **既存ファイル修正**: `gns/train.py`, `gns/train_multinode.py`
   - グローバル定数（52-54行）を削除
     - `INPUT_SEQUENCE_LENGTH = 6` → `config.input_sequence_length`
     - `NUM_PARTICLE_TYPES = 9` → `config.num_particle_types`
     - `KINEMATIC_PARTICLE_ID = 3` → `config.kinematic_particle_id`
   - 各関数で `config` を引数として受け取る

**解決する課題**: 課題4 - 異なるパラメータのデータセットに対応可能

**影響範囲**:
- `gns/train.py`, `gns/train_multinode.py` の定数参照箇所すべて

**検証方法**:
```bash
# 設定オブジェクトの作成確認
python -c "from gns.config import SimulatorConfig; \
cfg = SimulatorConfig(); print(cfg)"

# 学習実行で検証
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train
```

**規模**: 小（明確な抽出、リスク低）

---

### Step 4: 設定読み込みとシミュレータ生成の分離

**目的**: 設定の解釈とオブジェクト生成を分離し、カスタム設定でのシミュレータ作成を容易にする。

**変更内容**:

1. **既存ファイル修正**: `gns/config.py`
   - `build_normalization_stats()` 関数追加 - metadata解釈
   - `infer_feature_dimensions()` 関数追加 - 次元推論

2. **既存ファイル修正**: `gns/train.py`
   - `_get_simulator()` 関数（503-558行）を分割
     - 設定読み込み部分 → `config.py` の関数群
     - シミュレータ生成部分 → `create_simulator()` 関数

**解決する課題**: 課題4 - カスタム設定でシミュレータを作成可能

**影響範囲**:
- `gns/train.py` の `_get_simulator()` 関数

**検証方法**: Step 3と同様

**規模**: 小（既存関数のリファクタリング）

---

### Phase 2: モジュール/スクリプト明確化（Steps 5-6）

gns/ パッケージを再利用可能なモジュールのみに限定し、実行スクリプトを分離する。

---

### Step 5: 実行スクリプトの gns/ パッケージ外への移動

**目的**: インポート可能なモジュールと実行スクリプトを明確に分離する。

**変更内容**:

1. **新規ディレクトリ作成**: `scripts/`

2. **新規ファイル作成**: `gns/training.py`, `gns/rollout.py`
   - `gns/training.py` - 再利用可能な学習ロジック
     - `train_step()` - 1ステップの学習
     - `validation_step()` - 1ステップの検証
     - `save_checkpoint()`, `load_checkpoint()` - チェックポイント管理
     - `prepare_training_batch()` - バッチ準備（Step 8で実装）
   - `gns/rollout.py` - 再利用可能なrolloutロジック
     - `run_rollout()` - rollout実行
     - `save_rollout()` - 結果保存

3. **ファイル移動**:
   - `gns/train.py` → `scripts/train.py`（CLI部分のみ、ロジックは `gns/training.py` へ）
   - `gns/train_multinode.py` → `scripts/train_multinode.py`（同様）
   - `gns/render_rollout.py` → `scripts/render_rollout.py`

4. **既存ファイル修正**: `scripts/train.py`
   - FLAGSの解析とCLIエントリポイントのみ保持
   - 実際のロジックは `gns.training` の関数を呼び出す

**解決する課題**: 課題3 - `gns/` = インポート可能なモジュール、`scripts/` = 実行スクリプト

**影響範囲**:
- `gns/train.py` 全体（658行を分割）
- `gns/train_multinode.py` 全体
- `gns/render_rollout.py`

**検証方法**:
```bash
# 新しいスクリプト位置から実行
python scripts/train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# モジュールインポート確認
python -c "from gns.training import train_step; \
from gns.rollout import run_rollout; print('Import successful')"
```

**規模**: 中（大規模ファイルの分割だが、移動が主体）

---

### Step 6: ルートディレクトリのシェルスクリプト整理

**目的**: シェルスクリプトを目的別に整理し、わかりやすい名前を付ける。

**変更内容**:

1. **新規ディレクトリ作成**:
   - `scripts/setup/` - 環境構築用
   - `scripts/examples/` - 実行例

2. **ファイル移動とリネーム**:
   ```
   build_venv.sh → scripts/setup/create_environment.sh
   build_venv_frontera.sh → scripts/setup/create_environment_frontera.sh
   module.sh → scripts/setup/load_modules.sh
   start_venv.sh → scripts/setup/activate_environment.sh
   run.sh → scripts/examples/train_water_drop.sh
   ```

3. **新規ファイル作成**: `scripts/README.md`
   - 各スクリプトの目的と使い方を説明

**解決する課題**: 課題2 - スクリプトの目的が明確になり、整理される

**影響範囲**:
- ルートディレクトリのシェルスクリプト5ファイル

**検証方法**:
```bash
# 環境構築スクリプトのテスト
bash scripts/setup/create_environment.sh
source scripts/setup/activate_environment.sh

# サンプル実行スクリプトのテスト
bash scripts/examples/train_water_drop.sh
```

**規模**: 小（ファイル移動とドキュメント作成）

---

### Phase 3: 環境・設定の統一（Steps 7-8）

環境定義を統一し、設定の扱いを整理する。

---

### Step 7: 環境定義ファイルの統一

**目的**: 依存関係の定義を一元化し、不整合をなくす。

**変更内容**:

1. **既存ファイル修正**: `environment.yml`
   - ファイル名のtypo修正（`enviornment.yml` → `environment.yml`）
   - ファイル先頭にコメント追加: "Use requirements.txt as the source of truth"

2. **既存ファイル維持**: `requirements.txt`
   - 依存関係の唯一の真実の情報源として維持
   - CI（`.circleci/config.yml`）が既に使用中

3. **新規ファイル作成**: `scripts/setup/create_conda_environment.sh`
   - `requirements.txt` を使ってconda環境を作成するスクリプト

4. **ドキュメント更新**: `README.md`（存在する場合）
   - どの環境定義ファイルを使うべきかを明記

**解決する課題**: 課題1 - 環境定義の一元化

**影響範囲**:
- `environment.yml` (typo修正とコメント追加)
- `scripts/setup/` の新規スクリプト

**検証方法**:
```bash
# pip インストールのテスト
pip install -r requirements.txt
python -c "import torch; import torch_geometric; print('OK')"

# conda環境作成のテスト
bash scripts/setup/create_conda_environment.sh
conda activate gns
python -c "import torch; import torch_geometric; print('OK')"
```

**規模**: 小（主にドキュメントとスクリプト追加）

---

### Step 8: 学習ループ内の特徴抽出処理の関数化

**目的**: 学習バッチの準備処理を再利用可能にする。

**変更内容**:

1. **既存ファイル修正**: `gns/training.py`（Step 5で作成）
   - `prepare_training_batch()` 関数追加
   - 責務: データローダーから取得したexampleを処理
     - 特徴抽出（position, particle_type, material_property）
     - ノイズ生成
     - 運動学的粒子のマスク適用
   - 現在 `train.py` の387-419行にあるインライン処理を抽出

2. **既存ファイル修正**: `scripts/train.py`, `scripts/train_multinode.py`
   - 学習ループ内で `prepare_training_batch()` を呼び出す

**解決する課題**: 課題4 - 学習バッチの準備ロジックを再利用可能にする

**影響範囲**:
- `gns/train.py` の学習ループ（387-419行）
- `gns/train_multinode.py` の同等部分

**検証方法**: Step 5と同様の学習実行で検証

**規模**: 小（明確な抽出）

---

## 実装順序と依存関係

```
Phase 0: 環境構築
  Step 0 (uv環境構築) - 最優先、他のステップの前提条件
    ↓

Phase 1: アーキテクチャ分離
  Step 1 (rollout分離) - Step 0完了後
    ↓
  Step 2 (LearnedSimulator分離) - Step 1に依存
    ↓
  Step 3 (グローバル定数) - Step 0完了後（Step 1-2と独立）
    ↓
  Step 4 (設定読み込み分離) - Step 3に依存

Phase 2: モジュール/スクリプト明確化
  Step 5 (スクリプト移動) - Steps 1-4に依存
    ↓
  Step 6 (シェルスクリプト整理) - Step 0完了後（Step 5と並行可能）

Phase 3: 環境・設定統一（Step 0で先行実施済み）
  Step 7 (環境定義統一) - Step 0で大部分を実施済み、残作業のみ
    ↓
  Step 8 (バッチ準備関数化) - Step 5に依存
```

**重要**: Step 0（uv環境構築）は最初に必ず実施する。これにより：
- リファクタリング中に pyright-lsp が正常動作
- 型チェック・コード補完が利用可能
- 最新パッケージでの検証が可能

## 最終的なディレクトリ構造

```
/Users/masahiro/repos/gns/wt-cleanup/
├── gns/                          # 再利用可能なモジュールのみ
│   ├── config.py                 # NEW - Step 3
│   ├── inference_utils.py        # NEW - Step 1
│   ├── simulator.py              # NEW - Step 2
│   ├── learned_simulator.py      # MODIFIED - Step 2
│   ├── training.py               # NEW - Step 5, 8
│   ├── rollout.py                # NEW - Step 5
│   ├── graph_network.py          # 既存（変更なし）
│   ├── data_loader.py            # 既存（変更なし）
│   ├── reading_utils.py          # 既存（変更なし）
│   └── noise_utils.py            # 既存（変更なし）
│
├── scripts/                      # 実行スクリプト
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
├── pyproject.toml                # NEW - Step 0（依存関係の真実の情報源）
├── .python-version               # NEW - Step 0
├── .venv/                        # NEW - Step 0（gitignoreに追加）
├── uv.lock                       # NEW - Step 0（gitignoreに追加）
├── requirements.txt              # DEPRECATED - Step 0（後方互換性のため保持）
├── enviornment.yml               # DEPRECATED - Step 0（後方互換性のため保持）
├── slurm_scripts/                # 既存（変更なし）
├── test/                         # 既存（変更なし）
└── ...
```

## 重要な変更ファイル一覧

### 変更が必須のファイル

1. **gns/train.py** (658行)
   - Steps 1, 3, 4, 5で段階的に変更
   - 最終的に `scripts/train.py` に移動

2. **gns/learned_simulator.py** (388行)
   - Step 2で `GraphNeuralNetworkModel` にリネーム・分割

3. **gns/train_multinode.py** (~25KB)
   - `gns/train.py` と同様の変更が必要
   - 最終的に `scripts/train_multinode.py` に移動

4. **requirements.txt**
   - Step 7で依存関係の真実の情報源として位置づけ

5. **gns/reading_utils.py**
   - Step 3, 4で `config.py` から利用される（変更は不要）

### 新規作成ファイル

1. **gns/inference_utils.py** - Step 1
2. **gns/simulator.py** - Step 2
3. **gns/config.py** - Step 3
4. **gns/training.py** - Step 5
5. **gns/rollout.py** - Step 5
6. **scripts/README.md** - Step 6
7. **scripts/setup/create_conda_environment.sh** - Step 7

## 検証戦略

各ステップ完了後に以下を実施:

1. **小規模学習の実行**（10ステップ）
   ```bash
   python scripts/train.py --data_path=example/WaterDropSample/ \
     --model_path=models/test/ --ntraining_steps=10 --mode=train
   ```

2. **Rolloutの実行**
   ```bash
   python scripts/train.py --data_path=example/WaterDropSample/ \
     --model_path=models/test/ --model_file=model-10.pt \
     --mode=rollout --output_path=rollouts/test/
   ```

3. **出力の確認**
   - rollout pickle ファイルが生成される
   - 数値結果をベースライン（リファクタリング前）と比較

4. **モジュールのインポート確認**
   - 各ステップで追加されたモジュールがインポート可能

## リスク管理

- **ブランチ戦略**: 現在の `refactor/gns-cleanup` ブランチで継続
- **各ステップの独立性**: 各ステップは個別にコミット可能
- **ロールバック**: 検証失敗時は該当コミットをrevert
- **数値検証**: 推論結果が既存実装と一致することを確認（torch.allclose）

## 想定される課題と対策

### 課題1: train.py と train_multinode.py の重複コード

**対策**: 両ファイルに同じ変更を適用し、将来的な統合の基盤を作る（今回の範囲では統合まではしない）

### 課題2: 既存のSLURMスクリプトとの互換性

**対策**: `slurm_scripts/` は今回変更せず、`scripts/train.py` が元の `gns/train.py` と同じインターフェースを維持

### 課題3: テストコードの不足

**対策**: 実際のデータセットでの学習・推論実行により動作を検証

## 完了基準

全9ステップ完了後、以下が達成されていること:

1. ✅ **環境定義が `pyproject.toml` に一元化**（Step 0）
   - uv で管理される最新の Python 3.13 環境
   - pyright-lsp が正常動作し、型チェック・コード補完が利用可能

2. ✅ **シェルスクリプトが `scripts/setup/`, `scripts/examples/` に整理**（Step 6）
   - 目的別に整理され、わかりやすい命名

3. ✅ **`gns/` パッケージが再利用可能なモジュールのみを含む**（Step 5）
   - 実行スクリプトは `scripts/` に移動

4. ✅ **推論、運動学的制約、学習バッチ準備が独立した関数として抽出可能**（Steps 1, 8）
   - `inference_utils.py` で推論ロジックを提供
   - `training.py` で学習ロジックを提供

5. ✅ **既存の学習・推論が正常に動作（数値結果が一致）**（全ステップ）
   - 最新パッケージバージョンでの動作確認済み

この状態で、ユーザーは:
- ✅ 必要な機能だけをインポートして使える
- ✅ モジュールとスクリプトの区別が明確
- ✅ 環境構築の手順が明確（`bash scripts/setup/setup_uv_environment.sh` で完結）
- ✅ スクリプトの目的が明確
- ✅ 最新の Python エコシステムの恩恵を受けられる（Python 3.13, PyTorch 2.6+, numpy 2.1+）
- ✅ リファクタリング中も型チェック・補完が機能し、効率的に作業できる
- ✅ Python 3.13 の JIT コンパイラによるパフォーマンス向上の恩恵を受けられる

を実現できる。
