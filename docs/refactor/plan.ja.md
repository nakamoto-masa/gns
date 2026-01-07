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

## 実装ステップ（7段階）

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

3. **既存ファイル削除**: `requirements.txt`, `enviornment.yml`
   - `pyproject.toml` に一元化するため削除

**解決する課題**:
- pyright-lsp が正常に動作し、リファクタリング中に型チェック・コード補完が使える
- 最新のパッケージバージョンで潜在的なバグ修正やパフォーマンス向上の恩恵を受ける
- 課題1（環境定義の一元化）を先行して解決

**影響範囲**:
- 新規ファイル: `pyproject.toml`, `.python-version`
- 削除ファイル: `requirements.txt`, `enviornment.yml`

**検証方法**:
```bash
# uv 環境のセットアップ
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .

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

1. **新規ファイル作成**: `gns/graph_model.py`
   - `GraphNeuralNetworkModel` クラスを新規作成
   - 責務: グラフ構築、メッセージパッシング、forward pass
   - 純粋なGNNモデルのみを提供

2. **既存ファイル修正**: `gns/learned_simulator.py`
   - `LearnedSimulator` クラス名はそのまま維持
   - `GraphNeuralNetworkModel` をラップする形にリファクタリング
   - 責務: 正規化、オイラー積分、位置/加速度予測
   - メソッド: `predict_positions()`, `predict_accelerations()`, `save()`, `load()`

3. **既存ファイル修正**: `gns/train.py`, `gns/train_multinode.py`
   - `from gns.learned_simulator import LearnedSimulator` はそのまま維持（変更不要）

**解決する課題**: 課題4 - GNNモデルだけを使いたいユーザーは `GraphNeuralNetworkModel` を、物理予測が必要なユーザーは `LearnedSimulator` を使える

**影響範囲**:
- `gns/learned_simulator.py` 全体（388行）
- 新規ファイル: `gns/graph_model.py`

**検証方法**:
```bash
# Step 1と同じ学習・rolloutコマンドで検証
python -m gns.train --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# モジュールのインポート確認
python -c "from gns.learned_simulator import LearnedSimulator; \
from gns.graph_model import GraphNeuralNetworkModel; \
print('Import successful')"
```

**規模**: 大（388行のクラスを分割、既存コードへの影響は最小限）

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
     - `prepare_training_batch()` - バッチ準備（Step 7で実装）
   - `gns/rollout.py` - 再利用可能なrolloutロジック
     - `run_rollout()` - rollout実行
     - `save_rollout()` - 結果保存

3. **ファイル移動**:
   - `gns/train.py` → `scripts/gns_train.py`（CLI部分のみ、ロジックは `gns/training.py` へ）
   - `gns/train_multinode.py` → `scripts/gns_train_multinode.py`（同様）
   - `gns/render_rollout.py` → `scripts/gns_render_rollout.py`

4. **既存ファイル修正**: `scripts/gns_train.py`
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
python scripts/gns_train.py --data_path=example/WaterDropSample/ \
  --model_path=models/test/ --ntraining_steps=10 --mode=train

# モジュールインポート確認
python -c "from gns.training import train_step; \
from gns.rollout import run_rollout; print('Import successful')"
```

**規模**: 中（大規模ファイルの分割だが、移動が主体）

---

### Step 6: ルートディレクトリのシェルスクリプト退避

**目的**: 不要なシェルスクリプトをルートディレクトリから整理する。

**変更内容**:

1. **新規ディレクトリ作成**:
   - `scripts/legacy/` - 旧スクリプト退避用

2. **ファイル移動**:
   - 以下のスクリプトを `scripts/legacy/` に移動
   ```
   build_venv.sh → scripts/legacy/build_venv.sh
   build_venv_frontera.sh → scripts/legacy/build_venv_frontera.sh
   module.sh → scripts/legacy/module.sh
   start_venv.sh → scripts/legacy/start_venv.sh
   run.sh → scripts/legacy/run.sh
   ```

3. **新規ファイル作成**: `scripts/README.md`
   - 新しいスクリプト（`gns_train.py` など）の説明
   - `legacy/` フォルダには古い環境用スクリプトがあり、使用非推奨であることを明記

**解決する課題**: 課題2 - ルートディレクトリがクリーンになり、どのファイルが重要かが明確になる

**影響範囲**:
- ルートディレクトリのシェルスクリプト5ファイル

**検証方法**:
```bash
# ファイルが移動されたことを確認
ls scripts/legacy/

# ルートディレクトリがクリーンであることを確認
ls *.sh 2>/dev/null || echo "ルートディレクトリにシェルスクリプトなし（正常）"
```

**規模**: 小（ファイル移動とドキュメント作成）

---

### Phase 3: 設定の整理（Step 7）

学習バッチ準備処理を整理する。

---

### Step 7: 学習ループ内の特徴抽出処理の関数化

**目的**: 学習バッチの準備処理を再利用可能にする。

**変更内容**:

1. **既存ファイル修正**: `gns/training.py`（Step 5で作成）
   - `prepare_training_batch()` 関数追加
   - 責務: データローダーから取得したexampleを処理
     - 特徴抽出（position, particle_type, material_property）
     - ノイズ生成
     - 運動学的粒子のマスク適用
   - 現在 `train.py` の387-419行にあるインライン処理を抽出

2. **既存ファイル修正**: `scripts/gns_train.py`, `scripts/gns_train_multinode.py`
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

Phase 3: 設定の整理
  Step 7 (バッチ準備関数化) - Step 5に依存
```

**重要**: Step 0（uv環境構築）は最初に必ず実施する。これにより：
- リファクタリング中に pyright-lsp が正常動作
- 型チェック・コード補完が利用可能
- 最新パッケージでの検証が可能
- 課題1（環境定義の一元化）を完全に解決

## 最終的なディレクトリ構造

```
/Users/masahiro/repos/gns/wt-cleanup/
├── gns/                          # 再利用可能なモジュールのみ
│   ├── config.py                 # NEW - Step 3
│   ├── inference_utils.py        # NEW - Step 1
│   ├── graph_model.py            # NEW - Step 2
│   ├── learned_simulator.py      # MODIFIED - Step 2
│   ├── training.py               # NEW - Step 5, 7
│   ├── rollout.py                # NEW - Step 5
│   ├── graph_network.py          # 既存（変更なし）
│   ├── data_loader.py            # 既存（変更なし）
│   ├── reading_utils.py          # 既存（変更なし）
│   └── noise_utils.py            # 既存（変更なし）
│
├── scripts/                      # 実行スクリプト
│   ├── gns_train.py              # MOVED - Step 5
│   ├── gns_train_multinode.py    # MOVED - Step 5
│   ├── gns_render_rollout.py     # MOVED - Step 5
│   ├── README.md                 # NEW - Step 6（スクリプトの説明）
│   └── legacy/                   # MOVED - Step 6（旧スクリプト退避）
│       ├── build_venv.sh
│       ├── build_venv_frontera.sh
│       ├── module.sh
│       ├── start_venv.sh
│       └── run.sh
│
├── pyproject.toml                # NEW - Step 0（依存関係の唯一の情報源）
├── .python-version               # NEW - Step 0
├── slurm_scripts/                # 既存（変更なし）
├── test/                         # 既存（変更なし）
└── ...
```

## 重要な変更ファイル一覧

### 変更が必須のファイル

1. **gns/train.py** (658行)
   - Steps 1, 3, 4, 5で段階的に変更
   - 最終的に `scripts/gns_train.py` に移動

2. **gns/learned_simulator.py** (388行)
   - Step 2で内部をリファクタリング（`GraphNeuralNetworkModel`をラップ）

3. **gns/train_multinode.py** (~25KB)
   - `gns/train.py` と同様の変更が必要
   - 最終的に `scripts/gns_train_multinode.py` に移動

4. **gns/reading_utils.py**
   - Step 3, 4で `config.py` から利用される（変更は不要）

### 新規作成ファイル

1. **gns/inference_utils.py** - Step 1
2. **gns/graph_model.py** - Step 2
3. **gns/config.py** - Step 3
4. **gns/training.py** - Step 5
5. **gns/rollout.py** - Step 5
6. **scripts/README.md** - Step 6

## 検証戦略

各ステップ完了後に以下を実施:

1. **小規模学習の実行**（10ステップ）
   ```bash
   python scripts/gns_train.py --data_path=example/WaterDropSample/ \
     --model_path=models/test/ --ntraining_steps=10 --mode=train
   ```

2. **Rolloutの実行**
   ```bash
   python scripts/gns_train.py --data_path=example/WaterDropSample/ \
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

**対策**: `slurm_scripts/` は今回変更せず、`scripts/gns_train.py` が元の `gns/train.py` と同じインターフェースを維持

### 課題3: テストコードの不足

**対策**: 実際のデータセットでの学習・推論実行により動作を検証

## 完了基準

全7ステップ完了後、以下が達成されていること:

1. ✅ **環境定義が `pyproject.toml` に一元化**（Step 0）
   - uv で管理される最新の Python 3.13 環境
   - pyright-lsp が正常動作し、型チェック・コード補完が利用可能
   - 旧環境定義ファイル（`requirements.txt`, `enviornment.yml`）を削除

2. ✅ **不要なシェルスクリプトが `scripts/legacy/` に退避**（Step 6）
   - ルートディレクトリがクリーンになり、重要なファイルが明確

3. ✅ **`gns/` パッケージが再利用可能なモジュールのみを含む**（Step 5）
   - 実行スクリプトは `scripts/` に移動（`gns_train.py` など）

4. ✅ **推論、運動学的制約、学習バッチ準備が独立した関数として抽出可能**（Steps 1, 7）
   - `inference_utils.py` で推論ロジックを提供
   - `training.py` で学習ロジックを提供

5. ✅ **既存の学習・推論が正常に動作（数値結果が一致）**（全ステップ）
   - 最新パッケージバージョンでの動作確認済み

この状態で、ユーザーは:
- ✅ 必要な機能だけをインポートして使える
- ✅ モジュールとスクリプトの区別が明確
- ✅ 環境構築の手順が明確（`uv venv --python 3.13 && uv pip install -e .` で完結）
- ✅ ルートディレクトリがクリーンで、重要なファイルが見つけやすい
- ✅ 最新の Python エコシステムの恩恵を受けられる（Python 3.13, PyTorch 2.6+, numpy 2.1+）
- ✅ リファクタリング中も型チェック・補完が機能し、効率的に作業できる
- ✅ Python 3.13 の JIT コンパイラによるパフォーマンス向上の恩恵を受けられる

を実現できる。
