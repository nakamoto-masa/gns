# Decision 0005: rollout() Docstringの形状不一致の修正

**日付**: 2026-01-09
**ステータス**: 承認済み
**決定者**: 開発チーム

## 背景

コードレビュー中に、`gns/rollout.py` の `run_rollout()` 関数のdocstringと実装の間に不一致が発見されました。docstringでは `position` パラメータの形状が `(timesteps, nparticles, ndims)` と記載されていますが、実際の実装は `(nparticles, timesteps, ndims)` を期待しています。

### 調査結果

この問題は**リファクタリングによって引き起こされたものではありません**。docstringはStep 1以前の元のコードベースで既に間違っていました。

#### 証拠: 元のコード（Step 1以前）

**データローダー** (`gns/data_loader.py` - オリジナルから変更なし):
```python
# TrajectoriesDataset.__getitem__
positions = np.transpose(positions, (1, 0, 2))  # (nparticles, timesteps, ndims) に転置
```

**元の rollout() 関数** (コミット 2096600 以前の `gns/train.py`):
```python
def rollout(position, ...):
    """
    Args:
        position: Positions of particles (timesteps, nparticles, ndims)  # ← Docstring（間違い）
    """
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]  # ← 実装は (nparticles, timesteps, ndims) を期待
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]
```

**元の predict() 関数**:
```python
positions = features[0].to(device)  # データローダーから (nparticles, timesteps, ndims) を受け取る
sequence_length = positions.shape[1]  # 2番目の次元をtimestepsとして扱う
```

実装は明らかに `position[:, :INPUT_SEQUENCE_LENGTH]` でtimestep次元（2番目の軸）に沿ってスライスすることを期待しており、これは `position` が `(nparticles, timesteps, ndims)` でなければならないことを意味します。

## 問題の経過

| 段階 | ファイル | Docstring | 実際の形状 | 状態 |
|------|---------|-----------|-----------|------|
| **Step 1以前** | `gns/train.py::rollout()` | `(timesteps, nparticles, ndims)` | `(nparticles, timesteps, ndims)` | ❌ 既に間違い |
| **Step 1** | `gns/inference_utils.py::rollout()` | `(nparticles, timesteps, ndims)` | `(nparticles, timesteps, ndims)` | ✅ 修正済み |
| **Step 1** | `gns/train.py::rollout()` (ラッパー) | `(timesteps, nparticles, ndims)` | `(nparticles, timesteps, ndims)` | ❌ 更新されず |
| **Step 5** | `gns/rollout.py::run_rollout()` | `(timesteps, nparticles, ndims)` | `(nparticles, timesteps, ndims)` | ❌ 間違ったdocstringをコピー |

## なぜこうなったか

1. **元のdocstringが間違っていた** - `rollout()` 関数のdocstringは最初から間違っていた
2. **Step 1で部分的に修正** - `inference_utils.rollout()` のdocstringは正しい
3. **Step 5で間違ったバージョンをコピー** - `gns/rollout.py` を作成する際に、`inference_utils.py` の正しいdocstringではなく、元のラッパーの間違ったdocstringをコピーしてしまった

## 決定

**実際のデータ形状に合わせてdocstringを修正: `(nparticles, timesteps, ndims)`**

これは変更やrevertではなく、常に間違っていたドキュメントの修正です。

## 実装

### 更新するファイル

1. **`gns/rollout.py::run_rollout()`** (85行目)
   - 変更前: `position: Initial positions (timesteps, nparticles, ndims).`
   - 変更後: `position: Initial positions (nparticles, timesteps, ndims).`

2. **`gns/rollout.py::rollout()`** (レガシーラッパー、315行目)
   - 変更前: `position: Positions of particles (timesteps, nparticles, ndims)`
   - 変更後: `position: Positions of particles (nparticles, timesteps, ndims)`

### なぜこの形状なのか

形状 `(nparticles, timesteps, ndims)` が必要な理由:

1. **データローダーがこの形状を返す**: `TrajectoriesDataset` が `(nparticles, timesteps, ndims)` に転置
2. **実装がこの形状を期待**: `position[:, :input_sequence_length]` のスライスは2番目の次元を時間として扱う
3. **訓練データと一貫性**: `SamplesDataset` も転置後に `(nparticles, timesteps, ndims)` を使用（99行目）

### 検証

実装は常に正しく動作していました:
```python
# gns/inference_utils.py
initial_positions = position[:, :input_sequence_length]  # (nparticles, 6, ndims)
ground_truth_positions = position[:, input_sequence_length:]  # (nparticles, nsteps, ndims)
```

このスライスは `position` が `(nparticles, timesteps, ndims)` である場合にのみ意味を成します。

## 影響

### ポジティブ
- ✅ **ドキュメントが現実と一致** - ユーザーが混乱しない
- ✅ **inference_utils.pyと一貫性** - 両方のdocstringが一致
- ✅ **コード変更不要** - ドキュメント修正のみ

### 中立
- ℹ️ **破壊的変更なし** - 実装は変更なし

## 関連する決定

- **関連**: Step 1（inference_utils.py作成）
- **関連**: Step 5（rollout.py作成）

## 得られた教訓

1. **リファクタリング時はdocstringを実装と照合する** - docstringを盲目的にコピーしない
2. **正しいソースを確認する** - Step 5は古いラッパー（間違い）ではなく `inference_utils.py`（正しい）からコピーすべきだった
3. **Docstringのバグはリファクタリングを通じて持続する** - 自動テストではドキュメントエラーを検出できない

## 備考

これは純粋なドキュメント修正です。コードは常に `(nparticles, timesteps, ndims)` 形状で正しく動作していました。docstringが期待する形状について嘘をついていただけです。
