# Decision 0002: 型アノテーション修正

**日付**: 2026-01-08
**ステータス**: ~~承認済み~~ **Decision 0004により部分的に無効化**
**決定者**: 開発チーム

---
**⚠️ 注意**: Step 2は2026-01-09に[Decision 0004: Step 2アーキテクチャ分離のRevert](0004-revert-step2-architecture-separation.ja.md)によりrevertされました。しかし、型アノテーションの改善（現代的な構文、適切な大文字表記）はrevert時に保持されました。このドキュメントは歴史的参照のために保持されています。

---

## 背景

Step 2リファクタリング（LearnedSimulatorとGraphNeuralNetworkModelの分離）中に、型チェッカーの警告を引き起こす型アノテーションの不整合を導入しました:

1. **Optionalの型アノテーション欠落**: `= None`デフォルトを持つパラメータに適切な型アノテーションがない
2. **boundariesの型不一致**: `learned_simulator.py`では`np.ndarray`を使用しているが、`graph_model.py`では`torch.Tensor`を宣言
3. **未使用のインポート**: `from typing import Dict`がインポートされているが使用されていない
4. **deviceパラメータに型アノテーションがない**: `device="cpu"`に型ヒントがない

## 特定された問題

### 1. Optional/Union アノテーションの欠落
```python
# 現在（誤り）
material_property: torch.Tensor = None

# 正しい形（現代的なPython 3.10+構文を使用）
material_property: torch.Tensor | None = None
```

**影響箇所:**
- `learned_simulator.py:104, 166, 194`
- `graph_model.py:111`

### 2. Boundariesの型不一致
```python
# learned_simulator.py:27
boundaries: np.ndarray

# graph_model.py:26
boundaries: torch.Tensor  # ← 型不一致!

# learned_simulatorからgraph_modelへ渡される
self._graph_model = GraphNeuralNetworkModel(
    boundaries=boundaries,  # torch.Tensorが宣言されている場所にnp.ndarrayを渡している
    ...
)
```

**実行時に動作する理由:**
- `graph_model.py:139-140`で、boundariesは使用時に変換される:
  ```python
  boundaries = torch.tensor(
      self._boundaries, requires_grad=False).float().to(self._device)
  ```
- `torch.tensor()`は`np.ndarray`と`torch.Tensor`の両方を受け入れる
- 変換は初期化時ではなく、必要な時に遅延的に行われる

**分析:**
- `GraphNeuralNetworkModel`は純粋な`nn.Module`なので、`torch.Tensor`が適切に見える
- しかし、実際の実装は値をそのまま保存し、後で変換する
- 現在のパターン（`np.ndarray`を受け取り、必要な時に`torch.Tensor`に変換）は有効

### 3. 未使用のインポート
```python
# learned_simulator.py:5
from typing import Dict  # ファイル内で使用されていない
```

### 4. デバイスの型アノテーション欠落
```python
# learned_simulator.py:32
device="cpu"  # 型アノテーションなし
```

## 決定

### 1. Nullableパラメータに`X | None`構文を使用
`= None`デフォルトを持つすべてのパラメータに、`Optional[X]`の代わりに現代的なPython 3.10+のunion構文（`X | None`）を使用する。

**理由:**
- より簡潔で読みやすい
- PEP 604標準（Python 3.10+）
- 好まれる現代的なPythonスタイル

### 2. boundariesの型を`np.ndarray`に統一
実際の実装に合わせて、両方のファイルで`boundaries: np.ndarray`を維持する。

**理由:**
- 実装は`np.ndarray`を保存し、使用時に`torch.Tensor`に変換する
- この遅延変換パターンは有効で効率的
- 型アノテーションは中間変換ではなく、実際に保存される型を反映すべき
- ユーザーのAPI簡潔性を維持（メタデータから`np.ndarray`を渡す）

**検討したが却下した代替案:**
- `torch.Tensor`を使用: 即座の変換が必要になり、不要なオーバーヘッドが追加される
- `Union[np.ndarray, torch.Tensor]`を使用: 冗長すぎて、利点なしに複雑さが増す

### 3. 未使用のインポートを削除
`learned_simulator.py`から`from typing import Dict`を削除する。

### 4. deviceの型アノテーションを追加
一貫性のため、`device="cpu"`を`device: str = "cpu"`に変更する。

## 実装

### 必要な変更

#### learned_simulator.py
```python
# 削除: from typing import Dict
# インポート追加不要 - 組み込みのunion構文を使用

class LearnedSimulator(nn.Module):
  def __init__(
          self,
          ...
          boundaries: np.ndarray,
          normalization_stats: dict,
          ...
          device: str = "cpu"  # 型アノテーションを追加
  ):
      ...

  def _encoder_preprocessor(
          self,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None  # | None を使用
  ):
      ...

  def predict_positions(
          self,
          current_positions: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None  # | None を使用
  ) -> torch.Tensor:
      ...

  def predict_accelerations(
          self,
          next_positions: torch.Tensor,
          position_sequence_noise: torch.Tensor,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None  # | None を使用
  ):
      ...
```

#### graph_model.py
```python
import numpy as np  # 型アノテーション用のインポート追加

class GraphNeuralNetworkModel(nn.Module):
  def __init__(
          self,
          ...
          boundaries: np.ndarray,  # torch.Tensorから変更
          ...
  ):
      ...

  def build_graph_features(
          self,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          normalized_velocity_sequence: torch.Tensor,
          material_property: torch.Tensor | None = None  # | None を使用
  ):
      ...
```

## 影響

### ポジティブ
- より良い型安全性とIDE支援
- モジュール間で一貫した型アノテーション
- 現代的なPython構文（PEP 604）
- 実行時の挙動変更なし
- 型アノテーションが実際の実装に一致
- メンテナンスとデバッグが容易

### ネガティブ
- `X | None`構文にはPython 3.10+が必要
- union型でわずかに冗長性が増加

## 検証

実装後:
1. Pythonバージョンが3.10+であることを確認（プロジェクト要件ですでに満たされている）
2. 型チェッカー（mypy/pylance）を実行して型エラーがないことを確認
3. 既存のテストを実行して実行時の退行がないことを確認
4. 学習とrolloutが正しく動作することを確認

## 関連決定

- Decision 0001: LearnedSimulatorをnn.Moduleとして維持

## 備考

- Python 3.10+はプロジェクトですでに必要（`uv`と現代的なツールを使用）
- `X | None`構文は`Optional[X]`よりも明確で簡潔
- 型アノテーションは純粋に静的 - 実行時オーバーヘッドなし
