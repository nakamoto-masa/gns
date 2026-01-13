# Decision 0004: Step 2アーキテクチャ分離のRevert

**日付**: 2026-01-09
**ステータス**: 承認済み
**決定者**: 開発チーム

## 背景

リファクタリング計画のStep 2では、`LearnedSimulator`を2つのクラスに分割しました:
- `LearnedSimulator`(ラッパー) - 正規化、オイラー積分、物理統合
- `GraphNeuralNetworkModel`(コアGNN) - グラフ構築、メッセージパッシング

しかし、これにより意図しない重大な副作用が発生しました。

### 問題: State Dictの非互換性

```python
# Step 2以前
LearnedSimulator.state_dict() = {
    '_encode_process_decode.encoder.mlp.0.weight': ...,
    '_particle_type_embedding.weight': ...,
    ...
}

# Step 2以降
LearnedSimulator.state_dict() = {
    '_graph_model._encode_process_decode.encoder.mlp.0.weight': ...,  # ← キー名が変更された
    '_graph_model._particle_type_embedding.weight': ...,  # ← キー名が変更された
    ...
}
```

**影響**: 既存のモデルチェックポイント(model-*.pt)がリファクタリング後に読み込めなくなる。

## Step 2の当初の目標(未達成)

[plan.ja.md:5](plan.ja.md#L5)より:
> 利用者としてモデルを使いやすくすることを目的とした、最小限のリファクタリングを行う。

[plan.ja.md:182-201](plan.ja.md#L182-L201)より:
> **目的**: NNモデル本体と物理統合を分離し、用途に応じて使い分けられるようにする。
>
> **解決する課題**: 課題4 - GNNモデルだけを使いたいユーザーは `GraphNeuralNetworkModel` を、物理予測が必要なユーザーは `LearnedSimulator` を使える

**達成状況の分析**:
1. ❌ **「最小限のリファクタリング」を達成できず** - 破壊的変更を導入
2. ❌ **「nn.Moduleの機能だけを分離」を達成できず** - 両方ともnn.Moduleのまま
3. ❌ **「モデルを使いやすく」を達成できず** - チェックポイント互換性を破壊
4. ❌ **当初の目標が不明瞭** - 「nn.Moduleの機能だけを分離」とは何を意味するのか?

## 決定

**Step 2を完全にrevertし、LearnedSimulatorを単一クラスのまま維持する。**

### Revertする内容

1. **削除**: `gns/graph_model.py` (ファイル全体)
2. **復元**: `gns/learned_simulator.py` をStep 2以前の状態に戻す(コミット`2096600`)
3. **無効化**: Decision 0001 (LearnedSimulatorをnn.Moduleとして維持)
4. **無効化**: Decision 0002 (型アノテーション修正) ※ただし型改善は保持

### 維持する内容

- Steps 3-6はそのまま維持
- 他のリファクタリング作業には影響なし
- `GraphNeuralNetworkModel`は`learned_simulator.py`内でしか使われていないため外部依存なし

## 理由

### 1. State Dictの互換性は極めて重要

チェックポイント互換性を破壊することは、ユーザーにとって深刻な問題:
- 学習実行には数日から数週間かかる場合がある
- ユーザーはチェックポイントから再開する必要がある
- 移行パスなしでこれを破壊することは許容できない

### 2. Step 2の目標が不明瞭だった

当初の目標:
> nn.Moduleの機能だけを分離するという当初の狙いも実現できてないし

しかし、`LearnedSimulator`と`GraphNeuralNetworkModel`の両方が`nn.Module`サブクラスなので、「nn.Moduleの機能だけを分離」とは何を意味するのか?

可能な解釈:
- **関心の分離?** - 両クラスとも複数の責務が混在したまま
- **純粋GNN vs 物理?** - 2つのnn.Moduleを作らなくても達成可能
- **再利用性?** - `GraphNeuralNetworkModel`は`LearnedSimulator`外で使われていない

目標があいまいで、実装は明確な利益を達成しなかった。

### 3. Steps 3-6はStep 2に依存していない

分析結果:
- `GraphNeuralNetworkModel`は`learned_simulator.py`でのみインポート
- `scripts/gns_train.py`は`LearnedSimulator`のみ使用
- Steps 3-6はStep 2の内部構造とは独立して動作

### 4. 代替アプローチが存在する

高度なユーザーにGNN機能へのアクセスを提供したい場合:
- **内部メソッドを公開** - `_encoder_preprocessor`、`_compute_graph_connectivity`を公開メソッドに
- **継承よりも合成** - 新しいクラスではなくユーティリティ関数を作成
- **シンプルに保つ** - ほとんどのユーザーは`predict_positions()`だけで十分

## 実装

### 変更されたファイル

1. **修正**: [gns/learned_simulator.py](gns/learned_simulator.py)
   - コミット`2096600`の状態に復元
   - 現代的な型アノテーションを適用(`torch.Tensor`、`X | None`)
   - `from typing import Dict`を削除(未使用)
   - `device: str = "cpu"`型アノテーションを追加

2. **削除**: `gns/graph_model.py`

3. **無効化**:
   - `docs/refactor/decisions/0001-keep-learned-simulator-as-nn-module.md`
   - `docs/refactor/decisions/0002-type-annotation-fixes.md`

### 型の改善

Revert時に、型アノテーションの改善は維持:
```python
# 現代的なPython 3.10+構文
material_property: torch.Tensor | None = None

# 適切な型の大文字表記
torch.Tensor  # torch.tensorではない
```

## 影響

### ポジティブ

- ✅ **State dict互換性が回復** - 既存のチェックポイントが動作
- ✅ **よりシンプルなアーキテクチャ** - 2クラスではなく1クラス
- ✅ **Steps 3-6に影響なし** - カスケード障害なし
- ✅ **明確な前進パス** - 将来API改善を再検討可能

### ネガティブ

- ❌ **Step 2の作業が無駄に** - クラス分割に費やした時間
- ❌ **Decision 0001, 0002が無効化** - ドキュメントのオーバーヘッド
- ⚠️ **より良いAPIが依然として必要** - 元の問題(使いやすさ)は未解決

## 得られた教訓

1. **nn.Module構造をリファクタリングする際は必ずstate_dict互換性を確認する**
2. **大規模リファクタリング前に明確で測定可能な目標を定義する**
3. **最小限の変更から始める** - 利益が明確な場合にのみ複雑さを追加
4. **早期に仮定を検証する** - 「nn.Moduleの機能だけを分離」は検証されなかった

## 次のステップ

将来APIを改善したい場合:

### オプションA: 単一クラスを維持、ドキュメントを改善
- 内部メソッドを説明する明確なdocstringを追加
- 高度なユーザー向けの使用例を提供
- 破壊的変更なし

### オプションB: 内部メソッドを公開
```python
# 一部のメソッドを公開
def compute_graph_connectivity(self, ...):  # _プレフィックスを削除
def encode_features(self, ...):  # リネームして公開
```

### オプションC: ユーティリティ関数を作成(クラスではなく)
```python
# gns/graph_utils.py
def compute_graph_connectivity(positions, radius, ...):
    """グラフ構築のための純粋関数"""
    ...
```

## 関連する決定

- **置き換え**: [Decision 0001](0001-keep-learned-simulator-as-nn-module.md) (LearnedSimulatorをnn.Moduleとして維持)
- **置き換え**: [Decision 0002](0002-type-annotation-fixes.md) (型アノテーション修正 - 構造部分)
- **保持**: 型アノテーションの改善(現代的構文)

## 備考

このrevertは、Step 2の目標が間違っていたことを意味するものではありません - API使いやすさの向上は依然として価値があります。しかし:

1. **State dict互換性は譲れない**
2. **目標は具体的で測定可能でなければならない**
3. **最小限の変更が良い** - シンプルに始め、必要な時にのみ複雑さを追加

リファクタリング計画全体は依然として健全です - Steps 1, 3-6は目標を成功裏に達成しました。
