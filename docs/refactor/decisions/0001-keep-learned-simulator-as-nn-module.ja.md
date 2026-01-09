# Decision 0001: LearnedSimulatorをnn.Moduleとして維持

**日付**: 2026-01-08
**ステータス**: ~~承認済み~~ **Decision 0004により無効化**
**決定者**: 開発チーム

---
**⚠️ 注意**: この決定は2026-01-09に[Decision 0004: Step 2アーキテクチャ分離のRevert](0004-revert-step2-architecture-separation.ja.md)により無効化されました。Step 2はstate_dictの非互換性問題によりrevertされました。このドキュメントは歴史的参照のみのために保持されています。

---

## 背景

リファクタリングプロセス（Step 2: LearnedSimulatorクラスアーキテクチャの分離）において、`LearnedSimulator`が`nn.Module`の規約に違反していることが判明しました:

- `forward()`メソッドは存在するが使用されていない
- 代わりに`predict_positions()`や`predict_accelerations()`のようなメソッドが直接呼び出される
- これはPyTorchの`nn.Module`設計パターンに違反している

## 問題

以下の理由から、`LearnedSimulator`を`nn.Module`のサブクラスとして維持すべきかを検討しました:

1. **現在の違反**: `forward()`が適切に実装されていない
2. **代替アプローチ**: `GraphNeuralNetworkModel`をラップする通常のクラスにする

## 調査

`nn.Module`継承を削除した場合の影響を分析しました:

### nn.Moduleへの重要な依存関係

1. **デバイス管理** (`simulator.to(device)`)
   - `gns/train.py:108`, `259`, `320`
   - GPU/CPU転送に必要

2. **学習/評価モード** (`simulator.train()`, `simulator.eval()`)
   - `gns/train.py:109`, `319`
   - DropoutやBatchNormの挙動を制御

3. **オプティマイザ統合** (`simulator.parameters()`)
   - `gns/train.py:260`, `263`, `305`
   - 勾配ベース最適化に必要

4. **分散データ並列 (DDP)**
   - `gns/train.py:259` - `DDP(serial_simulator.to(rank), ...)`
   - `gns/train.py:557` - `simulator.module.predict_accelerations`
   - **重要**: DDPは`nn.Module`を必要とする - これが最も深刻なブロッカー

5. **状態の永続化** (`state_dict()`, `load_state_dict()`)
   - `gns/learned_simulator.py:274`, `284`
   - `gns/train.py:228`, `306`
   - モデルチェックポイントに必要

6. **型アノテーション**
   - `gns/train.py:58`
   - `gns/inference_utils.py:48`

## 決定

**`LearnedSimulator`を`nn.Module`のサブクラスとして維持します。**

`nn.Module`継承を削除する代わりに、`forward()`を適切に実装することで規約違反を修正します。

## 理由

1. **DDP要件**: 分散学習は`nn.Module`を必要とする重要な機能
2. **PyTorchエコシステム統合**: `nn.Module`のままであることで以下との互換性を維持:
   - デバイス管理 (`.to()`)
   - オプティマイザ統合 (`.parameters()`)
   - 状態の永続化 (`.state_dict()`, `.load_state_dict()`)
   - 学習/評価モード (`.train()`, `.eval()`)
3. **最小限の変更**: `forward()`を適切に実装する方が、`nn.Module`から離れてリファクタリングするよりもはるかに簡単
4. **将来の互換性**: PyTorchツールやライブラリとの互換性を維持

## 実装計画

以下のいずれかのアプローチで`nn.Module`規約違反に対処します:

### オプションA: forward()を実装する（推奨）
```python
def forward(self, current_positions, nparticles_per_example, particle_types, material_property=None):
    """Standard nn.Module forward pass."""
    return self.predict_positions(current_positions, nparticles_per_example,
                                  particle_types, material_property)
```

### オプションB: 逸脱を文書化する
`forward()`が使用されていない理由を明確に説明し、ユーザーを適切なメソッドに誘導する。

## 影響

### ポジティブ
- 既存のすべての機能を維持
- 分散学習機能を保持
- 必要なコード変更が最小限
- PyTorchエコシステムとの完全な互換性

### ネガティブ
- 技術的には依然として`nn.Module`規約に違反（ただしオプションAで修正可能）
- やや非標準的なAPI（ユーザーはモデルを直接呼び出す代わりに`predict_positions()`を呼び出す）

## 関連決定

この決定は以下に影響します:
- コードベースの型アノテーション
- シミュレータクラスの将来のリファクタリング
- ユーザー向けメソッドのAPI設計

## 備考

規約違反にもかかわらず現在の実装が正しく動作する理由:
1. PyTorchは実行時に`forward()`の使用を強制しない
2. 必要なすべての`nn.Module`機能（parameters、state_dictなど）が正しく継承されている
3. 内部の`GraphNeuralNetworkModel`はそのスコープ内で独自の`forward()`を適切に実装している
