# GNS Codebase Structure Migration

このドキュメントは、リファクタリングによって役割がどのように再配置されたかを示します。

## 二部グラフ: ファイル構造の移行

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'14px'}}}%%
graph LR
    subgraph BEFORE["<b>BEFORE: Monolithic Structure</b>"]
        OLD_TRAIN["gns/train.py<br/>(658 lines)<br/>━━━━━━━━━━<br/>• Configuration<br/>• Training Logic<br/>• Rollout Logic<br/>• CLI (single GPU)"]
        OLD_TRAIN_MULTI["gns/train_multinode.py<br/>(661 lines)<br/>━━━━━━━━━━<br/>• Configuration<br/>• Distributed Training<br/>• Rollout Logic<br/>• CLI (multi-node)"]
        OLD_RENDER["gns/render_rollout.py<br/>(246 lines)<br/>━━━━━━━━━━<br/>• Rendering Logic<br/>• CLI"]
        OLD_LEGACY["legacy/*.sh<br/>━━━━━━━━━━<br/>• build_venv.sh<br/>• module.sh<br/>• run.sh<br/>• start_venv.sh"]
    end

    subgraph AFTER["<b>AFTER: Modular Structure</b>"]
        subgraph GNS["gns/ (Reusable Modules)"]
            NEW_CONFIG["config.py<br/>(204 lines)"]
            NEW_TRAINING["training.py<br/>(788 lines)"]
            NEW_ROLLOUT["rollout.py<br/>(329 lines)"]
            NEW_INFERENCE["inference_utils.py<br/>(175 lines)"]
            NEW_RENDER["render.py<br/>(415 lines)"]
        end

        subgraph SCRIPTS["scripts/ (CLI Wrappers)"]
            NEW_TRAIN_CLI["gns_train.py<br/>(181 lines)"]
            NEW_TRAIN_MULTI["gns_train_multinode.py<br/>(212 lines)"]
            NEW_RENDER_CLI["gns_render_rollout.py<br/>(58 lines)"]
        end

        NEW_PYPROJECT["pyproject.toml<br/>━━━━━━━━━━<br/>uv package<br/>management"]
        NEW_ARCHIVED["scripts/legacy/<br/>━━━━━━━━━━<br/>(archived)"]
    end

    %% Configuration (from both train scripts)
    OLD_TRAIN -->|"Configuration<br/>Constants"| NEW_CONFIG
    OLD_TRAIN_MULTI -->|"Configuration<br/>Constants"| NEW_CONFIG

    %% Training Logic
    OLD_TRAIN -->|"Training<br/>Functions"| NEW_TRAINING
    OLD_TRAIN_MULTI -->|"Distributed<br/>Training"| NEW_TRAINING

    %% Rollout/Inference
    OLD_TRAIN -->|"Rollout<br/>Execution"| NEW_ROLLOUT
    OLD_TRAIN -->|"Trajectory<br/>Prediction"| NEW_INFERENCE
    OLD_TRAIN_MULTI -->|"Rollout<br/>Execution"| NEW_ROLLOUT

    %% Rendering
    OLD_RENDER -->|"Visualization<br/>Logic"| NEW_RENDER

    %% CLI Wrappers
    OLD_TRAIN -->|"Refactored<br/>to thin CLI"| NEW_TRAIN_CLI
    OLD_TRAIN_MULTI -->|"Refactored<br/>to thin CLI"| NEW_TRAIN_MULTI
    OLD_RENDER -->|"Refactored<br/>to thin CLI"| NEW_RENDER_CLI

    %% Environment
    OLD_LEGACY -->|"Archived"| NEW_ARCHIVED
    OLD_LEGACY -.->|"Replaced by"| NEW_PYPROJECT

    %% Styling
    classDef oldStyle fill:#ffcccc,stroke:#cc0000,stroke-width:3px,color:#000
    classDef newModuleStyle fill:#ccffcc,stroke:#00cc00,stroke-width:2px,color:#000
    classDef newCLIStyle fill:#cce5ff,stroke:#0066cc,stroke-width:2px,color:#000
    classDef newConfigStyle fill:#ffffcc,stroke:#cccc00,stroke-width:2px,color:#000
    classDef archivedStyle fill:#e6e6e6,stroke:#999999,stroke-width:2px,color:#666

    class OLD_TRAIN,OLD_TRAIN_MULTI,OLD_RENDER,OLD_LEGACY oldStyle
    class NEW_CONFIG,NEW_TRAINING,NEW_ROLLOUT,NEW_INFERENCE,NEW_RENDER newModuleStyle
    class NEW_TRAIN_CLI,NEW_TRAIN_MULTI,NEW_RENDER_CLI newCLIStyle
    class NEW_PYPROJECT newConfigStyle
    class NEW_ARCHIVED archivedStyle
```

**凡例:**
- 🔴 **赤**: リファクタリング前の実行スクリプト（gns/内に配置、ビジネスロジックとCLIが混在）
- 🟢 **緑**: リファクタリング後の再利用可能モジュール (gns/)
- 🔵 **青**: リファクタリング後のCLIラッパー (scripts/)
- 🟡 **黄**: 新しいパッケージ管理システム
- ⚫ **灰**: アーカイブされたファイル

**主な変更:**
- **実行スクリプト**: `gns/train.py` (658行) + `gns/train_multinode.py` (661行) + `gns/render_rollout.py` (246行) → ロジックをモジュール化 + 薄いCLIラッパー (scripts/: 181, 212, 58行)

## 主要な変更点

### 分離された責務

1. **Configuration** (gns/train.py, gns/train_multinode.py → gns/config.py)
   - 重複していたグローバル定数を統一された設定クラスに変換

2. **Training** (gns/train.py, gns/train_multinode.py → gns/training.py)
   - トレーニングループとユーティリティ
   - 単一GPU/分散訓練の共通ロジック抽出
   - データローダー管理
   - チェックポイント管理

3. **Rollout/Inference** (gns/train.py, gns/train_multinode.py → gns/rollout.py + gns/inference_utils.py)
   - ロールアウト実行ロジックの共通化
   - 軌道予測ユーティリティの分離

4. **Rendering** (gns/render_rollout.py → gns/render.py)
   - 可視化ロジックとCLIの分離
   - GIF、VTK、画像レンダリング機能

5. **CLI** (gns/*.py → scripts/*.py)
   - 薄いCLIラッパー（56-250行）
   - ビジネスロジックを上記モジュールに委譲
   - gns/ディレクトリからscripts/ディレクトリへ移動

6. **Environment** (legacy/*.sh → pyproject.toml)
   - uvによるモダンなパッケージ管理
