# GNS Codebase Structure Migration

This document illustrates how responsibilities were reorganized through refactoring.

## Bipartite Graph: File Structure Migration

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'14px'}}}%%
graph LR
    subgraph BEFORE["<b>BEFORE: Monolithic Structure</b>"]
        OLD_TRAIN["gns/train.py<br/>(658 lines)<br/>━━━━━━━━━━<br/>• Configuration<br/>• Training Logic<br/>• Rollout Logic<br/>• CLI (single GPU)"]
        OLD_TRAIN_MULTI["gns/train_multinode.py<br/>(661 lines)<br/>━━━━━━━━━━<br/>• Configuration<br/>• Distributed Training<br/>• Rollout Logic<br/>• CLI (multi-node)"]
        OLD_RENDER["gns/render_rollout.py<br/>(246 lines)<br/>━━━━━━━━━━<br/>• Rendering Logic<br/>• CLI"]
        OLD_LEGACY["./*.sh (root)<br/>━━━━━━━━━━<br/>• build_venv.sh<br/>• module.sh<br/>• run.sh<br/>• start_venv.sh"]
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

**Legend:**
- 🔴 **Red**: Pre-refactoring execution scripts (located in gns/, mixed business logic and CLI)
- 🟢 **Green**: Post-refactoring reusable modules (gns/)
- 🔵 **Blue**: Post-refactoring CLI wrappers (scripts/)
- 🟡 **Yellow**: New package management system
- ⚫ **Gray**: Archived files

**Key Changes:**
- **Execution scripts**: `gns/train.py` (658 lines) + `gns/train_multinode.py` (661 lines) + `gns/render_rollout.py` (246 lines) → Modularized logic + thin CLI wrappers (scripts/: 181, 212, 58 lines)

## Major Changes

### Separated Responsibilities

1. **Configuration** (gns/train.py, gns/train_multinode.py → gns/config.py)
   - Unified duplicated global constants into a configuration class

2. **Training** (gns/train.py, gns/train_multinode.py → gns/training.py)
   - Training loop and utilities
   - Extracted common logic for single-GPU/distributed training
   - Dataloader management
   - Checkpoint management

3. **Rollout/Inference** (gns/train.py, gns/train_multinode.py → gns/rollout.py + gns/inference_utils.py)
   - Unified rollout execution logic
   - Separated trajectory prediction utilities

4. **Rendering** (gns/render_rollout.py → gns/render.py)
   - Separated visualization logic from CLI
   - GIF, VTK, and image rendering functionality

5. **CLI** (gns/*.py → scripts/*.py)
   - Thin CLI wrappers (56-250 lines)
   - Delegate business logic to above modules
   - Moved from gns/ directory to scripts/ directory

6. **Environment** (./*.sh → pyproject.toml)
   - Modern package management with uv
