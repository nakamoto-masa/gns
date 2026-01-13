# Decision 0001: Keep LearnedSimulator as nn.Module

**Date**: 2026-01-08
**Status**: ~~Accepted~~ **SUPERSEDED by Decision 0004**
**Decision Makers**: Development Team

---
**⚠️ NOTICE**: This decision was superseded on 2026-01-09 by [Decision 0004: Revert Step 2 Architecture Separation](0004-revert-step2-architecture-separation.ja.md). Step 2 was reverted due to state_dict incompatibility issues. This document is kept for historical reference only.

---

## Context

During the refactoring process (Step 2: Separating LearnedSimulator class architecture), we observed that `LearnedSimulator` violates `nn.Module` conventions:

- The `forward()` method exists but is never used
- Instead, methods like `predict_positions()` and `predict_accelerations()` are called directly
- This violates PyTorch's `nn.Module` design pattern

## Problem

We considered whether `LearnedSimulator` should remain an `nn.Module` subclass, given:

1. **Current violation**: `forward()` is not properly implemented
2. **Alternative approach**: Make it a regular class that wraps `GraphNeuralNetworkModel`

## Investigation

We analyzed the impact of removing `nn.Module` inheritance:

### Critical Dependencies on nn.Module

1. **Device Management** (`simulator.to(device)`)
   - `gns/train.py:108`, `259`, `320`
   - Required for GPU/CPU transfers

2. **Training/Evaluation Modes** (`simulator.train()`, `simulator.eval()`)
   - `gns/train.py:109`, `319`
   - Controls Dropout and BatchNorm behavior

3. **Optimizer Integration** (`simulator.parameters()`)
   - `gns/train.py:260`, `263`, `305`
   - Required for gradient-based optimization

4. **Distributed Data Parallel (DDP)**
   - `gns/train.py:259` - `DDP(serial_simulator.to(rank), ...)`
   - `gns/train.py:557` - `simulator.module.predict_accelerations`
   - **CRITICAL**: DDP requires `nn.Module` - this is the most severe blocker

5. **State Persistence** (`state_dict()`, `load_state_dict()`)
   - `gns/learned_simulator.py:274`, `284`
   - `gns/train.py:228`, `306`
   - Required for model checkpointing

6. **Type Annotations**
   - `gns/train.py:58`
   - `gns/inference_utils.py:48`

## Decision

**We will keep `LearnedSimulator` as an `nn.Module` subclass.**

Instead of removing `nn.Module` inheritance, we will fix the convention violation by properly implementing `forward()`.

## Rationale

1. **DDP Requirement**: Distributed training is a critical feature that requires `nn.Module`
2. **PyTorch Ecosystem Integration**: Staying as `nn.Module` maintains compatibility with:
   - Device management (`.to()`)
   - Optimizer integration (`.parameters()`)
   - State persistence (`.state_dict()`, `.load_state_dict()`)
   - Training/eval modes (`.train()`, `.eval()`)
3. **Minimal Changes**: Implementing `forward()` properly is much simpler than refactoring away from `nn.Module`
4. **Future Compatibility**: Maintains compatibility with PyTorch tools and libraries

## Implementation Plan

We will address the `nn.Module` convention violation through one of these approaches:

### Option A: Implement forward() (Recommended)
```python
def forward(self, current_positions, nparticles_per_example, particle_types, material_property=None):
    """Standard nn.Module forward pass."""
    return self.predict_positions(current_positions, nparticles_per_example,
                                  particle_types, material_property)
```

### Option B: Document the deviation
Add clear documentation explaining why `forward()` is not used and direct users to the appropriate methods.

## Consequences

### Positive
- Maintains all existing functionality
- Preserves distributed training capability
- Minimal code changes required
- Full PyTorch ecosystem compatibility

### Negative
- Still technically violates `nn.Module` convention (but can be fixed with Option A)
- Slightly non-standard API (users call `predict_positions()` instead of calling the model directly)

## Related Decisions

This decision affects:
- Type annotations in the codebase
- Future refactoring of simulator classes
- API design for user-facing methods

## Notes

The current implementation works correctly despite the convention violation because:
1. PyTorch doesn't enforce `forward()` usage at runtime
2. All necessary `nn.Module` functionality (parameters, state_dict, etc.) is inherited correctly
3. The internal `GraphNeuralNetworkModel` properly implements its own `forward()` for its scope
