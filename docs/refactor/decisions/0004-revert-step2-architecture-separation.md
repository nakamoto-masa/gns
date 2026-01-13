# Decision 0004: Revert Step 2 Architecture Separation

**Date**: 2026-01-09
**Status**: Accepted
**Decision Makers**: Development Team

## Context

In Step 2 of the refactoring plan, we split `LearnedSimulator` into two classes:
- `LearnedSimulator` (wrapper) - normalization, Euler integration, physics integration
- `GraphNeuralNetworkModel` (core GNN) - graph construction, message passing

However, this introduced an unintended critical side effect.

### Problem: State Dict Incompatibility

```python
# Before Step 2
LearnedSimulator.state_dict() = {
    '_encode_process_decode.encoder.mlp.0.weight': ...,
    '_particle_type_embedding.weight': ...,
    ...
}

# After Step 2
LearnedSimulator.state_dict() = {
    '_graph_model._encode_process_decode.encoder.mlp.0.weight': ...,  # ← Key name changed
    '_graph_model._particle_type_embedding.weight': ...,  # ← Key name changed
    ...
}
```

**Impact**: Existing model checkpoints (model-*.pt) cannot be loaded after refactoring.

## Step 2's Original Goals (Not Achieved)

From [plan.ja.md:5](plan.ja.md#L5):
> The purpose is to make the model easier to use for users through minimal refactoring.

From [plan.ja.md:182-201](plan.ja.md#L182-L201):
> **Purpose**: Separate the NN model core from physics integration so they can be used according to different needs.
>
> **Problem Solved**: Issue 4 - Users who want to use only the GNN model can use `GraphNeuralNetworkModel`, while users who need physics prediction can use `LearnedSimulator`

**Achievement Analysis**:
1. ❌ **Failed to achieve "minimal refactoring"** - Introduced breaking changes
2. ❌ **Failed to achieve "separate only nn.Module functionality"** - Both remain nn.Modules
3. ❌ **Failed to "make the model easier to use"** - Broke checkpoint compatibility
4. ❌ **Original goal was unclear** - What does "separate only nn.Module functionality" mean?

## Decision

**Completely revert Step 2 and keep LearnedSimulator as a single class.**

### What to Revert

1. **Delete**: `gns/graph_model.py` (entire file)
2. **Restore**: `gns/learned_simulator.py` to pre-Step 2 state (commit `2096600`)
3. **Supersede**: Decision 0001 (Keep LearnedSimulator as nn.Module)
4. **Supersede**: Decision 0002 (Type annotation fixes) ※but keep type improvements

### What to Keep

- Steps 3-6 remain unchanged
- No impact on other refactoring work
- No external dependencies since `GraphNeuralNetworkModel` was only used within `learned_simulator.py`

## Rationale

### 1. State Dict Compatibility is Critical

Breaking checkpoint compatibility is a severe issue for users:
- Training runs can take days to weeks
- Users need to resume from checkpoints
- Breaking this without a migration path is unacceptable

### 2. Step 2's Goals Were Unclear

Original goal:
> The original intent to separate only nn.Module functionality was also not achieved

But since both `LearnedSimulator` and `GraphNeuralNetworkModel` are `nn.Module` subclasses, what does "separate only nn.Module functionality" mean?

Possible interpretations:
- **Separation of concerns?** - Both classes still have mixed responsibilities
- **Pure GNN vs physics?** - Could be achieved without creating two nn.Modules
- **Reusability?** - `GraphNeuralNetworkModel` is not used outside `LearnedSimulator`

The goal was ambiguous, and the implementation didn't achieve clear benefits.

### 3. Steps 3-6 Don't Depend on Step 2

Analysis shows:
- `GraphNeuralNetworkModel` is only imported by `learned_simulator.py`
- `scripts/gns_train.py` only uses `LearnedSimulator`
- Steps 3-6 work independently of Step 2's internal structure

### 4. Alternative Approaches Exist

If we want to provide advanced users access to GNN functionality:
- **Expose internal methods** - Make `_encoder_preprocessor`, `_compute_graph_connectivity` public
- **Composition over inheritance** - Create utility functions instead of new classes
- **Keep it simple** - Most users only need `predict_positions()`

## Implementation

### Files Changed

1. **Modified**: [gns/learned_simulator.py](gns/learned_simulator.py)
   - Restored to commit `2096600` state
   - Applied modern type annotations (`torch.Tensor`, `X | None`)
   - Removed `from typing import Dict` (unused)
   - Added `device: str = "cpu"` type annotation

2. **Deleted**: `gns/graph_model.py`

3. **Superseded**:
   - `docs/refactor/decisions/0001-keep-learned-simulator-as-nn-module.md`
   - `docs/refactor/decisions/0002-type-annotation-fixes.md`

### Type Improvements

When reverting, we kept type annotation improvements:
```python
# Modern Python 3.10+ syntax
material_property: torch.Tensor | None = None

# Proper capitalization for types
torch.Tensor  # not torch.tensor
```

## Consequences

### Positive

- ✅ **State dict compatibility restored** - Existing checkpoints work
- ✅ **Simpler architecture** - 1 class instead of 2
- ✅ **No impact on Steps 3-6** - No cascading failures
- ✅ **Clear path forward** - Can revisit API improvements in future

### Negative

- ❌ **Step 2 work wasted** - Time spent on class separation
- ❌ **Decisions 0001, 0002 superseded** - Documentation overhead
- ⚠️ **Better API still needed** - Original problem (usability) remains unsolved

## Lessons Learned

1. **Always verify state_dict compatibility when refactoring nn.Module structures**
2. **Define clear, measurable goals before major refactoring**
3. **Start with minimal changes** - Add complexity only when benefits are clear
4. **Validate assumptions early** - "Separate only nn.Module functionality" was never validated

## Next Steps

If we want to improve the API in the future:

### Option A: Keep single class, improve documentation
- Add clear docstrings explaining internal methods
- Provide usage examples for advanced users
- No breaking changes

### Option B: Expose internal methods
```python
# Make some methods public
def compute_graph_connectivity(self, ...):  # Remove _ prefix
def encode_features(self, ...):  # Rename and expose
```

### Option C: Create utility functions (not classes)
```python
# gns/graph_utils.py
def compute_graph_connectivity(positions, radius, ...):
    """Pure function for graph construction"""
    ...
```

## Related Decisions

- **Supersedes**: [Decision 0001](0001-keep-learned-simulator-as-nn-module.md) (Keep LearnedSimulator as nn.Module)
- **Supersedes**: [Decision 0002](0002-type-annotation-fixes.md) (Type annotation fixes - structural part)
- **Keeps**: Type annotation improvements (modern syntax)

## Notes

This revert does not mean Step 2's goals were wrong - improving API usability is still valuable. However:

1. **State dict compatibility is non-negotiable**
2. **Goals must be concrete and measurable**
3. **Minimal changes are better** - Start simple, add complexity only when needed

The overall refactoring plan is still sound - Steps 1, 3-6 successfully achieved their goals.
