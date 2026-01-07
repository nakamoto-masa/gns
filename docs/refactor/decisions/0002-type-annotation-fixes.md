# Decision 0002: Type Annotation Fixes

**Date**: 2026-01-08
**Status**: Accepted
**Decision Makers**: Development Team

## Context

During the Step 2 refactoring (separating LearnedSimulator and GraphNeuralNetworkModel), we introduced type annotation inconsistencies that cause type checker warnings:

1. **Optional type annotations missing**: Parameters with `= None` default lack proper type annotations
2. **Boundaries type mismatch**: `learned_simulator.py` uses `np.ndarray` but `graph_model.py` declares `torch.Tensor`
3. **Unused imports**: `from typing import Dict` is imported but never used
4. **Device parameter lacks type annotation**: `device="cpu"` has no type hint

## Problems Identified

### 1. Missing Optional/Union Annotations
```python
# Current (incorrect)
material_property: torch.Tensor = None

# Should be (using modern Python 3.10+ syntax)
material_property: torch.Tensor | None = None
```

**Affected locations:**
- `learned_simulator.py:104, 166, 194`
- `graph_model.py:111`

### 2. Boundaries Type Mismatch
```python
# learned_simulator.py:27
boundaries: np.ndarray

# graph_model.py:26
boundaries: torch.Tensor  # ← Type mismatch!

# Passed from learned_simulator to graph_model
self._graph_model = GraphNeuralNetworkModel(
    boundaries=boundaries,  # np.ndarray passed where torch.Tensor declared
    ...
)
```

**Why it works at runtime:**
- In `graph_model.py:139-140`, boundaries are converted at usage time:
  ```python
  boundaries = torch.tensor(
      self._boundaries, requires_grad=False).float().to(self._device)
  ```
- `torch.tensor()` accepts both `np.ndarray` and `torch.Tensor`
- The conversion happens lazily when needed, not at initialization

**Analysis:**
- `GraphNeuralNetworkModel` is a pure `nn.Module`, so `torch.Tensor` seems appropriate
- However, the actual implementation stores the value as-is and converts later
- The current pattern (receive `np.ndarray`, convert to `torch.Tensor` when needed) is valid

### 3. Unused Import
```python
# learned_simulator.py:5
from typing import Dict  # Never used in the file
```

### 4. Missing Device Type Annotation
```python
# learned_simulator.py:32
device="cpu"  # No type annotation
```

## Decision

### 1. Use `X | None` syntax for nullable parameters
Use modern Python 3.10+ union syntax (`X | None`) instead of `Optional[X]` for all parameters with `= None` defaults.

**Rationale:**
- More concise and readable
- PEP 604 standard for Python 3.10+
- Preferred modern Python style

### 2. Standardize boundaries type to `np.ndarray`
Keep `boundaries: np.ndarray` in both files to match the actual implementation.

**Rationale:**
- The implementation stores `np.ndarray` and converts to `torch.Tensor` at usage time
- This lazy conversion pattern is valid and efficient
- Type annotations should reflect actual stored types, not intermediate conversions
- Maintains API simplicity for users (they pass `np.ndarray` from metadata)

**Alternative considered but rejected:**
- Using `torch.Tensor`: Would require immediate conversion, adding unnecessary overhead
- Using `Union[np.ndarray, torch.Tensor]`: Too verbose, adds complexity without benefit

### 3. Remove unused imports
Remove `from typing import Dict` from `learned_simulator.py`.

### 4. Add device type annotation
Change `device="cpu"` to `device: str = "cpu"` for consistency.

## Implementation

### Changes Required

#### learned_simulator.py
```python
# Remove: from typing import Dict
# No need to add imports - using built-in union syntax

class LearnedSimulator(nn.Module):
  def __init__(
          self,
          ...
          boundaries: np.ndarray,
          normalization_stats: dict,
          ...
          device: str = "cpu"  # Add type annotation
  ):
      ...

  def _encoder_preprocessor(
          self,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None  # Use | None
  ):
      ...

  def predict_positions(
          self,
          current_positions: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None  # Use | None
  ) -> torch.Tensor:
      ...

  def predict_accelerations(
          self,
          next_positions: torch.Tensor,
          position_sequence_noise: torch.Tensor,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          material_property: torch.Tensor | None = None  # Use | None
  ):
      ...
```

#### graph_model.py
```python
import numpy as np  # Add import for type annotation

class GraphNeuralNetworkModel(nn.Module):
  def __init__(
          self,
          ...
          boundaries: np.ndarray,  # Change from torch.Tensor
          ...
  ):
      ...

  def build_graph_features(
          self,
          position_sequence: torch.Tensor,
          nparticles_per_example: torch.Tensor,
          particle_types: torch.Tensor,
          normalized_velocity_sequence: torch.Tensor,
          material_property: torch.Tensor | None = None  # Use | None
  ):
      ...
```

## Consequences

### Positive
- Better type safety and IDE support
- Consistent type annotations across modules
- Modern Python syntax (PEP 604)
- No runtime behavior changes
- Type annotations match actual implementation
- Easier maintenance and debugging

### Negative
- Requires Python 3.10+ for `X | None` syntax
- Minor verbosity increase with union types

## Validation

After implementation:
1. Verify Python version is 3.10+ (already met by project requirements)
2. Run type checker (mypy/pylance) to verify no type errors
3. Run existing tests to ensure no runtime regressions
4. Verify training and rollout still work correctly

## Related Decisions

- Decision 0001: Keep LearnedSimulator as nn.Module

## Notes

- Python 3.10+ is already required by the project (using `uv` and modern tooling)
- The `X | None` syntax is clearer and more concise than `Optional[X]`
- Type annotations are purely static - no runtime overhead
