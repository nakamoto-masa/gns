# Decision 0005: Fix rollout() Docstring Shape Mismatch

**Date**: 2026-01-09
**Status**: Approved
**Decision By**: Development Team

## Background

During code review, a discrepancy was found between the docstring and actual implementation of `run_rollout()` in `gns/rollout.py`. The docstring claims the `position` parameter has shape `(timesteps, nparticles, ndims)`, but the actual implementation expects `(nparticles, timesteps, ndims)`.

### Investigation Results

This issue is **not caused by refactoring**. The docstring was already incorrect in the original codebase before Step 1.

#### Evidence: Original Code (Before Step 1)

**Data Loader** (`gns/data_loader.py` - unchanged since original):
```python
# TrajectoriesDataset.__getitem__
positions = np.transpose(positions, (1, 0, 2))  # Transpose to (nparticles, timesteps, ndims)
```

**Original rollout() function** (`gns/train.py` before commit 2096600):
```python
def rollout(position, ...):
    """
    Args:
        position: Positions of particles (timesteps, nparticles, ndims)  # ← Docstring (WRONG)
    """
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]  # ← Implementation expects (nparticles, timesteps, ndims)
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]
```

**Original predict() function**:
```python
positions = features[0].to(device)  # Receives (nparticles, timesteps, ndims) from data loader
sequence_length = positions.shape[1]  # Treats 2nd dimension as timesteps
```

The implementation clearly expects `position[:, :INPUT_SEQUENCE_LENGTH]` to slice along the timestep dimension (2nd axis), which means `position` must be `(nparticles, timesteps, ndims)`.

## Timeline of the Issue

| Stage | File | Docstring | Actual Shape | Status |
|-------|------|-----------|--------------|--------|
| **Before Step 1** | `gns/train.py::rollout()` | `(timesteps, nparticles, ndims)` | `(nparticles, timesteps, ndims)` | ❌ Already wrong |
| **Step 1** | `gns/inference_utils.py::rollout()` | `(nparticles, timesteps, ndims)` | `(nparticles, timesteps, ndims)` | ✅ Corrected |
| **Step 1** | `gns/train.py::rollout()` (wrapper) | `(timesteps, nparticles, ndims)` | `(nparticles, timesteps, ndims)` | ❌ Not updated |
| **Step 5** | `gns/rollout.py::run_rollout()` | `(timesteps, nparticles, ndims)` | `(nparticles, timesteps, ndims)` | ❌ Copied wrong docstring |

## Why This Happened

1. **Original docstring was incorrect** - The `rollout()` function docstring was wrong from the beginning
2. **Step 1 partially fixed it** - `inference_utils.rollout()` has the correct docstring
3. **Step 5 copied the wrong version** - When creating `gns/rollout.py`, the wrong docstring from the original wrapper was copied instead of the correct one from `inference_utils.py`

## Decision

**Fix the docstrings to match the actual data shape: `(nparticles, timesteps, ndims)`**

This is not a change or revert - it's a correction of documentation that was always incorrect.

## Implementation

### Files to Update

1. **`gns/rollout.py::run_rollout()`** (line 85)
   - Change: `position: Initial positions (timesteps, nparticles, ndims).`
   - To: `position: Initial positions (nparticles, timesteps, ndims).`

2. **`gns/rollout.py::rollout()`** (legacy wrapper, line 314)
   - Change: `position: Positions of particles (timesteps, nparticles, ndims)`
   - To: `position: Positions of particles (nparticles, timesteps, ndims)`

### Why This Shape?

The shape `(nparticles, timesteps, ndims)` is required because:

1. **Data loader returns this shape**: `TrajectoriesDataset` transposes to `(nparticles, timesteps, ndims)`
2. **Implementation expects this shape**: Slicing `position[:, :input_sequence_length]` treats 2nd dimension as time
3. **Consistent with training data**: `SamplesDataset` also uses `(nparticles, timesteps, ndims)` after transpose (line 99)

### Verification

The implementation has always been correct:
```python
# gns/inference_utils.py
initial_positions = position[:, :input_sequence_length]  # (nparticles, 6, ndims)
ground_truth_positions = position[:, input_sequence_length:]  # (nparticles, nsteps, ndims)
```

This slicing only makes sense if `position` is `(nparticles, timesteps, ndims)`.

## Impact

### Positive
- ✅ **Documentation matches reality** - Users will not be confused
- ✅ **Consistent with inference_utils.py** - Both docstrings now agree
- ✅ **No code changes needed** - Only documentation fix

### Neutral
- ℹ️ **No breaking changes** - Implementation remains unchanged

## Related Decisions

- **Related to**: Step 1 (inference_utils.py creation)
- **Related to**: Step 5 (rollout.py creation)

## Lessons Learned

1. **When refactoring, verify docstrings against implementation** - Don't blindly copy docstrings
2. **Check the correct source** - Step 5 should have copied from `inference_utils.py` (correct) not from the old wrapper (incorrect)
3. **Docstring bugs can persist through refactoring** - Automated tests don't catch documentation errors

## Notes

This is purely a documentation fix. The code has always worked correctly with `(nparticles, timesteps, ndims)` shape. The docstring was simply lying about what shape it expected.
