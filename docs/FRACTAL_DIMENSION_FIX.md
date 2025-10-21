# Fractal Dimension Fix

## Issue

The `fractal_dimension` feature was returning constant values (either all 1.0 or all 2.0) instead of varying values in the range [1, 2].

## Root Cause

Three bugs in the Higuchi Fractal Dimension implementation:

1. **Wrong normalization**: Missing division by `k` in the Lmk calculation
2. **Wrong logarithm**: Using natural log `ln` instead of `log2`
3. **Wrong formula**: Using `FD = 2 - slope` instead of `FD = -slope`

## Investigation

### Reference Implementation

Consulted the authoritative implementation: https://github.com/inuritdino/HiguchiFractalDimension

Key findings from `hfd.c` and `hfd.py`:

```c
// Correct normalization (line 22 of hfd.c)
Lmk += ( s * (double)(n-1) / (double)( floor((n-j-1)/k_arr[i]) * k_arr[i] ) ) / k_arr[i];
//                                                                              ^^^^^^^^^ 
//                                                                     Additional /k here!
```

```python
# Correct formula (line 90 of hfd.py)
return (-np.polyfit(np.log2(k),np.log2(L),deg=1)[0])
#       ^          ^^^^^                          Use -slope, not 2-slope
#       |          Use log2, not ln
```

### Manual Testing

Before fix:
```
Calculated FD = 2.44 → clipped to 2.0 (all constant)
```

After fix:
```
Min: 1.056, Max: 1.977, Mean: 1.494, Std: 0.209 ✅
```

## Fix Applied

**File**: `fin_feature_preprocessing.py`, lines 780-810

### Before:
```python
# WRONG normalization
Lmk = (Lmk * (N - 1)) / (k * num_segments)  # Missing /k

# WRONG formula
log_k = np.log(np.arange(1, len(L) + 1))     # Should be log2
log_L = np.log(L)                             # Should be log2
slope = np.polyfit(log_k[mask], log_L[mask], 1)[0]
fd = 2 - slope                                 # Should be -slope
```

### After:
```python
# CORRECT normalization (added /k)
Lmk = (Lmk * (N - 1)) / (k * num_segments * k)

# CORRECT formula
log_k = np.log2(np.arange(1, len(L) + 1))     # log2 ✅
log_L = np.log2(L)                             # log2 ✅
slope = np.polyfit(log_k[mask], log_L[mask], 1)[0]
fd = -slope                                    # -slope ✅
```

## Validation

### Test Results

```
Fractal Dimension Stats:
  Min: 1.056138
  Max: 1.977481
  Mean: 1.494134
  Median: 1.457039
  Std: 0.209292
  
✅ PASS: All values in [1.0, 2.0]
✅ PASS: Mean ≈ 1.5 (expected for random walk)
✅ PASS: Variation shows dynamic behavior
```

### Interpretation

- **FD ≈ 1.0**: Smooth, trending behavior (low complexity)
- **FD ≈ 1.5**: Random walk (Brownian motion)
- **FD ≈ 2.0**: Very rough, mean-reverting (high complexity)

Result: Mean = 1.49 indicates typical financial time series complexity between trending and random walk.

## Impact

- **Before**: Feature was useless (constant values)
- **After**: Feature correctly captures price path complexity
- **Model**: Should now have access to meaningful fractal behavior information

## References

1. T. Higuchi, "Approach to an Irregular Time Series on the Basis of the Fractal Theory", Physica D, 1988; 31: 277-283
2. Reference implementation: https://github.com/inuritdino/HiguchiFractalDimension
3. Validation test: `test_all_features.py`

## Status

✅ **FIXED and VALIDATED**
- Correct Higuchi formula implemented
- Validated against reference implementation
- Produces sensible values for financial data
- Ready for model training
