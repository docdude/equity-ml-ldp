# Critical Bug Fix: Double Normalization

## Problem Discovered

When training with the modular normalization system, features were being normalized **TWICE**:

1. **Step 2.5**: Normalized all data (train + val together)
2. **Step 4.5**: Normalized again (train-only fitting)

This caused:
- Extreme standard deviations (154 instead of 1.0)
- Incorrect feature distributions
- Model trained on wrong data

## Evidence

```
Training output BEFORE fix:
📊 Feature statistics AFTER normalization:
   Mean: -0.0420, Std: 154.0770  ❌ WRONG!
   Min: -195296.3181, Max: 5674.9564  ❌ Extreme!
```

## Root Cause

In `fin_training.py`:
```python
# Step 2.5 - WRONG: Normalize before split
X_seq, normalizer = normalize_sequences(X_seq, fit=True)  # ❌

# Step 4 - Split
X_train, X_val = split(X_seq)

# Step 4.5 - Normalize AGAIN
X_train, X_val, normalizer = normalize_for_training(X_train, X_val)  # ❌ Double normalized!
```

## Fix Applied

**Removed step 2.5 entirely:**

```python
# Step 2 - Create sequences (NO normalization)
X_seq = create_sequences(features)

# Show raw statistics
print(f"Raw feature statistics:")
print(f"   Mean: {X_seq.mean():.4f}, Std: {X_seq.std():.4f}")

# Step 4 - Split
X_train, X_val = split(X_seq)

# Step 4.5 - Normalize ONCE (train-only fitting)
X_train, X_val, normalizer = normalize_for_training(X_train, X_val)  # ✅ Correct!
```

## Expected Results After Fix

```
📊 Raw feature statistics (before normalization):
   Mean: ~0-20, Std: ~2-10
   Min: ~-50, Max: ~100

📊 AFTER normalization (train data):
   Mean: ~0.00, Std: ~1.00  ✅ CORRECT!
   Min: ~-3, Max: ~3
```

## Impact

- **Before fix**: Model trained on garbage (double-normalized) data
- **After fix**: Model trains on properly normalized features
- **Need to**: Retrain model from scratch with fix

## Related Issues

The OBV/A/D rate of change features (`obv_roc`, `ad_roc`) still have extreme values (-748 to +328) but these are **before** normalization, so RobustScaler will handle them. No clipping needed at source - let the normalizer do its job.

## Files Changed

- `fin_training.py` - Removed step 2.5 (double normalization)

## Next Steps

1. ✅ Fix applied to `fin_training.py`
2. ⏳ Retrain model with fixed normalization
3. ⏳ Verify statistics look correct (std ~1.0)
4. ⏳ Test PBO with properly trained model
5. ⏳ Compare AUC before/after fix
