# Per-Ticker López de Prado Weight Calculation - Refactor Complete

## Summary

Refactored `fin_training_ldp.py` to calculate López de Prado sample weights **per-ticker** (inside the ticker processing loop) instead of globally (after combining all tickers). This fixes the alignment bug that was averaging out uniqueness signals.

## Problem Statement

**Original Implementation (BROKEN):**
```python
# Lines 320-420: Process each ticker
for ticker in tickers:
    all_events.append(events_aligned)
    all_close_series.append(prices_aligned)  # Save for later

# Lines 440-470: Combine all tickers
events_combined = pd.concat(all_events)
combined_close = pd.concat(all_close_series)

# Lines 620-720: Calculate weights GLOBALLY
num_conc_events = concurrent_events(events_combined.index, ...)
avg_unique = average_uniqueness(events_combined.index, ...)
ldp_weights = avg_unique * return_weights

# Lines 690-710: Date-based alignment to sequences
for seq_date in dates_sequences:
    weight = ldp_weights.loc[seq_date]  # PROBLEM: Duplicate dates
    if isinstance(weight, pd.Series):
        weight = weight.iloc[0]  # Just take first → loses info
```

**Observed Results:**
- Raw uniqueness: mean=0.019, range=[0.015, 0.058] ✅ Correct
- Aligned uniqueness: mean=0.987 ❌ **Averaged out!**
- Per-class weights: 1.06-1.11 (uniform) ❌ **No variation!**
- Expected per-class: 0.2-0.8 range with TIMEOUT<0.3, UP/DOWN>0.6

**Root Cause:**
Date-based lookup after combining all tickers lost per-sample granularity:
1. Multiple tickers have the same dates → duplicate index
2. `ldp_weights.loc[date]` returns Series when duplicates exist
3. Taking `.iloc[0]` throws away variation
4. Result: All weights average to ~1.0

## Solution: Per-Ticker Calculation

**New Implementation:**
```python
# Lines 318-324: Import LdP functions at module level
from FinancialMachineLearning.sample_weights.concurrency import (
    concurrent_events, average_uniqueness
)
from FinancialMachineLearning.sample_weights.attribution import weights_by_return

# Line 337: Store weights not close prices
all_ldp_weights = []  # NEW

# Lines 395-453: Calculate weights INSIDE ticker loop
for ticker in tickers:
    # ... create features, labels, events ...
    
    # CALCULATE WEIGHTS PER-TICKER (before combining!)
    try:
        # Step 1: Concurrent events
        num_conc_events = mp_pandas_obj(concurrent_events,
            ('molecule', events_aligned.index), num_threads=1,
            close_series_index=prices_aligned.index,
            label_endtime=events_aligned['t1'])
        
        # Step 2: Average uniqueness
        avg_unique = mp_pandas_obj(average_uniqueness,
            ('molecule', events_aligned.index), num_threads=1,
            label_endtime=events_aligned['t1'],
            num_conc_events=num_conc_events)
        
        # Step 3: Return attribution
        return_weights = weights_by_return(
            triple_barrier_events=events_aligned,
            close_series=prices_aligned, num_threads=1)
        
        # Step 4: Combine (uniqueness × return attribution)
        ldp_weight = avg_unique * return_weights
        ldp_weight = ldp_weight / ldp_weight.mean()
        
        # Diagnostics
        print(f"     Concurrency: mean={num_conc_events.mean():.1f}")
        print(f"     Uniqueness: mean={avg_unique.mean():.3f}, range=[{avg_unique.min():.3f}, {avg_unique.max():.3f}]")
        print(f"     LdP weights: range=[{ldp_weight.min():.3f}, {ldp_weight.max():.3f}]")
        
    except Exception as e:
        print(f"     ⚠️  Failed: {e}")
        ldp_weight = pd.Series(1.0, index=common_index)
    
    all_ldp_weights.append(ldp_weight)  # Stored WITH natural alignment!

# Lines 465-485: Combine with weights preserved
ldp_weights_combined = pd.concat(all_ldp_weights)
ldp_weights_combined = ldp_weights_combined.iloc[sort_idx].reset_index(drop=True)
# Weights stay aligned through same sorting!

# Lines 660-720: Simple positional mapping to sequences
ldp_weights_for_sequences = []
for i in range(len(X) - seq_len):
    seq_end_idx = i + seq_len - 1
    ldp_weights_for_sequences.append(ldp_weights_combined.iloc[seq_end_idx])

ldp_weights_for_sequences = np.array(ldp_weights_for_sequences)
ldp_weights_for_sequences = ldp_weights_for_sequences[valid_mask]

# Split and combine with class weights
ldp_weights_train = ldp_weights_for_sequences[:split_idx]
sample_weights_train = ldp_weights_train * simple_weights_train
```

## Key Benefits

1. **Natural Alignment**: Weights calculated with same index as data → no complex lookup
2. **Preserves Granularity**: No averaging across tickers → keeps 0.015-0.058 range
3. **Simple Mapping**: Positional indexing `[i + seq_len - 1]` → no date lookup
4. **No Duplicate Issues**: Each ticker's index is unique during calculation
5. **Future-Ready**: Enables Sequential Bootstrapping per-ticker
6. **Easier Debugging**: Can inspect weights per-ticker with diagnostics

## Expected Outcomes

After refactor:
- ✅ LdP weights (before sequences): Range [0.003-8.5], Mean=1.0, Std>1.0
- ✅ LdP weights (after sequences): Preserve range (not averaged to ~1.0)
- ✅ Per-class uniqueness variation:
  - Label 0 (DOWN/SL): uniqueness=0.6-0.8 (clean signals)
  - Label 1 (TIMEOUT): uniqueness=0.2-0.4 (overlap → down-weighted)
  - Label 2 (UP/TP): uniqueness=0.6-0.8 (clean signals)
- ✅ Training metrics improve: AUC>0.75 (vs 0.71), Accuracy>58% (vs 56%)

## Testing Plan

1. **Run Training**: `python fin_training_ldp.py`
2. **Check Per-Ticker Diagnostics**: Should see variation in uniqueness per ticker
3. **Verify Weight Alignment**: Compare "before sequences" vs "after sequences" ranges
4. **Validate Per-Class Weights**: TIMEOUT should have lower avg_weight than UP/DOWN
5. **Monitor Training**: Loss should decrease smoothly, no NaN/instability

## Next Steps (After Verification)

### Phase 1: Validate Refactor
- [x] Complete code refactor
- [ ] Test training pipeline end-to-end
- [ ] Verify per-class weight variation
- [ ] Compare metrics to baseline (AUC=0.71)

### Phase 2: Sequential Bootstrapping
Once weights work properly, implement per-ticker sampling:
```python
from FinancialMachineLearning.sample_weights.bootstrapping import (
    get_indicator_matrix, seq_bootstrap
)

for ticker in tickers:
    # ... calculate weights (existing) ...
    
    # Sample by uniqueness (NEW)
    ind_mat = get_indicator_matrix(events_aligned.index, events_aligned['t1'])
    phi = seq_bootstrap(ind_mat.values, sample_length=len(ind_mat))
    
    # Keep only sampled
    features_aligned = features_aligned.iloc[phi]
    labels_aligned = labels_aligned.iloc[phi]
    ldp_weight = ldp_weight.iloc[phi]
```

Expected improvements:
- Mean concurrency: 53 → 20 (less overlap)
- Mean uniqueness: 0.019 → 0.05 (higher quality)
- Training set smaller but cleaner
- Better generalization (lower PBO)

## Files Changed

- `fin_training_ldp.py`:
  - Lines 318-324: Added LdP imports at module level
  - Line 337: Changed `all_close_series` → `all_ldp_weights`
  - Lines 395-453: Added per-ticker weight calculation inside loop
  - Lines 465-485: Updated combine section to concat weights
  - Lines 660-720: Simplified weight alignment (positional indexing)
  - Lines 434: Fixed `num_threads=4` → `num_threads=1`

## References

- **Notebooks**: `MLFinance/Notes/01LabelConcurrency.ipynb` (proper implementation)
- **Diagnostic**: `compare_weights.py` (proved original was broken)
- **Discussion**: See conversation history for detailed debugging process
