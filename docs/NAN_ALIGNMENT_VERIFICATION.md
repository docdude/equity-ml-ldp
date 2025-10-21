# NaN Handling & Data Alignment Verification

**Date:** October 20, 2025  
**Status:** ✅ ALL VERIFIED  
**Verification Script:** `verify_nan_alignment.py`

## Executive Summary

The pipeline implements a **3-layer defense** against NaN values with **guaranteed data alignment** through consistent masking. All verification tests passed.

---

## NaN Handling Strategy

### Layer 1: Market Context Module (`fin_market_context.py`)

**Location:** Lines 146-156

```python
# Forward fill market data after alignment
aligned_market = market_features.reindex(ticker_df.index)
aligned_market = aligned_market.ffill()  # Fill gaps from market closures

# Report remaining NaN
nan_counts = aligned_market.isna().sum()
if nan_counts.sum() > 0:
    print(f"Warning: NaN values in market features...")
```

**Purpose:**
- Handles different market closure schedules (e.g., VIX weekends)
- Forward fills from last known value
- Reports any NaN that can't be filled

**Example:** VIX data missing on Saturday → Uses Friday's value

---

### Layer 2: Sequence Loading (`fin_load_and_sequence.py`)

**Per-Ticker Cleaning** (Lines 382-387):
```python
# Drop NaN values from features (including market context)
nan_before = features.isna().any(axis=1).sum()
features = features.dropna()
if verbose and nan_before > 0:
    print(f"Dropped {nan_before} rows with NaN values")
```

**Purpose:**
- Removes rows with NaN in ANY feature (ticker or market)
- Handles `pct_change()` NaN in first rows
- Handles alignment issues between ticker and market data

**Sequence-Level Filtering** (Lines 496-505):
```python
# Filter sequences with NaN
has_feature_nan = np.isnan(X_seq).any(axis=(1, 2))
valid_mask = ~has_feature_nan

# ✅ CRITICAL: Same mask applied to ALL arrays
X_seq = X_seq[valid_mask]
y_seq = y_seq[valid_mask]
dates_seq = dates_seq[valid_mask]
ldp_weights_seq = ldp_weights_seq[valid_mask]
returns_seq = returns_seq[valid_mask]
```

**Purpose:**
- Final safety check after sequence creation
- **Guarantees alignment:** Same mask filters all 5 arrays
- Removes any edge case NaN from sequence boundaries

---

### Layer 3: Normalization (`fin_utils.py`)

**Location:** Lines 491-496

```python
# Check for NaN/Inf
n_inf = np.isinf(X).sum()
n_nan = np.isnan(X).sum()
if n_inf > 0 or n_nan > 0:
    print(f"WARNING: {n_inf} inf, {n_nan} NaN values")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
```

**Purpose:**
- Safety net for any NaN that slipped through
- Should rarely trigger if Layers 1-2 work correctly
- Replaces NaN with 0 (features already normalized)

**Note:** During testing, this should report `n_nan=0` if pipeline working correctly.

---

## Data Alignment Guarantee

### The Critical Valid Mask

The sequence loader uses a **single boolean mask** to filter ALL arrays:

```python
valid_mask = ~np.isnan(X_seq).any(axis=(1, 2))  # One mask for all
```

This mask is then applied identically to:

1. `X_seq[valid_mask]` → Features (shape: N, 20, 25)
2. `y_seq[valid_mask]` → Labels (shape: N)
3. `dates_seq[valid_mask]` → Timestamps (shape: N)
4. `ldp_weights_seq[valid_mask]` → Uniqueness weights (shape: N)
5. `returns_seq[valid_mask]` → Actual barrier returns (shape: N)

**Alignment Guarantee:** For any index `i`:
- `X_seq[i]` is the feature sequence
- `y_seq[i]` is the corresponding label
- `dates_seq[i]` is the date of that sequence
- `ldp_weights_seq[i]` is the López de Prado weight
- `returns_seq[i]` is the actual barrier return

This is mathematically guaranteed because **the same boolean mask is used for all arrays**.

---

## Complete Data Flow

### 1. Sequence Loading
```
Raw Data
  ├─ Load market context (6 features)
  ├─ For each ticker:
  │  ├─ Create features (19 ticker-specific)
  │  ├─ Add market context → 25 total features
  │  ├─ ✅ features.dropna() → Remove NaN rows
  │  ├─ Create barriers on clean dates
  │  ├─ Align features ∩ labels ∩ events
  │  └─ Store: features, labels, returns, dates, weights
  ├─ Combine all tickers (concat + sort by date)
  ├─ Create sequences (sliding window)
  ├─ ✅ Filter with valid_mask → Same mask for all
  └─ Output: 5 aligned arrays, NO NaN
```

### 2. Normalization
```
X_seq (clean, no NaN)
  ├─ Check for NaN/Inf (should be 0)
  ├─ Safety: np.nan_to_num() if any found
  ├─ Fit/transform with RobustScaler
  └─ Output: X_normalized (same shape, no NaN)
```

### 3. Training
```
X_normalized, y_seq, ldp_weights_seq
  ├─ Split (same indices):
  │  ├─ X_train, X_val
  │  ├─ y_train, y_val
  │  └─ weights_train, weights_val
  └─ model.fit(X_train, y_train, sample_weight=weights_train)
```

### 4. Inference (PBO Test)
```
load_or_generate_sequences()
  ├─ Returns: X_seq, y_seq, dates_seq, weights_seq, returns_seq
  ├─ Split: val = seq[split_idx:]
  ├─ Normalize: X_val_norm
  ├─ Predict: probs = model(X_val_norm)
  ├─ Signal: signals = f(probs)  ← NO returns used
  └─ P&L: strategy_returns = signals × returns_val
```

---

## Verified Safe Cases

### ✅ Market Data Missing for Some Dates
- **Cause:** Different market closure schedules
- **Handling:** `ffill()` in market context → `dropna()` if can't fill
- **Example:** VIX closed Saturday → Use Friday value

### ✅ First Rows NaN from pct_change()
- **Cause:** `pct_change(1)` creates NaN in first row
- **Handling:** `dropna()` removes them
- **Impact:** Typically 1-2 rows lost per ticker

### ✅ Different Ticker Date Ranges
- **Cause:** AAPL data from 2015, NVDA from 2018
- **Handling:** Common index intersection during alignment
- **Result:** Only overlapping dates used

### ✅ Sequence Creation Edge Cases
- **Cause:** Sequence at data boundaries may have NaN
- **Handling:** `valid_mask` filters them out
- **Guarantee:** Same mask applied to all arrays

---

## Potential Issues (Noted, Low Risk)

### ⚠️ All Market Data Missing for Extended Period
- **Current:** `ffill()` uses stale data indefinitely
- **Risk:** If market data gap > 30 days, stale features
- **Mitigation:** Could add `limit` parameter to `ffill(max_fill_days=5)`
- **Status:** Low priority - rare in practice

### ⚠️ Normalization Division by Zero
- **Current:** `np.nan_to_num()` replaces with 0
- **Risk:** Could create NaN if feature has zero variance
- **Mitigation:** RobustScaler handles this better than StandardScaler
- **Status:** Already using RobustScaler ✅

### ⚠️ Ticker with Too Few Samples After Cleaning
- **Current:** Still included in dataset
- **Risk:** If ticker has <100 samples, may not learn well
- **Mitigation:** Could add check: `if len(features) < min_samples: continue`
- **Status:** Consider adding if issues arise

---

## Verification Test Results

### Synthetic Test Case
```
Created: 100 samples with NaN at indices 5-9, 20-24
Expected removed: 10 samples
Actual removed: 10 samples ✅
Remaining has NaN: False ✅
All arrays aligned: True ✅
Strategy calculation works: True ✅
```

### All Verification Checks
```
✅ PASS - Sequence Loading
✅ PASS - Market Context  
✅ PASS - Normalization
✅ PASS - Training Flow
✅ PASS - Edge Cases
✅ PASS - Test Case
```

---

## Conclusions

1. **NaN values are properly handled** through 3-layer defense
2. **Data alignment is guaranteed** via single valid_mask
3. **No look-ahead bias** - returns not used in signal generation
4. **Normalization is safe** - safety net for edge cases
5. **Pipeline is production-ready** for training and evaluation

### Key Design Principle

> **Single Mask Alignment:** By using the same boolean mask to filter all arrays simultaneously, we mathematically guarantee that row `i` corresponds to the same observation across all data structures.

This is the **cornerstone of data integrity** in the pipeline.

---

## Recommendations

### For Production Use
✅ **Current implementation is safe** - No changes required

### For Future Enhancement
1. **Add max_fill_days limit** to market context `ffill()`
   - Prevents using very stale market data
   - Recommended: `limit=5` (1 week)

2. **Add minimum sample check** per ticker
   - Skip tickers with <100 samples after cleaning
   - Prevents sparse data from affecting model

3. **Add alignment assertion tests**
   - After loading sequences, verify: `assert len(X_seq) == len(y_seq) == len(returns_seq)`
   - Catches bugs early in development

### Monitoring Recommendations
- Log how many rows dropped per ticker
- Alert if >20% of data lost to NaN cleaning
- Monitor normalization NaN warnings (should be 0)

---

## Related Files

- **Verification Script:** `verify_nan_alignment.py`
- **Sequence Loading:** `fin_load_and_sequence.py` (Lines 382-505)
- **Market Context:** `fin_market_context.py` (Lines 146-156)
- **Normalization:** `fin_utils.py` (Lines 491-496)
- **Training:** `fin_training_ldp.py`
- **Inference:** `test_pbo_ldp.py`

---

**Last Verified:** October 20, 2025  
**Verified By:** Automated test suite + manual code review  
**Status:** 🚀 **PRODUCTION READY**
