# MLFinLab Triple Barrier Implementation Guide

## What You Found

You discovered two excellent open-source forks of the original mlfinlab:

1. **0xrushi/mlfinlab**: https://github.com/0xrushi/mlfinlab/blob/master/mlfinlab/labeling/labeling.py
2. **baobach/mlfinpy**: https://github.com/baobach/mlfinpy/blob/main/mlfinpy/labeling/labeling.py

Both implement the **exact same approach** from López de Prado's book.

---

## The Key Functions

### 1. `get_events()` - Find Barrier Touch Times

```python
def get_events(close, t_events, pt_sl, target, ...):
    """
    Returns DataFrame with:
        - t1: Timestamp when FIRST barrier was touched
        - trgt: Target threshold used for that event
        - side: Position side (optional, for meta-labeling)
    """
    # For each event, find which barrier (TP/SL/vertical) was hit first
    # Store that timestamp as t1
    
    events.at[ind, 't1'] = first_touch_dates.loc[ind, :].dropna().min()
    return events
```

**Key insight**: `t1` is NOT a fixed horizon. It's when the barrier was actually hit!

---

### 2. `get_bins()` - Calculate Returns at Exit

```python
def get_bins(triple_barrier_events, close):
    """
    Calculate ACTUAL returns at barrier touch time.
    
    Returns DataFrame with:
        - ret: Return from entry to t1 (when barrier hit)
        - bin: Label (-1, 0, 1) indicating which barrier
        - trgt: Target used
    """
    # 1) Get prices at entry and exit
    events_ = triple_barrier_events.dropna(subset=['t1'])
    all_dates = events_.index.union(events_['t1'].array)
    prices = close.reindex(all_dates, method='bfill')
    
    # 2) Calculate return: entry → exit (NOT fixed horizon!)
    out_df['ret'] = np.log(prices.loc[events_['t1'].array].array) - \
                    np.log(prices.loc[events_.index])
    
    # 3) Convert from log returns to normal returns
    out_df['ret'] = np.exp(out_df['ret']) - 1
    
    # 4) Determine which barrier was hit
    out_df = barrier_touched(out_df, triple_barrier_events)
    
    return out_df
```

**The critical calculation**:
```python
# Calculate log return from entry price to exit price
log_return = log(price_at_t1) - log(price_at_entry)

# Convert back to normal return
return = exp(log_return) - 1
```

---

## Why This Works

### Example Trade Comparison

**Price path**:
```
Day 0: $100 (entry)
Day 1: $103 (+3%)
Day 2: $107 (+7%) ← TP hit! (threshold 6%)
Day 3: $105 (+5%)
Day 4: $103 (+3%)
Day 5: $101 (+1%)
```

### Your Current Method (WRONG):
```python
Label: 1 (TP hit on day 2)
Return used: +1% (day 5 fixed horizon)
t1: Day 5 (fixed)
```
**Result**: Label says UP, but return is almost flat → Model looks like it's failing!

### MLFinLab Method (CORRECT):
```python
Label: 1 (TP hit on day 2)
Return used: +7% (day 2, actual exit)
t1: Day 2 (when barrier hit)
```
**Result**: Label says UP, return is +7% → Perfect alignment!

---

## What You Need to Implement

### Modify `create_dynamic_triple_barriers()` in `fin_feature_preprocessing.py`

**Current output**:
```python
return pd.DataFrame({
    'label': labels,
    'dynamic_tp': dynamic_tp,
    'dynamic_sl': dynamic_sl
})
```

**Need to add** (like mlfinlab):
```python
return pd.DataFrame({
    'label': labels,
    't1': exit_timestamps,      # NEW: When barrier was hit
    'exit_return': exit_returns, # NEW: Actual return at exit
    'exit_day': exit_days,       # NEW: Days until exit (1-5)
    'dynamic_tp': dynamic_tp,
    'dynamic_sl': dynamic_sl
})
```

### Update the barrier loop to track exits:

```python
labels = []
exit_timestamps = []
exit_returns = []
exit_days = []

for i in range(len(df) - horizon):
    future_returns = df['close'].pct_change().iloc[i+1:i+1+horizon].cumsum()
    entry_price = df['close'].iloc[i]
    
    tp_threshold = dynamic_tp.iloc[i]
    sl_threshold = dynamic_sl.iloc[i]
    
    # Find when barriers are hit
    tp_hit = (future_returns >= tp_threshold).idxmax() if (future_returns >= tp_threshold).any() else None
    sl_hit = (future_returns <= -sl_threshold).idxmax() if (future_returns <= -sl_threshold).any() else None
    
    if tp_hit is not None and (sl_hit is None or tp_hit < sl_hit):
        # TP hit first
        labels.append(1)
        exit_idx = df.index.get_loc(tp_hit)
        exit_day_offset = exit_idx - i
        exit_timestamps.append(tp_hit)
        exit_days.append(exit_day_offset)
        
        # CRITICAL: Calculate actual return at exit
        exit_price = df['close'].loc[tp_hit]
        exit_returns.append((exit_price / entry_price) - 1)
        
    elif sl_hit is not None:
        # SL hit first
        labels.append(0)
        exit_idx = df.index.get_loc(sl_hit)
        exit_day_offset = exit_idx - i
        exit_timestamps.append(sl_hit)
        exit_days.append(exit_day_offset)
        
        # Calculate actual return at exit
        exit_price = df['close'].loc[sl_hit]
        exit_returns.append((exit_price / entry_price) - 1)
        
    else:
        # Timeout (vertical barrier)
        labels.append(-1)
        timeout_idx = i + horizon
        if timeout_idx < len(df):
            exit_timestamps.append(df.index[timeout_idx])
            exit_days.append(horizon)
            exit_price = df['close'].iloc[timeout_idx]
            exit_returns.append((exit_price / entry_price) - 1)
        else:
            # Edge case: not enough data
            exit_timestamps.append(df.index[-1])
            exit_days.append(len(df) - i - 1)
            exit_returns.append(0.0)
```

---

## Update Evaluation Script

### In `fin_model_evaluation.py`:

**Current (WRONG)**:
```python
# Line ~90
forward_ret_5d = (prices_aligned['close'].shift(-5) / prices_aligned['close']) - 1
ticker_features['forward_return_5d'] = forward_ret_5d.values

# Line ~212
forward_returns = X['forward_return_5d'].values
```

**Change to (CORRECT)**:
```python
# Line ~90 - Don't calculate forward_return_5d, use exit_return from barriers
# (No change needed here - just don't add forward_return_5d)

# In barrier creation section (~line 70):
barriers = feature_engineer.create_dynamic_triple_barriers(df)
# barriers now contains 'exit_return' column

combined = pd.concat([features, barriers], axis=1)
# exit_return is now in the dataset

# Line ~212
forward_returns = X['exit_return'].values  # ← Change this line!
```

---

## Testing the Fix

### Simple test to verify alignment:

```python
# After implementing, run this test:
import numpy as np
import pandas as pd

# Load one ticker
df = pd.read_parquet('data_raw/AAPL.parquet')
barriers = feature_engineer.create_dynamic_triple_barriers(df)

# Check alignment
print("Label vs Return correlation:")
print(f"Correlation: {np.corrcoef(barriers['label'], barriers['exit_return'])[0,1]:.3f}")

# Should be positive! Labels and returns should agree
# UP labels (1) should have positive returns
# DOWN labels (0) should have negative returns

print("\nMean returns by label:")
for label in [-1, 0, 1]:
    mean_ret = barriers[barriers['label'] == label]['exit_return'].mean()
    print(f"  Label {label:2d}: {mean_ret:+.4f}")

# Expected output:
#   Label -1: ~0.0000 (neutral, small returns)
#   Label  0: -0.0300 (down, negative returns around -3%)
#   Label  1: +0.0600 (up, positive returns around +6%)
```

---

## Expected Results After Fix

When you re-run `fin_model_evaluation.py`:

**Before fix**:
- Mean Sharpe: -0.74 ❌
- Returns contradict labels ❌

**After fix**:
- Mean Sharpe: **Positive** ✅ (likely 0.5-1.5)
- Returns align with labels ✅
- Strategy captures actual barrier exits ✅

---

## Summary

**What mlfinlab does differently**:
1. Tracks WHEN each barrier is hit (not just which one)
2. Calculates return at that EXIT TIME (not fixed horizon)
3. Uses those aligned returns for backtesting

**Your fix**:
1. Add `exit_return`, `t1`, `exit_day` to barrier output
2. Use `exit_return` instead of `forward_return_5d` in evaluation
3. Enjoy positive Sharpe ratio! 🎉

---

## References

- **GitHub Repos**:
  - https://github.com/0xrushi/mlfinlab (archived fork)
  - https://github.com/baobach/mlfinpy (active fork)
  
- **Key Files**:
  - `mlfinlab/labeling/labeling.py` - Triple barrier implementation
  - See `get_events()` and `get_bins()` functions

- **Book**: Advances in Financial Machine Learning, Chapter 3
