# López de Prado Triple Barrier Method - The Correct Way

## Summary from "Advances in Financial Machine Learning" (2018)

### Chapter 3: Labeling

López de Prado's triple barrier method is designed to create **better labels** for ML, but the key insight is:

**The barrier defines the LABEL, and you must track the ACTUAL RETURN at barrier touch for backtesting.**

---

## The mlfinlab Implementation (Official)

### Step 1: Define Barriers (`get_events`)

```python
def get_events(close, t_events, pt_sl, target, min_ret, num_threads, 
               vertical_barrier_times=False, side_prediction=None):
    """
    Get timestamps of when barriers are touched.
    
    Returns:
        DataFrame with columns:
        - t1: timestamp when first barrier was touched
        - trgt: the target (threshold) for that event
        - side: position side if using meta-labeling
    """
    # For each t_events (potential entry point):
    # 1. Set upper barrier at close * (1 + target)
    # 2. Set lower barrier at close * (1 - target * pt_sl[0])
    # 3. Set vertical barrier at t + vertical_barrier_times
    # 4. Find which barrier is touched FIRST
    # 5. Record that timestamp as t1
```

### Step 2: Calculate Returns at Barrier Touch (`get_bins`)

```python
def get_bins(events, close):
    """
    Compute labels and returns.
    
    Returns:
        DataFrame with columns:
        - ret: ACTUAL return from entry to t1 (when barrier hit)
        - bin: label (-1, 0, 1) indicating which barrier hit
    """
    # For each event:
    # 1. Get entry price at t_events
    # 2. Get exit price at t1 (barrier touch time)
    # 3. Calculate: ret = (exit_price - entry_price) / entry_price
    # 4. Assign label based on which barrier was hit
```

### Step 3: Meta-Labeling (Advanced)

```python
def get_meta_labels(events, close, side_prediction):
    """
    Create meta-labels for position sizing.
    
    Meta-label = 1 if the primary prediction was CORRECT and PROFITABLE
    Meta-label = 0 if should not bet
    """
    # For each event:
    # 1. Check if side_prediction matched actual return sign
    # 2. Check if return magnitude justified the bet
    # 3. Meta-label = 1 only if both conditions met
```

---

## Key Differences from Your Current Implementation

### ❌ Your Current Approach:
```python
# In create_dynamic_triple_barriers:
labels.append(1)  # TP hit

# In fin_model_evaluation:
forward_returns = X['forward_return_5d'].values  # Fixed 5-day return!
positions = np.where(pred_class == 2, 1, 0)
strategy_returns = positions * forward_returns
```

**Problem**: Label says "TP hit on day 2" but you use day 5 return!

---

### ✅ López de Prado's Approach:
```python
# In create_dynamic_triple_barriers (ENHANCED):
for i in range(len(df) - horizon):
    # Find which barrier hit and WHEN
    tp_hit_day = ...
    sl_hit_day = ...
    
    if tp_hit_day < sl_hit_day:
        labels.append(1)
        exit_day.append(tp_hit_day)
        # Calculate return at TP touch
        exit_return.append((df['close'].iloc[i + tp_hit_day] / df['close'].iloc[i]) - 1)
    elif sl_hit_day is not None:
        labels.append(0)
        exit_day.append(sl_hit_day)
        # Calculate return at SL touch
        exit_return.append((df['close'].iloc[i + sl_hit_day] / df['close'].iloc[i]) - 1)
    else:
        labels.append(-1)
        exit_day.append(horizon)
        # Calculate return at timeout
        exit_return.append((df['close'].iloc[i + horizon] / df['close'].iloc[i]) - 1)

# Return DataFrame with:
return pd.DataFrame({
    'label': labels,
    'exit_day': exit_day,
    'exit_return': exit_return,  # ← THIS is what you backtest with!
    't1': [df.index[i + ed] for i, ed in enumerate(exit_day)],  # Exit timestamp
})

# In fin_model_evaluation:
barrier_returns = X['exit_return'].values  # Use actual barrier returns!
positions = np.where(pred_class == 2, 1, 0)
strategy_returns = positions * barrier_returns  # Now aligned!
```

---

## Why This Matters

### Example Trade:

```
Day 0: Entry at $100
Day 1: Price $103 (+3%)
Day 2: Price $107 (+7%) ← TP hit! (threshold was 6%)
Day 3: Price $105 (+5%)
Day 4: Price $103 (+3%)
Day 5: Price $101 (+1%)
```

**Your current method:**
- Label: 1 (TP hit) ✅
- Return used: +1% (day 5) ❌
- **P&L: +1%** (missed most of the gain!)

**López de Prado's method:**
- Label: 1 (TP hit) ✅
- Return used: +7% (day 2, when exited) ✅
- **P&L: +7%** (captured the actual trade!)

---

## Implementation Priority

To fix your code:

1. ✅ **Modify `create_dynamic_triple_barriers`** to track:
   - `exit_day`: When barrier was hit (1-5 days)
   - `exit_return`: Actual return at exit
   - `t1`: Timestamp of exit

2. ✅ **Update feature engineering** to include `exit_return` column

3. ✅ **Modify `fin_model_evaluation.py`** to use `exit_return` instead of `forward_return_5d`

4. ⚠️ **Consider meta-labeling** (advanced, optional):
   - Train secondary model to predict bet size
   - Use only when confident

---

## References

1. **Advances in Financial Machine Learning** (2018) by Marcos López de Prado
   - Chapter 3: Labeling
   - Chapter 5: Fractionally Differentiated Features
   
2. **mlfinlab library** (Hudson & Thames)
   - `mlfinlab.labeling.labeling.get_events()`
   - `mlfinlab.labeling.labeling.get_bins()`
   - `mlfinlab.labeling.labeling.drop_labels()`

3. **Machine Learning for Asset Managers** (2020)
   - Chapter 4: Optimal Clustering

---

## Bottom Line

**López de Prado does NOT use fixed-horizon returns with barriers.**

He tracks when the barrier is hit and uses the return AT THAT MOMENT. This is the only way to properly align the label with the return for backtesting.

Your Option 2 from my previous answer is exactly what mlfinlab does!
