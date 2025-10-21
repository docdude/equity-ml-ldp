# Forward Returns Fix Applied to fin_model_evaluation.py

**Date:** October 15, 2025  
**Issue:** Need to ensure forward returns are calculated per-ticker to prevent cross-ticker contamination  
**Solution:** Applied same fix from `test_pbo_quick.py` to `fin_model_evaluation.py`

---

## Changes Made

### 1. Track Ticker and Prices During Feature Engineering

**Added variables:**
```python
all_prices = []
all_tickers = []
```

**Per-ticker processing:**
```python
for ticker in tickers:
    # ... feature engineering ...
    
    # Get aligned prices for forward return calculation
    prices_aligned = df.loc[combined.index, ['close']]
    
    # Store features (without forward_return_5d yet)
    ticker_features = combined[features.columns].copy()
    
    # Track ticker for each row (to prevent cross-ticker forward returns)
    all_tickers.extend([ticker] * len(combined))
    
    # Calculate 5-day forward returns BEFORE concatenation (within ticker)
    # This prevents cross-ticker contamination
    forward_ret_5d = (prices_aligned['close'].shift(-5) / prices_aligned['close']) - 1
    forward_ret_5d = forward_ret_5d.fillna(0)  # Fill NaN at end of series
    ticker_features['forward_return_5d'] = forward_ret_5d.values
    ticker_features['ticker'] = ticker
    
    all_features.append(ticker_features)
```

### 2. Updated Sorting to Include New Variables

```python
# Sort all aligned data structures
sort_idx = dates.argsort()
X = X.iloc[sort_idx].reset_index(drop=True)
y = y.iloc[sort_idx].reset_index(drop=True)
prices = prices.iloc[sort_idx].reset_index(drop=True)  # NEW
dates = dates[sort_idx]
tickers = tickers.iloc[sort_idx].reset_index(drop=True)  # NEW
```

### 3. Separate Model Features from Metadata

```python
# Separate ticker and forward_return_5d from model input features
model_feature_cols = [col for col in X.columns 
                      if col not in ['ticker', 'forward_return_5d']]
X_features = X[model_feature_cols]
print(f"Using {len(model_feature_cols)} features for model input")
```

**Result:** Model sees 69 features (not 71), excluding ticker and forward_return_5d

### 4. Use Real Forward Returns for Strategy Generation

**Keras Models:**
```python
# Get pre-calculated forward returns from features
forward_returns_all = X['forward_return_5d'].values

# Get forward returns for validation set (accounting for seq_len offset)
val_start_idx = val_split + seq_len
forward_returns = forward_returns_all[val_start_idx:val_start_idx + len(X_val)]

# Truncate predictions to match available forward returns
direction_probs = direction_probs[:len(forward_returns)]

print(f"Aligned {len(forward_returns)} predictions with actual returns")
print(f"Forward returns - mean: {forward_returns.mean():.6f}, std: {forward_returns.std():.6f}")

# Create strategies using actual returns
for min_conf in min_confidences:
    pred_class = direction_probs.argmax(axis=1)
    max_prob = direction_probs.max(axis=1)
    
    positions = np.where(
        (pred_class == 2) & (max_prob > min_conf), 
        max_prob - 0.33,  # Long: scaled by confidence
        np.where(
            (pred_class == 0) & (max_prob > min_conf),
            -(max_prob - 0.33),  # Short: scaled by confidence
            0  # No position
        )
    )
    
    # Apply positions to ACTUAL forward returns (not synthetic!)
    returns = positions * forward_returns
    strategy_returns.append(returns)
```

**Sklearn Models:**
```python
# Flatten sequences and apply same logic
X_val_flat = X_val.reshape(X_val.shape[0], -1)
probs = model.predict_proba(X_val_flat)
probs = probs[:len(forward_returns)]

# Use same position sizing with actual forward returns
for min_conf in min_confidences:
    pred_class = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)
    positions = ...  # Same logic
    returns = positions * forward_returns
    strategy_returns.append(returns)
```

### 5. Updated X_df Column Names

```python
# Use model_feature_cols (excludes ticker and forward_return_5d)
feature_names = []
for t in range(seq_len):
    for col in model_feature_cols:
        feature_names.append(f'{col}_t{t}')

X_df = pd.DataFrame(X_flat, index=dates_seq, columns=feature_names)
```

---

## Why This Matters

### Problem with Cross-Ticker Contamination

**Before Fix:**
```python
# BAD: Calculate forward returns AFTER concatenating all tickers
X = pd.concat(all_features)
forward_returns = X['close'].shift(-5) / X['close'] - 1
```

**Issue:** At ticker boundaries, forward return uses next ticker's price:
```
AAPL row 2432: forward_return = (DELL_price_day0 / AAPL_price_day2432) - 1
```
This creates impossible returns spanning different securities!

**After Fix:**
```python
# GOOD: Calculate forward returns WITHIN each ticker BEFORE concatenation
for ticker in tickers:
    prices_aligned = df.loc[combined.index, ['close']]
    forward_ret_5d = (prices_aligned['close'].shift(-5) / prices_aligned['close']) - 1
    ticker_features['forward_return_5d'] = forward_ret_5d.values
```

**Result:** Forward returns stay within ticker boundaries:
```
AAPL row 2432: forward_return = (AAPL_price_day2437 / AAPL_price_day2432) - 1  ✓
```

### Benefits of Using Real Returns

**Before (Synthetic Returns):**
```python
# Simulated returns based on predictions (placeholder)
returns = np.where(p_up > threshold, 0.001, 
                  np.where(p_down > threshold, -0.001, 0))
returns += np.random.normal(0, 0.005, len(returns))  # Add noise
```

**Problems:**
- Not realistic (constant magnitude)
- Ignores actual market conditions
- PBO analysis on fake data is meaningless

**After (Real Returns):**
```python
# Apply positions to ACTUAL forward returns
positions = model_confidence * direction
returns = positions * forward_returns  # Real market returns!
```

**Benefits:**
- Tests model on real market outcomes
- PBO analysis is meaningful
- Captures actual risk/reward profile
- Shows true strategy performance

---

## Validation

### Check 1: No Cross-Ticker Contamination
```python
# Verify forward returns are calculated per-ticker
for ticker in tickers:
    ticker_data = X[X['ticker'] == ticker]
    forward_rets = ticker_data['forward_return_5d']
    
    # Check no extreme values at boundaries
    assert forward_rets.max() < 2.0, f"{ticker} has impossible return"
    assert forward_rets.min() > -0.9, f"{ticker} has impossible return"
```

### Check 2: Returns Aligned with Predictions
```python
# Verify same length after alignment
assert len(direction_probs) == len(forward_returns)
print(f"✅ Aligned {len(forward_returns)} predictions with returns")
```

### Check 3: Realistic Return Statistics
```python
print(f"Forward returns - mean: {forward_returns.mean():.6f}")
print(f"Forward returns - std: {forward_returns.std():.6f}")
print(f"Returns > 100%: {(forward_returns > 1.0).sum()}")  # Should be 0
print(f"Returns < -90%: {(forward_returns < -0.9).sum()}")  # Should be 0
```

---

## Impact on Evaluation Pipeline

### PBO Analysis
- **Before:** PBO on synthetic returns (meaningless)
- **After:** PBO on real strategy returns (actionable)

### Purged CV
- Still uses proper time-series CV
- No lookahead bias
- Embargo prevents label leakage

### Walk-Forward Analysis
- Tests on real out-of-sample data
- Shows actual performance degradation
- Meaningful for deployment decisions

### Feature Importance
- MDI and MDA use real target relationship
- More accurate feature rankings
- Better for feature selection

---

## Files Updated

1. **fin_model_evaluation.py**
   - Lines 51-98: Per-ticker forward return calculation
   - Lines 108-115: Updated sorting and tracking
   - Lines 127-133: Separate model features from metadata
   - Lines 164-171: Updated X_df column names
   - Lines 191-247: Real forward returns for Keras models
   - Lines 249-281: Real forward returns for Sklearn models

---

## Testing

Run the full evaluation suite:
```bash
cd /mnt/ssd_backup/equity-ml-ldp
.venv/bin/python -c "
from fin_model import load_model_with_custom_objects
from fin_model_evaluation import main

model = load_model_with_custom_objects('./run_financial_wavenet_v1/best_model.keras')
main(model=model, model_name='CNN-LSTM-WaveNet')
"
```

Expected output:
```
✅ Forward returns calculated per-ticker (no cross-contamination)
Features: 71 (including ticker and forward_return_5d)
Using 69 features for model input
Aligned X predictions with actual returns
Forward returns - mean: ~0.001-0.007, std: ~0.02-0.07
```

---

## Conclusion

The fix ensures:
1. ✅ **No cross-ticker contamination** in forward returns
2. ✅ **Real market returns** used for strategy evaluation
3. ✅ **Proper feature separation** (model doesn't see ticker/forward_return_5d)
4. ✅ **Meaningful PBO analysis** based on actual performance
5. ✅ **Consistent with test_pbo_quick.py** implementation

This makes the evaluation results trustworthy for deployment decisions.
