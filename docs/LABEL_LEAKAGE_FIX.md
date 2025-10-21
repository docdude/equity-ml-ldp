# CRITICAL: Label Leakage Detected in Features

## 🚨 Issue Found

### Forward Returns in Features = Label Leakage!

**Location:** `fin_feature_preprocessing.py` lines 108-112

```python
# Forward returns (for labeling/targets) ← THESE ARE LABELS, NOT FEATURES!
for horizon in [1, 3, 5]:
    features[f'forward_return_{horizon}d'] = np.log(
        df['close'].shift(-horizon) / df['close']
    )
```

### Why This Is a Problem

1. **Future Information:** Forward returns use `.shift(-horizon)` which looks into the future
2. **Perfect Prediction:** The model sees tomorrow's returns when predicting today
3. **Won't Work in Production:** Future data won't be available in real trading

### Evidence from Feature Importance

```
Top 10 Features:
    1. forward_return_5d_t19: 0.1144  ← FUTURE DATA!
    2. forward_return_3d_t19: 0.1023  ← FUTURE DATA!
    3. forward_return_1d_t19: 0.0344  ← FUTURE DATA!
    4. atr_ratio_t19: 0.0234          ← Real feature
    5. volatility_gk_20_t19: 0.0221   ← Real feature
```

**26% of model's predictive power comes from looking at the future!**

### Impact on Evaluation

- **AUC = 0.9698** → Artificially inflated
- **PBO = 0.695** → Correctly detected overfitting
- **Real AUC (without leakage):** Likely 0.70-0.80

---

## ✅ Solution

### Option 1: Remove Forward Returns Entirely (Recommended)

**These features should NEVER be in the feature set** - they're for label creation only.

```python
# In fin_feature_preprocessing.py, REMOVE lines 108-112:
# DELETE THIS SECTION:
# for horizon in [1, 3, 5]:
#     features[f'forward_return_{horizon}d'] = np.log(
#         df['close'].shift(-horizon) / df['close']
#     )
```

Expected outcome:
- Features: 73 → 70 (removes 3 forward return features)
- AUC will drop but be more realistic
- PBO should improve (drop below 0.50)

### Option 2: Use for Labeling Only (If Needed Elsewhere)

If forward returns are used for creating labels:
1. Calculate them separately
2. Never add them to the features DataFrame
3. Use only for y labels

---

## 🔍 Verification Steps

### After Removing Forward Returns:

1. **Re-run feature generation:**
   ```bash
   python fin_training.py
   ```

2. **Expected output:**
   ```
   ✅ Created 70 features  (was 73)
   ```

3. **Check feature importance:**
   ```
   Top features should be:
   - Volatility indicators (volatility_yz, volatility_gk)
   - Momentum indicators (RSI, MACD)
   - Trend indicators (ADX, Aroon)
   - NOT forward returns!
   ```

4. **Verify metrics:**
   ```
   AUC: Should remain > 0.65 (if features are truly predictive)
   PBO: Should drop < 0.50 (indicates less overfitting)
   ```

### If AUC Drops Significantly (< 0.60):

This means the model was heavily relying on label leakage. Actions:

1. **Add more sophisticated features:**
   - Enable 'microstructure' group (spreads, liquidity)
   - Enable 'entropy' group (complexity measures)
   - Enable 'regime' group (market state)

2. **Increase feature engineering quality:**
   - Use longer lookback periods
   - Add cross-sectional features (relative to SPY)
   - Engineer interaction features

3. **Consider different labeling:**
   - Triple barrier method
   - Meta-labeling
   - Regime-specific labels

---

## 📊 López de Prado Validation Checklist

Before training CNN-LSTM, verify:

- [ ] No forward-looking features (no `.shift(-n)`)
- [ ] Purged K-Fold CV passes (AUC > 0.65)
- [ ] Walk-Forward stable (AUC range < 0.15)
- [ ] PBO acceptable (< 0.50)
- [ ] Feature importance makes intuitive sense
- [ ] Top features are available in production
- [ ] Embargo properly applied (2% of fold size)

---

## 🎯 Next Steps

1. **Immediate:** Remove forward returns from features
2. **Validate:** Re-run evaluation, check metrics
3. **If needed:** Enable additional feature groups
4. **Final check:** Run comprehensive López de Prado evaluation
5. **Only then:** Proceed with CNN-LSTM training

**DO NOT train the CNN-LSTM until the validation passes with clean features!**

Otherwise you'll have a model that:
- Shows amazing backtest results
- Fails completely in production
- Wastes compute resources on retraining
- Produces false confidence in the strategy
