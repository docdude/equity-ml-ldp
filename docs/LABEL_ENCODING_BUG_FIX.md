# Label Encoding Bug: A Case Study in ML Debugging

## 🐛 The Bug

Strategy was losing money (Sharpe ratio: -0.6166) despite good classification performance (AUC: 0.6899).

## 🔍 The Investigation

### Symptom 1: Negative Returns Despite Positive Market
```
Mean return per strategy: -0.000659 (NEGATIVE)
Mean Sharpe ratio: -0.6166 (NEGATIVE)
Forward returns mean: 0.009020 (POSITIVE)
AUC: 0.6899 (GOOD)
```

**Initial Reaction:** "Maybe it's just a bear market period?"
**Reality:** Root cause in label encoding.

### Symptom 2: Feature Importance Looked Good
```
TOP 10 FEATURES:
1. volatility_cc_60_t19: 0.0346
2. atr_ratio_t19: 0.0317
3. volatility_gk_20_t19: 0.0261
...
```

**Initial Reaction:** "Volatility features are working!"
**Reality:** They were working, but strategy mapping was broken.

### Symptom 3: Prediction vs Label Mismatch
```
Label distribution: {-1: 12084, 0: 7398, 1: 3949}
Prediction probabilities: [0.xxx, 0.xxx, 0.xxx]
```

**Initial Reaction:** "Model bias due to market conditions?"
**Reality:** Label encoding created wrong class mapping.

## 🎯 Root Cause Analysis

### The Data Flow
```
1. Triple Barriers Output
   Labels: {-1: neutral/timeout, 0: down/stop loss, 1: up/take profit}

2. Keras to_categorical(labels, num_classes=3)
   -1 → [0, 0, 1] → argmax returns 2
    0 → [1, 0, 0] → argmax returns 0
    1 → [0, 1, 0] → argmax returns 1

3. Model Training
   Learns to predict class indices: {0, 1, 2}
   
4. Strategy Logic (BEFORE FIX - WRONG!)
   pred_class == 2 → LONG  ❌ (This is neutral!)
   pred_class == 1 → SHORT ❌ (This is up!)
   pred_class == 0 → (unused) ❌ (This is down!)

5. Result
   - Never takes long positions (looking for class 2 = neutral)
   - Takes short positions on UP predictions (class 1)
   - Loses money despite good classification
```

## ✅ The Fix

### Correct Strategy Mapping
```python
# AFTER FIX - CORRECT!
positions = np.where(
    (pred_class == 1) & (max_prob > min_conf),  # Class 1 = UP → LONG ✅
    max_prob - 0.33,
    np.where(
        (pred_class == 0) & (max_prob > min_conf),  # Class 0 = DOWN → SHORT ✅
        -(max_prob - 0.33),
        0  # Class 2 = NEUTRAL → NO POSITION ✅
    )
)
```

## 📚 Lessons Learned

### 1. Trust But Verify, Always
- **Don't Assume:** `to_categorical()` handles negative values unexpectedly
- **Verify:** Test with actual data, not documentation alone
- **Cross-check:** Predictions should align with label distribution (accounting for model bias)

### 2. Fix Root Causes, No Bandaid Fixes
- ❌ **Bandaid:** "Let's just flip the sign on returns"
- ✅ **Root Cause:** "Let's trace exactly how labels are encoded through the pipeline"

### 3. Follow the Data Flow, Look for Anomalies
```
Data → Features → Labels → Encoding → Training → Predictions → Strategy → Returns
  ↓        ↓         ↓         ↓          ↓            ↓            ↓         ↓
 OK       OK        OK     BROKEN!       OK        WRONG!      INVERTED!  NEGATIVE!
```

### 4. Garbage In = Garbage Out
- **Feature Engineering:** Verified volatility estimators matched reference implementations
- **Label Encoding:** Verified to_categorical behavior with test script  
- **Strategy Mapping:** Verified class indices matched model output encoding
- **No Assumptions:** Only verification with data

## 🧪 Verification Method

### Test Script Used
```python
import numpy as np
import tensorflow as tf

# Test what to_categorical does with {-1, 0, 1}
labels = np.array([-1, 0, 1, -1, 0, 1])
categorical = tf.keras.utils.to_categorical(labels, num_classes=3)
print('to_categorical output:', categorical)
print('Argmax:', categorical.argmax(axis=1))

# Output:
# to_categorical output:
# [[0. 0. 1.]  ← -1 becomes class 2
#  [1. 0. 0.]  ←  0 becomes class 0
#  [0. 1. 0.]  ←  1 becomes class 1
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]]
# Argmax: [2 0 1 2 0 1]
```

## 📋 Debugging Checklist for Future

When you see good metrics but poor strategy performance:

1. ✅ **Verify label encoding matches strategy logic**
   - Print actual one-hot encodings
   - Test with small sample
   - Don't assume standard behavior with edge cases

2. ✅ **Trace the complete data flow**
   - Raw labels → Encoding → Training → Predictions → Strategy
   - Check each transformation step
   - Look for silent conversions/mappings

3. ✅ **Check for inconsistencies**
   - Label distribution vs prediction distribution
   - Forward returns vs strategy returns  
   - AUC vs Sharpe ratio alignment

4. ✅ **Validate with ground truth**
   - Test on known patterns
   - Manually calculate expected outcomes
   - Compare to reference implementations

## 🎓 Key Takeaways

### For ML Practitioners
1. **Library functions can have unexpected behavior with edge cases** (negative labels)
2. **Good classification metrics don't guarantee profitable strategies** (encoding bugs)
3. **Always verify the complete pipeline** (features → model → strategy → returns)

### For This Codebase
1. **Labels:** {-1: neutral, 0: down, 1: up}
2. **After Encoding:** {0: down, 1: up, 2: neutral}
3. **Strategy:** Long on 1, Short on 0, None on 2

### Red Flags to Watch For
- ✅ Good AUC but negative Sharpe → Check label/prediction mapping
- ✅ Prediction distribution != Label distribution → Check encoding
- ✅ Unexpected class indices in outputs → Verify one-hot encoding
- ✅ Strategy performance opposite to expectations → Trace data flow

---

## Files Fixed
- ✅ `fin_model_evaluation.py` - Strategy position logic corrected
- ✅ `test_pbo_quick.py` - Strategy position logic corrected
- ✅ `docs/TRAINING_PIPELINE_COMPLETE.md` - Added label encoding warnings

## Expected Results After Fix
- Positive Sharpe ratio (>0.5 target)
- Strategy returns correlate with forward returns
- Prediction distribution roughly matches label distribution (with model bias)
- Long positions on class 1 (up), short on class 0 (down)

**Date Fixed:** October 16, 2025
**Root Cause:** `to_categorical()` with negative values creates non-intuitive class mapping
**Impact:** High - Would have caused all strategies to fail despite good models
**Prevention:** Always verify encoding with test data, not assumptions
