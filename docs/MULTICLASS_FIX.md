# Multi-Class Classification Fix

## Issue

When running `fin_training.py` with the enhanced López de Prado evaluator, got this error:

```python
ValueError: multi_class must be in ('ovo', 'ovr')
```

## Root Cause

The labels have **3 classes, not 2**:
- **Label distribution: {-1: 12084, 0: 7398, 1: 3949}**
  - `-1`: Bearish (price will decrease)
  - `0`: Neutral (no significant move)
  - `1`: Bullish (price will increase)

The `roc_auc_score` function in sklearn requires the `multi_class` parameter when dealing with more than 2 classes, but our code was written for binary classification only.

## Fixes Applied

### 1. Fixed `roc_auc_score` call in `purged_cross_validation()`

**File**: `lopez_de_prado_evaluation.py`, lines 143-159

**Before**:
```python
if hasattr(model, 'predict_proba'):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
```

**After**:
```python
if hasattr(model, 'predict_proba'):
    y_pred_proba = model.predict_proba(X_test)
    n_classes = y_pred_proba.shape[1]
    
    if n_classes == 2:
        # Binary classification
        score = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        # Multi-class classification
        score = roc_auc_score(y_test, y_pred_proba, 
                            multi_class='ovr',  # One-vs-Rest
                            average='weighted')  # Weight by class frequency
```

### 2. Fixed Predictions Saving for Multi-Class

**File**: `lopez_de_prado_evaluation.py`, lines 179-197

**Issue**: `y_pred_proba` is a 2D array (n_samples, n_classes) for multi-class, but was trying to store it as a single column, causing:
```python
ValueError: Per-column arrays must each be 1-dimensional
```

**Solution**: Store each class probability as a separate column

**Before**:
```python
pred_df = pd.DataFrame({
    'index': X.index[test_idx],
    'y_true': y_test.values,
    'y_pred_proba': y_pred_proba,  # ERROR: 2D array!
    'y_pred_class': y_pred,
    'fold': fold + 1
})
```

**After**:
```python
pred_dict = {
    'index': X.index[test_idx],
    'y_true': y_test.values,
    'y_pred_class': y_pred,
    'fold': fold + 1
}

# Store probabilities for each class
if len(y_pred_proba.shape) == 2:
    # Multi-class: store each class probability
    for class_idx in range(y_pred_proba.shape[1]):
        pred_dict[f'y_pred_proba_class_{class_idx}'] = y_pred_proba[:, class_idx]
else:
    # Binary: store single probability
    pred_dict['y_pred_proba'] = y_pred_proba

pred_df = pd.DataFrame(pred_dict)
```

## Results

After fixes, the pipeline runs successfully:

```
Label distribution: {-1: 12084, 0: 7398, 1: 3949}

📊 Purged K-Fold Cross-Validation
   Fold 1/5 - AUC: 0.9025, Accuracy: 0.8123
   Fold 2/5 - AUC: 0.9216, Accuracy: 0.7830
   Fold 3/5 - AUC: 0.9154, Accuracy: 0.7766
   Fold 4/5 - AUC: 0.9218, Accuracy: 0.7619
   Fold 5/5 - AUC: 0.9148, Accuracy: 0.7589
   💾 Saved 23411 predictions
✅ PCV Results - AUC: 0.9152 ± 0.0070

📊 Walk-Forward Analysis
   Window type: Expanding
   Training window: 2000 periods (initial)
   Step size: 200 periods
   Embargo: 2.0% = 40 periods
   
   Period 1:
   Train:   2016-02-03 to 2017-01-24 (1960 samples)
   Embargo: 2017-01-24 to 2017-01-31 (40 samples) [EXCLUDED]
   Test:    2017-01-31 to 2017-03-08 (200 samples)
   AUC: 0.6473 | Accuracy: 0.4299
   ...
```

## Predictions DataFrame Structure

The saved predictions now have this structure:

```
columns:
- index: DatetimeIndex (original sample dates)
- y_true: True label (-1, 0, or 1)
- y_pred_class: Predicted label
- y_pred_proba_class_0: Probability of class -1 (bearish)
- y_pred_proba_class_1: Probability of class 0 (neutral)
- y_pred_proba_class_2: Probability of class 1 (bullish)
- fold: Cross-validation fold number
```

## Multi-Class AUC Interpretation

For multi-class with `multi_class='ovr'` and `average='weighted'`:

1. **One-vs-Rest (OVR)**: Calculate AUC for each class vs all other classes
   - Class -1 vs {0, 1}
   - Class 0 vs {-1, 1}
   - Class 1 vs {-1, 0}

2. **Weighted Average**: Weight each class's AUC by its frequency in y_true
   - Gives more importance to classes with more samples
   - Current distribution: -1 (51.6%), 0 (31.6%), 1 (16.8%)

3. **Score**: 0.9152 means the model has 91.52% average ability to distinguish each class from the others

## Triple Barrier Labeling

The 3-class labels come from the **triple barrier method** (López de Prado):

```python
# In fin_feature_preprocessing.py
barriers = feature_engineer.create_dynamic_triple_barriers(df)
```

This creates:
- **Upper barrier**: Price increases by threshold → label = 1
- **Lower barrier**: Price decreases by threshold → label = -1  
- **Time barrier**: No significant move before timeout → label = 0

## Next Steps

1. ✅ **Multi-class support working** - Pipeline now handles 3-class labels
2. ✅ **Predictions saving working** - All class probabilities captured
3. ⏳ **Complete test run** - Verify all methods (PCV, Walk-Forward, PBO) work end-to-end
4. 🚨 **CRITICAL**: Still need to remove forward return features (label leakage)

## Files Modified

- `lopez_de_prado_evaluation.py`:
  - Line 143-159: Fixed roc_auc_score for multi-class
  - Line 179-197: Fixed predictions saving for multi-dimensional probabilities
