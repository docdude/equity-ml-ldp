# Volatility and Magnitude Output Issues - Root Cause Analysis

**Date**: October 15, 2025  
**Status**: 🔴 CRITICAL - Model outputs are broken  
**Impact**: High training losses (631% vol, 362% mag), saturated predictions

## Problem Summary

The CNN-LSTM model has **serious issues** with its auxiliary outputs (volatility and magnitude):

1. **Volatility predictions saturated at ~1.0** (100% of predictions >0.999)
2. **Extremely high training losses** (MAE: 631% volatility, 362% magnitude)
3. **Mismatch between activation function and targets**
4. **Wrong loss function for sigmoid output**

## Root Cause Analysis

### Issue 1: Volatility - Sigmoid + MSE Mismatch ❌

**Current Architecture:**
```python
# Model output (fin_model.py line 217)
volatility_out = layers.Dense(1, activation='sigmoid', name='volatility')(fusion)

# Loss function (fin_training.py line 341)
'volatility': 'mse'

# Actual targets (calculated from 5-day returns)
volatility_targets = prices.pct_change().rolling(5).std()
# Range: [0.0009, 0.107]  # 0.09% to 10.7%
# Mean: 0.0154  # 1.54%
# 95th percentile: 0.0329  # 3.29%
```

**The Problem:**
```
Target: 0.02 (2% volatility)
Prediction: sigmoid output → [0, 1]
Model learns: Always predict 1.0 to minimize loss
MSE loss: (1.0 - 0.02)^2 = 0.96 PER SAMPLE!

With 18,745 training samples:
Total volatility loss = 0.96 × 18,745 = 18,000+
Scaled by loss_weight (0.3) = 5,400

This DOMINATES the total loss!
```

**Why Sigmoid?**
- Sigmoid activation outputs [0, 1]
- Typically used for **probabilities or binary classification**
- NOT appropriate for volatility which is an **unbounded positive value**

**Why Not BCE?**
- Binary Cross-Entropy (BCE) requires targets in [0, 1]
- Our targets are [0, 0.107] - NOT probabilities
- BCE would be appropriate IF we normalized targets to [0, 1] first
- But then we lose interpretability (what does 0.5 mean?)

### Issue 2: Magnitude - Linear Activation Without Bounds ⚠️

**Current Architecture:**
```python
# Model output (fin_model.py line 220)
magnitude_out = layers.Dense(1, activation='linear', name='magnitude')(fusion)

# Loss function (fin_training.py line 342)
'magnitude': 'huber'  # ✅ Good choice - robust to outliers

# Actual targets (absolute value of 5-day returns)
magnitude_targets = abs(forward_returns_5d)
# Range: [0, ~0.50]  # 0% to 50%
# Mean: ~0.05  # 5%
```

**The Problem:**
- Linear activation can output ANY value (-∞, +∞)
- But magnitude is conceptually bounded [0, +∞)
- Model can predict negative magnitudes (nonsensical)
- Predictions range [0.39, 3.93] while targets are [0, 0.50]
- Model is predicting 390% magnitude when reality is 5-10%!

**Why Huber Loss is OK:**
- Huber loss is robust to outliers (good for returns)
- Combines MSE (small errors) + MAE (large errors)
- But doesn't fix the activation mismatch

### Issue 3: Metrics Don't Align 📊

**Current Metrics:**
```python
metrics={
    'direction': ['accuracy', tf.keras.metrics.AUC(name='auc')],  # ✅ Correct
    'volatility': ['mae'],  # ⚠️ Shows 631% - what does this mean?
    'magnitude': ['mae']    # ⚠️ Shows 362% - completely off scale
}
```

**The Problem:**
- MAE on volatility: comparing sigmoid output [0,1] to targets [0,0.1]
- MAE on magnitude: comparing linear output [-∞,+∞] to targets [0,0.5]
- No way to interpret what "631% MAE" means for volatility
- Should use **relative metrics** (MAPE, R²) instead

## Evidence from Predictions

### Volatility Predictions (4,683 validation samples):
```
Min:     0.9999  (99.99%)
Max:     1.0000  (100.00%)
Mean:    0.99999595  (99.9996%)
Median:  0.9999974  (99.9997%)
Std:     0.00000514  (0.0005%)

% predictions > 0.999: 100.00%
```

**Interpretation:** Model has learned to **always predict maximum volatility** to minimize MSE loss. This is completely saturated and useless.

### Magnitude Predictions:
```
Min:     0.39  (39% return magnitude)
Max:     3.93  (393% return magnitude!)
Mean:    1.22  (122% return magnitude)
Median:  1.13  (113% return magnitude)

Reality: Most 5-day magnitudes are 2-10%
```

**Interpretation:** Model is predicting absurdly high return magnitudes (100%+ moves in 5 days) when reality is 2-10%.

## Solutions

### Option 1: Fix Activation Functions (RECOMMENDED) ✅

**For Volatility:**
```python
# Change to RELU (outputs [0, +∞))
volatility_out = layers.Dense(1, activation='relu', name='volatility')(fusion)

# Keep MSE loss (now makes sense)
'volatility': 'mse'

# Targets stay in natural scale [0, 0.107]
# Predictions will learn appropriate range
```

**For Magnitude:**
```python
# Change to RELU (outputs [0, +∞))
magnitude_out = layers.Dense(1, activation='relu', name='magnitude')(fusion)

# Keep Huber loss (robust to outliers)
'magnitude': 'huber'

# Targets stay in natural scale [0, 0.50]
# Predictions will match target range
```

**Pros:**
- ✅ Simple one-line fix
- ✅ Keeps targets in interpretable scale
- ✅ No data preprocessing needed
- ✅ RELU naturally outputs positive values
- ✅ Can represent any magnitude [0, +∞)

**Cons:**
- ❌ Need to retrain model from scratch
- ❌ RELU can output very large values (but Huber loss handles this)

### Option 2: Scale Targets to [0, 1] + Keep Sigmoid

**For Volatility:**
```python
# Keep sigmoid activation
volatility_out = layers.Dense(1, activation='sigmoid', name='volatility')(fusion)

# Scale targets during training
max_vol = 0.15  # 99th percentile
volatility_targets_scaled = volatility_targets / max_vol
volatility_targets_scaled = np.clip(volatility_targets_scaled, 0, 1)

# Use BCE loss for [0,1] targets
'volatility': 'binary_crossentropy'

# Unscale predictions for evaluation
volatility_preds_real = volatility_preds * max_vol
```

**Pros:**
- ✅ Sigmoid naturally handles [0, 1] range
- ✅ BCE loss is theoretically sound for [0, 1] targets
- ✅ Bounded predictions prevent extreme values

**Cons:**
- ❌ Requires preprocessing and postprocessing
- ❌ Less interpretable during training
- ❌ Need to store/track scaling factors
- ❌ Clipping at 1.0 loses information about extreme volatility
- ❌ BCE treats this as probability (wrong conceptual model)

### Option 3: Logarithmic Transform + Linear Activation

**For Both Outputs:**
```python
# Use linear activation
volatility_out = layers.Dense(1, activation='linear', name='volatility')(fusion)
magnitude_out = layers.Dense(1, activation='linear', name='magnitude')(fusion)

# Log-transform targets
volatility_targets_log = np.log1p(volatility_targets)  # log(1 + x)
magnitude_targets_log = np.log1p(magnitude_targets)

# Use MSE on log scale
'volatility': 'mse',
'magnitude': 'mse'

# Exp-transform predictions
volatility_preds_real = np.expm1(volatility_preds)  # exp(x) - 1
```

**Pros:**
- ✅ Handles different scales naturally
- ✅ Focuses on relative errors (good for percentage-based targets)
- ✅ Linear activation is unbounded

**Cons:**
- ❌ Complex preprocessing/postprocessing
- ❌ Log transform changes loss landscape
- ❌ Can still predict negative values (need RELU or exp)
- ❌ Less interpretable

## Recommended Fix

**Use Option 1: ReLU + He Initialization**

### ReLU vs LeakyReLU Decision

**Initial consideration: LeakyReLU**
- Prevents dying neurons (gradient always non-zero)
- Better for deep networks with 100+ epochs
- Industry standard for long training runs

**BUT, for this specific model:**
- ✅ Training is SHORT (15-20 epochs with early stopping)
- ✅ Stopping on val_direction_auc (not vol/mag loss)
- ✅ Vol/mag are AUXILIARY outputs (low priority)
- ✅ Using Adam optimizer (adaptive, can recover from near-zero gradients)
- ✅ Single output neuron (not deep ReLU network)

**Dying ReLU risk is LOW because:**
1. Not enough epochs for neurons to die permanently
2. Early stopping prevents prolonged training of auxiliary heads
3. Adam's momentum helps escape dead zones
4. Small output layer (1 neuron) won't cascade failures

**ReLU + He Initialization Advantages:**
- ✅ Simpler (no hyperparameter alpha)
- ✅ Conceptually correct (outputs strictly ≥ 0)
- ✅ He initialization designed specifically for ReLU
- ✅ Sufficient for short training runs
- ✅ No post-processing needed (no negatives to clip)

**Conclusion:** ReLU + He init is the right choice for this use case.

### Changes Required:

**1. Update Model Architecture** (`fin_model.py` lines 217, 220):
```python
# OLD (BROKEN)
volatility_out = layers.Dense(1, activation='sigmoid', name='volatility')(fusion)
magnitude_out = layers.Dense(1, activation='linear', name='magnitude')(fusion)

# NEW (FIXED) ✅
volatility_out = layers.Dense(
    1, 
    activation='relu',
    kernel_initializer='he_normal',  # Designed for ReLU
    name='volatility'
)(fusion)

magnitude_out = layers.Dense(
    1, 
    activation='relu',
    kernel_initializer='he_normal',  # Designed for ReLU
    name='magnitude'
)(fusion)
```

**Why He initialization?**
- `he_normal`: Draws weights from normal distribution with std = sqrt(2 / fan_in)
- Keeps variance consistent through ReLU layers
- Prevents vanishing/exploding gradients
- Standard practice for ReLU activations

**2. Keep Loss Functions** (`fin_training.py` - NO CHANGE):
```python
loss={
    'direction': 'categorical_crossentropy',  # ✅ Correct
    'volatility': 'mse',                       # ✅ Now correct with RELU
    'magnitude': 'huber'                       # ✅ Correct
}
```

**3. Improve Metrics** (`fin_training.py` lines 349-353):
```python
# OLD
metrics={
    'direction': ['accuracy', tf.keras.metrics.AUC(name='auc')],
    'volatility': ['mae'],
    'magnitude': ['mae']
}

# NEW
metrics={
    'direction': ['accuracy', tf.keras.metrics.AUC(name='auc')],
    'volatility': ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
    'magnitude': ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
}
```

**4. Retrain Model:**
```bash
python fin_training.py
```

### Expected Results After Fix:

**Volatility Predictions:**
```
Range: [0.005, 0.08]  # 0.5% to 8% (strictly non-negative from ReLU)
Mean:  ~0.015         # 1.5%
Not saturated at 1.0!
Distribution matches actual volatility
No post-processing needed (ReLU ensures ≥ 0)
```

**Magnitude Predictions:**
```
Range: [0.01, 0.30]   # 1% to 30%
Mean:  ~0.05          # 5%
Matches target distribution!
No more 393% predictions!
```

**Training Losses:**
```
Volatility MAE: ~0.005 (0.5%)     # Down from 631%!
Magnitude MAE:  ~0.02 (2%)        # Down from 362%!
Better gradient flow from LeakyReLU
```

## Why This Matters

1. **Current Model is Broken:** Volatility predictions are 100% useless
2. **High Losses Dominate Training:** Direction head may be learning wrong patterns
3. **No Interpretability:** Can't use volatility/magnitude for position sizing
4. **Wasted Capacity:** Model dedicating neurons to predict useless outputs

## Action Items

- [ ] Update `fin_model.py` with RELU activations
- [ ] Add better metrics to `fin_training.py`
- [ ] Retrain model (Run #3)
- [ ] Validate predictions are in correct range
- [ ] Compare Run #2 (broken) vs Run #3 (fixed) PBO
- [ ] Update documentation with corrected architecture
- [ ] Consider removing volatility/magnitude if still problematic

## Appendix: Why Not Just Remove These Outputs?

**Arguments for Keeping:**
- Multi-task learning can improve direction predictions
- Volatility useful for dynamic position sizing
- Magnitude useful for risk management

**Arguments for Removing:**
- Adds complexity and potential bugs
- Direction head alone achieved 0.65 AUC (target met)
- PBO test only uses direction output
- Simpler model = easier to debug

**Recommendation:** Fix first, then evaluate if they improve direction accuracy. If not, remove them.
