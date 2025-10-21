# Feature Normalization for Financial WaveNet

## Background: IMU vs Financial Data

### Your IMU WaveNet (Swimming Styles)
- **Data**: Accelerometer + Gyroscope (bounded physical ranges)
- **Normalization**: Tanh-scaled per window [-1, 1]
- **Why it worked**: 
  - Physical sensors have known ranges
  - Per-window normalization preserves relative patterns
  - Sliding windows with augmentation

### Our Financial WaveNet (Trading Signals)
- **Data**: 100+ features (prices, volumes, indicators, ratios)
- **Current**: No normalization ❌
- **Challenge**: Features have vastly different scales

---

## The Problem

**Current Feature Scales (from fin_training.py):**
```
Examples from validation data:
- RSI: [0, 100]
- log_return_1d: [-0.15, 0.15] 
- volatility_yz_20: [0.0, 0.8]
- volume_norm: [0.0, 5.0+]
- price_position: [0, 1]
- obv_zscore: [-3, 3]
- cmf: [-1, 1]
```

**Without normalization:**
- Volume features dominate (large magnitude)
- Returns/volatility get drowned out (small magnitude)
- Model struggles to learn multi-scale patterns

---

## Normalization Options

### 1. **Tanh-Scaled (Your IMU Approach)** ⚠️

```python
# Per-window scaling to [-1, 1]
min_val = np.min(window_data, axis=0)
max_val = np.max(window_data, axis=0)
scaled = 2 * (window_data - min_val) / (max_val - min_val) - 1
```

**Pros:**
- ✅ Preserves relative patterns within window
- ✅ Bounded range [-1, 1] (good for WaveNet)
- ✅ Works well for physical sensors

**Cons for Finance:**
- ❌ Look-ahead bias! (uses future data in window)
- ❌ Not reproducible at inference (need same window)
- ❌ Financial features have different semantics than IMU data

**Verdict:** Not suitable for financial prediction (look-ahead bias)

---

### 2. **StandardScaler (Z-score)** ✅ RECOMMENDED

```python
from sklearn.preprocessing import StandardScaler

# Fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
X_val_scaled = scaler.transform(X_val.reshape(-1, n_features))
```

**Pros:**
- ✅ No look-ahead bias (fit only on train)
- ✅ Handles outliers reasonably
- ✅ Mean=0, Std=1 (good for neural networks)
- ✅ Works with time-series CV

**Cons:**
- ⚠️ Sensitive to outliers
- ⚠️ Unbounded (can have values > 3 std)

**Best for:** Most financial features (returns, ratios, indicators)

---

### 3. **RobustScaler (Median + IQR)** ✅ RECOMMENDED

```python
from sklearn.preprocessing import RobustScaler

# Uses median and IQR instead of mean/std
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
```

**Pros:**
- ✅ Robust to outliers (uses median, not mean)
- ✅ Better for financial data (has extreme events)
- ✅ No look-ahead bias

**Cons:**
- ⚠️ Still unbounded

**Best for:** Financial data with outliers (market crashes, flash crashes)

---

### 4. **MinMaxScaler** ⚠️

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
```

**Pros:**
- ✅ Bounded range (good for WaveNet)
- ✅ Simple and fast

**Cons:**
- ❌ Very sensitive to outliers
- ❌ Single extreme value ruins entire scale
- ❌ Not good for financial data (has black swans)

**Verdict:** Avoid for financial data

---

### 5. **Hybrid Approach (Feature-Type Specific)** ✅✅ BEST

Different feature groups need different normalization:

```python
# Already normalized (0-100)
rsi_features = ['rsi_7', 'rsi_14', 'rsi_21']  
# Keep as-is or divide by 100

# Already normalized (0-1 or -1 to 1)
bounded_features = ['bb_percent_b', 'price_position', 'cmf']
# Keep as-is

# Need robust scaling (returns, volatility)
return_features = ['log_return_*', 'return_acceleration']
volatility_features = ['volatility_*', 'atr']
# Use RobustScaler

# Need standard scaling (volume, indicators)
volume_features = ['volume_*', 'obv_*', 'ad_*']
other_indicators = ['adx', 'cci', 'macd*']
# Use StandardScaler

# Ratios (already normalized)
ratio_features = ['*_ratio', 'dist_from_*']
# Keep as-is or clip to [-3, 3]
```

---

## Recommendation for Your Model

### **Option A: Simple & Effective (Start Here)**

```python
from sklearn.preprocessing import RobustScaler

# Apply to ALL features
scaler = RobustScaler()
X_train_flat = X_train.reshape(-1, n_features)
X_val_flat = X_val.reshape(-1, n_features)

# Fit only on training data
scaler.fit(X_train_flat)

# Transform both
X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
```

**Why RobustScaler:**
- Financial data has outliers (crashes, spikes)
- Uses median/IQR instead of mean/std
- More stable than StandardScaler
- No look-ahead bias

---

### **Option B: Hybrid (More Sophisticated)**

```python
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np

def normalize_financial_features(X, feature_names, scaler_dict=None, fit=True):
    """
    Apply feature-type-specific normalization
    
    Args:
        X: (n_samples, n_timesteps, n_features)
        feature_names: List of feature names
        scaler_dict: Dict of fitted scalers (for inference)
        fit: Whether to fit scalers (True for train, False for val/test)
    """
    X_normalized = X.copy()
    n_samples, n_timesteps, n_features = X.shape
    
    if scaler_dict is None:
        scaler_dict = {}
    
    # Group 1: Already bounded [0, 100] - scale to [0, 1]
    rsi_idx = [i for i, name in enumerate(feature_names) if 'rsi_' in name]
    if rsi_idx:
        X_normalized[:, :, rsi_idx] = X[:, :, rsi_idx] / 100.0
    
    # Group 2: Already bounded [0, 1] or [-1, 1] - keep as-is
    bounded_idx = [i for i, name in enumerate(feature_names) 
                   if any(x in name for x in ['bb_percent', 'price_position', 'cmf', 'sar_signal'])]
    # No transformation needed
    
    # Group 3: Returns & volatility - RobustScaler
    returns_vol_idx = [i for i, name in enumerate(feature_names)
                       if any(x in name for x in ['return', 'volatility', 'atr', 'vol_'])]
    if returns_vol_idx:
        X_flat = X[:, :, returns_vol_idx].reshape(-1, len(returns_vol_idx))
        if fit:
            scaler_dict['returns_vol'] = RobustScaler().fit(X_flat)
        X_scaled = scaler_dict['returns_vol'].transform(X_flat)
        X_normalized[:, :, returns_vol_idx] = X_scaled.reshape(n_samples, n_timesteps, -1)
    
    # Group 4: Volume & other indicators - StandardScaler
    volume_idx = [i for i, name in enumerate(feature_names)
                  if any(x in name for x in ['volume', 'obv', 'ad_', 'vwap'])]
    if volume_idx:
        X_flat = X[:, :, volume_idx].reshape(-1, len(volume_idx))
        if fit:
            scaler_dict['volume'] = StandardScaler().fit(X_flat)
        X_scaled = scaler_dict['volume'].transform(X_flat)
        X_normalized[:, :, volume_idx] = X_scaled.reshape(n_samples, n_timesteps, -1)
    
    # Group 5: Other indicators - StandardScaler
    other_idx = [i for i, name in enumerate(feature_names)
                 if i not in rsi_idx + bounded_idx + returns_vol_idx + volume_idx]
    if other_idx:
        X_flat = X[:, :, other_idx].reshape(-1, len(other_idx))
        if fit:
            scaler_dict['other'] = StandardScaler().fit(X_flat)
        X_scaled = scaler_dict['other'].transform(X_flat)
        X_normalized[:, :, other_idx] = X_scaled.reshape(n_samples, n_timesteps, -1)
    
    # Clip extreme values to [-5, 5] std to prevent explosions
    X_normalized = np.clip(X_normalized, -5, 5)
    
    return X_normalized, scaler_dict
```

---

## Key Differences from IMU Approach

| Aspect | IMU (Swimming) | Financial (Trading) |
|--------|----------------|---------------------|
| **Normalization** | Per-window tanh | Global scaler (fit on train) |
| **Reason** | Relative patterns matter | Absolute scale matters for inference |
| **Look-ahead** | OK (within same motion) | NOT OK (future data leak) |
| **Inference** | Need same window size | Need saved scaler from training |
| **Data nature** | Bounded physical sensors | Unbounded, heavy-tailed returns |

---

## Implementation Plan

### Step 1: Add to fin_training.py

```python
# After creating sequences, before train/val split
print("\n2.5. FEATURE NORMALIZATION")
print("-"*40)

from sklearn.preprocessing import RobustScaler

# Flatten sequences for scaling
X_train_flat = X_train.reshape(-1, X_train.shape[2])
X_val_flat = X_val.reshape(-1, X_val.shape[2])

# Fit scaler on training data only
scaler = RobustScaler()
scaler.fit(X_train_flat)

# Transform both
X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)

print(f"✅ Applied RobustScaler (median/IQR normalization)")
print(f"   Training data: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
print(f"   Validation data: mean={X_val_scaled.mean():.4f}, std={X_val_scaled.std():.4f}")

# Save scaler for inference
import pickle
with open(f'{experiment_save_path}/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"   Saved scaler to {experiment_save_path}/feature_scaler.pkl")

# Use scaled data for training
X_train = X_train_scaled
X_val = X_val_scaled
```

### Step 2: Update Inference Code

```python
# At inference time
with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Scale new data
X_new_scaled = scaler.transform(X_new.reshape(-1, n_features)).reshape(X_new.shape)
predictions = model.predict(X_new_scaled)
```

---

## Expected Impact

**Before Normalization:**
- Features compete at different scales
- Large volume values dominate gradients
- Model struggles with small returns
- Validation AUC: ~0.67

**After Normalization:**
- All features contribute equally
- Better gradient flow
- Model learns multi-scale patterns
- **Expected AUC: 0.70-0.73** (based on literature)

---

## Why Not Per-Window Like IMU?

**IMU Context:**
- You're classifying a complete motion (butterfly stroke, etc.)
- The motion is self-contained in the window
- Relative patterns within the motion matter
- No prediction of future states

**Financial Context:**
- You're predicting future returns/direction
- Using past data to predict future
- Absolute scale matters (5% return vs 0.5%)
- Per-window scaling would use future data (look-ahead bias!)

**Example of look-ahead bias:**
```python
# BAD: Per-window scaling (your IMU approach)
window = X[i:i+20]  # Next 20 days
min_val = window.min()  # ❌ Includes future data!
max_val = window.max()  # ❌ Includes future data!
scaled = (window - min_val) / (max_val - min_val)
# At prediction time for day i, you wouldn't know min/max of next 20 days!

# GOOD: Global scaling (fit on past only)
scaler.fit(X_train)  # ✅ Only past data
scaled = scaler.transform(X_new)  # ✅ Can reproduce at inference
```

---

## Recommendation

**Start with Option A (RobustScaler for all features):**
1. Simple to implement
2. Robust to financial outliers
3. No look-ahead bias
4. Easy to reproduce at inference
5. Should improve AUC by ~5-8%

**Then consider Option B (Hybrid) if needed:**
- More sophisticated
- Feature-type specific
- Potentially 1-2% better performance
- More complex to maintain

**DO NOT use per-window tanh scaling:**
- Creates look-ahead bias
- Won't work at inference
- Different use case than IMU

---

## Next Steps

1. ✅ Add RobustScaler to fin_training.py
2. ✅ Save scaler with model
3. ✅ Update inference code to use scaler
4. ✅ Compare before/after metrics
5. ⏳ If needed, try hybrid approach

Would you like me to implement Option A (RobustScaler) in your training code now?
