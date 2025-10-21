# Feature Normalization with Winsorization

## Problem Identified

Initial RobustScaler implementation showed abnormal statistics:

```
BEFORE normalization:
   Mean: 13.1035, Std: 48.9213
   Min: -21712.96, Max: 670.98

AFTER RobustScaler:
   Mean: -0.0417, Std: 154.08  ❌ WRONG!
   Min: -195296.32, Max: 5674.96  ❌ EXTREME OUTLIERS!
```

**Issue**: Even RobustScaler (which uses median/IQR) struggled with extreme outliers in financial data. Standard deviation should be ~1 after scaling, not 154!

---

## Root Cause

Financial data contains extreme outliers:
- Flash crashes (instant -10% moves)
- Stock splits (sudden 2x or 10x price jumps)
- Data errors (wrong decimal places)
- Low-liquidity tickers (huge spreads)
- Market halts (zero volume)

These extreme values (even 1% of data) can completely dominate the feature space:
- One value of -21,712 drowns out typical features in [-1, 1] range
- Volume features can be 1000x larger than price features
- Model sees mostly noise from outliers

---

## Solution: Winsorization + RobustScaler

**Winsorization** = Clip extreme values at percentiles (1st and 99th)

```python
# 1. Clip outliers BEFORE scaling
percentile_1 = np.percentile(X_train, 1, axis=0)   # Per-feature
percentile_99 = np.percentile(X_train, 99, axis=0)  # Per-feature
X_train_clipped = np.clip(X_train, percentile_1, percentile_99)

# 2. THEN apply RobustScaler
scaler = RobustScaler()
scaler.fit(X_train_clipped)
X_train_scaled = scaler.transform(X_train_clipped)
```

**Why this works:**
1. Removes top/bottom 1% extreme values (2% total)
2. Preserves 98% of data unchanged
3. Each feature clipped independently (per-column)
4. RobustScaler then works properly on cleaned data

**Expected results:**
```
AFTER winsorization + RobustScaler:
   Mean: ~0.00, Std: ~1.00  ✅ CORRECT!
   Min: ~-3 to -5, Max: ~3 to 5
   Outliers handled gracefully
```

---

## Implementation Details

### Training Pipeline

```python
# Step 1: Create sequences
X_seq = create_sequences(features, seq_len=20)
# Shape: (samples, timesteps, features)

# Step 2: Flatten for processing
X_flat = X_seq.reshape(-1, n_features)

# Step 3: Winsorize (clip outliers)
p1 = np.percentile(X_flat, 1, axis=0)
p99 = np.percentile(X_flat, 99, axis=0)
X_clipped = np.clip(X_flat, p1, p99)

# Step 4: Scale with RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_clipped)

# Step 5: Reshape back
X_seq_scaled = X_scaled.reshape(X_seq.shape)

# Step 6: Train/Val split
X_train, X_val = split(X_seq_scaled)

# Step 7: REFIT on training data only (prevent data leakage)
X_train_flat = X_train.reshape(-1, n_features)
X_val_flat = X_val.reshape(-1, n_features)

# Compute percentiles on TRAINING data only
p1_train = np.percentile(X_train_flat, 1, axis=0)
p99_train = np.percentile(X_train_flat, 99, axis=0)

# Clip both train and val using TRAINING percentiles
X_train_clipped = np.clip(X_train_flat, p1_train, p99_train)
X_val_clipped = np.clip(X_val_flat, p1_train, p99_train)

# Refit scaler on TRAINING data only
scaler_final = RobustScaler()
scaler_final.fit(X_train_clipped)

# Transform both
X_train_scaled = scaler_final.transform(X_train_clipped).reshape(X_train.shape)
X_val_scaled = scaler_final.transform(X_val_clipped).reshape(X_val.shape)

# Step 8: Save scaler + clipping params
scaler_config = {
    'scaler': scaler_final,
    'clip_percentile_1': p1_train,
    'clip_percentile_99': p99_train
}
pickle.dump(scaler_config, open('feature_scaler.pkl', 'wb'))
```

### Inference Pipeline

```python
# Load saved scaler + clipping params
with open('feature_scaler.pkl', 'rb') as f:
    scaler_config = pickle.load(f)

scaler = scaler_config['scaler']
clip_min = scaler_config['clip_percentile_1']
clip_max = scaler_config['clip_percentile_99']

# Prepare new data
X_new = create_sequences(features, seq_len=20)
X_new_flat = X_new.reshape(-1, n_features)

# Apply SAME clipping as training
X_new_clipped = np.clip(X_new_flat, clip_min, clip_max)

# Apply SAME scaling as training
X_new_scaled = scaler.transform(X_new_clipped)

# Reshape and predict
X_new_scaled = X_new_scaled.reshape(X_new.shape)
predictions = model.predict(X_new_scaled)
```

---

## Why Winsorization?

### Compared to Other Approaches

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **StandardScaler** | Simple, well-known | Sensitive to outliers | Clean data only |
| **RobustScaler** | Uses median/IQR | Still affected by extremes | Most financial data |
| **MinMaxScaler** | Bounded [0,1] | Very sensitive to outliers | Never for finance |
| **Winsorization + RobustScaler** | Handles extreme outliers | More complex | Real-world finance ✅ |
| **Remove outliers** | Cleanest approach | Lose data, hard to define | Research only |

### Why Not Just Remove Outliers?

```python
# ❌ Problem: How to handle at inference?
# If you remove outliers during training, what do you do when
# you see an outlier at inference time?

# Training
X_train = X_train[X_train < threshold]  # Remove outliers
scaler.fit(X_train)

# Inference - New outlier arrives!
if X_new > threshold:
    # What do we do here? Can't just skip it!
    # Can't predict without the sample!
    ???
```

**Winsorization solves this:**
- Training: Clip to [p1, p99]
- Inference: Clip to SAME [p1, p99]
- Always have valid data to predict on
- No special cases or missing predictions

---

## Expected Improvements

### Before Normalization
```
Features at different scales:
- RSI: [0, 100]
- Returns: [-0.15, 0.15]
- Volume: [0, 1,000,000]
- OBV: [-21,712, 670]  ← Dominates!

Model sees:
- Mostly volume/OBV (huge magnitude)
- Returns/RSI contribute almost nothing
- Gradients dominated by large features
```

### After Winsorization + RobustScaler
```
All features normalized:
- RSI: [-2, 2]
- Returns: [-2, 2]
- Volume: [-2, 2]
- OBV: [-2, 2]

Model sees:
- All features contribute equally
- Balanced gradient flow
- Better learning dynamics
```

### Performance Impact

**Expected improvements:**
- **Direction AUC**: 0.67 → 0.70-0.73 (+5-8%)
- **Training stability**: Smoother loss curves
- **Convergence speed**: Faster (fewer epochs needed)
- **Prediction quality**: More balanced, less saturated

**Why?**
1. All features contribute equally to gradients
2. No single feature dominates learning
3. Optimizer (Adam) works better with normalized data
4. Batch normalization (if used) is more effective

---

## Validation

### Check Normalization Quality

```python
def validate_normalization(X_scaled, name="Normalized data"):
    """Verify normalization looks correct"""
    print(f"\n🔍 {name} Statistics:")
    print(f"   Shape: {X_scaled.shape}")
    print(f"   Mean: {X_scaled.mean():.4f} (target: 0.00)")
    print(f"   Std: {X_scaled.std():.4f} (target: 1.00)")
    print(f"   Median: {np.median(X_scaled):.4f} (target: 0.00)")
    print(f"   Min: {X_scaled.min():.4f}")
    print(f"   Max: {X_scaled.max():.4f}")
    print(f"   [P25, P75]: [{np.percentile(X_scaled, 25):.4f}, {np.percentile(X_scaled, 75):.4f}]")
    
    # Check for issues
    if np.isnan(X_scaled).any():
        print("   ❌ ERROR: NaN values detected!")
        return False
    
    if np.isinf(X_scaled).any():
        print("   ❌ ERROR: Inf values detected!")
        return False
    
    if abs(X_scaled.mean()) > 0.5:
        print("   ⚠️  WARNING: Mean far from 0!")
    
    if X_scaled.std() < 0.5 or X_scaled.std() > 3.0:
        print(f"   ⚠️  WARNING: Std unusual (should be ~1.0, got {X_scaled.std():.2f})")
    
    if X_scaled.min() < -10 or X_scaled.max() > 10:
        print(f"   ⚠️  WARNING: Extreme values still present!")
        print(f"   Check if winsorization is working correctly")
    
    print("   ✅ Normalization looks good!")
    return True

# Use it
validate_normalization(X_train_scaled, "Training data")
validate_normalization(X_val_scaled, "Validation data")
```

### Check Feature Distributions

```python
import matplotlib.pyplot as plt

def plot_feature_distributions(X_before, X_after, feature_idx=0):
    """Compare before/after distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Before
    axes[0].hist(X_before[:, :, feature_idx].flatten(), bins=50)
    axes[0].set_title(f'Feature {feature_idx} - Before Normalization')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    
    # After
    axes[1].hist(X_after[:, :, feature_idx].flatten(), bins=50)
    axes[1].set_title(f'Feature {feature_idx} - After Normalization')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('feature_normalization_comparison.png')
    print("📊 Saved comparison plot")

# Check a few features
plot_feature_distributions(X_seq_original, X_seq_scaled, feature_idx=0)
plot_feature_distributions(X_seq_original, X_seq_scaled, feature_idx=10)
plot_feature_distributions(X_seq_original, X_seq_scaled, feature_idx=20)
```

---

## Common Issues

### Issue 1: Std still > 5 after normalization

**Cause**: Extreme outliers not properly clipped
```python
# Check percentile values
print(f"Clip range per feature:")
for i in range(n_features):
    print(f"  Feature {i}: [{p1[i]:.2f}, {p99[i]:.2f}]")
    
# Look for extremely wide ranges
# If range > 1000, feature has serious outliers
```

**Fix**: More aggressive clipping (0.5th and 99.5th percentile)
```python
p0_5 = np.percentile(X_train_flat, 0.5, axis=0)
p99_5 = np.percentile(X_train_flat, 99.5, axis=0)
X_clipped = np.clip(X_train_flat, p0_5, p99_5)
```

### Issue 2: Different stats on train vs validation

**Cause**: Not using training percentiles for validation
```python
# ❌ WRONG
p1_val = np.percentile(X_val, 1, axis=0)  # Calculated on val!
X_val_clipped = np.clip(X_val, p1_val, p99_val)

# ✅ CORRECT
p1_train = np.percentile(X_train, 1, axis=0)  # From training
X_val_clipped = np.clip(X_val, p1_train, p99_train)
```

### Issue 3: Model still saturates after normalization

**Cause**: Might be model architecture, not normalization
```python
# Check if problem is features or model
print(f"Input range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Output range: [{predictions.min():.2f}, {predictions.max():.2f}]")

# If inputs normalized but outputs saturated:
# - Problem is activation functions (use softplus, not sigmoid)
# - Problem is too many layers (gradient vanishing)
# - Problem is learning rate (too high)
```

---

## Summary

**What we implemented:**
1. ✅ Winsorization at 1st/99th percentile (clip extreme outliers)
2. ✅ RobustScaler normalization (median/IQR based)
3. ✅ Proper train-only fitting (no data leakage)
4. ✅ Save clipping params with scaler (reproducible inference)
5. ✅ Apply same clipping at inference time

**Why it works:**
- Removes 2% most extreme outliers
- RobustScaler handles remaining outliers gracefully
- All features contribute equally to learning
- No look-ahead bias (use training params at inference)

**Expected results:**
- Mean ≈ 0.00, Std ≈ 1.00
- Min/Max in range [-5, 5]
- Direction AUC +5-8% improvement
- Faster convergence, more stable training

**Next steps:**
1. Train model with new normalization
2. Compare metrics before/after
3. Verify predictions not saturated
4. Consider class balancing (sample_weights)
5. Tune triple barrier parameters
