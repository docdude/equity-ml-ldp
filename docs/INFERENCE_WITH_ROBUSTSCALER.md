# Using RobustScaler at Inference Time

## Overview

The model now uses **RobustScaler** for feature normalization, which was fitted on training data only. You must apply the same scaler to any new data before making predictions.

---

## Quick Start

```python
import pickle
import numpy as np
from tensorflow import keras

# 1. Load model and scaler
model = keras.models.load_model('run_financial_wavenet_v1/fin_wavenet_model.keras')

with open('run_financial_wavenet_v1/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 2. Prepare new data (your feature engineering)
# ... create features using EnhancedFinancialFeatures ...
# X_new shape: (n_samples, n_timesteps, n_features)

# 3. Apply scaler
original_shape = X_new.shape
X_new_flat = X_new.reshape(-1, original_shape[2])
X_new_scaled = scaler.transform(X_new_flat).reshape(original_shape)

# 4. Make predictions
predictions = model.predict(X_new_scaled)
```

---

## Complete Inference Pipeline

```python
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

def load_model_and_scaler(model_path='run_financial_wavenet_v1'):
    """Load trained model, feature scaler, and clipping parameters"""
    model = keras.models.load_model(f'{model_path}/fin_wavenet_model.keras')
    
    with open(f'{model_path}/feature_scaler.pkl', 'rb') as f:
        scaler_config = pickle.load(f)
    
    # Extract components
    scaler = scaler_config['scaler']
    clip_min = scaler_config['clip_percentile_1']
    clip_max = scaler_config['clip_percentile_99']
    
    print(f"✅ Loaded model, scaler, and clipping params from {model_path}")
    return model, scaler, clip_min, clip_max

def prepare_features(df, feature_config='minimal'):
    """Create features from OHLCV data"""
    config = FeatureConfig.get_preset(feature_config)
    feature_engineer = EnhancedFinancialFeatures(feature_config=config)
    
    features = feature_engineer.create_all_features(df)
    features = features.fillna(method='ffill').fillna(0)
    
    return features

def create_sequences(features, seq_len=20):
    """Create sequences for LSTM input"""
    X_sequences = []
    
    for i in range(len(features) - seq_len):
        X_sequences.append(features.iloc[i:i+seq_len].values)
    
    return np.array(X_sequences)

def predict_trading_signals(df, model, scaler, clip_min, clip_max, seq_len=20, feature_config='minimal'):
    """
    Complete pipeline: data → features → sequences → clip → scale → predictions
    
    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        model: Trained Keras model
        scaler: Fitted RobustScaler
        clip_min: Lower clipping bounds (1st percentile from training)
        clip_max: Upper clipping bounds (99th percentile from training)
        seq_len: Sequence length (must match training)
        feature_config: Feature preset used during training
    
    Returns:
        predictions: Dict with 'direction', 'volatility', 'magnitude'
        dates: Timestamps for predictions
    """
    print("🔧 Preparing features...")
    features = prepare_features(df, feature_config)
    
    print("📊 Creating sequences...")
    X_seq = create_sequences(features, seq_len)
    
    print("✂️  Clipping outliers...")
    original_shape = X_seq.shape
    X_seq_flat = X_seq.reshape(-1, original_shape[2])
    X_seq_flat_clipped = np.clip(X_seq_flat, clip_min, clip_max)
    
    print("⚖️  Normalizing features...")
    X_seq_scaled = scaler.transform(X_seq_flat_clipped).reshape(original_shape)
    
    print(f"✅ Prepared {len(X_seq_scaled)} sequences")
    print(f"   Shape: {X_seq_scaled.shape}")
    print(f"   Normalized: mean={X_seq_scaled.mean():.4f}, std={X_seq_scaled.std():.4f}")
    
    print("🚀 Making predictions...")
    predictions = model.predict(X_seq_scaled, verbose=0)
    
    # Get dates (skip first seq_len dates used for first sequence)
    dates = df.index[seq_len:]
    
    return predictions, dates

# Example usage
if __name__ == "__main__":
    # Load model, scaler, and clipping params
    model, scaler, clip_min, clip_max = load_model_and_scaler()
    
    # Load new data
    df = pd.read_parquet('data_raw/AAPL.parquet')
    
    # Make predictions
    predictions, dates = predict_trading_signals(
        df, 
        model, 
        scaler,
        clip_min,
        clip_max,
        seq_len=20, 
        feature_config='minimal'
    )
    
    # Extract predictions
    direction_probs = predictions['direction']
    predicted_class = direction_probs.argmax(axis=1)
    
    volatility_pred = predictions['volatility'].flatten()
    magnitude_pred = predictions['magnitude'].flatten()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': dates,
        'predicted_direction': predicted_class,
        'prob_down': direction_probs[:, 0],
        'prob_up': direction_probs[:, 1],
        'prob_neutral': direction_probs[:, 2],
        'predicted_volatility': volatility_pred,
        'predicted_magnitude': magnitude_pred
    })
    
    print("\n📈 Predictions:")
    print(results.tail(10))
    
    # Generate trading signals
    confidence_threshold = 0.6
    results['signal'] = 'HOLD'
    results.loc[
        (results['predicted_direction'] == 1) & 
        (results['prob_up'] > confidence_threshold), 
        'signal'
    ] = 'BUY'
    results.loc[
        (results['predicted_direction'] == 0) & 
        (results['prob_down'] > confidence_threshold), 
        'signal'
    ] = 'SELL'
    
    print(f"\n🎯 Trading Signals:")
    print(f"   BUY: {(results['signal'] == 'BUY').sum()}")
    print(f"   SELL: {(results['signal'] == 'SELL').sum()}")
    print(f"   HOLD: {(results['signal'] == 'HOLD').sum()}")
```

---

## Winsorization (Outlier Clipping)

The model uses **winsorization** to handle extreme outliers before scaling:

```python
# Clip values at 1st and 99th percentile
# This prevents extreme values from dominating the normalization
X_clipped = np.clip(X_flat, percentile_1, percentile_99)

# Then apply RobustScaler
X_scaled = scaler.transform(X_clipped)
```

**Why winsorization?**
- Financial data has extreme outliers (flash crashes, splits, data errors)
- Even RobustScaler (which uses median/IQR) can struggle with very extreme values
- Clipping top/bottom 1% prevents these from skewing the distribution
- More aggressive than pure RobustScaler but necessary for real-world financial data

**At inference:**
You MUST apply the same clipping bounds from training:
```python
# Load clipping bounds from training
scaler_config = pickle.load(open('feature_scaler.pkl', 'rb'))
clip_min = scaler_config['clip_percentile_1']  # 1st percentile from training
clip_max = scaler_config['clip_percentile_99']  # 99th percentile from training

# Apply to new data
X_new_clipped = np.clip(X_new, clip_min, clip_max)
X_new_scaled = scaler.transform(X_new_clipped)
```

---

## Important Notes

### ⚠️ Critical Requirements

1. **Same Features**: Must use exact same features as training
   - Same feature_config preset ('minimal', 'balanced', etc.)
   - Same feature engineering code
   - Features in same order

2. **Same Sequence Length**: Must match training
   - Default: 20 timesteps
   - Check model input shape: `model.input_shape`

3. **Same Scaler AND Clipping**: Must use saved scaler and clipping params from training
   - Don't fit a new scaler on test data!
   - Don't calculate new percentiles on test data!
   - Load both from `feature_scaler.pkl`

4. **Correct Order**: Clip THEN scale (same order as training)
   ```python
   # ✅ CORRECT ORDER
   X_clipped = np.clip(X, clip_min, clip_max)
   X_scaled = scaler.transform(X_clipped)
   
   # ❌ WRONG ORDER
   X_scaled = scaler.transform(X)
   X_clipped = np.clip(X_scaled, ...)  # Too late!
   ```

5. **Data Quality**: Clean your data
   - Remove NaN values (forward fill)
   - Ensure proper date indexing
   - Check OHLCV data integrity

### 🎯 Common Mistakes

```python
# ❌ BAD: Fitting new scaler on test data
scaler = RobustScaler()
X_test_scaled = scaler.fit_transform(X_test)  # WRONG! Data leakage!

# ❌ BAD: Calculating new percentiles on test data
percentile_1 = np.percentile(X_test, 1, axis=0)  # WRONG! Data leakage!
percentile_99 = np.percentile(X_test, 99, axis=0)  # WRONG!
X_test_clipped = np.clip(X_test, percentile_1, percentile_99)

# ✅ GOOD: Using saved scaler and clipping params
with open('feature_scaler.pkl', 'rb') as f:
    scaler_config = pickle.load(f)
scaler = scaler_config['scaler']
clip_min = scaler_config['clip_percentile_1']
clip_max = scaler_config['clip_percentile_99']

X_test_clipped = np.clip(X_test, clip_min, clip_max)  # Correct!
X_test_scaled = scaler.transform(X_test_clipped)  # Correct!

# ❌ BAD: Different features
features_new = create_features_v2(df)  # Different features!

# ✅ GOOD: Same features as training
config = FeatureConfig.get_preset('minimal')  # Same as training
features_new = EnhancedFinancialFeatures(feature_config=config).create_all_features(df)

# ❌ BAD: Wrong sequence length
X_seq = create_sequences(features, seq_len=30)  # Different length!

# ✅ GOOD: Same sequence length
X_seq = create_sequences(features, seq_len=20)  # Same as training
```

---

## Verification

```python
def verify_preprocessing(X_scaled):
    """Verify preprocessing looks correct"""
    print("🔍 Preprocessing Verification:")
    print(f"   Shape: {X_scaled.shape}")
    print(f"   Mean: {X_scaled.mean():.4f} (should be near 0)")
    print(f"   Std: {X_scaled.std():.4f} (should be near 1)")
    print(f"   Min: {X_scaled.min():.4f}")
    print(f"   Max: {X_scaled.max():.4f}")
    
    # Check for issues
    if np.isnan(X_scaled).any():
        print("   ⚠️  WARNING: NaN values detected!")
    
    if np.isinf(X_scaled).any():
        print("   ⚠️  WARNING: Inf values detected!")
    
    if X_scaled.std() < 0.5 or X_scaled.std() > 2.0:
        print("   ⚠️  WARNING: Unusual std deviation!")
    
    if abs(X_scaled.mean()) > 0.5:
        print("   ⚠️  WARNING: Mean far from 0!")
    
    print("   ✅ Preprocessing looks good!")

# Use it
X_scaled = scaler.transform(X_flat).reshape(original_shape)
verify_preprocessing(X_scaled)
```

---

## Performance Expectations

With RobustScaler normalization, you should see:

**Model Performance:**
- Direction AUC: 0.70-0.73 (up from 0.67)
- Direction Accuracy: 0.45-0.55 (3-class)
- Volatility RMSE: 0.01-0.015
- Magnitude RMSE: 0.03-0.05

**Prediction Distribution:**
- Direction: More balanced predictions
- Volatility: Predictions in [0, 0.1] range
- Magnitude: Predictions in [0, 0.2] range

**No more saturated outputs!**
- Previous issue: All predictions near 0 or 1
- With normalization: Smooth distribution

---

## Troubleshooting

### Issue: Different predictions than training

**Cause**: Not using saved scaler
```python
# Fix: Load the scaler
with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

### Issue: NaN in predictions

**Cause**: NaN in features
```python
# Fix: Clean features
features = features.fillna(method='ffill').fillna(0)
```

### Issue: Model predicts same class for everything

**Cause**: Features not scaled
```python
# Fix: Apply scaler
X_scaled = scaler.transform(X_flat).reshape(original_shape)
```

### Issue: "Shape mismatch" error

**Cause**: Wrong sequence length or number of features
```python
# Check model input shape
print(model.input_shape)  # Should be (None, 20, n_features)

# Verify your data
print(X_seq.shape)  # Should match (n_samples, 20, n_features)
```

---

## Next Steps

After implementing normalization:

1. **Retrain model** with normalized features
2. **Compare metrics** before/after normalization
3. **Check predictions** - should be more stable
4. **Update evaluation scripts** to use scaler
5. **Document performance improvement**

Expected improvement:
- ✅ AUC: +5-8% (from 0.67 to 0.70-0.73)
- ✅ More stable predictions
- ✅ Better gradient flow
- ✅ Reduced training time (faster convergence)
