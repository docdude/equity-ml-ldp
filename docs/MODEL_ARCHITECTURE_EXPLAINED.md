# CNN-LSTM Model Architecture - CORRECT EXPLANATION

## ⚠️ CRITICAL: Understanding the Model Outputs

### Model Overview
```
Input: (batch, 20 timesteps, 15 features)
  ↓
CNN Layers (Conv1D)
  ↓
LSTM Layers
  ↓
Dense Shared Layer
  ↓
  ├─→ Output 1: direction_output (3-class probabilities)
  └─→ Output 2: heatmap_output (signed confidence value)
```

---

## Output 1: direction_output

**Shape:** `(n_samples, 3)`  
**Type:** Softmax probabilities (sum to 1.0)  
**Activation:** Softmax

### Class Mapping (from Triple Barrier Method)

| Class Index | Label | Meaning | How it's determined |
|-------------|-------|---------|---------------------|
| 0 | **SL** (Stop Loss) | Price hit **-3%** barrier first | Downward move |
| 1 | **Timeout** | Neither barrier hit within 5 days | Sideways/small move |
| 2 | **TP** (Take Profit) | Price hit **+6%** barrier first | Upward move |

### Example:
```python
predictions = model.predict(X_test)
direction_probs = predictions[0]  # Shape: (n_samples, 3)

# Example for one sample:
# [0.15, 0.25, 0.60] means:
#   - 15% probability of SL (price drops to -3%)
#   - 25% probability of Timeout (price stays within ±3-6%)
#   - 60% probability of TP (price rises to +6%)
```

### For Binary Classification (SL vs TP):
```python
# Exclude timeout samples during training
# Then use only class 0 (SL) and class 2 (TP)

prob_sl = direction_probs[:, 0]
prob_tp = direction_probs[:, 2]

# Binary prediction: 1 if TP more likely than SL
y_pred_binary = (prob_tp > 0.5).astype(int)

# For AUC calculation, use prob(TP)
auc = roc_auc_score(y_true_binary, prob_tp)
```

---

## Output 2: heatmap_output

**Shape:** `(n_samples, 1)`  
**Type:** Continuous value  
**Activation:** Linear (no activation)

### Interpretation:
- **Positive values** → Model confident in **Take Profit** direction
- **Negative values** → Model confident in **Stop Loss** direction
- **Magnitude** → Strength of conviction (higher = more confident)

### Example:
```python
heatmap_confidence = predictions[1]  # Shape: (n_samples, 1)

# Example values:
#   +0.8 → Strong conviction for TP (upward move)
#   -0.5 → Moderate conviction for SL (downward move)
#    0.1 → Weak/uncertain signal
```

### Relationship to Direction Output:
- If `direction_probs[2]` (TP) is high → `heatmap_output` should be positive
- If `direction_probs[0]` (SL) is high → `heatmap_output` should be negative
- Both outputs trained jointly, should be correlated

---

## Ground Truth Generation: Triple Barrier Method

### Process (from `financial_preprocessing.py`):

```python
def augment_labels_to_heatmaps(tp=0.06, sl=0.03, horizon=5):
    """
    For each timestep, look forward up to 5 days:
    
    1. Calculate cumulative returns: r[t+1], r[t+1:t+2], ..., r[t+1:t+5]
    2. Check barriers:
       - If cumulative_return >= +6% first → Label = 1 (TP)
       - If cumulative_return <= -3% first → Label = 0 (SL)
       - If neither hit within 5 days → Label = -1 (Timeout)
    3. Map labels for training:
       - 0 (SL) → class_0
       - -1 (Timeout) → class_1
       - 1 (TP) → class_2
    """
```

### Asymmetric Barriers:
- **Take Profit:** +6.0% (upside target)
- **Stop Loss:** -3.0% (downside limit)
- **Ratio:** 2:1 reward/risk
- **Horizon:** 5 trading days

---

## Training Configuration

### Loss Functions:
```python
model.compile(
    loss={
        'direction_output': 'categorical_crossentropy',  # For 3-class
        'heatmap_output': 'mse'  # For signed confidence
    },
    loss_weights={
        'direction_output': 1.0,  # Primary task
        'heatmap_output': 0.5     # Auxiliary task
    }
)
```

### Sample Weights:
- **Purpose:** Handle class imbalance (more SL samples than TP)
- **Applied to:** Both outputs during training
- **Method:** `tf.data.Dataset` with weights as 3rd tuple element

```python
# CORRECT way to pass sample weights:
train_dataset = tf.data.Dataset.from_tensor_slices((
    X_train,
    {
        'direction_output': y_direction,
        'heatmap_output': y_heatmap
    },
    sample_weights  # 3rd element
)).batch(batch_size)

model.fit(train_dataset, epochs=10)
```

---

## Common Mistakes to Avoid

### ❌ WRONG: Treating classes as "Up/Down/Sideways"
```python
# NO! Classes are from triple barrier outcomes
# Class 0 = SL hit first (not generic "down")
# Class 2 = TP hit first (not generic "up")
```

### ❌ WRONG: Using heatmap_output for binary classification
```python
# NO! Heatmap is auxiliary confidence signal
prob_tp = predictions[1]  # WRONG - this is signed confidence
```

### ✅ CORRECT: Use direction_output probabilities
```python
# YES! Use class 2 probability for binary TP prediction
direction_probs = predictions[0]
prob_tp = direction_probs[:, 2]  # CORRECT
```

### ❌ WRONG: No embargo in walk-forward
```python
# NO! Testing on data immediately after training = temporal leakage
train_end = t
test_start = t  # WRONG - no gap
```

### ✅ CORRECT: Include embargo period
```python
# YES! Add gap to prevent lookahead
train_end = t - embargo_samples
test_start = t  # CORRECT - temporal separation
```

---

## Evaluation Metrics

### For Multi-class (3-class):
```python
from sklearn.metrics import classification_report

# Get predicted class
y_pred_class = np.argmax(direction_probs, axis=1)

# Multi-class metrics
report = classification_report(y_true, y_pred_class,
                               target_names=['SL', 'Timeout', 'TP'])
```

### For Binary (SL vs TP only):
```python
from sklearn.metrics import roc_auc_score, accuracy_score

# Use only non-timeout samples (done in preprocessing)
# y_true is binary: 0=SL, 1=TP

prob_tp = direction_probs[:, 2]
y_pred_binary = (prob_tp > 0.5).astype(int)

auc = roc_auc_score(y_true, prob_tp)
accuracy = accuracy_score(y_true, y_pred_binary)
```

---

## Walk-Forward Validation (López de Prado)

### Correct Implementation:
```python
# With embargo to prevent temporal leakage
embargo_pct = 0.01  # 1% of training size

for fold in range(n_folds):
    # Train on past
    train_start = fold * test_size
    train_end = train_start + train_size - embargo_samples
    
    # GAP (embargo period)
    # ... embargo_samples removed ...
    
    # Test on future
    test_start = train_start + train_size
    test_end = test_start + test_size
    
    # Train fresh model
    model = build_financial_cnn_lstm(...)
    model.fit(X[train_start:train_end], ...)
    
    # Evaluate on truly future data
    predictions = model.predict(X[test_start:test_end])
    auc = roc_auc_score(y[test_start:test_end], predictions[0][:, 2])
```

---

## Summary

| Component | Details |
|-----------|---------|
| **Model Type** | Dual-output CNN-LSTM |
| **Input** | (batch, 20 days, 15 features) |
| **Output 1** | direction_output: (batch, 3) - SL/Timeout/TP probs |
| **Output 2** | heatmap_output: (batch, 1) - signed confidence |
| **Ground Truth** | Triple barrier method (TP=+6%, SL=-3%, horizon=5) |
| **Training** | Both outputs, with sample weights |
| **Evaluation** | Binary AUC using prob(TP) from direction_output |
| **Validation** | Walk-forward with embargo period |

---

## Quick Reference

```python
# Load model
model = keras.models.load_model('models/best_model.keras')

# Predict
predictions = model.predict(X_test)
direction_probs = predictions[0]  # (n, 3)
heatmap_conf = predictions[1]     # (n, 1)

# Binary prediction
prob_tp = direction_probs[:, 2]
y_pred = (prob_tp > 0.5).astype(int)

# Metrics
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, prob_tp)
print(f"AUC: {auc:.4f}")
```
