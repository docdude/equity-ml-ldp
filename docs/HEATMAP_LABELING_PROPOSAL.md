# Gaussian Heatmap Labeling for Triple Barriers

## Problem with Current Approach

**Current**: Discrete 3-class classification `{DOWN/SL, TIMEOUT, UP/TP}`

**Issues**:
- Model biased toward TIMEOUT (majority class ~63%)
- Hard boundary: either TP or SL, no uncertainty
- Temporal rigidity: Must predict exact outcome at exact timepoint
- Sample weights and focal loss can't fully overcome class imbalance

## Proposed Solution: Gaussian Heatmap Labels

Adapted from swimming stroke detection - replace discrete classes with continuous heatmaps.

### Architecture Change

**Old Output**: Softmax over 3 classes `[DOWN, TIMEOUT, UP]`

**New Output**: Two sigmoid heatmap channels
- `TP heatmap`: (N, 1) continuous [0, 1] - likelihood of take-profit in nearby future
- `SL heatmap`: (N, 1) continuous [0, 1] - likelihood of stop-loss in nearby future

### Label Generation

For each barrier event at time `t` with outcome:
- **UP/TP hit**: Create Gaussian bump in TP heatmap centered at `t`
- **DOWN/SL hit**: Create Gaussian bump in SL heatmap centered at `t`
- **TIMEOUT**: No bump (both heatmaps stay at 0)

```python
# Pseudocode
def create_barrier_heatmaps(events, labels, sigma=2.0):
    """
    events: DataFrame with barrier events
    labels: Series with {-1, 0, 1} outcomes
    sigma: Gaussian width in days
    """
    n_samples = len(labels)
    tp_heatmap = np.zeros(n_samples)
    sl_heatmap = np.zeros(n_samples)
    
    # Build Gaussian kernel
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    
    # Mark TP events
    tp_events = (labels == 1).astype(float)
    tp_heatmap = np.convolve(tp_events, kernel, mode='same')
    
    # Mark SL events
    sl_events = (labels == -1).astype(float)
    sl_heatmap = np.convolve(sl_events, kernel, mode='same')
    
    # Clip to [0, 1]
    tp_heatmap = np.clip(tp_heatmap, 0.0, 1.0)
    sl_heatmap = np.clip(sl_heatmap, 0.0, 1.0)
    
    return tp_heatmap, sl_heatmap
```

### Model Architecture

```python
# Input: (batch, seq_len, features)
x = WaveNet(inputs)

# Global pooling
pooled = GlobalAveragePooling1D()(x)

# Two separate sigmoid heads (NOT softmax!)
tp_head = Dense(1, activation='sigmoid', name='tp_heatmap')(pooled)
sl_head = Dense(1, activation='sigmoid', name='sl_heatmap')(pooled)

model = Model(inputs, [tp_head, sl_head])
```

### Loss Function

Use **Binary Focal Loss** for each heatmap (handles class imbalance better):

```python
tp_loss = BinaryFocalCrossentropy(alpha=0.75, gamma=2.0)(y_true_tp, y_pred_tp)
sl_loss = BinaryFocalCrossentropy(alpha=0.75, gamma=2.0)(y_true_sl, y_pred_sl)
total_loss = tp_loss + sl_loss
```

### Inference / Trading Signals

```python
# Get predictions
tp_pred = model.predict(X)[:, 0]  # (N,) TP probabilities
sl_pred = model.predict(X)[:, 1]  # (N,) SL probabilities

# Convert to trading signals
confidence_threshold = 0.3
net_signal = tp_pred - sl_pred  # [-1, 1] range

# Long when TP > SL and confidence high
long_mask = (tp_pred > sl_pred) & (np.abs(net_signal) > confidence_threshold)
signals[long_mask] = net_signal[long_mask]

# Short when SL > TP and confidence high
short_mask = (sl_pred > tp_pred) & (np.abs(net_signal) > confidence_threshold)
signals[short_mask] = net_signal[short_mask]

# Neutral when both low or similar
# (automatically handled - signals stay at 0)
```

### Advantages

1. **No explicit TIMEOUT class** → Can't get stuck predicting majority class
2. **Soft targets** → Better gradient flow during training
3. **Temporal flexibility** → "TP likely in next few days" vs "TP on exact day"
4. **Natural confidence** → Heatmap magnitude = model certainty
5. **Handles near-misses** → Price almost hit barrier → partial heatmap activation
6. **Better class balance** → Binary problems easier than 3-class
7. **Interpretable** → Can visualize predicted TP/SL probabilities over time

### Implementation Steps

1. Modify `create_mlfinlab_barriers()` to return heatmap labels
2. Update model to output 2 sigmoid heads instead of 3-class softmax
3. Change loss to binary focal loss for each channel
4. Update inference to use `tp_pred - sl_pred` for signals
5. Adjust PBO analysis to work with continuous predictions

### Hyperparameters

- `sigma`: Gaussian width (2-3 days recommended)
- `alpha`: Focal loss weight (0.75 good starting point)
- `gamma`: Focal loss focusing (2.0 reduces easy examples)
- `confidence_threshold`: Min |tp - sl| to take position (0.2-0.4)

### Expected Results

- **More balanced predictions**: No more 87% TIMEOUT
- **Better Sharpe ratios**: More selective signals
- **Lower PBO**: Less overfitting to specific class distribution
- **Smoother training**: Continuous targets vs discrete classes
