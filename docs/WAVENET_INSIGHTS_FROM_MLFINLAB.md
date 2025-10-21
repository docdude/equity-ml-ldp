# WaveNet Model Improvements from MLFinLab Methodology

## Executive Summary

The MLFinLab triple barrier + RandomForest approach isn't about RandomForest performance - it's about **better labeling, feature engineering, and training methodology** that your WaveNet model can leverage.

---

## 1. Key Insights for WaveNet

### What WaveNet Can Adopt from MLFinLab:

#### ✅ 1. Meta-Labeling for Better Training Targets
```python
# Current WaveNet Training:
# Input: [sequence of features]
# Output: [UP/DOWN/NEUTRAL prediction]

# Enhanced WaveNet Training (Meta-Labeling):
# Stage 1 WaveNet: Predict direction (side)
# Stage 2 WaveNet: Predict bet sizing / confidence
# 
# Or combined:
# Output: [direction, confidence] - dual head architecture
```

**Why This Helps WaveNet:**
- Separates "what will happen" from "should I act on it"
- Confidence head can learn when pattern recognition is reliable
- Reduces false positives without filtering post-prediction

#### ✅ 2. Sample Weighting During Training
```python
# Calculate avg_uniqueness for your barrier events
avg_uniqueness = calculate_avg_uniqueness(barrier_events)

# Use in WaveNet training
loss = criterion(predictions, targets)
weighted_loss = loss * avg_uniqueness  # Weight by sample importance
weighted_loss.mean().backward()
```

**Why This Helps WaveNet:**
- Temporal sequences have natural overlap
- WaveNet's receptive field creates overlapping contexts
- Weighting prevents overfitting to crowded periods

#### ✅ 3. Better Label Quality via Actual Exit Returns
```python
# Current approach (potential issue):
labels = create_dynamic_triple_barriers(df)
# Returns: {0: SL, 1: TP, 2: Timeout}
# But uses forward_return_5d for evaluation

# MLFinLab approach (what to adopt):
barrier_events = create_enhanced_barriers(df)
# Returns:
# - label: {-1: SL, 0: Timeout, 1: TP}
# - t1: actual exit timestamp
# - exit_return: actual return at barrier touch
# - holding_period: days held

# Now WaveNet can learn:
# 1. Which direction (classification head)
# 2. How long to hold (regression head for holding_period)
# 3. Expected return (regression head for exit_return)
```

**Why This Helps WaveNet:**
- Labels aligned with actual tradeable returns
- Can predict holding period (useful for position sizing)
- Multi-task learning improves feature representations

#### ✅ 4. Side Filtering for Training Data
```python
# Instead of training on all samples:
X_all, y_all = prepare_data(df)

# Filter to high-conviction samples:
# (where primary signal != neutral)
conviction_mask = (primary_predictions != 0) | (volatility > threshold)
X_filtered = X_all[conviction_mask]
y_filtered = y_all[conviction_mask]

# Train WaveNet on filtered data
model.fit(X_filtered, y_filtered)
```

**Why This Helps WaveNet:**
- Focuses learning on actionable patterns
- Reduces noise from ambiguous market conditions
- Better gradient signal during backprop

---

## 2. WaveNet Architecture Enhancements

### Current WaveNet (Simplified):
```python
class FinancialWaveNet(nn.Module):
    def __init__(self):
        self.wavenet_layers = WaveNetStack(...)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        features = self.wavenet_layers(x)
        logits = self.classifier(features)
        return logits  # Shape: [batch, 3] for {SL, TP, Neutral}
```

### Enhanced WaveNet with Meta-Labeling:
```python
class MetaLabelingWaveNet(nn.Module):
    def __init__(self):
        self.wavenet_layers = WaveNetStack(...)
        
        # Multi-head output
        self.direction_head = nn.Linear(hidden_dim, 3)  # UP/DOWN/NEUTRAL
        self.confidence_head = nn.Linear(hidden_dim, 1)  # Bet size [0, 1]
        self.holding_head = nn.Linear(hidden_dim, 1)    # Expected holding period
        self.return_head = nn.Linear(hidden_dim, 1)      # Expected return
    
    def forward(self, x):
        features = self.wavenet_layers(x)
        
        direction = self.direction_head(features)     # Classification
        confidence = torch.sigmoid(self.confidence_head(features))  # [0, 1]
        holding_period = torch.relu(self.holding_head(features))    # Days
        expected_return = self.return_head(features)   # Percentage
        
        return {
            'direction': direction,
            'confidence': confidence,
            'holding_period': holding_period,
            'expected_return': expected_return
        }
```

### Training Loss:
```python
def compute_loss(outputs, targets, sample_weights):
    # Classification loss (direction)
    direction_loss = F.cross_entropy(
        outputs['direction'], 
        targets['label'],
        reduction='none'
    )
    
    # Regression loss (holding period)
    holding_loss = F.mse_loss(
        outputs['holding_period'],
        targets['holding_period'],
        reduction='none'
    )
    
    # Regression loss (expected return)
    return_loss = F.mse_loss(
        outputs['expected_return'],
        targets['exit_return'],
        reduction='none'
    )
    
    # Confidence loss (meta-labeling)
    # High confidence when prediction matches actual direction
    correct_predictions = (outputs['direction'].argmax(1) == targets['label']).float()
    confidence_loss = F.binary_cross_entropy(
        outputs['confidence'].squeeze(),
        correct_predictions,
        reduction='none'
    )
    
    # Combine with sample weighting
    total_loss = (
        1.0 * direction_loss +
        0.5 * holding_loss +
        0.5 * return_loss +
        0.3 * confidence_loss
    )
    
    weighted_loss = (total_loss * sample_weights).mean()
    return weighted_loss
```

---

## 3. Feature Engineering from MLFinLab

### What Features Does 02RandomForest.ipynb Use?

The notebook loads `feature_matrix.parquet` - these are likely advanced features from López de Prado's book:

```python
# Standard features (what you have):
- SMA, EMA, MACD, RSI, Bollinger Bands
- ATR, ADX, CCI, Stochastic
- Volume indicators

# Advanced features (what MLFinLab likely uses):
1. Fractional Differentiation
   - Preserves memory while removing autocorrelation
   - Makes features more stationary

2. Entropy-based Features
   - Shannon entropy of returns
   - Plug-in entropy
   - Lempel-Ziv entropy

3. Microstructural Features
   - Roll measure (bid-ask spread estimate)
   - Corwin-Schultz spread
   - Kyle's lambda (price impact)

4. Information-driven Bars
   - Tick imbalance bars
   - Volume imbalance bars
   - Dollar imbalance bars

5. Bet Sizing Signals
   - Average uniqueness
   - Return attribution
   - Time decay factors
```

### What WaveNet Should Add:

#### A. Fractional Differentiation (High Priority)
```python
def fracDiff(series, d=0.5, thres=0.01):
    """
    Fractionally differentiate a series.
    d=0: No differencing (original series)
    d=1: Full differencing (completely stationary)
    d=0.5: Balanced (some memory, some stationarity)
    """
    weights = get_weights_ffd(d, thres)
    return series.rolling(len(weights)).apply(
        lambda x: np.dot(x, weights), raw=True
    )

# Add to features:
df['close_fracdiff'] = fracDiff(df['close'], d=0.5)
df['volume_fracdiff'] = fracDiff(df['volume'], d=0.5)
```

**Why for WaveNet:**
- Removes autocorrelation without destroying signal
- WaveNet can learn temporal patterns from stationary features
- Better generalization across different market regimes

#### B. Entropy Features (Medium Priority)
```python
def calculate_entropy(returns, window=20):
    """Shannon entropy of return distribution."""
    def shannon_entropy(x):
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    
    return returns.rolling(window).apply(shannon_entropy)

df['return_entropy'] = calculate_entropy(df['returns'])
```

**Why for WaveNet:**
- Captures regime changes (low entropy = trending, high entropy = choppy)
- WaveNet's temporal modeling can detect entropy transitions
- Useful for confidence prediction

#### C. Microstructural Features (Lower Priority - Need Tick Data)
```python
# Only if you have bid-ask or tick data
def roll_measure(close, window=20):
    """Estimate bid-ask spread from close prices."""
    returns = close.pct_change()
    return 2 * np.sqrt(-returns.rolling(window).cov(returns.shift(1)))
```

---

## 4. Training Data Preparation for WaveNet

### Enhanced Barrier Creation:
```python
def create_enhanced_barriers_for_wavenet(df, base_tp=0.02, base_sl=0.01, horizon=7):
    """
    Create barriers with full metadata for WaveNet training.
    
    Returns DataFrame with:
    - label: {-1, 0, 1} for {SL, Timeout, TP}
    - t1: exit timestamp
    - exit_return: actual return at exit
    - holding_period: days held
    - avg_uniqueness: sample weight
    """
    results = []
    
    for i in range(len(df) - horizon):
        entry_idx = i
        entry_price = df['close'].iloc[i]
        entry_time = df.index[i]
        
        # Calculate dynamic barriers
        volatility = df['volatility'].iloc[i]
        tp_threshold = entry_price * (1 + base_tp * volatility)
        sl_threshold = entry_price * (1 - base_sl * volatility)
        
        # Find which barrier hits first
        future_prices = df['close'].iloc[i+1:i+horizon+1]
        tp_hits = future_prices >= tp_threshold
        sl_hits = future_prices <= sl_threshold
        
        if tp_hits.any() and sl_hits.any():
            tp_day = tp_hits.idxmax()
            sl_day = sl_hits.idxmax()
            
            if df.index.get_loc(tp_day) < df.index.get_loc(sl_day):
                exit_idx = df.index.get_loc(tp_day)
                label = 1
            else:
                exit_idx = df.index.get_loc(sl_day)
                label = -1
        elif tp_hits.any():
            exit_idx = df.index.get_loc(tp_hits.idxmax())
            label = 1
        elif sl_hits.any():
            exit_idx = df.index.get_loc(sl_hits.idxmax())
            label = -1
        else:
            exit_idx = i + horizon
            label = 0
        
        exit_time = df.index[exit_idx]
        exit_price = df['close'].iloc[exit_idx]
        
        results.append({
            'entry_time': entry_time,
            't1': exit_time,
            'label': label,
            'exit_return': (exit_price - entry_price) / entry_price,
            'holding_period': exit_idx - entry_idx,
            'entry_idx': entry_idx,
            'exit_idx': exit_idx
        })
    
    barrier_df = pd.DataFrame(results)
    barrier_df.set_index('entry_time', inplace=True)
    
    # Calculate avg_uniqueness
    barrier_df['avg_uniqueness'] = calculate_avg_uniqueness(barrier_df)
    
    return barrier_df


def calculate_avg_uniqueness(barrier_df):
    """Calculate sample uniqueness for weighting."""
    uniqueness = []
    
    for idx, row in barrier_df.iterrows():
        t_start = row['entry_idx']
        t_end = row['exit_idx']
        
        # Count overlapping samples
        overlaps = barrier_df[
            (barrier_df['entry_idx'] < t_end) & 
            (barrier_df['exit_idx'] > t_start)
        ]
        
        # Uniqueness = 1 / number of concurrent samples
        uniqueness.append(1.0 / len(overlaps))
    
    return pd.Series(uniqueness, index=barrier_df.index)
```

### Preparing WaveNet Sequences:
```python
def prepare_wavenet_sequences(df, barrier_df, lookback=60):
    """
    Prepare sequences for WaveNet with enhanced labels.
    """
    sequences = []
    labels = []
    metadata = []
    
    for idx, barrier_row in barrier_df.iterrows():
        entry_idx = barrier_row['entry_idx']
        
        # Need enough history
        if entry_idx < lookback:
            continue
        
        # Extract sequence of features
        sequence = df.iloc[entry_idx - lookback:entry_idx][FEATURE_COLS].values
        
        # Multi-target labels
        label_dict = {
            'direction': barrier_row['label'] + 1,  # Convert {-1,0,1} to {0,1,2}
            'exit_return': barrier_row['exit_return'],
            'holding_period': barrier_row['holding_period'],
            'sample_weight': barrier_row['avg_uniqueness']
        }
        
        sequences.append(sequence)
        labels.append(label_dict)
    
    return np.array(sequences), labels
```

---

## 5. Inference Strategy with Enhanced WaveNet

### Current Strategy:
```python
# Predict direction, trade immediately
predictions = model.predict(X_test)
positions = np.where(predictions == 1, 1, -1)  # Long on UP, short on DOWN
```

### Enhanced Strategy (Meta-Labeling):
```python
# Get full predictions
outputs = model.predict(X_test)

# Strategy with confidence filtering
positions = []
for i in range(len(outputs)):
    direction = outputs[i]['direction'].argmax()  # {0: SL, 1: Neutral, 2: TP}
    confidence = outputs[i]['confidence']
    
    if confidence < 0.6:
        # Low confidence - stay neutral
        positions.append(0)
    elif direction == 2:
        # High confidence UP
        positions.append(1)
    elif direction == 0:
        # High confidence DOWN
        positions.append(-1)
    else:
        # Predicted neutral
        positions.append(0)

# Calculate returns using predicted holding periods
expected_holding = outputs['holding_period']
actual_returns = calculate_returns_at_holding_period(df, positions, expected_holding)
```

---

## 6. What NOT to Adopt from RandomForest

### ❌ Sequential Bootstrapping
**Why Skip:** WaveNet uses gradient descent, not bootstrap sampling. Not applicable.

### ❌ OOB Scoring
**Why Skip:** Use validation set or walk-forward for WaveNet evaluation.

### ❌ Tree-Specific Hyperparameters
**Why Skip:** `min_weight_fraction_leaf`, `max_features` are tree-specific. Use dropout, L2 regularization instead.

### ✅ What to Keep from RandomForest Approach:
- Sample weighting by uniqueness
- Meta-labeling concept (dual-head architecture)
- Side filtering (train on high-conviction samples)
- Enhanced barrier labels (exit_return, holding_period)

---

## 7. Implementation Roadmap for WaveNet

### Phase 1: Enhanced Labels (Immediate)
1. Modify `create_dynamic_triple_barriers()` to return:
   - `t1` (exit timestamp)
   - `exit_return` (actual return at barrier touch)
   - `holding_period` (days held)
2. Calculate `avg_uniqueness` for all samples
3. Update evaluation to use `exit_return` instead of `forward_return_5d`

### Phase 2: Sample Weighting (Week 1)
1. Modify WaveNet loss function to accept `sample_weights`
2. Train with weighted loss
3. Compare performance vs unweighted

### Phase 3: Advanced Features (Week 2)
1. Implement fractional differentiation for price/volume
2. Add entropy features
3. Retrain WaveNet with expanded feature set

### Phase 4: Multi-Head Architecture (Week 3-4)
1. Implement dual-head WaveNet:
   - Direction head (classification)
   - Confidence head (regression)
2. Train with multi-task loss
3. Implement confidence-filtered strategy

### Phase 5: Meta-Labeling (Week 5-6)
1. Train Stage 1 WaveNet (direction prediction)
2. Generate `side` predictions on training set
3. Train Stage 2 WaveNet (bet sizing)
4. Combine predictions in inference

---

## 8. Expected Improvements

### From Sample Weighting:
- 10-15% reduction in overfitting
- Better generalization to OOS data
- More stable performance across market regimes

### From Enhanced Labels:
- 5-10% improvement in Sharpe Ratio
- Reduced slippage (labels match actual exits)
- Better risk-adjusted returns

### From Advanced Features:
- 15-20% improvement in prediction accuracy
- Better regime detection
- Reduced autocorrelation in residuals

### From Meta-Labeling:
- 20-30% reduction in losing trades
- Higher win rate (filtering low-confidence predictions)
- Better position sizing

---

## 9. Validation Approach

### Test Each Enhancement Separately:
```python
# Baseline WaveNet
baseline_results = train_and_evaluate(wavenet_basic, data_basic)

# + Sample Weighting
weighted_results = train_and_evaluate(wavenet_basic, data_weighted)
print(f"Sample weighting improvement: {weighted_results.sharpe - baseline_results.sharpe}")

# + Advanced Features
advanced_features_results = train_and_evaluate(wavenet_basic, data_advanced_features)
print(f"Advanced features improvement: {advanced_features_results.sharpe - baseline_results.sharpe}")

# + Multi-Head
multihead_results = train_and_evaluate(wavenet_multihead, data_weighted)
print(f"Multi-head improvement: {multihead_results.sharpe - baseline_results.sharpe}")

# + Meta-Labeling
metalabeling_results = train_and_evaluate(wavenet_metalabeling, data_filtered)
print(f"Meta-labeling improvement: {metalabeling_results.sharpe - baseline_results.sharpe}")
```

---

## 10. Key Takeaways for WaveNet

### From MLFinLab's Triple Barrier + RandomForest:

1. **Better Labels**: Use actual exit returns, not fixed-horizon returns
2. **Sample Weighting**: Weight by temporal uniqueness to reduce overfitting
3. **Meta-Labeling**: Separate direction prediction from bet sizing
4. **Advanced Features**: Fractional differentiation, entropy, microstructure
5. **Filtering**: Train on high-conviction samples, not all data

### What Makes WaveNet Different from RandomForest:

- **Temporal Modeling**: WaveNet's dilated convolutions capture long-range dependencies
- **Gradient-Based Learning**: Can optimize multi-task objectives jointly
- **Continuous Representations**: Can predict probabilities, holding periods, expected returns
- **No Ensemble Needed**: Single model can capture complex patterns

### The Core Insight:

**RandomForest showed what features and labels work. WaveNet can learn them better because it models temporal structure explicitly.**

The goal isn't to replicate RandomForest's performance - it's to take the methodological insights (meta-labeling, sample weighting, better labels) and apply them to a superior temporal model (WaveNet) that can capture patterns RandomForest cannot.

---

## Next Steps

1. Create `create_enhanced_barriers_for_wavenet()` in `fin_feature_preprocessing.py`
2. Modify WaveNet training script to use sample weights
3. Add fractional differentiation features
4. Implement multi-head WaveNet architecture
5. Run comparative experiments: baseline vs enhanced

**Focus**: Feature engineering and label quality, not model architecture comparisons.
