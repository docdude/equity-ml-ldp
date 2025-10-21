# fin_training_ldp.py - López de Prado Implementation Guide

## Overview

`fin_training_ldp.py` is a refactored training script that implements proper **triple barrier labeling** from López de Prado's methodology using the MLFinLab library. This replaces the custom `create_dynamic_triple_barriers()` with the academically-validated MLFinLab implementation.

## Key Differences from fin_training.py

### 1. **Labeling Method**

| Aspect | fin_training.py (Custom) | fin_training_ldp.py (MLFinLab) |
|--------|-------------------------|--------------------------------|
| **Implementation** | Custom `create_dynamic_triple_barriers()` | MLFinLab `get_events()` + `meta_labeling()` |
| **Labels** | 3-class: {-1=TIMEOUT, 0=SL, 1=TP} | 2-class: {-1=DOWN/SL, 1=UP/TP} |
| **Exit Strategy** | All 3 barriers (including timeout) | Only profit/loss barriers (timeout labeled by side) |
| **Returns** | Calculated at barrier touch | Meta-labeled with actual returns |
| **Validation** | Custom alignment checks | Built-in alignment validation |

### 2. **Triple Barrier Parameters**

```python
# fin_training_ldp.py
BARRIER_PARAMS = {
    'lookback': 60,       # Days for volatility EWMA
    'pt_sl': [2, 1],      # Asymmetric: TP=2×σ, SL=1×σ
    'min_ret': 0.005,     # 0.5% minimum return (noise filter)
    'num_days': 7,        # 7-day vertical barrier
    'num_threads': 1      # Parallel processing threads
}
```

**Key advantages:**
- **Asymmetric barriers** (`[2, 1]`): Reflects real trading (TP > SL)
- **Minimum return threshold**: Filters out noisy signals
- **Dynamic volatility**: EWMA adapts to market regimes
- **MLFinLab parallelization**: Faster event computation

### 3. **Label Philosophy**

#### Custom Implementation (fin_training.py)
```python
label = -1  # TIMEOUT (neutral, no clear direction)
label = 0   # STOP_LOSS (down/loss)
label = 1   # TAKE_PROFIT (up/profit)
```
- **Problem**: 74.6% timeout labels → mostly neutral
- **Result**: Model struggles to find signal (49% val accuracy)

#### MLFinLab Implementation (fin_training_ldp.py)
```python
label = -1  # DOWN (stop-loss OR negative timeout)
label = 1   # UP (take-profit OR positive timeout)
```
- **Advantage**: Binary classification (simpler, more balanced)
- **Philosophy**: Every position ends up/down (no "neutral")
- **Result**: Better class balance, clearer signal

### 4. **Concurrency Implications**

**Finding from analysis:**
- Current stride=1 creates **211x redundancy** (uniqueness=0.005)
- 5268 samples → **~25 effective samples** across all tickers

**Solutions (NOT in current script, future enhancement):**

```python
# Option 1: Increase stride (eliminate overlap)
events_non_overlapping = events.iloc[::5]  # Sample every 5th row

# Option 2: Add uniqueness weighting
from MLFinance.FinancialMachineLearning.sample_weights.attribution import weights_by_return
from MLFinance.FinancialMachineLearning.sample_weights.concurrency import average_uniqueness_triple_barrier

# Calculate uniqueness
uniqueness = average_uniqueness_triple_barrier(events, close_series, num_threads=1)

# Calculate return-based weights
return_weights = weights_by_return(events, close_series, num_threads=1)

# Combine with class weights
final_weights = class_weights * uniqueness['tW'] * return_weights
```

## MLFinLab Functions Used

### `add_vertical_barrier()`
```python
vertical_barriers = add_vertical_barrier(
    t_events=df.index,
    close=df['Close'],
    num_days=7  # Expiration limit
)
```
- Creates time-based exit points
- Returns Series of timestamps

### `daily_volatility()`
```python
volatility = daily_volatility(
    close=df['Close'],
    lookback=60  # EWMA span
)
```
- Exponentially weighted moving average
- Used as dynamic barrier threshold

### `get_events()`
```python
events = get_events(
    close=df['Close'],
    t_events=df.index[60:],  # After warmup
    pt_sl=[2, 1],            # [TP_multiple, SL_multiple]
    target=volatility,       # Dynamic threshold
    min_ret=0.005,           # Minimum 0.5% return
    num_threads=1,
    vertical_barrier_times=vertical_barriers,
    side_prediction=None     # No primary model
)
```
- Returns DataFrame with columns:
  - `t1`: Exit timestamp (when barrier hit)
  - `trgt`: Target threshold used
  - `pt`: Profit-taking level
  - `sl`: Stop-loss level

### `meta_labeling()`
```python
labels = meta_labeling(
    triple_barrier_events=events,
    close=df['Close']
)
```
- Returns DataFrame with columns:
  - `ret`: Actual return at barrier touch
  - `trgt`: Target threshold
  - `bin`: Label (1=UP, -1=DOWN)
- Validates alignment (UP labels → positive returns)

## Usage

### Basic Training
```bash
python fin_training_ldp.py
```

### Configuration Options

Edit parameters in script:

```python
# Tickers
tickers = ['AAPL', 'NVDA', 'TSLA']  # Adjust list

# Triple barriers
BARRIER_PARAMS = {
    'lookback': 60,      # Increase for smoother volatility
    'pt_sl': [2, 1],     # Try [3, 1] for wider TP
    'min_ret': 0.005,    # Lower to 0.003 for more samples
    'num_days': 7,       # Increase to 10 for longer horizon
}

# Feature preset
CONFIG_PRESET = 'wavenet_optimized'  # or 'minimal', 'balanced'
```

## Expected Outputs

### Training Artifacts
```
run_financial_wavenet_ldp_v1/
├── final_model.keras          # Final trained model
├── best_model.keras           # Best validation checkpoint
├── training_log.csv           # Epoch-by-epoch metrics
├── metadata.pkl               # Full configuration
├── triple_barrier_events.csv  # Event data for analysis
├── normalizer.pkl             # Feature normalization params
└── tensorboard_logs/          # TensorBoard visualizations
```

### Label Distribution Example
```
DOWN/Stop-Loss   (-1):  2,145 (40.7%) ████████████████████
UP/Take-Profit   ( 1):  3,123 (59.3%) █████████████████████████████
```
- More balanced than 3-class (TIMEOUT dominated)
- UP bias expected (markets trend up over time)

### Alignment Verification
```
UP labels (expect positive returns):
  Mean return: 0.0234 (2.34%)
  % positive:  87.3%
  ✅ GOOD: UP labels mostly positive

DOWN labels (expect negative returns):
  Mean return: -0.0178 (-1.78%)
  % negative:  84.6%
  ✅ GOOD: DOWN labels mostly negative
```

## Advanced Features (Future Implementation)

### 1. Sample Weighting Enhancement

```python
# Add to fin_training_ldp.py after label generation

from MLFinance.FinancialMachineLearning.sample_weights.attribution import (
    weights_by_return,
    weights_by_time_decay
)
from MLFinance.FinancialMachineLearning.sample_weights.concurrency import (
    average_uniqueness_triple_barrier
)

# Calculate uniqueness (down-weights redundant overlapping labels)
uniqueness = average_uniqueness_triple_barrier(
    triple_barrier_events=events_combined,
    close_series=all_closes_series,  # Need to combine
    num_threads=4
)

# Calculate return attribution (up-weights high-magnitude events)
return_weights = weights_by_return(
    triple_barrier_events=events_combined,
    close_series=all_closes_series,
    num_threads=4
)

# Combine: class_balance × uniqueness × return_magnitude
final_weights = (
    sample_weights_train *        # Class balance
    uniqueness.loc[train_idx] *   # Uniqueness
    return_weights.loc[train_idx] # Return magnitude
)

# Normalize
final_weights = final_weights / final_weights.mean()
```

### 2. Meta-Labeling (Bet Sizing)

If you have a primary model that predicts direction:

```python
# Step 1: Get primary model predictions
primary_predictions = primary_model.predict(X)  # {-1, 1}

# Step 2: Create events with side prediction
events = get_events(
    close=df['Close'],
    t_events=df.index,
    pt_sl=[2, 1],
    target=volatility,
    min_ret=0.005,
    num_threads=1,
    vertical_barrier_times=vertical_barriers,
    side_prediction=primary_predictions  # ← Add primary model
)

# Step 3: Meta-labeling returns {0, 1} (bet or no bet)
meta_labels = meta_labeling(events, df['Close'])
# meta_labels['bin'] ∈ {0, 1}
# 1 = take the bet (sized by model confidence)
# 0 = skip the bet
```

**Benefits:**
- Separates direction (primary model) from sizing (ML model)
- Reduces overfitting (ML only learns when to bet, not direction)
- Better F1 scores (filters false positives)

### 3. Fractional Differentiation

```python
from MLFinance.FinancialMachineLearning.features.fracdiff import frac_diff_ffd

# Make prices stationary while preserving memory
prices_stationary = frac_diff_ffd(
    df['Close'],
    d=0.5,  # Fractional degree (0.5 often optimal)
    thres=0.01
)

# Use stationary prices for feature engineering
```

**Benefits:**
- Removes trend while preserving short-term memory
- Better than integer differentiation (d=1)
- Improves model generalization

## Comparison with Swimming WaveNet

### Similarities with `train_wavenet.py`

```python
# Both use multiplicative sample weighting
final_swim_weights = val_sample_weights * validation_style_sample_weights
final_swim_weights = final_swim_weights / np.mean(final_swim_weights)
```

**From swimming model:**
- `val_sample_weights`: Per-sample balanced weights
- `validation_style_sample_weights`: Global class distribution weights
- **Multiply and normalize**

**Equivalent for finance:**
```python
final_weights = (
    class_weights *      # Per-sample balanced
    uniqueness_weights * # Label overlap correction (López addition)
    return_weights       # Return magnitude weighting (López addition)
)
final_weights = final_weights / final_weights.mean()
```

### Key Difference

Swimming model has:
- **Spatial features** (ACC, GYRO 6-axis)
- **Temporal patterns** (stroke cycles)
- **Class imbalance** (stroke styles)

Finance model has:
- **Temporal features** (prices, volumes)
- **Label overlap** (concurrency from sliding windows)
- **Return heterogeneity** (some events matter more)

→ Finance needs **uniqueness + return weights** in addition to class weights

## Next Steps

### Immediate Actions

1. **Run the new script:**
   ```bash
   python fin_training_ldp.py
   ```

2. **Compare results:**
   ```bash
   # Old method
   cat run_financial_wavenet_v1/training_log.csv | tail -5
   
   # New method
   cat run_financial_wavenet_ldp_v1/training_log.csv | tail -5
   ```

3. **Check label balance:**
   - Old: 74.6% TIMEOUT (neutral)
   - New: Should be ~40-60% split (DOWN/UP)

### Performance Expectations

| Metric | Old (Custom) | Expected (MLFinLab) | Reason |
|--------|-------------|---------------------|--------|
| **Val Accuracy** | 49.37% | 55-65% | Better label balance |
| **Val AUC** | 0.6991 | 0.70-0.75 | Clearer signal |
| **Sharpe** | -1.8448 | 0.0-1.0 | Aligned labels |
| **Overfitting** | 14.58% | 10-15% | Similar |

### Future Enhancements (Priority Order)

1. **Add uniqueness weighting** (High impact, 2 hours)
   - Fixes 211x redundancy problem
   - Expected: 5268 → ~1000 effective samples
   - Should improve generalization significantly

2. **Add return attribution weighting** (High impact, 1 hour)
   - Focus learning on high-magnitude events
   - Down-weight small noisy returns
   - Improves signal-to-noise ratio

3. **Try stride=5 sampling** (Quick test, 30 mins)
   ```python
   events_non_overlapping = events.iloc[::5]
   ```
   - Eliminates overlap entirely
   - Trades quantity for quality
   - May work better for WaveNet

4. **Implement meta-labeling** (Strategic, 4 hours)
   - Requires primary model (technical indicators?)
   - Separates direction from sizing
   - More robust than pure ML

5. **Add fractional differentiation** (Advanced, 3 hours)
   - Makes prices stationary
   - Preserves short-term memory
   - Better feature quality

6. **Try CNN-LSTM Gaussian heatmap** (Alternative path)
   - Already implemented in `cnn-lstm/`
   - Continuous targets instead of discrete
   - May avoid class imbalance issues

## References

### López de Prado's "Advances in Financial Machine Learning"

- **Chapter 3**: Triple Barrier Method
  - Section 3.2: Vertical Barriers
  - Section 3.3: Profit Taking and Stop Loss
  - Section 3.4: Learning Side and Size

- **Chapter 4**: Sample Weights
  - Section 4.3: Label Uniqueness
  - Section 4.4: Return Attribution
  - Section 4.5: Time Decay

- **Chapter 5**: Fractional Differentiation
  - Section 5.2: The Fixed-Width Window Fracdiff Method
  - Section 5.3: Stationarity with Maximum Memory Preservation

### Notebooks Reviewed

1. `Week05Labeling/02TripleBarrierMethods.ipynb`
   - Basic triple barrier setup
   - Visualization of barriers
   - Label interpretation

2. `Week05Labeling/04MetaLabeling.ipynb`
   - Meta-labeling for bet sizing
   - Primary + secondary model architecture
   - F1 score improvement

3. `Week05Labeling/03Compare_Labeling_Methods.ipynb`
   - MLFinLab vs custom implementation
   - Performance comparison
   - Best practices

4. `Week15BetSizing/03AveragingActiveBets.ipynb`
   - Sequential bootstrap
   - Sample weight calculation
   - Concurrent bet handling

## Troubleshooting

### Issue: Low validation accuracy

**Check:**
1. Label distribution balanced?
2. Alignment verification passing?
3. Overfitting gap reasonable (<20%)?

**Solutions:**
- Increase `min_ret` to filter noise
- Adjust `pt_sl` ratio (try [3, 1])
- Add uniqueness weighting

### Issue: NaN in training

**Check:**
1. Volatility has NaN values?
2. Events DataFrame has missing `t1`?

**Solutions:**
- Increase `lookback` period
- Check data quality (gaps, holidays)
- Filter events: `events = events.dropna()`

### Issue: Too few samples

**Check:**
1. `min_ret` threshold too high?
2. Volatility too low (stable period)?

**Solutions:**
- Lower `min_ret` to 0.003
- Increase `num_days` horizon to 10
- Add more tickers

## Summary

`fin_training_ldp.py` implements the **gold standard** López de Prado triple barrier methodology from academic literature. Key improvements:

✅ **MLFinLab validation**: Academically tested implementation  
✅ **Binary classification**: Simpler than 3-class  
✅ **Asymmetric barriers**: Realistic trading conditions  
✅ **Minimum return filter**: Reduces noise  
✅ **Built-in alignment checks**: Ensures correctness  

Next step: **Run training and add uniqueness weighting** to address the 211x redundancy problem discovered in concurrency analysis.
