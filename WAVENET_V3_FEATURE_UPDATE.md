# WaveNet Optimized V3 Feature Configuration Update

**Date:** October 21, 2025  
**Analysis:** systematic_feature_selection.py + equity_feature_importance_analysis.ipynb  
**Method:** Hybrid (Method 3) + Top 5 Quadrant 1 (Robust)

## Summary

Updated `feature_config.py` with new preset `wavenet_optimized_v3` based on systematic feature selection analysis using MDI, MDA, and Orthogonal MDA metrics.

### Key Changes

- **Reduced:** 42 features (v2) → 35 features (v3)
- **Rationale:** Optimal feature count for sample size ~500-1000 (sqrt rule)
- **Methodology:** Method 3 (Hybrid) - Top 50 Ortho MDA excluding Quadrant 4, plus Top 5 Quadrant 1
- **Improvements:**
  - ✅ Added `vpin` (now fixed and working, Quadrant 1 rank #5)
  - ✅ Removed dead weight (8 features with negative MDA)
  - ✅ Added market context (GC=F, ^GSPC, JPY=X)
  - ✅ Better category balance (10 categories represented)
  - ✅ Focus on unique information (Ortho MDA) + validation (MDA)

## Feature Selection Methodology

### Three Methods Evaluated

1. **Method 1 (Ortho Top-N):** Select top N by Orthogonal MDA
   - Maximizes unique predictive information
   - Removes correlation bias
   - 35 features selected

2. **Method 2 (Quadrant-Based):** Conservative approach
   - Start with Quadrant 1 (High MDI + High MDA = Robust)
   - Add from Quadrant 3 (Substituted) to reach target
   - 35 features selected

3. **Method 3 (Hybrid) - RECOMMENDED:**
   - Top 50 by Ortho MDA
   - Exclude Quadrant 4 (Low MDI + Low MDA)
   - Best balance of unique info + validation
   - 30 features → Add top 5 Quadrant 1 → **35 features**

### Four Quadrants Framework

```
                          MDA (Out-of-Sample)
                    Low                   High
            ┌─────────────────┬─────────────────────┐
            │   QUADRANT 2    │    QUADRANT 1       │
 MDI   High │   OVERFITTED    │     ROBUST          │
(Training)  │   18 features   │   30 features       │
            ├─────────────────┼─────────────────────┤
            │   QUADRANT 4    │    QUADRANT 3       │
       Low  │   UNIMPORTANT   │   SUBSTITUTED       │
            │   31 features   │   27 features       │
            └─────────────────┴─────────────────────┘
```

**Quadrant 1 (ROBUST):** High training + High validation → KEEP  
**Quadrant 2 (OVERFITTED):** High training + Low validation → INVESTIGATE  
**Quadrant 3 (SUBSTITUTED):** Low training + High validation → CONSIDER  
**Quadrant 4 (UNIMPORTANT):** Low training + Low validation → REMOVE

## V3 Feature List (35 Features)

### Volume (6 features)
- `volume_percentile` - Ortho: 0.0353, Top unique info
- `volume_zscore` - **Quadrant 1 rank #1** (MDI: 0.0104, MDA: 0.0132)
- `volume_norm` - **Quadrant 1 rank #3** (MDI: 0.0094, MDA: 0.0125)
- `volume_roc` - Momentum of volume changes
- `obv_normalized` - Cumulative volume indicator
- `ad_roc` - Accumulation/distribution momentum

### Momentum (5 features)
- `cci` - **Quadrant 1 rank #2** (MDI: 0.0102, MDA: 0.0132)
- `stoch_k_d_diff` - Stochastic divergence
- `stoch_d` - Stochastic signal line
- `macd_signal` - MACD signal crosses
- `roc_10` - 10-day rate of change

### Trend (3 features)
- `sar_signal` - Parabolic SAR signals
- `aroon_down` - Downtrend strength
- `dist_from_ma200` - Long-term trend position

### Volatility (2 features)
- `vol_of_vol` - Volatility instability
- `vol_ratio_short_long` - Vol regime changes

### Bollinger Bands (2 features)
- `bb_position` - **Quadrant 1 rank #4** (MDI: 0.0099, MDA: 0.0122)
- `bb_percent_b` - **Quadrant 1 rank #5** (MDI: 0.0094, MDA: 0.0113)

### Returns (3 features)
- `log_return_2d` - 2-day momentum
- `log_return_5d` - Weekly momentum
- `return_acceleration` - Momentum of momentum

### Statistical (3 features)
- `skewness_60` - Tail risk indicator
- `kurtosis_60` - Extreme event likelihood
- `serial_corr_5` - Short-term autocorrelation

### Microstructure (3 features)
- `vpin` - **NEW! Quadrant 1 rank #5** (MDI: 0.0107, MDA: 0.0119)
- `oc_range` - Open-close range
- `order_flow_imbalance` - Buy/sell pressure

### Market Context (3 features)
- `GC=F` - Gold futures (Ortho: 0.0148, MDA: 0.0109)
- `^GSPC` - S&P 500 (Ortho: 0.0268, MDA: 0.0075)
- `JPY=X` - USD/JPY (Ortho: 0.0133, MDA: 0.0043)

### Regime (1 feature)
- `trend_percentile` - Trend strength percentile

### Risk-Adjusted (2 features)
- `sharpe_60` - 60-day Sharpe ratio
- `sortino_20` - Downside risk-adjusted

### Other (2 features)
- `cmf` - Chaikin Money Flow
- `variance_ratio` - Random walk test

## Changes from V2 to V3

### Overlap
- **V2:** 44 features
- **V3:** 35 features  
- **Shared:** 15 features (34% overlap)

### Removed from V2 (29 features)

**Dead Weight (Negative MDA):**
- `log_return_1d` (MDA: -0.0041)
- `log_return_10d` (MDA: -0.0010)
- `adx` (replaced with aroon_down)
- `adx_plus` (negative MDA)

**Redundant/Substituted:**
- `serial_corr_1` (replaced with serial_corr_5)
- `skewness_20` (replaced with skewness_60)
- `macd`, `macd_hist`, `macd_divergence` (kept only macd_signal)
- `sar` (replaced with sar_signal)
- `bb_upper`, `bb_lower` (kept bb_position, bb_percent_b)
- `dist_from_ma10`, `dist_from_ma50` (replaced with dist_from_ma200)
- `hl_volatility_ratio`, `roll_spread` (replaced with oc_range)
- `relative_volume` (redundant with volume_norm, volume_zscore)
- `price_vwap_ratio`, `vwap_20` (volume features sufficient)
- `volatility_ht_60` (replaced with vol_of_vol, vol_ratio)
- `hurst_exponent`, `return_entropy` (replaced with variance_ratio)
- `max_drawdown_20`, `risk_adj_momentum_20` (replaced with sharpe_60, sortino_20)

### Added in V3 (20 features)

**Market Context:**
- `GC=F`, `^GSPC`, `JPY=X` - Broad market signals

**Unique Information (High Ortho MDA):**
- `volume_percentile` - Top Ortho MDA (0.0353)
- `cmf` - Chaikin Money Flow (Ortho: 0.0318)
- `vol_of_vol` - Volatility instability
- `aroon_down` - Better trend indicator than ADX
- `trend_percentile` - Regime classification

**Robust (Quadrant 1):**
- `vpin` - **NOW WORKING!** Fixed volume bucketing implementation

**Better Alternatives:**
- `sar_signal` vs `sar` - Signal crosses more actionable
- `roc_10` vs `roc_20` - Faster signal
- `dist_from_ma200` vs `dist_from_ma50` - Long-term trend
- `volume_roc`, `obv_normalized` - Additional volume dynamics

## Dead Weight Removed

Features with **negative MDA** (hurt out-of-sample performance):

1. `downside_vol_20` (MDA: -0.0065)
2. `volatility_cc_20` (MDA: -0.0047)
3. `log_return_1d` (MDA: -0.0041)
4. `ad_normalized` (MDA: -0.0039)
5. `amihud_illiquidity` (MDA: -0.0023)
6. `volatility_rs_60` (MDA: -0.0019)
7. `market_state` (MDA: -0.0014)
8. `williams_r` (MDA: -0.0003)

## VPIN Fix Verification

**Previous Status (V2):**
- `vpin` included in feature list but broken
- Bug: `bucket_size=50` shares → 1 billion buckets!
- Result: Constant value 1.0, blank correlation cells

**Fixed Implementation:**
- Changed to `n_buckets=50` total buckets
- Proper volume bucketing: `volume_per_bucket = total_volume / n_buckets`
- Result: Mean 0.228, Std 0.106, Range [0.064, 0.685]

**Current Status (V3):**
- ✅ **Quadrant 1 (Robust) rank #5**
- ✅ MDI: 0.0107, MDA: 0.0119 (positive predictive power!)
- ✅ Ortho MDA: 0.0183 (unique information)
- ✅ Now included and working properly

## Feature Category Balance

V3 provides better balance across categories:

| Category       | Count | % of Total |
|----------------|-------|------------|
| Volume         | 6     | 17%        |
| Momentum       | 5     | 14%        |
| Returns        | 3     | 9%         |
| Statistical    | 3     | 9%         |
| Microstructure | 3     | 9%         |
| Market         | 3     | 9%         |
| Trend          | 3     | 9%         |
| Volatility     | 2     | 6%         |
| Bollinger      | 2     | 6%         |
| Risk           | 2     | 6%         |
| Regime         | 1     | 3%         |
| Other          | 2     | 6%         |

## Usage

```python
from feature_config import FeatureConfig

# Get V3 configuration
config = FeatureConfig.get_preset('wavenet_optimized_v3')

# Use in training
features = config['feature_list']  # List of 35 feature names

# Print summary
FeatureConfig.print_config_summary(config)
```

## Expected Performance

- **Feature Count:** 35 (optimal for sample size ~500-1000)
- **Overfitting Risk:** Reduced vs V2 (fewer features, dead weight removed)
- **Unique Information:** Maximized via Ortho MDA selection
- **Validation:** All features have positive or near-positive MDA
- **Sample Efficiency:** sqrt(n) rule optimized

## Next Steps

1. ✅ **Configuration updated** in `feature_config.py`
2. Train model with `wavenet_optimized_v3` features
3. Compare performance to V2:
   - Per-barrier precision/recall
   - Overall accuracy
   - Overfitting metrics (train vs validation gap)
4. Monitor for:
   - Better generalization
   - Reduced overfitting
   - Improved per-barrier performance

## References

- **Analysis Script:** `systematic_feature_selection.py`
- **Methodology Doc:** `FEATURE_SELECTION_METHODOLOGY.md`
- **Notebook:** `equity_feature_importance_analysis.ipynb`
- **VPIN Fix:** Lines 650-695 in `fin_feature_preprocessing.py`
- **CS_SPREAD Fix:** Lines 732-754 in `fin_feature_preprocessing.py`

---

**Analysis Date:** October 21, 2025  
**Tickers:** AAPL, NVDA, TSLA, AMZN, AVGO  
**Samples:** 2,636  
**Method:** Hybrid (Method 3) + Top 5 Quadrant 1
