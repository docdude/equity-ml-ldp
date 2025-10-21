# WaveNet Optimized V2 - Usage Guide

## Quick Start

```python
from feature_config import FeatureConfig

# Get the preset
config = FeatureConfig.get_preset('wavenet_optimized_v2')

# Use in your training pipeline
# This will create 38 stock features + market features (if MARKET_FEATURES configured)
```

## What Changed

### Fixed Issues
- ❌ **Removed:** `GC=F`, `^FVX` (market features - add via `MARKET_FEATURES` config)
- ✅ **Kept:** All 38 engineered stock features that exist in the codebase

### Feature Count
- **Stock features:** 38 (from feature_list)
- **Market features:** 0-6 (configure separately via `MARKET_FEATURES`)
- **Total:** 38 + market features

### Market Features Recommendation

Based on the analysis:

**High priority (add to `MARKET_FEATURES`):**
```python
MARKET_FEATURES = ['fvx', 'tyx']  # Treasury yields (unique macro signal)
# Or just:
MARKET_FEATURES = ['fvx']  # 5-year Treasury (rank 14 orthogonal)
```

**Optional:**
```python
MARKET_FEATURES = ['gold']  # GC=F (rank 13/21, consistent performer)
```

**Skip (redundant):**
- `'spy'` (^GSPC) - Rank 102 orthogonal, redundant with stock features
- `'vix'` (^VIX) - Rank 110 orthogonal, redundant with volatility features
- `'jpyx'` - Rank 28 orthogonal, marginal importance

## Configuration Example

```python
# In your training config
CONFIG = {
    'feature_preset': 'wavenet_optimized_v2',  # 38 stock features
    'MARKET_FEATURES': ['fvx', 'tyx'],         # Add 2 macro features
    # Total: 40 features (38 stock + 2 market)
}
```

## 38 Stock Features Breakdown

### Momentum (7)
- macd, macd_signal, macd_hist, macd_divergence
- cci, rsi_7, stoch_k_d_diff

### Bollinger Bands (4)
- bb_lower, bb_upper, bb_position, bb_percent_b

### Volume (5)
- volume_zscore, volume_norm, dollar_volume_ma_ratio
- relative_volume, ad_roc

### Price Position (5)
- dist_from_ma10, dist_from_ma20, dist_from_ma50
- serial_corr_1, serial_corr_5

### Returns (4)
- log_return_2d, log_return_3d, log_return_5d, log_return_10d

### Statistical (3)
- kurtosis_60, skewness_20, skewness_60

### Volatility (3)
- volatility_ht_60, hl_volatility_ratio, relative_volatility

### Risk-Adjusted (2)
- risk_adj_momentum_20, max_drawdown_20

### Microstructure (1)
- amihud_illiq

### Trend (2)
- sar, adx_plus

### Other (2)
- vwap_20, variance_ratio

## Why 38 Instead of 40?

The original analysis included 2 market features:
- **GC=F** (Gold futures) - Rank 13 original MDA, 21 orthogonal
- **^FVX** (5-year Treasury) - Rank 63 → 14 orthogonal (hidden gem!)

These are **added separately** via the `MARKET_FEATURES` config parameter, not through feature engineering.

## Performance Expectations

Based on RandomForest baseline:
- **OOB Score:** 0.4613
- **OOS Score:** 0.4344 ± 0.0394
- **Overfitting Gap:** 0.027 (low)

These 38 features were selected by consensus across:
1. MDI (in-sample importance)
2. MDA original (OOS with multicollinearity)
3. MDA orthogonal (OOS after removing correlation)

## Troubleshooting

If you see:
```
⚠️  WARNING: X features not found: ['GC=F', '^FVX']
```

This is **EXPECTED** - market features are added separately. To include them:

```python
# In your data loading code
MARKET_FEATURES = ['gold', 'fvx']  # Internal keys
# These map to: GC=F and ^FVX
```

## Next Steps

1. ✅ Use `wavenet_optimized_v2` for 38 consensus stock features
2. ✅ Add `MARKET_FEATURES = ['fvx']` or `['fvx', 'tyx']` for macro signals
3. ✅ Train WaveNet and compare to `wavenet_optimized` (18 features)
4. ✅ Run feature ablation: test removing bottom 10 to see if 28 features perform better
