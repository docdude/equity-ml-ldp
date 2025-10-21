# WaveNet Optimized V2 Feature Set

**Generated:** October 2025  
**Analysis Source:** `equity_feature_importance_analysis.ipynb`  
**Config Preset:** `wavenet_optimized_v2` in `feature_config.py`

## Summary

This feature set contains **40 features** selected through consensus ranking across three importance methods:
1. **MDI** (Mean Decrease Impurity) - In-sample splits
2. **MDA Original** (Mean Decrease Accuracy) - Out-of-sample performance with multicollinearity
3. **MDA Orthogonal** (Mean Decrease Accuracy on PCs) - Out-of-sample after removing correlation

## Model Performance (RandomForest baseline)

- **OOB Score:** 0.4613
- **OOS Score:** 0.4344 ± 0.0394 (5-fold PurgedKFold CV)
- **Overfitting Gap:** 0.027 (low)
- **Sample Weights:** López de Prado uniqueness weights
- **Tickers:** AAPL, NVDA, TSLA, AMZN, AVGO (2,636 samples)

## Top 40 Features (Ranked by Consensus)

### Tier 1: Consensus 3/3 (Appears in all three methods)
1. **kurtosis_60** - Statistical tail risk measure

### Tier 2: Consensus 2/3 (Appears in two methods)
2. **macd_signal** - MACD signal line
3. **dist_from_ma50** - Distance from 50-day moving average
4. **macd** - MACD indicator
5. **macd_hist** - MACD histogram
6. **dollar_volume_ma_ratio** - Dollar volume relative to moving average
7. **bb_lower** - Bollinger Band lower bound
8. **volume_zscore** - Volume z-score
9. **bb_position** - Price position within Bollinger Bands
10. **GC=F** - Gold futures (market feature)

### Tier 3: Consensus 1/3 (Top performers in at least one method)
11. **volatility_ht_60** - High-low volatility (60-day)
12. **cci** - Commodity Channel Index
13. **dist_from_ma20** - Distance from 20-day MA
14. **relative_volume** - Volume relative to average
15. **volume_norm** - Normalized volume
16. **sar** - Parabolic SAR
17. **dist_from_ma10** - Distance from 10-day MA
18. **serial_corr_1** - 1-day serial correlation (hidden gem from orthogonal!)
19. **log_return_2d** - 2-day log return
20. **skewness_20** - 20-day skewness
21. **vwap_20** - 20-day VWAP
22. **bb_upper** - Bollinger Band upper bound
23. **macd_divergence** - MACD divergence indicator
24. **log_return_3d** - 3-day log return
25. **log_return_5d** - 5-day log return
26. **rsi_7** - 7-day RSI
27. **stoch_k_d_diff** - Stochastic K-D difference
28. **hl_volatility_ratio** - High-low volatility ratio (hidden gem!)
29. **ad_roc** - Accumulation/Distribution rate of change
30. **relative_volatility** - Relative volatility measure
31. **serial_corr_5** - 5-day serial correlation
32. **skewness_60** - 60-day skewness
33. **bb_percent_b** - Bollinger %B indicator
34. **^FVX** - 5-year Treasury yield (market feature)
35. **variance_ratio** - Variance ratio test
36. **amihud_illiq** - Amihud illiquidity measure
37. **log_return_10d** - 10-day log return
38. **risk_adj_momentum_20** - Risk-adjusted momentum
39. **max_drawdown_20** - 20-day maximum drawdown
40. **adx_plus** - ADX positive directional indicator

## Key Insights

### Market Features (2 included, 4 excluded)

**Included:**
- **GC=F (Gold)**: Rank 13 original MDA, rank 21 orthogonal - consistent performer
- **^FVX (5-year Treasury)**: Rank 63 → 14 orthogonal - **hidden gem!**

**Excluded (Redundant):**
- **^GSPC (S&P 500)**: Rank 102 orthogonal - redundant with stock features
- **^VIX**: Rank 110 orthogonal - redundant with volatility features  
- **^TYX (30-year Treasury)**: Rank 8 orthogonal but not in top 40 consensus
- **JPY=X (Yen)**: Rank 28 orthogonal - marginal importance

**Rationale:** Treasury yields (^FVX, ^TYX) capture unique macro risk/liquidity conditions that individual stock features can't. However, only ^FVX made the consensus top 40.

### Hidden Gems (Low original MDA, high orthogonal MDA)

These features were masked by multicollinearity in the original analysis:

- **serial_corr_1**: Rank 64 original → Rank 1 orthogonal (0.0188 importance)
- **hl_volatility_ratio**: Rank 56 → Rank 2 orthogonal (0.0181)
- **macd_divergence**: Rank 67 → Rank 4 orthogonal (0.0176)
- **^FVX (Treasury)**: Rank 63 → Rank 14 orthogonal (0.0149)

### Feature Categories

- **Momentum (7)**: macd, macd_signal, macd_hist, macd_divergence, cci, rsi_7, stoch_k_d_diff
- **Volatility (3)**: volatility_ht_60, hl_volatility_ratio, relative_volatility
- **Bollinger Bands (4)**: bb_lower, bb_upper, bb_position, bb_percent_b
- **Volume (5)**: volume_zscore, volume_norm, dollar_volume_ma_ratio, relative_volume, ad_roc
- **Price Position (5)**: dist_from_ma10, dist_from_ma20, dist_from_ma50, serial_corr_1, serial_corr_5
- **Returns (4)**: log_return_2d, log_return_3d, log_return_5d, log_return_10d
- **Statistical (4)**: kurtosis_60, skewness_20, skewness_60, variance_ratio
- **Microstructure (2)**: hl_volatility_ratio, amihud_illiq
- **Risk-Adjusted (2)**: risk_adj_momentum_20, max_drawdown_20
- **Trend (2)**: sar, adx_plus
- **Other (2)**: vwap_20, GC=F, ^FVX

## Why Original vs Orthogonal Results Differ

This is **EXPECTED and CORRECT**:

### Original MDA
- Shows importance WITH multicollinearity effects
- Correlated features dilute each other's importance
- Captures "which features help when used WITH others"

### Orthogonal MDA
- Shows importance AFTER removing multicollinearity via PCA
- Reveals true independent predictors
- Captures "which features contain unique information"

### Example: serial_corr_1
- **Original**: Rank 64 (hidden by correlation with serial_corr_5)
- **Orthogonal**: Rank 1 (emerges as most important after removing correlation)
- Both serial_corr features are correlated → split credit in original
- After PCA: serial_corr_1's unique signal revealed

## Usage

```python
from feature_config import FeatureConfig

# Load the preset
config = FeatureConfig.get_preset('wavenet_optimized_v2')

# Use in feature engineering
features_df = feature_engineer.create_all_features(df, feature_config=config)

# Access the explicit feature list
feature_list = config['feature_list']  # 40 features
```

## Comparison to wavenet_optimized (v1)

| Metric | v1 (Original) | v2 (This) |
|--------|---------------|-----------|
| Features | 18 | 40 |
| Selection Method | Single analysis | Consensus of 3 methods |
| Market Features | 0 | 2 (GC=F, ^FVX) |
| Orthogonal Analysis | Basic | Full PCA + mapping |
| Hidden Gems | Not identified | 4+ features revealed |

## Next Steps

1. **Test on WaveNet**: Use `wavenet_optimized_v2` preset in training pipeline
2. **Compare Performance**: Benchmark against `wavenet_optimized` (18 features) and `comprehensive` (115 features)
3. **Feature Ablation**: Test removing bottom 10 features to see if 30 features perform better
4. **Market Context**: Consider adding ^TYX (rank 8 orthogonal) for enhanced macro signal

## Files

- **Config:** `feature_config.py` → `FeatureConfig.get_preset('wavenet_optimized_v2')`
- **Analysis:** `equity_feature_importance_analysis.ipynb`
- **Results:** `artifacts/feature_importance/*.csv`
- **Feature List:** `wavenet_optimized_v2_features.txt`
