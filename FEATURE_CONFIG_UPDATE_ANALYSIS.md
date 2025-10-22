# Feature Configuration Update Analysis

## Executive Summary

**Date:** October 21, 2025

The latest feature importance analysis shows **MAJOR DIFFERENCES** from the current `wavenet_optimized_v2` configuration:

- **Current config:** 38 features
- **New consensus:** 29 features (excluding market features)
- **Overlap:** Only 5 features (13%)
- **Recommendation:** **⚠️ UPDATE REQUIRED**

## Analysis Details

### What Changed?

The notebook `equity_feature_importance_analysis.ipynb` was rerun with:
1. **All 6 market features** (`spy`, `vix`, `fvx`, `tyx`, `gold`, `jpyx`)
2. **5 tickers:** AAPL, NVDA, TSLA, AMZN, AVGO
3. **Comprehensive feature set:** All 109+ features analyzed

### Overlapping Features (Keep These - 5 features)

These appeared in BOTH the current config AND new consensus:

1. ✅ `ad_roc` - Accumulation/Distribution Rate of Change
2. ✅ `amihud_illiq` - Amihud Illiquidity metric
3. ✅ `max_drawdown_20` - 20-day maximum drawdown
4. ✅ `skewness_20` - 20-day return skewness
5. ✅ `variance_ratio` - Variance ratio test statistic

### Features to REMOVE (33 features)

Current features that did NOT make the new consensus top 40:

**Technical Indicators:**
- MACD family: `macd`, `macd_signal`, `macd_hist`, `macd_divergence` (ranked 95-112 in MDA)
- Bollinger Bands: `bb_lower`, `bb_upper`, `bb_position`, `bb_percent_b` (ranked 75-114 in MDA)
- `cci` (ranked 82 MDA, 102 MDI)
- `rsi_7` (ranked 41 MDA, 105 MDI)
- `sar` (ranked 60 MDA, 72 MDI)

**Returns:**
- `log_return_2d`, `log_return_3d`, `log_return_5d`, `log_return_10d` (mixed rankings)

**Volume:**
- `volume_zscore` (ranked 115 MDA!)
- `volume_norm` (ranked 47 MDA, 86 MDI)
- `relative_volume` (ranked 45 MDA, 107 MDI)
- `dollar_volume_ma_ratio` (ranked 8 MDA but 79 MDI)
- `vwap_20` (ranked 42 MDA, 40 MDI)

**Distance from MA:**
- `dist_from_ma10`, `dist_from_ma20`, `dist_from_ma50` (ranked 92-105)

**Volatility:**
- `volatility_ht_60` (ranked 24 MDA, 76 MDI)
- `hl_volatility_ratio` (ranked 96 MDA, 77 MDI)

**Others:**
- `serial_corr_1`, `serial_corr_5` (low MDA ranks)
- `kurtosis_60` (ranked 99 MDA despite 106 MDI)
- `relative_volatility`, `risk_adj_momentum_20`
- `stoch_k_d_diff`, `adx_plus`
- `skewness_60`

### Features to ADD (24 features)

New consensus features not in current config:

**Top Performers (Consensus 3/3):**
1. 🆕 `log_return_1d` - [3/3] avg_rank:19.0 (MDI:9, MDA:10, Ortho:38)
2. 🆕 `roll_spread` - [3/3] avg_rank:19.0 (MDI:33, MDA:11, Ortho:13)

**Strong Performers (Consensus 2/3):**
3. 🆕 `lz_complexity` - avg_rank:4.0 (MDI:3, Ortho:5) - **Complexity measure**
4. 🆕 `market_state` - avg_rank:8.0 (MDI:1!, Ortho:15) - **Regime indicator**
5. 🆕 `ad_normalized` - avg_rank:10.5 (MDA:4, Ortho:17)
6. 🆕 `obv_normalized` - avg_rank:10.5 (MDI:12, Ortho:9)
7. 🆕 `aroon_up` - avg_rank:15.0 (MDI:5, MDA:25)
8. 🆕 `aroon_oscillator` - avg_rank:17.0 (MDI:18, MDA:16)
9. 🆕 `amihud_illiquidity` - avg_rank:17.0 (MDI:7, MDA:27)
10. 🆕 `kyle_lambda` - avg_rank:17.5 (MDI:8, Ortho:27) - **Microstructure**
11. 🆕 `obv_roc` - avg_rank:18.0 (MDI:10, Ortho:26)
12. 🆕 `price_position` - avg_rank:18.0 (MDI:17, MDA:19)
13. 🆕 `hl_range_ma` - avg_rank:18.5 (MDI:35, MDA:2!)
14. 🆕 `return_acceleration` - avg_rank:21.5 (MDI:20, Ortho:23)
15. 🆕 `hui_heubel` - avg_rank:24.0 (MDI:15, MDA:33)
16. 🆕 `oc_range` - avg_rank:25.5 (MDI:14, Ortho:37)
17. 🆕 `adx` - avg_rank:26.5 (MDI:31, Ortho:22)
18. 🆕 `trend_percentile` - avg_rank:27.0 (MDA:40, Ortho:14)
19. 🆕 `order_flow_imbalance` - avg_rank:27.0 (MDI:36, MDA:18)
20. 🆕 `stoch_k` - avg_rank:27.5 (MDI:21, Ortho:34)
21. 🆕 `sharpe_20` - avg_rank:28.0 (MDI:28, MDA:28)
22. 🆕 `vol_ratio_short_long` - avg_rank:28.5 (MDA:38, Ortho:19)
23. 🆕 `adx_minus` - avg_rank:31.0 (MDI:30, MDA:32)
24. 🆕 `return_zscore_20` - avg_rank:35.5 (MDI:32, Ortho:39)

### Market Features (Excluded from Core List)

These ranked highly but are external market indicators (not stock-specific):
- `GC=F` (Gold futures) - avg_rank:18.0
- `^VIX` (VIX) - avg_rank:30.0
- `JPY=X` (JPY/USD) - avg_rank:29.5
- `^TYX` (30-Year Treasury) - avg_rank:26.5

**Note:** These should be added separately via the `MARKET_FEATURES` config in `fin_load_and_sequence.py`, not in the main feature list.

## Key Insights

### Why the Big Change?

1. **MACD Features Dropped:** All 4 MACD features (previously top performers) now rank 95-112 in MDA
   - Suggests they may have been overfit in previous analysis
   - Or multicollinearity with other momentum indicators

2. **Bollinger Bands Dropped:** All 4 BB features now rank 75-114
   - Similar to MACD - possible overfitting or redundancy

3. **Microstructure Features Rose:** `kyle_lambda`, `roll_spread`, `oc_range` now in top consensus
   - Suggests market microstructure is more predictive than previously thought

4. **Regime Indicators Important:** `market_state` ranks #1 in MDI!
   - Regime detection is critical for predictions

5. **Volume Metrics Mixed:** Some fell (volume_zscore #115!), others rose (obv_normalized #12 MDI)

### Validation Concerns

**The 13% overlap is concerning and suggests:**

1. ✅ **Different methodology:** Current config may have used different validation or ranking
2. ✅ **Data drift:** Market dynamics may have changed
3. ⚠️ **Overfitting risk:** Previous features may have been overfit
4. ⚠️ **Need for testing:** New features must be validated before deployment

## Recommendations

### Option 1: Conservative Update (RECOMMENDED)

**Keep the best of both worlds:**

1. **Keep all 5 overlapping features** (proven in both analyses)
2. **Add top 10 new consensus features** with highest avg_rank:
   - `log_return_1d`, `roll_spread`, `lz_complexity`, `market_state`
   - `ad_normalized`, `obv_normalized`, `aroon_up`, `aroon_oscillator`
   - `amihud_illiquidity`, `kyle_lambda`

3. **Keep top 10 from current config** that still rank reasonably:
   - `macd`, `macd_signal` (despite lower rank, still fundamental)
   - `bb_position`, `bb_lower` (price position context)
   - `vwap_20` (ranked 40 MDI)
   - `dist_from_ma20` (price context)
   - `log_return_3d`, `log_return_5d` (multi-horizon)
   - `volume_norm`, `relative_volume`

**Result:** ~25 features (balanced, lower overfitting risk)

### Option 2: Full Replacement

**Adopt the new consensus top 29 completely:**

1. Replace `wavenet_optimized_v2` feature_list with new consensus
2. Add market features separately via `MARKET_FEATURES` config
3. Retrain and validate performance

**Risk:** May lose some proven signal from current features

### Option 3: Expand to Top 40

**Include both overlapping and unique features:**

1. Keep all 38 current features
2. Add the 24 new consensus features
3. Total: 57 features

**Risk:** Higher dimensionality, possible overfitting

## Next Steps

1. **Review the changes** with domain expertise
2. **Run ablation study:** Test current vs new vs hybrid configs
3. **Validate on out-of-sample data** (2025 data)
4. **Check PBO score** for overfitting with new config
5. **Update `feature_config.py`** with final decision

## Files to Update

If proceeding with update:

1. `feature_config.py` - Update `wavenet_optimized_v2` preset
2. `fin_training_ldp.py` - Verify config selection
3. Documentation - Update feature selection rationale

---

**Generated:** October 21, 2025  
**Analysis Script:** `compare_feature_rankings.py`  
**Source Data:** `artifacts/feature_importance/*.csv`
