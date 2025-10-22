# Actual Notebook Feature Recommendations vs Current Config

**Analysis Date:** October 21, 2025  
**Source:** `equity_feature_importance_analysis.ipynb` (most recent run)

## Notebook's TOP 20 Recommended Features (Orthogonal MDA)

Based on the notebook's **FINAL RECOMMENDATIONS** section:

1. ✅ `serial_corr_1` - Ortho MDA: 0.01881 (MDI rank: 41, MDA rank: 64)
2. ❌ `hl_volatility_ratio` - Ortho MDA: 0.01813 (MDI rank: 73, MDA rank: 56)
3. ✅ `macd_hist` - Ortho MDA: 0.01756 (MDI rank: 50, MDA rank: 11)
4. ✅ `macd_divergence` - Ortho MDA: 0.01756 (MDI rank: 47, MDA rank: 67)
5. ✅ `macd_signal` - Ortho MDA: 0.01713 (MDI rank: 39, MDA rank: 14)
6. ✅ `amihud_illiq` - Ortho MDA: 0.01703 (MDI rank: 92, MDA rank: 53)
7. ❌ `roll_spread` - Ortho MDA: 0.01648 (MDI rank: 49, MDA rank: 108)
8. ❌ `^TYX` - Ortho MDA: 0.01642 (MDI rank: 103, MDA rank: 83) **[MARKET]**
9. ✅ `macd` - Ortho MDA: 0.01580 (MDI rank: 20, MDA rank: 33)
10. ✅ `relative_volatility` - Ortho MDA: 0.01565 (MDI rank: 26, MDA rank: 101)
11. ✅ `skewness_20` - Ortho MDA: 0.01543 (MDI rank: 62, MDA rank: 43)
12. ✅ `sar` - Ortho MDA: 0.01494 (MDI rank: 44, MDA rank: 32)
13. ✅ `bb_upper` - Ortho MDA: 0.01494 (MDI rank: 48, MDA rank: 57)
14. ❌ `^FVX` - Ortho MDA: 0.01488 (MDI rank: 71, MDA rank: 63) **[MARKET]**
15. ❌ `hurst_exponent` - Ortho MDA: 0.01487 (MDI rank: 102, MDA rank: 55)
16. ✅ `vwap_20` - Ortho MDA: 0.01476 (MDI rank: 60, MDA rank: 41)
17. ✅ `ad_roc` - Ortho MDA: 0.01455 (MDI rank: 37, MDA rank: 77)
18. ✅ `bb_lower` - Ortho MDA: 0.01431 (MDI rank: 57, MDA rank: 2)
19. ❌ `return_entropy` - Ortho MDA: 0.01430 (MDI rank: NaN, MDA rank: 39)
20. ❌ `adx` - Ortho MDA: 0.01407 (MDI rank: 74, MDA rank: 75)

**Legend:**
- ✅ = In current `wavenet_optimized_v2` config
- ❌ = NOT in current config
- **[MARKET]** = External market feature (should be in MARKET_FEATURES config)

## Comparison with Current wavenet_optimized_v2

### Current Config (38 features):
```python
'feature_list': [
    # Consensus 3/3
    'kurtosis_60',
    # Consensus 2/3
    'macd_signal', 'dist_from_ma50', 'macd', 'macd_hist',
    'dollar_volume_ma_ratio', 'bb_lower', 'volume_zscore',
    'bb_position',
    # Consensus 1/3 but high average rank
    'volatility_ht_60', 'cci', 'dist_from_ma20', 'relative_volume',
    'volume_norm', 'sar', 'dist_from_ma10', 'serial_corr_1',
    'log_return_2d', 'skewness_20', 'vwap_20', 'bb_upper',
    'macd_divergence', 'log_return_3d', 'log_return_5d', 'rsi_7',
    'stoch_k_d_diff', 'hl_volatility_ratio', 'ad_roc',
    'relative_volatility', 'serial_corr_5', 'skewness_60',
    'bb_percent_b', 'variance_ratio', 'amihud_illiq',
    'log_return_10d', 'risk_adj_momentum_20', 'max_drawdown_20',
    'adx_plus'
]
```

### Overlap Analysis

**Features in BOTH (14 out of 20):**
1. ✅ `serial_corr_1` - #1 in notebook, in current config
2. ✅ `hl_volatility_ratio` - #2 in notebook, in current config
3. ✅ `macd_hist` - #3 in notebook, in current config
4. ✅ `macd_divergence` - #4 in notebook, in current config
5. ✅ `macd_signal` - #5 in notebook, in current config
6. ✅ `amihud_illiq` - #6 in notebook, in current config
7. ✅ `macd` - #9 in notebook, in current config
8. ✅ `relative_volatility` - #10 in notebook, in current config
9. ✅ `skewness_20` - #11 in notebook, in current config
10. ✅ `sar` - #12 in notebook, in current config
11. ✅ `bb_upper` - #13 in notebook, in current config
12. ✅ `vwap_20` - #16 in notebook, in current config
13. ✅ `ad_roc` - #17 in notebook, in current config
14. ✅ `bb_lower` - #18 in notebook, in current config

**Overlap Rate: 70% (14/20) ✅**

### Missing from Current Config (6 features):

1. ❌ `roll_spread` - #7 in notebook (Ortho MDA: 0.01648)
2. ❌ `^TYX` - #8 in notebook **[Should be in MARKET_FEATURES]**
3. ❌ `^FVX` - #14 in notebook **[Should be in MARKET_FEATURES]**
4. ❌ `hurst_exponent` - #15 in notebook (Ortho MDA: 0.01487)
5. ❌ `return_entropy` - #19 in notebook (Ortho MDA: 0.01430)
6. ❌ `adx` - #20 in notebook (Ortho MDA: 0.01407)

### In Current Config but NOT in Notebook's Top 20 (24 features):

**High Priority (should investigate):**
- `kurtosis_60` - In current config but not in top 20 (was "Consensus 3/3" before)
- `bb_position` - In current config (#8) but not in top 20
- `volume_zscore` - In current config (#8) but not in top 20
- `dollar_volume_ma_ratio` - In current config (#5) but not in top 20

**Medium Priority:**
- `volatility_ht_60`, `cci`, `dist_from_ma20`, `dist_from_ma50`, `dist_from_ma10`
- `relative_volume`, `volume_norm`
- `log_return_2d`, `log_return_3d`, `log_return_5d`, `log_return_10d`

**Lower Priority:**
- `rsi_7`, `stoch_k_d_diff`, `adx_plus`
- `serial_corr_5`, `skewness_60`, `bb_percent_b`
- `variance_ratio`, `risk_adj_momentum_20`, `max_drawdown_20`

## Key Findings

### ✅ GOOD NEWS: 70% Overlap!

The current `wavenet_optimized_v2` config has **14 out of 20 (70%)** overlap with the notebook's top recommendations. This is MUCH BETTER than the 13% I calculated earlier (which was based on a different analysis file).

### Why the Earlier Analysis Was Wrong

I was comparing against the wrong files:
- ❌ Used: `MDI_importance.csv`, `MDA_importance.csv`, `Ortho_MDA_importance_mapped.csv` 
- ✅ Should use: **The notebook's FINAL RECOMMENDATIONS output** (which uses Orthogonal MDA)

The notebook's recommendations are based on **Orthogonal MDA** (mapped back to original features), which is the **most robust metric** as it:
1. Removes multicollinearity via PCA
2. Uses out-of-sample validation (MDA)
3. Maps PC importance back to original features

### What This Means

**The current config is GOOD but could be improved:**

1. **Keep all 14 overlapping features** ✅ - These are validated by both analyses

2. **Consider adding 4 features:**
   - `roll_spread` (#7) - Microstructure feature
   - `hurst_exponent` (#15) - Complexity measure
   - `return_entropy` (#19) - Information theory
   - `adx` (#20) - Trend strength

3. **Market features handled separately:**
   - `^TYX` and `^FVX` should be in `MARKET_FEATURES` config, not feature_list
   - This is correct - they're external context, not stock-specific features

4. **Review these current features** (not in top 20):
   - `kurtosis_60` - Was highly ranked before, investigate why it dropped
   - `volume_zscore`, `bb_position` - Still useful or redundant?
   - Distance from MA features - May be captured by other features

## Recommendations

### Option 1: Minimal Update (RECOMMENDED)

**Add the 4 missing features to current config:**
```python
'feature_list': [
    # ... existing 38 features ...
    'roll_spread',      # #7 in notebook
    'hurst_exponent',   # #15 in notebook  
    'return_entropy',   # #19 in notebook
    'adx'              # #20 in notebook
]
# Total: 42 features
```

**Pros:**
- Low risk - keeps all proven features
- Adds high-performing features from new analysis
- 42 features is still manageable

**Cons:**
- Slightly higher dimensionality
- May have some redundancy

### Option 2: Replace with Top 20

**Use exactly the notebook's top 20:**
```python
'feature_list': [
    'serial_corr_1', 'hl_volatility_ratio', 'macd_hist',
    'macd_divergence', 'macd_signal', 'amihud_illiq',
    'roll_spread', 'macd', 'relative_volatility',
    'skewness_20', 'sar', 'bb_upper', 'vwap_20',
    'ad_roc', 'bb_lower', 'hurst_exponent',
    'return_entropy', 'adx'
]
# Total: 18 features (excluding 2 market features)
```

**Pros:**
- Based on most robust analysis (Orthogonal MDA)
- Cleaner, simpler feature set
- Less overfitting risk

**Cons:**
- Loses 24 current features
- Need to validate performance matches or improves

### Option 3: Top 30 Hybrid

**Combine top 20 from notebook + top 10 from current config:**
- Keep all 14 overlapping features
- Add 6 from notebook not in current
- Keep 10 best from current config not in notebook top 20

Total: ~30 features (balanced approach)

## Next Steps

1. ✅ **DONE:** Identified that current config has 70% overlap (not 13%)
2. **Decide:** Which option to pursue (recommend Option 1 - minimal update)
3. **Update:** `feature_config.py` with chosen features
4. **Validate:** Run training and compare performance
5. **Document:** Update selection rationale

---

**Conclusion:** The current `wavenet_optimized_v2` config is **well-aligned (70% overlap)** with the latest analysis. Small improvements can be made by adding 4 high-performing features.
