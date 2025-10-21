# Feature Migration Checklist

## Enhanced Script Features (100+ features expected)

### 1. RETURNS (10 features)
- [x] log_return_1d, 2d, 3d, 5d, 10d, 20d (6 features)
- [ ] **forward_return_1d, 3d, 5d** (3 features) - MISSING
- [x] return_acceleration (1 feature)

### 2. VOLATILITY (12 features)
- [x] volatility_yz_10, 20, 60 (3 features)
- [x] volatility_parkinson_10, 20 (2 features)
- [x] volatility_gk_20 (1 feature)
- [x] volatility_cc_20 (1 feature)
- [ ] **volatility_cc_60** (1 feature) - MISSING
- [x] vol_ratio_short_long (1 feature)
- [x] vol_of_vol (1 feature)
- [ ] **realized_vol_positive** (1 feature) - MISSING
- [ ] **realized_vol_negative** (1 feature) - MISSING

### 3. PRICE POSITION & LEVELS (8 features)
- [x] price_position (1 feature)
- [x] dist_from_ma10, 20, 50 (3 features)
- [ ] **dist_from_ma200** (1 feature) - MISSING
- [x] dist_from_20d_high (1 feature)
- [x] dist_from_20d_low (1 feature)

### 4. MOMENTUM INDICATORS (15 features)
- [x] rsi_7, 14, 21 (3 features)
- [x] stoch_k, stoch_d (2 features)
- [ ] **stoch_k_d_diff** (1 feature) - MISSING
- [x] williams_r (1 feature)
- [x] roc_10, 20 (2 features)
- [x] macd (1 feature) - as macd_signal in current
- [ ] **macd (separate)** (1 feature) - MISSING (we have macd_signal and macd_hist but not raw macd)
- [ ] **macd_divergence** (1 feature) - MISSING
- [x] cci (1 feature)

### 5. TREND INDICATORS (10 features)
- [x] adx (1 feature)
- [x] adx_plus, adx_minus (2 features)
- [x] sma_10_20_diff, sma_20_50_diff (2 features)
- [x] sar, sar_signal (2 features)
- [x] aroon_up, aroon_down, aroon_oscillator (3 features)

### 6. VOLUME FEATURES (12 features)
- [ ] **volume_norm** (1 feature) - MISSING (we have relative_volume which is similar)
- [ ] **volume_std** (1 feature) - MISSING
- [x] volume_roc (1 feature)
- [x] dollar_volume, dollar_volume_ma_ratio (2 features)
- [x] vwap_20 (1 feature) - as vwap
- [x] price_vwap_ratio (1 feature)
- [x] obv, obv_ma_20 (2 features)
- [x] ad_line (1 feature)
- [ ] **ad_normalized** (1 feature) - MISSING
- [x] cmf (1 feature)

### 7. VOLATILITY-ADJUSTED METRICS (8 features)
- [x] sharpe_20 (1 feature)
- [ ] **sharpe_60** (1 feature) - MISSING
- [x] risk_adj_momentum_20 (1 feature)
- [x] downside_vol_20, sortino_20 (2 features)
- [x] max_drawdown_20 (1 feature)
- [ ] **calmar_20** (1 feature) - MISSING
- [x] atr (1 feature)

### 8. MICROSTRUCTURE (10 features)
- [ ] **hl_range** (1 feature) - MISSING
- [ ] **hl_range_ma** (1 feature) - MISSING
- [ ] **oc_range** (1 feature) - MISSING
- [x] roll_spread (1 feature)
- [x] cs_spread (1 feature)
- [ ] **amihud_illiq** (1 feature) - MISSING (we have amihud_illiquidity but not the 20-day MA version)
- [x] kyle_lambda (1 feature)
- [x] order_flow_imbalance (1 feature)
- [x] vpin (1 feature)
- [x] hui_heubel (1 feature)

### 9. BOLLINGER BANDS (5 features)
- [ ] **bb_upper** (1 feature) - MISSING
- [ ] **bb_lower** (1 feature) - MISSING
- [x] bb_position (1 feature)
- [x] bb_width (1 feature)
- [ ] **bb_percent_b** (1 feature) - MISSING (duplicate of bb_position)

### 10. STATISTICAL PROPERTIES (8 features)
- [x] serial_corr_1 (1 feature)
- [x] serial_corr_5 (1 feature) - already exists
- [x] skewness_20 (1 feature)
- [ ] **skewness_60** (1 feature) - MISSING
- [x] kurtosis_20 (1 feature)
- [ ] **kurtosis_60** (1 feature) - MISSING
- [x] variance_ratio (1 feature)
- [x] hurst_60 (1 feature) - as hurst_exponent

### 11. ENTROPY & COMPLEXITY (4 features)
- [x] return_entropy_20 (1 feature) - as return_entropy
- [x] lz_complexity_20 (1 feature) - as lz_complexity
- [ ] **approx_entropy_20** (1 feature) - MISSING
- [x] fractal_dimension_20 (1 feature) - as fractal_dimension

### 12. REGIME INDICATORS (6 features)
- [x] vol_percentile (1 feature)
- [x] trend_percentile (1 feature)
- [x] volume_percentile (1 feature)
- [ ] **rvi (Relative Volatility Index)** (1 feature) - MISSING
- [x] return_zscore_20 (1 feature)
- [ ] **market_state** (1 feature) - MISSING

## Summary
- **Current**: ~88 features
- **Missing**: ~26 features
- **Total Expected**: ~114 features

## Helper Methods Status
- [x] _yang_zhang_volatility
- [x] _parkinson_volatility
- [x] _garman_klass_volatility
- [x] _roll_spread
- [x] _corwin_schultz_spread
- [x] _kyle_lambda
- [x] _order_flow_imbalance
- [x] _calculate_vpin
- [x] _hui_heubel_liquidity
- [x] _max_drawdown
- [x] _variance_ratio
- [x] _hurst_exponent
- [x] _shannon_entropy
- [x] _lempel_ziv_complexity
- [ ] **_approximate_entropy** - MISSING
- [x] _fractal_dimension
- [ ] **_relative_volatility_index** - MISSING
- [ ] **_classify_market_state** - MISSING
