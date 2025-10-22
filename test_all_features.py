"""
Comprehensive Feature Validation - Test EVERY Feature
Validates each individual feature against known calculations or expected ranges
"""

import numpy as np
import pandas as pd
import talib
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

print("\n" + "╔"+"="*78+"╗")
print("║" + " "*15 + "COMPREHENSIVE FEATURE VALIDATION" + " "*30 + "║")
print("║" + " "*20 + "(Testing ALL Features)" + " "*36 + "║")
print("╚"+"="*78+"╝\n")

# Load test data
print("Loading test data (AAPL)...")
df = pd.read_parquet('data_raw/AAPL.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# FIX: Drop NaN values in OHLCV columns (TA-Lib cannot handle NaN)
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

# Use middle section after cleaning
df = df.iloc[200:400]  # 200 days

print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total samples: {len(df)}")

# Create features
config = FeatureConfig.get_preset('comprehensive')
feature_eng = EnhancedFinancialFeatures(feature_config=config)
print("\nGenerating features...")
features = feature_eng.create_all_features(df)

print(f"Total features created: {len(features.columns)}\n")

# Store test results
results = []

def validate_feature(name, feature_data, expected_min=None, expected_max=None, 
                     reference_data=None, check_type='range', tolerance=1e-5):
    """
    Validate a single feature
    
    Args:
        name: Feature name
        feature_data: Calculated feature values
        expected_min: Minimum expected value
        expected_max: Maximum expected value
        reference_data: Reference calculation to compare against
        check_type: 'range', 'match', or 'both'
        tolerance: Relative tolerance for matching
    """
    issues = []
    
    # Skip categorical features
    if pd.api.types.is_categorical_dtype(feature_data):
        return name, True, "Categorical (skipped)"
    
    # Check for NaN percentage
    nan_pct = feature_data.isna().sum() / len(feature_data) * 100
    if nan_pct > 50:
        issues.append(f"High NaN: {nan_pct:.1f}%")
    
    # Check for Inf
    if pd.api.types.is_numeric_dtype(feature_data):
        inf_count = np.isinf(feature_data.values).sum()
        if inf_count > 0:
            issues.append(f"Contains Inf: {inf_count}")
    
    # Range check
    if check_type in ['range', 'both'] and expected_min is not None and expected_max is not None:
        actual_min = feature_data.min()
        actual_max = feature_data.max()
        
        # Allow 10% tolerance on range
        if actual_min < expected_min * 1.1 or actual_max > expected_max * 1.1:
            issues.append(f"Range [{actual_min:.4f}, {actual_max:.4f}] outside expected [{expected_min}, {expected_max}]")
    
    # Match check against reference
    if check_type in ['match', 'both'] and reference_data is not None:
        valid_idx = ~(feature_data.isna() | reference_data.isna())
        if valid_idx.sum() > 0:
            if not np.allclose(feature_data[valid_idx], reference_data[valid_idx], rtol=tolerance, atol=1e-8):
                max_diff = np.abs(feature_data[valid_idx] - reference_data[valid_idx]).max()
                issues.append(f"Mismatch with reference (max diff: {max_diff:.2e})")
    
    # Check for extreme values
    if pd.api.types.is_numeric_dtype(feature_data):
        max_abs = abs(feature_data[np.isfinite(feature_data)]).max() if len(feature_data[np.isfinite(feature_data)]) > 0 else 0
        if max_abs > 1e6:
            issues.append(f"Extreme value: {max_abs:.2e}")
    
    passed = len(issues) == 0
    status = "✅ PASS" if passed else f"❌ {', '.join(issues)}"
    
    return name, passed, status


print("="*80)
print("VALIDATING EACH FEATURE")
print("="*80 + "\n")

# ============================================================================
# 1. RETURNS GROUP
# ============================================================================
print("1. RETURNS GROUP")
print("-" * 80)

for horizon in [1, 2, 3, 5, 10, 20]:
    feat_name = f'log_return_{horizon}d'
    if feat_name in features.columns:
        # Manual calculation
        manual = np.log(df['Close'] / df['Close'].shift(horizon))
        results.append(validate_feature(feat_name, features[feat_name], 
                                       reference_data=manual, check_type='match'))

# Return acceleration
if 'return_acceleration' in features.columns:
    manual = features['log_return_1d'].diff()
    results.append(validate_feature('return_acceleration', features['return_acceleration'],
                                   reference_data=manual, check_type='match'))

# ============================================================================
# 2. VOLATILITY GROUP
# ============================================================================
print("\n2. VOLATILITY GROUP")
print("-" * 80)

# Close-to-close volatility
for period in [20, 60]:
    feat_name = f'volatility_cc_{period}'
    if feat_name in features.columns:
        manual = df['Close'].pct_change().rolling(period).std()
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=0, expected_max=1.0,
                                       reference_data=manual, check_type='both'))

# Yang-Zhang volatility (range check only)
for period in [10, 20, 60]:
    feat_name = f'volatility_yz_{period}'
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=0, expected_max=1.0,
                                       check_type='range'))

# Parkinson volatility
for period in [10, 20]:
    feat_name = f'volatility_parkinson_{period}'
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=0, expected_max=1.0,
                                       check_type='range'))

# Garman-Klass volatility
if 'volatility_gk_20' in features.columns:
    results.append(validate_feature('volatility_gk_20', features['volatility_gk_20'],
                                   expected_min=0, expected_max=1.0,
                                   check_type='range'))

# Volatility ratios
if 'vol_ratio_short_long' in features.columns:
    results.append(validate_feature('vol_ratio_short_long', features['vol_ratio_short_long'],
                                   expected_min=0, expected_max=5.0,
                                   check_type='range'))

# Volatility of volatility
if 'vol_of_vol' in features.columns:
    results.append(validate_feature('vol_of_vol', features['vol_of_vol'],
                                   expected_min=0, expected_max=0.1,
                                   check_type='range'))

# Realized volatility components
for feat_name in ['realized_vol_positive', 'realized_vol_negative']:
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=0, expected_max=0.5,
                                       check_type='range'))

# ============================================================================
# 3. VOLUME GROUP
# ============================================================================
print("\n3. VOLUME GROUP")
print("-" * 80)

# Volume ROC
if 'volume_roc' in features.columns:
    manual = df['Volume'].pct_change(5)
    results.append(validate_feature('volume_roc', features['volume_roc'],
                                   reference_data=manual, check_type='match'))

# Dollar volume ratio
if 'dollar_volume_ma_ratio' in features.columns:
    dollar_vol = df['Volume'] * df['Close']
    manual = dollar_vol / (dollar_vol.rolling(20).mean() + 1e-8)
    results.append(validate_feature('dollar_volume_ma_ratio', features['dollar_volume_ma_ratio'],
                                   expected_min=0, expected_max=10.0,
                                   reference_data=manual, check_type='both'))

# Volume normalization
if 'volume_norm' in features.columns:
    manual = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    results.append(validate_feature('volume_norm', features['volume_norm'],
                                   expected_min=0, expected_max=10.0,
                                   reference_data=manual, check_type='both'))

# Volume z-score
if 'volume_zscore' in features.columns:
    manual = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    results.append(validate_feature('volume_zscore', features['volume_zscore'],
                                   expected_min=-5, expected_max=5,
                                   reference_data=manual, check_type='both'))

# VWAP
if 'vwap_20' in features.columns:
    manual = (df['Close'] * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
    results.append(validate_feature('vwap_20', features['vwap_20'],
                                   reference_data=manual, check_type='match'))

# Price/VWAP ratio
if 'price_vwap_ratio' in features.columns:
    results.append(validate_feature('price_vwap_ratio', features['price_vwap_ratio'],
                                   expected_min=0.5, expected_max=1.5,
                                   check_type='range'))

# OBV z-score
if 'obv_zscore' in features.columns:
    results.append(validate_feature('obv_zscore', features['obv_zscore'],
                                   expected_min=-5, expected_max=5,
                                   check_type='range'))

# OBV ROC
if 'obv_roc' in features.columns:
    results.append(validate_feature('obv_roc', features['obv_roc'],
                                   expected_min=-200, expected_max=200,
                                   check_type='range'))

# AD z-score
if 'ad_zscore' in features.columns:
    results.append(validate_feature('ad_zscore', features['ad_zscore'],
                                   expected_min=-5, expected_max=5,
                                   check_type='range'))

# AD ROC
if 'ad_roc' in features.columns:
    results.append(validate_feature('ad_roc', features['ad_roc'],
                                   expected_min=-1000, expected_max=1000,
                                   check_type='range'))

# CMF - CRITICAL TEST
if 'cmf' in features.columns:
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    mfv = mfm * df['Volume']
    manual = mfv.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
    results.append(validate_feature('cmf', features['cmf'],
                                   expected_min=-1.0, expected_max=1.0,
                                   reference_data=manual, check_type='both'))

# Relative volume
if 'relative_volume' in features.columns:
    manual = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    results.append(validate_feature('relative_volume', features['relative_volume'],
                                   expected_min=0, expected_max=10.0,
                                   reference_data=manual, check_type='both'))

# ============================================================================
# 4. MOMENTUM GROUP
# ============================================================================
print("\n4. MOMENTUM GROUP")
print("-" * 80)

# RSI
for period in [7, 14, 21]:
    feat_name = f'rsi_{period}'
    if feat_name in features.columns:
        ref = talib.RSI(df['Close'], period)
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=0, expected_max=100,
                                       reference_data=ref, check_type='both', tolerance=1e-3))

# MACD
for feat_name in ['macd', 'macd_signal', 'macd_hist', 'macd_divergence']:
    if feat_name in features.columns:
        if feat_name == 'macd':
            ref, _, _ = talib.MACD(df['Close'])
        elif feat_name == 'macd_signal':
            _, ref, _ = talib.MACD(df['Close'])
        elif feat_name in ['macd_hist', 'macd_divergence']:
            macd, signal, _ = talib.MACD(df['Close'])
            ref = macd - signal
        
        results.append(validate_feature(feat_name, features[feat_name],
                                       reference_data=ref, check_type='match', tolerance=1e-3))

# Stochastic
for feat_name in ['stoch_k', 'stoch_d']:
    if feat_name in features.columns:
        slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'])
        ref = slowk if feat_name == 'stoch_k' else slowd
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=0, expected_max=100,
                                       reference_data=ref, check_type='both', tolerance=1e-3))

if 'stoch_k_d_diff' in features.columns:
    results.append(validate_feature('stoch_k_d_diff', features['stoch_k_d_diff'],
                                   expected_min=-100, expected_max=100,
                                   check_type='range'))

# Williams %R
if 'williams_r' in features.columns:
    ref = talib.WILLR(df['High'], df['Low'], df['Close'])
    results.append(validate_feature('williams_r', features['williams_r'],
                                   expected_min=-100, expected_max=0,
                                   reference_data=ref, check_type='both', tolerance=1e-3))

# ATR
if 'atr' in features.columns:
    ref = talib.ATR(df['High'], df['Low'], df['Close'])
    results.append(validate_feature('atr', features['atr'],
                                   expected_min=0, expected_max=50,
                                   reference_data=ref, check_type='both', tolerance=1e-3))

if 'atr_ratio' in features.columns:
    results.append(validate_feature('atr_ratio', features['atr_ratio'],
                                   expected_min=0, expected_max=1.0,
                                   check_type='range'))

# CCI
if 'cci' in features.columns:
    ref = talib.CCI(df['High'], df['Low'], df['Close'])
    results.append(validate_feature('cci', features['cci'],
                                   expected_min=-500, expected_max=500,
                                   reference_data=ref, check_type='both', tolerance=1e-3))

# Return z-score
if 'return_zscore_20' in features.columns:
    returns = df['Close'].pct_change()
    manual = (returns - returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-8)
    results.append(validate_feature('return_zscore_20', features['return_zscore_20'],
                                   expected_min=-5, expected_max=5,
                                   reference_data=manual, check_type='both'))

# ROC
for period in [10, 20]:
    feat_name = f'roc_{period}'
    if feat_name in features.columns:
        ref = talib.ROC(df['Close'], period)
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=-50, expected_max=50,
                                       reference_data=ref, check_type='both', tolerance=1e-3))

# ============================================================================
# 5. TREND GROUP
# ============================================================================
print("\n5. TREND GROUP")
print("-" * 80)

# ADX
if 'adx' in features.columns:
    ref = talib.ADX(df['High'], df['Low'], df['Close'])
    results.append(validate_feature('adx', features['adx'],
                                   expected_min=0, expected_max=100,
                                   reference_data=ref, check_type='both', tolerance=1e-3))

# ADX Plus/Minus
if 'adx_plus' in features.columns:
    ref = talib.PLUS_DI(df['High'], df['Low'], df['Close'])
    results.append(validate_feature('adx_plus', features['adx_plus'],
                                   expected_min=0, expected_max=100,
                                   reference_data=ref, check_type='both', tolerance=1e-3))

if 'adx_minus' in features.columns:
    ref = talib.MINUS_DI(df['High'], df['Low'], df['Close'])
    results.append(validate_feature('adx_minus', features['adx_minus'],
                                   expected_min=0, expected_max=100,
                                   reference_data=ref, check_type='both', tolerance=1e-3))

# Parabolic SAR
if 'sar' in features.columns:
    ref = talib.SAR(df['High'], df['Low'])
    results.append(validate_feature('sar', features['sar'],
                                   expected_min=0, expected_max=df['High'].max() * 1.2,
                                   reference_data=ref, check_type='both', tolerance=1e-2))

if 'sar_signal' in features.columns:
    results.append(validate_feature('sar_signal', features['sar_signal'],
                                   expected_min=0, expected_max=1,
                                   check_type='range'))

# Aroon
for feat_name in ['aroon_up', 'aroon_down']:
    if feat_name in features.columns:
        down, up = talib.AROON(df['High'], df['Low'])
        ref = up if feat_name == 'aroon_up' else down
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=0, expected_max=100,
                                       reference_data=ref, check_type='both', tolerance=1e-3))

if 'aroon_oscillator' in features.columns:
    results.append(validate_feature('aroon_oscillator', features['aroon_oscillator'],
                                   expected_min=-100, expected_max=100,
                                   check_type='range'))

# Distance from MA
for period in [10, 20]:
    feat_name = f'dist_from_ma{period}'
    if feat_name in features.columns:
        ma = df['Close'].rolling(period).mean()
        manual = (df['Close'] - ma) / (ma + 1e-8)
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=-1.0, expected_max=1.0,
                                       reference_data=manual, check_type='both'))

# ============================================================================
# 6. BOLLINGER BANDS
# ============================================================================
print("\n6. BOLLINGER BANDS")
print("-" * 80)

# Bollinger Bands
bb_upper_ref, bb_middle_ref, bb_lower_ref = talib.BBANDS(df['Close'], timeperiod=20)

for feat_name in ['bb_upper', 'bb_lower']:
    if feat_name in features.columns:
        ref = bb_upper_ref if feat_name == 'bb_upper' else bb_lower_ref
        results.append(validate_feature(feat_name, features[feat_name],
                                       reference_data=ref, check_type='match', tolerance=1e-3))

if 'bb_position' in features.columns:
    manual = (df['Close'] - bb_lower_ref) / (bb_upper_ref - bb_lower_ref + 1e-8)
    results.append(validate_feature('bb_position', features['bb_position'],
                                   expected_min=-0.5, expected_max=1.5,
                                   reference_data=manual, check_type='both'))

if 'bb_width' in features.columns:
    manual = (bb_upper_ref - bb_lower_ref) / bb_middle_ref
    results.append(validate_feature('bb_width', features['bb_width'],
                                   expected_min=0, expected_max=1.0,
                                   reference_data=manual, check_type='both'))

if 'bb_percent_b' in features.columns:
    results.append(validate_feature('bb_percent_b', features['bb_percent_b'],
                                   expected_min=-0.5, expected_max=1.5,
                                   check_type='range'))

# ============================================================================
# 7. PRICE POSITION GROUP
# ============================================================================
print("\n7. PRICE POSITION GROUP")
print("-" * 80)

# Price position in daily range
if 'price_position' in features.columns:
    manual = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    results.append(validate_feature('price_position', features['price_position'],
                                   expected_min=0, expected_max=1.0,
                                   reference_data=manual, check_type='both'))

# Distance from longer MAs
for period in [50, 200]:
    feat_name = f'dist_from_ma{period}'
    if feat_name in features.columns:
        ma = df['Close'].rolling(period).mean()
        manual = (df['Close'] - ma) / (ma + 1e-8)
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=-1.0, expected_max=2.0,
                                       reference_data=manual, check_type='both'))

# Distance from highs/lows
if 'dist_from_20d_high' in features.columns:
    manual = df['Close'] / df['High'].rolling(20).max() - 1
    results.append(validate_feature('dist_from_20d_high', features['dist_from_20d_high'],
                                   expected_min=-1.0, expected_max=0.1,
                                   reference_data=manual, check_type='both'))

if 'dist_from_20d_low' in features.columns:
    manual = df['Close'] / df['Low'].rolling(20).min() - 1
    results.append(validate_feature('dist_from_20d_low', features['dist_from_20d_low'],
                                   expected_min=0, expected_max=1.0,
                                   reference_data=manual, check_type='both'))

# Serial correlation
for lag in [1, 5]:
    feat_name = f'serial_corr_{lag}'
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=-1.0, expected_max=1.0,
                                       check_type='range'))

# ============================================================================
# 8. MICROSTRUCTURE GROUP
# ============================================================================
print("\n8. MICROSTRUCTURE GROUP")
print("-" * 80)

# These are complex calculations, mainly check ranges
microstructure_features = [
    ('vpin', 0, 1.0),
    ('kyle_lambda', -0.01, 0.01),
    ('amihud_illiquidity', 0, 1e-6),
    ('hl_range', 0, 0.5),
    ('hl_range_ma', 0, 0.5),
    ('oc_range', -0.5, 0.5),
    ('roll_spread', 0, 0.5),
    ('cs_spread', 0, 1.0),
    ('hl_volatility_ratio', 0.5, 2.0),
    ('amihud_illiq', 0, 1e-6),
    ('order_flow_imbalance', -1.0, 1.0),
]

for feat_name, min_val, max_val in microstructure_features:
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=min_val, expected_max=max_val,
                                       check_type='range'))

# ============================================================================
# 9. ENTROPY/COMPLEXITY GROUP
# ============================================================================
print("\n9. ENTROPY/COMPLEXITY GROUP")
print("-" * 80)

entropy_features = [
    ('return_entropy', 0, 10),
    ('lz_complexity', 0, 5),
    ('variance_ratio', 0, 5),
    ('hurst_exponent', 0, 1.0),
]

for feat_name, min_val, max_val in entropy_features:
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=min_val, expected_max=max_val,
                                       check_type='range'))

# ============================================================================
# 10. REGIME GROUP
# ============================================================================
print("\n10. REGIME GROUP")
print("-" * 80)

regime_features = [
    ('relative_volatility', 0, 5),
    ('volume_percentile', 0, 1.0),
    ('trend_percentile', 0, 1.0),
    ('rvi', 0, 100),
    ('market_state', 0, 2),
    ('fractal_dimension', 1.0, 2.0),
]

for feat_name, min_val, max_val in regime_features:
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=min_val, expected_max=max_val,
                                       check_type='range'))

# ============================================================================
# 11. STATISTICAL GROUP
# ============================================================================
print("\n11. STATISTICAL GROUP")
print("-" * 80)

statistical_features = [
    ('skewness_20', -5, 5),
    ('skewness_60', -5, 5),
    ('kurtosis_20', -5, 20),
    ('kurtosis_60', -5, 20),
]

for feat_name, min_val, max_val in statistical_features:
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=min_val, expected_max=max_val,
                                       check_type='range'))

# ============================================================================
# 12. RISK-ADJUSTED GROUP
# ============================================================================
print("\n12. RISK-ADJUSTED GROUP")
print("-" * 80)

risk_features = [
    ('sharpe_20', -5, 5),
    ('sharpe_60', -5, 5),
    ('risk_adj_momentum_20', -5, 5),
    ('downside_vol_20', 0, 0.1),
    ('sortino_20', -5, 5),
    ('max_drawdown_20', -1.0, 0),
    ('calmar_20', -10, 10),
    ('hui_heubel', 0, 10),
]

for feat_name, min_val, max_val in risk_features:
    if feat_name in features.columns:
        results.append(validate_feature(feat_name, features[feat_name],
                                       expected_min=min_val, expected_max=max_val,
                                       check_type='range'))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80 + "\n")

passed = sum(1 for _, p, _ in results if p)
total = len(results)

# Group by status
passed_features = [(name, status) for name, p, status in results if p]
failed_features = [(name, status) for name, p, status in results if not p]

print(f"Total features tested: {total}")
print(f"Passed: {passed} ({passed/total*100:.1f}%)")
print(f"Failed: {len(failed_features)} ({len(failed_features)/total*100:.1f}%)")

if failed_features:
    print(f"\n{'='*80}")
    print("FAILED FEATURES:")
    print("="*80)
    for name, status in failed_features:
        print(f"  {name:<35} {status}")

print(f"\n{'='*80}")
if passed == total:
    print("🎉 ALL FEATURES VALIDATED SUCCESSFULLY!")
    print("   Every feature is accurately calculated and ready for training.")
else:
    print(f"⚠️  {len(failed_features)} feature(s) need attention")
    print("   Review failed features above for details.")
print("="*80 + "\n")

# Save results to file
results_df = pd.DataFrame(results, columns=['Feature', 'Passed', 'Status'])
results_df.to_csv('feature_validation_results.csv', index=False)
print("Detailed results saved to: feature_validation_results.csv\n")
