"""
Simplified Feature Accuracy Validation
Test key features against known calculations
"""

import numpy as np
import pandas as pd
import talib

print("\n" + "╔"+"="*78+"╗")
print("║" + " "*20 + "FEATURE ACCURACY VALIDATION" + " "*31 + "║")
print("╚"+"="*78+"╝\n")

# Load real data
print("Loading test data (AAPL)...")
df = pd.read_parquet('data_raw/AAPL.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# FIX: Drop NaN values in OHLCV columns (TA-Lib cannot handle NaN)
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

# Use middle section after cleaning
df = df.iloc[200:400]  # 200 days

print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total samples: {len(df)}\n")

# Import after loading data to avoid issues
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

config = FeatureConfig.get_preset('minimal')
feature_eng = EnhancedFinancialFeatures(feature_config=config)
features = feature_eng.create_all_features(df)

print(f"\n{'='*80}")
print("TEST 1: RETURNS")
print('='*80)

# Test log returns
manual_return = np.log(df['Close'] / df['Close'].shift(1))
calc_return = features['log_return_1d']

print(f"\nSample returns (days 50-55):")
print(f"Manual:     {manual_return.iloc[50:55].values}")
print(f"Calculated: {calc_return.iloc[50:55].values}")

match1 = np.allclose(manual_return.iloc[50:55].values, calc_return.iloc[50:55].values, rtol=1e-5)
print(f"Match: {'✅ PASS' if match1 else '❌ FAIL'}")

print(f"\n{'='*80}")
print("TEST 2: VOLATILITY (Close-to-Close)")
print('='*80)

# Test simple volatility
manual_vol = df['Close'].pct_change().rolling(20).std()
calc_vol = features['volatility_cc_20']

print(f"\nSample volatility (days 50-55):")
print(f"Manual:     {manual_vol.iloc[50:55].values}")
print(f"Calculated: {calc_vol.iloc[50:55].values}")

match2 = np.allclose(manual_vol.iloc[50:55].values, calc_vol.iloc[50:55].values, rtol=1e-5)
print(f"Match: {'✅ PASS' if match2 else '❌ FAIL'}")

print(f"\n{'='*80}")
print("TEST 3: VOLUME NORMALIZATION")
print('='*80)

# Test volume normalization
manual_vol_norm = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
calc_vol_norm = features['volume_norm']

print(f"\nSample volume norm (days 50-55):")
print(f"Manual:     {manual_vol_norm.iloc[50:55].values}")
print(f"Calculated: {calc_vol_norm.iloc[50:55].values}")

match3 = np.allclose(manual_vol_norm.iloc[50:55].values, calc_vol_norm.iloc[50:55].values, rtol=1e-5)
print(f"Match: {'✅ PASS' if match3 else '❌ FAIL'}")

print(f"\n{'='*80}")
print("TEST 4: CHAIKIN MONEY FLOW (CMF)")
print('='*80)

# Test CMF calculation
mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
mfv = mfm * df['Volume']
manual_cmf = mfv.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
calc_cmf = features['cmf']

print(f"\nSample CMF (days 50-55):")
print(f"Manual:     {manual_cmf.iloc[50:55].values}")
print(f"Calculated: {calc_cmf.iloc[50:55].values}")

match4 = np.allclose(manual_cmf.iloc[50:55].values, calc_cmf.iloc[50:55].values, rtol=1e-5)

# Check CMF range
cmf_in_range = (calc_cmf.min() >= -1.0) and (calc_cmf.max() <= 1.0)
print(f"\nCMF Range: [{calc_cmf.min():.4f}, {calc_cmf.max():.4f}]")
print(f"Expected: [-1, 1]")
print(f"In range: {'✅' if cmf_in_range else '❌'}")
print(f"Match: {'✅ PASS' if (match4 and cmf_in_range) else '❌ FAIL'}")

print(f"\n{'='*80}")
print("TEST 5: RSI (Relative Strength Index)")
print('='*80)

# Test RSI against talib
rsi_talib = talib.RSI(df['Close'], 14)
calc_rsi = features['rsi_14']

print(f"\nSample RSI (days 50-55):")
print(f"TAlib:      {rsi_talib.iloc[50:55].values}")
print(f"Calculated: {calc_rsi.iloc[50:55].values}")

match5 = np.allclose(rsi_talib.iloc[50:55].values, calc_rsi.iloc[50:55].values, rtol=1e-3)

# Check RSI range
rsi_in_range = (calc_rsi.min() >= 0) and (calc_rsi.max() <= 100)
print(f"\nRSI Range: [{calc_rsi.min():.2f}, {calc_rsi.max():.2f}]")
print(f"Expected: [0, 100]")
print(f"In range: {'✅' if rsi_in_range else '❌'}")
print(f"Match: {'✅ PASS' if (match5 and rsi_in_range) else '❌ FAIL'}")

print(f"\n{'='*80}")
print("TEST 6: MACD")
print('='*80)

# Test MACD
macd_talib, signal_talib, hist_talib = talib.MACD(df['Close'])
calc_macd = features['macd']
calc_signal = features['macd_signal']

print(f"\nSample MACD (days 50-55):")
print(f"TAlib:      {macd_talib.iloc[50:55].values}")
print(f"Calculated: {calc_macd.iloc[50:55].values}")

match6 = np.allclose(macd_talib.iloc[50:55].values, calc_macd.iloc[50:55].values, rtol=1e-3)
print(f"Match: {'✅ PASS' if match6 else '❌ FAIL'}")

print(f"\n{'='*80}")
print("TEST 7: OBV (On-Balance Volume)")
print('='*80)

# Test OBV against talib
obv_talib = talib.OBV(df['Close'], df['Volume'])

# Manual OBV calculation
obv_manual = [0]
for i in range(1, len(df)):
    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
        obv_manual.append(obv_manual[-1] + df['Volume'].iloc[i])
    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
        obv_manual.append(obv_manual[-1] - df['Volume'].iloc[i])
    else:
        obv_manual.append(obv_manual[-1])

obv_manual = pd.Series(obv_manual, index=df.index)

print(f"\nSample OBV raw (days 50-55):")
print(f"TAlib:  {obv_talib.iloc[50:55].values}")
print(f"Manual: {obv_manual.iloc[50:55].values}")

# OBV differs by constant offset (TAlib starts at first volume, manual at 0)
# Check if differences are consistent
obv_diff = obv_talib.values - obv_manual.values
offset_consistent = obv_diff.std() < 1.0  # Should be constant offset
match7 = offset_consistent

print(f"\nOBV offset (TAlib - Manual): {obv_diff[50]:.0f}")
print(f"Offset is constant: {'✅' if offset_consistent else '❌'}")

# Check OBV normalization in features
obv_zscore = features['obv_zscore']
obv_zscore_reasonable = abs(obv_zscore.dropna()).quantile(0.95) < 5

print(f"\nOBV Z-score 95th percentile: {abs(obv_zscore.dropna()).quantile(0.95):.2f}")
print(f"Reasonable (< 5): {'✅' if obv_zscore_reasonable else '❌'}")
print(f"Match: {'✅ PASS' if (match7 and obv_zscore_reasonable) else '❌ FAIL'}")

print(f"\n{'='*80}")
print("TEST 8: EXTREME VALUE CHECK")
print('='*80)

# Check for extreme values
extreme_features = []
for col in features.columns:
    if pd.api.types.is_numeric_dtype(features[col]):
        max_val = abs(features[col]).max()
        if max_val > 1e6:
            extreme_features.append((col, max_val))

if extreme_features:
    print(f"\n❌ Found {len(extreme_features)} features with extreme values (>1M):")
    for col, val in extreme_features[:5]:
        print(f"   {col}: {val:.2e}")
    match8 = False
else:
    print(f"\n✅ No extreme values found (all features < 1M)")
    match8 = True

print(f"Result: {'✅ PASS' if match8 else '❌ FAIL'}")

# Summary
print(f"\n{'='*80}")
print("VALIDATION SUMMARY")
print('='*80)

tests = [
    ("Returns", match1),
    ("Volatility (C-to-C)", match2),
    ("Volume Normalization", match3),
    ("CMF", match4 and cmf_in_range),
    ("RSI", match5 and rsi_in_range),
    ("MACD", match6),
    ("OBV", match7 and obv_zscore_reasonable),
    ("Extreme Values", match8),
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

print()
for test_name, result in tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{test_name:<25} {status}")

print(f"\n{'='*80}")
print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
print('='*80)

if passed == total:
    print("\n🎉 All validation tests passed!")
    print("   Features are accurately calculated and ready for model training.")
else:
    print(f"\n⚠️  {total - passed} test(s) failed - review issues above")

print()
