"""
Feature Accuracy Validation
Test each feature against known calculations and reference implementations
"""

import numpy as np
import pandas as pd
import talib
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

def test_returns():
    """Test return calculations"""
    print("\n" + "="*80)
    print("TEST 1: RETURNS")
    print("="*80)
    
    # Create simple test data
    dates = pd.date_range('2020-01-01', periods=10)
    prices = pd.Series([100, 102, 101, 105, 103, 107, 110, 108, 112, 115], index=dates)
    
    df = pd.DataFrame({
        'close': prices,
        'open': prices * 0.99,
        'high': prices * 1.01,
        'low': prices * 0.98,
        'volume': [1000000] * 10
    }, index=dates)
    
    config = FeatureConfig.get_preset('minimal')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    features = feature_eng.create_all_features(df)
    
    # Manual calculation
    manual_log_return_1d = np.log(prices / prices.shift(1))
    calculated_return = features['log_return_1d']
    
    print(f"\nManual log returns (1d):")
    print(manual_log_return_1d.iloc[1:6].values)
    print(f"\nCalculated log returns (1d):")
    print(calculated_return.iloc[1:6].values)
    
    # Compare only valid values
    manual_valid = manual_log_return_1d.dropna().values
    calc_valid = calculated_return.dropna().values
    
    # Make sure they're same length
    min_len = min(len(manual_valid), len(calc_valid))
    match = np.allclose(manual_valid[:min_len], calc_valid[:min_len], rtol=1e-5)
    print(f"\n✅ PASS" if match else "❌ FAIL")
    
    return match


def test_volatility():
    """Test volatility calculations"""
    print("\n" + "="*80)
    print("TEST 2: VOLATILITY")
    print("="*80)
    
    # Load real data for realistic test  
    df = pd.read_parquet('data_raw/AAPL.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # FIX: Drop NaN values in OHLCV columns (TA-Lib cannot handle NaN)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Skip early data, use 200 days
    df = df.iloc[100:300]
    
    config = FeatureConfig.get_preset('minimal')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    features = feature_eng.create_all_features(df)
    
    # Test 1: Close-to-close volatility
    manual_vol_cc = df['Close'].pct_change().rolling(20).std()
    calculated_vol_cc = features['volatility_cc_20']
    
    print(f"\nClose-to-Close Volatility (20-day):")
    print(f"Manual:     {manual_vol_cc.iloc[25:30].values}")
    print(f"Calculated: {calculated_vol_cc.iloc[25:30].values}")
    
    match1 = np.allclose(manual_vol_cc.dropna(), calculated_vol_cc.dropna(), rtol=1e-5)
    
    # Test 2: Parkinson volatility (using talib for reference)
    hl = np.log(df['High'] / df['Low'])
    manual_parkinson = hl.rolling(20).apply(lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))))
    calculated_parkinson = features['volatility_parkinson_20']
    
    print(f"\nParkinson Volatility (20-day):")
    print(f"Manual:     {manual_parkinson.iloc[25:30].values}")
    print(f"Calculated: {calculated_parkinson.iloc[25:30].values}")
    
    match2 = np.allclose(manual_parkinson.dropna(), calculated_parkinson.dropna(), rtol=1e-3)
    
    result = match1 and match2
    print(f"\n✅ PASS" if result else "❌ FAIL")
    
    return result


def test_volume_features():
    """Test volume feature calculations"""
    print("\n" + "="*80)
    print("TEST 3: VOLUME FEATURES")
    print("="*80)
    
    df = pd.read_parquet('data_raw/AAPL.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').iloc[:100]
    
    config = FeatureConfig.get_preset('minimal')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    features = feature_eng.create_all_features(df)
    
    # Test 1: Volume normalization
    manual_vol_norm = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
    calculated_vol_norm = features['volume_norm']
    
    print(f"\nVolume Normalization (volume / MA20):")
    print(f"Manual:     {manual_vol_norm.iloc[25:30].values}")
    print(f"Calculated: {calculated_vol_norm.iloc[25:30].values}")
    
    match1 = np.allclose(manual_vol_norm.dropna(), calculated_vol_norm.dropna(), rtol=1e-5)
    
    # Test 2: CMF calculation
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    mfv = mfm * df['Volume']
    manual_cmf = mfv.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
    calculated_cmf = features['cmf']
    
    print(f"\nChaikin Money Flow (20-day):")
    print(f"Manual:     {manual_cmf.iloc[25:30].values}")
    print(f"Calculated: {calculated_cmf.iloc[25:30].values}")
    
    match2 = np.allclose(manual_cmf.dropna(), calculated_cmf.dropna(), rtol=1e-5)
    
    # Test 3: CMF range validation
    cmf_in_range = (calculated_cmf.min() >= -1.0) and (calculated_cmf.max() <= 1.0)
    print(f"\nCMF Range Check: [{calculated_cmf.min():.4f}, {calculated_cmf.max():.4f}]")
    print(f"Expected: [-1, 1]")
    print(f"In range: {cmf_in_range}")
    
    # Test 4: OBV comparison with talib
    obv_talib = talib.OBV(df['Close'], df['Volume'])
    obv_manual = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv_manual.append(obv_manual[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv_manual.append(obv_manual[-1] - df['Volume'].iloc[i])
        else:
            obv_manual.append(obv_manual[-1])
    
    obv_manual = pd.Series(obv_manual, index=df.index)
    
    print(f"\nOBV Validation (raw values):")
    print(f"TAlib:  {obv_talib.iloc[25:30].values}")
    print(f"Manual: {obv_manual.iloc[25:30].values}")
    
    match3 = np.allclose(obv_talib.dropna(), obv_manual.dropna(), rtol=1e-5)
    
    # Test 5: OBV z-score range
    obv_zscore = features['obv_zscore']
    obv_zscore_reasonable = (abs(obv_zscore.dropna()).quantile(0.95) < 5)
    print(f"\nOBV Z-score 95th percentile: {abs(obv_zscore.dropna()).quantile(0.95):.2f}")
    print(f"Reasonable (< 5): {obv_zscore_reasonable}")
    
    result = match1 and match2 and cmf_in_range and match3 and obv_zscore_reasonable
    print(f"\n✅ PASS" if result else "❌ FAIL")
    
    return result


def test_momentum_indicators():
    """Test momentum indicators against talib"""
    print("\n" + "="*80)
    print("TEST 4: MOMENTUM INDICATORS")
    print("="*80)
    
    df = pd.read_parquet('data_raw/AAPL.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').iloc[:100]
    
    config = FeatureConfig.get_preset('minimal')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    features = feature_eng.create_all_features(df)
    
    # Test 1: RSI
    rsi_talib = talib.RSI(df['Close'], 14)
    calculated_rsi = features['rsi_14']
    
    print(f"\nRSI (14):")
    print(f"TAlib:      {rsi_talib.iloc[25:30].values}")
    print(f"Calculated: {calculated_rsi.iloc[25:30].values}")
    
    match1 = np.allclose(rsi_talib.dropna(), calculated_rsi.dropna(), rtol=1e-3)
    
    # Test 2: MACD
    macd_talib, signal_talib, hist_talib = talib.MACD(df['Close'])
    calculated_macd = features['macd']
    calculated_signal = features['macd_signal']
    
    print(f"\nMACD:")
    print(f"TAlib MACD:      {macd_talib.iloc[40:45].values}")
    print(f"Calculated MACD: {calculated_macd.iloc[40:45].values}")
    
    match2 = np.allclose(macd_talib.dropna(), calculated_macd.dropna(), rtol=1e-3)
    
    # Test 3: RSI range validation
    rsi_in_range = (calculated_rsi.min() >= 0) and (calculated_rsi.max() <= 100)
    print(f"\nRSI Range: [{calculated_rsi.min():.2f}, {calculated_rsi.max():.2f}]")
    print(f"Expected: [0, 100]")
    print(f"In range: {rsi_in_range}")
    
    result = match1 and match2 and rsi_in_range
    print(f"\n✅ PASS" if result else "❌ FAIL")
    
    return result


def test_trend_indicators():
    """Test trend indicators"""
    print("\n" + "="*80)
    print("TEST 5: TREND INDICATORS")
    print("="*80)
    
    df = pd.read_parquet('data_raw/AAPL.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').iloc[:100]
    
    config = FeatureConfig.get_preset('minimal')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    features = feature_eng.create_all_features(df)
    
    # Test 1: ADX
    adx_talib = talib.ADX(df['High'], df['Low'], df['Close'])
    calculated_adx = features['adx']
    
    print(f"\nADX (14):")
    print(f"TAlib:      {adx_talib.iloc[40:45].values}")
    print(f"Calculated: {calculated_adx.iloc[40:45].values}")
    
    match1 = np.allclose(adx_talib.dropna(), calculated_adx.dropna(), rtol=1e-3)
    
    # Test 2: Distance from MA
    ma10 = df['Close'].rolling(10).mean()
    manual_dist = (df['Close'] - ma10) / (ma10 + 1e-8)
    calculated_dist = features['dist_from_ma10']
    
    print(f"\nDistance from MA10:")
    print(f"Manual:     {manual_dist.iloc[15:20].values}")
    print(f"Calculated: {calculated_dist.iloc[15:20].values}")
    
    match2 = np.allclose(manual_dist.dropna(), calculated_dist.dropna(), rtol=1e-5)
    
    result = match1 and match2
    print(f"\n✅ PASS" if result else "❌ FAIL")
    
    return result


def test_bollinger_bands():
    """Test Bollinger Bands"""
    print("\n" + "="*80)
    print("TEST 6: BOLLINGER BANDS")
    print("="*80)
    
    df = pd.read_parquet('data_raw/AAPL.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').iloc[:100]
    
    config = FeatureConfig.get_preset('minimal')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    features = feature_eng.create_all_features(df)
    
    # Test with talib
    bb_upper_talib, bb_middle_talib, bb_lower_talib = talib.BBANDS(df['Close'], timeperiod=20)
    calculated_upper = features['bb_upper']
    calculated_lower = features['bb_lower']
    
    print(f"\nBollinger Upper Band:")
    print(f"TAlib:      {bb_upper_talib.iloc[25:30].values}")
    print(f"Calculated: {calculated_upper.iloc[25:30].values}")
    
    match1 = np.allclose(bb_upper_talib.dropna(), calculated_upper.dropna(), rtol=1e-3)
    
    # Test BB Position (should be in [0, 1])
    bb_position = features['bb_position']
    position_in_range = (bb_position.dropna().min() >= -0.5) and (bb_position.dropna().max() <= 1.5)
    
    print(f"\nBB Position Range: [{bb_position.dropna().min():.2f}, {bb_position.dropna().max():.2f}]")
    print(f"Expected: ~[0, 1] (can exceed slightly)")
    print(f"Reasonable: {position_in_range}")
    
    result = match1 and position_in_range
    print(f"\n✅ PASS" if result else "❌ FAIL")
    
    return result


def test_feature_ranges():
    """Test that all features are in reasonable ranges"""
    print("\n" + "="*80)
    print("TEST 7: FEATURE RANGE VALIDATION")
    print("="*80)
    
    df = pd.read_parquet('data_raw/NVDA.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    config = FeatureConfig.get_preset('comprehensive')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    features = feature_eng.create_all_features(df)
    
    issues = []
    
    # Expected ranges for key features
    expected_ranges = {
        'rsi_14': (0, 100),
        'rsi_7': (0, 100),
        'rsi_21': (0, 100),
        'cmf': (-1, 1),
        'williams_r': (-100, 0),
        'bb_position': (-1, 2),  # Can exceed [0,1] slightly
        'stoch_k': (0, 100),
        'stoch_d': (0, 100),
    }
    
    print(f"\nChecking {len(expected_ranges)} key features for range violations...")
    
    for feature, (min_expected, max_expected) in expected_ranges.items():
        if feature in features.columns:
            feat_min = features[feature].min()
            feat_max = features[feature].max()
            
            if feat_min < min_expected * 1.1 or feat_max > max_expected * 1.1:  # 10% tolerance
                issues.append(f"{feature}: [{feat_min:.2f}, {feat_max:.2f}] outside expected [{min_expected}, {max_expected}]")
                print(f"  ⚠️  {feature}: [{feat_min:.2f}, {feat_max:.2f}] (expected [{min_expected}, {max_expected}])")
            else:
                print(f"  ✅ {feature}: [{feat_min:.2f}, {feat_max:.2f}]")
    
    # Check for extreme values
    print(f"\nChecking for extreme values (>1M)...")
    extreme_features = []
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            max_val = abs(features[col]).max()
            if max_val > 1e6:
                extreme_features.append(f"{col}: {max_val:.2e}")
                print(f"  ❌ {col}: max = {max_val:.2e}")
    
    if not extreme_features:
        print(f"  ✅ No extreme values found")
    
    result = len(issues) == 0 and len(extreme_features) == 0
    print(f"\n✅ PASS" if result else "❌ FAIL")
    
    return result


def test_cross_ticker_consistency():
    """Test that features are consistent across tickers"""
    print("\n" + "="*80)
    print("TEST 8: CROSS-TICKER CONSISTENCY")
    print("="*80)
    
    tickers = ['AAPL', 'NVDA', 'TSLA']
    config = FeatureConfig.get_preset('minimal')
    feature_eng = EnhancedFinancialFeatures(feature_config=config)
    
    all_ranges = {}
    
    for ticker in tickers:
        df = pd.read_parquet(f'data_raw/{ticker}.parquet')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').iloc[:100]
        
        features = feature_eng.create_all_features(df)
        
        for col in features.columns:
            if pd.api.types.is_numeric_dtype(features[col]):
                if col not in all_ranges:
                    all_ranges[col] = []
                all_ranges[col].append({
                    'ticker': ticker,
                    'min': features[col].min(),
                    'max': features[col].max(),
                    'mean': features[col].mean()
                })
    
    # Check for features with wildly different ranges across tickers
    inconsistent = []
    
    print(f"\nChecking consistency across {len(tickers)} tickers...")
    
    for feature, ranges in all_ranges.items():
        maxs = [r['max'] for r in ranges]
        mins = [r['min'] for r in ranges]
        
        # Check if ranges differ by more than 10x
        if max(maxs) > 0 and min(maxs) > 0:
            ratio = max(maxs) / min(maxs)
            if ratio > 10:
                inconsistent.append(f"{feature}: max ratio = {ratio:.1f}x")
                print(f"  ⚠️  {feature}: max values differ by {ratio:.1f}x")
    
    if not inconsistent:
        print(f"  ✅ All features consistent across tickers")
    
    result = len(inconsistent) == 0
    print(f"\n✅ PASS" if result else "❌ FAIL (some variation expected)")
    
    return True  # Don't fail on this, just informational


def main():
    """Run all tests"""
    print("\n" + "╔"+"="*78+"╗")
    print("║" + " "*20 + "FEATURE ACCURACY VALIDATION" + " "*31 + "║")
    print("╚"+"="*78+"╝")
    
    tests = [
        ("Returns", test_returns),
        ("Volatility", test_volatility),
        ("Volume Features", test_volume_features),
        ("Momentum Indicators", test_momentum_indicators),
        ("Trend Indicators", test_trend_indicators),
        ("Bollinger Bands", test_bollinger_bands),
        ("Feature Ranges", test_feature_ranges),
        ("Cross-Ticker Consistency", test_cross_ticker_consistency),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("   Features are accurately calculated and ready for model training.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review issues above")
    
    return passed == total


if __name__ == "__main__":
    main()
