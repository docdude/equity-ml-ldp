"""
Test individual feature engineering methods
"""

import numpy as np
import pandas as pd
import sys
from fin_feature_preprocessing import EnhancedFinancialFeatures

def test_feature_method(feature_engineer, df, method_name, *args):
    """Test a single feature method"""
    try:
        print(f"Testing {method_name}...", end=" ")
        result = getattr(feature_engineer, method_name)(df, *args)
        
        # Check result
        if result is None:
            print(f"❌ FAILED - returned None")
            return False
        
        # Check for NaN percentage
        if isinstance(result, pd.Series):
            nan_pct = result.isna().sum() / len(result) * 100
            inf_count = np.isinf(result.replace([np.inf, -np.inf], np.nan)).sum()
            print(f"✓ OK (NaN: {nan_pct:.1f}%, Inf: {inf_count})")
        elif isinstance(result, pd.DataFrame):
            nan_pct = result.isna().sum().sum() / result.size * 100
            print(f"✓ OK (NaN: {nan_pct:.1f}%)")
        else:
            print(f"✓ OK (type: {type(result)})")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        return False

def main():
    print("="*80)
    print("TESTING INDIVIDUAL FEATURE ENGINEERING METHODS")
    print("="*80)
    
    # Load test data
    print("\n📊 Loading test data (AAPL)...")
    df = pd.read_parquet('data_raw/AAPL.parquet')
    
    # FIX: Drop NaN values in OHLCV columns (TA-Lib cannot handle NaN)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    feature_engineer = EnhancedFinancialFeatures()
    
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL METHODS")
    print("="*80)
    
    results = {}
    
    # Test each method
    print("\n1. VOLATILITY METHODS")
    print("-"*40)
    results['yang_zhang_volatility'] = test_feature_method(
        feature_engineer, df, '_yang_zhang_volatility', 20
    )
    results['parkinson_volatility'] = test_feature_method(
        feature_engineer, df, '_parkinson_volatility', 20
    )
    
    print("\n2. VOLUME METHODS")
    print("-"*40)
    results['calculate_vpin'] = test_feature_method(
        feature_engineer, df, '_calculate_vpin', 50
    )
    results['kyle_lambda'] = test_feature_method(
        feature_engineer, df, '_kyle_lambda', 20
    )
    results['order_flow_imbalance'] = test_feature_method(
        feature_engineer, df, '_order_flow_imbalance'
    )
    
    print("\n3. MICROSTRUCTURE METHODS")
    print("-"*40)
    results['roll_spread'] = test_feature_method(
        feature_engineer, df, '_roll_spread', 20
    )
    results['corwin_schultz_spread'] = test_feature_method(
        feature_engineer, df, '_corwin_schultz_spread', 20
    )
    
    print("\n4. INFORMATION METHODS")
    print("-"*40)
    # Test with a small series for Shannon entropy
    returns = df['Close'].pct_change().dropna()
    results['shannon_entropy'] = test_feature_method(
        feature_engineer, returns.head(50), '_shannon_entropy'
    )
    results['lempel_ziv_complexity'] = test_feature_method(
        feature_engineer, returns.head(50), '_lempel_ziv_complexity'
    )
    
    print("\n5. STRUCTURAL METHODS")
    print("-"*40)
    results['variance_ratio'] = test_feature_method(
        feature_engineer, returns, '_variance_ratio', 5
    )
    results['hurst_exponent'] = test_feature_method(
        feature_engineer, returns.head(100), '_hurst_exponent'
    )
    
    print("\n6. FRACTAL METHODS")
    print("-"*40)
    results['fractal_dimension'] = test_feature_method(
        feature_engineer, df['Close'], '_fractal_dimension', 20
    )
    
    print("\n7. LIQUIDITY METHODS")
    print("-"*40)
    results['hui_heubel_liquidity'] = test_feature_method(
        feature_engineer, df, '_hui_heubel_liquidity', 5
    )
    
    print("\n8. TRIPLE BARRIERS")
    print("-"*40)
    results['create_dynamic_triple_barriers'] = test_feature_method(
        feature_engineer, df, 'create_dynamic_triple_barriers', 0.06, 0.03, 5
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed:")
        for name, passed in results.items():
            if not passed:
                print(f"   - {name}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
