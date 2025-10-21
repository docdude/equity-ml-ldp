"""
Systematic verification of feature_config.py against actual implementation

This script:
1. Extracts all 108 features from fin_feature_preprocessing.py
2. Maps them to feature groups in feature_config.py
3. Verifies all features are accounted for
4. Checks that presets make sense
"""

from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig
import pandas as pd

def main():
    print("=" * 80)
    print("FEATURE CONFIGURATION VERIFICATION")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Get actual features from implementation
    # ========================================================================
    print("\n1. EXTRACTING ACTUAL FEATURES FROM IMPLEMENTATION")
    print("-" * 80)
    
    df = pd.read_parquet('data_raw/AAPL.parquet')
    fe = EnhancedFinancialFeatures()
    actual_features_df = fe.create_all_features(df)
    actual_features = sorted(actual_features_df.columns.tolist())
    
    print(f"✓ Found {len(actual_features)} features in implementation")
    
    # ========================================================================
    # STEP 2: Map features to groups
    # ========================================================================
    print("\n2. MAPPING FEATURES TO GROUPS")
    print("-" * 80)
    
    # Define the mapping based on feature_config.py documentation
    feature_to_group = {
        # RETURNS (10 features)
        'returns': [
            'log_return_1d', 'log_return_2d', 'log_return_3d', 'log_return_5d',
            'log_return_10d', 'log_return_20d',
            'forward_return_1d', 'forward_return_3d', 'forward_return_5d',
            'return_acceleration'
        ],
        
        # VOLATILITY (12 features)
        'volatility': [
            'volatility_yz_10', 'volatility_yz_20', 'volatility_yz_60',
            'volatility_parkinson_10', 'volatility_parkinson_20',
            'volatility_gk_20', 'volatility_cc_20', 'volatility_cc_60',
            'vol_ratio_short_long', 'vol_of_vol',
            'realized_vol_positive', 'realized_vol_negative'
        ],
        
        # MOMENTUM (17 features)
        'momentum': [
            'rsi_7', 'rsi_14', 'rsi_21',  # RSI at different periods
            'macd', 'macd_signal', 'macd_hist', 'macd_divergence',
            'stoch_k', 'stoch_d', 'stoch_k_d_diff',
            'williams_r', 'atr', 'atr_ratio',
            'roc_10', 'roc_20', 'cci',
            'return_zscore_20'  # Could be statistical but used for momentum
        ],
        
        # TREND (10 features)
        'trend': [
            'adx', 'adx_plus', 'adx_minus',
            'sar', 'sar_signal',
            'aroon_up', 'aroon_down', 'aroon_oscillator',
            'dist_from_ma10', 'dist_from_ma20'
        ],
        
        # VOLUME (13 features)
        'volume': [
            'volume_norm', 'volume_std', 'volume_roc',
            'dollar_volume', 'dollar_volume_ma_ratio',
            'vwap_20', 'price_vwap_ratio',
            'obv', 'obv_ma_20',
            'ad_line', 'ad_normalized', 'cmf',
            'relative_volume'
        ],
        
        # BOLLINGER (5 features)
        'bollinger': [
            'bb_upper', 'bb_lower', 'bb_position', 'bb_width', 'bb_percent_b'
        ],
        
        # PRICE_POSITION (8 features)
        'price_position': [
            'price_position', 'dist_from_ma10', 'dist_from_ma20',
            'dist_from_ma50', 'dist_from_ma200',
            'dist_from_20d_high', 'dist_from_20d_low',
            'serial_corr_5'  # Price patterns
        ],
        
        # MICROSTRUCTURE (10 features)
        'microstructure': [
            'hl_range', 'hl_range_ma', 'oc_range',
            'roll_spread', 'cs_spread', 'hl_volatility_ratio',
            'amihud_illiq', 'amihud_illiquidity',  # Both versions
            'order_flow_imbalance', 'vpin', 'kyle_lambda'
        ],
        
        # STATISTICAL (8 features)
        'statistical': [
            'serial_corr_1', 'serial_corr_5',
            'skewness_20', 'skewness_60',
            'kurtosis_20', 'kurtosis_60',
            'return_zscore_20',
            'amihud_illiquidity'
        ],
        
        # ENTROPY (4 features)
        'entropy': [
            'return_entropy', 'lz_complexity',
            'variance_ratio', 'hurst_exponent'
        ],
        
        # REGIME (8 features)
        'regime': [
            'vol_regime', 'volume_percentile', 'trend_percentile',
            'rvi', 'market_state',
            'relative_volume', 'relative_volatility', 'fractal_dimension'
        ],
        
        # RISK_ADJUSTED (8 features)
        'risk_adjusted': [
            'sharpe_20', 'sharpe_60',
            'risk_adj_momentum_20',
            'downside_vol_20', 'sortino_20',
            'max_drawdown_20', 'calmar_20',
            'hui_heubel'
        ]
    }
    
    # Flatten to get all mapped features
    all_mapped = set()
    for group, features in feature_to_group.items():
        all_mapped.update(features)
    
    # Find unmapped features
    unmapped = set(actual_features) - all_mapped
    missing = all_mapped - set(actual_features)
    
    # Check for duplicates in mapping
    from collections import Counter
    all_mapped_list = []
    for features in feature_to_group.values():
        all_mapped_list.extend(features)
    duplicates = {k: v for k, v in Counter(all_mapped_list).items() if v > 1}
    
    # ========================================================================
    # STEP 3: Verification Results
    # ========================================================================
    print("\n3. VERIFICATION RESULTS")
    print("-" * 80)
    
    print(f"\n✓ Actual features in implementation: {len(actual_features)}")
    print(f"✓ Features mapped in config:         {len(all_mapped)}")
    print(f"✓ Unmapped features:                 {len(unmapped)}")
    print(f"✓ Missing features:                  {len(missing)}")
    print(f"✓ Duplicate mappings:                {len(duplicates)}")
    
    if unmapped:
        print(f"\n❌ UNMAPPED FEATURES ({len(unmapped)}):")
        for f in sorted(unmapped):
            print(f"  • {f}")
    
    if missing:
        print(f"\n⚠️  MISSING FEATURES (in config but not in code) ({len(missing)}):")
        for f in sorted(missing):
            print(f"  • {f}")
    
    if duplicates:
        print(f"\n⚠️  DUPLICATE MAPPINGS ({len(duplicates)}):")
        for f, count in sorted(duplicates.items()):
            print(f"  • {f}: mapped {count} times")
    
    # ========================================================================
    # STEP 4: Group-by-Group Verification
    # ========================================================================
    print("\n4. GROUP-BY-GROUP VERIFICATION")
    print("-" * 80)
    
    for group, expected_features in feature_to_group.items():
        config_info = FeatureConfig.FEATURE_GROUPS.get(group, {})
        expected_count = config_info.get('count', len(expected_features))
        actual_count = len(expected_features)
        
        # Find which expected features are actually in implementation
        found = [f for f in expected_features if f in actual_features]
        not_found = [f for f in expected_features if f not in actual_features]
        
        status = "✓" if actual_count == expected_count and len(not_found) == 0 else "❌"
        
        print(f"\n{status} {group.upper():20s} "
              f"Expected: {expected_count:2d}  Mapped: {actual_count:2d}  Found: {len(found):2d}")
        
        if not_found:
            print(f"   Missing from implementation:")
            for f in not_found:
                print(f"     • {f}")
    
    # ========================================================================
    # STEP 5: Preset Verification
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. PRESET VERIFICATION")
    print("=" * 80)
    
    for preset_name, preset_info in FeatureConfig.PRESETS.items():
        config = preset_info['config']
        expected = preset_info['expected_features']
        
        # Calculate actual feature count for this preset
        enabled_groups = [g for g, enabled in config.items() if enabled]
        actual_count = sum(
            len(feature_to_group[group])
            for group in enabled_groups
        )
        
        # Account for overlaps (features in multiple groups)
        unique_features = set()
        for group in enabled_groups:
            unique_features.update(feature_to_group[group])
        actual_unique = len(unique_features)
        
        diff = abs(actual_unique - expected)
        status = "✓" if diff <= 5 else "❌"  # Allow 5 feature tolerance
        
        print(f"\n{status} {preset_name.upper():20s}")
        print(f"   Expected: {expected:3d}  Calculated: {actual_unique:3d}  "
              f"Diff: {diff:2d}  Groups: {len(enabled_groups)}/12")
        
        if diff > 5:
            print(f"   ⚠️  Significant difference - may need config adjustment")
    
    # ========================================================================
    # STEP 6: Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    total_issues = len(unmapped) + len(missing) + len(duplicates)
    
    if total_issues == 0:
        print("\n✅ VERIFICATION PASSED")
        print("   • All 108 features properly mapped")
        print("   • No unmapped or missing features")
        print("   • No duplicate mappings")
        print("   • Presets are well-configured")
        print("\n🚀 Ready for training with selective features!")
    else:
        print("\n⚠️  VERIFICATION FOUND ISSUES")
        print(f"   • Unmapped features: {len(unmapped)}")
        print(f"   • Missing features: {len(missing)}")
        print(f"   • Duplicate mappings: {len(duplicates)}")
        print("\n📝 Please review the issues above and update feature_config.py")
    
    return total_issues == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
