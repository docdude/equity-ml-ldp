"""
Comprehensive Feature Validation for wavenet_optimized_v2
==========================================================

Systematically validates all 40 features in wavenet_optimized_v2:
1. Cross-validation against TA-Lib (where applicable)
2. Statistical quality checks (NaN, Inf, range, distribution)
3. Correlation with price/volume (sanity checks)
4. Time-series properties (stationarity, autocorrelation)

This ensures the model trains on accurate, high-quality features.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig


class FeatureValidator:
    """Validates feature quality and accuracy"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Args:
            tolerance: Relative tolerance for TA-Lib comparison (1% default)
        """
        self.tolerance = tolerance
        self.results = []
        
    def load_test_data(self, ticker: str = 'AAPL') -> pd.DataFrame:
        """Load sample data for validation"""
        import os
        data_path = f'data_raw/{ticker}.parquet'
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        df = pd.read_parquet(data_path)
        
        # Set date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Use recent data (last 500 days for speed)
        df = df.tail(500).copy()
        
        print(f"✅ Loaded {ticker} data: {len(df)} rows")
        print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        return df
    
    def validate_against_talib(
        self, 
        feature_name: str,
        our_values: pd.Series,
        talib_func,
        talib_params: dict,
        df: pd.DataFrame
    ) -> Dict:
        """
        Compare our feature implementation against TA-Lib
        
        Returns:
            Dict with validation results
        """
        # Calculate TA-Lib version
        try:
            talib_values = talib_func(**talib_params)
            
            # Align (both may have NaN at start due to warmup)
            valid_mask = ~(np.isnan(our_values) | np.isnan(talib_values))
            our_clean = our_values[valid_mask]
            talib_clean = talib_values[valid_mask]
            
            if len(our_clean) == 0:
                return {
                    'status': 'ERROR',
                    'reason': 'No valid overlapping values',
                    'match': False
                }
            
            # Calculate differences
            abs_diff = np.abs(our_clean - talib_clean)
            rel_diff = abs_diff / (np.abs(talib_clean) + 1e-10)
            
            mean_rel_diff = rel_diff.mean()
            max_rel_diff = rel_diff.max()
            
            # Correlation check
            correlation = np.corrcoef(our_clean, talib_clean)[0, 1]
            
            # Pass criteria
            passed = (
                mean_rel_diff < self.tolerance and
                correlation > 0.99
            )
            
            return {
                'status': 'PASS' if passed else 'FAIL',
                'mean_rel_diff': mean_rel_diff,
                'max_rel_diff': max_rel_diff,
                'correlation': correlation,
                'match': passed,
                'n_compared': len(our_clean)
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'reason': str(e),
                'match': False
            }
    
    def statistical_quality_checks(
        self,
        feature_name: str,
        values: pd.Series,
        df: pd.DataFrame
    ) -> Dict:
        """
        Comprehensive statistical quality checks
        
        Checks:
        1. Missing values (NaN, Inf)
        2. Value range (detect unrealistic values)
        3. Distribution properties (skew, kurtosis)
        4. Stationarity (ADF test)
        5. Autocorrelation
        """
        results = {
            'feature': feature_name,
            'n_samples': len(values)
        }
        
        # 1. Missing value check
        n_nan = values.isna().sum()
        n_inf = np.isinf(values).sum()
        pct_valid = ((len(values) - n_nan - n_inf) / len(values)) * 100
        
        results['missing'] = {
            'n_nan': int(n_nan),
            'n_inf': int(n_inf),
            'pct_valid': pct_valid,
            'pass': pct_valid > 80  # At least 80% valid
        }
        
        # Get valid values for remaining checks
        valid_values = values.dropna()
        valid_values = valid_values[np.isfinite(valid_values)]
        
        if len(valid_values) < 50:
            results['error'] = 'Insufficient valid values'
            return results
        
        # 2. Range check
        results['range'] = {
            'min': float(valid_values.min()),
            'max': float(valid_values.max()),
            'mean': float(valid_values.mean()),
            'std': float(valid_values.std()),
            'median': float(valid_values.median())
        }
        
        # Check for unrealistic values (beyond 100 standard deviations)
        z_scores = np.abs((valid_values - valid_values.mean()) / valid_values.std())
        n_outliers = (z_scores > 100).sum()
        results['range']['extreme_outliers'] = int(n_outliers)
        results['range']['pass'] = n_outliers < len(valid_values) * 0.01  # < 1% extreme
        
        # 3. Distribution properties
        results['distribution'] = {
            'skewness': float(stats.skew(valid_values)),
            'kurtosis': float(stats.kurtosis(valid_values)),
        }
        
        # Check if distribution is reasonable (not pathological)
        results['distribution']['pass'] = (
            abs(results['distribution']['skewness']) < 20 and
            abs(results['distribution']['kurtosis']) < 100
        )
        
        # 4. Stationarity test (ADF test)
        try:
            adf_result = adfuller(valid_values, maxlag=20, regression='c', autolag='AIC')
            results['stationarity'] = {
                'adf_statistic': float(adf_result[0]),
                'p_value': float(adf_result[1]),
                'is_stationary': adf_result[1] < 0.05  # Reject null = stationary
            }
        except Exception as e:
            results['stationarity'] = {
                'error': str(e),
                'is_stationary': None
            }
        
        # 5. Autocorrelation (lag 1)
        try:
            autocorr = valid_values.autocorr(lag=1)
            results['autocorr_lag1'] = float(autocorr) if not np.isnan(autocorr) else None
        except:
            results['autocorr_lag1'] = None
        
        # 6. Correlation with price (sanity check)
        try:
            price_corr = valid_values.corr(df['Close'].loc[valid_values.index])
            results['price_correlation'] = float(price_corr) if not np.isnan(price_corr) else None
        except:
            results['price_correlation'] = None
        
        # Overall quality score
        quality_checks = [
            results['missing']['pass'],
            results['range']['pass'],
            results['distribution']['pass']
        ]
        results['quality_score'] = sum(quality_checks) / len(quality_checks)
        results['overall_pass'] = results['quality_score'] >= 0.66  # At least 2/3 checks pass
        
        return results
    
    def validate_feature(
        self,
        feature_name: str,
        features_df: pd.DataFrame,
        df_raw: pd.DataFrame
    ) -> Dict:
        """
        Complete validation for a single feature
        
        Returns comprehensive validation report
        """
        print(f"\n{'='*60}")
        print(f"Validating: {feature_name}")
        print('='*60)
        
        if feature_name not in features_df.columns:
            print(f"❌ Feature not found in features_df")
            return {
                'feature': feature_name,
                'status': 'NOT_FOUND',
                'pass': False
            }
        
        values = features_df[feature_name]
        
        # Statistical quality checks (always run)
        quality_results = self.statistical_quality_checks(feature_name, values, df_raw)
        
        print(f"\n📊 Statistical Quality:")
        print(f"   Valid values: {quality_results['missing']['pct_valid']:.1f}%")
        print(f"   Range: [{quality_results['range']['min']:.4f}, {quality_results['range']['max']:.4f}]")
        print(f"   Mean: {quality_results['range']['mean']:.4f}, Std: {quality_results['range']['std']:.4f}")
        print(f"   Skewness: {quality_results['distribution']['skewness']:.2f}, Kurtosis: {quality_results['distribution']['kurtosis']:.2f}")
        
        if quality_results.get('stationarity'):
            stat_info = quality_results['stationarity']
            if stat_info.get('is_stationary') is not None:
                stat_str = 'Stationary' if stat_info['is_stationary'] else 'Non-stationary'
                print(f"   Stationarity: {stat_str} (p={stat_info['p_value']:.4f})")
        
        quality_pass = quality_results['overall_pass']
        print(f"   Quality Score: {quality_results['quality_score']*100:.0f}% - {'✅ PASS' if quality_pass else '❌ FAIL'}")
        
        # TA-Lib comparison (if applicable)
        talib_result = self._check_talib_equivalent(feature_name, values, df_raw)
        
        if talib_result:
            print(f"\n📐 TA-Lib Comparison:")
            if talib_result['status'] == 'PASS':
                print(f"   ✅ MATCH - Mean diff: {talib_result['mean_rel_diff']*100:.3f}%, Corr: {talib_result['correlation']:.5f}")
            elif talib_result['status'] == 'FAIL':
                print(f"   ❌ MISMATCH - Mean diff: {talib_result['mean_rel_diff']*100:.3f}%, Corr: {talib_result['correlation']:.5f}")
            else:
                print(f"   ⚠️  {talib_result['status']}: {talib_result.get('reason', 'N/A')}")
        else:
            print(f"\n📐 TA-Lib Comparison: N/A (custom feature)")
            talib_result = {'status': 'N/A', 'match': None}
        
        # Overall result
        overall_pass = quality_pass and (talib_result['match'] in [True, None])
        
        result = {
            'feature': feature_name,
            'quality': quality_results,
            'talib': talib_result,
            'overall_pass': overall_pass
        }
        
        self.results.append(result)
        return result
    
    def _check_talib_equivalent(
        self,
        feature_name: str,
        values: pd.Series,
        df: pd.DataFrame
    ) -> Dict:
        """Check if feature has TA-Lib equivalent and validate"""
        
        # Define TA-Lib mappings
        talib_mappings = {
            # Momentum indicators
            'rsi_7': (talib.RSI, {'real': df['Close'].values, 'timeperiod': 7}),
            'rsi_14': (talib.RSI, {'real': df['Close'].values, 'timeperiod': 14}),
            'rsi_21': (talib.RSI, {'real': df['Close'].values, 'timeperiod': 21}),
            
            'macd': (lambda: talib.MACD(df['Close'].values, 12, 26, 9)[0], {}),
            'macd_signal': (lambda: talib.MACD(df['Close'].values, 12, 26, 9)[1], {}),
            'macd_hist': (lambda: talib.MACD(df['Close'].values, 12, 26, 9)[2], {}),
            
            'stoch_k': (lambda: talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values, 14, 3, 0, 3, 0)[0], {}),
            'stoch_d': (lambda: talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values, 14, 3, 0, 3, 0)[1], {}),
            
            'cci': (talib.CCI, {'high': df['High'].values, 'low': df['Low'].values, 'close': df['Close'].values, 'timeperiod': 14}),
            
            'atr': (talib.ATR, {'high': df['High'].values, 'low': df['Low'].values, 'close': df['Close'].values, 'timeperiod': 14}),
            
            # Bollinger Bands
            'bb_upper': (lambda: talib.BBANDS(df['Close'].values, 20, 2, 2)[0], {}),
            'bb_lower': (lambda: talib.BBANDS(df['Close'].values, 20, 2, 2)[2], {}),
            
            # Trend indicators
            'adx': (talib.ADX, {'high': df['High'].values, 'low': df['Low'].values, 'close': df['Close'].values, 'timeperiod': 14}),
            'adx_plus': (talib.PLUS_DI, {'high': df['High'].values, 'low': df['Low'].values, 'close': df['Close'].values, 'timeperiod': 14}),
            
            'sar': (talib.SAR, {'high': df['High'].values, 'low': df['Low'].values, 'acceleration': 0.02, 'maximum': 0.2}),
            
            # Volume indicators
            'obv': (talib.OBV, {'close': df['Close'].values, 'volume': df['Volume'].values}),
        }
        
        if feature_name not in talib_mappings:
            return None
        
        func, params = talib_mappings[feature_name]
        
        return self.validate_against_talib(
            feature_name,
            values,
            func,
            params,
            df
        )
    
    def generate_report(self) -> pd.DataFrame:
        """Generate summary report of all validations"""
        
        if not self.results:
            print("No validation results to report")
            return pd.DataFrame()
        
        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            feature = result['feature']
            quality = result['quality']
            talib = result['talib']
            
            summary_data.append({
                'feature': feature,
                'overall_pass': '✅' if result['overall_pass'] else '❌',
                'quality_score': f"{quality['quality_score']*100:.0f}%",
                'pct_valid': f"{quality['missing']['pct_valid']:.1f}%",
                'range_min': f"{quality['range']['min']:.4f}",
                'range_max': f"{quality['range']['max']:.4f}",
                'skewness': f"{quality['distribution']['skewness']:.2f}",
                'kurtosis': f"{quality['distribution']['kurtosis']:.2f}",
                'talib_status': talib['status'],
                'talib_match': '✅' if talib.get('match') else ('❌' if talib.get('match') is False else 'N/A')
            })
        
        df = pd.DataFrame(summary_data)
        
        return df


def main():
    """Run comprehensive validation on wavenet_optimized_v2 features"""
    
    print("="*80)
    print("WAVENET_OPTIMIZED_V2 FEATURE VALIDATION")
    print("="*80)
    print("\nValidating all 40 features for:")
    print("  ✓ Statistical quality (NaN, range, distribution)")
    print("  ✓ TA-Lib accuracy (where applicable)")
    print("  ✓ Stationarity and autocorrelation")
    print("  ✓ Correlation with price (sanity check)")
    
    # Load configuration
    config = FeatureConfig.get_preset('wavenet_optimized_v2')
    feature_list = config['feature_list']
    
    print(f"\n📋 Features to validate: {len(feature_list)}")
    
    # Load test data
    validator = FeatureValidator(tolerance=0.01)  # 1% tolerance for TA-Lib
    df_raw = validator.load_test_data('AAPL')
    
    # Clean data - drop NaN values in OHLCV columns (TA-Lib cannot handle NaN)
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_raw = df_raw.dropna(subset=required_cols)
    print(f"✅ Cleaned data: {len(df_raw)} rows after dropping NaN")
    
    # Generate features
    print(f"\n🔧 Generating features with wavenet_optimized_v2 config...")
    feature_engineer = EnhancedFinancialFeatures(feature_config=config)
    features_df = feature_engineer.create_all_features(df_raw)
    
    print(f"✅ Generated {len(features_df.columns)} features")
    
    # Validate each feature
    print(f"\n{'='*80}")
    print("FEATURE-BY-FEATURE VALIDATION")
    print('='*80)
    
    for i, feature_name in enumerate(feature_list, 1):
        if i % 5 == 0:
            print(f"  Progress: {i}/{len(feature_list)} features validated...")
        
        # Suppress detailed output during validation loop
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            validator.validate_feature(feature_name, features_df, df_raw)
        finally:
            sys.stdout = old_stdout
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print('='*80)
    
    report_df = validator.generate_report()
    
    # Display report (first 20 rows to avoid broken pipe)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(f"\nShowing first 20 features:")
    print(f"\n{report_df.head(20).to_string(index=False)}")
    
    # Save to file
    output_file = 'artifacts/wavenet_v2_feature_validation.csv'
    report_df.to_csv(output_file, index=False)
    print(f"\n✅ Detailed report saved to: {output_file}")
    
    # Summary statistics
    n_total = len(report_df)
    n_passed = (report_df['overall_pass'] == '✅').sum()
    n_failed = (report_df['overall_pass'] == '❌').sum()
    n_talib_checked = (report_df['talib_status'].isin(['PASS', 'FAIL'])).sum()
    n_talib_passed = (report_df['talib_match'] == '✅').sum()
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print('='*80)
    print(f"\n📊 Overall:")
    print(f"   Total features validated: {n_total}")
    print(f"   Passed: {n_passed} ({n_passed/n_total*100:.1f}%)")
    print(f"   Failed: {n_failed} ({n_failed/n_total*100:.1f}%)")
    
    print(f"\n📐 TA-Lib Comparison:")
    print(f"   Features with TA-Lib equivalent: {n_talib_checked}")
    print(f"   Matched TA-Lib: {n_talib_passed} ({n_talib_passed/n_talib_checked*100:.1f}% if n_talib_checked else 0)")
    
    if n_failed > 0:
        print(f"\n⚠️  FAILED FEATURES:")
        failed_features = report_df[report_df['overall_pass'] == '❌']['feature'].tolist()
        for feat in failed_features:
            print(f"   - {feat}")
    
    if n_passed == n_total:
        print(f"\n🎉 ALL FEATURES PASSED VALIDATION!")
        print(f"   The model will train on accurate, high-quality data.")
    else:
        print(f"\n⚠️  Some features failed validation - review artifacts/wavenet_v2_feature_validation.csv")
    
    return validator, report_df


if __name__ == '__main__':
    validator, report = main()
