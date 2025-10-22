"""
Test the full create_all_features method
"""

import numpy as np
import pandas as pd
import sys
from fin_feature_preprocessing import EnhancedFinancialFeatures

def main():
    print("="*80)
    print("TESTING FULL FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Load test data
    print("\n📊 Loading test data (AAPL)...")
    df = pd.read_parquet('data_raw/AAPL.parquet')
    
    # FIX: Drop NaN values in OHLCV columns (TA-Lib cannot handle NaN)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    feature_engineer = EnhancedFinancialFeatures()
    
    print("\n🔧 Running create_all_features()...")
    try:
        features = feature_engineer.create_all_features(df)
        print(f"✅ SUCCESS!")
        print(f"\n📊 Feature Statistics:")
        print(f"   Shape: {features.shape}")
        print(f"   Number of features: {features.shape[1]}")
        print(f"   Total values: {features.size:,}")
        
        # Check for NaN and Inf
        nan_count = features.isna().sum().sum()
        nan_pct = nan_count / features.size * 100
        
        # Check inf only for numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric_features.replace([np.inf, -np.inf], np.nan)).sum().sum()
        
        print(f"\n📈 Data Quality:")
        print(f"   NaN values: {nan_count:,} ({nan_pct:.2f}%)")
        print(f"   Inf values: {inf_count:,}")
        
        # Check for zero-filled columns (potential issue)
        print(f"\n⚠️  Zero-only columns (might indicate NaN fillna):")
        for col in numeric_features.columns:
            non_zero = (numeric_features[col] != 0).sum()
            if non_zero == 0:
                print(f"   - {col}: ALL ZEROS")
            elif non_zero < 10:
                print(f"   - {col}: only {non_zero} non-zero values")
        
        # Show feature columns
        print(f"\n📋 Feature columns ({len(features.columns)}):")
        for i, col in enumerate(features.columns, 1):
            nan_col = features[col].isna().sum()
            print(f"   {i:2d}. {col:30s} (NaN: {nan_col:4d})")
        
        # Show first few rows
        print(f"\n🔍 Sample data (first 5 rows):")
        print(features.head())
        
        # Show data from middle to see if features are actually calculated
        print(f"\n🔍 Sample data (rows 100-105, after warm-up period):")
        print(features.iloc[100:105])
        
        # Show statistics
        print(f"\n📊 Feature Statistics (non-zero values):")
        print(features.describe().loc[['mean', 'std', 'min', 'max']].T)
        
        print("\n✅ ALL FEATURES CREATED SUCCESSFULLY!")
        return 0
        
    except Exception as e:
        print(f"❌ FAILED!")
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        import traceback
        print("\n📍 Full traceback:")
        traceback.print_exc()
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
