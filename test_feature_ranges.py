"""
Quick test to identify which of the 18 features has extreme values
"""
import numpy as np
import pandas as pd
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

# Load one ticker
print("Loading AAPL data...")
df = pd.read_parquet('data_raw/AAPL.parquet')
print(f"Loaded {len(df)} rows")

# Create features using wavenet_optimized preset
print("\nCreating features with wavenet_optimized preset...")
feature_engineer = EnhancedFinancialFeatures(feature_preset='wavenet_optimized')
features = feature_engineer.create_all_features(df)

print(f"\nTotal features created: {len(features.columns)}")
print(f"Expected 18 features")

# Analyze each feature
print(f"\n{'='*80}")
print(f"FEATURE ANALYSIS - LOOKING FOR EXTREME VALUES")
print(f"{'='*80}")
print(f"{'Feature':<30} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12} {'Range':>12}")
print(f"{'-'*80}")

problematic = []
for col in features.columns:
    data = features[col].values
    data_min = np.min(data)
    data_max = np.max(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_range = data_max - data_min
    
    # Flag if extreme
    is_extreme = (abs(data_max) > 10 or abs(data_min) > 10 or data_std > 5 or data_range > 50)
    
    marker = "⚠️ " if is_extreme else "   "
    print(f"{marker}{col:<28} {data_min:>12.4f} {data_max:>12.4f} {data_mean:>12.4f} {data_std:>12.4f} {data_range:>12.4f}")
    
    if is_extreme:
        problematic.append({
            'feature': col,
            'min': data_min,
            'max': data_max,
            'mean': data_mean,
            'std': data_std,
            'range': data_range
        })

# Summary
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Problematic features (need transformation): {len(problematic)}")

if problematic:
    print(f"\nTop issues ranked by absolute max:")
    for pf in sorted(problematic, key=lambda x: max(abs(x['min']), abs(x['max'])), reverse=True):
        print(f"  {pf['feature']:<30} range: [{pf['min']:.2f}, {pf['max']:.2f}]")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for pf in sorted(problematic, key=lambda x: max(abs(x['min']), abs(x['max'])), reverse=True):
        feat_name = pf['feature'].lower()
        
        if 'volume' in feat_name:
            print(f"  • {pf['feature']}: Apply log1p transform (volume features are naturally skewed)")
        elif 'spread' in feat_name or 'amihud' in feat_name or 'kyle' in feat_name:
            print(f"  • {pf['feature']}: Apply log1p or winsorize (microstructure can have outliers)")
        elif 'volatility' in feat_name or 'atr' in feat_name or 'bb' in feat_name:
            print(f"  • {pf['feature']}: Apply sqrt transform (volatility has moderate skew)")
        elif 'return' in feat_name:
            print(f"  • {pf['feature']}: Apply winsorization at 1st/99th percentile (fat tails)")
        else:
            print(f"  • {pf['feature']}: Check distribution, may need log or sqrt transform")
else:
    print("✅ All features within reasonable range!")
