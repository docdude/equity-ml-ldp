"""
Test normalization to see which features become extreme AFTER RobustScaler
"""
import numpy as np
import pandas as pd
from fin_feature_preprocessing import EnhancedFinancialFeatures
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# Load one ticker
print("Loading AAPL data...")
df = pd.read_parquet('data_raw/AAPL.parquet')
print(f"Loaded {len(df)} rows")

# Create features
print("\nCreating features...")
feature_engineer = EnhancedFinancialFeatures(feature_preset='wavenet_optimized')
features = feature_engineer.create_all_features(df)
print(f"Created {len(features.columns)} features")

# Convert to numpy
X = features.values
print(f"\nFeature matrix shape: {X.shape}")

# Apply RobustScaler (same as training)
print("\nApplying RobustScaler...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n{'='*80}")
print(f"BEFORE NORMALIZATION")
print(f"{'='*80}")
print(f"  Mean: {X.mean():.4f}")
print(f"  Std:  {X.std():.4f}")
print(f"  Min:  {X.min():.4f}")
print(f"  Max:  {X.max():.4f}")

print(f"\n{'='*80}")
print(f"AFTER ROBUST NORMALIZATION")
print(f"{'='*80}")
print(f"  Mean: {X_scaled.mean():.4f}")
print(f"  Std:  {X_scaled.std():.4f}")
print(f"  Min:  {X_scaled.min():.4f}")
print(f"  Max:  {X_scaled.max():.4f}")

# Check each feature AFTER normalization
print(f"\n{'='*80}")
print(f"PER-FEATURE ANALYSIS (AFTER NORMALIZATION)")
print(f"{'='*80}")
print(f"{'Feature':<30} {'Before Min':>12} {'Before Max':>12} {'After Min':>12} {'After Max':>12} {'After Std':>12}")
print(f"{'-'*100}")

problematic = []
for i, col in enumerate(features.columns):
    before_min = X[:, i].min()
    before_max = X[:, i].max()
    after_min = X_scaled[:, i].min()
    after_max = X_scaled[:, i].max()
    after_std = X_scaled[:, i].std()
    
    # Flag if extreme AFTER normalization
    is_extreme = (abs(after_max) > 10 or abs(after_min) > 10 or after_std > 5)
    
    marker = "⚠️ " if is_extreme else "   "
    print(f"{marker}{col:<28} {before_min:>12.4f} {before_max:>12.4f} {after_min:>12.4f} {after_max:>12.4f} {after_std:>12.4f}")
    
    if is_extreme:
        problematic.append({
            'feature': col,
            'before_min': before_min,
            'before_max': before_max,
            'after_min': after_min,
            'after_max': after_max,
            'after_std': after_std
        })

# Summary
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
if problematic:
    print(f"⚠️  {len(problematic)} features with extreme values AFTER normalization:")
    for pf in sorted(problematic, key=lambda x: max(abs(x['after_min']), abs(x['after_max'])), reverse=True):
        print(f"\n  {pf['feature']}:")
        print(f"    Before: [{pf['before_min']:.4f}, {pf['before_max']:.4f}]")
        print(f"    After:  [{pf['after_min']:.4f}, {pf['after_max']:.4f}] (std={pf['after_std']:.2f})")
        
        # Diagnose why
        feat_name = pf['feature'].lower()
        if 'volume' in feat_name:
            print(f"    💡 Volume features are highly skewed → Use log1p BEFORE normalization")
        elif 'spread' in feat_name:
            print(f"    💡 Spread can have extreme outliers → Use log1p or winsorize BEFORE normalization")
        elif pf['after_std'] > 5:
            print(f"    💡 High variance even after RobustScaler → Likely has heavy outliers")
            print(f"       Consider: winsorization (clip at 1st/99th percentile) BEFORE normalization")
else:
    print("✅ All features within reasonable range after normalization!")

print(f"\n{'='*80}")
print(f"ROBUST SCALER INFO")
print(f"{'='*80}")
print("RobustScaler uses median and IQR, which should handle outliers well.")
print("If features still have extreme values AFTER RobustScaler, it means:")
print("  1. Distribution has very heavy tails (IQR is tiny compared to range)")
print("  2. Feature needs transformation (log, sqrt) BEFORE normalization")
print("  3. Feature may need winsorization to clip extreme outliers")
