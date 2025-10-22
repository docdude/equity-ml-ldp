"""
Debug return_entropy to understand why it's constant
"""

import numpy as np
import pandas as pd
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

print("\n" + "="*80)
print("DEBUGGING RETURN_ENTROPY")
print("="*80)

# Load test data
df = pd.read_parquet('data_raw/AAPL.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
df = df.iloc[100:300]  # 200 samples

print(f"\nData samples: {len(df)}")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Calculate returns
returns = np.log(df['Close'] / df['Close'].shift(1))
print(f"\nReturns statistics:")
print(f"  Mean: {returns.mean():.6f}")
print(f"  Std: {returns.std():.6f}")
print(f"  Min: {returns.min():.6f}")
print(f"  Max: {returns.max():.6f}")

# Test the Shannon entropy calculation on a rolling window
def shannon_entropy_debug(returns_window):
    """Debug version with detailed output"""
    if len(returns_window) < 2:
        return 0
    
    # Discretize returns into bins
    try:
        bins = pd.qcut(returns_window, q=5, labels=False, duplicates='drop')
        unique_bins = len(bins.unique())
        
        # Calculate probabilities
        probs = bins.value_counts(normalize=True).sort_index()
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy, unique_bins, probs.values
    except Exception as e:
        return 0, 0, []

print("\n" + "="*80)
print("Testing Rolling Entropy (20-day windows)")
print("="*80)

# Test first 10 windows
for i in range(20, 30):
    window = returns.iloc[i-20:i]
    entropy, n_bins, probs = shannon_entropy_debug(window)
    
    print(f"\nWindow {i-19} (days {i-20} to {i}):")
    print(f"  Return std: {window.std():.6f}")
    print(f"  Return range: [{window.min():.6f}, {window.max():.6f}]")
    print(f"  Unique bins: {n_bins}")
    print(f"  Bin probabilities: {probs}")
    print(f"  Entropy: {entropy:.6f}")

# Now test with actual feature engineering
print("\n" + "="*80)
print("Testing with EnhancedFinancialFeatures")
print("="*80)

config = FeatureConfig.get_preset('comprehensive')
feature_eng = EnhancedFinancialFeatures(feature_config=config)
features = feature_eng.create_all_features(df)

if 'return_entropy' in features.columns:
    entropy_values = features['return_entropy'].dropna()
    print(f"\nReturn Entropy statistics:")
    print(f"  Count: {len(entropy_values)}")
    print(f"  Mean: {entropy_values.mean():.6f}")
    print(f"  Std: {entropy_values.std():.6f}")
    print(f"  Min: {entropy_values.min():.6f}")
    print(f"  Max: {entropy_values.max():.6f}")
    print(f"  Unique values: {len(entropy_values.unique())}")
    
    print(f"\nFirst 20 values:")
    print(entropy_values.head(20).values)
    
    print(f"\nValue counts:")
    print(entropy_values.value_counts().head(10))
else:
    print("\nreturn_entropy not in features!")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("""
The issue is that pd.qcut() creates 5 equal-frequency bins by design.
When you have 20 returns and create 5 bins, each bin gets exactly 4 values.
This means every window has probabilities of [0.2, 0.2, 0.2, 0.2, 0.2].

Shannon entropy for uniform distribution:
H = -sum(p * log2(p)) = -5 * (0.2 * log2(0.2)) = 2.3219

This is mathematically correct but captures NO variation between windows!

Solutions:
1. Use histogram binning instead of qcut (equal-width bins)
2. Increase number of bins (but need more data points)
3. Use a different entropy measure (approximate entropy)
4. Remove this feature as it's uninformative
""")
