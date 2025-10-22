"""
Diagnostic script to debug vpin and cs_spread returning constant values
"""

import pandas as pd
import numpy as np
from fin_feature_preprocessing import EnhancedFinancialFeatures

# Load sample data
print("Loading AAPL data...")
df = pd.read_csv('AAPL_2022_2024.csv', parse_dates=['Date'])
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
print(f"Data loaded: {len(df)} rows")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print()

# Test VPIN calculation step-by-step
print("=" * 80)
print("DEBUGGING VPIN CALCULATION")
print("=" * 80)

price_change = df['Close'].diff()
buy_volume = df['Volume'].where(price_change > 0, 0)
sell_volume = df['Volume'].where(price_change < 0, 0)

print(f"\nPrice change stats:")
print(f"  Positive changes: {(price_change > 0).sum()}")
print(f"  Negative changes: {(price_change < 0).sum()}")
print(f"  Zero changes: {(price_change == 0).sum()}")
print(f"  NaN: {price_change.isna().sum()}")

print(f"\nBuy volume stats:")
print(f"  Non-zero: {(buy_volume > 0).sum()}")
print(f"  Zero: {(buy_volume == 0).sum()}")
print(f"  Mean: {buy_volume.mean():.2f}")

print(f"\nSell volume stats:")
print(f"  Non-zero: {(sell_volume > 0).sum()}")
print(f"  Zero: {(sell_volume == 0).sum()}")
print(f"  Mean: {sell_volume.mean():.2f}")

# Check cumulative volumes for bucketing
cum_volume = df['Volume'].cumsum()
volume_bucket_size = 50
bucket_indices = (cum_volume / volume_bucket_size).astype(int)

print(f"\nBucket stats:")
print(f"  Total buckets: {bucket_indices.nunique()}")
print(f"  Bucket size parameter: {volume_bucket_size}")
print(f"  Total volume: {df['Volume'].sum():.0f}")
print(f"  Expected buckets: {df['Volume'].sum() / volume_bucket_size:.0f}")

# Sample buckets
print(f"\nFirst 10 bucket indices: {bucket_indices.head(10).tolist()}")

# Test Corwin-Schultz spread step-by-step
print("\n" + "=" * 80)
print("DEBUGGING CORWIN-SCHULTZ SPREAD CALCULATION")
print("=" * 80)

high = df['High']
low = df['Low']

h2 = high.rolling(2).max()
l2 = low.rolling(2).min()

beta = (np.log(high / low)) ** 2
gamma = (np.log(h2 / l2)) ** 2

term1 = (np.sqrt(2 * beta) - np.sqrt(beta))
term2 = np.sqrt(gamma)

k = 3 - 2 * np.sqrt(2)

print(f"\nConstants:")
print(f"  k = {k:.6f}")

print(f"\nBeta (daily HL ratio squared):")
print(f"  Mean: {beta.mean():.8f}")
print(f"  Std: {beta.std():.8f}")
print(f"  Min: {beta.min():.8f}")
print(f"  Max: {beta.max():.8f}")

print(f"\nGamma (2-day HL ratio squared):")
print(f"  Mean: {gamma.mean():.8f}")
print(f"  Std: {gamma.std():.8f}")
print(f"  Min: {gamma.min():.8f}")
print(f"  Max: {gamma.max():.8f}")

print(f"\nTerm1 (sqrt(2*beta) - sqrt(beta)):")
print(f"  Mean: {term1.mean():.8f}")
print(f"  Std: {term1.std():.8f}")
print(f"  Min: {term1.min():.8f}")
print(f"  Max: {term1.max():.8f}")

print(f"\nTerm2 (sqrt(gamma)):")
print(f"  Mean: {term2.mean():.8f}")
print(f"  Std: {term2.std():.8f}")
print(f"  Min: {term2.min():.8f}")
print(f"  Max: {term2.max():.8f}")

alpha = (term1 - term2) / k

print(f"\nAlpha ((term1 - term2) / k):")
print(f"  Mean: {alpha.mean():.8f}")
print(f"  Std: {alpha.std():.8f}")
print(f"  Min: {alpha.min():.8f}")
print(f"  Max: {alpha.max():.8f}")

spread_raw = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

print(f"\nSpread before clipping:")
print(f"  Mean: {spread_raw.mean():.8f}")
print(f"  Std: {spread_raw.std():.8f}")
print(f"  Min: {spread_raw.min():.8f}")
print(f"  Max: {spread_raw.max():.8f}")
print(f"  Unique values: {spread_raw.nunique()}")

spread = spread_raw.clip(lower=0, upper=1)

print(f"\nSpread after clipping [0, 1]:")
print(f"  Mean: {spread.mean():.8f}")
print(f"  Std: {spread.std():.8f}")
print(f"  Min: {spread.min():.8f}")
print(f"  Max: {spread.max():.8f}")
print(f"  Unique values: {spread.nunique()}")

# Now test actual feature generation
print("\n" + "=" * 80)
print("TESTING ACTUAL FEATURE GENERATION")
print("=" * 80)

fe = EnhancedFinancialFeatures(feature_list=['vpin', 'cs_spread'])
features = fe.generate_features(df)

print(f"\nvpin from feature engineering:")
print(f"  Non-NaN: {features['vpin'].notna().sum()}")
print(f"  Unique: {features['vpin'].nunique()}")
print(f"  Mean: {features['vpin'].mean():.8f}")
print(f"  Std: {features['vpin'].std():.8f}")
print(f"  First 20: {features['vpin'].head(20).tolist()}")

print(f"\ncs_spread from feature engineering:")
print(f"  Non-NaN: {features['cs_spread'].notna().sum()}")
print(f"  Unique: {features['cs_spread'].nunique()}")
print(f"  Mean: {features['cs_spread'].mean():.8f}")
print(f"  Std: {features['cs_spread'].std():.8f}")
print(f"  First 20: {features['cs_spread'].head(20).tolist()}")
