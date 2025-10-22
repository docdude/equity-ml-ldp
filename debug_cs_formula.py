"""
Debug cs_spread with the correct reference formula
"""

import pandas as pd
import numpy as np

# Load sample data
print("Loading AAPL data...")
df = pd.read_csv('AAPL_2022_2024.csv', parse_dates=['Date'])
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
print(f"Data loaded: {len(df)} rows\n")

high = df['High']
low = df['Low']

# Constants (from reference)
CONST = 3 - 2 * np.sqrt(2)
print(f"CONST = {CONST}")

# Beta: Single-day high-low variance proxy
beta = np.log(high / low) ** 2

# Gamma: Two-day high-low variance proxy
h2 = high.rolling(2).max()
l2 = low.rolling(2).min()
gamma = np.log(h2 / l2) ** 2

print(f"\nBeta stats:")
print(f"  Mean: {beta.mean():.8f}")
print(f"  Min: {beta.min():.8f}")
print(f"  Max: {beta.max():.8f}")

print(f"\nGamma stats:")
print(f"  Mean: {gamma.mean():.8f}")
print(f"  Min: {gamma.min():.8f}")
print(f"  Max: {gamma.max():.8f}")

# Reference formula: alpha = (sqrt(2*beta) - sqrt(beta))/CONST - sqrt(gamma/CONST)
term1 = (np.sqrt(2 * beta) - np.sqrt(beta)) / CONST
term2 = np.sqrt(gamma / CONST)

print(f"\nTerm1 = (sqrt(2*beta) - sqrt(beta))/CONST:")
print(f"  Mean: {term1.mean():.8f}")
print(f"  Min: {term1.min():.8f}")
print(f"  Max: {term1.max():.8f}")

print(f"\nTerm2 = sqrt(gamma/CONST):")
print(f"  Mean: {term2.mean():.8f}")
print(f"  Min: {term2.min():.8f}")
print(f"  Max: {term2.max():.8f}")

alpha = term1 - term2

print(f"\nAlpha = term1 - term2:")
print(f"  Mean: {alpha.mean():.8f}")
print(f"  Std: {alpha.std():.8f}")
print(f"  Min: {alpha.min():.8f}")
print(f"  Max: {alpha.max():.8f}")
print(f"  Positive values: {(alpha > 0).sum()}")
print(f"  Negative values: {(alpha < 0).sum()}")

# Spread
spread_raw = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

print(f"\nSpread (before clipping):")
print(f"  Mean: {spread_raw.mean():.8f}")
print(f"  Std: {spread_raw.std():.8f}")
print(f"  Min: {spread_raw.min():.8f}")
print(f"  Max: {spread_raw.max():.8f}")
print(f"  Positive values: {(spread_raw > 0).sum()}")
print(f"  Zero or negative: {(spread_raw <= 0).sum()}")

spread = spread_raw.clip(lower=0)

print(f"\nSpread (after clipping to 0):")
print(f"  Mean: {spread.mean():.8f}")
print(f"  Std: {spread.std():.8f}")
print(f"  Min: {spread.min():.8f}")
print(f"  Max: {spread.max():.8f}")
print(f"  Unique: {spread.nunique()}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("The issue is that term2 >> term1, making alpha always negative.")
print("This happens because gamma (2-day variance) > 2*beta (single-day variance)")
print("which is expected, but the formula subtracts sqrt(gamma/CONST).")
print("\nThe reference implementation might be handling this differently,")
print("or there's missing data cleaning (overnight returns adjustment, etc.)")
