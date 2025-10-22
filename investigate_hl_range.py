"""
Investigate hl_range outliers
"""
import numpy as np
import pandas as pd

df = pd.read_parquet('data_raw/AAPL.parquet')
hl_range_raw = (df['high'] - df['low']) / df['Close']

print("HL Range statistics:")
print(f"  Mean: {hl_range_raw.mean():.6f}")
print(f"  Median: {hl_range_raw.median():.6f}")
print(f"  Std: {hl_range_raw.std():.6f}")
print(f"  Min: {hl_range_raw.min():.6f}")
print(f"  Max: {hl_range_raw.max():.6f}")
print(f"\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]:
    val = np.percentile(hl_range_raw, p)
    print(f"  {p:5.1f}%: {val:.6f}")

print(f"\nOutliers (> 99th percentile):")
p99 = np.percentile(hl_range_raw, 99)
outliers = hl_range_raw[hl_range_raw > p99]
print(f"  Count: {len(outliers)}")
print(f"  Values: {sorted(outliers.values, reverse=True)[:10]}")

# Check dates
print(f"\nTop 10 extreme hl_range days:")
top_days = hl_range_raw.nlargest(10)
for date, value in top_days.items():
    print(f"  {date.date()}: {value:.6f} ({value*100:.2f}%)")
