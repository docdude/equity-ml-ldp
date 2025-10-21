#!/usr/bin/env python3
"""
Verify that downloaded market data matches MLFinance feature matrix transformations
"""

import pandas as pd
import numpy as np

print("="*80)
print("MARKET DATA TRANSFORMATION VERIFICATION")
print("="*80)

# Load MLFinance feature matrix
mlf_features = pd.read_parquet('MLFinance/Data/AAPL_feature_matrix.parquet')

# Load downloaded market data (with date as index and lowercase columns)
def load_market_data(filepath):
    df = pd.read_parquet(filepath)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    # Rename columns to match standard format
    df.columns = df.columns.str.capitalize()
    return df

spy_data = load_market_data('data_raw/^GSPC.parquet')
vix_data = load_market_data('data_raw/^VIX.parquet')
fvx_data = load_market_data('data_raw/^FVX.parquet')
tyx_data = load_market_data('data_raw/^TYX.parquet')
gold_data = load_market_data('data_raw/GC=F.parquet')
jpyx_data = load_market_data('data_raw/JPY=X.parquet')

print(f"\nMLFinance date range: {mlf_features.index.min()} to {mlf_features.index.max()}")
print(f"Downloaded SPY range:  {spy_data.index.min()} to {spy_data.index.max()}")

# Find common date range for comparison
common_dates = mlf_features.index.intersection(spy_data.index)
print(f"\nCommon dates for comparison: {len(common_dates)} days")

# Test each transformation
results = {}

print("\n" + "="*80)
print("1. ^GSPC (S&P 500) - Testing daily returns transformation")
print("="*80)

spy_returns = spy_data['Close'].pct_change(1)
spy_returns_aligned = spy_returns.loc[common_dates]
mlf_spy = mlf_features['^GSPC'].loc[common_dates]

# Compare
correlation = spy_returns_aligned.corr(mlf_spy)
mae = (spy_returns_aligned - mlf_spy).abs().mean()

print(f"Correlation: {correlation:.6f}")
print(f"Mean Absolute Error: {mae:.8f}")
print(f"\nSample comparison (first 10 dates):")
comparison = pd.DataFrame({
    'Downloaded_Returns': spy_returns_aligned.head(10),
    'MLFinance_Feature': mlf_spy.head(10),
    'Difference': (spy_returns_aligned - mlf_spy).head(10)
})
print(comparison)

results['^GSPC'] = {
    'transformation': 'pct_change(1)',
    'correlation': correlation,
    'mae': mae,
    'match': correlation > 0.99 and mae < 0.0001
}

print("\n" + "="*80)
print("2. ^VIX - Testing VIX/100 transformation")
print("="*80)

vix_normalized = vix_data['Close'] / 100
vix_normalized_aligned = vix_normalized.loc[common_dates]
mlf_vix = mlf_features['^VIX'].loc[common_dates]

correlation = vix_normalized_aligned.corr(mlf_vix)
mae = (vix_normalized_aligned - mlf_vix).abs().mean()

print(f"Correlation: {correlation:.6f}")
print(f"Mean Absolute Error: {mae:.8f}")
print(f"\nSample comparison (first 10 dates):")
comparison = pd.DataFrame({
    'Downloaded_VIX': vix_data['Close'].loc[common_dates].head(10),
    'Normalized': vix_normalized_aligned.head(10),
    'MLFinance_Feature': mlf_vix.head(10),
    'Difference': (vix_normalized_aligned - mlf_vix).head(10)
})
print(comparison)

results['^VIX'] = {
    'transformation': 'Close / 100',
    'correlation': correlation,
    'mae': mae,
    'match': correlation > 0.99 and mae < 0.001
}

print("\n" + "="*80)
print("3. ^FVX (5yr Treasury) - Testing daily returns transformation")
print("="*80)

fvx_returns = fvx_data['Close'].pct_change(1)
fvx_returns_aligned = fvx_returns.loc[common_dates]
mlf_fvx = mlf_features['^FVX'].loc[common_dates]

correlation = fvx_returns_aligned.corr(mlf_fvx)
mae = (fvx_returns_aligned - mlf_fvx).abs().mean()

print(f"Correlation: {correlation:.6f}")
print(f"Mean Absolute Error: {mae:.8f}")
print(f"\nSample comparison (first 10 dates):")
comparison = pd.DataFrame({
    'Downloaded_Returns': fvx_returns_aligned.head(10),
    'MLFinance_Feature': mlf_fvx.head(10),
    'Difference': (fvx_returns_aligned - mlf_fvx).head(10)
})
print(comparison)

results['^FVX'] = {
    'transformation': 'pct_change(1)',
    'correlation': correlation,
    'mae': mae,
    'match': correlation > 0.99 and mae < 0.0001
}

print("\n" + "="*80)
print("4. ^TYX (30yr Treasury) - Testing daily returns transformation")
print("="*80)

tyx_returns = tyx_data['Close'].pct_change(1)
tyx_returns_aligned = tyx_returns.loc[common_dates]
mlf_tyx = mlf_features['^TYX'].loc[common_dates]

correlation = tyx_returns_aligned.corr(mlf_tyx)
mae = (tyx_returns_aligned - mlf_tyx).abs().mean()

print(f"Correlation: {correlation:.6f}")
print(f"Mean Absolute Error: {mae:.8f}")
print(f"\nSample comparison (first 10 dates):")
comparison = pd.DataFrame({
    'Downloaded_Returns': tyx_returns_aligned.head(10),
    'MLFinance_Feature': mlf_tyx.head(10),
    'Difference': (tyx_returns_aligned - mlf_tyx).head(10)
})
print(comparison)

results['^TYX'] = {
    'transformation': 'pct_change(1)',
    'correlation': correlation,
    'mae': mae,
    'match': correlation > 0.99 and mae < 0.0001
}

print("\n" + "="*80)
print("5. GC=F (Gold) - Testing daily returns transformation")
print("="*80)

gold_returns = gold_data['Close'].pct_change(1)
gold_returns_aligned = gold_returns.loc[common_dates]
mlf_gold = mlf_features['GC=F'].loc[common_dates]

correlation = gold_returns_aligned.corr(mlf_gold)
mae = (gold_returns_aligned - mlf_gold).abs().mean()

print(f"Correlation: {correlation:.6f}")
print(f"Mean Absolute Error: {mae:.8f}")
print(f"\nSample comparison (first 10 dates):")
comparison = pd.DataFrame({
    'Downloaded_Returns': gold_returns_aligned.head(10),
    'MLFinance_Feature': mlf_gold.head(10),
    'Difference': (gold_returns_aligned - mlf_gold).head(10)
})
print(comparison)

results['GC=F'] = {
    'transformation': 'pct_change(1)',
    'correlation': correlation,
    'mae': mae,
    'match': correlation > 0.99 and mae < 0.0001
}

print("\n" + "="*80)
print("6. JPY=X (USD/JPY) - Testing daily returns transformation")
print("="*80)

jpyx_returns = jpyx_data['Close'].pct_change(1)
jpyx_returns_aligned = jpyx_returns.loc[common_dates]
mlf_jpyx = mlf_features['JPY=X'].loc[common_dates]

correlation = jpyx_returns_aligned.corr(mlf_jpyx)
mae = (jpyx_returns_aligned - mlf_jpyx).abs().mean()

print(f"Correlation: {correlation:.6f}")
print(f"Mean Absolute Error: {mae:.8f}")
print(f"\nSample comparison (first 10 dates):")
comparison = pd.DataFrame({
    'Downloaded_Returns': jpyx_returns_aligned.head(10),
    'MLFinance_Feature': mlf_jpyx.head(10),
    'Difference': (jpyx_returns_aligned - mlf_jpyx).head(10)
})
print(comparison)

results['JPY=X'] = {
    'transformation': 'pct_change(1)',
    'correlation': correlation,
    'mae': mae,
    'match': correlation > 0.99 and mae < 0.0001
}

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

all_match = True
for feature, result in results.items():
    status = "✓ MATCH" if result['match'] else "✗ MISMATCH"
    print(f"{feature:10s} | {result['transformation']:20s} | Corr: {result['correlation']:.6f} | MAE: {result['mae']:.8f} | {status}")
    if not result['match']:
        all_match = False

print("\n" + "="*80)
if all_match:
    print("✓ ALL TRANSFORMATIONS VERIFIED!")
    print("\nConfirmed transformations:")
    print("  - ^GSPC, ^FVX, ^TYX, GC=F, JPY=X: Use pct_change(1)")
    print("  - ^VIX: Use Close / 100")
else:
    print("✗ Some transformations did not match. Review output above.")
print("="*80)
