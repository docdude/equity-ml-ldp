"""
Test the fixed vpin and cs_spread implementations
"""

import pandas as pd
import numpy as np
from fin_feature_preprocessing import EnhancedFinancialFeatures

# Load sample data
print("Loading AAPL data...")
df = pd.read_csv('AAPL_2022_2024.csv', parse_dates=['Date'])
df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
print(f"Data loaded: {len(df)} rows\n")

# Create feature engineer with microstructure features
print("Creating features with microstructure enabled...")
fe = EnhancedFinancialFeatures(feature_preset='minimal')
fe.feature_config['microstructure'] = True
fe.feature_config['returns'] = False
fe.feature_config['volatility'] = False
fe.feature_config['volume'] = False
fe.feature_config['momentum'] = False
fe.feature_config['trend'] = False
fe.feature_config['bollinger'] = False
fe.feature_config['price_position'] = False
fe.feature_config['entropy'] = False
fe.feature_config['regime'] = False
fe.feature_config['statistical'] = False
fe.feature_config['risk_adjusted'] = False

# Generate only microstructure features
features = fe.create_all_features(df)

print("\n" + "=" * 80)
print("VPIN STATISTICS (AFTER FIX)")
print("=" * 80)
vpin = features['vpin']
print(f"Non-NaN: {vpin.notna().sum()}")
print(f"Unique values: {vpin.nunique()}")
print(f"Mean: {vpin.mean():.6f}")
print(f"Std: {vpin.std():.6f}")
print(f"Min: {vpin.min():.6f}")
print(f"Max: {vpin.max():.6f}")
print(f"First 30 values:\n{vpin.head(30).tolist()}")

print("\n" + "=" * 80)
print("CS_SPREAD STATISTICS (AFTER FIX)")
print("=" * 80)
cs = features['cs_spread']
print(f"Non-NaN: {cs.notna().sum()}")
print(f"Unique values: {cs.nunique()}")
print(f"Mean: {cs.mean():.6f}")
print(f"Std: {cs.std():.6f}")
print(f"Min: {cs.min():.6f}")
print(f"Max: {cs.max():.6f}")
print(f"First 30 values:\n{cs.head(30).tolist()}")

# Check for constant values
print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)
if vpin.std() > 0:
    print("✅ VPIN: Has variation (std > 0)")
else:
    print("❌ VPIN: Still constant!")

if cs.std() > 0:
    print("✅ CS_SPREAD: Has variation (std > 0)")
else:
    print("❌ CS_SPREAD: Still constant!")
