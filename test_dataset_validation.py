"""
Dataset Validation Test Script
Validates that the training pipeline produces correctly aligned features and labels
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

print("="*80)
print("DATASET VALIDATION TEST")
print("="*80)

# Import the same components used in training
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

# Use the same configuration as training
CONFIG_PRESET = 'wavenet_optimized'
tickers = ['AAPL', 'DELL', 'JOBY', 'LCID', 'SMCI', 'NVDA', 'TSLA', 'WDAY', 'AMZN', 'AVGO', 'SPY']

print("\n" + "="*80)
print("1. LOADING DATA (Same as fin_training.py)")
print("="*80)

config = FeatureConfig.get_preset(CONFIG_PRESET)
feature_engineer = EnhancedFinancialFeatures(feature_config=config)
print(f"\n📋 Using {CONFIG_PRESET} feature preset")
feature_engineer.print_config()

all_features = []
all_labels = []
all_dates = []
all_forward_returns = []
all_tickers = []

for ticker in tickers[:3]:  # Test with just 3 tickers for speed
    print(f"\nProcessing {ticker}...")
    
    df = pd.read_parquet(f'data_raw/{ticker}.parquet')
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    # Clean data - drop rows with NaN in OHLCV columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    if len(df) < 50:
        print(f"  ⚠️  Skipping {ticker}: Only {len(df)} valid rows")
        continue
    
    # Create features
    features = feature_engineer.create_all_features(df)
    
    # Create barriers
    barriers = feature_engineer.create_dynamic_triple_barriers(df)
    
    # Combine
    combined = pd.concat([features, barriers], axis=1)
    combined = combined.dropna()
    
    if len(combined) == 0:
        print(f"  ⚠️  Skipping {ticker}: No valid samples after feature engineering")
        continue
    
    # Get exit returns from barriers
    exit_returns = combined['exit_return'].values
    
    # Add to lists
    all_features.append(combined[features.columns])
    all_labels.append(combined['label'])
    all_forward_returns.append(pd.Series(exit_returns, index=combined.index))
    all_dates.extend(combined.index)
    all_tickers.extend([ticker] * len(combined))
    
    print(f"  ✅ {ticker}: {len(combined)} samples")
    print(f"     Label distribution: {combined['label'].value_counts().to_dict()}")
    print(f"     Exit return stats: mean={exit_returns.mean():.6f}, std={exit_returns.std():.6f}")

# Combine all data
X = pd.concat(all_features)
y = pd.concat(all_labels)
forward_returns = pd.concat(all_forward_returns)
dates = pd.DatetimeIndex(all_dates)
tickers_array = np.array(all_tickers)

# Sort by date
sort_idx = dates.argsort()
X = X.iloc[sort_idx].reset_index(drop=True)
y = y.iloc[sort_idx].reset_index(drop=True)
forward_returns = forward_returns.iloc[sort_idx].reset_index(drop=True)
dates = dates[sort_idx]
tickers_array = tickers_array[sort_idx]

print("\n" + "="*80)
print("2. DATASET OVERVIEW")
print("="*80)
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Date range: {dates.min().date()} to {dates.max().date()}")

print("\n" + "="*80)
print("3. LABEL DISTRIBUTION")
print("="*80)
label_counts = y.value_counts().sort_index()
label_pcts = y.value_counts(normalize=True).sort_index() * 100
print(f"Label -1 (NEUTRAL): {label_counts.get(-1, 0):5d} ({label_pcts.get(-1, 0):5.2f}%)")
print(f"Label  0 (DOWN/SL): {label_counts.get(0, 0):5d} ({label_pcts.get(0, 0):5.2f}%)")
print(f"Label  1 (UP/TP):   {label_counts.get(1, 0):5d} ({label_pcts.get(1, 0):5.2f}%)")

print("\n" + "="*80)
print("4. CRITICAL ALIGNMENT CHECK: Labels vs Returns")
print("="*80)

print("\n🔍 Testing if labels align with actual returns:")

for label in [-1, 0, 1]:
    mask = y == label
    if mask.sum() > 0:
        returns = forward_returns[mask]
        pct_positive = (returns > 0).sum() / len(returns) * 100
        pct_negative = (returns < 0).sum() / len(returns) * 100
        
        label_name = {-1: "NEUTRAL/Timeout", 0: "DOWN/Stop Loss", 1: "UP/Take Profit"}[label]
        expected = {-1: "~neutral (mixed)", 0: "NEGATIVE", 1: "POSITIVE"}[label]
        
        print(f"\n  Label {label:2d} ({label_name}):")
        print(f"    Count: {mask.sum()}")
        print(f"    Mean return: {returns.mean():.6f} (expected: {expected})")
        print(f"    Median return: {returns.median():.6f}")
        print(f"    Positive returns: {pct_positive:.1f}%")
        print(f"    Negative returns: {pct_negative:.1f}%")
        print(f"    Min: {returns.min():.6f}, Max: {returns.max():.6f}")
        
        # Validation
        if label == 0:  # DOWN/SL should be negative
            if returns.mean() < 0 and pct_negative > 70:
                print(f"    ✅ CORRECT: DOWN labels have negative returns")
            else:
                print(f"    ❌ ERROR: DOWN labels should be mostly negative!")
        elif label == 1:  # UP/TP should be positive
            if returns.mean() > 0 and pct_positive > 70:
                print(f"    ✅ CORRECT: UP labels have positive returns")
            else:
                print(f"    ❌ ERROR: UP labels should be mostly positive!")

print("\n" + "="*80)
print("5. SEQUENCE CREATION TEST (Same as training)")
print("="*80)

seq_len = 20
X_sequences = []
y_sequences = []
dates_sequences = []
returns_sequences = []

for i in range(len(X) - seq_len):
    X_sequences.append(X.iloc[i:i+seq_len].values)
    y_sequences.append(y.iloc[i+seq_len-1])
    dates_sequences.append(dates[i+seq_len-1])
    returns_sequences.append(forward_returns.iloc[i+seq_len-1])

X_seq = np.array(X_sequences)
y_seq = np.array(y_sequences)
dates_seq = pd.DatetimeIndex(dates_sequences)
returns_seq = np.array(returns_sequences)

print(f"Sequences shape: {X_seq.shape}")
print(f"Labels shape: {y_seq.shape}")
print(f"Returns shape: {returns_seq.shape}")

print("\n" + "="*80)
print("6. TRAIN/VAL SPLIT TEST (80/20)")
print("="*80)

split_idx = int(len(X_seq) * 0.8)

X_train = X_seq[:split_idx]
X_val = X_seq[split_idx:]
y_train = y_seq[:split_idx]
y_val = y_seq[split_idx:]
returns_train = returns_seq[:split_idx]
returns_val = returns_seq[split_idx:]
dates_train = dates_seq[:split_idx]
dates_val = dates_seq[split_idx:]

print(f"\nTraining set: {len(X_train)} samples")
print(f"  Date range: {dates_train.min().date()} to {dates_train.max().date()}")
print(f"  Label distribution:")
train_label_counts = pd.Series(y_train).value_counts().sort_index()
for label in [-1, 0, 1]:
    count = train_label_counts.get(label, 0)
    pct = count / len(y_train) * 100 if len(y_train) > 0 else 0
    print(f"    Label {label:2d}: {count:5d} ({pct:5.2f}%)")

print(f"\nValidation set: {len(X_val)} samples")
print(f"  Date range: {dates_val.min().date()} to {dates_val.max().date()}")
print(f"  Label distribution:")
val_label_counts = pd.Series(y_val).value_counts().sort_index()
for label in [-1, 0, 1]:
    count = val_label_counts.get(label, 0)
    pct = count / len(y_val) * 100 if len(y_val) > 0 else 0
    print(f"    Label {label:2d}: {count:5d} ({pct:5.2f}%)")

print("\n" + "="*80)
print("7. VALIDATION SET ALIGNMENT CHECK")
print("="*80)

print("\n🔍 Checking if validation labels align with returns:")

for label in [-1, 0, 1]:
    mask = y_val == label
    if mask.sum() > 0:
        returns = returns_val[mask]
        pct_positive = (returns > 0).sum() / len(returns) * 100
        
        label_name = {-1: "NEUTRAL", 0: "DOWN", 1: "UP"}[label]
        
        print(f"\n  Label {label:2d} ({label_name}):")
        print(f"    Count: {mask.sum()}")
        print(f"    Mean return: {returns.mean():.6f}")
        print(f"    % Positive: {pct_positive:.1f}%")
        
        if label == 0:
            status = "✅" if returns.mean() < 0 else "❌"
        elif label == 1:
            status = "✅" if returns.mean() > 0 else "❌"
        else:
            status = "✅"
        
        print(f"    {status} Alignment check")

print("\n" + "="*80)
print("8. to_categorical TRANSFORMATION TEST")
print("="*80)

from tensorflow.keras.utils import to_categorical

# Show what happens when we encode labels
print("\nLabel encoding with to_categorical(y, num_classes=3):")
print("  Original label → One-hot encoding → Argmax")

for label in [-1, 0, 1]:
    encoded = to_categorical([label], num_classes=3)[0]
    argmax_idx = encoded.argmax()
    print(f"  Label {label:2d} → {encoded} → argmax={argmax_idx}")

print("\nThis means the model will learn:")
print("  Output index 0 = Label  0 (DOWN/SL)")
print("  Output index 1 = Label  1 (UP/TP)")
print("  Output index 2 = Label -1 (NEUTRAL)")

# Encode validation labels
y_val_encoded = to_categorical(y_val, num_classes=3)

print(f"\nEncoded validation labels shape: {y_val_encoded.shape}")
print(f"  (samples, classes) = ({y_val_encoded.shape[0]}, {y_val_encoded.shape[1]})")

# Check encoding distribution
print("\nClass distribution in encoded labels:")
for idx in range(3):
    count = (y_val_encoded.argmax(axis=1) == idx).sum()
    pct = count / len(y_val_encoded) * 100
    label_value = {0: 0, 1: 1, 2: -1}[idx]
    print(f"  Output index {idx} (label {label_value:2d}): {count:5d} ({pct:5.2f}%)")

print("\n" + "="*80)
print("9. SAMPLE INSPECTION")
print("="*80)

# Show first 10 validation samples
print("\nFirst 10 validation samples:")
print(f"{'Idx':<6} {'Date':<12} {'Label':<7} {'Return':<10} {'Encoded':<20}")
print("-" * 60)
for i in range(min(10, len(y_val))):
    encoded_str = f"{y_val_encoded[i]}"
    print(f"{i:<6} {str(dates_val[i].date()):<12} {y_val[i]:<7} {returns_val[i]:<10.6f} {encoded_str:<20}")

print("\n" + "="*80)
print("10. FEATURE STATISTICS")
print("="*80)

print(f"\nValidation features (X_val):")
print(f"  Shape: {X_val.shape} (samples, timesteps, features)")
print(f"  Mean: {X_val.mean():.4f}")
print(f"  Std: {X_val.std():.4f}")
print(f"  Min: {X_val.min():.4f}")
print(f"  Max: {X_val.max():.4f}")

print(f"\nFeature ranges per timestep (showing first 5 timesteps):")
for t in range(min(5, X_val.shape[1])):
    timestep_data = X_val[:, t, :]
    print(f"  Timestep {t}: mean={timestep_data.mean():.4f}, "
          f"std={timestep_data.std():.4f}, "
          f"min={timestep_data.min():.4f}, "
          f"max={timestep_data.max():.4f}")

print("\n" + "="*80)
print("11. CORRELATION TEST: Labels vs Returns")
print("="*80)

# Calculate correlation between label and return
label_numeric = np.where(y_val == 1, 1, np.where(y_val == 0, -1, 0))
correlation = np.corrcoef(label_numeric, returns_val)[0, 1]

print(f"\nCorrelation between label and actual return: {correlation:.4f}")
print("  (Label mapped: UP=+1, DOWN=-1, NEUTRAL=0)")

if correlation > 0.3:
    print("  ✅ GOOD: Strong positive correlation")
elif correlation > 0.1:
    print("  ⚠️  MODERATE: Weak positive correlation")
else:
    print("  ❌ BAD: Very weak or negative correlation - DATA ISSUE!")

# Directional accuracy
pred_direction = label_numeric
actual_direction = np.sign(returns_val)
directional_acc = (pred_direction == actual_direction).mean()

print(f"\nDirectional accuracy (if we just used labels as predictions): {directional_acc:.2%}")
print("  (This should be >> 50% if labels are correctly aligned)")

if directional_acc > 0.65:
    print("  ✅ GOOD: Labels are well-aligned with return direction")
elif directional_acc > 0.55:
    print("  ⚠️  MODERATE: Some alignment but not strong")
else:
    print("  ❌ BAD: Labels barely better than random - SERIOUS DATA ISSUE!")

print("\n" + "="*80)
print("12. SUMMARY & DIAGNOSIS")
print("="*80)

issues_found = []
warnings_found = []

# Check 1: Label-return alignment
for label in [0, 1]:
    mask = y_val == label
    if mask.sum() > 0:
        mean_ret = returns_val[mask].mean()
        if label == 0 and mean_ret >= 0:
            issues_found.append(f"DOWN labels (0) have non-negative mean return: {mean_ret:.6f}")
        elif label == 1 and mean_ret <= 0:
            issues_found.append(f"UP labels (1) have non-positive mean return: {mean_ret:.6f}")

# Check 2: Overall correlation
if correlation < 0.1:
    issues_found.append(f"Very low correlation between labels and returns: {correlation:.4f}")
elif correlation < 0.3:
    warnings_found.append(f"Moderate correlation between labels and returns: {correlation:.4f}")

# Check 3: Directional accuracy
if directional_acc < 0.55:
    issues_found.append(f"Directional accuracy barely above random: {directional_acc:.2%}")
elif directional_acc < 0.65:
    warnings_found.append(f"Directional accuracy could be better: {directional_acc:.2%}")

# Check 4: Class imbalance
neutral_pct = (y_val == -1).sum() / len(y_val) * 100
if neutral_pct > 60:
    warnings_found.append(f"High neutral class percentage: {neutral_pct:.1f}%")

print("\n🔍 DIAGNOSTIC RESULTS:")

if issues_found:
    print(f"\n❌ CRITICAL ISSUES FOUND ({len(issues_found)}):")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✅ No critical issues found")

if warnings_found:
    print(f"\n⚠️  WARNINGS ({len(warnings_found)}):")
    for i, warning in enumerate(warnings_found, 1):
        print(f"  {i}. {warning}")
else:
    print("\n✅ No warnings")

if not issues_found and not warnings_found:
    print("\n🎉 DATASET APPEARS TO BE CORRECTLY ALIGNED!")
    print("   The training pipeline should work correctly.")
else:
    print("\n🔧 RECOMMENDATIONS:")
    if issues_found:
        print("   1. Check barrier labeling logic in create_dynamic_triple_barriers()")
        print("   2. Verify exit_return calculation")
        print("   3. Review triple barrier parameters (TP/SL thresholds)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
