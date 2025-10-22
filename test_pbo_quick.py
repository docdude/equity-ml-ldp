"""
Quick PBO Test with Real Model Predictions
Tests only the PBO calculation with strategy returns from trained model
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tensorflow as tf
print("="*80)
print("PBO TEST WITH REAL MODEL PREDICTIONS")
print("="*80)

# 1. Load trained model and normalizer
print("\n1. Loading trained CNN-LSTM model and normalizer...")
try:
    from fin_inference_utils import load_model_and_normalizer
    model, normalizer_path = load_model_and_normalizer(
        model_path='run_financial_wavenet_v1',
        model_file='best_model.keras'
    )
except Exception as e:
    print(f"❌ Could not load model or normalizer: {e}")
    print("   Run fin_training.py first to train a model")
    sys.exit(1)

# 2. Load data for predictions
print("\n2. Loading validation data...")
from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig

tickers = ['AAPL', 'DELL', 'JOBY', 'LCID', 'SMCI', 'NVDA', 'TSLA', 'WDAY', 'AMZN', 'AVGO', 'SPY']
#tickers = ['AAPL']
CONFIG_PRESET = 'wavenet_optimized'
config = FeatureConfig.get_preset(CONFIG_PRESET)
feature_engineer = EnhancedFinancialFeatures(feature_config=config)
print(f"\n📋 Using {CONFIG_PRESET} feature preset (18 features)")
print("   Selected by López de Prado analysis (MDI/MDA/SFI/Orthogonal)")
feature_engineer.print_config()

all_features = []
all_labels = []
all_dates = []
all_prices = []
all_tickers = []

for ticker in tickers:
    df = pd.read_parquet(f'data_raw/{ticker}.parquet')
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    # Clean data - drop rows with NaN in OHLCV columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    features = feature_engineer.create_all_features(df)
    barriers = feature_engineer.create_dynamic_triple_barriers(df)
    combined = pd.concat([features, barriers], axis=1).dropna()
    prices_aligned = df.loc[combined.index, ['Close']]
    
    # Store features and use exit_return from barriers (aligned with training/evaluation!)
    ticker_features = combined[features.columns].copy()
    
    all_labels.append(combined['label'])
    all_prices.append(prices_aligned)
    all_dates.extend(combined.index)
    
    # Track ticker for each row
    all_tickers.extend([ticker] * len(combined))
    
    # ✅ USE EXIT_RETURN FROM BARRIERS (López de Prado / MLFinLab method)
    # This is the actual return at barrier touch, NOT fixed 5-day returns
    # Aligns with fin_training.py and fin_model_evaluation.py
    ticker_features['exit_return'] = combined['exit_return'].values
    ticker_features['exit_day'] = combined['exit_day'].values  # For analysis
    ticker_features['ticker'] = ticker
    
    all_features.append(ticker_features)

# Combine and sort
X = pd.concat(all_features)
y = pd.concat(all_labels)
prices = pd.concat(all_prices)
dates = pd.DatetimeIndex(all_dates)
tickers = pd.Series(all_tickers)

sort_idx = dates.argsort()
X = X.iloc[sort_idx].reset_index(drop=True)
y = y.iloc[sort_idx].reset_index(drop=True)
prices = prices.iloc[sort_idx].reset_index(drop=True)
dates = dates[sort_idx]
tickers = tickers.iloc[sort_idx].reset_index(drop=True)

print(f"✅ Loaded {X.shape[0]} samples, {X.shape[1]} features (including ticker)")

# Check label distribution
label_counts = y.value_counts().sort_index()
label_pcts = y.value_counts(normalize=True).sort_index() * 100
print(f"\n📊 Label distribution:")
print(f"   Down (0):    {label_counts.get(0, 0):5d} ({label_pcts.get(0, 0):5.2f}%)")
print(f"   Up (1):      {label_counts.get(1, 0):5d} ({label_pcts.get(1, 0):5.2f}%)")
print(f"   Neutral (-1): {label_counts.get(-1, 0):5d} ({label_pcts.get(-1, 0):5.2f}%)")

# 3. Create sequences (remove ticker column for model input)
print("\n3. Creating sequences...")
seq_len = 20

# Separate ticker and exit from model input features
model_feature_cols = [col for col in X.columns 
                      if col not in ['ticker', 'exit_return', 'exit_day']]
X_features = X[model_feature_cols]
print(f"   Using {len(model_feature_cols)} features for model input")

X_sequences = []
for i in range(len(X_features) - seq_len):
    X_sequences.append(X_features.iloc[i:i+seq_len].values)
X_seq = np.array(X_sequences)

# Use only validation set (last 20%)
val_split = int(len(X_seq) * 0.8)
X_val = X_seq[val_split:]
y_val = y.iloc[val_split + seq_len:val_split + seq_len + len(X_val)]
dates_val = dates[val_split + seq_len:val_split + seq_len + len(X_val)]

print(f"✅ Validation set: {len(X_val)} sequences (BEFORE normalization)")
print(f"   Date range: {dates_val.min().date()} to {dates_val.max().date()}")

# 3.5 & 4. Normalize and predict (using inference utilities)
print("\n3.5. Normalizing features and generating predictions...")
from fin_inference_utils import predict_with_normalization
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

predictions = predict_with_normalization(
    model=model,
    X=X_val,
    normalizer_path=normalizer_path,
    batch_size=32,
    verbose=0
)

# Evaluate Model
y_pred_prob = predictions['direction']
y_pred = np.argmax(y_pred_prob, axis=1)

# Map y_val from {-1, 0, 1} to {0, 1, 2} to match model output indices
# This ensures y_true and y_pred are in the same space
y_true = y_val.values - np.min(y_val.values)  # Shift minimum to 0

acc = accuracy_score(y_true, y_pred)
print(f"\nFinal Test Accuracy: {acc:.2f}")
print(f"\nLabel Mapping: 0=NEUTRAL, 1=DOWN, 2=UP")
print("\nClassification Report:\n", classification_report(
    y_true, y_pred, 
    target_names=['NEUTRAL', 'DOWN', 'UP']
))
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)
print("   Rows=True, Cols=Predicted | Order: NEUTRAL, DOWN, UP")


import seaborn as sns
import matplotlib.pyplot as plt
# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Hold", "Sell", "Buy"], yticklabels=["Hold", "Sell", "Buy"])
plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
plt.savefig("confusion_matrix.png")

if isinstance(predictions, dict):
    direction_probs = predictions['direction']
else:
    direction_probs = predictions[0] if isinstance(predictions, list) else predictions

print(f"✅ Predictions shape: {direction_probs.shape}")
# Model trained with to_categorical on labels {-1, 0, 1}:
#   Label  0 (DOWN/SL)  → [0,1,0] → model output index 1
#   Label  1 (UP/TP)    → [0,0,1] → model output index 2
#   Label -1 (NEUTRAL)  → [1,0,0] → model output index 0
print(f"   Classes: Down={direction_probs[:, 1].mean():.3f}, "
      f"Up={direction_probs[:, 2].mean():.3f}, "
      f"Neutral={direction_probs[:, 0].mean():.3f}")

# DIAGNOSTIC: Check prediction alignment with returns
print(f"\n🔍 PREDICTION DIAGNOSTICS:")
prob_down = direction_probs[:, 1]
prob_up = direction_probs[:, 2]
prob_neutral = direction_probs[:, 0]

# ✅ USE EXIT_RETURN FROM BARRIERS
exit_returns_all = X['exit_return'].values
val_start_idx = val_split + seq_len
forward_returns_diag = exit_returns_all[val_start_idx:val_start_idx + len(X_val)]
direction_probs = direction_probs[:len(forward_returns_diag)]

# Recalculate probabilities after truncation
prob_down = direction_probs[:, 1]
prob_up = direction_probs[:, 2]
prob_neutral = direction_probs[:, 0]

# Check correlation
net_prediction = prob_up - prob_down
correlation = np.corrcoef(net_prediction, forward_returns_diag)[0, 1]
print(f"   Net signal (up-down) vs returns correlation: {correlation:.4f}")

# Directional accuracy
pred_direction = np.sign(net_prediction)
actual_direction = np.sign(forward_returns_diag)
directional_acc = (pred_direction == actual_direction).mean()
print(f"   Directional accuracy: {directional_acc:.2%}")

# Mean returns by predicted class
pred_class = direction_probs.argmax(axis=1)
print(f"   Mean return when predicting DOWN (idx 1):    {forward_returns_diag[pred_class == 1].mean():.6f}")
print(f"   Mean return when predicting UP (idx 2):      {forward_returns_diag[pred_class == 2].mean():.6f}")
print(f"   Mean return when predicting NEUTRAL (idx 0): {forward_returns_diag[pred_class == 0].mean():.6f}")

# 5. Create strategy returns from predictions
print("\n5. Creating strategy returns (10 strategies with different thresholds)...")
n_strategies = 10
strategy_returns = []

# Use forward_returns from diagnostics (already calculated and aligned)
forward_returns = forward_returns_diag

print(f"   Aligned {len(forward_returns)} predictions with actual returns")
print(f"   Forward returns range: [{forward_returns.min():.4f}, {forward_returns.max():.4f}]")
print(f"   Forward returns mean: {forward_returns.mean():.6f}, std: {forward_returns.std():.6f}")
print(f"   Returns > 1.0 (100%): {(forward_returns > 1.0).sum()} ({(forward_returns > 1.0).sum()/len(forward_returns)*100:.2f}%)")
print(f"   Returns < -0.5 (-50%): {(forward_returns < -0.5).sum()} ({(forward_returns < -0.5).sum()/len(forward_returns)*100:.2f}%)")

# Create strategies with different confidence thresholds
# Since softmax outputs probabilities that sum to 1, we use RELATIVE confidence
# Strategy: Take position based on highest probability class, weighted by confidence
thresholds = np.linspace(0.0, 0.3, n_strategies)  # Margin above 1/3 (random guess)

for threshold in thresholds:
    # Model output: [P(down), P(up), P(neutral)] from softmax
    prob_down = direction_probs[:, 1]     # P(label=0) = DOWN/SL
    prob_up = direction_probs[:, 2]       # P(label=1) = UP/TP
    prob_neutral = direction_probs[:, 0]  # P(label=-1) = NEUTRAL
    
    # Get the winning class and its probability
    pred_class = direction_probs.argmax(axis=1)
    max_prob = direction_probs.max(axis=1)
    
    # Calculate confidence margin: how much better is the top choice vs random (1/3)?
    confidence_margin = max_prob - (1.0 / 3.0)
    
    # Only take positions when confidence margin exceeds threshold
    # Position sizing: Use the probability difference (up - down) scaled by confidence
    net_signal = prob_up - prob_down  # Positive = bullish, Negative = bearish
    
    positions = np.where(
        confidence_margin > threshold,
        net_signal,  # Take position proportional to net signal when confident
        0  # No position when not confident enough
    )
    
    # Apply positions to actual forward returns
    returns = positions * forward_returns
    strategy_returns.append(returns)

# Convert to DataFrame (observations x strategies)
returns_df = pd.DataFrame(strategy_returns).T
returns_df.columns = [f'Threshold_{t:.2f}' for t in thresholds]

print(f"✅ Strategy returns matrix: {returns_df.shape}")
print(f"   Mean return per strategy: {returns_df.mean().mean():.6f}")
print(f"   Mean Sharpe ratio: {(returns_df.mean() / returns_df.std() * np.sqrt(252)).mean():.4f}")

# 6. Run PBO analysis
print("\n6. Running PBO Analysis...")
print("-"*40)

from lopez_de_prado_evaluation import LopezDePradoEvaluator

evaluator = LopezDePradoEvaluator()

# Check return statistics (no clipping)
print(f"   Return range: [{returns_df.values.min():.4f}, {returns_df.values.max():.4f}]")
print(f"   Mean return: {returns_df.values.mean():.6f}")
print(f"   Std return: {returns_df.values.std():.6f}")

# Run PBO with more splits for better statistics
pbo_result = evaluator.probability_backtest_overfitting(
    strategy_returns=returns_df.values,
    n_splits=16  # Use 16 splits for CSCV (default)
)

print("\n📊 PBO RESULTS:")
print(f"   PBO: {pbo_result['pbo']:.4f}")
print(f"   Probability of OOS Loss: {pbo_result.get('prob_oos_loss', 0):.4f}")
print(f"   Mean Logit (λ): {pbo_result.get('mean_logit', 0):.4f}")

# Check for degradation metrics
if 'performance_degradation' in pbo_result:
    perf_deg = pbo_result['performance_degradation']
    print("   Performance Degradation:")
    print(f"      Slope: {perf_deg['slope']:.4f}")
    print(f"      R²: {perf_deg['r_squared']:.4f}")

# Interpretation
print(f"\n💡 INTERPRETATION:")
if pbo_result['pbo'] < 0.3:
    print("   ✅ EXCELLENT: Very low overfitting risk (PBO < 0.3)")
    print("   → Model is likely to generalize well to new data")
elif pbo_result['pbo'] < 0.5:
    print("   ✅ GOOD: Acceptable overfitting risk (PBO < 0.5)")
    print("   → Model shows reasonable out-of-sample consistency")
elif pbo_result['pbo'] < 0.7:
    print("   ⚠️  MODERATE: Elevated overfitting risk (PBO 0.5-0.7)")
    print("   → Model may not generalize well, needs validation")
else:
    print("   ❌ HIGH: Severe overfitting detected (PBO > 0.7)")
    print("   → Model is likely overfit to training data")

# Show all available metrics
print(f"\n📈 ALL METRICS:")
for key, value in pbo_result.items():
    if isinstance(value, (int, float, np.number)):
        print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")

print("\n" + "="*80)
print("PBO TEST COMPLETE")
print("="*80)

print("\n💾 SAVED RESULTS:")
Path('artifacts').mkdir(exist_ok=True)
returns_df.to_csv('artifacts/pbo_strategy_returns.csv')
print(f"   artifacts/pbo_strategy_returns.csv - Strategy returns matrix")

# Save PBO results
import json
with open('artifacts/pbo_results.json', 'w') as f:
    # Convert numpy types to native Python types
    result_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in pbo_result.items()}
    json.dump(result_clean, f, indent=2)
print(f"   artifacts/pbo_results.json - PBO metrics")

print("\n✅ Use these results to assess model production readiness!")
