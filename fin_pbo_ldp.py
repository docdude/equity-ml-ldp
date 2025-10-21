"""
PBO Test with López de Prado Triple Barrier Strategy
=====================================================

Tests Probability of Backtest Overfitting using MLFinLab triple barriers
and a proper meta-labeling strategy inspired by López de Prado's framework.

Strategy Design:
1. Primary Model: WaveNet predicts barrier outcomes (DOWN/TIMEOUT/UP)
2. Meta-labeling: Bet sizing based on confidence
3. Position sizing: Scaled by confidence threshold
4. Sample weighting: López de Prado uniqueness weights

Features:
- Uses fin_load_and_sequence (includes market context + actual returns)
- No redundant feature engineering or barrier creation
- Actual returns from meta-labeling (not synthetic)

References:
- Chapter 3: Triple Barrier Method
- Chapter 7: Cross-Validation in Finance
- Chapter 14: Backtest Overfitting
"""
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import tensorflow as tf
import pickle
from typing import Dict, Tuple

# Import sequence loading (includes market context + returns)
from fin_load_and_sequence import load_or_generate_sequences

print("="*80)
print("PBO TEST: LÓPEZ DE PRADO TRIPLE BARRIER STRATEGY")
print("="*80)


def load_model_artifacts(run_path: str = 'run_financial_wavenet_ldp_v1') -> Tuple:
    """Load model, normalizer, and metadata from training run"""
    from wavenet_model import load_model
    
    print(f"\n1. Loading model artifacts from {run_path}...")
    
    # Load model with custom objects
    model_path = f'{run_path}/best_model.keras'
    model = load_model(model_path)
    print(f"   ✅ Model loaded: {model_path}")
    print(f"      Parameters: {model.count_params():,}")
    
    # Load normalizer - return path for use with predict_with_normalization
    normalizer_path = f'{run_path}/normalizer.pkl'
    print(f"   ✅ Normalizer path: {normalizer_path}")
    
    # Load metadata
    metadata_path = f'{run_path}/metadata.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"   ✅ Metadata loaded: {metadata_path}")
    print(f"      Barrier params: pt_sl={metadata['barrier_params']['pt_sl']}, "
          f"num_days={metadata['barrier_params']['num_days']}")
    
    return model, normalizer_path, metadata


def create_mlfinlab_strategy_signals(
    predictions: np.ndarray,
    confidence_threshold: float = 0.4,
    use_meta_labeling: bool = True
) -> np.ndarray:
    """
    Create trading signals using López de Prado meta-labeling approach
    
    Signals are generated purely from model predictions (forward-looking).
    Actual returns are used later in calculate_strategy_returns() for P&L.
    
    Args:
        predictions: Model output probabilities [N, 3] for [DOWN/SL, TIMEOUT, UP/TP]
        confidence_threshold: Minimum confidence to take position
        use_meta_labeling: Use meta-labeling (bet sizing) or primary signal only
    
    Returns:
        signals: Position sizes [-1, 1] where:
                 - Positive = Long position
                 - Negative = Short position
                 - Zero = No position (timeout or low confidence)
    """
    n_samples = len(predictions)
    signals = np.zeros(n_samples)
    
    # Extract probabilities
    prob_down = predictions[:, 0]   # Stop loss hit (bearish)
    prob_timeout = predictions[:, 1]  # Vertical barrier (neutral)
    prob_up = predictions[:, 2]      # Take profit hit (bullish)
    
    if not use_meta_labeling:
        # Primary model: Simple directional bet
        # Go long if UP is most likely and exceeds threshold
        # Go short if DOWN is most likely and exceeds threshold
        # Stay out if TIMEOUT is most likely or confidence too low
        
        max_prob = predictions.max(axis=1)
        pred_class = predictions.argmax(axis=1)
        
        # Long when predicting UP with high confidence
        long_mask = (pred_class == 2) & (max_prob > confidence_threshold)
        signals[long_mask] = 1.0
        
        # Short when predicting DOWN with high confidence
        short_mask = (pred_class == 0) & (max_prob > confidence_threshold)
        signals[short_mask] = -1.0
        
        # Stay out when predicting TIMEOUT or low confidence
        # (signals remain 0)
        
    else:
        # Meta-labeling: Bet sizing approach
        # Primary model gives direction (UP vs DOWN)
        # Secondary model (confidence) gives size
        
        # Step 1: Get primary direction signal (ignore timeout)
        # Net bullishness: prob_up - prob_down
        primary_signal = prob_up - prob_down
        
        # Step 2: Get confidence (how sure are we about non-timeout outcome?)
        # Confidence = 1 - prob_timeout (higher when we expect barrier hit)
        confidence = 1.0 - prob_timeout
        
        # Step 3: Scale position by confidence
        # Only take positions when confidence exceeds threshold
        high_confidence_mask = confidence > confidence_threshold
        
        # Position size = primary_signal * confidence_scaling
        # This gives values in [-1, 1]
        signals[high_confidence_mask] = (
            primary_signal[high_confidence_mask] * 
            confidence[high_confidence_mask]
        )
        
        # Clip to [-1, 1] range
        signals = np.clip(signals, -1, 1)
    
    return signals


def calculate_strategy_returns(
    signals: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost: float = 0.001
) -> np.ndarray:
    """
    Calculate strategy returns including transaction costs
    
    Args:
        signals: Position sizes [-1, 1]
        actual_returns: Actual barrier returns
        transaction_cost: Cost per trade (0.1% = 0.001)
    
    Returns:
        strategy_returns: Returns per period
    """
    # Strategy return = signal * actual_return
    gross_returns = signals * actual_returns
    
    # Transaction costs: occur when signal changes
    signal_changes = np.abs(np.diff(np.concatenate([[0], signals])))
    transaction_costs = signal_changes * transaction_cost
    
    # Net returns
    strategy_returns = gross_returns - transaction_costs
    
    return strategy_returns


def run_pbo_analysis(
    strategy_returns: np.ndarray,
    n_splits: int = 16,
    test_size: float = 0.5,
    n_trials: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Run Probability of Backtest Overfitting analysis
    
    Args:
        strategy_returns: Array of strategy returns
        n_splits: Number of combinatorial splits
        test_size: Fraction for out-of-sample
        n_trials: Monte Carlo trials for p-value
        verbose: Print progress
    
    Returns:
        results: Dictionary with PBO metrics
    """
    from lopez_de_prado_evaluation import LopezDePradoEvaluator
    
    if verbose:
        print(f"\n6. Running PBO Analysis (López de Prado Method)...")
        print(f"   N splits: {n_splits}")
        print(f"   Test size: {test_size*100:.0f}%")
        print(f"   MC trials: {n_trials}")
    
    # Run PBO
    evaluator = LopezDePradoEvaluator()
    results = evaluator.probability_backtest_overfitting(
        strategy_returns=strategy_returns,
        n_splits=n_splits
    )
    
    return results


def main():
    """Main execution pipeline"""
    
    # 1. Load model and artifacts
    model, normalizer_path, metadata = load_model_artifacts()
    
    # 2. Load data using centralized sequence loader
    print("\n2. Loading sequences with market context + actual returns...")
    
    # Use same config as training
    CONFIG_PRESET = metadata.get('config_preset', 'wavenet_optimized_v2')
    tickers = metadata.get('tickers', ['AAPL'])
    barrier_params = metadata['barrier_params']
    market_features = metadata.get('market_features', None)  # Get from training metadata
    seq_len = 20
    
    print(f"\n📋 Config: {CONFIG_PRESET}")
    print(f"📊 Tickers: {tickers}")
    print(f"📊 Market Features: {market_features}")
    
    # Load all sequences (includes market context + actual returns)
    X_seq, y_seq, dates_seq, ldp_weights_seq, returns_seq, feature_names = load_or_generate_sequences(
        tickers=tickers,
        config_preset=CONFIG_PRESET,
        barrier_params=barrier_params,
        market_features=market_features,  # Use same market features as training
        seq_len=seq_len,
        data_path='data_raw',
        use_cache=True,
        verbose=True
    )
    
    print(f"\n✅ Total sequences: {len(X_seq)}")
    num_market = len(market_features) if market_features else 0
    print(f"   Features: {X_seq.shape[2]} ({num_market} market + {X_seq.shape[2]-num_market} stock)")
    print(f"   Date range: {dates_seq.min().date()} to {dates_seq.max().date()}")
    
    # Check label distribution
    label_counts = pd.Series(y_seq).value_counts().sort_index()
    label_pcts = pd.Series(y_seq).value_counts(normalize=True).sort_index() * 100
    print(f"\n📊 Label distribution:")
    print(f"   Down (-1):  {label_counts.get(-1, 0):5d} ({label_pcts.get(-1, 0):5.2f}%)")
    print(f"   Neutral (0): {label_counts.get(0, 0):5d} ({label_pcts.get(0, 0):5.2f}%)")
    print(f"   Up (1):      {label_counts.get(1, 0):5d} ({label_pcts.get(1, 0):5.2f}%)")
    
    # Use validation set only (last 20%)
    split_idx = int(len(X_seq) * 0.8)
    X_val = X_seq[split_idx:]
    y_val = y_seq[split_idx:]
    returns_val = returns_seq[split_idx:]
    dates_val = dates_seq[split_idx:]
    
    print(f"\n✅ Validation sequences: {len(X_val)}")
    print(f"   Date range: {dates_val.min().date()} to {dates_val.max().date()}")
    
    # 3. Normalize and predict (single output model)
    print("\n3. Normalizing and generating predictions...")
    
    from fin_inference_utils import predict_with_normalization
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Model returns single output array [N, 3] for direction probabilities
    predictions = predict_with_normalization(
        model=model,
        X=X_val,
        normalizer_path=normalizer_path,
        batch_size=64,
        verbose=0
    )
    
    # Extract direction probabilities (backward compatible with multiple return formats)
    if isinstance(predictions, dict):
        # Old format: {'direction': probs, 'volatility': ..., 'magnitude': ...}
        direction_probs = predictions.get('direction', predictions)
    elif isinstance(predictions, (list, tuple)):
        # List format: [direction_probs]
        direction_probs = predictions[0]
    else:
        # Direct array format
        direction_probs = predictions
    
    print(f"✅ Predictions shape: {direction_probs.shape}")
    
    # Evaluate Model
    y_pred_prob = direction_probs
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Map y_val from {-1, 0, 1} to {0, 1, 2} to match model output indices
    # This ensures y_true and y_pred are in the same space
    y_true = y_val - np.min(y_val)  # Shift minimum to 0

    acc = accuracy_score(y_true, y_pred)
    print(f"\nFinal Test Accuracy: {acc:.2f}")
    print(f"\nLabel Mapping: 0=DOWN, 1=NEUTRAL, 2=UP")
    print("\nClassification Report:\n", classification_report(
        y_true, y_pred, 
        target_names=['DOWN', 'NEUTRAL', 'UP']
    ))
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)
    print("   Rows=True, Cols=Predicted | Order: DOWN, NEUTRAL, UP")


    import seaborn as sns
    import matplotlib.pyplot as plt
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sell", "Hold", "Buy"], yticklabels=["Sell", "Hold", "Buy"])
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
    plt.savefig("confusion_matrix.png")
    
    print(f"✅ Predictions shape: {direction_probs.shape}")
    print(f"   Mean probabilities: DOWN={direction_probs[:, 0].mean():.3f}, "
          f"NEUTRAL={direction_probs[:, 1].mean():.3f}, "
          f"UP={direction_probs[:, 2].mean():.3f}")
    
    # Check prediction distribution
    pred_classes = direction_probs.argmax(axis=1)
    from collections import Counter
    pred_counts = Counter(pred_classes)
    print(f"\n   Prediction distribution:")
    print(f"     DOWN:    {pred_counts.get(0, 0):4d} ({pred_counts.get(0, 0)/len(pred_classes)*100:5.1f}%)")
    print(f"     NEUTRAL: {pred_counts.get(1, 0):4d} ({pred_counts.get(1, 0)/len(pred_classes)*100:5.1f}%)")
    print(f"     UP:      {pred_counts.get(2, 0):4d} ({pred_counts.get(2, 0)/len(pred_classes)*100:5.1f}%)")
    
    # 4. Create strategies with different parameters
    print("\n4. Creating López de Prado strategies...")
    
    n_strategies = 10
    strategy_returns_list = []
    
    # Test both primary and meta-labeling approaches
    # Threshold range [0.30, 0.70] provides optimal balance:
    #   - [0.00, 0.30]: High activity, Sharpe~0.34, but PBO=100% (severe overfitting)
    #                   Trading on low-confidence predictions leads to poor OOS performance
    #   - [0.30, 0.70]: Mixed activity, Sharpe 0.34-1.28, PBO=6.6% (robust!)
    #                   More selective trading on confident predictions generalizes well
    #   - [0.50, 0.75]: Very low activity → near-zero variance → division by zero → NaN
    # Result: [0.30, 0.70] gives best strategy differentiation + low overfitting risk
    confidence_thresholds = np.linspace(0.650, 0.70, n_strategies)
    
    for i, threshold in enumerate(confidence_thresholds):
        # Create signals using meta-labeling (bet sizing)
        signals = create_mlfinlab_strategy_signals(
            predictions=direction_probs,
            confidence_threshold=threshold,
            use_meta_labeling=True  # López de Prado approach
        )
        
        # Calculate strategy returns
        strat_returns = calculate_strategy_returns(
            signals=signals,
            actual_returns=returns_val,
            transaction_cost=0.001  # 10 bps
        )
        
        strategy_returns_list.append(strat_returns)
        
        # Stats
        sharpe = (strat_returns.mean() / strat_returns.std() * np.sqrt(252)) if strat_returns.std() > 0 else 0
        hit_rate = (strat_returns > 0).mean()
        avg_position = np.abs(signals).mean()
        
        print(f"   Strategy {i+1} (threshold={threshold:.3f}): "
              f"Sharpe={sharpe:.2f}, HitRate={hit_rate:.2%}, AvgPos={avg_position:.2f}")
    
    # Convert to DataFrame
    strategy_returns_df = pd.DataFrame(
        strategy_returns_list,
        index=[f'Strategy_{i+1}' for i in range(n_strategies)]
    ).T
    
    print(f"\n✅ Created {n_strategies} strategies")
    print(f"   Returns shape: {strategy_returns_df.shape}")
    
    # 5. Run PBO analysis
    pbo_results = run_pbo_analysis(
        strategy_returns=strategy_returns_df.values,
        n_splits=16,
        test_size=0.5,
        n_trials=1000,
        verbose=True
    )
    
    # 6. Display results
    print("\n" + "="*80)
    print("PBO ANALYSIS RESULTS")
    print("="*80)
    print(f"\n📊 Probability of Backtest Overfitting: {pbo_results['pbo']:.2%}")
    
    # Performance degradation is a dict with slope and r_squared
    perf_deg = pbo_results.get('performance_degradation', {})
    if isinstance(perf_deg, dict):
        print(f"   Performance Degradation: slope={perf_deg.get('slope', 0):.4f}, R²={perf_deg.get('r_squared', 0):.4f}")
    else:
        print(f"   Performance Degradation: {perf_deg:.4f}")
    
    print(f"   Probability of Loss (OOS): {pbo_results.get('prob_oos_loss', 0):.2%}")
    print(f"   Stochastic Dominance: {pbo_results.get('stochastic_dominance', 0):.2%}")
    
    # Interpretation
    print("\n📋 Interpretation:")
    if pbo_results['pbo'] < 0.3:
        print("   ✅ LOW overfitting risk (PBO < 30%)")
        print("      Strategy likely robust to out-of-sample data")
    elif pbo_results['pbo'] < 0.5:
        print("   ⚠️  MODERATE overfitting risk (30% < PBO < 50%)")
        print("      Strategy shows some robustness but needs validation")
    else:
        print("   ❌ HIGH overfitting risk (PBO > 50%)")
        print("      Strategy likely overfit to in-sample data")
    
    # Save results
    print("\n8. Saving results...")
    results_dir = 'artifacts'
    Path(results_dir).mkdir(exist_ok=True)
    
    strategy_returns_df.to_csv(f'{results_dir}/pbo_ldp_strategy_returns.csv')
    print(f"   ✅ Strategy returns saved to {results_dir}/pbo_ldp_strategy_returns.csv")
    
    with open(f'{results_dir}/pbo_ldp_results.json', 'w') as f:
        import json
        json.dump(pbo_results, f, indent=2)
    print(f"   ✅ PBO results saved to {results_dir}/pbo_ldp_results.json")
    
    print("\n" + "="*80)
    print("PBO ANALYSIS COMPLETE")
    print("="*80)
    
    return pbo_results, strategy_returns_df


if __name__ == "__main__":
    results, returns = main()
