"""
Inference Utilities for Financial ML Models
Provides consistent interface for model predictions with normalization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from tensorflow import keras

from fin_utils import normalize_for_inference


def load_model_and_normalizer(
    model_path: str = 'run_financial_wavenet_v1',
    model_file: str = 'best_model.keras'
) -> Tuple[keras.Model, str]:
    """
    Load trained model and get path to normalizer.
    
    Args:
        model_path: Directory containing model and normalizer
        model_file: Model filename
    
    Returns:
        (model, normalizer_path)
    """
    from fin_model import load_model_with_custom_objects
    
    model_full_path = Path(model_path) / model_file
    normalizer_path = Path(model_path) / 'feature_scaler.pkl'
    
    # Load model
    model = load_model_with_custom_objects(str(model_full_path))
    
    # Check normalizer exists
    if not normalizer_path.exists():
        raise FileNotFoundError(
            f"Normalizer not found at {normalizer_path}. "
            f"Train model with fin_training.py to generate normalizer."
        )
    
    print(f"✅ Loaded model: {model.count_params():,} parameters")
    print(f"✅ Normalizer available: {normalizer_path}")
    
    return model, str(normalizer_path)


def predict_with_normalization(
    model: keras.Model,
    X: np.ndarray,
    normalizer_path: str,
    batch_size: int = 32,
    verbose: int = 0
) -> Dict[str, np.ndarray]:
    """
    Make predictions with proper normalization.
    
    Args:
        model: Trained Keras model
        X: Features of shape (n_samples, n_timesteps, n_features)
        normalizer_path: Path to saved normalizer
        batch_size: Batch size for prediction
        verbose: Verbosity level
    
    Returns:
        Dictionary with predictions {'direction': probs, 'volatility': values, ...}
    """
    # Normalize features
    print(f"\n📊 Normalizing {len(X)} sequences for inference...")
    X_normalized = normalize_for_inference(X, normalizer_path)
    
    # Make predictions
    print(f"🔮 Generating predictions...")
    predictions = model.predict(X_normalized, batch_size=batch_size, verbose=verbose)
    
    # Format predictions
    if isinstance(predictions, dict):
        pred_dict = predictions
    elif isinstance(predictions, list):
        # Multi-output model
        pred_dict = {
            'direction': predictions[0],
            'volatility': predictions[1] if len(predictions) > 1 else None,
            'magnitude': predictions[2] if len(predictions) > 2 else None
        }
    else:
        # Single output
        pred_dict = {'direction': predictions}
    
    print(f"✅ Predictions generated: {len(predictions)} samples")
    
    return pred_dict


def create_trading_signals(
    direction_probs: np.ndarray,
    confidence_threshold: float = 0.5,
    scale_by_confidence: bool = True
) -> np.ndarray:
    """
    Convert direction probabilities to trading signals.
    
    Args:
        direction_probs: Probabilities array (n_samples, 3) [down, neutral, up]
        confidence_threshold: Minimum confidence to take position
        scale_by_confidence: Scale position size by confidence
    
    Returns:
        Position array: -1 (short), 0 (neutral), +1 (long)
        Or fractional if scale_by_confidence=True
    """
    # Get predicted class and confidence
    pred_class = direction_probs.argmax(axis=1)
    max_prob = direction_probs.max(axis=1)
    
    if scale_by_confidence:
        # Fractional positions scaled by confidence above neutral
        positions = np.where(
            (pred_class == 2) & (max_prob > confidence_threshold),
            max_prob - 0.33,  # Long: scaled by confidence
            np.where(
                (pred_class == 0) & (max_prob > confidence_threshold),
                -(max_prob - 0.33),  # Short: scaled by confidence
                0  # No position
            )
        )
    else:
        # Binary positions {-1, 0, 1}
        positions = np.where(
            (pred_class == 2) & (max_prob > confidence_threshold),
            1,  # Long
            np.where(
                (pred_class == 0) & (max_prob > confidence_threshold),
                -1,  # Short
                0  # No position
            )
        )
    
    return positions


def evaluate_model_predictions(
    model_path: str = 'run_financial_wavenet_v1',
    X: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    forward_returns: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Complete evaluation pipeline: load model, normalize, predict, evaluate.
    
    Args:
        model_path: Path to model directory
        X: Feature sequences (if None, must load data)
        y_true: True labels (optional)
        forward_returns: Forward returns for strategy evaluation (optional)
        dates: Timestamps (optional)
        confidence_threshold: Trading signal threshold
    
    Returns:
        Dictionary with all evaluation results
    """
    # Load model and normalizer
    model, normalizer_path = load_model_and_normalizer(model_path)
    
    # Make predictions
    predictions = predict_with_normalization(
        model, X, normalizer_path,
        batch_size=32, verbose=1
    )
    
    direction_probs = predictions['direction']
    
    # Create trading signals
    positions = create_trading_signals(
        direction_probs,
        confidence_threshold=confidence_threshold,
        scale_by_confidence=True
    )
    
    results = {
        'predictions': predictions,
        'positions': positions,
        'direction_probs': direction_probs
    }
    
    # Evaluate accuracy if labels provided
    if y_true is not None:
        pred_labels = direction_probs.argmax(axis=1)
        accuracy = (pred_labels == y_true).mean()
        results['accuracy'] = accuracy
        print(f"\n📊 Classification Accuracy: {accuracy:.4f}")
    
    # Evaluate strategy if returns provided
    if forward_returns is not None:
        strategy_returns = positions * forward_returns
        results['strategy_returns'] = strategy_returns
        results['cumulative_returns'] = (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        win_rate = (strategy_returns > 0).sum() / (strategy_returns != 0).sum()
        
        results['sharpe'] = sharpe
        results['win_rate'] = win_rate
        
        print(f"\n📈 Strategy Performance:")
        print(f"   Sharpe Ratio: {sharpe:.4f}")
        print(f"   Win Rate: {win_rate:.4f}")
        print(f"   Total Return: {results['cumulative_returns'][-1] - 1:.4f}")
    
    return results


# Example usage functions for different evaluation scripts

def evaluate_for_pbo(
    X: np.ndarray,
    forward_returns: np.ndarray,
    model_path: str = 'run_financial_wavenet_v1',
    n_strategies: int = 10
) -> pd.DataFrame:
    """
    Evaluate model for PBO analysis.
    Creates multiple strategies with different confidence thresholds.
    
    Returns:
        DataFrame of strategy returns (observations x strategies)
    """
    model, normalizer_path = load_model_and_normalizer(model_path)
    predictions = predict_with_normalization(model, X, normalizer_path)
    direction_probs = predictions['direction']
    
    # Create strategies with different thresholds
    strategy_returns = []
    min_confidences = np.linspace(0.33, 0.85, n_strategies)
    
    for min_conf in min_confidences:
        positions = create_trading_signals(
            direction_probs,
            confidence_threshold=min_conf,
            scale_by_confidence=True
        )
        returns = positions * forward_returns
        strategy_returns.append(returns)
    
    # Convert to DataFrame
    returns_df = pd.DataFrame(strategy_returns).T
    returns_df.columns = [f'MinConf_{c:.2f}' for c in min_confidences]
    
    return returns_df


def evaluate_for_walkforward(
    X_test: np.ndarray,
    y_test: np.ndarray,
    forward_returns: np.ndarray,
    model_path: str = 'run_financial_wavenet_v1'
) -> Dict:
    """
    Evaluate model for walk-forward analysis.
    
    Returns:
        Dictionary with predictions, positions, and metrics
    """
    return evaluate_model_predictions(
        model_path=model_path,
        X=X_test,
        y_true=y_test,
        forward_returns=forward_returns,
        confidence_threshold=0.5
    )


if __name__ == "__main__":
    print("="*60)
    print("INFERENCE UTILITIES DEMO")
    print("="*60)
    
    # Create dummy data
    np.random.seed(42)
    X_test = np.random.randn(100, 20, 69)  # (samples, timesteps, features)
    y_test = np.random.randint(0, 3, 100)  # Labels
    forward_returns = np.random.randn(100) * 0.02  # Forward returns
    
    print("\nDemo data created:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  forward_returns: {forward_returns.shape}")
    
    print("\n✅ Inference utilities ready!")
    print("   Use these functions in evaluation scripts:")
    print("   - load_model_and_normalizer()")
    print("   - predict_with_normalization()")
    print("   - evaluate_for_pbo()")
    print("   - evaluate_for_walkforward()")
