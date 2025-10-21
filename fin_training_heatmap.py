"""
Financial Barrier Heatmap Training
===================================

Training pipeline using 2-channel heatmap regression for barrier prediction.

Adapted from swimming stroke detection approach:
- Uses Gaussian heatmaps instead of discrete class labels
- ConsistentPeakBFCE loss for peak consistency
- EnhancedPeakConsistencyCallback for monitoring
- No explicit TIMEOUT class (emerges when both channels low)
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import json
from datetime import datetime

# Import our modules
from fin_load_and_sequence import load_or_generate_sequences
import fin_market_context  # Required by load_or_generate_sequences
import fin_utils
from wavenet_model_heatmap import (
    build_wavenet_heatmap_model,
    predict_signals_from_heatmap
)
from barrier_heatmap_targets import create_heatmaps_simple


# ==================== Custom Loss Functions ====================

class ConsistentPeakBFCE(tf.keras.losses.Loss):
    """
    Binary Focal Cross-Entropy with Peak Consistency.
    
    Adapted from swimming stroke detection.
    Ensures predicted peaks reach target height with consistency.
    """
    def __init__(self, 
                 alpha=0.85, 
                 gamma=1.25, 
                 target_height=0.85, 
                 peak_weight=12.0,
                 peak_threshold=0.6,
                 name="consistent_peak_bfce"):
        super().__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.target_height = target_height
        self.peak_weight = peak_weight
        self.peak_threshold = peak_threshold
        
        # Create the base BFCE loss
        self.bfce = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )
    
    def call(self, y_true, y_pred, sample_weight=None):
        # Cast to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute focal loss
        focal_loss = self.bfce(y_true, y_pred)  # Shape: (B, T, 2)
        
        # Apply sample weights to focal loss if provided
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            # Broadcast if needed
            if len(sample_weight.shape) < len(focal_loss.shape):
                for _ in range(len(focal_loss.shape) - len(sample_weight.shape)):
                    sample_weight = tf.expand_dims(sample_weight, -1)
            focal_loss = focal_loss * sample_weight
        
        # Peak consistency loss - only penalize LOW predictions in peak regions
        peak_regions = tf.cast(y_true > self.peak_threshold, tf.float32)  # Where we expect peaks
        low_predictions = tf.cast(y_pred < self.target_height, tf.float32)  # Predictions below target
        penalty_mask = peak_regions * low_predictions  # Only penalize low predictions in peak regions
        
        # Calculate penalty only for low peaks
        target_values = self.target_height * penalty_mask
        actual_values = y_pred * penalty_mask
        peak_mse = tf.square(actual_values - target_values)
        
        # Apply sample weights to peak loss if provided
        if sample_weight is not None:
            peak_mse = peak_mse * sample_weight
        
        # Compute mean peak loss only where penalties apply
        num_penalty_points = tf.reduce_sum(penalty_mask)
        # Use tf.cond instead of Python if for symbolic tensors
        peak_loss = tf.cond(
            tf.greater(num_penalty_points, 0),
            lambda: tf.reduce_sum(peak_mse) / num_penalty_points,
            lambda: 0.0
        )
        
        # Combine losses
        total_loss = focal_loss + self.peak_weight * peak_loss
        
        # Return per-sample loss for SUM_OVER_BATCH_SIZE reduction
        # focal_loss is already (B, T) after bfce, so we just reduce over timesteps
        return tf.reduce_mean(total_loss, axis=1)  # Shape: (B,)


class BarrierPeakConsistencyCallback(tf.keras.callbacks.Callback):
    """
    Monitor barrier heatmap predictions during training.
    
    Adapted from EnhancedPeakConsistencyCallback for financial barriers.
    Tracks:
    - Peak heights at true barrier positions
    - Peak consistency (low std = good)
    - False positive rate
    - Threshold stability
    """
    def __init__(self, x_val, y_val_heatmaps, target_threshold=0.8):
        super().__init__()
        self.x_val = x_val
        self.y_val_heatmaps = y_val_heatmaps
        self.target_threshold = target_threshold
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions
        y_pred = self.model.predict(self.x_val, verbose=0)  # (batch, timesteps, 2)
        
        # Split channels
        sl_pred = y_pred[:, :, 0]  # (batch, timesteps)
        tp_pred = y_pred[:, :, 1]  # (batch, timesteps)
        sl_true = self.y_val_heatmaps[:, :, 0]
        tp_true = self.y_val_heatmaps[:, :, 1]
        
        # Analyze SL channel
        sl_metrics = self._analyze_channel(sl_true, sl_pred, "SL")
        tp_metrics = self._analyze_channel(tp_true, tp_pred, "TP")
        
        # Combined metrics
        logs['val_sl_peak_consistency'] = sl_metrics['peak_consistency']
        logs['val_tp_peak_consistency'] = tp_metrics['peak_consistency']
        logs['val_sl_peaks_above_threshold'] = sl_metrics['peaks_above_threshold']
        logs['val_tp_peaks_above_threshold'] = tp_metrics['peaks_above_threshold']
        logs['val_sl_mean_peak_height'] = sl_metrics['mean_peak']
        logs['val_tp_mean_peak_height'] = tp_metrics['mean_peak']
        logs['val_sl_fpr'] = sl_metrics['fpr']
        logs['val_tp_fpr'] = tp_metrics['fpr']
        
        # Overall objective (like swimming project)
        overall_objective = (
            0.3 * (sl_metrics['f1'] + tp_metrics['f1']) / 2 +
            0.25 * (sl_metrics['peaks_above_threshold'] + tp_metrics['peaks_above_threshold']) / 2 +
            0.15 * (sl_metrics['peak_consistency'] + tp_metrics['peak_consistency']) / 2 +
            0.2 * (1.0 - (sl_metrics['fpr'] + tp_metrics['fpr']) / 2) +
            0.1 * (sl_metrics['threshold_stability'] + tp_metrics['threshold_stability']) / 2
        )
        logs['val_heatmap_objective'] = overall_objective
        
    def _analyze_channel(self, y_true, y_pred, channel_name):
        """Analyze single channel (SL or TP)"""
        
        # Find true peak centers (Gaussian peaks at 0.95+)
        predicted_peaks = []
        for batch_idx in range(len(y_true)):
            true_centers = np.where(y_true[batch_idx] >= 0.95)[0]
            for center in true_centers:
                pred_value = y_pred[batch_idx, center]
                predicted_peaks.append(pred_value)
        
        if len(predicted_peaks) > 0:
            predicted_peaks = np.array(predicted_peaks)
            
            # Peak height metrics
            mean_peak = np.mean(predicted_peaks)
            std_peak = np.std(predicted_peaks)
            peaks_above_threshold = np.mean(predicted_peaks >= self.target_threshold)
            
            # Peak consistency (normalized CV)
            if mean_peak > 0.1:
                peak_consistency = 1.0 - (std_peak / mean_peak)
            else:
                peak_consistency = 0.0
            
            # F1 at target threshold
            y_true_binary = (y_true >= 0.95).astype(int)
            y_pred_binary = (y_pred >= self.target_threshold).astype(int)
            
            y_true_flat = y_true_binary.flatten()
            y_pred_flat = y_pred_binary.flatten()
            
            tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
            fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
            fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
            tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Threshold stability
            threshold_stability = self._check_threshold_stability(y_true, y_pred)
            
        else:
            # No peaks in validation set for this channel
            mean_peak = 0.0
            peaks_above_threshold = 0.0
            peak_consistency = 0.0
            f1 = 0.0
            fpr = 1.0
            threshold_stability = 0.0
        
        return {
            'mean_peak': mean_peak,
            'peaks_above_threshold': peaks_above_threshold,
            'peak_consistency': peak_consistency,
            'f1': f1,
            'fpr': fpr,
            'threshold_stability': threshold_stability
        }
    
    def _check_threshold_stability(self, y_true, y_pred, threshold_range=[0.65, 0.85]):
        """Check if performance is stable across threshold range"""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 9)
        f1_scores = []
        
        y_true_binary = (y_true >= 0.95).astype(int)
        
        for thresh in thresholds:
            y_pred_binary = (y_pred >= thresh).astype(int)
            
            y_true_flat = y_true_binary.flatten()
            y_pred_flat = y_pred_binary.flatten()
            
            tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
            fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
            fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        # Low std means stable performance
        threshold_stability = 1.0 - np.std(f1_scores) if len(f1_scores) > 0 else 0.0
        return threshold_stability


# ==================== Training Configuration ====================

def train_heatmap_model(
    ticker='AAPL',
    use_market_features=False,
    run_name=None,
    save_dir='./models_heatmap'
):
    """
    Train financial barrier heatmap model.
    
    Args:
        ticker: Stock ticker to train on
        use_market_features: Whether to include market context features
        run_name: Name for this training run
        save_dir: Directory to save models
    """
    
    if run_name is None:
        run_name = f"heatmap_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("="*80)
    print(f"FINANCIAL BARRIER HEATMAP TRAINING")
    print("="*80)
    print(f"Ticker: {ticker}")
    print(f"Market features: {use_market_features}")
    print(f"Run name: {run_name}")
    print()
    
    # ==================== Data Loading ====================
    
    print("Loading and preparing data...")
    
    # Choose config preset based on market features
    config_preset = 'wavenet_with_market' if use_market_features else 'wavenet_optimized'
    
    # Load data using existing pipeline (same as fin_training_ldp.py)
    X_seq, y_seq, dates_seq, ldp_weights_seq, returns_seq, feature_names = load_or_generate_sequences(
        tickers=[ticker],
        config_preset=config_preset,
        barrier_params={
            'lookback': 60,
            'pt_sl': [2, 1],  # TP=2x, SL=1x (2% / 1%)
            'min_ret': 0.005,
            'num_days': 10,
            'num_threads': 6
        },
        seq_len=20,
        data_path='data_raw',
        cache_dir='cache',
        use_cache=True,
        verbose=True
    )
    
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    
    # ==================== Create Heatmap Targets ====================
    
    print("\nCreating Gaussian heatmap targets...")
    
    # Convert class labels to heatmaps
    # y_seq is (n_samples,) with {0, 1, 2} = {DOWN/SL, TIMEOUT, UP/TP}
    heatmaps, heatmap_stats = create_heatmaps_simple(
        labels=y_seq,
        seq_len=20,
        sigma=2.0,
        verbose=True
    )
    
    print(f"\nHeatmaps shape: {heatmaps.shape}")
    
    # ==================== Train/Test Split ====================
    
    # Use simple chronological split (like original code)
    split_idx = int(len(X_seq) * 0.8)
    
    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train_heatmaps = heatmaps[:split_idx]
    y_test_heatmaps = heatmaps[split_idx:]
    ldp_weights_train = ldp_weights_seq[:split_idx]
    ldp_weights_test = ldp_weights_seq[split_idx:]
    returns_train = returns_seq[:split_idx]
    returns_test = returns_seq[split_idx:]
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # ==================== Model Building ====================
    
    print("\nBuilding WaveNet heatmap model...")
    
    input_shape = X_train.shape[1:]  # (seq_len, n_features)
    
    model = build_wavenet_heatmap_model(
        input_shape=input_shape,
        wavenet_filters=32,
        wavenet_blocks=4,
        dropout_rate=0.2,
        n_channels=2  # [SL, TP]
    )
    
    model.summary()
    
    # ==================== Compile Model ====================
    
    print("\nCompiling model with ConsistentPeakBFCE loss...")
    
    loss_fn = ConsistentPeakBFCE(
        alpha=0.85,
        gamma=1.25,
        target_height=0.5,
        peak_weight=12.0,
        peak_threshold=0.3
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_fn,
        metrics=['mae']
    )
    
    # ==================== Callbacks ====================
    
    # Create save directory
    save_path = Path(save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Peak consistency monitoring (like swimming project)
        BarrierPeakConsistencyCallback(
            X_test,
            y_test_heatmaps,
            target_threshold=0.5
        ),
        
        # Early stopping on heatmap objective
        tf.keras.callbacks.EarlyStopping(
            monitor='val_heatmap_objective',
            mode='max',
            patience=30,
            restore_best_weights=True,
            verbose=2
        ),
        
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path / 'best_model.keras'),
            monitor='val_heatmap_objective',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # ==================== Training ====================
    
    print("\nStarting training...")
    print(f"Epochs: 100")
    print(f"Batch size: 128")
    print()
    
    history = model.fit(
        X_train,
        y_train_heatmaps,
        validation_data=(X_test, y_test_heatmaps),
        epochs=100,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # ==================== Evaluation ====================
    
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    # Get predictions
    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)
    
    # Convert heatmaps to trading signals
    signals_train = predict_signals_from_heatmap(y_pred_train, tp_threshold=0.5, sl_threshold=0.5)
    signals_test = predict_signals_from_heatmap(y_pred_test, tp_threshold=0.5, sl_threshold=0.5)
    
    # Class balance
    unique_train, counts_train = np.unique(signals_train, return_counts=True)
    unique_test, counts_test = np.unique(signals_test, return_counts=True)
    
    print("\nTRAIN Signal Distribution:")
    signal_names = {-1: 'SHORT', 0: 'NEUTRAL', 1: 'LONG'}
    for sig, count in zip(unique_train, counts_train):
        pct = count / len(signals_train) * 100
        print(f"  {signal_names.get(sig, sig):>7s}: {count:5d} ({pct:5.1f}%)")
    
    print("\nTEST Signal Distribution:")
    for sig, count in zip(unique_test, counts_test):
        pct = count / len(signals_test) * 100
        print(f"  {signal_names.get(sig, sig):>7s}: {count:5d} ({pct:5.1f}%)")
    
    # Calculate strategy returns (simple)
    train_strategy_returns = signals_train * returns_train
    test_strategy_returns = signals_test * returns_test
    
    # Calculate Sharpe ratios (annualized)
    train_sharpe = (train_strategy_returns.mean() / train_strategy_returns.std() * np.sqrt(252)) if train_strategy_returns.std() > 0 else 0
    test_sharpe = (test_strategy_returns.mean() / test_strategy_returns.std() * np.sqrt(252)) if test_strategy_returns.std() > 0 else 0
    
    print(f"\nStrategy Sharpe Ratios:")
    print(f"  Train: {train_sharpe:.3f}")
    print(f"  Test:  {test_sharpe:.3f}")
    
    # ==================== Save Results ====================
    
    # Save final model
    model.save(save_path / 'final_model.keras')
    
    # Save predictions
    np.save(save_path / 'predictions_train.npy', y_pred_train)
    np.save(save_path / 'predictions_test.npy', y_pred_test)
    np.save(save_path / 'signals_train.npy', signals_train)
    np.save(save_path / 'signals_test.npy', signals_test)
    
    # Save history
    with open(save_path / 'history.json', 'w') as f:
        json.dump(history.history, f, indent=2, default=float)
    
    # Save configuration
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    config = {
        'ticker': ticker,
        'use_market_features': use_market_features,
        'run_name': run_name,
        'input_shape': list(input_shape),
        'heatmap_stats': convert_to_native(heatmap_stats),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_sharpe': float(train_sharpe),
        'test_sharpe': float(test_sharpe),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Results saved to: {save_path}")
    print("="*80)
    
    return model, history, signals_test, test_sharpe


if __name__ == '__main__':
    # Test WITHOUT market features first (better baseline)
    print("\n" + "="*80)
    print("EXPERIMENT 1: WITHOUT MARKET FEATURES")
    print("="*80)
    
    model1, history1, signals1, sharpe1 = train_heatmap_model(
        ticker='AAPL',
        use_market_features=False,
        run_name='heatmap_aapl_no_market'
    )
    
    # Test WITH market features
    print("\n\n" + "="*80)
    print("EXPERIMENT 2: WITH MARKET FEATURES")
    print("="*80)
    
    model2, history2, signals2, sharpe2 = train_heatmap_model(
        ticker='AAPL',
        use_market_features=True,
        run_name='heatmap_aapl_with_market'
    )
    
    # Compare
    print("\n\n" + "="*80)
    print("COMPARISON: HEATMAP vs 3-CLASS CLASSIFICATION")
    print("="*80)
    print(f"\nHeatmap WITHOUT market features:")
    print(f"  Test Sharpe: {sharpe1:.3f}")
    print(f"\nHeatmap WITH market features:")
    print(f"  Test Sharpe: {sharpe2:.3f}")
    print("\nPrevious 3-class model (from test_pbo_ldp.py):")
    print(f"  Test Sharpe: 0.34-1.28 (PBO=6.6%)")
    print(f"  Issue: 87% TIMEOUT predictions")
    print("\n✅ Check signal distributions above to see if TIMEOUT bias eliminated!")
