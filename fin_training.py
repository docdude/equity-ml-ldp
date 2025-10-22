from tensorflow.keras import callbacks, optimizers
import numpy as np
from typing import Dict, Tuple, List
import os
import pickle
import pandas as pd
import tensorflow as tf
import random
from fin_feature_preprocessing import EnhancedFinancialFeatures
from fin_model import build_enhanced_cnn_lstm
from feature_config import FeatureConfig
from training_configs import balanced_config, recommend_config

# MLFinLab imports
from MLFinance.FinancialMachineLearning.labeling.labeling import add_vertical_barrier, get_events, meta_labeling
from MLFinance.FinancialMachineLearning.features.volatility import daily_volatility

# Model Parameters 
model_parameters = {
    'input_shape': (20, 100),  # (sequence_len, n_features) 
    'n_classes': 3,  
    'wavenet_filters': 32,
    'wavenet_blocks': 4,
    'wavenet_layers_per_block': 3,
    'conv_filters': [64, 128, 256],
    'lstm_units': [256, 128],
    'attention_units': 128,
    'dropout_rate': 0.3,
    'l2_reg': 0.0001
}

# Import balanced config from training_configs
# You can switch to conservative_config or aggressive_config as needed
training_config = balanced_config
training_parameters = training_config['training_parameters']

# Extract callback-specific parameters
callback_parameters = {
    'monitor': 'val_direction_auc',
    'early_stopping_patience': training_parameters['early_stopping_patience'],
    'reduce_lr_patience': training_parameters['reduce_lr_patience'],
    'reduce_lr_factor': training_parameters['reduce_lr_factor'],
    'min_lr': training_parameters['min_lr']
}

def create_callbacks(
    output_dir: str = 'models',
    callback_params: Dict = None
) -> List[callbacks.Callback]:
    """
    Create training callbacks using parameters from training_configs
    
    Args:
        output_dir: Directory to save model outputs
        callback_params: Callback configuration dict from training_configs
    """
    if callback_params is None:
        callback_params = callback_parameters
    
    monitor = callback_params['monitor']
    patience = callback_params['early_stopping_patience']
    reduce_lr_patience = callback_params['reduce_lr_patience']
    reduce_lr_factor = callback_params['reduce_lr_factor']
    min_lr = callback_params['min_lr']
    
    callback_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce LR on plateau
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode='max',
            verbose=1
        ),
        
        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=f'{output_dir}/best_model.keras',
            monitor=monitor,
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # CSV logger
        callbacks.CSVLogger(
            filename=f'{output_dir}/training_log.csv',
            append=False
        ),
        
        # TensorBoard for visualization
        callbacks.TensorBoard(
            log_dir=f'{output_dir}/tensorboard_logs',
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0,  # Disable profiling to reduce overhead
            embeddings_freq=0
        )
    ]
    
    return callback_list



def main():
    """
    Model training pipeline using balanced_config from training_configs.py
    """
    print("="*80)
    print("CNN-LSTM FINANCIAL MODEL TRAINING")
    print("="*80)
    print("\n⚙️  Configuration: balanced_config")
    print(f"   Learning Rate: {training_parameters['learning_rate']}")
    print(f"   Batch Size: {training_parameters['batch_size']}")
    print(f"   Max Epochs: {training_parameters['max_epochs']}")
    patience = callback_parameters['early_stopping_patience']
    print(f"   Early Stop Patience: {patience}")
    
    # Check GPU
    print("🔍 Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    else:
        print("⚠️  No GPU found, using CPU")
    # Set Random Seed for Reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    print()
    print("="*80)
    # Paths
    data_path = 'data_raw'
    run_name = 'financial_wavenet_v1'
    base_path = '.'
    experiment_save_path = os.path.join(base_path, f'run_{run_name}')
    
    # 1. Load and preprocess data
    print("\n1. ENHANCED FEATURE ENGINEERING")
    print("-"*40)
    
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
    all_forward_returns = []  # Calculate per-ticker!
    all_forward_volatility = []  # Calculate per-ticker!
    all_tickers = []  # Track ticker for each sample
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        # Load raw data
        df = pd.read_parquet(f'{data_path}/{ticker}.parquet')
        
        # Set date as index (critical for proper date tracking!)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Clean data: Remove rows with NaN prices (IPO dates, data gaps)
        df = df.dropna(subset=['Close', 'Open', 'High', 'Low'])
        
        if len(df) < 50:  # Skip tickers with insufficient data
            print(f"  ⚠️  Skipping {ticker}: Only {len(df)} valid rows")
            continue
        
        # Create comprehensive features
        features = feature_engineer.create_all_features(df)
        
        # Create dynamic triple barriers
        barriers = feature_engineer.create_dynamic_triple_barriers(df)
        
        # Combine
        combined = pd.concat([features, barriers], axis=1)
        combined = combined.dropna()
        
        if len(combined) == 0:
            print(f"  ⚠️  Skipping {ticker}: No valid samples after feature engineering")
            continue
        
        # Align prices with features (critical for forward calculations)
        prices_aligned = df.loc[combined.index, 'Close']
        
        # ✅ USE EXIT_RETURN FROM BARRIERS (align with fin_model_evaluation.py)
        # This is the López de Prado / MLFinLab method
        # Uses actual returns at barrier touch, not fixed-horizon returns
        exit_returns = combined['exit_return'].values
        exit_days = combined['exit_day'].values
        
        # Calculate volatility over actual holding period (not fixed 5-day window)
        exit_volatility = []
        for idx in range(len(combined)):
            days = int(exit_days[idx])
            if days > 1 and pd.notna(combined['t1'].iloc[idx]):
                # Get actual exit window
                start_date = combined.index[idx]
                end_date = combined['t1'].iloc[idx]
                
                # Calculate volatility over this period
                try:
                    period_prices = prices_aligned.loc[start_date:end_date]
                    if len(period_prices) > 1:
                        period_returns = period_prices.pct_change().dropna()
                        vol = period_returns.std() if len(period_returns) > 0 else 0.0
                    else:
                        vol = 0.0
                except:
                    vol = 0.0
            else:
                vol = 0.0
            
            exit_volatility.append(vol)
        
        # Add to lists (aligned with barriers)
        all_features.append(combined[features.columns])
        all_labels.append(combined['label'])
        all_forward_returns.append(pd.Series(exit_returns, index=combined.index))
        all_forward_volatility.append(pd.Series(exit_volatility, index=combined.index))
        all_dates.extend(combined.index)
        all_tickers.extend([ticker] * len(combined))
        
        print(f"  ✅ {ticker}: {len(combined)} samples")
    
    # Combine all data
    X = pd.concat(all_features)
    y = pd.concat(all_labels)
    forward_returns = pd.concat(all_forward_returns)  # ✅ From barriers (exit_return)
    forward_volatility = pd.concat(all_forward_volatility)  # ✅ From actual holding periods
    dates = pd.DatetimeIndex(all_dates)
    tickers_array = np.array(all_tickers)
    
    # CRITICAL: Sort by date for proper time-series cross-validation
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    forward_returns = forward_returns.iloc[sort_idx].reset_index(drop=True)
    forward_volatility = forward_volatility.iloc[sort_idx].reset_index(drop=True)
    dates = dates[sort_idx]
    tickers_array = tickers_array[sort_idx]
    
    print("\n⚠️  Data sorted chronologically for proper CV")
    print(f"✅ Using exit_return from barriers (aligned with evaluation)")
    print(f"✅ Using actual holding-period volatility (not fixed window)")
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Validate feature count matches wavenet_optimized preset
    expected_features = 18
    actual_features = X.shape[1]
    if actual_features != expected_features:
        print(f"\n⚠️  WARNING: Expected {expected_features} features (wavenet_optimized), got {actual_features}")
        print(f"   Check feature_config.py - wavenet_optimized preset may not be working correctly")
    else:
        print(f"\n✅ Feature count verified: {actual_features} features (wavenet_optimized preset)")
    
    # 🔬 ALIGNMENT VERIFICATION
    print(f"\n🔬 Barrier Alignment Verification:")
    print(f"   Exit returns - Mean: {forward_returns.mean():.4f}, Std: {forward_returns.std():.4f}")
    mask_tp = y == 1  # TP labels
    if mask_tp.sum() > 0:
        tp_returns = forward_returns[mask_tp]
        print(f"   TP labels (expect positive returns):")
        print(f"      Mean: {tp_returns.mean():.4f}")
        print(f"      % positive: {(tp_returns > 0).sum() / len(tp_returns) * 100:.1f}%")
        if (tp_returns > 0).sum() / len(tp_returns) > 0.7:
            print(f"      ✅ GOOD: TP labels mostly positive (aligned)")
        else:
            print(f"      ⚠️  WARNING: TP labels not mostly positive (misaligned?)")
    
    mask_sl = y == 0  # SL labels  
    if mask_sl.sum() > 0:
        sl_returns = forward_returns[mask_sl]
        print(f"   SL labels (expect negative returns):")
        print(f"      Mean: {sl_returns.mean():.4f}")
        print(f"      % negative: {(sl_returns < 0).sum() / len(sl_returns) * 100:.1f}%")
        if (sl_returns < 0).sum() / len(sl_returns) > 0.7:
            print(f"      ✅ GOOD: SL labels mostly negative (aligned)")
        else:
            print(f"      ⚠️  WARNING: SL labels not mostly negative (misaligned?)")
    
    # 2. Prepare sequences (for deep learning)
    print("\n2. SEQUENCE PREPARATION")
    print("-"*40)
    
    seq_len = 20
    X_sequences = []
    y_sequences = []
    dates_sequences = []
    
    for i in range(len(X) - seq_len):
        X_sequences.append(X.iloc[i:i+seq_len].values)
        y_sequences.append(y.iloc[i+seq_len-1])
        dates_sequences.append(dates[i+seq_len-1])
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    dates_seq = pd.DatetimeIndex(dates_sequences)
    
    print(f"Sequences shape: {X_seq.shape}")
    print(f"Sequence date range: {dates_seq.min().date()} to {dates_seq.max().date()}")
    
    # Note: Feature normalization will be done AFTER train/val split
    # to prevent data leakage (fit on train only)
    print(f"\n📊 Raw feature statistics (before normalization):")
    print(f"   Mean: {X_seq.mean():.4f}, Std: {X_seq.std():.4f}")
    print(f"   Min: {X_seq.min():.4f}, Max: {X_seq.max():.4f}")
    
    # 3. Prepare auxiliary targets (already calculated per-ticker!)
    print("\n3. PREPARING AUXILIARY TARGETS")
    print("-"*40)
    
    # Forward targets were calculated per-ticker, now align with sequences
    # Extract target for each sequence endpoint
    forward_returns_seq = []
    forward_volatility_seq = []
    
    for i in range(len(X) - seq_len):
        # Get target at sequence endpoint
        target_idx = i + seq_len - 1
        forward_returns_seq.append(forward_returns.iloc[target_idx])
        forward_volatility_seq.append(forward_volatility.iloc[target_idx])
    
    forward_returns_seq = np.array(forward_returns_seq)
    forward_volatility_seq = np.array(forward_volatility_seq)
    
    # Filter out NaN values (from end of each ticker's data)
    valid_mask = ~(np.isnan(forward_returns_seq) | np.isnan(forward_volatility_seq))
    
    X_seq = X_seq[valid_mask]
    y_seq = y_seq[valid_mask]
    dates_seq = dates_seq[valid_mask]
    forward_returns_seq = forward_returns_seq[valid_mask]
    forward_volatility_seq = forward_volatility_seq[valid_mask]
    
    print(f"✅ Filtered {(~valid_mask).sum()} samples with NaN targets")
    print(f"   (Expected from last 5 days of each ticker)")
    
    # Map labels to 0-based indices for to_categorical
    # Original: {-1, 0, 1} → Shift to {0, 1, 2}
    # This works for any label range, not just {-1, 0, 1}
    print("\n📋 Mapping labels for to_categorical:")
    print(f"   Before mapping: {np.unique(y_seq, return_counts=True)}")
    
    y_seq = y_seq - np.min(y_seq)  # Shift minimum to 0
    
    print(f"   After mapping: {np.unique(y_seq, return_counts=True)}")
    print(f"   Label interpretation: [0=DOWN, 1=UP, 2=NEUTRAL]")
    
    n_samples = len(X_seq)
    forward_returns = forward_returns_seq
    forward_volatility = forward_volatility_seq
    
    print(f"Forward returns shape: {forward_returns.shape}")
    print(f"Forward volatility shape: {forward_volatility.shape}")
    print(f"Aligned sequences: {X_seq.shape[0]}")
    
    # Diagnostic: Check target ranges
    print(f"\n📊 Target Statistics:")
    print(f"   Forward returns: min={forward_returns.min():.4f}, "
          f"max={forward_returns.max():.4f}, "
          f"mean={forward_returns.mean():.4f}, "
          f"std={forward_returns.std():.4f}")
    print(f"   Forward volatility: min={forward_volatility.min():.4f}, "
          f"max={forward_volatility.max():.4f}, "
          f"mean={forward_volatility.mean():.4f}, "
          f"std={forward_volatility.std():.4f}")
    
    # 4. Time-based train/validation split (80/20)
    print("\n4. TRAIN/VALIDATION SPLIT")
    print("-"*40)
    
    split_idx = int(len(X_seq) * 0.8)
    
    # Split sequences
    X_train = X_seq[:split_idx]
    X_val = X_seq[split_idx:]
    
    # 4.5. REFIT NORMALIZER ON TRAINING DATA ONLY (prevent data leakage)
    print("\n4.5. REFITTING NORMALIZER ON TRAINING DATA ONLY")
    print("-"*40)
    
    from fin_utils import normalize_for_training
    
    # Normalize with proper train-only fitting
    X_train, X_val, normalizer_final = normalize_for_training(
        X_train, X_val,
        method='robust',  # Same method as initial normalization
        save_path=None  # Will save later with model
    )
    
    print(f"\n✅ Normalizer refit on training data only (no data leakage)")
    
    # Convert to categorical (labels already mapped to {0, 1, 2})
    y_train_direction = tf.keras.utils.to_categorical(y_seq[:split_idx], num_classes=3)
    y_val_direction = tf.keras.utils.to_categorical(y_seq[split_idx:], num_classes=3)
    
    y_train_volatility = forward_volatility[:split_idx]
    y_val_volatility = forward_volatility[split_idx:]
    
    y_train_magnitude = np.abs(forward_returns[:split_idx])
    y_val_magnitude = np.abs(forward_returns[split_idx:])
    
    train_dates = dates_seq[:split_idx]
    val_dates = dates_seq[split_idx:]
    
    # Calculate class weights to handle imbalance
    print("\n4.6. CALCULATING CLASS WEIGHTS FOR IMBALANCED DATA")
    print("-"*40)
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Compute sample weights directly using sklearn's compute_sample_weight
    # Important: Compute weights separately for train and val using their own distributions
    sample_weights_train = compute_sample_weight(
        class_weight='balanced',
        y=y_seq[:split_idx]  # Training labels only
    )
    
    sample_weights_val = compute_sample_weight(
        class_weight='balanced',
        y=y_seq[split_idx:]  # Validation labels only
    )
    
    print(f"Class distribution (training):")
    for label in [0, 1, 2]:
        count = (y_seq[:split_idx] == label).sum()
        pct = count / len(y_seq[:split_idx]) * 100
        label_name = {0: "DOWN", 1: "UP", 2: "NEUTRAL"}[label]
        # Get weight for this label from sample_weights_train
        weight = sample_weights_train[y_seq[:split_idx] == label][0]
        print(f"  Label {label:2d} ({label_name}): {count:5d} ({pct:5.2f}%) → weight={weight:.4f}")
    
    print(f"\nSample weights statistics:")
    print(f"  Training: mean={sample_weights_train.mean():.4f}, std={sample_weights_train.std():.4f}")
    print(f"  Validation: mean={sample_weights_val.mean():.4f}, std={sample_weights_val.std():.4f}")
    print(f"✅ Class weights will force model to pay attention to minority classes (DOWN, UP)")
    
    # Create uniform weights for volatility and magnitude (no class weighting for regression outputs)
    uniform_weights_train = np.ones(len(sample_weights_train))
    uniform_weights_val = np.ones(len(sample_weights_val))
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"  Date range: {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"Validation samples: {len(X_val)}")
    print(f"  Date range: {val_dates.min().date()} to {val_dates.max().date()}")
    
    # 5. Train Enhanced Model
    print("\n5. TRAINING CNN-LSTM MODEL")
    print("-"*40)
    
    print("🔧 Building financial CNN-LSTM model...")
    
    model = build_enhanced_cnn_lstm(
        input_shape=(seq_len, X.shape[1]),
        n_classes=model_parameters['n_classes'],
        wavenet_filters=model_parameters['wavenet_filters'],
        wavenet_blocks=model_parameters['wavenet_blocks'],
        wavenet_layers_per_block=model_parameters['wavenet_layers_per_block'],
        conv_filters=model_parameters['conv_filters'],
        lstm_units=model_parameters['lstm_units'],
        attention_units=model_parameters['attention_units'],
        dropout_rate=model_parameters['dropout_rate'],
    )
    
    model.summary()
    print(f"Model parameters: {model.count_params():,}")
    
    # Compile model with Focal Loss for class imbalance
    optimizer = optimizers.Adam(
        learning_rate=training_parameters['learning_rate'],
        beta_1=training_parameters['beta_1'],
        beta_2=training_parameters['beta_2'],
        clipnorm=training_parameters.get('clipnorm', 1.0)  # Gradient clipping
    )
    
    # Focal Loss parameters
    # gamma: focusing parameter (higher = more focus on hard examples)
    # alpha: weighting factor (can be scalar or array for class weights)
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=0.25,  # Default weighting
        gamma=2.0,   # Standard focal loss gamma
        from_logits=False,  # Model outputs softmax probabilities
        name='focal_loss'
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            'direction': focal_loss,  # Use Focal Loss instead of categorical_crossentropy
            'volatility': 'mse',  
            'magnitude': 'huber'  
        },
    #    loss_weights={
     #       'direction': training_parameters['direction_loss_weight'],
      #      'volatility': training_parameters['volatility_loss_weight'],
       #     'magnitude': training_parameters['magnitude_loss_weight']
      #  },
        metrics={
            'direction': [
                'accuracy',  # Standard accuracy (unweighted - for comparison)
                tf.keras.metrics.AUC(name='auc')
            ],
            'volatility': [
                'mae',  # Mean Absolute Error (interpretable in %)
                'mse',  # Mean Squared Error (penalizes large errors)
                tf.keras.metrics.RootMeanSquaredError(name='rmse')  # In same units as target
            ],
            'magnitude': [
                'mae',  # Mean Absolute Error (interpretable in %)
                'mse',  # Mean Squared Error
                tf.keras.metrics.RootMeanSquaredError(name='rmse')  # In same units as target
            ]
        },
        weighted_metrics={
            'direction': ['accuracy']  # Keras will automatically track weighted accuracy
        }
    )
    
    print("\n🚀 Starting training with Focal Loss + Sample Weights...")
    print(f"   Focal Loss: alpha=0.25, gamma=2.0 (focuses on hard examples)")
    print(f"   Sample Weights: Balanced (handles class imbalance)")
    print(f"   Combined approach for better generalization")
    print(f"   Epochs: {training_parameters['max_epochs']}")
    print(f"   Batch size: {training_parameters['batch_size']}")
    print(f"   Learning rate: {training_parameters['learning_rate']}")
    print(f"   Gradient clipping: {training_parameters.get('clipnorm', 'None')}")
    es_patience = callback_parameters['early_stopping_patience']
    print(f"   Early stopping patience: {es_patience}")
    
    print("\n📊 Expected Metric Ranges (with ReLU outputs):")
    print("   Direction:")
    print("      - Accuracy: 0.4-0.6 (3-class)")
    print("      - AUC: Target ≥ 0.65")
    print("   Volatility (ReLU output, MSE loss):")
    print(f"      - Targets: mean={y_train_volatility.mean():.4f}, std={y_train_volatility.std():.4f}")
    print("      - Good MAE: < 0.01 (1%)")
    print("      - Good RMSE: < 0.015 (1.5%)")
    print("   Magnitude (ReLU output, Huber loss):")
    print(f"      - Targets: mean={y_train_magnitude.mean():.4f}, std={y_train_magnitude.std():.4f}")
    print("      - Good MAE: < 0.03 (3%)")
    print("      - Good RMSE: < 0.05 (5%)")
    
    # Calculate steps per epoch if specified
    steps_per_epoch = training_parameters.get('steps_per_epoch', None)
    if steps_per_epoch is not None:
        print(f"   Steps per epoch: {steps_per_epoch}")
    else:
        steps = len(X_train) // training_parameters['batch_size']
        print(f"   Steps per epoch: {steps} (auto-calculated)")
    
    # Train with Focal Loss + sample weights (complementary approaches)
    # Focal Loss: focuses on hard examples, down-weights easy ones
    # Sample Weights: balances class distribution
    history = model.fit(
        X_train,
        {
            'direction': y_train_direction,
            'volatility': y_train_volatility,
            'magnitude': y_train_magnitude
        },
        sample_weight={
            'direction': sample_weights_train,
            'volatility': uniform_weights_train,
            'magnitude': uniform_weights_train
        },
        validation_data=(
            X_val,
            {
                'direction': y_val_direction,
                'volatility': y_val_volatility,
                'magnitude': y_val_magnitude
            },
            {
                'direction': sample_weights_val,
                'volatility': uniform_weights_val,
                'magnitude': uniform_weights_val
            }
        ),
        epochs=training_parameters['max_epochs'],
        batch_size=training_parameters['batch_size'],
        steps_per_epoch=steps_per_epoch,
        callbacks=create_callbacks(
            output_dir=experiment_save_path,
            callback_params=callback_parameters
        ),
        verbose=2
    )
    # Post-training verification of output ranges
    print("\n🔍 Verifying output ranges...")
    val_predictions = model.predict(X_val[:1000], verbose=0)  # Sample first 1000
    
    vol_preds = val_predictions['volatility'].flatten()
    mag_preds = val_predictions['magnitude'].flatten()
    
    print(f"   Volatility predictions:")
    print(f"      Range: [{vol_preds.min():.6f}, {vol_preds.max():.6f}]")
    print(f"      Mean: {vol_preds.mean():.6f}")
    print(f"      % saturated (>0.999): {(vol_preds > 0.999).sum() / len(vol_preds) * 100:.2f}%")
    if (vol_preds > 0.999).sum() / len(vol_preds) > 0.5:
        print("      ⚠️  WARNING: >50% predictions saturated! Model may not have fixed properly.")
    else:
        print("      ✅ GOOD: Predictions not saturated")
    
    print(f"   Magnitude predictions:")
    print(f"      Range: [{mag_preds.min():.4f}, {mag_preds.max():.4f}]")
    print(f"      Mean: {mag_preds.mean():.4f}")
    print(f"      % > 1.0: {(mag_preds > 1.0).sum() / len(mag_preds) * 100:.2f}%")
    if mag_preds.mean() > 0.5:
        print("      ⚠️  WARNING: Mean prediction >50%! Seems too high.")
    else:
        print("      ✅ GOOD: Predictions in reasonable range")
    
    # Save
    model.save(os.path.join(experiment_save_path, f'fin_wavenet_model.keras'))
    
    with open(os.path.join(experiment_save_path, 'history.pkl'), 'wb') as f:
        pickle.dump([history.history], f)
    
    # Save feature normalizer for inference
    normalizer_path = os.path.join(experiment_save_path, 'feature_scaler.pkl')
    normalizer_final.save(normalizer_path)
    
    # Save complete training config for reproducibility
    complete_config = {
        'training_parameters': training_parameters,
        'callback_parameters': callback_parameters,
        'model_parameters': model_parameters,
        'config_name': 'balanced_config',
        'normalization': {
            'method': 'RobustScaler',
            'scaler_type': 'sklearn.preprocessing.RobustScaler',
            'fitted_on': 'training_data_only',
            'note': 'Features properly scale-normalized at source (OBV, A/D use cumulative volume normalization)'
        }
    }
    with open(os.path.join(experiment_save_path, 'training_config.pkl'), 'wb') as f:
        pickle.dump(complete_config, f)
    
    print("\n✅ Training completed!")
    print(f"   Model saved to: {experiment_save_path}")
    print(f"   Config used: balanced_config")
    print(f"   Best val AUC: {max(history.history['val_direction_auc']):.4f}")

if __name__ == '__main__':
    main()