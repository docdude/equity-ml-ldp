"""
López de Prado Triple Barrier Training (Direction-Only)
========================================================

Implements proper López de Prado methodology from "Advances in Financial ML":

1. Triple Barrier Labeling (Chapter 3)
   - Dynamic volatility-based barriers (asymmetric pt_sl = [2, 1])
   - Minimum return threshold to filter noise
   - Proper meta-labeling via MLFinLab

2. Sample Weighting (Chapter 4.5)
   - Uniqueness weights (average_uniqueness)
   - Calculated per-ticker before combining
   - Combined with sklearn class balancing

3. Direction-Only Prediction
   - Focus: Generate signals for Bayesian Kelly position sizing
   - No auxiliary outputs (volatility/magnitude)
   - Single categorical output: DOWN/TIMEOUT/UP

References:
- MLFinLab: https://github.com/hudson-and-thames/mlfinlab
- Book: "Advances in Financial Machine Learning" by Marcos López de Prado
"""
from typing import Dict, Tuple, List, Optional
import os
import sys
import pickle
from tensorflow.keras import callbacks, optimizers
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from fin_feature_preprocessing import EnhancedFinancialFeatures
#from fin_model import build_enhanced_cnn_lstm
from wavenet_model import build_enhanced_cnn_lstm
from feature_config import FeatureConfig
from training_configs import balanced_config, recommend_config, aggressive_config
from fin_load_and_sequence import load_or_generate_sequences

# Add MLFinance to path for imports
mlfinance_path = os.path.join(os.path.dirname(__file__), 'MLFinance')
if mlfinance_path not in sys.path:
    sys.path.insert(0, mlfinance_path)


# MLFinLab imports for proper triple barrier implementation
from FinancialMachineLearning.labeling.labeling import (
    add_vertical_barrier, 
    get_events, 
    meta_labeling
)
from FinancialMachineLearning.features.volatility import daily_volatility
from FinancialMachineLearning.sample_weights.concurrency import average_uniqueness_triple_barrier

# Training configuration
training_config = balanced_config
training_parameters = training_config['training_parameters']
model_config = training_config['model_parameters']

# Model Parameters (use from config)
model_parameters = {
    'input_shape': (20, 100),
    'n_classes': 3,
    **model_config  # Unpack config values (dropout_rate, l2_reg, etc.)
}

# Callback parameters
callback_parameters = {
    'monitor': 'val_auc',  # Single output, no 'direction_' prefix
    'early_stopping_patience': training_parameters['early_stopping_patience'],
    'reduce_lr_patience': training_parameters['reduce_lr_patience'],
    'reduce_lr_factor': training_parameters['reduce_lr_factor'],
    'min_lr': training_parameters['min_lr']
}

def create_callbacks(
    output_dir: str = 'models',
    callback_params: Dict = None
) -> List[callbacks.Callback]:
    """Create training callbacks"""
    if callback_params is None:
        callback_params = callback_parameters
    
    monitor = callback_params['monitor']
    patience = callback_params['early_stopping_patience']
    reduce_lr_patience = callback_params['reduce_lr_patience']
    reduce_lr_factor = callback_params['reduce_lr_factor']
    min_lr = callback_params['min_lr']
    
    callback_list = [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode='max',
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{output_dir}/best_model.keras',
            monitor=monitor,
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.CSVLogger(
            filename=f'{output_dir}/training_log.csv',
            append=False
        ),
        callbacks.TensorBoard(
            log_dir=f'{output_dir}/tensorboard_logs',
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0
        )
    ]
    
    return callback_list


def create_mlfinlab_barriers(
    df: pd.DataFrame,
    lookback: int = 60,
    pt_sl: List[float] = [2, 1],
    min_ret: float = 0.005,
    num_days: int = 7,
    num_threads: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create triple barrier labels using MLFinLab methodology
    
    Args:
        df: DataFrame with OHLCV data (must have 'Close' column)
        lookback: Days for volatility calculation
        pt_sl: [profit_taking_multiple, stop_loss_multiple]
               e.g., [2, 1] means TP = 2×σ, SL = 1×σ (asymmetric)
        min_ret: Minimum return threshold (filters noise)
        num_days: Vertical barrier horizon (days)
        num_threads: Number of threads for parallel processing
    
    Returns:
        events: Triple barrier events DataFrame (with t1, trgt, pt, sl)
        labels: Label DataFrame (with ret, trgt, bin)
    """
    print("\n" + "="*80)
    print("MLFINLAB TRIPLE BARRIER LABELING")
    print("="*80)
    
    # Step 1: Calculate dynamic volatility threshold
    volatility = daily_volatility(df['Close'], lookback=lookback)
    print(f"\n✓ Volatility calculated ({lookback}-day EWMA)")
    print(f"  Mean: {volatility.mean():.4f} ({volatility.mean()*100:.2f}%)")
    print(f"  Std:  {volatility.std():.4f}")
    print(f"  Min:  {volatility.min():.4f}, Max: {volatility.max():.4f}")
    
    # Step 2: Add vertical barrier (expiration limit)
    vertical_barriers = add_vertical_barrier(
        t_events=df.index,
        close=df['Close'],
        num_days=num_days
    )
    print(f"\n✓ Vertical barriers created: {len(vertical_barriers)} timestamps")
    print(f"  Horizon: {num_days} days")
    
    # Step 3: Get triple barrier events
    # Start after volatility warmup period
    t_events = df.index[lookback:]
    
    events = get_events(
        close=df['Close'],
        t_events=t_events,
        pt_sl=pt_sl,  # [profit_taking, stop_loss] multiples
        target=volatility,  # Dynamic threshold
        min_ret=min_ret,  # Minimum return to consider
        num_threads=num_threads,
        vertical_barrier_times=vertical_barriers,
        side_prediction=None  # No primary model (learn direction)
    )
    
    print(f"\n✓ Triple barrier events generated: {len(events)}")
    print(f"  Profit taking: {pt_sl[0]}× volatility")
    print(f"  Stop loss:     {pt_sl[1]}× volatility")
    print(f"  Min return:    {min_ret:.3f} ({min_ret*100:.1f}%)")
    print(f"\nEvent columns:")
    for col in events.columns:
        print(f"  - {col}")
    
    # Step 4: Apply meta-labeling to get labels
    labels = meta_labeling(events, df['Close'])
    
    print(f"\n✓ Labels generated: {len(labels)}")
    print(f"\nLabel columns:")
    print(f"  - ret: Actual return at barrier touch")
    print(f"  - trgt: Target threshold that was used")
    print(f"  - bin: Label (1=UP/TP, 0=TIMEOUT, -1=DOWN/SL)")
    
    # Label distribution
    label_counts = labels['bin'].value_counts().sort_index()
    print(f"\n{'='*40}")
    print("LABEL DISTRIBUTION (3 classes)")
    print(f"{'='*40}")
    
    label_names = {-1: 'DOWN/Stop-Loss', 0: 'TIMEOUT/Vertical', 1: 'UP/Take-Profit'}
    for label in sorted(label_counts.index):
        count = label_counts[label]
        pct = count / len(labels) * 100
        name = label_names.get(label, f'Unknown({label})')
        bar = '█' * int(pct / 2)
        print(f"{name:20} ({label:2d}): {count:5d} ({pct:5.1f}%) {bar}")
    
    # Verify alignment
    print(f"\n{'='*40}")
    print("ALIGNMENT VERIFICATION")
    print(f"{'='*40}")
    
    returns_up = labels[labels['bin'] == 1]['ret']
    returns_down = labels[labels['bin'] == -1]['ret']
    returns_timeout = labels[labels['bin'] == 0]['ret']
    
    if len(returns_up) > 0:
        pct_positive_up = (returns_up > 0).sum() / len(returns_up) * 100
        print(f"UP labels (expect positive returns):")
        print(f"  Mean return: {returns_up.mean():.4f} ({returns_up.mean()*100:.2f}%)")
        print(f"  % positive:  {pct_positive_up:.1f}%")
        if pct_positive_up > 70:
            print(f"  ✅ GOOD: UP labels mostly positive")
        else:
            print(f"  ⚠️  WARNING: UP labels not mostly positive")
    
    if len(returns_down) > 0:
        pct_negative_down = (returns_down < 0).sum() / len(returns_down) * 100
        print(f"\nDOWN labels (expect negative returns):")
        print(f"  Mean return: {returns_down.mean():.4f} ({returns_down.mean()*100:.2f}%)")
        print(f"  % negative:  {pct_negative_down:.1f}%")
        if pct_negative_down > 70:
            print(f"  ✅ GOOD: DOWN labels mostly negative")
        else:
            print(f"  ⚠️  WARNING: DOWN labels not mostly negative")
    
    if len(returns_timeout) > 0:
        pct_positive_timeout = (returns_timeout > 0).sum() / len(returns_timeout) * 100
        print(f"\nTIMEOUT labels (mixed expected):")
        print(f"  Mean return: {returns_timeout.mean():.4f} ({returns_timeout.mean()*100:.2f}%)")
        print(f"  % positive:  {pct_positive_timeout:.1f}%")
        print(f"  Note: Timeout = vertical barrier reached before TP/SL")
    
    return events, labels


def main():
    """
    Model training pipeline using MLFinLab triple barrier labeling
    """
    print("="*80)
    print("CNN-LSTM FINANCIAL MODEL TRAINING (López de Prado Method)")
    print("="*80)
    print("\n⚙️  Configuration: balanced_config")
    print(f"   Learning Rate: {training_parameters['learning_rate']}")
    print(f"   Batch Size: {training_parameters['batch_size']}")
    print(f"   Max Epochs: {training_parameters['max_epochs']}")
    patience = callback_parameters['early_stopping_patience']
    print(f"   Early Stop Patience: {patience}")
    
    # Check GPU
    print("\n🔍 Checking GPU availability...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    else:
        print("⚠️  No GPU found, using CPU")
    
    # Set Random Seed
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    print("\n" + "="*80)
    
    # Paths
    data_path = 'data_raw'
    run_name = 'financial_wavenet_ldp_v1'
    base_path = '.'
    experiment_save_path = os.path.join(base_path, f'run_{run_name}')
    
    if not os.path.exists(experiment_save_path):
        os.makedirs(experiment_save_path)
    
    # 1. Load and preprocess data
    print("\n1. ENHANCED FEATURE ENGINEERING")
    print("-"*40)
    
    tickers = ['AAPL', 'DELL', 'JOBY', 'LCID', 'SMCI', 'NVDA', 'TSLA', 'WDAY', 'AMZN', 'AVGO']
    #tickers = ['AAPL']  # For testing
    
    CONFIG_PRESET = 'wavenet_optimized_v2'
    config = FeatureConfig.get_preset(CONFIG_PRESET)
    feature_engineer = EnhancedFinancialFeatures(feature_config=config)
    
    print(f"\n📋 Using {CONFIG_PRESET} feature preset")
    print("   Selected by López de Prado analysis (MDI/MDA/SFI)")
    feature_engineer.print_config()
    
    # Triple barrier parameters
    BARRIER_PARAMS = {
        'lookback': 60,      # Days for volatility calculation
        'pt_sl': [2, 1],     # Asymmetric barriers: TP=2×σ, SL=1×σ
        'min_ret': 0.005,    # 0.5% minimum return threshold
        'num_days': 7,       # 7-day vertical barrier
        'num_threads': 6
    }
    
    print(f"\n📊 Triple Barrier Parameters:")
    print(f"   Profit Taking: {BARRIER_PARAMS['pt_sl'][0]}× volatility")
    print(f"   Stop Loss:     {BARRIER_PARAMS['pt_sl'][1]}× volatility")
    print(f"   Min Return:    {BARRIER_PARAMS['min_ret']} ({BARRIER_PARAMS['min_ret']*100}%)")
    print(f"   Horizon:       {BARRIER_PARAMS['num_days']} days")
    print(f"   Volatility:    {BARRIER_PARAMS['lookback']}-day EWMA")
    
    # Market features configuration (based on wavenet_optimized_v2 analysis)
    # See WAVENET_V2_USAGE.md for rationale
    MARKET_FEATURES = ['fvx', 'tyx']  # Treasury yields (rank 8/14 orthogonal)
    # Options: ['spy', 'vix', 'fvx', 'tyx', 'gold', 'jpyx']
    # Recommended: ['fvx', 'tyx'] - unique macro risk signals
    # Optional: ['gold'] - rank 13/21, consistent performer
    # Skip: ['spy', 'vix', 'jpyx'] - redundant with stock features
    
    print(f"\n📊 Market Features: {MARKET_FEATURES}")
    if MARKET_FEATURES:
        market_map = {
            'spy': '^GSPC (S&P 500)',
            'vix': '^VIX (Volatility)',
            'fvx': '^FVX (5yr Treasury)',
            'tyx': '^TYX (30yr Treasury)',
            'gold': 'GC=F (Gold)',
            'jpyx': 'JPY=X (Yen)'
        }
        for mkt in MARKET_FEATURES:
            print(f"   • {market_map.get(mkt, mkt)}")
    
    # Load or generate sequences (with caching!)
    seq_len = 20
    X_seq, y_seq, dates_seq, ldp_weights_seq, returns_seq, feature_names = load_or_generate_sequences(
        tickers=tickers,
        config_preset=CONFIG_PRESET,
        barrier_params=BARRIER_PARAMS,
        market_features=MARKET_FEATURES,  # Add market features here
        seq_len=seq_len,
        data_path=data_path,
        cache_dir='cache',
        use_cache=True,
        verbose=True
    )
    
    print(f"\n✅ Sequences ready: {X_seq.shape}")
    print(f"   Features: {X_seq.shape[2]}")
    print(f"   Date range: {dates_seq.min().date()} to {dates_seq.max().date()}")
    print(f"   LdP weights: range=[{ldp_weights_seq.min():.3f}, {ldp_weights_seq.max():.3f}]")
    
    # Map labels: {-1, 0, 1} → {0, 1, 2} for 3-class classification
    print(f"\n📋 Mapping labels for 3-class classification:")
    print(f"   Before mapping: {np.unique(y_seq, return_counts=True)}")
    
    y_seq = y_seq - np.min(y_seq)  # Shift minimum to 0
    
    print(f"   After mapping:  {np.unique(y_seq, return_counts=True)}")
    print(f"   Label interpretation: [0=DOWN/SL, 1=TIMEOUT, 2=UP/TP]")
    
    # Calculate output bias for imbalanced classes
    print(f"\n📊 Calculating output bias from label distribution...")
    unique_labels, label_counts = np.unique(y_seq, return_counts=True)
    label_freqs = label_counts / label_counts.sum()
    
    # Initial bias = log(class_frequency)
    # This helps the model start with reasonable class predictions
    output_bias = np.log(label_freqs)
    
    print(f"   Class frequencies:")
    for i, (label, freq, bias) in enumerate(zip(unique_labels, label_freqs, output_bias)):
        label_name = {0: "DOWN/SL", 1: "TIMEOUT", 2: "UP/TP"}[label]
        print(f"     {label_name}: {freq:.3f} (bias={bias:.4f})")
    
    print(f"   Output bias will initialize final layer to match class distribution")
    
    # 3. Train/validation split
    print("\n3. TRAIN/VALIDATION SPLIT")
    print("-"*40)
    
    split_idx = int(len(X_seq) * 0.8)
    
    X_train = X_seq[:split_idx]
    X_val = X_seq[split_idx:]
    
    # Normalize
    print("\n3.5. NORMALIZING DATA")
    print("-"*40)
    
    from fin_utils import normalize_for_training
    
    normalizer_path = os.path.join(experiment_save_path, 'normalizer.pkl')
    X_train, X_val, normalizer = normalize_for_training(
        X_train, X_val,
        method='minmax',
        save_path=normalizer_path
    )
    
    print(f"✅ Normalizer saved to {normalizer_path}")
    
    # DIAGNOSTIC: Check normalized features
    print(f"\n{'='*40}")
    print("NORMALIZED FEATURES CHECK")
    print(f"{'='*40}")
    print(f"X_train:")
    print(f"  NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(f"  Shape: {X_train.shape}")
    print(f"X_val:")
    print(f"  NaN: {np.isnan(X_val).sum()}, Inf: {np.isinf(X_val).sum()}")
    print(f"  Shape: {X_val.shape}")
    
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print(f"⚠️  WARNING: Found NaN/Inf in X_train, replacing with 0")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(X_val).any() or np.isinf(X_val).any():
        print(f"⚠️  WARNING: Found NaN/Inf in X_val, replacing with 0")
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert to categorical (3 classes)
    y_train = tf.keras.utils.to_categorical(y_seq[:split_idx], num_classes=3)
    y_val = tf.keras.utils.to_categorical(y_seq[split_idx:], num_classes=3)
    
    train_dates = dates_seq[:split_idx]
    val_dates = dates_seq[split_idx:]
    
    # Calculate sample weights using López de Prado method
    print("\n3.6. APPLYING SAMPLE WEIGHTS (LÓPEZ DE PRADO)")
    print("-"*40)
    
    from sklearn.utils.class_weight import compute_sample_weight
    
    simple_weights_train = compute_sample_weight(
        class_weight='balanced',
        y=y_seq[:split_idx]  # Training labels only
    )
    
    simple_weights_val = compute_sample_weight(
        class_weight='balanced',
        y=y_seq[split_idx:]  # Validation labels only
    )
    
    # Use LdP weights from sequence preparation (already aligned and filtered!)
    print(f"📊 Using López de Prado weights from sequence preparation...")
    ldp_weights_train = ldp_weights_seq[:split_idx]
    ldp_weights_val = ldp_weights_seq[split_idx:]
    
    
    # Combine with class balancing (LdP weights × class weights)
    sample_weights_train = ldp_weights_train * simple_weights_train
    sample_weights_val = ldp_weights_val * simple_weights_val
    
    # Normalize
   # sample_weights_train = sample_weights_train / sample_weights_train.mean()
   # sample_weights_val = sample_weights_val / sample_weights_val.mean()
    
    print(f"\n✅ Final weights (LdP × class balance):")
    print(f"   Range: [{sample_weights_train.min():.3f}, {sample_weights_train.max():.3f}]")
    print(f"   Mean: {sample_weights_train.mean():.3f}, Std: {sample_weights_train.std():.3f}")

    print(f"Class distribution (training):")
    for label in [0, 1, 2]:
        count = (y_seq[:split_idx] == label).sum()
        pct = count / len(y_seq[:split_idx]) * 100
        label_name = {0: "DOWN", 1: "UP", 2: "NEUTRAL"}[label]
        # Get weight for this label from sample_weights_train
        weight = sample_weights_train[y_seq[:split_idx] == label][0]
        print(f"  Label {label:2d} ({label_name}): {count:5d} ({pct:5.2f}%) → weight={weight:.4f}")
     
    print(f"\nClass distribution (training):")
    for label in [0, 1, 2]:
        count = (y_seq[:split_idx] == label).sum()
        pct = count / len(y_seq[:split_idx]) * 100
        label_name = {0: "DOWN/SL", 1: "TIMEOUT", 2: "UP/TP"}[label]
        avg_weight = sample_weights_train[y_seq[:split_idx] == label].mean()
        try:
            avg_unique = ldp_weights_train[y_seq[:split_idx] == label].mean()
            print(f"  Label {label} ({label_name}): {count:5d} ({pct:5.2f}%) → avg_weight={avg_weight:.4f} (uniqueness={avg_unique:.4f})")
        except:
            print(f"  Label {label} ({label_name}): {count:5d} ({pct:5.2f}%) → avg_weight={avg_weight:.4f}")
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"  Date range: {train_dates.min().date()} to {train_dates.max().date()}")
    print(f"Validation samples: {len(X_val)}")
    print(f"  Date range: {val_dates.min().date()} to {val_dates.max().date()}")
    
    # 4. Build and train model
    print("\n4. BUILDING MODEL")
    print("-"*40)
    
    
    model = build_enhanced_cnn_lstm(
        input_shape=(seq_len, X_seq.shape[2]),  # X_seq shape is (N, seq_len, features)
        n_classes=model_parameters['n_classes'],
        wavenet_filters=model_parameters['wavenet_filters'],
        wavenet_blocks=model_parameters['wavenet_blocks'],
       # wavenet_layers_per_block=model_parameters['wavenet_layers_per_block'],
      #  conv_filters=model_parameters['conv_filters'],
      #  lstm_units=model_parameters['lstm_units'],
      #  attention_units=model_parameters['attention_units'],
        dropout_rate=model_parameters['dropout_rate'],
        l2_reg=model_parameters['l2_reg'],
        output_bias=output_bias  # Initialize with class distribution
    )
    
    model.summary()
    print(f"Model parameters: {model.count_params():,}")
    
    # Compile
    optimizer = optimizers.Adam(
        learning_rate=training_parameters['learning_rate'],
        beta_1=training_parameters['beta_1'],
        beta_2=training_parameters['beta_2'],
       # clipnorm=training_parameters.get('clipnorm', 1.0)
    )
    

    
    # Categorical crossentropy with label smoothing (reduces overconfidence)
    label_smoothing = training_parameters.get('label_smoothing', 0.1)
    categorical_loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing,
        name='categorical_crossentropy'
    )

    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
        alpha=0.25,
        gamma=1.5,
        from_logits=False,
        label_smoothing=label_smoothing,
        name='focal_loss'
    ) 
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss,  # Single loss for direction
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
        weighted_metrics=['accuracy']
    )
    
    print("\n5. TRAINING")
    print("-"*40)
    print(f"🚀 Starting training with MLFinLab triple barriers...")
    print(f"   Loss Function: Focal Loss (label_smoothing={label_smoothing})")
    print(f"   Sample Weights: López de Prado uniqueness × class balance")
    print(f"   3-Class Classification: DOWN/SL, TIMEOUT, UP/TP")
    print(f"   Focus: Direction signals only (position sizing via Bayesian Kelly)")
    
    # Create callbacks
    callback_list = create_callbacks(experiment_save_path, callback_parameters)
    
    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_train,
        validation_data=(X_val, y_val, sample_weights_val),
        batch_size=training_parameters['batch_size'],
        epochs=training_parameters['max_epochs'],
        callbacks=callback_list,
        verbose=2
    )
    
    # 6. Save artifacts
    print("\n6. SAVING ARTIFACTS")
    print("-"*40)
    
    # Save model
    model.save(f'{experiment_save_path}/final_model.keras')
    print(f"✅ Model saved to {experiment_save_path}/final_model.keras")
    
    # Save metadata
    metadata = {
        'tickers': tickers,
        'config_preset': CONFIG_PRESET,
        'market_features': MARKET_FEATURES,
        'barrier_params': BARRIER_PARAMS,
        'training_params': training_parameters,
        'model_parameters': model_parameters,
        'dataset_info': {
            'total_samples': len(X_seq),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'features': X_seq.shape[2],  # Number of features
            'feature_names': feature_names,  # Actual feature names
            'date_range': (str(dates_seq.min().date()), str(dates_seq.max().date()))
        }
    }
    
    with open(f'{experiment_save_path}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Metadata saved to {experiment_save_path}/metadata.pkl")
    
    # Note: events data is embedded in the cached sequence file
    # To access events, regenerate from fin_load_and_sequence.py if needed
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {experiment_save_path}/")
    print(f"  - final_model.keras")
    print(f"  - best_model.keras")
    print(f"  - training_log.csv")
    print(f"  - metadata.pkl")
    print(f"  - normalizer.pkl")
    
    return model, history, metadata


if __name__ == "__main__":
    model, history, metadata = main()
