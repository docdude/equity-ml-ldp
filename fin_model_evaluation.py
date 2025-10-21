

# fin_training.py
"""
Production training script with proper López de Prado methods
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier


from fin_feature_preprocessing import EnhancedFinancialFeatures
from fin_model import build_enhanced_cnn_lstm
from feature_config import FeatureConfig

# Import enhanced López de Prado evaluator

from lopez_de_prado_evaluation import LopezDePradoEvaluator

def main(model=None, model_name="RandomForest"):
    """
    Complete evaluation pipeline with López de Prado methods
    
    Args:
        model: Pre-trained model to evaluate (sklearn or keras)
               If None, creates a RandomForestClassifier
        model_name: Name for logging/display purposes
    """
    print("="*80)
    print("MODEL EVALUATION PIPELINE")
    print("Based on López de Prado's Methods")
    print("="*80)
    print(f"\n📊 Model: {model_name}")
    
    # 1. Load and preprocess data
    print("\n1. ENHANCED FEATURE ENGINEERING")
    print("-"*40)
    
    tickers = ['AAPL', 'DELL', 'JOBY', 'LCID', 'SMCI', 'NVDA', 'TSLA', 'WDAY', 'AMZN', 'AVGO', 'SPY']
    CONFIG_PRESET = 'comprehensive'
    # Setup feature engineering with presets or custom config
    config = FeatureConfig.get_preset('comprehensive')

    feature_engineer = EnhancedFinancialFeatures(feature_config=config)
    print(f"\n📋 Using {CONFIG_PRESET} feature preset for evaluation")
    feature_engineer.print_config()
    
    all_features = []
    all_labels = []
    all_dates = []
    all_prices = []
    all_tickers = []
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        
        # Load raw data
        df = pd.read_parquet(f'data_raw/{ticker}.parquet')
        
        # Set date as index (critical for proper date tracking!)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Create comprehensive features
        features = feature_engineer.create_all_features(df)
        
        # Create dynamic triple barriers (now includes exit_return!)
        barriers = feature_engineer.create_dynamic_triple_barriers(df)
        
        # Combine
        combined = pd.concat([features, barriers], axis=1)
        combined = combined.dropna()
        
        # Get aligned prices for forward return calculation
        prices_aligned = df.loc[combined.index, ['close']]
        
        # Store features including exit_return from barriers
        # CRITICAL: Use exit_return instead of fixed-horizon forward_return_5d
        # This aligns with López de Prado / mlfinlab method
        ticker_features = combined[features.columns].copy()
        
        all_labels.append(combined['label'])
        all_prices.append(prices_aligned)
        all_dates.extend(combined.index)
        
        # Track ticker for each row (to prevent cross-ticker forward returns)
        all_tickers.extend([ticker] * len(combined))
        
        # NEW: Use exit_return from barriers (actual return at barrier touch)
        # This is the mlfinlab approach - returns are calculated at EXIT time
        ticker_features['exit_return'] = combined['exit_return'].values
        ticker_features['exit_day'] = combined['exit_day'].values  # For analysis
        ticker_features['ticker'] = ticker
        
        all_features.append(ticker_features)
    
    # Combine all data
    X = pd.concat(all_features)
    y = pd.concat(all_labels)
    prices = pd.concat(all_prices)
    dates = pd.DatetimeIndex(all_dates)
    tickers = pd.Series(all_tickers)
    
    # CRITICAL: Sort by date for proper time-series cross-validation
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    prices = prices.iloc[sort_idx].reset_index(drop=True)
    dates = dates[sort_idx]
    tickers = tickers.iloc[sort_idx].reset_index(drop=True)
    
    print("\n⚠️  Data sorted chronologically for proper CV")
    print("✅ Exit returns calculated per-ticker at barrier touch (mlfinlab method)")
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Features: {X.shape[1]} (including ticker and exit_return)")
    print(f"Samples: {X.shape[0]}")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # 2. Prepare sequences (for deep learning)
    print("\n2. SEQUENCE PREPARATION")
    print("-"*40)
    
    seq_len = 20
    
    # Separate ticker, exit_return, and exit_day from model input features
    # These are targets/metadata, not features for prediction
    model_feature_cols = [col for col in X.columns 
                          if col not in ['ticker', 'exit_return', 'exit_day']]
    X_features = X[model_feature_cols]
    print(f"Using {len(model_feature_cols)} features for model input")
    
    X_sequences = []
    y_sequences = []
    dates_sequences = []
    
    for i in range(len(X_features) - seq_len):
        X_sequences.append(X_features.iloc[i:i+seq_len].values)
        y_sequences.append(y.iloc[i+seq_len-1])
        dates_sequences.append(dates[i+seq_len-1])
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    dates_seq = pd.DatetimeIndex(dates_sequences)
    
    print(f"Sequences shape: {X_seq.shape}")
    print(f"Sequence date range: {dates_seq.min().date()} to {dates_seq.max().date()}")
    
    # 3. López de Prado Evaluation
    print("\n3. LÓPEZ DE PRADO EVALUATION")
    print("-"*40)
    
    evaluator = LopezDePradoEvaluator(embargo_pct=0.02, n_splits=5)
    
    # Convert sequences to DataFrame for evaluator
    # Flatten 3D sequences to 2D for RandomForest
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    
    # Create proper column names: feature_name_t0, feature_name_t1, etc.
    # Use model_feature_cols (excludes ticker and forward_return_5d)
    feature_names = []
    for t in range(seq_len):
        for col in model_feature_cols:
            feature_names.append(f'{col}_t{t}')
    
    X_df = pd.DataFrame(X_flat, index=dates_seq, columns=feature_names)
    y_series = pd.Series(y_seq, index=dates_seq)
    
    # Check if model is keras (has predict method) or sklearn
    is_keras = model is not None and hasattr(model, 'layers')
    
    # Create model if not provided
    if model is None:
        print("\n⚠️  No model provided, creating RandomForest proxy...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        is_keras = False
        
        # Train the model on training set before generating predictions
        print("   Training RandomForest on training set...")
        train_split = int(len(X_seq) * 0.8)
        X_train_flat = X_seq[:train_split].reshape(X_seq[:train_split].shape[0], -1)
        y_train = y_seq[:train_split]
        model.fit(X_train_flat, y_train)
        print(f"   ✅ Model trained on {len(X_train_flat)} samples")
    else:
        print(f"\n✅ Using provided {model_name} model")
        print(f"   Model type: {'Keras/TensorFlow' if is_keras else 'Sklearn'}")
    
    # Generate strategy returns from model predictions
    print("\n📊 Generating strategy returns from model predictions...")
    
    # Get model predictions for validation set (last 20%)
    val_split = int(len(X_seq) * 0.8)
    X_val = X_seq[val_split:]
    
    # CRITICAL: Use exit_return from barriers (mlfinlab method)
    # These are actual returns at barrier touch time, NOT fixed-horizon returns
    # This aligns labels with returns for proper backtesting
    exit_returns_all = X['exit_return'].values
    
    # Get exit returns for validation set (accounting for seq_len offset)
    val_start_idx = val_split + seq_len
    forward_returns = exit_returns_all[val_start_idx:val_start_idx + len(X_val)]
    
    print(f"Validation samples: {len(X_val)}")
    print(f"Exit returns available: {len(forward_returns)}")
    print(f"Mean exit return: {forward_returns.mean():.6f}, Std: {forward_returns.std():.6f}")
    
    if is_keras:
        # Keras CNN-LSTM - use raw predictions
        predictions = model.predict(X_val, verbose=0)
        
        # Extract direction probabilities (multi-output model)
        if isinstance(predictions, dict):
            direction_probs = predictions['direction']
        elif isinstance(predictions, list):
            direction_probs = predictions[0]  # First output
        else:
            direction_probs = predictions
        
        # Truncate predictions to match available forward returns
        direction_probs = direction_probs[:len(forward_returns)]
        
        print(f"Aligned {len(forward_returns)} predictions with actual returns")
        print(f"Forward returns - mean: {forward_returns.mean():.6f}, std: {forward_returns.std():.6f}")
        
        # Create strategies with different confidence thresholds
        n_strategies = 10
        strategy_returns = []
        min_confidences = np.linspace(0.33, 0.85, n_strategies)
        
        for min_conf in min_confidences:
            # Get predicted class (argmax)
            pred_class = direction_probs.argmax(axis=1)
            max_prob = direction_probs.max(axis=1)
            
            # Only take positions when confidence > min_conf
            positions = np.where(
                (pred_class == 2) & (max_prob > min_conf), 
                max_prob - 0.33,  # Long: scaled by confidence above neutral
                np.where(
                    (pred_class == 0) & (max_prob > min_conf),
                    -(max_prob - 0.33),  # Short: scaled by confidence
                    0  # No position (neutral or low confidence)
                )
            )
            
            # Apply positions to actual forward returns
            returns = positions * forward_returns
            strategy_returns.append(returns)
    else:
        # Sklearn - flatten sequences to 2D for RandomForest
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_val_flat)
            
            # Truncate to match available forward returns
            probs = probs[:len(forward_returns)]
            
            print(f"Aligned {len(forward_returns)} predictions with actual returns")
            print(f"Forward returns - mean: {forward_returns.mean():.6f}, std: {forward_returns.std():.6f}")
            
            n_strategies = 10
            strategy_returns = []
            min_confidences = np.linspace(0.33, 0.85, n_strategies)
            
            for min_conf in min_confidences:
                # Get predicted class (argmax) and probabilities
                pred_class = probs.argmax(axis=1)
                max_prob = probs.max(axis=1)
                
                # CRITICAL: sklearn RandomForest with labels {-1, 0, 1} creates:
                #   rf.classes_ = [-1, 0, 1]  (sorted label values!)
                #
                # predict_proba returns 3 columns:
                #   Column 0 → P(label = -1) [NEUTRAL/TIMEOUT]
                #   Column 1 → P(label = 0)  [DOWN/STOP LOSS]
                #   Column 2 → P(label = 1)  [UP/TAKE PROFIT]
                #
                # When we do probs.argmax(axis=1), we get class indices:
                #   pred_class == 0 → predicting label -1 (neutral)
                #   pred_class == 1 → predicting label 0 (down)
                #   pred_class == 2 → predicting label 1 (up)
                
                # Only take positions when confidence > min_conf
                positions = np.where(
                    (pred_class == 2) & (max_prob > min_conf), 
                    max_prob - 0.33,  # Long when predicting UP (class 2 = label 1)
                    np.where(
                        (pred_class == 1) & (max_prob > min_conf),
                        -(max_prob - 0.33),  # Short when predicting DOWN (class 1 = label 0)
                        0  # No position (neutral = class 0, or low confidence)
                    )
                )
                
                # Apply positions to actual forward returns
                returns = positions * forward_returns
                strategy_returns.append(returns)
        else:
            # Fallback to synthetic if no probabilities available
            print("⚠️  Model has no predict_proba, using synthetic returns")
            np.random.seed(42)
            n_strategies = 20
            n_observations = len(forward_returns)
            strategy_returns = []
            for i in range(n_strategies):
                sharpe = np.random.uniform(-0.5, 2.0)
                daily_return = sharpe * 0.01 / np.sqrt(252)
                returns = np.random.normal(daily_return, 0.01, n_observations)
                strategy_returns.append(returns)

    # Convert to DataFrame (observations x strategies)
    returns_df = pd.DataFrame(strategy_returns).T
    returns_df.columns = [f'Strategy_{i+1}' for i in range(len(strategy_returns))]

    print(f"Created returns matrix: {returns_df.shape}")
    print(f"Strategies: {len(strategy_returns)}, Observations: {len(returns_df)}")
    
    # Calculate Sharpe ratios for all strategies
    strategy_sharpes = returns_df.mean() / returns_df.std() * np.sqrt(252)  # Annualized Sharpe
    mean_sharpe = strategy_sharpes.mean()
    
    print(f"\nMean return per strategy: {returns_df.mean().mean():.6f}")
    print(f"Mean Sharpe ratio: {mean_sharpe:.4f}")
    
    # Run appropriate evaluation based on model type
    results = {}
    
    if is_keras:
        # For Keras models (already trained), only run PBO analysis
        print("\n� Running PBO Analysis (Keras model - skip retraining steps)...")
        print("-"*40)
        
        pbo_result = evaluator.probability_backtest_overfitting(
            strategy_returns=returns_df.values,
            n_splits=16
        )
        # Add Sharpe ratio to results for final assessment
        pbo_result['mean_sharpe'] = mean_sharpe
        pbo_result['strategy_sharpes'] = strategy_sharpes
        results['pbo'] = pbo_result
        
        print(f"\n📊 PBO RESULTS:")
        print(f"   PBO: {pbo_result['pbo']:.4f}")
        print(f"   Probability of OOS Loss: {pbo_result.get('prob_oos_loss', 0):.4f}")
        
    else:
        # For sklearn models, run comprehensive evaluation
        print("\n� Running Comprehensive López de Prado Evaluation...")
        results = evaluator.comprehensive_evaluation(
            model=model,
            X=X_df,
            y=y_series,
            pred_times=dates_seq,
            sample_weights=None,
            strategy_returns=returns_df.values  # Pass numpy array (obs x strat)
        )
    
    # Save predictions from PCV
    if 'pcv' in results and 'predictions' in results['pcv']:
        pred_file = 'artifacts/pcv_predictions.parquet'
        Path('artifacts').mkdir(exist_ok=True)
        results['pcv']['predictions'].to_parquet(pred_file, index=False)
        print(f"\n💾 Saved PCV predictions to {pred_file}")
    
    # Save predictions from walk-forward
    if 'walk_forward' in results and 'predictions' in results['walk_forward']:
        pred_file = 'artifacts/walkforward_predictions.parquet'
        Path('artifacts').mkdir(exist_ok=True)
        results['walk_forward']['predictions'].to_parquet(pred_file)
        print(f"💾 Saved Walk-Forward predictions to {pred_file}")
    
    # Additional interpretation
    print("\n" + "="*80)
    print("📋 DETAILED INTERPRETATION")
    print("="*80)
    
    if 'pcv' in results:
        mean_auc = results['pcv']['mean_score']
        if mean_auc < 0.52:
            print("\n⚠️  PURGED CV: POOR - No signal detected (AUC ~= random)")
        elif mean_auc < 0.55:
            print("\n⚠️  PURGED CV: WEAK - Marginal signal, may not be tradeable")
        elif mean_auc < 0.60:
            print("\n✅ PURGED CV: MODERATE - Clear signal, worth pursuing")
        else:
            print("\n✅ PURGED CV: STRONG - Excellent signal!")
    
    if 'walk_forward' in results:
        wf_scores = results['walk_forward']['auc_scores']
        wf_score_range = max(wf_scores) - min(wf_scores)
       
        if wf_score_range > 0.15:
            print(f"⚠️  WALK-FORWARD: HIGH VARIABILITY - Score range = {min(wf_scores):.3f} - {max(wf_scores):.3f}")
            print("   Model may not generalize across time")
        else:
            print(f"✅ WALK-FORWARD: STABLE - Score range = {min(wf_scores):.3f} - {max(wf_scores):.3f}")
    
    if 'feature_importance' in results:
        print("\n📊 TOP 10 FEATURES (MDI):")
        top_features = results['feature_importance']['mdi'].head(10)
        
        for i, (feat, imp) in enumerate(top_features.items(), 1):
            print(f"   {i:2d}. {feat}: {imp:.4f}")
        
        # Sanity checks
        mdi_scores = results['feature_importance']['mdi']
        if mdi_scores.max() > 0.5:
            print(f"\n   ⚠️  WARNING: One feature dominates "
                  f"({mdi_scores.idxmax()}: {mdi_scores.max():.3f})")
            print("      Check for data leakage!")
         

    # 5. Final Assessment
    print("\n5. FINAL ASSESSMENT")
    print("-"*40)
    
    # For Keras models, use training AUC from model itself (if we skip PCV)
    # For sklearn models, use PCV AUC
    pcv_auc = results.get('pcv', {}).get('mean_score', None)
    pbo = results.get('pbo', {}).get('pbo', 1.0)
    mean_sharpe = results.get('pbo', {}).get('mean_sharpe', 0)
    
    # If no PCV (Keras model), use alternative criteria
    if pcv_auc is None:
        # For Keras: assess based on PBO and strategy Sharpe
        if pbo < 0.5 and mean_sharpe > 0.5:
            print("✅ Model shows promise - proceed with careful live testing")
            print(f"   - PBO: {pbo:.3f} (acceptable overfitting risk)")
            print(f"   - Mean Sharpe: {mean_sharpe:.3f}")
            print("   - Implement with small position sizes")
            print("   - Monitor for regime changes")
        elif pbo < 0.5:
            print("⚠️  Model shows potential but needs improvement")
            print(f"   - PBO: {pbo:.3f} (acceptable overfitting risk)")
            print(f"   - Mean Sharpe: {mean_sharpe:.3f} (low profitability)")
            print("   - Consider improving strategy parameters")
            print("   - Test different confidence thresholds")
        else:
            print("❌ Model not ready for deployment")
            print(f"   - PBO: {pbo:.3f} (high overfitting risk)")
            print("   - Review feature engineering")
            print("   - Consider different architectures")
    else:
        # For sklearn: use original PCV + PBO check
        if pcv_auc > 0.65 and pbo < 0.5:
            print("✅ Model shows promise - proceed with careful live testing")
            print("   - Implement with small position sizes")
            print("   - Monitor for regime changes")
            print("   - Use ensemble with other models")
        else:
            print("❌ Model not ready for deployment")
            print("   - Review feature engineering")
            print("   - Consider different architectures")
            print("   - Collect more diverse data")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
