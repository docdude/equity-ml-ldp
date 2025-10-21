"""
Sequence Loading and Generation with Caching
=============================================

Extracts sequence generation logic from fin_training_ldp.py with caching support.

Usage:
    from fin_load_and_sequence import load_or_generate_sequences
    
    X_seq, y_seq, dates_seq, ldp_weights_seq = load_or_generate_sequences(
        tickers=['AAPL', 'NVDA'],
        config_preset='wavenet_optimized',
        barrier_params={'pt_sl': [2, 1], 'min_ret': 0.005, 'num_days': 7, 'lookback': 60},
        seq_len=20,
        data_path='data_raw',
        use_cache=True
    )
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

# Add MLFinance to path
mlfinance_path = os.path.join(os.path.dirname(__file__), 'MLFinance')
if mlfinance_path not in sys.path:
    sys.path.insert(0, mlfinance_path)

from fin_feature_preprocessing import EnhancedFinancialFeatures
from feature_config import FeatureConfig
from fin_market_context import (
    load_market_data,
    create_market_context_features,
    add_market_context_to_ticker
)

# MLFinLab imports
from FinancialMachineLearning.labeling.labeling import (
    add_vertical_barrier, 
    get_events, 
    meta_labeling
)
from FinancialMachineLearning.features.volatility import daily_volatility
from FinancialMachineLearning.sample_weights.concurrency import average_uniqueness_triple_barrier

class FinancialHeatmapAugmentation:
    """
    Adapt YOUR stroke heatmap augmentation for financial events
    This is your BRILLIANT innovation - Gaussian smoothing of labels!
    """
    
    def __init__(self, event_sigma=2.0):
        """
        event_sigma: Like your stroke_sigma - controls smoothing
        """
        self.event_sigma = event_sigma
        
    def augment_events_to_heatmaps(self, events, sigma=None):
        """
        YOUR EXACT METHOD adapted for financial events
        Instead of stroke events, we have TP/SL hit events
        """
        if sigma is None:
            sigma = self.event_sigma
            
        print(f"Augmenting events to heatmaps with σ = {sigma} days")
        
        # Build Gaussian kernel (YOUR approach)
        radius = int(np.ceil(3 * sigma))
        x = np.arange(-radius, radius+1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma)**2)
        # Note: Not normalizing (following YOUR implementation)
        
        # Convolve with Gaussian
        heat = np.convolve(events, kernel, mode='same')
        
        # Clip to [0,1] for overlapping events
        heat = np.clip(heat, 0.0, 1.0)
        
        return heat
    
    def create_signed_heatmap(self, tp_events, sl_events, sigma=None):
        """
        Extension of your method for signed events
        Positive for TP, negative for SL
        """
        if sigma is None:
            sigma = self.event_sigma
            
        # Create separate heatmaps
        tp_heat = self.augment_events_to_heatmaps(tp_events, sigma)
        sl_heat = self.augment_events_to_heatmaps(sl_events, sigma)
        
        # Combine: positive for TP, negative for SL
        signed_heat = tp_heat - sl_heat
        
        # Clip to [-1, 1]
        signed_heat = np.clip(signed_heat, -1.0, 1.0)
        
        return signed_heat

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

def load_or_generate_sequences(
    tickers: List[str],
    config_preset: str = 'wavenet_optimized',
    barrier_params: Dict = None,
    seq_len: int = 20,
    data_path: str = 'data_raw',
    cache_dir: str = 'cache',
    use_cache: bool = True,
    verbose: bool = True,
    market_features: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Load sequences from cache or generate them
    
    Args:
        tickers: List of ticker symbols
        config_preset: Feature configuration preset name
        barrier_params: Triple barrier parameters dict
        seq_len: Sequence length
        data_path: Path to raw data files
        cache_dir: Cache directory
        use_cache: Whether to use caching
        verbose: Print detailed progress
        market_features: List of market features to include. 
                        Options: ['spy', 'vix', 'fvx', 'tyx', 'gold', 'jpyx']
                        If None, no market features added.
                        Example: ['spy', 'vix'] for just S&P500 and VIX
    
    Returns:
        X_seq: Feature sequences (N, seq_len, n_features)
        y_seq: Labels (N,) with values {-1, 0, 1}
        dates_seq: DatetimeIndex (N,)
        ldp_weights_seq: López de Prado weights (N,)
        returns_seq: Actual barrier returns (N,) from meta-labeling
        feature_names: List of feature names (length n_features)
    """
    
    # Default barrier params
    if barrier_params is None:
        barrier_params = {
            'lookback': 60,
            'pt_sl': [2, 1],
            'min_ret': 0.005,
            'num_days': 7,
            'num_threads': 6
        }
    
    # Setup cache
    os.makedirs(cache_dir, exist_ok=True)
    ticker_hash = '_'.join(sorted(tickers))[:50]
    
    # Include market features in cache key
    mkt_suffix = '_'.join(sorted(market_features)) if market_features else 'nomkt'
    
    cache_file = (
        f"{cache_dir}/seq_{ticker_hash}_{config_preset}"
        f"_pt{barrier_params['pt_sl'][0]}_sl{barrier_params['pt_sl'][1]}"
        f"_d{barrier_params['num_days']}_len{seq_len}_{mkt_suffix}.parquet"
    )
    
    # Try loading cache
    if use_cache and os.path.exists(cache_file):
        if verbose:
            print(f"\n🗄️  Loading cached sequences: {cache_file}")
            print(f"   ⚡ Skipping feature engineering and barrier generation")
        
        try:
            df_cache = pd.read_parquet(cache_file)
            
            X_seq = np.array([np.array(eval(x)) for x in df_cache['X_seq']])
            y_seq = df_cache['y'].values
            dates_seq = pd.DatetimeIndex(df_cache['date'])
            ldp_weights_seq = df_cache['ldp_weight'].values
            returns_seq = df_cache['return'].values
            
            if verbose:
                print(f"✅ Loaded {len(X_seq)} sequences from cache")
                print(f"   Shape: {X_seq.shape}")
                print(f"   Date range: {dates_seq.min().date()} to {dates_seq.max().date()}")
                print(f"   LdP weights: range=[{ldp_weights_seq.min():.3f}, {ldp_weights_seq.max():.3f}]")
                print(f"   Returns: range=[{returns_seq.min():.4f}, {returns_seq.max():.4f}]")
            
            # Reconstruct feature names (same logic as generation path)
            config = FeatureConfig.get_preset(config_preset)
            
            # Check if config has explicit feature_list
            if 'feature_list' in config and config['feature_list']:
                feature_names = config['feature_list'].copy()
            else:
                # Build from enabled feature groups
                feature_names = []
                for group_name, enabled in config.items():
                    if enabled is True and group_name in FeatureConfig.FEATURE_GROUPS:
                        feature_names.extend(FeatureConfig.FEATURE_GROUPS[group_name]['features'])
            
            # Add market feature names if present
            if market_features:
                # Market feature mapping: internal key -> ticker symbol
                market_map = {
                    'spy': '^GSPC',
                    'vix': '^VIX',
                    'fvx': '^FVX',
                    'tyx': '^TYX',
                    'gold': 'GC=F',
                    'jpyx': 'JPY=X'
                }
                for mkt_feat in market_features:
                    if mkt_feat in market_map:
                        feature_names.append(market_map[mkt_feat])
            
            if verbose:
                print(f"   Reconstructed {len(feature_names)} feature names")
            
            return X_seq, y_seq, dates_seq, ldp_weights_seq, returns_seq, feature_names
            
        except Exception as e:
            if verbose:
                print(f"⚠️  Cache load failed: {e}")
                print(f"   Regenerating sequences...")
    
    # Generate sequences
    if verbose:
        print(f"\n🔄 Generating sequences (no cache)")
        print(f"   Tickers: {len(tickers)}")
        print(f"   Config: {config_preset}")
        print(f"   Barriers: pt={barrier_params['pt_sl'][0]}, sl={barrier_params['pt_sl'][1]}, days={barrier_params['num_days']}")
    
    # Setup feature engineering
    config = FeatureConfig.get_preset(config_preset)
    feature_engineer = EnhancedFinancialFeatures(feature_config=config)
    
    # Load market context features (once for all tickers) if requested
    market_context_df = None
    if market_features:
        if verbose:
            print(f"\n📊 Loading market context features: {market_features}")
        try:
            market_data = load_market_data(data_path)
            market_context_df = create_market_context_features(market_data, selected_features=market_features)
            if verbose:
                print(f"   ✓ Market context loaded: {market_context_df.shape[1]} features")
                print(f"   Features: {list(market_context_df.columns)}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Warning: Could not load market context: {e}")
                print(f"   Continuing without market features...")
            market_context_df = None
    else:
        if verbose:
            print(f"\n📊 No market features requested - using only ticker-specific features")
    
    # Initialize collectors
    all_features = []
    all_labels = []
    all_dates = []
    all_tickers = []
    all_events = []
    all_ldp_weights = []
    all_returns = []
    
    # Process each ticker
    for ticker in tickers:
        if verbose:
            print(f"\nProcessing {ticker}...")
        
        # Load raw data
        df = pd.read_parquet(f'{data_path}/{ticker}.parquet')
        
        # Set date as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Ensure proper column names
        if 'close' in df.columns:
            df['Close'] = df['close']
        
        # Clean data
        df = df.dropna(subset=['Close'])
        
        if len(df) < 100:
            if verbose:
                print(f"  ⚠️  Skipping {ticker}: Only {len(df)} valid rows")
            continue
        
        # Create features
        features = feature_engineer.create_all_features(df)
        
        # Add market context features if available
        if market_context_df is not None:
            if verbose:
                print(f"  📈 Adding market context ({market_context_df.shape[1]} features)...")
            original_shape = features.shape
            features = add_market_context_to_ticker(features, market_context_df)
            if verbose:
                print(f"     {original_shape[1]} → {features.shape[1]} features")
        
        # Drop NaN values from features (including market context)
        # This handles: pct_change NaNs, missing market data, alignment issues
        nan_before = features.isna().any(axis=1).sum()
        features = features.dropna()
        if verbose and nan_before > 0:
            print(f"     Dropped {nan_before} rows with NaN values")
        
        # Create triple barriers
        try:
            events, labels = create_mlfinlab_barriers(
                df,
                **barrier_params
            )
        except Exception as e:
            if verbose:
                print(f"  ❌ Error creating barriers: {e}")
            continue
        
        # Align features with labels
        common_index = features.index.intersection(labels.index)
        
        if len(common_index) == 0:
            if verbose:
                print(f"  ⚠️  Skipping: No common index")
            continue
        
        features_aligned = features.loc[common_index]
        labels_aligned = labels.loc[common_index]
        events_aligned = events.loc[common_index]
        prices_aligned = df.loc[common_index, 'Close']
        
        # Calculate López de Prado weights per-ticker
        if verbose:
            print(f"  📊 Calculating LdP weights...")
        
        try:
            uniqueness_df = average_uniqueness_triple_barrier(
                triple_barrier_events=events_aligned,
                close_series=prices_aligned,
                num_threads=4
            )
            
            avg_unique = uniqueness_df['tW']
            ldp_weight = avg_unique / avg_unique.mean()
            
            if verbose:
                print(f"     Uniqueness: mean={avg_unique.mean():.3f}, range=[{avg_unique.min():.3f}, {avg_unique.max():.3f}]")
            
        except Exception as e:
            if verbose:
                print(f"     ⚠️  Failed: {e}, using uniform weights")
            ldp_weight = pd.Series(1.0, index=common_index)
        
        # Store data
        all_features.append(features_aligned)
        all_labels.append(labels_aligned['bin'])
        all_returns.append(labels_aligned['ret'])
        all_dates.extend(common_index)
        all_tickers.extend([ticker] * len(common_index))
        all_events.append(events_aligned)
        all_ldp_weights.append(ldp_weight)
        
        if verbose:
            print(f"  ✅ {ticker}: {len(common_index)} samples")
    
    # Combine all data
    if verbose:
        print(f"\n📊 Combining data from {len(all_features)} tickers...")
    
    X = pd.concat(all_features)
    y = pd.concat(all_labels)
    returns_combined = pd.concat(all_returns)
    dates = pd.DatetimeIndex(all_dates)
    tickers_array = np.array(all_tickers)
    events_combined = pd.concat(all_events)
    ldp_weights_combined = pd.concat(all_ldp_weights)
    
    # Capture feature names before converting to numpy
    feature_names = list(X.columns)
    
    # Sort by date
    sort_idx = dates.argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)
    returns_combined = returns_combined.iloc[sort_idx].reset_index(drop=True)
    dates = dates[sort_idx]
    tickers_array = tickers_array[sort_idx]
    events_combined = events_combined.iloc[sort_idx]
    ldp_weights_combined = ldp_weights_combined.iloc[sort_idx].reset_index(drop=True)
    
    if verbose:
        print(f"✅ Combined: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Date range: {dates.min().date()} to {dates.max().date()}")
        print(f"   Feature names: {feature_names[:5]}... ({len(feature_names)} total)")
    
    # Create sequences
    if verbose:
        print(f"\n🔄 Creating sequences (seq_len={seq_len})...")
    
    X_sequences = []
    y_sequences = []
    dates_sequences = []
    ldp_weights_sequences = []
    returns_sequences = []
    
    for i in range(len(X) - seq_len):
        seq_end_idx = i + seq_len - 1
        X_sequences.append(X.iloc[i:i+seq_len].values)
        y_sequences.append(y.iloc[seq_end_idx])
        dates_sequences.append(dates[seq_end_idx])
        ldp_weights_sequences.append(ldp_weights_combined.iloc[seq_end_idx])
        returns_sequences.append(returns_combined.iloc[seq_end_idx])
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    dates_seq = pd.DatetimeIndex(dates_sequences)
    ldp_weights_seq = np.array(ldp_weights_sequences, dtype=float)
    returns_seq = np.array(returns_sequences, dtype=float)
    
    # Filter NaN
    has_feature_nan = np.isnan(X_seq).any(axis=(1, 2))
    valid_mask = ~has_feature_nan
    
    X_seq = X_seq[valid_mask]
    y_seq = y_seq[valid_mask]
    dates_seq = dates_seq[valid_mask]
    ldp_weights_seq = ldp_weights_seq[valid_mask]
    returns_seq = returns_seq[valid_mask]
    
    if verbose:
        print(f"✅ Generated {len(X_seq)} sequences")
        print(f"   Filtered {(~valid_mask).sum()} NaN sequences")
        print(f"   Shape: {X_seq.shape}")
        print(f"   LdP weights: range=[{ldp_weights_seq.min():.3f}, {ldp_weights_seq.max():.3f}]")
        print(f"   Returns: range=[{returns_seq.min():.4f}, {returns_seq.max():.4f}]")
    
    # Save to cache
    if use_cache:
        if verbose:
            print(f"\n💾 Saving to cache: {cache_file}")
        
        try:
            cache_data = {
                'X_seq': [str(x.tolist()) for x in X_seq],
                'y': y_seq,
                'date': dates_seq,
                'ldp_weight': ldp_weights_seq,
                'return': returns_seq
            }
            df_cache = pd.DataFrame(cache_data)
            df_cache.to_parquet(cache_file, compression='snappy')
            
            file_size_mb = os.path.getsize(cache_file) / 1024**2
            if verbose:
                print(f"✅ Cache saved ({file_size_mb:.1f} MB)")
        except Exception as e:
            if verbose:
                print(f"⚠️  Cache save failed: {e}")
    
    return X_seq, y_seq, dates_seq, ldp_weights_seq, returns_seq, feature_names


if __name__ == "__main__":
    """Test the module"""
    
    print("="*80)
    print("TESTING fin_load_and_sequence.py")
    print("="*80)
    
    # Test with 2 tickers
    X_seq, y_seq, dates_seq, ldp_weights_seq, returns_seq = load_or_generate_sequences(
        tickers=['AAPL', 'NVDA'],
        config_preset='wavenet_optimized',
        barrier_params={
            'lookback': 60,
            'pt_sl': [2, 1],
            'min_ret': 0.005,
            'num_days': 7,
            'num_threads': 1
        },
        seq_len=20,
        data_path='data_raw',
        use_cache=True,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")
    print(f"dates_seq range: {dates_seq.min().date()} to {dates_seq.max().date()}")
    print(f"ldp_weights_seq: min={ldp_weights_seq.min():.3f}, max={ldp_weights_seq.max():.3f}")
    print(f"returns_seq: min={returns_seq.min():.4f}, max={returns_seq.max():.4f}, mean={returns_seq.mean():.4f}")
    print(f"\nLabel distribution: {np.unique(y_seq, return_counts=True)}")
    
    print("\n✅ Module test complete!")
    print("\nRun again to test cache loading (should be instant)")
