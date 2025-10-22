"""
Investigate why vpin and cs_spread show blank in correlation plot
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("INVESTIGATING VPIN AND CS_SPREAD")
print("="*80)

# Load the feature importance comparison to see which features are in top 20
comparison_file = Path('artifacts/feature_importance/feature_importance_comparison.csv')
if comparison_file.exists():
    comparison = pd.read_csv(comparison_file, index_col=0)
    comparison_sorted = comparison.sort_values('MDA_mean', ascending=False)
    top_features = comparison_sorted.head(20).index.tolist()
    
    print(f"\nTop 20 features by MDA:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # Check if vpin and cs_spread are in the list
    print(f"\nvpin in top 20: {'vpin' in top_features}")
    print(f"cs_spread in top 20: {'cs_spread' in top_features}")
    
    # Check their MDA scores
    if 'vpin' in comparison.index:
        print(f"\nvpin stats:")
        print(f"  MDA mean: {comparison.loc['vpin', 'MDA_mean']:.6f}")
        print(f"  MDA std: {comparison.loc['vpin', 'MDA_std']:.6f}")
        print(f"  MDA rank: {comparison.loc['vpin', 'MDA_rank']}")
    
    if 'cs_spread' in comparison.index:
        print(f"\ncs_spread stats:")
        print(f"  MDA mean: {comparison.loc['cs_spread', 'MDA_mean']:.6f}")
        print(f"  MDA std: {comparison.loc['cs_spread', 'MDA_std']:.6f}")
        print(f"  MDA rank: {comparison.loc['cs_spread', 'MDA_rank']}")

# Now check the actual feature data
print("\n" + "="*80)
print("CHECKING ACTUAL FEATURE VALUES")
print("="*80)

# Load some test data to check feature generation
from fin_load_and_sequence import load_or_generate_sequences
from feature_config import FeatureConfig

tickers = ['AAPL', 'NVDA']
seq_len = 20
config_preset = 'comprehensive'

# Load sequences
all_sequences = []
all_labels = []
all_tickers_list = []

for ticker in tickers:
    print(f"\nLoading {ticker}...")
    X, y, meta = load_or_generate_sequences(
        ticker=ticker,
        seq_len=seq_len,
        config_preset=config_preset,
        market_features=['spy', 'vix'],  # Just 2 for quick test
        use_cache=True,
        cache_dir='cache'
    )
    
    if X is not None and len(X) > 0:
        all_sequences.append(X)
        all_labels.append(y)
        all_tickers_list.extend([ticker] * len(X))
        print(f"  Loaded {len(X)} sequences")

# Combine
if len(all_sequences) > 0:
    X_combined = np.concatenate(all_sequences, axis=0)
    y_combined = np.concatenate(all_labels, axis=0)
    
    print(f"\nCombined shape: {X_combined.shape}")
    
    # Get feature names
    config = FeatureConfig.get_preset(config_preset)
    feature_list = meta.get('feature_names', [])
    
    print(f"\nTotal features: {len(feature_list)}")
    print(f"Features in X: {X_combined.shape[2]}")
    
    # Check if vpin and cs_spread are in the feature list
    vpin_idx = feature_list.index('vpin') if 'vpin' in feature_list else None
    cs_spread_idx = feature_list.index('cs_spread') if 'cs_spread' in feature_list else None
    
    print(f"\nvpin in features: {vpin_idx is not None} (index: {vpin_idx})")
    print(f"cs_spread in features: {cs_spread_idx is not None} (index: {cs_spread_idx})")
    
    # Flatten sequences to 2D for correlation analysis
    n_samples, n_timesteps, n_features = X_combined.shape
    X_flat = X_combined.reshape(-1, n_features)
    X_df = pd.DataFrame(X_flat, columns=feature_list)
    
    # Check vpin
    if vpin_idx is not None:
        vpin_data = X_df['vpin']
        print(f"\nvpin statistics:")
        print(f"  Count: {len(vpin_data)}")
        print(f"  Non-NaN: {vpin_data.notna().sum()}")
        print(f"  NaN: {vpin_data.isna().sum()}")
        print(f"  Unique values: {vpin_data.nunique()}")
        print(f"  Mean: {vpin_data.mean():.6f}")
        print(f"  Std: {vpin_data.std():.6f}")
        print(f"  Min: {vpin_data.min():.6f}")
        print(f"  Max: {vpin_data.max():.6f}")
        print(f"  Sample values: {vpin_data.head(20).values}")
    
    # Check cs_spread
    if cs_spread_idx is not None:
        cs_spread_data = X_df['cs_spread']
        print(f"\ncs_spread statistics:")
        print(f"  Count: {len(cs_spread_data)}")
        print(f"  Non-NaN: {cs_spread_data.notna().sum()}")
        print(f"  NaN: {cs_spread_data.isna().sum()}")
        print(f"  Unique values: {cs_spread_data.nunique()}")
        print(f"  Mean: {cs_spread_data.mean():.6f}")
        print(f"  Std: {cs_spread_data.std():.6f}")
        print(f"  Min: {cs_spread_data.min():.6f}")
        print(f"  Max: {cs_spread_data.max():.6f}")
        print(f"  Sample values: {cs_spread_data.head(20).values}")
    
    # Test correlation
    print("\n" + "="*80)
    print("CORRELATION TEST")
    print("="*80)
    
    if vpin_idx is not None and cs_spread_idx is not None:
        # Compute correlation matrix
        corr_matrix = X_df.corr()
        
        print(f"\nvpin correlation with itself: {corr_matrix.loc['vpin', 'vpin']}")
        print(f"cs_spread correlation with itself: {corr_matrix.loc['cs_spread', 'cs_spread']}")
        
        print(f"\nvpin correlation with cs_spread: {corr_matrix.loc['vpin', 'cs_spread']}")
        
        # Check if they correlate with anything
        print(f"\nvpin - features with non-NaN correlation:")
        vpin_corr = corr_matrix['vpin'].dropna()
        print(f"  Count: {len(vpin_corr)}")
        if len(vpin_corr) > 0:
            print(f"  Top 5:")
            print(vpin_corr.sort_values(ascending=False).head(5))
        
        print(f"\ncs_spread - features with non-NaN correlation:")
        cs_corr = corr_matrix['cs_spread'].dropna()
        print(f"  Count: {len(cs_corr)}")
        if len(cs_corr) > 0:
            print(f"  Top 5:")
            print(cs_corr.sort_values(ascending=False).head(5))

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("""
If vpin and cs_spread show as blank in the correlation heatmap:

1. All NaN values → correlation is undefined (blank/white in heatmap)
2. All same value (zero variance) → correlation is undefined
3. Not in the dataset → KeyError would occur

Check the output above to see which case applies.
""")
