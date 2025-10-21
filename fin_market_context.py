"""
Market Context Features Module

Loads and transforms external market data (SPY, VIX, Treasury yields, Gold, Forex)
to add as context features for all tickers, following MLFinance methodology.

Transformations verified against MLFinance/Data/AAPL_feature_matrix.parquet:
  - ^GSPC (S&P 500): Daily returns (pct_change)
  - ^VIX: Normalized level (Close / 100)
  - ^FVX (5yr Treasury): Daily returns (pct_change)
  - ^TYX (30yr Treasury): Daily returns (pct_change)
  - GC=F (Gold): Daily returns (pct_change)
  - JPY=X (USD/JPY): Daily returns (pct_change)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_market_data(data_path: str = 'data_raw') -> dict:
    """
    Load all external market data files.
    
    Args:
        data_path: Directory containing market data parquet files
        
    Returns:
        Dictionary with keys: 'spy', 'vix', 'fvx', 'tyx', 'gold', 'jpyx'
        Each value is a DataFrame with date index and OHLCV columns
    """
    data_path = Path(data_path)
    
    def load_and_prepare(filepath):
        """Load parquet file and prepare with datetime index."""
        df = pd.read_parquet(filepath)
        
        # Set date as index if needed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Standardize column names (capitalize)
        df.columns = df.columns.str.capitalize()
        
        return df
    
    market_data = {}
    
    # Define file mappings
    files = {
        'spy': '^GSPC.parquet',
        'vix': '^VIX.parquet', 
        'fvx': '^FVX.parquet',
        'tyx': '^TYX.parquet',
        'gold': 'GC=F.parquet',
        'jpyx': 'JPY=X.parquet'
    }
    
    print("Loading market context data...")
    for key, filename in files.items():
        filepath = data_path / filename
        if filepath.exists():
            market_data[key] = load_and_prepare(filepath)
            print(f"  ✓ Loaded {key:4s}: {len(market_data[key])} rows, "
                  f"date range {market_data[key].index.min().date()} to {market_data[key].index.max().date()}")
        else:
            print(f"  ✗ Warning: {filename} not found at {filepath}")
            market_data[key] = None
    
    return market_data


def create_market_context_features(
    market_data: dict,
    selected_features: list = None
) -> pd.DataFrame:
    """
    Transform raw market data into features matching MLFinance methodology.
    
    Args:
        market_data: Dictionary from load_market_data()
        selected_features: Optional list of features to include. 
                          Valid options: ['spy', 'vix', 'fvx', 'tyx', 'gold', 'jpyx']
                          If None, includes all available features
        
    Returns:
        DataFrame with date index and selected market context feature columns:
        - ^GSPC: S&P 500 daily returns (if 'spy' selected)
        - ^VIX: VIX level normalized (0-1 scale) (if 'vix' selected)
        - ^FVX: 5yr Treasury yield daily changes (if 'fvx' selected)
        - ^TYX: 30yr Treasury yield daily changes (if 'tyx' selected)
        - GC=F: Gold futures daily returns (if 'gold' selected)
        - JPY=X: USD/JPY forex daily changes (if 'jpyx' selected)
    """
    # If no selection, use all available
    if selected_features is None:
        selected_features = ['spy', 'vix', 'fvx', 'tyx', 'gold', 'jpyx']
    
    print(f"  Creating market features for: {selected_features}")
    
    features = {}
    
    # ^GSPC (S&P 500) - Daily returns
    if 'spy' in selected_features and market_data['spy'] is not None:
        features['^GSPC'] = market_data['spy']['Close'].pct_change(1, fill_method=None)
        print("  ✓ Created ^GSPC feature (S&P 500 returns)")
    
    # ^VIX - Normalized level (VIX / 100)
    if 'vix' in selected_features and market_data['vix'] is not None:
        features['^VIX'] = market_data['vix']['Close'] / 100
        print("  ✓ Created ^VIX feature (volatility level / 100)")
    
    # ^FVX (5yr Treasury) - Daily returns
    if 'fvx' in selected_features and market_data['fvx'] is not None:
        features['^FVX'] = market_data['fvx']['Close'].pct_change(1, fill_method=None)
        print("  ✓ Created ^FVX feature (5yr yield changes)")
    
    # ^TYX (30yr Treasury) - Daily returns
    if 'tyx' in selected_features and market_data['tyx'] is not None:
        features['^TYX'] = market_data['tyx']['Close'].pct_change(1, fill_method=None)
        print("  ✓ Created ^TYX feature (30yr yield changes)")
    
    # GC=F (Gold) - Daily returns
    if 'gold' in selected_features and market_data['gold'] is not None:
        features['GC=F'] = market_data['gold']['Close'].pct_change(1, fill_method=None)
        print("  ✓ Created GC=F feature (gold returns)")
    
    # JPY=X (USD/JPY) - Daily returns
    if 'jpyx' in selected_features and market_data['jpyx'] is not None:
        features['JPY=X'] = market_data['jpyx']['Close'].pct_change(1, fill_method=None)
        print("  ✓ Created JPY=X feature (forex changes)")
    
    # Combine into single DataFrame
    market_features = pd.DataFrame(features)
    
    print(f"\n  Market context features shape: {market_features.shape}")
    print(f"  Date range: {market_features.index.min().date()} to {market_features.index.max().date()}")
    print(f"  Features created: {list(market_features.columns)}")
    
    return market_features


def add_market_context_to_ticker(
    ticker_df: pd.DataFrame, 
    market_features: pd.DataFrame,
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Add market context features to a ticker's feature DataFrame.
    
    Args:
        ticker_df: Ticker-specific features with date index
        market_features: Market context features from create_market_context_features()
        fill_method: How to handle missing market data ('ffill', 'bfill', or None)
        
    Returns:
        DataFrame with original ticker features + 6 market context columns
    """
    # Align market features to ticker dates
    aligned_market = market_features.reindex(ticker_df.index)
    
    # Forward fill market data (markets closed on different days)
    if fill_method == 'ffill':
        aligned_market = aligned_market.ffill()
    elif fill_method == 'bfill':
        aligned_market = aligned_market.bfill()
    
    # Combine ticker features with market context
    combined = pd.concat([ticker_df, aligned_market], axis=1)
    
    # Report any remaining NaN values
    nan_counts = aligned_market.isna().sum()
    if nan_counts.sum() > 0:
        print(f"    Warning: NaN values in market features after alignment:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"      {col}: {count} NaN values")
    
    return combined


def get_market_context_feature_names() -> list:
    """
    Return the list of market context feature names.
    
    Returns:
        List of 6 feature names matching MLFinance convention
    """
    return ['^GSPC', '^VIX', '^FVX', '^TYX', 'GC=F', 'JPY=X']


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("MARKET CONTEXT FEATURES - Testing")
    print("="*80)
    
    # Step 1: Load market data
    print("\nStep 1: Loading market data files...")
    market_data = load_market_data('data_raw')
    
    # Step 2: Create features
    print("\nStep 2: Creating market context features...")
    market_features = create_market_context_features(market_data)
    
    # Step 3: Display sample
    print("\nStep 3: Sample market features (last 10 days):")
    print(market_features.tail(10))
    
    # Step 4: Statistics
    print("\nStep 4: Feature statistics:")
    print(market_features.describe())
    
    # Step 5: Check correlations
    print("\nStep 5: Feature correlations:")
    print(market_features.corr())
    
    # Step 6: Test adding to ticker data
    print("\nStep 6: Testing integration with ticker data...")
    test_ticker = pd.read_parquet('data_raw/AAPL.parquet')
    if 'date' in test_ticker.columns:
        test_ticker['date'] = pd.to_datetime(test_ticker['date'])
        test_ticker = test_ticker.set_index('date')
    
    # Create simple test features
    test_features = pd.DataFrame({
        'test_feature_1': test_ticker['close'].pct_change(1),
        'test_feature_2': test_ticker['volume']
    }, index=test_ticker.index)
    
    print(f"  Original AAPL features shape: {test_features.shape}")
    
    combined = add_market_context_to_ticker(test_features, market_features)
    print(f"  After adding market context: {combined.shape}")
    print(f"  Added {combined.shape[1] - test_features.shape[1]} market features")
    
    print("\n  Sample combined features (last 5 rows):")
    print(combined.tail())
    
    print("\n" + "="*80)
    print("✓ Market context module ready for integration!")
    print("="*80)
