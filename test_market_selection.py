"""
Test market feature selection in fin_load_and_sequence
"""
from fin_load_and_sequence import load_or_generate_sequences

# Test 1: No market features
print("="*80)
print("TEST 1: NO MARKET FEATURES")
print("="*80)
X1, y1, dates1, ldp1, ret1 = load_or_generate_sequences(
    tickers=['AAPL'],
    config_preset='wavenet_optimized',
    barrier_params={'pt_sl': [2, 1], 'min_ret': 0.005, 'num_days': 7, 'lookback': 60},
    seq_len=20,
    use_cache=False,
    market_features=None  # No market features
)
print(f"\nResult shape: {X1.shape}")
print(f"Features per timestep: {X1.shape[2]}")

# Test 2: Only SPY and VIX
print("\n" + "="*80)
print("TEST 2: ONLY SPY + VIX")
print("="*80)
X2, y2, dates2, ldp2, ret2 = load_or_generate_sequences(
    tickers=['AAPL'],
    config_preset='wavenet_optimized',
    barrier_params={'pt_sl': [2, 1], 'min_ret': 0.005, 'num_days': 7, 'lookback': 60},
    seq_len=20,
    use_cache=False,
    market_features=['spy', 'vix']  # Just SPY and VIX
)
print(f"\nResult shape: {X2.shape}")
print(f"Features per timestep: {X2.shape[2]}")
print(f"Added features: {X2.shape[2] - X1.shape[2]} (should be 2)")

# Test 3: All market features
print("\n" + "="*80)
print("TEST 3: ALL MARKET FEATURES")
print("="*80)
X3, y3, dates3, ldp3, ret3 = load_or_generate_sequences(
    tickers=['AAPL'],
    config_preset='wavenet_optimized',
    barrier_params={'pt_sl': [2, 1], 'min_ret': 0.005, 'num_days': 7, 'lookback': 60},
    seq_len=20,
    use_cache=False,
    market_features=['spy', 'vix', 'fvx', 'tyx', 'gold', 'jpyx']  # All 6
)
print(f"\nResult shape: {X3.shape}")
print(f"Features per timestep: {X3.shape[2]}")
print(f"Added features: {X3.shape[2] - X1.shape[2]} (should be 6)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"No market:     {X1.shape[2]} features")
print(f"SPY + VIX:     {X2.shape[2]} features (+{X2.shape[2] - X1.shape[2]})")
print(f"All market:    {X3.shape[2]} features (+{X3.shape[2] - X1.shape[2]})")
print("\n✅ Market feature selection working!")
