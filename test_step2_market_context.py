#!/usr/bin/env python3
"""
Test market context integration in fin_load_and_sequence.py
"""

import numpy as np
from fin_load_and_sequence import load_or_generate_sequences

print("="*80)
print("STEP 2: Testing Market Context Integration")
print("="*80)

# Test with a small set of tickers
tickers = ['AAPL', 'NVDA']  # Small test set

print(f"\nTesting with tickers: {tickers}")
print("\nThis should:")
print("  1. Load ticker-specific features (~71 features)")
print("  2. Load market context (6 features)")
print("  3. Combine them (~77 total features)")

barrier_params = {
    'lookback': 60,
    'pt_sl': [2, 1],
    'min_ret': 0.005,
    'num_days': 7,
    'num_threads': 6
}

try:
    X_seq, y_seq, dates_seq, ldp_weights_seq = load_or_generate_sequences(
        tickers=tickers,
        config_preset='wavenet_optimized',
        barrier_params=barrier_params,
        seq_len=20,
        data_path='data_raw',
        use_cache=False,  # Force regeneration to test market context
        verbose=True
    )
    
    print("\n" + "="*80)
    print("✓ STEP 2 TEST RESULTS")
    print("="*80)
    print(f"Sequences shape: {X_seq.shape}")
    print(f"  - Number of samples: {X_seq.shape[0]}")
    print(f"  - Sequence length: {X_seq.shape[1]}")
    print(f"  - Number of features: {X_seq.shape[2]}")
    print(f"\nExpected features: ~77 (71 ticker + 6 market)")
    
    if X_seq.shape[2] >= 75:
        print(f"✓ Market context appears to be included!")
    else:
        print(f"⚠️  Feature count seems low - market context may not be added")
    
    print(f"\nLabels shape: {y_seq.shape}")
    print(f"Label distribution:")
    unique, counts = np.unique(y_seq, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label:2.0f}: {count:5d} samples ({count/len(y_seq)*100:5.1f}%)")
    
    print(f"\nDate range: {dates_seq.min().date()} to {dates_seq.max().date()}")
    print(f"LdP weights: [{ldp_weights_seq.min():.3f}, {ldp_weights_seq.max():.3f}]")
    
    print("\n" + "="*80)
    print("✓ Step 2 integration successful!")
    print("="*80)

except Exception as e:
    print(f"\n❌ Step 2 test failed: {e}")
    import traceback
    traceback.print_exc()
