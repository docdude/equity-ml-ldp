"""
Debug LdP weight alignment issue
"""
import pandas as pd
import numpy as np

# Load the events to check uniqueness weights
events = pd.read_csv('run_financial_wavenet_ldp_v1/triple_barrier_events.csv', index_col=0, parse_dates=True)

print("="*80)
print("EVENTS DATAFRAME CHECK")
print("="*80)
print(f"Total events: {len(events)}")
print(f"Columns: {events.columns.tolist()}")
print(f"\nFirst few rows:")
print(events.head())

# Check if we have the uniqueness weight column
if 'tW' in events.columns:
    print(f"\n✅ Found 'tW' column (uniqueness weights)")
    print(f"   Range: [{events['tW'].min():.3f}, {events['tW'].max():.3f}]")
    print(f"   Mean: {events['tW'].mean():.3f}, Std: {events['tW'].std():.3f}")
    
    # Check distribution by label (need to map back to labels)
    # We don't have labels in events, they're in a separate structure
else:
    print(f"\n❌ No 'tW' column found!")
    print(f"   This means uniqueness weights weren't saved/computed correctly")

print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("""
The problem is likely one of these:

1. **Averaging over sequences** (MOST LIKELY):
   - You calculate uniqueness per timestamp
   - But then average over 20-step sequences
   - This washes out all the variation!
   
2. **Index misalignment**:
   - events_combined has raw sample indices
   - X_seq/y_seq have sequence indices (shifted by seq_len)
   - Mapping between them loses granularity

3. **Return attribution canceling uniqueness**:
   - If returns are uniform across overlaps
   - return_weights might be uniform too
   
FIX:
- Calculate weights AFTER creating sequences
- OR: Use the sequence end date to look up the weight
- OR: Average uniqueness over the sequence window (not just last point)
""")
