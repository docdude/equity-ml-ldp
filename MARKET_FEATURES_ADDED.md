# Market Features Added to Training Pipeline ✅

## What Changed

Added `MARKET_FEATURES` configuration to `fin_training_ldp.py`:

```python
# Market features configuration (based on wavenet_optimized_v2 analysis)
MARKET_FEATURES = ['fvx', 'tyx']  # Treasury yields (rank 8/14 orthogonal)
```

## Feature Configuration

### Current Setup (Recommended)
```python
MARKET_FEATURES = ['fvx', 'tyx']  # 5yr + 30yr Treasury yields
```

**Total features:** 38 (stock) + 2 (market) = **40 features**

### Rationale (from Feature Importance Analysis)

**Included:**
- **^FVX (5yr Treasury)**: Rank 63 → **14 orthogonal** (hidden gem!)
- **^TYX (30yr Treasury)**: Rank **8 orthogonal** (top 10!)

These capture unique **macro risk/liquidity signals** not present in stock features.

### Alternative Configurations

**Minimal (just 5yr):**
```python
MARKET_FEATURES = ['fvx']  # Just 5-year Treasury
# Total: 38 + 1 = 39 features
```

**With Gold:**
```python
MARKET_FEATURES = ['fvx', 'tyx', 'gold']  # Add Gold futures
# Total: 38 + 3 = 41 features
```

**All Available:**
```python
MARKET_FEATURES = ['spy', 'vix', 'fvx', 'tyx', 'gold', 'jpyx']
# Total: 38 + 6 = 44 features
# Not recommended: spy, vix, jpyx are redundant
```

## Market Feature Rankings

| Feature | Internal Key | Original MDA Rank | Orthogonal MDA Rank | Status |
|---------|-------------|-------------------|---------------------|--------|
| ^FVX (5yr Treasury) | `'fvx'` | 63 | **14** | ✅ **Include** |
| ^TYX (30yr Treasury) | `'tyx'` | 83 | **8** | ✅ **Include** |
| GC=F (Gold) | `'gold'` | 13 | 21 | ⚪ Optional |
| ^GSPC (S&P 500) | `'spy'` | 26 | 102 | ❌ Skip (redundant) |
| ^VIX (Volatility) | `'vix'` | 103 | 110 | ❌ Skip (redundant) |
| JPY=X (Yen) | `'jpyx'` | 84 | 28 | ❌ Skip (marginal) |

## Where Features Are Added

1. **Configuration** (`fin_training_ldp.py`, line ~329):
   ```python
   MARKET_FEATURES = ['fvx', 'tyx']
   ```

2. **Loading** (`fin_training_ldp.py`, line ~355):
   ```python
   X_seq, y_seq, dates_seq, ldp_weights_seq, returns_seq, feature_names = \
       load_or_generate_sequences(
           tickers=tickers,
           config_preset=CONFIG_PRESET,
           barrier_params=BARRIER_PARAMS,
           market_features=MARKET_FEATURES,  # ← Added here
           ...
       )
   ```

3. **Processing** (`fin_load_and_sequence.py`):
   - Loads market data from `data_raw/^FVX.parquet`, `^TYX.parquet`
   - Creates features using `create_market_context_features()`
   - Adds to each ticker via `add_market_context_to_ticker()`

4. **Saved** (`metadata.pkl`):
   ```python
   metadata = {
       'market_features': MARKET_FEATURES,  # ← Saved for inference
       ...
   }
   ```

## Expected Output

When you run training, you should see:

```
📊 Market Features: ['fvx', 'tyx']
   • ^FVX (5yr Treasury)
   • ^TYX (30yr Treasury)

📊 Loading market context features: ['fvx', 'tyx']
  ✓ Loaded fvx  : XXXX rows, date range YYYY-MM-DD to YYYY-MM-DD
  ✓ Loaded tyx  : XXXX rows, date range YYYY-MM-DD to YYYY-MM-DD
  Creating market features for: ['fvx', 'tyx']

✅ Sequences ready: (N, 20, 40)
   Features: 40  ← 38 stock + 2 market
```

## Cache Impact

**Important:** Cache filenames include market features:
```
sequences_AAPL_NVDA_..._wavenet_optimized_v2_fvx_tyx_20_pt2.0_sl1.0_mr0.005_h7.npz
                                             ^^^^^^^^ market features in filename
```

If you change `MARKET_FEATURES`, the cache will regenerate automatically (different filename).

## Verify Market Features Are Working

```python
# Load metadata after training
import pickle
with open('run_financial_wavenet_ldp_v1/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Market features used: {metadata['market_features']}")
print(f"Total features: {metadata['dataset_info']['features']}")
print(f"Feature names: {metadata['dataset_info']['feature_names']}")

# Check if ^FVX and ^TYX are in feature names
feature_names = metadata['dataset_info']['feature_names']
market_in_features = [f for f in feature_names if '^' in f or '=' in f]
print(f"Market features found: {market_in_features}")
# Expected: ['^FVX', '^TYX']
```

## Next Steps

1. ✅ Run training with `python fin_training_ldp.py`
2. ✅ Verify feature count is 40 (38 + 2)
3. ✅ Check `metadata.pkl` has `market_features: ['fvx', 'tyx']`
4. ⚪ Optional: Test with just `['fvx']` (39 features) or add `'gold'` (41 features)
5. ⚪ Compare model performance with/without market features
