# Test Scripts Analysis - Feature Validation Methods

## Executive Summary

**9 test scripts are VULNERABLE** to the NaN bug and need fixing.
**2 test scripts are PROTECTED** (use `load_or_generate_sequences()`).

---

## Vulnerability Status

### ❌ VULNERABLE (9 files) - Need Fix

These scripts load data directly and create features without dropping NaN from OHLCV columns:

1. **test_all_features.py** - Comprehensive validation of ALL 109 features
   - Validation: TA-Lib comparison, numeric comparison, range validation, NaN checks
   - **Priority: CRITICAL** - Most comprehensive test

2. **test_feature_accuracy.py** - Accuracy validation against reference implementations
   - Validation: TA-Lib comparison, numeric comparison, range validation, NaN checks
   - **Priority: HIGH** - Validates correctness

3. **test_feature_accuracy_simple.py** - Simplified accuracy tests
   - Validation: TA-Lib comparison, numeric comparison, range validation, NaN checks
   - **Priority: HIGH** - Core accuracy validation

4. **test_full_features.py** - Tests full feature engineering pipeline
   - Validation: Range validation, NaN checks
   - **Priority: HIGH** - End-to-end test

5. **test_features.py** - Individual feature method tests
   - Validation: NaN checks
   - **Priority: MEDIUM** - Unit tests for methods

6. **test_feature_ranges.py** - Range validation for features
   - Validation: Range validation, NaN checks
   - **Priority: MEDIUM** - Detects scaling issues

7. **test_dataset_validation.py** - Dataset quality validation
   - Validation: Range validation, NaN checks
   - **Priority: MEDIUM** - Data quality checks

8. **test_normalization_effect.py** - Tests normalization impact
   - Validation: Range validation, NaN checks
   - **Priority: LOW** - Normalization specific

9. **test_pbo_quick.py** - Quick PBO analysis
   - Validation: Range validation, NaN checks
   - **Priority: LOW** - Performance testing

### ✅ PROTECTED (2 files) - No Fix Needed

These scripts use `load_or_generate_sequences()` which already has the fix:

1. **test_market_selection.py** - Market feature selection
2. **test_step2_market_context.py** - Market context validation

### ℹ️ N/A (5 files) - No Data Loading

These don't load data or don't create features:
- test_fixes.py
- test_minmax_per_ticker.py
- test_pbo.py
- test_validation_fix.py
- test_wavenet_config.py

---

## Validation Methods Inventory

### 1. **TA-Lib Comparison** (3 files)
**Purpose**: Validate our features match reference TA-Lib implementation

**Files**: 
- test_all_features.py (COMPREHENSIVE)
- test_feature_accuracy.py
- test_feature_accuracy_simple.py

**Features Tested**:
- RSI (7, 14, 21 periods)
- MACD (macd, signal, histogram)
- Stochastic (K, D)
- ADX / +DI / -DI
- Bollinger Bands (upper, middle, lower)
- OBV (On-Balance Volume)
- CCI, SAR, Williams %R, ATR

**Method**: `np.allclose()` with tolerance (typically 1e-3 for TA-Lib)

### 2. **Numeric Comparison** (3 files)
**Purpose**: Validate manual calculations match automated feature generation

**Files**:
- test_all_features.py
- test_feature_accuracy.py
- test_feature_accuracy_simple.py

**Features Tested**:
- Log returns (1d, 2d, 5d, etc.)
- Volatility (close-to-close, Parkinson, Yang-Zhang)
- Volume normalization
- CMF (Chaikin Money Flow)
- Distance from moving averages
- BB position and width

**Method**: Manual calculation vs feature engineer output

### 3. **Range Validation** (12 files)
**Purpose**: Ensure features are within expected/reasonable bounds

**Common Ranges Tested**:
- RSI: [0, 100]
- Williams %R: [-100, 0]
- CMF: [-1, 1]
- BB Position: [0, 1] (can exceed slightly)
- Stochastic: [0, 100]
- ADX: [0, 100]
- Returns: typically [-0.2, 0.2] daily

**Method**: Check `min()`, `max()`, percentiles

### 4. **NaN Checks** (12 files)
**Purpose**: Ensure features don't have excessive missing values

**Thresholds**:
- Typical: < 50% NaN acceptable (warm-up period)
- Some features: 0% NaN required (e.g., normalized features)

**Method**: `isna().sum()`, `isna().any()`

---

## Recommended Fix for Vulnerable Scripts

### Standard Fix Pattern

Add immediately after `read_parquet()`:

```python
df = pd.read_parquet('data_raw/AAPL.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# FIX: Drop NaN values in OHLCV columns (TA-Lib cannot handle NaN)
df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
```

### Why This Fix?

1. **TA-Lib functions fail on NaN input** - Return all NaN
2. **NaN gets converted to 0** - By feature cleaning fillna()
3. **Downstream checks fail** - They expect NaN but see 0
4. **Silent failures** - Features look valid but are wrong

---

## Comprehensive Verification Plan

### Phase 1: Fix Critical Tests (Priority)

1. **test_all_features.py** - Most comprehensive, tests all 109 features
2. **test_feature_accuracy.py** - Validates correctness vs references
3. **test_full_features.py** - End-to-end pipeline

### Phase 2: Run Validation Suite

```bash
# Run comprehensive test
.venv/bin/python test_all_features.py

# Run accuracy tests
.venv/bin/python test_feature_accuracy.py
.venv/bin/python test_feature_accuracy_simple.py

# Run full pipeline test
.venv/bin/python test_full_features.py
```

### Phase 3: Verify All Features

Expected results after fix:
- ✅ All MACD features (macd, macd_signal, macd_hist) non-zero
- ✅ All Bollinger Bands features (bb_upper, bb_lower, bb_position) non-zero
- ✅ All features within expected ranges
- ✅ TA-Lib comparisons pass (< 1% difference)
- ✅ NaN percentage acceptable (< 50% for warm-up features)

---

## Current Status

- ✅ `fin_load_and_sequence.py` - FIXED (line 411)
- ✅ `validate_wavenet_v2_features.py` - FIXED (line 412)
- ❌ 9 test scripts - NEED FIX
- ✅ Notebook (`equity_feature_importance_analysis.ipynb`) - PROTECTED

**Next Action**: Fix the 9 vulnerable test scripts systematically.
