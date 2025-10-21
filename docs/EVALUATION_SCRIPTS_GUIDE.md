# Evaluation Scripts Guide: Which to Use When

**Date**: October 15, 2025  
**Status**: All scripts refactored with forward returns fix ✅

---

## ⚠️ CRITICAL DISTINCTION

### fin_model_evaluation.py Capabilities:

**Sklearn Models (model=None)**:
- ✅ **COMPREHENSIVE** evaluation
- ✅ Full López de Prado suite: PCV, CPCV, Walk-Forward, Feature Importance, PBO
- ✅ Use this for feature testing!

**Keras Models (model=keras_model)**:
- ⚠️ **LIMITED** evaluation
- ⚠️ **PBO ONLY** (skips PCV, CPCV, Walk-Forward, Feature Importance)
- ⚠️ Same as test_pbo_quick.py (not comprehensive)
- ⚠️ Can't run full suite without days of retraining

### Bottom Line:
- **Comprehensive evaluation** → Sklearn only
- **Keras evaluation** → Limited to PBO (use test_pbo_quick.py for simplicity)

---

## Quick Reference Table

| Script | Purpose | Model Type | Forward Returns Bug | López de Prado Methods | Use When |
|--------|---------|------------|---------------------|----------------------|----------|
| **fin_model_evaluation.py** | **Sklearn**: Comprehensive<br>**Keras**: Limited (PBO only) | Sklearn OR Keras | ✅ FIXED | **Sklearn**: Full suite (PCV, CPCV, Walk-Forward, Feature Importance, PBO)<br>**Keras**: PBO ONLY | Testing features with RandomForest (comprehensive)<br>OR quick Keras PBO check (limited) |
| **test_pbo_quick.py** | Quick PBO test | Keras only | ✅ FIXED | PBO only | Quick Keras model validation (focused, simpler than fin_model_evaluation) |
| **test_pbo.py** | PBO unit tests | Synthetic data | N/A (no real data) | PBO only | Testing PBO implementation |

---

## 1. fin_model_evaluation.py

### Purpose
**Primary**: Feature evaluation using RandomForest (COMPREHENSIVE)  
**Secondary**: Pre-trained Keras model validation (PBO ONLY - LIMITED)

### What It Does

#### For Sklearn Models (RandomForest) - COMPREHENSIVE:
```python
# Run FULL López de Prado suite
model = None  # Will create RandomForestClassifier

main(model=model, model_name="RandomForest")

# Results:
# ✅ Purged Cross-Validation (PCV) - 5 folds
# ✅ Combinatorial Purged CV (CPCV) - 10 combinations  
# ✅ Walk-Forward Analysis - expanding window
# ✅ Feature Importance (MDI, MDA, SFI)
# ✅ PBO Analysis - 16 splits
# ✅ COMPREHENSIVE EVALUATION
```

**Why this is useful for features**:
- RandomForest trains in seconds (fast iteration)
- Get feature importance rankings (MDI/MDA/SFI)
- Test if features have predictive power before spending hours training Keras
- Full López de Prado validation suite

**Example workflow**:
```bash
# 1. Test new feature set with RandomForest
python -c "
from fin_model_evaluation import main
main(model=None, model_name='RandomForest-NewFeatures')
"

# If results are good (PCV AUC > 0.65, PBO < 0.5):
#   ✅ Features are worth using → Train Keras model
# If results are poor:
#   ❌ Save time - don't train Keras, improve features first
```

#### For Keras Models - LIMITED (PBO ONLY):
```python
# Run PBO only (skip retraining methods)
from fin_model import load_model_with_custom_objects
model = load_model_with_custom_objects('best_model.keras')

main(model=model, model_name="CNN-LSTM-WaveNet")

# Results:
# ❌ PCV - Skipped (too expensive to retrain)
# ❌ CPCV - Skipped
# ❌ Walk-Forward - Skipped
# ❌ Feature Importance - Skipped
# ✅ PBO Analysis - 16 splits ← ONLY THIS RUNS
# ⚠️  LIMITED EVALUATION (not comprehensive)
```

### Key Features

1. **Forward Returns Bug: FIXED** ✅
   ```python
   # Calculates per-ticker BEFORE concatenation
   for ticker in tickers:
       forward_ret_5d = (prices['close'].shift(-5) / prices['close']) - 1
       ticker_features['forward_return_5d'] = forward_ret_5d.values
       all_features.append(ticker_features)
   ```

2. **Detects Model Type**:
   ```python
   is_keras = model is not None and hasattr(model, 'layers')
   
   if is_keras:
       # PBO only
   else:
       # Full suite
   ```

3. **Creates 10 Strategies**:
   - Confidence thresholds: 0.33 to 0.85
   - Tests different trading aggressiveness

### When to Use

✅ **Use fin_model_evaluation.py when**:
- **Testing new features with RandomForest** (PRIMARY USE CASE)
  - Need feature importance rankings
  - Want FULL López de Prado validation suite
  - Comprehensive evaluation before training Keras
- **Quick PBO check for Keras model** (SECONDARY USE CASE)
  - Same as test_pbo_quick.py but integrated in pipeline
  - **NOT comprehensive - PBO only!**

❌ **Don't use when**:
- Just want quick PBO check for Keras → use `test_pbo_quick.py` instead (simpler, focused)
- Expecting comprehensive Keras evaluation → **NOT POSSIBLE** (would need days of retraining)
- Testing PBO implementation → use `test_pbo.py` instead

---

## 2. test_pbo_quick.py

### Purpose
**Quick validation of Keras model predictions using PBO**

### What It Does

```python
# Load trained Keras model
model = load_model('best_model.keras')

# Generate predictions (once)
predictions = model.predict(X_val)

# Create 10 strategies (different confidence thresholds)
strategy_returns = create_strategies(predictions, forward_returns)

# Run PBO analysis
pbo_result = pbo_func(strategy_returns, n_splits=16)

# Results:
# ✅ PBO: 0.408
# ✅ Prob OOS Loss: 0.412
# ✅ Mean Sharpe: 0.363
# ✅ Strategy returns saved to artifacts/
```

### Key Features

1. **Forward Returns Bug: FIXED** ✅
   ```python
   # Same fix as fin_model_evaluation.py
   for ticker in tickers:
       forward_ret_5d = (prices['close'].shift(-5) / prices['close']) - 1
       ticker_features['forward_return_5d'] = forward_ret_5d.values
       all_features.append(ticker_features)
   ```

2. **Focused on Keras**:
   - Only loads pre-trained models
   - No RandomForest fallback
   - Optimized for speed

3. **Saves Results**:
   ```
   artifacts/pbo_strategy_returns.csv
   artifacts/pbo_results.json
   ```

4. **10 Strategies**:
   - Same thresholds as fin_model_evaluation.py
   - Confidence: 0.33, 0.39, 0.45, 0.50, 0.56, 0.62, 0.68, 0.73, 0.79, 0.85

### When to Use

✅ **Use test_pbo_quick.py when**:
- Trained a Keras model, want quick PBO check
- Don't need feature importance
- Don't need PCV/CPCV/Walk-Forward
- Want to save strategy returns for analysis

❌ **Don't use when**:
- Testing features → use `fin_model_evaluation.py` with RandomForest
- Need comprehensive validation → use `fin_model_evaluation.py`

### Example Usage

```bash
# After training Keras model:
python test_pbo_quick.py

# Check results:
cat artifacts/pbo_results.json

# Analyze strategy returns:
jupyter notebook pbo_analysis_explained.ipynb
```

---

## 3. test_pbo.py

### Purpose
**Unit tests for PBO implementation**

### What It Does

Tests PBO calculation with synthetic data:

```python
# Test 1: Basic PBO with mixed strategies
test_pbo_basic()  # Random Sharpe ratios [-0.5, 2.0]

# Test 2: Overfit strategies (should have high PBO)
test_pbo_overfit()  # Pure random walks → PBO should be > 0.5

# Test 3: Robust strategies (should have low PBO)
test_pbo_robust()  # Consistent positive drift → PBO should be < 0.5

# Test 4: API compatibility
test_pbo_from_list()  # Accepts list input
```

### Key Features

1. **No Real Data**:
   - Uses synthetic returns
   - No forward returns calculation
   - No data loading

2. **Tests PBO Implementation**:
   - Verifies pypbo integration
   - Checks CSCV logic
   - Validates interpretation thresholds

3. **Fast**:
   - Runs in seconds
   - No model training
   - No feature engineering

### When to Use

✅ **Use test_pbo.py when**:
- Debugging PBO implementation
- Testing pypbo library integration
- Verifying CSCV splits work correctly
- Learning how PBO works

❌ **Don't use when**:
- Evaluating real models → use other scripts
- Need production validation → use `test_pbo_quick.py` or `fin_model_evaluation.py`

---

## 4. Complete Workflow Recommendations

### Workflow A: Feature Development

```bash
# Step 1: Add new features to feature_config.py
# Edit feature_config.py to enable new features

# Step 2: Test with RandomForest (fast!)
python -c "
from fin_model_evaluation import main
main(model=None, model_name='RF-Test-v1')
"

# Check results:
# - PCV AUC: Should be > 0.60 (preferably > 0.65)
# - PBO: Should be < 0.5
# - Feature Importance: Which features matter?

# If good → proceed to Step 3
# If bad → revise features, repeat Step 2
```

**Expected output**:
```
🎯 Purged Cross-Validation:
   AUC: 0.6800 ± 0.0200  ← GOOD!
   
📊 Feature Importance (MDI):
   1. yang_zhang_volatility: 0.0850
   2. atr_normalized: 0.0720
   3. macd_signal: 0.0680
   ...

🔍 PBO Analysis:
   PBO: 0.380  ← GOOD (< 0.5)
   
✅ Features look promising! Worth training Keras model.
```

### Workflow B: Keras Model Evaluation

```bash
# Step 1: Train Keras model (takes hours)
python fin_training.py

# Step 2: Quick PBO check
python test_pbo_quick.py

# Check results:
# - PBO: Should be < 0.5
# - Mean Sharpe: Should be > 0.3

# If good → deploy to paper trading
# If bad → investigate features/architecture
```

**Expected output**:
```
✅ Model loaded: 2,072,773 parameters

📊 Strategy returns matrix: (4683, 10)
   Mean Sharpe ratio: 0.3629

🔍 PBO Analysis:
   PBO: 0.408  ← GOOD (< 0.5)
   Prob OOS Loss: 0.412
   
💾 SAVED RESULTS:
   artifacts/pbo_strategy_returns.csv
   artifacts/pbo_results.json

✅ Model shows potential but needs improvement
   - PBO: 0.408 (acceptable overfitting risk)
   - Mean Sharpe: 0.363 (low profitability)
```

### Workflow C: Comprehensive Analysis

```bash
# For research/publication quality analysis

# Step 1: Train Keras model
python fin_training.py

# Step 2: Evaluate Keras model (PBO only - limited!)
python -c "
from fin_model import load_model_with_custom_objects
from fin_model_evaluation import main

model = load_model_with_custom_objects('best_model.keras')
main(model=model, model_name='CNN-LSTM-WaveNet-Final')
"
# ⚠️ NOTE: This only runs PBO for Keras!
# NOT comprehensive - just PBO

# Step 3: For comprehensive feature analysis, evaluate with RandomForest
python -c "
from fin_model_evaluation import main
main(model=None, model_name='RF-Feature-Analysis')
"
# ✅ This runs FULL suite: PCV, CPCV, Walk-Forward, Feature Importance, PBO
# Use this to understand which features matter
```

---

## 5. Decision Tree: Which Script to Use?

```
Do you have a trained Keras model?
│
├─ YES → Do you just need PBO?
│   │
│   ├─ YES → Use test_pbo_quick.py
│   │         (Fast, focused, saves results)
│   │
│   └─ NO → Want comprehensive analysis?
│       │
│       ├─ YES → Use fin_model_evaluation.py
│       │         (Keras: PBO only, but integrated)
│       │
│       └─ NO → Testing PBO implementation?
│           └─ Use test_pbo.py (unit tests)
│
└─ NO → Testing new features?
    │
    └─ YES → Use fin_model_evaluation.py with model=None
              (Creates RandomForest, runs full suite)
```

---

## 6. Forward Returns Bug Status

| Script | Status | Fix Applied |
|--------|--------|-------------|
| **fin_model_evaluation.py** | ✅ FIXED | Per-ticker calculation before concat |
| **test_pbo_quick.py** | ✅ FIXED | Per-ticker calculation before concat |
| **test_pbo.py** | N/A | Uses synthetic data only |

**The Fix**:
```python
# ❌ WRONG (OLD CODE - cross-ticker contamination):
X = pd.concat(all_features)  # Concatenate first
forward_ret_5d = (prices.shift(-5) / prices) - 1  # Calculate after

# ✅ CORRECT (CURRENT CODE - per-ticker):
for ticker in tickers:
    # Calculate WITHIN ticker
    forward_ret_5d = (prices['close'].shift(-5) / prices['close']) - 1
    ticker_features['forward_return_5d'] = forward_ret_5d.values
    all_features.append(ticker_features)
    
X = pd.concat(all_features)  # Concatenate after
```

---

## 7. Summary

### For Feature Testing:
→ **Use fin_model_evaluation.py with model=None**
- Fast iteration with RandomForest
- Full López de Prado suite
- Feature importance rankings
- Decide if features are worth training Keras

### For Keras Model Validation:
→ **Use test_pbo_quick.py**
- Quick PBO check
- Saves results for analysis
- Focused and efficient
- **Same PBO analysis as fin_model_evaluation.py but simpler**

### For PBO Implementation Testing:
→ **Use test_pbo.py**
- Unit tests with synthetic data
- Verifies pypbo integration
- Educational/debugging tool

### For Comprehensive Research (Sklearn ONLY):
→ **Use fin_model_evaluation.py with model=None**
- Creates RandomForestClassifier
- **FULL López de Prado suite** (PCV, CPCV, Walk-Forward, Feature Importance, PBO)
- Use this to evaluate FEATURES before training expensive Keras models
- ⚠️ **Keras models get PBO only - NOT comprehensive!**

---

## 8. Example Commands

```bash
# 1. Test new features (fast):
python -c "from fin_model_evaluation import main; main()"

# 2. Quick Keras validation:
python test_pbo_quick.py

# 3. Comprehensive Keras evaluation:
python -c "
from fin_model import load_model_with_custom_objects
from fin_model_evaluation import main
model = load_model_with_custom_objects('run_financial_wavenet_v1/best_model.keras')
main(model=model, model_name='CNN-LSTM-WaveNet')
"

# 4. Test PBO implementation:
python test_pbo.py
```

---

## 9. Key Takeaway

**YES, fin_model_evaluation.py is COMPREHENSIVE for sklearn feature testing!**

- ✅ When `model=None`: Creates RandomForest → **FULL López de Prado suite**
  - PCV, CPCV, Walk-Forward, Feature Importance, PBO
  - **This is COMPREHENSIVE evaluation**
- ⚠️ When `model=keras_model`: Uses Keras → **PBO ONLY (LIMITED, not comprehensive)**
  - Skips PCV, CPCV, Walk-Forward, Feature Importance
  - **Only runs PBO** (same as test_pbo_quick.py)
- ✅ Forward returns bug FIXED in both paths
- ✅ **Use it for sklearn feature evaluation - it's perfect for that!**

**YES, test_pbo_quick.py is all you need for Keras validation!**

- ✅ PBO only (skip expensive PCV/CPCV)
- ✅ Forward returns bug FIXED
- ✅ Saves results for analysis
- ✅ Fast and focused
- ✅ Same PBO analysis as fin_model_evaluation.py for Keras

**Use strategically**:
1. **Feature Development (Sklearn)**: `fin_model_evaluation.py` (model=None)
   - **COMPREHENSIVE** evaluation
   - Full López de Prado suite
   - Decide if features worth training Keras
2. **Keras Validation**: `test_pbo_quick.py`
   - **LIMITED** evaluation (PBO only)
   - Simpler, focused script
   - Same result as fin_model_evaluation.py for Keras
3. **DON'T expect comprehensive Keras evaluation** - it's impossible without retraining!
