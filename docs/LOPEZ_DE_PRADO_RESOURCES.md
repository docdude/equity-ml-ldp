# López de Prado Statistical Testing Resources

## Overview
This document consolidates resources for implementing rigorous statistical tests on our ML trading models, based on the work of Dr. Marcos López de Prado in "Advances in Financial Machine Learning" (2018).

---

## 📚 Primary Resources

### 1. Hudson & Thames - MLFinLab
**Repository:** https://github.com/hudson-and-thames/mlfinlab

**Description:** 
The most comprehensive open-source implementation of López de Prado's research. This library contains production-ready implementations of all the tests we need.

**Key Modules for Our Use:**

#### a) Probability of Backtest Overfitting (PBO)
- **Module:** `mlfinlab.backtest_statistics.backtests`
- **Function:** `probability_of_backtest_overfitting()`
- **What it does:**
  - Takes your model's probability predictions (e.g., 0.65 = 65% chance stock goes up)
  - Splits predictions into In-Sample (first 50%) and Out-of-Sample (second 50%)
  - Measures performance degradation from IS to OOS
  - Returns PBO score: <0.3 = good, >0.7 = overfit
  
- **Our Use Case:**
  ```python
  from mlfinlab.backtest_statistics import probability_of_backtest_overfitting
  
  # After running evaluate.py walkforward with Trial #150
  predictions_df = pd.read_parquet('artifacts/trial_150_predictions.parquet')
  
  # PBO needs: dates, y_true, y_pred_proba
  pbo_score = probability_of_backtest_overfitting(
      predictions_df['y_pred_proba'],
      predictions_df['y_true'],
      n_splits=2  # Split into IS/OOS
  )
  
  if pbo_score < 0.3:
      print("✅ Safe to deploy - low overfitting risk")
  elif pbo_score > 0.7:
      print("❌ DO NOT DEPLOY - high overfitting detected")
  ```

#### b) Deflated Sharpe Ratio (DSR)
- **Module:** `mlfinlab.backtest_statistics.statistics`
- **Function:** `deflated_sharpe_ratio()`
- **What it does:**
  - Adjusts Sharpe ratio for multiple testing (we tried 252 different strategy configurations in Bayesian optimization!)
  - Accounts for skewness and kurtosis in returns
  - Returns probability that observed SR is due to luck (adjusted for trying many configurations)
  
- **IMPORTANT: The Workflow**
  ```
  Step 1: Bayesian Optimization (DONE)
  ├─ Tested 252 hyperparameter configurations
  ├─ Used AUC to compare them (NOT Sharpe ratio)
  └─ Trial #150 won with 0.7131 AUC
  
  Step 2: Trading Backtest (TODO - only for Trial #150!)
  ├─ Train model with Trial #150's hyperparameters
  ├─ Generate predictions on OOS data
  ├─ Convert predictions → trading signals (e.g., buy if prob > 0.65)
  ├─ Calculate returns from those trades
  └─ Calculate Sharpe ratio from trading returns
  
  Step 3: Apply DSR (TODO)
  ├─ Use Trial #150's Sharpe ratio
  ├─ Tell DSR we tried 252 configurations
  └─ DSR adjusts for "best of 252" selection bias
  ```
  
- **Our Use Case:**
  ```python
  from mlfinlab.backtest_statistics import deflated_sharpe_ratio
  
  # Step 1: Run ONLY Trial #150 in trading backtest (not all 252!)
  predictions = run_walkforward_with_trial_150_params()
  
  # Step 2: Convert ML predictions to trading strategy
  signals = (predictions['y_pred_proba'] > 0.65).astype(int)  # Trading rule
  returns = signals * predictions['actual_returns']  # Strategy returns
  
  # Step 3: Calculate Sharpe ratio of the TRADING STRATEGY
  observed_sharpe = returns.mean() / returns.std() * np.sqrt(252)
  
  # Step 4: Apply DSR - adjusts for trying 252 configurations
  # NOTE: We DON'T need Sharpe ratios from all 252 trials!
  #       DSR uses order statistics to adjust automatically
  dsr = deflated_sharpe_ratio(
      observed_sr=observed_sharpe,      # Only Trial #150's SR from trading
      number_of_tests=252,               # We tried 252 configurations
      number_of_returns=len(returns),    # Days of trading data
      skewness=returns.skew(),
      kurtosis=returns.kurt()
  )
  
  if dsr > 0.95:
      print("✅ Statistically significant at 95% confidence")
  else:
      print("⚠️ Performance may be due to luck")
  ```
  
- **Key Point:**
  - We optimized on **AUC** (ML metric) across 252 trials
  - We backtest **only the winner** (Trial #150) for **Sharpe ratio** (finance metric)
  - DSR adjusts Trial #150's Sharpe for "best of 252" selection bias
  - We do NOT need to backtest all 252 trials (would take weeks!)

#### c) Minimum Backtest Length (MinBTL)
- **Module:** `mlfinlab.backtest_statistics.statistics`
- **Function:** `minimum_track_record_length()`
- **What it does:**
  - Calculates how many data points you need for statistical significance
  - Formula: `((Z_α / SR)² × (1 + (1-γ) × SR² / 2))`
  - Z_α = 1.96 for 95% confidence
  - γ = skewness adjustment
  
- **Our Use Case:**
  ```python
  from mlfinlab.backtest_statistics import minimum_track_record_length
  
  min_length = minimum_track_record_length(
      sharpe_ratio=1.2,  # Trial #150's SR
      confidence_level=0.95,
      skewness=0.1,
      target_sharpe=0.8  # What we'd accept as "real"
  )
  
  actual_length = len(backtest_returns)
  
  if actual_length >= min_length:
      print(f"✅ Backtest length sufficient: {actual_length} >= {min_length}")
  else:
      print(f"⚠️ Need {min_length - actual_length} more data points")
  ```

#### d) Combinatorial Purged Cross-Validation (CPCV)
- **Module:** `mlfinlab.cross_validation.combinatorial`
- **Function:** `CombinatorialPurgedKFold`
- **What it does:**
  - Tests ALL possible combinations of k folds (not just k sequential splits)
  - For k=5, tests 10 different train/test combinations
  - Validates that model is stable across all splits
  
- **Our Use Case:**
  ```python
  from mlfinlab.cross_validation import CombinatorialPurgedKFold
  
  cv = CombinatorialPurgedKFold(
      n_splits=5,
      n_test_splits=2,  # Use 2 folds for testing each time
      embargo_pct=0.01  # 1% embargo between train/test
  )
  
  # Test all combinations of folds
  scores = []
  for train_idx, test_idx in cv.split(X, pred_times=dates):
      X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
      y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
      
      model.fit(X_train, y_train)
      scores.append(model.score(X_test, y_test))
  
  print(f"Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
  if np.std(scores) < 0.05:
      print("✅ Model is stable across all fold combinations")
  ```

**Installation:**
```bash
pip install mlfinlab
```

**Documentation:** https://mlfinlab.readthedocs.io/

---

### 2. QuantConnect Research
**Repository:** https://github.com/QuantConnect/Research

**Description:**
Alternative implementations and research notebooks from QuantConnect's team. Good for understanding the intuition behind the tests.

**Key Notebooks:**
- `Multiple-Testing/Deflated-Sharpe-Ratio.ipynb` - DSR implementation
- `Multiple-Testing/Haircut-Sharpe-Ratios.ipynb` - Adjusting for data mining
- `Cross-Validation/Combinatorial-CV.ipynb` - CPCV examples

**Our Use Case:**
Review these notebooks to understand:
- Why multiple testing matters (we tested 252 different strategy configurations!)
- How to interpret DSR values
- Visual examples of overfitting detection

---

### 3. Original Book Code
**Repository:** https://github.com/BlackArbsCEO/Advances_in_Financial_Machine_Learning

**Description:**
Unofficial collection of code snippets from the book. More educational than production-ready.

**Key Files:**
- `Chapter7_CrossValidation.py` - Purged K-Fold implementation
- `Chapter11_BacktestOverfitting.py` - PBO calculation
- `Chapter14_BacktestStatistics.py` - DSR and MinBTL

**Note:** Use mlfinlab for production; use this for understanding the math.

---

## 🎯 Our Implementation Plan

### Phase 1: Prepare Data (Modify evaluate.py)
**Goal:** Save probability predictions from walk-forward validation

**Current State:**
```python
# evaluate.py currently saves only metrics
results.append({
    'fold': i,
    'train_dates': f"{train_dates.min()}:{train_dates.max()}",
    'test_dates': f"{test_dates.min()}:{test_dates.max()}",
    'auc': auc_score
})
```

**Needed Change:**
```python
# Need to save: date, ticker, y_true, y_pred_proba for PBO
predictions.append({
    'date': test_panel['date'],
    'ticker': test_panel['ticker'],
    'y_true': test_y,
    'y_pred_proba': model.predict_proba(test_X)[:, 1],  # Probability of class 1
    'fold': i
})

# Save to artifacts/trial_150_predictions.parquet
```

### Phase 2: Extract Trial #150 Configuration
**Goal:** Get hyperparameters from Optuna study

```python
import optuna
from sqlalchemy import create_engine

# Load study from PostgreSQL
study = optuna.load_study(
    study_name="production_enhanced_20251004_051123",
    storage="postgresql://user:pass@localhost/optuna_db"
)

# Get best trial (#150)
best_trial = study.best_trial
params = best_trial.params

# Save to config/trial_150_params.yaml
with open('config/trial_150_params.yaml', 'w') as f:
    yaml.dump(params, f)
```

### Phase 3: Run Walk-Forward with Trial #150
**Goal:** Get honest predictions from leak-free validation

```bash
# Run the corrected evaluate.py with Trial #150 params
python evaluate.py --config config/trial_150_params.yaml

# This will:
# 1. Train 4 separate models (one per fold)
# 2. Each model trained only on historical data
# 3. Save predictions to artifacts/trial_150_predictions.parquet
```

### Phase 4: Run López de Prado Tests
**Goal:** Validate Trial #150 before deployment

```python
# Create new file: scripts/lopez_prado_validation.py
import pandas as pd
from mlfinlab.backtest_statistics import (
    probability_of_backtest_overfitting,
    deflated_sharpe_ratio,
    minimum_track_record_length
)

# Load predictions
predictions = pd.read_parquet('artifacts/trial_150_predictions.parquet')

# Test 1: PBO
pbo_score = probability_of_backtest_overfitting(
    predictions['y_pred_proba'],
    predictions['y_true']
)
print(f"PBO Score: {pbo_score:.3f}")
print(f"Interpretation: {'✅ SAFE' if pbo_score < 0.3 else '❌ OVERFIT' if pbo_score > 0.7 else '⚠️ BORDERLINE'}")

# Test 2: DSR
returns = calculate_strategy_returns(predictions)
dsr = deflated_sharpe_ratio(
    observed_sr=returns.mean() / returns.std() * np.sqrt(252),
    sr_estimates=get_all_trial_sharpes(),  # Sharpe ratios from all 252 Bayesian opt configurations
    number_of_returns=len(returns),
    skewness=returns.skew(),
    kurtosis=returns.kurt(),
    number_of_tests=252  # Number of different configurations we tested
)
print(f"DSR: {dsr:.3f}")
print(f"Interpretation: {'✅ SIGNIFICANT' if dsr > 0.95 else '⚠️ NOT SIGNIFICANT'}")

# Test 3: MinBTL
min_length = minimum_track_record_length(
    sharpe_ratio=returns.mean() / returns.std() * np.sqrt(252),
    confidence_level=0.95
)
print(f"Minimum Length: {min_length} days")
print(f"Actual Length: {len(returns)} days")
print(f"Status: {'✅ SUFFICIENT' if len(returns) >= min_length else '❌ INSUFFICIENT'}")
```

### Phase 5: Make Deployment Decision

**Decision Matrix:**

| Test | Threshold | Trial #150 Result | Status |
|------|-----------|-------------------|--------|
| PBO  | < 0.3 (good) | TBD | ⏳ |
| DSR  | > 0.95 (significant) | TBD | ⏳ |
| MinBTL | Actual > Required | TBD | ⏳ |
| AUC | > 0.65 | 0.7131 | ✅ |
| Convergence | 100+ trials plateau | 116 trials | ✅ |

**Deployment Criteria:**
- ✅ ALL of: PBO < 0.5, DSR > 0.90, MinBTL satisfied
- ✅ IDEAL: PBO < 0.3, DSR > 0.95, MinBTL satisfied

**If Tests Fail:**
- PBO > 0.7: Model is overfit, DO NOT DEPLOY
- DSR < 0.90: Performance may be luck, need more validation
- MinBTL not satisfied: Need longer backtest period

---

## 📖 Further Reading

### Books
1. **"Advances in Financial Machine Learning" (2018)** - Marcos López de Prado
   - Chapter 7: Cross-Validation in Finance
   - Chapter 11: The Dangers of Backtesting
   - Chapter 14: Backtest Statistics

2. **"Machine Learning for Asset Managers" (2020)** - Marcos López de Prado
   - More practical examples
   - Portfolio construction with ML

### Papers
1. **"The Probability of Backtest Overfitting"** (2013)
   - Bailey, Borwein, López de Prado, Zhu
   - Journal of Computational Finance
   - Original PBO paper with mathematical proofs

2. **"Pseudo-Mathematics and Financial Charlatanism"** (2014)
   - Bailey, Borwein, López de Prado
   - Critique of spurious Sharpe ratios

3. **"The Deflated Sharpe Ratio"** (2014)
   - Bailey, López de Prado
   - Adjusting performance for multiple testing

### Online Resources
1. **Hudson & Thames Blog:** https://hudsonthames.org/blog/
   - Practical implementations
   - Case studies with real data

2. **QuantConnect Tutorials:** https://www.quantconnect.com/tutorials/
   - Integration with backtesting platform
   - Live examples

3. **Dr. López de Prado's Website:** https://www.quantresearch.org/
   - Updated research papers
   - Code examples

---

## 🔄 ML Optimization vs Trading Backtest (CRITICAL DISTINCTION!)

### The Two-Stage Process

Many people get confused about DSR because it involves TWO different stages:

#### Stage 1: Bayesian Optimization (What we already did)
```python
# Objective: Find best hyperparameters using ML metric (AUC)
# Metric: AUC (Area Under ROC Curve)
# NOT using Sharpe ratio yet!

Trial #1:   {n_estimators: 100, learning_rate: 0.01, ...}  → AUC = 0.650
Trial #2:   {n_estimators: 200, learning_rate: 0.05, ...}  → AUC = 0.672
...
Trial #150: {n_estimators: 150, learning_rate: 0.03, ...}  → AUC = 0.7131 ✅
...
Trial #252: {n_estimators: 80, learning_rate: 0.02, ...}   → AUC = 0.645

# Result: Trial #150 wins with AUC = 0.7131
# Time: ~5 days to run 252 trials
```

#### Stage 2: Trading Backtest (What we need to do next)
```python
# Objective: Validate that Trial #150 makes money in trading
# Metric: Sharpe ratio (risk-adjusted returns)
# NOW we use finance metrics!

# Take ONLY Trial #150's hyperparameters
params_150 = study.best_trial.params

# Run full trading backtest with those params
model = train_with_params(params_150)
predictions = model.predict_proba(X_test)

# Convert predictions to trading signals
signals = (predictions[:, 1] > 0.65).astype(int)  # Buy if prob > 65%

# Calculate trading returns
strategy_returns = signals * market_returns

# Calculate Sharpe ratio
sharpe_150 = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
# Example result: sharpe_150 = 1.15

# Apply DSR
dsr = deflated_sharpe_ratio(
    observed_sr=1.15,     # Trial #150's Sharpe
    number_of_tests=252,  # We tried 252 configurations to find Trial #150
    ...
)
```

### Common Confusion: "Do I need to backtest all 252 trials?"

**Short Answer: NO! Only backtest Trial #150.**

**Why it's confusing:**
- You optimized 252 configurations → You think you need 252 Sharpe ratios
- But López de Prado's DSR uses **order statistics**, not all values

**How DSR actually works:**
```python
# DSR doesn't need all 252 Sharpe ratios
# It only needs:
# 1. The BEST Sharpe ratio (from Trial #150)
# 2. The NUMBER of trials (252)
# 3. Return statistics (skewness, kurtosis)

# DSR then calculates:
# "If I randomly generated 252 strategies with similar return characteristics,
#  what's the probability that the MAXIMUM Sharpe ratio would be >= 1.15?"

# This is called the "distribution of the maximum" in statistics
# No need to actually compute all 252 Sharpe ratios!
```

### Practical Example

```python
# ❌ WRONG (would take weeks):
for trial in all_252_trials:
    model = train(trial.params)
    returns = backtest(model)
    sharpe_ratios.append(calculate_sharpe(returns))
dsr = deflated_sharpe_ratio(observed_sr=max(sharpe_ratios), ...)

# ✅ RIGHT (takes a few hours):
best_trial = study.best_trial  # Trial #150
model = train(best_trial.params)
returns = backtest(model)
best_sharpe = calculate_sharpe(returns)
dsr = deflated_sharpe_ratio(
    observed_sr=best_sharpe,
    number_of_tests=252,  # ← Just tell it how many we tried
    ...
)
```

### The Math Behind It

López de Prado's DSR formula uses:
- **Expected maximum Sharpe**: E[max(SR₁, SR₂, ..., SR₂₅₂)]
- **Variance of maximum**: Var[max(SR₁, SR₂, ..., SR₂₅₂)]

These can be calculated from:
- Number of trials (252)
- Mean/std of returns
- Skewness/kurtosis

No need to compute all 252 individual Sharpe ratios!

### Why This Saves Enormous Time

```
Full backtest all 252 trials:
├─ 252 trials × 3 hours per backtest
├─ = 756 hours (31 days)
└─ = $thousands in compute costs

Backtest only Trial #150:
├─ 1 trial × 3 hours
├─ = 3 hours
└─ DSR adjusts mathematically for the other 251 trials we didn't backtest
```

---

## ⚠️ Critical Warnings from López de Prado

### 1. **Data Leakage is the #1 Enemy**
> "Most academic ML papers in finance are wrong because they don't understand temporal leakage."

**What we fixed:**
- ❌ OLD: Load model trained on ALL data, test on subsets
- ✅ NEW: Train fresh model on ONLY historical data per fold

### 2. **Multiple Testing Bias**
> "If you try 100 strategies, one will look good by chance even if none work."

**What we're addressing:**
- We tested 252 different hyperparameter configurations in Bayesian optimization
- Each configuration is a different "strategy" we tried
- Must adjust Sharpe ratio for testing 252 configurations (DSR)
- PBO will detect if Trial #150 just got lucky among the 252 attempts

### 3. **The Backtest Paradox**
> "The more you torture the data, the more it will confess to anything."

**Our mitigation:**
- Use PBO to detect overfit
- Use DSR to adjust for trials
- Require convergence (116 trials plateau = good sign)

### 4. **Cross-Validation is Not Enough**
> "Standard k-fold CV in finance is wrong because of temporal dependencies."

**What we're doing:**
- Using PurgedKFold (removes overlapping samples)
- Expanding window (respects time order)
- No shuffling (maintains chronology)

---

## 🚀 Next Steps

1. **Install mlfinlab:**
   ```bash
   pip install mlfinlab
   ```

2. **Review these notebooks:**
   - Hudson & Thames PBO example
   - QuantConnect DSR tutorial
   - Book code Chapter 11

3. **Modify evaluate.py:**
   - Save probability predictions
   - Not just metrics

4. **Extract Trial #150:**
   - Get hyperparameters from Optuna
   - Save to config file

5. **Run validation:**
   - Execute walk-forward with Trial #150
   - Run lópez_prado_validation.py
   - Review results

6. **Make decision:**
   - If all tests pass: Deploy Trial #150
   - If PBO > 0.7: Back to research
   - If DSR < 0.90: Need more validation

---

## 📊 Expected Timeline

- **Modify evaluate.py:** 1 hour
- **Extract Trial #150:** 30 minutes
- **Run walk-forward:** 2-4 hours (depends on data size)
- **Install mlfinlab & run tests:** 30 minutes
- **Analysis & decision:** 1 hour

**Total:** ~5-7 hours to complete validation

---

## 🎓 Why This Matters

**Without López de Prado tests:**
- 🎲 70-80% chance model is overfit (industry average)
- 💸 Risk of deploying strategy that works in backtest but fails live
- 📉 No way to distinguish skill from luck

**With López de Prado tests:**
- 📊 Quantified probability of overfitting (PBO)
- 🔬 Adjusted performance for multiple trials (DSR)
- ✅ Confidence in deployment decision
- 🛡️ Protection against self-deception

**The Bottom Line:**
Trial #150 shows 0.7131 AUC, but is this real or luck? López de Prado tests will tell us definitively. This is the difference between professional quant research and hobby backtesting.

---

## 📞 Questions to Answer

Before starting implementation, review:

1. ✅ Do we understand what PBO measures? (Probability model is overfit)
2. ✅ Do we understand why DSR matters? (Multiple testing adjustment)
3. ✅ Do we have leak-free predictions? (Yes, root evaluate.py is fixed)
4. ⏳ Can we save probability outputs? (Need to modify evaluate.py)
5. ⏳ Do we have mlfinlab installed? (Need to install)
6. ⏳ Are we ready to deploy if tests pass? (Yes, Trial #150 ready)

**Once all ✅, proceed with implementation!**
