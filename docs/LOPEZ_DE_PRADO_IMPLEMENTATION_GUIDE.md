# López de Prado Implementation Guide

## 🎯 Quick Start: Where to Look First

This guide helps you navigate the actual code implementations to understand how these tests work in practice.

---

## 📦 1. MLFinLab (Production-Ready Code)

### Repository
```bash
git clone https://github.com/hudson-and-thames/mlfinlab
cd mlfinlab
```

### Key Files to Review

#### A) Deflated Sharpe Ratio (DSR)
**File:** `mlfinlab/backtest_statistics/statistics.py`

**Function:** `deflated_sharpe_ratio()`

**What to look for:**
```python
# The key formula for adjusting Sharpe ratio
# Look for how it calculates:
# 1. Expected maximum Sharpe given N trials
# 2. Variance of maximum Sharpe
# 3. Final deflated SR statistic

# Key parameters you'll see:
# - observed_sharpe_ratio: Your Trial #150's SR
# - number_of_trials: 252 (how many configs you tested)
# - skewness, kurtosis: Return distribution shape
```

**Lines to focus on:**
- Formula for expected maximum SR: `E[max(SR)]`
- Standard error calculation
- Final p-value computation

#### B) Probability of Backtest Overfitting (PBO)
**File:** `mlfinlab/backtest_statistics/backtests.py`

**Function:** `probability_of_backtest_overfitting()`

**What to look for:**
```python
# How PBO splits data into In-Sample / Out-of-Sample
# How it measures performance degradation
# How it calculates probability of overfit

# Key logic:
# 1. Split returns into first 50% (IS) and last 50% (OOS)
# 2. Calculate performance in each period
# 3. Measure how much performance degrades from IS to OOS
# 4. Return probability that degradation indicates overfit
```

**Lines to focus on:**
- Data splitting logic
- Performance metric calculation
- Degradation threshold

#### C) Minimum Backtest Length (MinBTL)
**File:** `mlfinlab/backtest_statistics/statistics.py`

**Function:** `minimum_track_record_length()`

**What to look for:**
```python
# Formula: T* = ((Z_α / SR)² × (1 + (1-γ) × SR²/2))
# Where:
# - Z_α = 1.96 for 95% confidence
# - SR = target Sharpe ratio
# - γ = skewness adjustment
```

---

## 📓 2. QuantConnect Research (Educational Notebooks)

### Repository
```bash
git clone https://github.com/QuantConnect/Research
cd Research
```

### Key Notebooks to Review

#### A) Deflated Sharpe Ratio Tutorial
**File:** `Multiple-Testing/Deflated-Sharpe-Ratio.ipynb`

**What you'll learn:**
- Visual explanation of multiple testing problem
- Step-by-step DSR calculation
- Interpretation of results
- Real examples with financial data

**Run it locally:**
```bash
cd Research/Multiple-Testing
jupyter notebook Deflated-Sharpe-Ratio.ipynb
```

#### B) Haircut Sharpe Ratios
**File:** `Multiple-Testing/Haircut-Sharpe-Ratios.ipynb`

**What you'll learn:**
- Different methods for adjusting Sharpe ratio
- Comparison of haircut methods
- When to use each method

#### C) Combinatorial Cross-Validation
**File:** `Cross-Validation/Combinatorial-CV.ipynb`

**What you'll learn:**
- Why standard k-fold fails in finance
- How CPCV tests all fold combinations
- Implementation with time series data

---

## 📖 3. Original Book Code (Mathematical Details)

### Repository
```bash
git clone https://github.com/BlackArbsCEO/Advances_in_Financial_Machine_Learning
cd Advances_in_Financial_Machine_Learning
```

### Key Files to Review

#### A) Chapter 11: The Dangers of Backtesting
**File:** `Chapter11/backtest_overfitting.py`

**What to look for:**
```python
# Original PBO implementation
# Shows the mathematical derivation
# Includes:
# - Combinatorial analysis
# - Performance matrix calculation
# - Statistical tests
```

**Key functions:**
- `get_performance_stats()` - Calculate strategy performance
- `pbo()` - Core PBO calculation
- `plot_pbo()` - Visualize results

#### B) Chapter 14: Backtest Statistics
**File:** `Chapter14/backtest_statistics.py`

**What to look for:**
```python
# DSR implementation
# MinBTL calculation
# Includes:
# - Sharpe ratio adjustments
# - Multiple testing corrections
# - Statistical significance tests
```

**Key functions:**
- `deflated_sharpe_ratio()` - DSR calculation
- `min_backtest_length()` - MinBTL calculation
- `sharpe_std()` - Standard error of Sharpe ratio

#### C) Chapter 7: Cross-Validation in Finance
**File:** `Chapter7/cross_validation.py`

**What to look for:**
```python
# PurgedKFold implementation
# Embargo logic
# How to handle overlapping samples in time series
```

**Key functions:**
- `PurgedKFold` - Time series CV with purging
- `embargo_samples()` - Add embargo period
- `split()` - Generate train/test indices

---

## 🔬 4. Detailed Implementation Walkthrough

### Understanding DSR Step-by-Step

```python
# Step 1: Calculate your observed Sharpe ratio
returns = strategy_returns  # From Trial #150 backtest
observed_sr = returns.mean() / returns.std() * np.sqrt(252)

# Step 2: Calculate expected maximum Sharpe given N trials
# This uses extreme value theory (Gumbel distribution)
n_trials = 252
expected_max_sr = calculate_expected_maximum(n_trials, returns)

# Step 3: Calculate standard error of Sharpe ratio
# Accounts for skewness and kurtosis
se_sr = calculate_sharpe_std_error(
    returns,
    skewness=returns.skew(),
    kurtosis=returns.kurt()
)

# Step 4: Calculate deflated Sharpe ratio
# This is a z-score: how many std errors is observed SR above expected?
dsr = (observed_sr - expected_max_sr) / se_sr

# Step 5: Convert to p-value
from scipy.stats import norm
p_value = norm.cdf(dsr)  # Probability that SR is "real"

# Interpretation:
# p_value > 0.95: Statistically significant at 95% confidence ✅
# p_value < 0.95: Could be due to luck ⚠️
```

### Understanding PBO Step-by-Step

```python
# Step 1: Get predictions from walk-forward validation
predictions_df = pd.read_parquet('artifacts/trial_150_predictions.parquet')

# Step 2: Split into In-Sample (first 50%) and Out-of-Sample (last 50%)
split_point = len(predictions_df) // 2
is_predictions = predictions_df.iloc[:split_point]
oos_predictions = predictions_df.iloc[split_point:]

# Step 3: Calculate strategy performance in each period
is_performance = calculate_performance_metric(is_predictions)
oos_performance = calculate_performance_metric(oos_predictions)

# Step 4: Calculate performance degradation
degradation = is_performance - oos_performance

# Step 5: Calculate PBO
# PBO = Probability that IS performance > OOS performance
# If PBO is high, model is overfit to IS data
pbo_score = calculate_probability_of_degradation(degradation)

# Interpretation:
# pbo_score < 0.3: Safe to deploy ✅
# pbo_score > 0.7: DO NOT DEPLOY ❌
```

---

## 🧪 5. Testing Your Understanding

### Hands-On Exercise

```python
# Create a simple test to understand DSR

import numpy as np
from scipy.stats import norm

# Simulate 252 random strategies
n_trials = 252
n_days = 1000

# Generate random returns for each strategy
all_sharpes = []
for trial in range(n_trials):
    returns = np.random.normal(0.0005, 0.02, n_days)  # Mean=0.05%/day, std=2%
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    all_sharpes.append(sharpe)

# Find the best one (this is your "Trial #150")
best_sharpe = max(all_sharpes)
print(f"Best Sharpe from 252 random strategies: {best_sharpe:.3f}")

# Question: Is this Sharpe ratio "real" or just luck?
# Expected maximum Sharpe from 252 random (zero-skill) strategies:
# E[max(SR)] ≈ 0.45 * sqrt(log(252)) ≈ 1.18

# If best_sharpe ≈ 1.18, it's probably just luck!
# This is what DSR detects.
```

### Expected Output
```
Best Sharpe from 252 random strategies: 1.23
Expected max from pure luck: 1.18
Conclusion: This "good" Sharpe is actually just luck! ⚠️
```

---

## 📚 6. Reading Order (Recommended)

### Day 1: Understand the Problem
1. Read Chapter 11 of the book (PDF/physical copy)
2. Review QuantConnect's "Deflated-Sharpe-Ratio.ipynb"
3. Understand WHY multiple testing matters

### Day 2: See the Implementation
1. Read MLFinLab's `statistics.py` DSR implementation
2. Read MLFinLab's `backtests.py` PBO implementation
3. Understand HOW the math is coded

### Day 3: Hands-On Practice
1. Clone mlfinlab repo
2. Run example notebooks
3. Test with your own data

### Day 4: Apply to Your System
1. Modify your evaluate.py to save predictions
2. Extract Trial #150 hyperparameters
3. Run walk-forward validation

### Day 5: Statistical Validation
1. Install mlfinlab
2. Run PBO test on Trial #150
3. Run DSR test on Trial #150
4. Make deployment decision

---

## 🔍 7. Key Code Snippets to Find

### In MLFinLab DSR Implementation

Look for this pattern:
```python
def deflated_sharpe_ratio(observed_sr, number_of_trials, ...):
    # Calculate expected maximum SR under null hypothesis
    expected_max_sr = norm.ppf(1 - 1/number_of_trials) * some_adjustment
    
    # Calculate standard error
    sr_std = calculate_sr_standard_error(returns, skew, kurt)
    
    # Calculate deflated SR (z-score)
    dsr_statistic = (observed_sr - expected_max_sr) / sr_std
    
    # Return probability
    return norm.cdf(dsr_statistic)
```

### In MLFinLab PBO Implementation

Look for this pattern:
```python
def probability_of_backtest_overfitting(returns, n_splits=2):
    # Split into IS/OOS
    is_returns, oos_returns = split_data(returns, n_splits)
    
    # Calculate performance
    is_perf = calculate_performance(is_returns)
    oos_perf = calculate_performance(oos_returns)
    
    # Calculate degradation
    degradation = is_perf - oos_perf
    
    # Calculate probability
    pbo = calculate_overfitting_probability(degradation)
    
    return pbo
```

---

## 🎓 8. Understanding the Math

### DSR Formula (Simplified)

```
DSR = (Observed_SR - Expected_Max_SR) / SE(SR)

Where:
- Observed_SR = Your Trial #150's Sharpe ratio
- Expected_Max_SR = E[max(SR₁, SR₂, ..., SR₂₅₂)]
                  ≈ Φ⁻¹(1 - 1/N) × some_adjustment
                  ≈ 2.5 for N=252 (needs actual calculation)
- SE(SR) = Standard error of Sharpe ratio
         = √(1 + SR²/2 - γ×SR + κ×SR²/4) / √T
         
If DSR > 1.96: Statistically significant at 95% ✅
If DSR < 1.96: Could be luck ⚠️
```

### PBO Formula (Simplified)

```
PBO = P(Rank[IS] ≠ Rank[OOS])

Intuition:
- If best strategy in IS period is also best in OOS: Low PBO ✅
- If best strategy in IS period is mediocre in OOS: High PBO ❌

PBO < 0.3: Model generalizes well ✅
PBO > 0.7: Model is overfit ❌
```

---

## 🚀 9. Next Actions

### Immediate (Today)
- [ ] Star/bookmark the three repos (mlfinlab, QuantConnect, book code)
- [ ] Read this guide completely
- [ ] Identify which specific files to review

### Short-term (This Week)
- [ ] Clone mlfinlab repo
- [ ] Read DSR implementation in `statistics.py`
- [ ] Read PBO implementation in `backtests.py`
- [ ] Run QuantConnect notebooks locally

### Medium-term (Next Week)
- [ ] Modify your evaluate.py to save predictions
- [ ] Extract Trial #308 hyperparameters (✅ DONE - saved in parallel_bayes_best.json)
- [ ] Run walk-forward with Trial #308
- [ ] Apply López de Prado tests

---

## 💡 Pro Tips

1. **Start with the notebooks**: They're easier to understand than raw code
2. **Focus on DSR first**: It's more straightforward than PBO
3. **Test with toy data**: Generate random strategies to see how the tests work
4. **Read the papers**: The original papers have better explanations than code comments
5. **Check the math**: Verify formulas against the book

---

## 📞 When You Get Stuck

### Common Issues

**Issue 1: "DSR formula looks different in each implementation"**
- Solution: They're mathematically equivalent, just different parameterizations
- Focus on understanding the CONCEPT, not memorizing the exact formula

**Issue 2: "PBO returns different values with different implementations"**
- Solution: Different implementations use different performance metrics
- Make sure you're using the same metric (Sharpe, returns, etc.)

**Issue 3: "MinBTL says I need 10 years of data but I only have 4"**
- Solution: This is a real limitation - you may need to lower your target Sharpe ratio
- Or accept higher uncertainty in results

---

## 🎯 Success Criteria

You'll know you understand the implementation when you can:

1. ✅ Explain why DSR adjusts for multiple testing
2. ✅ Calculate DSR by hand for simple example
3. ✅ Explain what PBO < 0.3 means in plain English
4. ✅ Run mlfinlab tests on your Trial #150 data
5. ✅ Make a confident deployment decision based on test results

---

## 📝 Final Checklist

Before running tests on Trial #150:

- [ ] I understand what DSR measures
- [ ] I understand what PBO measures  
- [ ] I know we only need to backtest Trial #150 (not all 252)
- [ ] I know DSR uses number_of_tests=252 to adjust
- [ ] I have mlfinlab installed
- [ ] I have Trial #150 predictions saved
- [ ] I'm ready to interpret the results
- [ ] I know the deployment thresholds (PBO<0.3, DSR>0.95)

**Once all checked, you're ready to run the tests!** 🚀
