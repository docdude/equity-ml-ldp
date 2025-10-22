# Feature Selection Methodology for Triple Barrier Prediction

## Understanding Feature Importance Metrics

### 1. MDI (Mean Decrease in Impurity) - Training Performance
**What it measures:** How much each feature improves the decision tree splits during training.

**Characteristics:**
- ✅ Fast to compute (byproduct of Random Forest training)
- ✅ Shows which features the model "uses" during training
- ❌ **BIASED towards high-cardinality features** (many unique values)
- ❌ **BIASED towards continuous features** over categorical
- ❌ **Can be inflated by correlated features** - model picks one arbitrarily
- ❌ **In-sample only** - doesn't test out-of-sample predictive power

**What MDI tells you:**
- High MDI = "Model finds this feature useful for splitting nodes"
- Low MDI = "Model doesn't use this feature much in tree construction"
- **WARNING:** High MDI ≠ predictive power! Could be overfitting.

---

### 2. MDA (Mean Decrease in Accuracy) - Predictive Power
**What it measures:** How much out-of-sample accuracy drops when you permute (shuffle) this feature.

**Characteristics:**
- ✅ **Out-of-sample test** - evaluates on data model hasn't seen
- ✅ **Measures true predictive value** - if shuffling hurts accuracy, feature matters
- ✅ Less biased than MDI
- ❌ Slower to compute (requires multiple permutations)
- ❌ **Can be diluted by correlated features** - if two features are 99% correlated, shuffling one doesn't hurt much because the other compensates

**What MDA tells you:**
- High MDA = "This feature contains unique predictive information"
- Low MDA = "Model doesn't lose accuracy without this feature"
- **KEY:** This is your TRUE measure of feature importance for prediction

---

### 3. Orthogonalized MDA - Eliminating Correlation Effects
**What it measures:** MDA after removing the effect of correlated features.

**The Problem MDA Has:**
```
Example: 
- Feature A: volume_norm (normalized volume)
- Feature B: volume_zscore (z-scored volume)
- Correlation: 0.95

If you shuffle volume_norm, volume_zscore still carries 95% of the information!
→ MDA shows volume_norm as "unimportant" even though volume IS important
→ This is the "substitution effect"
```

**Orthogonalization Process:**
1. For each feature X, find all correlated features (|correlation| > threshold, typically 0.7)
2. Remove from X the linear component explained by correlated features
3. Compute MDA on the orthogonalized (decorrelated) feature
4. This gives you the **unique information** that feature provides

**What Orthogonal MDA tells you:**
- High Ortho MDA = "This feature has unique predictive information not captured by other features"
- Low Ortho MDA but High MDA = "Feature is important but mostly redundant with others"

---

## The Four Quadrants: MDI vs MDA Analysis

### Quadrant 1: **HIGH MDI + HIGH MDA** → ✅ ROBUST FEATURES
**Interpretation:** Important in training AND predictive out-of-sample
**Action:** **KEEP THESE** - Core features with real predictive power
**Example from your results:**
- volume_zscore (MDI: 0.0104, MDA: 0.0132)
- cci (MDI: 0.0102, MDA: 0.0132)
- vpin (MDI: 0.0107, MDA: 0.0119) ← **NOW WORKING!**

---

### Quadrant 2: **HIGH MDI + LOW MDA** → ⚠️ OVERFITTED FEATURES
**Interpretation:** Model uses them heavily in training, but they don't predict out-of-sample
**Possible Causes:**
1. **Overfitting** - Feature captures training noise, not signal
2. **Data leakage** - Feature contains future information
3. **Substitution effect** - Feature is highly correlated with another feature that has high MDA

**Action:** 
- **INVESTIGATE** - Check correlation with other features
- If correlated with High MDA feature → Keep the high MDA one, drop this
- If not correlated → **REMOVE** (likely overfitting)

**Example from your results:**
- adx_plus (MDI: 0.0115, MDA: 0.0015) ← Used in training, but no prediction power!
- vol_of_vol (MDI: 0.0109, MDA: 0.0014) ← Similar issue

---

### Quadrant 3: **LOW MDI + HIGH MDA** → 🔄 SUBSTITUTED FEATURES
**Interpretation:** Predictive, but model chose correlated features instead
**Cause:** Random Forest picks ONE feature from a correlated group arbitrarily during training

**Action:**
- **CHECK CORRELATION** - Find what feature the model used instead
- **CONSIDER SWAPPING** - The high MDA feature might be better conceptually
- Often indicates you have multiple ways to capture the same information

**Example from your results:**
- log_return_3d (MDI: 0.0093, MDA: 0.0107) ← Predictive but model preferred other return horizons
- max_drawdown_20 (MDI: 0.0082, MDA: 0.0089) ← Risk measure diluted by volatility features

---

### Quadrant 4: **LOW MDI + LOW MDA** → ❌ UNIMPORTANT FEATURES
**Interpretation:** Not useful in training AND not predictive
**Action:** **REMOVE** - Dead weight, adds noise and computation

**Example from your results:**
- amihud_illiquidity (MDI: 0.0070, MDA: -0.0023) ← Actually HURTS accuracy!
- market_state (MDI: 0.0035, MDA: -0.0014) ← Categorical, not predictive

---

## How To Select Features: Step-by-Step Process

### Step 1: Start with Orthogonal MDA Rankings
**Why:** Orthogonal MDA shows unique predictive information, accounting for correlations.

From your notebook's TOP 20 Orthogonal MDA:
```
1. serial_corr_1 (0.0188)
2. hl_volatility_ratio (0.0181)
3. macd_hist (0.0176)
4. macd_divergence (0.0176)
5. macd_signal (0.0171)
...
```

**These are features with unique, non-redundant predictive power.**

---

### Step 2: Cross-Check with MDI-MDA Quadrants
For each high Ortho MDA feature, check its quadrant:

- **Quadrant 1 (High MDI + High MDA):** ✅ Definitely keep
- **Quadrant 2 (High MDI + Low MDA):** ⚠️ Check correlation - might be overfitted
- **Quadrant 3 (Low MDI + High MDA):** ✅ Keep - provides unique info model didn't fully exploit
- **Quadrant 4 (Low MDI + Low MDA):** ❌ Remove

---

### Step 3: Remove Redundant Features
**Check correlation matrix for features with |corr| > 0.7:**

Example redundancy groups:
```
Group 1: Volume features
- volume_norm, volume_zscore, relative_volume (all measure volume deviation)
- Solution: Keep the ONE with highest Ortho MDA

Group 2: Bollinger Bands
- bb_upper, bb_lower, bb_position, bb_percent_b (bb_percent_b = bb_position)
- Solution: Keep bb_position or bb_percent_b, drop the other

Group 3: MACD family
- macd, macd_signal, macd_hist, macd_divergence (derived from each other)
- Solution: Keep 2-3 with highest Ortho MDA (they capture different aspects)
```

---

### Step 4: Set Feature Count Based on Sample Size
**Rule of thumb:** Number of features ≈ sqrt(number of samples) for tree-based models

Your dataset:
- Training samples: ~500-1000 (depending on ticker count and time period)
- Suggested features: **20-40 features**

**Why not more?**
- Curse of dimensionality
- Overfitting risk increases
- Computation cost
- Model becomes harder to interpret

**Why not fewer?**
- Miss important predictive signals
- Underfit the data
- Waste available information

---

## Feature Selection for Triple Barrier Prediction

### What Makes a Good Feature for Triple Barrier?

**Triple Barrier Task:** Predict which barrier will be hit first (TP, SL, or Timeout)

**Key predictive patterns:**
1. **Momentum/Trend** - Direction of next move
   - Returns, MACD, trend indicators
   
2. **Volatility Regime** - How far price will move
   - Volatility measures, ATR, Bollinger width
   
3. **Market Microstructure** - Order flow toxicity
   - VPIN, spread estimators, volume imbalance
   
4. **Mean Reversion** - Probability of reversal
   - Stochastic, RSI, distance from MA

**Features to prioritize:**
- **High Ortho MDA** - Unique predictive information
- **Low correlation** - Diverse information sources
- **Interpretable** - Understand WHY model makes predictions
- **Stable across regimes** - Works in different market conditions

---

## Recommended Feature Selection Strategy

### Option A: Orthogonal MDA Top-N (Recommended)
**Method:** Take top N features by Orthogonal MDA, remove highly correlated pairs

**Steps:**
1. Rank by Ortho MDA
2. Starting from top, add feature if |correlation| < 0.7 with all selected features
3. Stop at target count (e.g., 30-40 features)

**Pros:**
- ✅ Maximizes unique predictive information
- ✅ Automatically handles redundancy
- ✅ Based on out-of-sample performance

**Cons:**
- ⚠️ Might miss feature interactions
- ⚠️ Computationally expensive to calculate

---

### Option B: Quadrant-Based Selection (Conservative)
**Method:** Only take Quadrant 1 (High MDI + High MDA) features

**Steps:**
1. Set MDI threshold (e.g., > median)
2. Set MDA threshold (e.g., > 0.005)
3. Take all features above both thresholds
4. Remove correlated pairs

**Pros:**
- ✅ Safe - only proven features
- ✅ Fast to compute
- ✅ Interpretable selection

**Cons:**
- ⚠️ Might miss substituted features (Quadrant 3)
- ⚠️ Smaller feature set

---

### Option C: Hybrid Approach (Balanced)
**Method:** Combine Ortho MDA rankings with MDI-MDA validation

**Steps:**
1. Start with top 50 Ortho MDA features
2. Remove Quadrant 4 features (Low MDI + Low MDA)
3. For Quadrant 2 features (High MDI + Low MDA):
   - Check if correlated with Quadrant 1 feature → Remove
   - If not correlated → Flag for investigation
4. Remove correlated pairs (keep higher Ortho MDA)
5. Target 30-40 features

**Pros:**
- ✅ Best of both worlds
- ✅ Removes clear dead weight
- ✅ Keeps diverse information

**Cons:**
- ⚠️ More complex process
- ⚠️ Requires judgment calls

---

## Your Current Results: What They Tell Us

### Finding 1: VPIN is NOW WORKING! 🎉
**Previous:** MDA = 0.0 (constant values, broken implementation)
**Now:** MDI = 0.0107, MDA = 0.0119 (Quadrant 1 - Robust!)

**Ranking:** #5 in Robust Features, #8 in Ortho MDA

**Interpretation:** VPIN provides unique predictive information about informed trading.
- High order imbalance → Directional move likely
- Complements other features (low correlation with technical indicators)

**Recommendation:** **KEEP VPIN** in feature set for informed trading signals.

---

### Finding 2: Market Features (GC=F, ^GSPC) Are Predictive But Underused
**GC=F (Gold):** MDI = 0.0071, MDA = 0.0109 (Quadrant 3)
**^GSPC (S&P500):** MDI = 0.0080, MDA = 0.0075 (Quadrant 3)

**Interpretation:** 
- Predictive (decent MDA)
- Model didn't exploit them fully during training (low MDI)
- Likely because feature set is dominated by single-stock features

**Recommendation:** 
- If predicting individual stocks: Keep as market context
- These capture systematic risk not in stock-specific features

---

### Finding 3: Several MACD Features in Top 20, But Risk of Redundancy
**MACD family in TOP 20 Ortho MDA:**
- macd_hist (#3: 0.0176)
- macd_divergence (#4: 0.0176)
- macd_signal (#5: 0.0171)
- macd (#9: 0.0158)

**All 4 are derived from the same MACD calculation!**

**Recommendation:**
- Keep **macd_hist** and **macd_signal** (highest Ortho MDA, capture different aspects)
- Consider dropping macd and macd_divergence (redundant)
- This frees up "feature budget" for other information sources

---

### Finding 4: Volume Features Are Robust
**Robust volume features:**
- volume_zscore (MDI: 0.0104, MDA: 0.0132) - #1 Robust!
- volume_norm (MDI: 0.0094, MDA: 0.0125) - #3 Robust
- relative_volume (MDI: 0.0103, MDA: 0.0108) - #7 Robust

**All measure volume deviation, highly correlated (>0.9)**

**Recommendation:**
- Keep ONE: volume_zscore (highest MDA)
- Drop volume_norm and relative_volume
- Saves 2 feature slots, no loss of information

---

### Finding 5: Some Features Are Dead Weight
**Low MDI + Low/Negative MDA:**
- amihud_illiquidity (MDI: 0.0070, MDA: -0.0023) ← **HURTS** performance!
- market_state (MDI: 0.0035, MDA: -0.0014)
- log_return_1d (MDI: 0.0074, MDA: -0.0041)
- williams_r (MDI: 0.0079, MDA: -0.0003)

**Recommendation:** **REMOVE ALL** - Free up 4+ feature slots for better features

---

## Actionable Feature Set Recommendations

### Recommendation 1: Start with TOP 25 Ortho MDA, Remove Redundancy

**Process:**
```python
# Top 25 Ortho MDA features
top_25_ortho = [
    'serial_corr_1', 'hl_volatility_ratio', 'macd_hist', 'macd_divergence',
    'macd_signal', 'amihud_illiq', 'roll_spread', 'vpin',
    'macd', 'relative_volatility', 'skewness_20', 'sar',
    'bb_upper', 'hurst_exponent', 'vwap_20', 'ad_roc',
    'bb_lower', 'return_entropy', 'adx', 'cci'
    # ... up to 25
]

# Remove redundancy:
# 1. Keep macd_hist, drop macd_divergence (identical Ortho MDA)
# 2. Keep bb_upper, drop bb_lower (correlated, similar rank)
# 3. Remove amihud_illiq (negative MDA)

# Add critical features that might be lower ranked but essential:
# - volume_zscore (robust #1)
# - volatility measures
# - distance from MA features
```

**Target:** 30-35 features after redundancy removal

---

### Recommendation 2: Feature Groups to Balance

**Aim for diversity across:**

1. **Momentum (6-8 features):**
   - MACD family (2-3): macd_hist, macd_signal
   - RSI/Stochastic (1-2): rsi_7 or stoch_k_d_diff
   - Returns (2-3): log_return_3d, log_return_5d, log_return_10d

2. **Volatility (5-7 features):**
   - Historical vol (2-3): volatility_ht_60, volatility_yz_20
   - Realized vol (1-2): realized_vol_positive
   - Relative vol (1): relative_volatility
   - Volatility ratios (1): hl_volatility_ratio

3. **Volume (3-4 features):**
   - Normalized (1): volume_zscore
   - VWAP-based (1): vwap_20 or price_vwap_ratio
   - Accumulation (1): ad_roc or obv_roc

4. **Microstructure (3-5 features):**
   - VPIN (1): vpin ← **KEY ADDITION!**
   - Spread estimates (1-2): roll_spread, hl_range
   - Illiquidity (1): amihud_illiq (if positive MDA)

5. **Trend/Position (5-7 features):**
   - Bollinger (2): bb_position, bb_width
   - MA distance (2-3): dist_from_ma20, dist_from_ma50
   - Trend strength (1-2): adx, sar

6. **Statistical (3-5 features):**
   - Autocorrelation (1): serial_corr_1
   - Distribution (2): skewness_20, kurtosis_60
   - Entropy (1-2): return_entropy, hurst_exponent

7. **Market Context (2-3 features):**
   - Market index: ^GSPC or GC=F
   - Volatility index: ^VIX (if available)

**Total: 30-40 features across 7 groups**

---

### Recommendation 3: Validation Process

**Before finalizing feature set:**

1. **Check Correlation Matrix:**
   ```python
   import seaborn as sns
   corr_matrix = X[selected_features].corr()
   sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
   
   # Flag any |corr| > 0.7
   high_corr_pairs = []
   for i in range(len(corr_matrix.columns)):
       for j in range(i+1, len(corr_matrix.columns)):
           if abs(corr_matrix.iloc[i, j]) > 0.7:
               high_corr_pairs.append((
                   corr_matrix.columns[i],
                   corr_matrix.columns[j],
                   corr_matrix.iloc[i, j]
               ))
   ```

2. **Run Feature Importance Again:**
   - Verify features are still predictive with reduced set
   - Check if removing correlated features boosted others

3. **Cross-Validation with Multiple Metrics:**
   - Accuracy (overall)
   - Precision/Recall per barrier (TP, SL, Timeout)
   - Check for class imbalance issues

4. **Walk-Forward Validation:**
   - Test on multiple time periods
   - Ensure features stable across market regimes

---

## Summary: Recommended Feature Set (35 features)

Based on your results, here's my recommended starting point:

### Tier 1: Core Robust Features (High MDI + High MDA) - 10 features
```
1. volume_zscore       (Robust #1)
2. cci                 (Robust #2)
3. volume_norm         (Robust #3) - OR keep volume_zscore only
4. bb_position         (Robust #4)
5. vpin                (Robust #5) ← **NEWLY WORKING!**
6. bb_percent_b        (Robust #6) - Check if same as bb_position
7. relative_volume     (Robust #7)
8. price_vwap_ratio    (Robust #8)
9. realized_vol_positive (Robust #9)
10. dist_from_ma50     (Robust #10)
```

### Tier 2: High Ortho MDA (Unique Information) - 15 features
```
11. serial_corr_1      (Ortho #1)
12. hl_volatility_ratio (Ortho #2)
13. macd_hist          (Ortho #3)
14. macd_signal        (Ortho #5)
15. roll_spread        (Ortho #7)
16. relative_volatility (Ortho #10)
17. skewness_20        (Ortho #11)
18. sar                (Ortho #12)
19. bb_upper           (Ortho #13)
20. hurst_exponent     (Ortho #15)
21. vwap_20            (Ortho #16)
22. ad_roc             (Ortho #17)
23. bb_lower           (Ortho #18)
24. return_entropy     (Ortho #19)
25. adx                (Ortho #20)
```

### Tier 3: Supporting Features (Fill Gaps) - 10 features
```
26. log_return_3d      (Returns horizon)
27. log_return_5d      (Returns horizon)
28. log_return_10d     (Returns horizon)
29. volatility_ht_60   (Long-term vol)
30. dollar_volume_ma_ratio (Volume context)
31. max_drawdown_20    (Risk measure)
32. kurtosis_20        (Tail risk)
33. dist_from_ma20     (Short-term position)
34. GC=F or ^GSPC      (Market context)
35. risk_adj_momentum_20 (Risk-adjusted return)
```

### Features to REMOVE (Dead weight):
```
❌ amihud_illiquidity  (Negative MDA)
❌ market_state        (Negative MDA)
❌ log_return_1d       (Negative MDA, too noisy)
❌ williams_r          (Negative MDA)
❌ aroon_up            (Near-zero MDA)
```

---

## How This Selection Works for Triple Barrier

**For TP barrier (profit target):**
- Momentum: macd_hist, macd_signal, cci
- Trend: sar, adx, dist_from_ma*
- Volume surge: volume_zscore, vpin (informed buying)

**For SL barrier (stop loss):**
- Volatility: hl_volatility_ratio, realized_vol_positive, volatility_ht_60
- Mean reversion: bb_position, serial_corr_1
- Risk: max_drawdown_20, kurtosis_20

**For Timeout (no barrier hit):**
- Low vol regime: relative_volatility (low)
- Range-bound: bb_position (mid-range)
- Low volume: volume_zscore (low)

**The feature set covers ALL three scenarios with complementary information!**

---

## Next Steps

1. **Implement recommended 35-feature set**
2. **Train model and compare to current 42-feature set**
3. **Monitor per-barrier precision/recall** (not just overall accuracy)
4. **Iterate:** Add/remove based on per-barrier performance
5. **Document:** Which features predict which barriers

This systematic approach ensures you're not just chasing numbers, but building a feature set that makes logical sense for the prediction task!
