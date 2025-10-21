# Feature Importance Analysis for WaveNet Model Selection

## Executive Summary

The **01FeatureImportance.ipynb** notebook demonstrates López de Prado's advanced feature importance methods. This is **CRITICAL for your WaveNet model** - it shows you which features to feed into WaveNet and which to discard.

**Key Insight**: RandomForest is not your prediction model - it's your **feature selection tool**. Use these methods to identify the best features, then feed only those into WaveNet.

---

## The Four Feature Importance Methods

### 1. **MDI (Mean Decrease Impurity)** - In-Sample Feature Ranking

**What it does:**
- Measures how much each feature reduces impurity (entropy/Gini) in tree splits
- Fast to compute (comes free with RandomForest)
- Sum of all feature importances = 1.0

**How it works:**
```python
forest = RandomForestClassifier(
    max_features=1,  # CRITICAL: Prevents masking effect
    n_estimators=1000,
    oob_score=True
)
forest.fit(X, y)
mdi = mean_decrease_impurity(forest, X.columns)
```

**Strengths:**
✅ No distributional assumptions
✅ Bootstrap-based (variance reduces with more trees)
✅ Normalized (0 to 1 scale)
✅ Works well for identifying informative vs noise features

**Weaknesses:**
❌ **In-sample only** (prone to overfitting)
❌ **Substitution effect**: Correlated features split importance (both get diluted)
❌ Only works with tree-based models
❌ Biased toward high-cardinality features

**When to use for WaveNet:**
- Initial screening to eliminate noise features
- Quick iteration during feature engineering
- Understanding which feature families matter (price-based, volume-based, etc.)

---

### 2. **MDA (Mean Decrease Accuracy)** - Out-of-Sample Permutation Importance

**What it does:**
- Shuffles each feature and measures accuracy drop
- Uses cross-validation (out-of-sample)
- If feature is important, shuffling destroys performance

**How it works:**
```python
forest = RandomForestClassifier(
    max_features=3,  # Can use more features than MDI
    n_estimators=1000
)
forest.fit(X, y)

# Shuffle each feature and measure accuracy drop
mda = mean_decrease_accuracy(
    forest, 
    X, 
    y, 
    cv_gen=PurgedKFold(n_splits=5, samples_info_sets=t1),
    scoring=accuracy_score
)
```

**Strengths:**
✅ **Out-of-sample** (better generalization assessment)
✅ Works with any model (not just trees)
✅ Directly measures prediction impact
✅ Can use any scoring metric (Sharpe, accuracy, F1, etc.)

**Weaknesses:**
❌ **Substitution effect**: Highly correlated features underestimated
  - Example: If features A and B are identical, shuffling A doesn't hurt (B compensates)
  - Both get low importance despite being valuable
❌ Slower to compute (requires re-prediction for each feature)
❌ Can be noisy with small datasets

**When to use for WaveNet:**
- **MOST IMPORTANT** - Final feature selection before WaveNet training
- Validates that features generalize out-of-sample
- Use with time-series aware cross-validation (PurgedKFold)

---

### 3. **SFI (Single Feature Importance)** - No Substitution Effects

**What it does:**
- Trains separate model on each feature alone
- Measures out-of-sample performance per feature
- No substitution effects (each feature evaluated independently)

**How it works:**
```python
forest = RandomForestClassifier(max_features=1, n_estimators=1000)
forest.fit(X, y)

# Train on each feature individually
sfi = single_feature_importance(
    forest, 
    X, 
    y, 
    cv_gen=PurgedKFold(n_splits=5, samples_info_sets=t1),
    scoring=accuracy_score
)
```

**Strengths:**
✅ **No substitution effects** (each feature independent)
✅ Works with any model
✅ Can identify features with standalone predictive power
✅ Out-of-sample validation

**Weaknesses:**
❌ **Misses interaction effects**
  - Feature B might only be useful with feature A
  - Feature combinations can outperform individual features
❌ Computationally expensive (N separate model fits)
❌ Underestimates features that are part of hierarchical relationships

**When to use for WaveNet:**
- Identifying "hero features" with strong standalone signal
- Debugging: If SFI is high but MDA is low → feature needs interactions
- Feature engineering validation (is this new feature useful by itself?)

---

### 4. **Orthogonal Features (PCA-Based)** - Removing Linear Substitution

**What it does:**
- Applies PCA to decorrelate features
- Removes linear multicollinearity
- Then applies MDI/MDA/SFI to orthogonal components
- Maps importance back to original features

**How it works:**
```python
# Step 1: Orthogonalize via PCA
ortho_features = get_orthogonal_features(X, variance_thresh=0.95)

# Step 2: Train on orthogonal features
forest.fit(ortho_features, y)

# Step 3: Get importance of PCA components
ortho_mdi = mean_decrease_impurity(forest, ortho_features.columns)

# Step 4: Map back to original features
pca_importance = feature_pca_analysis(
    ortho_features, 
    feature_importance=ortho_mdi,
    variance_thresh=0.95
)
```

**Strengths:**
✅ **Removes linear substitution effects**
✅ Reveals true feature contribution
✅ Can combine with MDI, MDA, or SFI
✅ Better handling of multicollinear features

**Weaknesses:**
❌ **Loses interpretability** (PC1, PC2 are combinations of features)
❌ Only handles linear correlations (nonlinear substitution remains)
❌ May not be suitable if you need to understand individual features

**When to use for WaveNet:**
- When you have highly correlated feature groups (e.g., multiple moving averages)
- Comparing relative importance within feature families
- Understanding if you have redundant features to prune

---

## Comparison: p-value vs Feature Importance Methods

### Why p-values FAIL in Financial ML:

The notebook demonstrates this with synthetic data:
```python
# Generate data:
# - 5 informative features
# - 5 redundant features (correlated with informative)
# - 10 noise features
trnsX, cont = get_test_data(
    n_features=20, 
    n_informative=5, 
    n_redundant=5, 
    n_samples=1000
)

# p-values from Logistic Regression:
ols = sm.Logit(cont['bin'], trnsX).fit()
```

**Result**: p-values are **unreliable** due to:
1. **Strong distributional assumptions** (often violated in finance)
2. **Multicollinearity sensitivity** (financial features are highly correlated)
3. **In-sample bias** (uses same data for coefficient estimation and significance testing)
4. **Wrong probability** (estimates P(data|H₀) not P(H₀|data))

**MDI/MDA/SFI solve these problems** by:
- No distributional assumptions
- Bootstrap-based variance reduction
- Out-of-sample validation (MDA, SFI)
- Direct prediction impact measurement

---

## Feature Importance Results Interpretation

### From the Notebook (Synthetic Data):

**Test Data Structure:**
- 5 informative features (I0-I4): Truly predictive
- 5 redundant features (R0-R4): Correlated with informative, also predictive
- 10 noise features (N0-N9): Random, no signal

### MDI Results:
```
Top Features:
1. Informative + Redundant features rank high
2. Noise features rank low
3. Importance diluted among correlated features (substitution effect)
```

**Interpretation**: MDI successfully separates signal from noise, but splits credit among correlated features.

### MDA Results:
```
Top Features:
1. Informative features rank highest
2. Redundant features get lower importance (substitution effect)
3. Noise features near zero
```

**Interpretation**: MDA penalizes redundant features more heavily than MDI.

### SFI Results:
```
Top Features:
1. Informative features with standalone power rank high
2. Features requiring interactions rank lower
3. Noise features near zero
```

**Interpretation**: SFI identifies features that work alone, misses those requiring combinations.

### Orthogonal Features Results:
```
Top PCA Components:
1. PC1-PC5 capture most variance
2. Importance more evenly distributed
3. Need to map back to understand original features
```

**Interpretation**: Decorrelation reveals true independent contributions.

---

## Critical Implementation Details for Time-Series

### 1. **PurgedKFold Cross-Validation**

**Standard K-Fold is WRONG for time-series:**
```python
# ❌ WRONG: Standard K-Fold
from sklearn.model_selection import KFold
cv = KFold(n_splits=5)  # Leaks future information!
```

**Correct: Purged K-Fold:**
```python
# ✅ CORRECT: Purged K-Fold
from FinancialMachineLearning.cross_validation.cross_validation import PurgedKFold

cv_gen = PurgedKFold(
    n_splits=5,
    samples_info_sets=cont['t1']  # Barrier exit times
)
```

**Why it matters:**
- Triple barriers create overlapping time periods
- Sample at time `t` depends on prices up to `t1` (exit time)
- Standard K-Fold can put overlapping samples in train/test
- **PurgedKFold removes overlapping samples from validation set**

### 2. **max_features=1 for MDI**

**Why this matters:**
```python
# ❌ Without max_features=1:
forest = RandomForestClassifier()  # default: max_features='sqrt'
# Problem: Tree can choose from multiple features at each split
# Result: "Masking effect" - popular features get all the attention
# Outcome: Some features never get a chance to split

# ✅ With max_features=1:
forest = RandomForestClassifier(max_features=1)
# Benefit: Each split considers only 1 random feature
# Result: All features get equal opportunity
# Outcome: True importance emerges over many trees
```

**For MDA/SFI**: Can use `max_features=3` or more (no masking issue)

### 3. **Sample Weighting Consideration**

The notebook doesn't show it, but you should add:
```python
# Calculate avg_uniqueness (from your barrier events)
avg_uniqueness = calculate_avg_uniqueness(barrier_events)

# Use in RandomForest training
forest.fit(
    X, 
    y, 
    sample_weight=avg_uniqueness.loc[X.index]
)
```

This weights feature importance by sample uniqueness, reducing overlap bias.

---

## Application Strategy for Your WaveNet Model

### Phase 1: Initial Feature Screening (MDI)

**Goal**: Eliminate obvious noise features

```python
# Quick MDI on all features
forest = RandomForestClassifier(
    max_features=1,
    n_estimators=1000,
    oob_score=True,
    random_state=42
)
forest.fit(X_train, y_train)

mdi = mean_decrease_impurity(forest, X_train.columns)

# Keep features with MDI > threshold
threshold = 0.01  # Keep features with >1% importance
important_features = mdi[mdi['mean'] > threshold].index.tolist()

print(f"Reduced from {len(X_train.columns)} to {len(important_features)} features")
```

**Expected outcome**: 
- Remove 30-50% of features (noise)
- Fast iteration
- Prepare for deeper analysis

---

### Phase 2: Out-of-Sample Validation (MDA)

**Goal**: Validate features generalize

```python
# MDA on important features from Phase 1
X_filtered = X_train[important_features]

forest = RandomForestClassifier(
    max_features=3,
    n_estimators=1000,
    random_state=42
)
forest.fit(X_filtered, y_train)

# Use PurgedKFold for time-series
cv_gen = PurgedKFold(n_splits=5, samples_info_sets=barrier_events['t1'])

mda = mean_decrease_accuracy(
    forest,
    X_filtered,
    y_train,
    cv_gen=cv_gen,
    scoring=accuracy_score
)

# Keep features with positive MDA (negative = hurts performance)
oos_features = mda[mda['mean'] > 0].index.tolist()

print(f"OOS validated: {len(oos_features)} features")
```

**Expected outcome**:
- Further reduce by 20-30%
- Only keep features that help out-of-sample
- These are your **WaveNet input features**

---

### Phase 3: Standalone Power Check (SFI)

**Goal**: Identify hero features and interaction-dependent features

```python
# SFI on validated features
forest = RandomForestClassifier(max_features=1, n_estimators=1000)
forest.fit(X_filtered, y_train)

sfi = single_feature_importance(
    forest,
    X_filtered,
    y_train,
    cv_gen=cv_gen,
    scoring=accuracy_score
)

# Categorize features
strong_solo = sfi[sfi['mean'] > 0.55].index.tolist()  # Good alone
needs_interaction = sfi[sfi['mean'] < 0.53].index.tolist()  # Needs context
```

**Expected outcome**:
- Identify features WaveNet can learn easily (strong solo)
- Identify features requiring temporal context (needs interaction)
- WaveNet's strength: Learning temporal interactions!

---

### Phase 4: Handle Multicollinearity (Orthogonal)

**Goal**: Understand feature redundancy

```python
# Orthogonalize and re-analyze
ortho_features = get_orthogonal_features(X_filtered, variance_thresh=0.95)

forest = RandomForestClassifier(max_features=1, n_estimators=1000)
forest.fit(ortho_features, y_train)

ortho_mdi = mean_decrease_impurity(forest, ortho_features.columns)

# Map back to original features
pca_importance = feature_pca_analysis(
    ortho_features,
    feature_importance=ortho_mdi,
    variance_thresh=0.95
)
```

**Expected outcome**:
- Identify redundant feature groups
- Example: If SMA_20, SMA_50, EMA_20 all map to same PC, keep only one
- Reduces WaveNet input dimensionality

---

### Phase 5: Final Feature Set for WaveNet

**Combination Strategy**:

```python
# Combine insights from all methods
final_features = []

# Must-have: High MDA + High SFI (strong standalone + generalizes)
hero_features = set(mda[mda['mean'] > 0.02].index) & set(sfi[sfi['mean'] > 0.55].index)
final_features.extend(hero_features)

# Context features: High MDA + Low SFI (needs interactions, WaveNet's strength)
context_features = set(mda[mda['mean'] > 0.01].index) - set(sfi[sfi['mean'] > 0.53].index)
final_features.extend(context_features)

# Remove redundant (from orthogonal analysis)
# If PC1 includes [SMA_20, SMA_50, EMA_20], keep only highest individual SFI
# ... (manual inspection based on pca_importance)

print(f"Final WaveNet features: {len(final_features)}")
print(final_features)
```

**Expected final set**: 10-20 high-quality features for WaveNet

---

## Specific Recommendations for Your WaveNet

### Current Features (from your codebase):
```python
features = [
    # Price-based
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'BB_upper', 'BB_middle', 'BB_lower',
    
    # Momentum
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
    'MOM_10', 'ROC_10', 'Stoch_K', 'Stoch_D',
    
    # Volatility
    'ATR_14', 'volatility',
    
    # Trend
    'ADX_14', 'CCI_20',
    
    # Volume
    'OBV', 'VWAP', 'Volume_SMA',
    
    # Price structure
    'high_low_pct', 'close_open_pct'
]
```

### Expected Feature Importance Patterns:

**High MDI + High MDA (Keep for WaveNet):**
- `volatility` - Direct barrier input
- `ATR_14` - Volatility measure
- `RSI_14` - Momentum (mean reversion signal)
- `MACD_hist` - Trend changes

**High MDI + Low MDA (Overfitting, Remove):**
- Likely: `BB_upper`, `BB_lower` (derived from SMA + volatility)
- Likely: `MACD_signal` (smoothed MACD)

**Low MDI + High MDA (Keep, Stable):**
- Possibly: `ADX_14` (trend strength)
- Possibly: `OBV` (volume confirmation)

**Low MDI + Low MDA (Remove):**
- Likely: Some of the redundant moving averages
- Possibly: `CCI_20`, `MOM_10` (if correlated with ROC)

### Redundancy Groups (Expect High Correlation):

1. **Moving Averages**: `SMA_20`, `SMA_50`, `EMA_12`, `EMA_26`, `BB_middle`
   - **Action**: Keep 1-2 (different timescales)
   - **Suggestion**: Keep `EMA_12` (short-term) + `SMA_50` (long-term)

2. **Bollinger Bands**: `BB_upper`, `BB_middle`, `BB_lower`
   - **Action**: Redundant with SMA + volatility
   - **Suggestion**: Remove, keep `volatility` instead

3. **MACD Family**: `MACD`, `MACD_signal`, `MACD_hist`
   - **Action**: Keep histogram only (contains signal - EMA difference)
   - **Suggestion**: Keep `MACD_hist`

4. **Momentum Oscillators**: `RSI_14`, `Stoch_K`, `Stoch_D`, `MOM_10`, `ROC_10`
   - **Action**: Keep 2-3 (different behaviors)
   - **Suggestion**: Keep `RSI_14` + `ROC_10`

5. **Volatility Measures**: `ATR_14`, `volatility`
   - **Action**: Both useful (different calculations)
   - **Suggestion**: Keep both

---

## Code Implementation: Run Feature Importance on Your Data

### Step 1: Prepare Your Data
```python
import pandas as pd
from fin_feature_preprocessing import prepare_features, create_dynamic_triple_barriers

# Load your data
df = pd.read_csv('AAPL_2022_2024.csv', index_col=0, parse_dates=True)

# Generate features
df_features = prepare_features(df)

# Create barriers
barriers = create_enhanced_barriers_for_wavenet(df_features)

# Merge
df_merged = df_features.join(barriers, how='inner')

# Define feature columns (exclude labels)
feature_cols = [col for col in df_merged.columns 
                if col not in ['label', 't1', 'exit_return', 'holding_period', 
                               'avg_uniqueness', 'entry_idx', 'exit_idx']]

X = df_merged[feature_cols].dropna()
y = df_merged.loc[X.index, 'label']
t1 = df_merged.loc[X.index, 't1']
```

### Step 2: MDI Analysis
```python
from sklearn.ensemble import RandomForestClassifier
from FinancialMachineLearning.feature_importance.importance import mean_decrease_impurity

forest = RandomForestClassifier(
    criterion='entropy',
    max_features=1,  # Critical for MDI
    n_estimators=1000,
    class_weight='balanced_subsample',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

forest.fit(X, y)
mdi = mean_decrease_impurity(forest, X.columns)

# Visualize
plot_feature_importance(mdi, oob_score=forest.oob_score_)
```

### Step 3: MDA Analysis
```python
from FinancialMachineLearning.cross_validation.cross_validation import PurgedKFold
from FinancialMachineLearning.feature_importance.importance import mean_decrease_accuracy
from sklearn.metrics import accuracy_score

# Purged K-Fold for time-series
cv_gen = PurgedKFold(n_splits=5, samples_info_sets=t1)

forest = RandomForestClassifier(
    criterion='entropy',
    max_features=3,
    n_estimators=1000,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

forest.fit(X, y)

mda = mean_decrease_accuracy(
    forest,
    X,
    y,
    cv_gen=cv_gen,
    scoring=accuracy_score
)

# Visualize
plot_feature_importance(mda)
```

### Step 4: Compare and Select
```python
# Merge MDI and MDA
comparison = pd.DataFrame({
    'MDI': mdi['mean'],
    'MDA': mda['mean'],
    'MDI_std': mdi['std'],
    'MDA_std': mda['std']
})

# Features good in both
strong_features = comparison[
    (comparison['MDI'] > 0.02) & 
    (comparison['MDA'] > 0.01)
].sort_values('MDA', ascending=False)

print("Strong features for WaveNet:")
print(strong_features)
```

---

## Key Takeaways for WaveNet

1. **RandomForest is NOT your model** - it's your feature selector
2. **Use MDA (not MDI) for final selection** - out-of-sample matters
3. **PurgedKFold is mandatory** - time-series leakage destroys validity
4. **Expect 50-70% feature reduction** - most features are noise or redundant
5. **SFI identifies hero features** - WaveNet will learn these easily
6. **Low SFI + High MDA = WaveNet's sweet spot** - temporal interactions
7. **Orthogonal analysis reveals redundancy** - prune correlated feature groups

### Expected WaveNet Feature Set (Prediction):
After running analysis, you'll likely keep:
- `volatility`, `ATR_14` (volatility)
- `RSI_14`, `ROC_10` (momentum)
- `MACD_hist` (trend changes)
- `EMA_12`, `SMA_50` (price trends, different scales)
- `ADX_14` (trend strength)
- `OBV`, `Volume_SMA` (volume)
- `high_low_pct`, `close_open_pct` (intraday structure)
- Plus: `exit_return`, `holding_period`, `avg_uniqueness` (enhanced labels)

**~10-15 features** instead of 25+ → Faster training, better generalization

---

## Next Steps

1. **Run the feature importance analysis** on your AAPL data
2. **Document the results** - which features survive MDI/MDA/SFI
3. **Reduce your feature set** to 10-15 high-quality features
4. **Retrain WaveNet** with selected features
5. **Compare performance** - reduced features often outperform (less overfitting)
6. **Iterate** - add fractional differentiation, entropy features, re-analyze

This is the foundation for **production-grade feature engineering** - not guessing, but systematic validation of what actually predicts out-of-sample.
