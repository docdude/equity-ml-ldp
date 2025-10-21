# Phased Implementation Plan: Advanced López de Prado Methods

**Current Status Analysis: fin_training_ldp.py**

## ✅ Currently Implemented (Phase 0 - COMPLETE)

### 1. Triple Barrier Labeling (Chapter 3)
- ✅ MLFinLab `get_events()` with dynamic volatility thresholds
- ✅ Asymmetric barriers (TP=2σ, SL=1σ)
- ✅ Minimum return filtering (0.5%)
- ✅ Vertical barrier timeout (7 days)
- ✅ 3-class classification (UP/TP, DOWN/SL, TIMEOUT)
- ✅ Proper alignment verification

### 2. Basic Sample Weighting
- ✅ Sklearn `compute_sample_weight` with `class_weight='balanced'`
- ✅ Applied only to direction output
- ✅ Uniform weights for auxiliary targets (volatility, magnitude)

### 3. Auxiliary Targets
- ✅ Realized volatility: std of holding period returns
- ✅ Return magnitude: absolute value of barrier returns
- ✅ NaN filtering after sequence creation

### 4. Data Quality
- ✅ Multi-ticker support (11 tickers)
- ✅ Chronological sorting for proper time-series CV
- ✅ Comprehensive diagnostics (NaN/Inf detection)

### 5. Model Architecture
- ✅ WaveNet-style dilated convolutions
- ✅ Multi-head attention mechanism
- ✅ Multi-output (direction, volatility, magnitude)
- ✅ Focal loss for imbalanced classes

### 6. Strategy Testing
- ✅ PBO analysis with 10 confidence thresholds
- ✅ Meta-labeling strategy (primary signal × confidence)
- ✅ Results: PBO=30.19%, Sharpe=0.86-1.26

---

## 🎯 Phase 1: Sample Uniqueness & Concurrency (HIGH PRIORITY)

**Goal:** Address 211x label redundancy from overlapping barriers

**Rationale:** Most impactful improvement - reduces overfitting from correlated samples

### 1.1 Calculate Sample Uniqueness (2-3 hours)
```python
from FinancialMachineLearning.sample_weights.attribution import get_av_uniqueness_from_triple_barrier

# After events_combined is created
uniqueness_train = get_av_uniqueness_from_triple_barrier(
    events=events_train,  # Subset of events_combined
    close=prices_train,
    num_threads=1
)
```

**Implementation Steps:**
1. Store barrier events per sequence (not just combined at end)
2. Split events_combined into train/val before sequence creation
3. Calculate uniqueness for each split
4. Apply as weight multiplier: `final_weight = class_weight × uniqueness`

**Files to Modify:**
- `fin_training_ldp.py`: Lines 400-415 (store events separately), Lines 616-635 (weight calculation)

**Expected Impact:**
- Down-weight samples in crowded periods (many concurrent positions)
- Up-weight samples in sparse periods (unique information)
- Reduce effective training set size by ~3-5x
- Lower overfitting, better generalization

**Validation:**
- Plot uniqueness distribution (should be 0-1, mean ~0.3-0.5)
- Compare training curves with/without uniqueness weights
- Expect: slower convergence, better validation metrics

### 1.2 Verify Concurrency Handling (1 hour)
```python
# Diagnostic after weight calculation
print(f"Uniqueness statistics:")
print(f"  Mean: {uniqueness_train.mean():.3f}")
print(f"  Median: {uniqueness_train.median():.3f}")
print(f"  Range: [{uniqueness_train.min():.3f}, {uniqueness_train.max():.3f}]")
print(f"  Effective N: {(uniqueness_train ** 2).sum():.0f} / {len(uniqueness_train)}")
```

**Success Criteria:**
- Mean uniqueness 0.3-0.6 (if <0.3, barriers too crowded)
- No uniqueness values = 0 (would zero-out samples)
- Effective N reduced 2-5x from nominal N

---

## 🎯 Phase 2: Return Attribution Weighting (MEDIUM PRIORITY)

**Goal:** Weight samples by information content (magnitude of move)

**Rationale:** Large moves are more informative than small moves

### 2.1 Calculate Return Attribution (1 hour)
```python
# After label calculation
return_attribution_train = np.abs(returns_seq[:split_idx]) / volatility_seq[:split_idx]
return_attribution_train = return_attribution_train / return_attribution_train.mean()  # Normalize
```

**Implementation:**
1. Calculate `|return| / expected_volatility`
2. Normalize to mean=1 (so average weight unchanged)
3. Clip extreme values (e.g., [0.1, 10.0])
4. Multiply with existing weights

**Expected Impact:**
- Up-weight large breakout moves (high information)
- Down-weight small noisy moves near barriers
- Model focuses on significant events

### 2.2 Combine Weight Components (30 mins)
```python
# Final weight = class_balance × uniqueness × return_attribution
final_weights_train = (
    sample_weights_train *      # Class balance
    uniqueness_train *           # Concurrency adjustment
    return_attribution_train     # Information content
)
```

**Validation:**
- Plot weight distribution vs return magnitude
- Should show positive correlation
- No extreme outliers (clip if needed)

---

## 🎯 Phase 3: Purged K-Fold Cross-Validation (MEDIUM PRIORITY)

**Goal:** Replace train/val split with proper time-series CV

**Rationale:** Current 80/20 split is naive - doesn't account for information leakage

### 3.1 Implement Purged CV (3-4 hours)
```python
from FinancialMachineLearning.cross_validation.purged import PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    samples_info_sets=events_combined['t1'],  # Barrier end times
    pct_embargo=0.01  # 1% embargo between folds
)

for train_idx, test_idx in cv.split(X_seq):
    # Train on train_idx, validate on test_idx
    # Purging removes samples that overlap test period
```

**Key Features:**
- Purging: Remove train samples that end during test period
- Embargo: Add buffer period between train/test
- Respects time-series structure

**Implementation Approach:**
- Option A: Use CV for hyperparameter tuning only
- Option B: Ensemble predictions from all folds
- Recommend: Start with Option A (simpler)

**Expected Impact:**
- More robust validation metrics
- Better estimate of true OOS performance
- May show current model is overfit

---

## 🎯 Phase 4: Fractional Differentiation (LOWER PRIORITY)

**Goal:** Make features stationary while preserving memory

**Rationale:** Financial time series are non-stationary, but we want to preserve trends

### 4.1 Apply to Price-Based Features (2-3 hours)
```python
from FinancialMachineLearning.features.fracdiff import frac_diff_ffd

# For price-based features (returns, price ratios)
for col in price_features:
    X[f'{col}_fracdiff'] = frac_diff_ffd(X[col], d=0.4, thres=0.01)
```

**What to Differentiate:**
- ✅ Price returns (already differenced, but could use frac_diff)
- ✅ Moving average ratios (price/MA)
- ❌ Already stationary indicators (RSI, Stochastic)
- ❌ Volume ratios (different stationarity properties)

**Parameter Selection:**
- d=0.4-0.6 typical range
- Test stationarity with ADF test before/after
- Balance: too low (non-stationary), too high (lost memory)

**Expected Impact:**
- Potentially better feature quality
- May improve long-term predictions
- Not critical if features already well-engineered

---

## 🎯 Phase 5: Sequential Bootstrap (ADVANCED)

**Goal:** Proper resampling that respects sample overlap

**Rationale:** Standard bootstrap assumes independence - doesn't hold with barriers

### 5.1 Implement for Ensemble Training (4-5 hours)
```python
from FinancialMachineLearning.cross_validation.sequential import seq_bootstrap

# Generate bootstrap samples
bootstrap_indices = seq_bootstrap(
    ind_matrix=concurrency_matrix,  # Which samples overlap
    sample_length=len(X_train)
)

# Train ensemble
for i, indices in enumerate(bootstrap_indices):
    model_i = train_model(X_train[indices], y_train[indices])
    models.append(model_i)
```

**Use Cases:**
- Train ensemble of models with proper diversity
- Generate confidence intervals for predictions
- Estimate model uncertainty

**Complexity:**
- Requires building concurrency matrix (memory intensive)
- Need to train multiple models (compute intensive)
- Best for smaller datasets or subset training

---

## 📊 Recommended Implementation Order

### Sprint 1: Foundation (Week 1)
- ✅ Phase 1.1: Sample uniqueness weighting (DONE)
- ✅ Phase 1.2: Concurrency diagnostics (DONE)
- Retrain model and compare validation metrics

### Sprint 2: Enhancement (Week 2)
- Phase 2.1: Return attribution weighting
- Phase 2.2: Combined weighting scheme
- Retrain and measure impact on PBO

### Sprint 3: Validation (Week 3)
- Phase 3.1: Implement purged K-fold
- Run 5-fold CV for robust metrics
- Compare CV results with current train/val split

### Sprint 4: Advanced Features (Week 4)
- Phase 4.1: Fractional differentiation (if needed)
- Phase 5.1: Sequential bootstrap (if doing ensemble)

---

## 📈 Success Metrics by Phase

| Phase | Metric | Current | Target | Validation |
|-------|--------|---------|--------|------------|
| **Phase 1** | Uniqueness weighting | No weighting | Mean ~0.4 | Distribution plot |
| | Effective N | 20,559 samples | 6,000-10,000 | Calculate (w²).sum() |
| | Training speed | Baseline | Slower convergence | Epoch time |
| | Val AUC | ~0.50 | 0.52-0.55 | Model metrics |
| **Phase 2** | Return weighting | Uniform | Weighted by |return| | Weight-return scatter |
| | Large move focus | Equal weight | 2-3x weight | Weight percentiles |
| | Val AUC | From Phase 1 | +0.01-0.02 | Model metrics |
| **Phase 3** | PBO score | 30.19% | <25% | PBO analysis |
| | CV std | N/A | <0.05 | Across folds |
| | OOS loss prob | 11.90% | <8% | PBO output |
| **Phase 4** | Feature stationarity | Mixed | ADF p<0.05 | Statistical test |
| | Feature correlation | High | Lower | Correlation matrix |
| **Phase 5** | Ensemble diversity | N/A | Low correlation | Model predictions |
| | Prediction uncertainty | N/A | Calibrated | Reliability diagram |

---

## 🔧 Implementation Templates

### Template 1: Uniqueness Weighting
```python
# After events_combined is created, before sequence creation
from FinancialMachineLearning.sample_weights.attribution import get_av_uniqueness_from_triple_barrier

print("\n3.7. CALCULATING SAMPLE UNIQUENESS")
print("-"*40)

# Need to reconstruct close prices aligned with events
close_series = pd.Series(index=events_combined.index)
for idx in events_combined.index:
    close_series[idx] = df.loc[idx, 'Close']  # Need to track which df this came from

uniqueness = get_av_uniqueness_from_triple_barrier(
    events=events_combined,
    close=close_series,
    num_threads=1
)

print(f"✅ Uniqueness calculated: {len(uniqueness)} samples")
print(f"   Mean: {uniqueness.mean():.3f}")
print(f"   Median: {uniqueness.median():.3f}")
print(f"   Effective N: {(uniqueness ** 2).sum():.0f} (from {len(uniqueness)} nominal)")

# Store for later sequence creation
all_uniqueness.append(uniqueness)
```

### Template 2: Combined Weighting
```python
# After Phase 1 uniqueness and Phase 2 return attribution
print("\n3.8. COMBINING WEIGHT COMPONENTS")
print("-"*40)

# Normalize each component to mean=1
class_weights_norm = sample_weights_train / sample_weights_train.mean()
uniqueness_norm = uniqueness_train / uniqueness_train.mean()
attribution_norm = return_attribution_train / return_attribution_train.mean()

# Combine multiplicatively
final_weights_train = class_weights_norm * uniqueness_norm * attribution_norm

# Renormalize to sum = N (Keras expectation)
final_weights_train = final_weights_train * len(final_weights_train) / final_weights_train.sum()

print(f"Weight statistics:")
print(f"  Class balance contribution: {class_weights_norm.std():.3f}")
print(f"  Uniqueness contribution: {uniqueness_norm.std():.3f}")
print(f"  Attribution contribution: {attribution_norm.std():.3f}")
print(f"  Final weight range: [{final_weights_train.min():.3f}, {final_weights_train.max():.3f}]")
```

### Template 3: Purged K-Fold
```python
from FinancialMachineLearning.cross_validation.purged import PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    samples_info_sets=events_combined['t1'],
    pct_embargo=0.01
)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X_seq)):
    print(f"\nFold {fold+1}/5:")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Train model
    # ... (same training code)
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val)
    cv_scores.append(val_metrics)
    
print(f"\nCV Results:")
print(f"  Mean AUC: {np.mean([s['auc'] for s in cv_scores]):.4f}")
print(f"  Std AUC: {np.std([s['auc'] for s in cv_scores]):.4f}")
```

---

## 🚨 Critical Considerations

### Data Alignment Challenges
**Problem:** Events are per-ticker, but sequences mix tickers
**Solution:** Track ticker ID through sequence pipeline, calculate uniqueness per ticker

### Memory Constraints
**Problem:** Concurrency matrices are N×N (can be huge)
**Solution:** 
- Use sparse matrix representation
- Calculate uniqueness in chunks
- Consider downsampling for uniqueness calculation

### Computational Cost
**Problem:** Uniqueness calculation is O(N²) worst case
**Solution:**
- Use MLFinLab's optimized numba implementation
- Parallel processing with `num_threads` parameter
- Cache results between experiments

### Weight Clipping
**Problem:** Extreme weights destabilize training
**Solution:**
```python
# Clip final weights to reasonable range
final_weights = np.clip(final_weights, 
                        final_weights.mean() * 0.1,  # Min = 10% of mean
                        final_weights.mean() * 10.0)  # Max = 10× mean
```

---

## 📝 Documentation Requirements

For each phase:
1. **Before:** Save baseline metrics (accuracy, AUC, PBO, Sharpe)
2. **During:** Log weight statistics, training curves
3. **After:** Compare metrics, document improvements
4. **Decision:** Keep if improvement > 2%, otherwise rollback

Create comparison reports:
- `docs/PHASE1_UNIQUENESS_RESULTS.md`
- `docs/PHASE2_ATTRIBUTION_RESULTS.md`
- etc.

---

## 🎓 Learning Resources

- **Chapter 4 (Sample Weights):** Pages 45-67 in Advances in Financial ML
- **MLFinLab Documentation:** https://mlfinlab.readthedocs.io/
- **Sample Weights Module:** `FinancialMachineLearning/sample_weights/`
- **Cross-Validation Module:** `FinancialMachineLearning/cross_validation/`

---

## ✅ Acceptance Criteria

**Phase 1 Complete When:**
- [ ] Uniqueness calculated for all samples
- [ ] Mean uniqueness 0.3-0.6
- [ ] Weights applied to training
- [ ] Training completes without errors
- [ ] Validation AUC improves OR training more stable
- [ ] Documentation updated

**Phase 2 Complete When:**
- [ ] Return attribution calculated
- [ ] Combined with uniqueness weights
- [ ] Positive correlation between weight and |return|
- [ ] Model focuses on large moves (diagnostic plot)
- [ ] PBO improves by 2-5%

**Phase 3 Complete When:**
- [ ] Purged K-Fold implemented
- [ ] 5-fold CV runs successfully
- [ ] CV std < 0.05 on AUC
- [ ] Documentation of OOS performance

---

**Next Immediate Action:** Start Phase 1.1 - Calculate sample uniqueness for training data
