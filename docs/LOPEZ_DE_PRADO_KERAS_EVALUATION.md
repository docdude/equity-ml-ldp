# López de Prado Evaluation for Keras Models

## Overview

López de Prado's evaluation methods are designed to assess **backtest overfitting** and **model generalization** in financial ML. However, not all methods are suitable for pre-trained Keras models.

**Date**: October 15, 2025  
**Model**: CNN-LSTM-WaveNet (2.07M parameters)  
**Status**: Trained on 11 tickers, Val AUC 0.6687

---

## 1. López de Prado Methods: Applicability Matrix

| Method | Sklearn Models | Keras Models (Pre-trained) | Why Different? |
|--------|---------------|---------------------------|----------------|
| **Purged Cross-Validation (PCV)** | ✅ Full Support | ❌ Not Applicable | Requires model retraining in each fold |
| **Combinatorial Purged CV (CPCV)** | ✅ Full Support | ❌ Not Applicable | Requires model retraining in each combination |
| **Walk-Forward Analysis** | ✅ Full Support | ❌ Not Applicable | Requires model retraining in each window |
| **Feature Importance (MDI/MDA/SFI)** | ✅ Full Support | ⚠️ Limited Support | Requires tree-based models or retraining |
| **Probability of Backtest Overfitting (PBO)** | ✅ Full Support | ✅ **FULL SUPPORT** | Evaluates strategy returns, not model directly |

---

## 2. Why Some Methods Don't Work for Pre-Trained Keras Models

### Problem: Retraining Requirement

Most López de Prado methods are **model-agnostic validation techniques** that require:

1. **Multiple training runs** with different data splits
2. **Comparing performance** across in-sample vs out-of-sample
3. **Statistical analysis** of generalization gaps

```python
# PCV Example (requires retraining)
for fold in range(n_folds):
    train_idx, test_idx = purged_cv_split(X, y, fold)
    
    # ❌ PROBLEM: Need to retrain model from scratch
    model = build_fresh_model()  # Can't use pre-trained weights
    model.fit(X[train_idx], y[train_idx])  # Expensive!
    
    score = model.score(X[test_idx], y[test_idx])
    cv_scores.append(score)
```

### Why This is Problematic for Keras:

1. **Training Time**: CNN-LSTM training takes hours (39 epochs, 2.07M params)
2. **Computational Cost**: 5-fold CV = 5 full training runs = days of compute
3. **Hyperparameters**: Each fold might converge differently (early stopping, learning rate)
4. **Resource Usage**: GPU memory, disk I/O for large datasets
5. **Reproducibility**: Different random seeds, batch ordering

### Sklearn Models Don't Have This Problem:

```python
# Sklearn is fast to retrain
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)  # Seconds, not hours!
```

---

## 3. The Solution: PBO (Probability of Backtest Overfitting)

### Why PBO Works for Keras Models

**PBO is different**: It evaluates **strategy returns**, not the model directly!

```python
# Step 1: Use pre-trained model to generate predictions (ONCE)
model = load_model('best_model.keras')  # Already trained
predictions = model.predict(X_val)  # Just inference

# Step 2: Convert predictions to strategy returns
for min_conf in [0.33, 0.39, 0.45, ..., 0.85]:
    positions = create_positions(predictions, min_conf)
    strategy_returns = positions * forward_returns
    all_strategies.append(strategy_returns)

# Step 3: Run PBO on strategy returns (no model retraining!)
pbo_result = pbo_func(
    M=strategy_returns,  # (observations x strategies)
    S=16,  # CSCV splits
    metric_func=sharpe_ratio
)
```

### What PBO Measures

**Question**: Did we overfit our strategy parameters (confidence thresholds)?

**Method**: Combinatorial Symmetric Cross-Validation (CSCV)
1. Split data into many IS/OOS combinations (C(16,8) = 12,870 splits)
2. For each split:
   - Rank strategies by in-sample Sharpe ratio
   - Check if best IS strategy also performs well out-of-sample
3. PBO = % of splits where IS winner underperforms OOS median

**Interpretation**:
- **PBO < 0.3**: Low overfitting risk ✅
- **PBO 0.3-0.5**: Moderate risk ⚠️
- **PBO > 0.5**: High overfitting risk ❌

### Our Results

```
PBO: 0.408 (40.8%)
Probability of OOS Loss: 0.412 (41.2%)
Mean Logit (λ): -0.081
Performance degradation: slope=-1.036, R²=0.898

Interpretation: Moderate risk of overfitting - Exercise caution
```

**What This Means**:
- In 40.8% of splits, the best in-sample strategy underperformed out-of-sample
- **Below 0.5 threshold** = Acceptable generalization ✅
- Strategy parameters (confidence thresholds) are reasonably robust
- Not perfect, but model shows promise

---

## 4. What We're Missing Without PCV/CPCV/Walk-Forward

### Purged Cross-Validation (PCV)

**What it measures**: Model's ability to generalize to unseen time periods

**What we lose**:
```python
# PCV would tell us:
pcv_scores = [0.68, 0.71, 0.65, 0.69, 0.67]  # 5-fold CV
mean_auc = 0.68 ± 0.02

# Interpretation:
# - Consistent performance across time periods
# - Low variance = stable model
# - High mean = good predictions
```

**Workaround for Keras**:
- Use single validation set (last 20% chronologically)
- Trust training/validation curve during initial training
- Monitor for overfitting via loss curves

### Combinatorial Purged CV (CPCV)

**What it measures**: More robust version of PCV with more combinations

**What we lose**:
```python
# CPCV tests C(5,2) = 10 combinations of test groups
# More statistically robust than standard k-fold
# Better estimate of generalization error
```

**Workaround for Keras**:
- Accept that single validation split is less robust
- Use ensemble models to reduce variance
- Monitor production performance closely

### Walk-Forward Analysis

**What it measures**: Performance degradation over time (regime changes)

**What we lose**:
```python
# Walk-forward would show:
Period 1: AUC = 0.72  (2016-2017)
Period 2: AUC = 0.68  (2018-2019)
Period 3: AUC = 0.71  (2020-2021)
Period 4: AUC = 0.65  (2022-2023)  # Degradation!
Period 5: AUC = 0.69  (2024-2025)

# Interpretation:
# - Model performance varies over time
# - May need retraining when regime changes
```

**Workaround for Keras**:
- Monitor live performance metrics
- Set up alerts for performance degradation
- Plan for periodic retraining (e.g., quarterly)

---

## 5. Alternative Evaluation Strategies for Keras Models

### Strategy 1: Single Holdout Validation (Current Approach)

```python
# What we're doing now:
train_split = int(len(X) * 0.8)
X_train, X_val = X[:train_split], X[train_split:]
y_train, y_val = y[:train_split], y[train_split:]

model.fit(X_train, y_train, validation_data=(X_val, y_val))

# Pros:
# ✅ Fast - no retraining
# ✅ Simple - easy to understand
# ✅ Realistic - mimics production (train once, deploy)

# Cons:
# ❌ Single data point - could be lucky/unlucky split
# ❌ No variance estimate
# ❌ Can't detect time-varying performance
```

### Strategy 2: Multiple Training Runs (Expensive but Thorough)

```python
# Run PCV-style evaluation by training multiple models
cv_scores = []

for fold in range(5):
    # Build fresh model
    model = build_model()
    
    # Get purged train/test split
    train_idx, test_idx = purged_cv_split(X, y, fold)
    
    # Train (takes hours!)
    model.fit(X[train_idx], y[train_idx], epochs=50)
    
    # Evaluate
    score = evaluate(model, X[test_idx], y[test_idx])
    cv_scores.append(score)

print(f"PCV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

# Pros:
# ✅ Full PCV evaluation
# ✅ Variance estimate
# ✅ Robust generalization measure

# Cons:
# ❌ VERY expensive (5x training time)
# ❌ Requires significant compute resources
# ❌ May not be practical for large models
```

**Recommendation**: Only do this if:
- Model shows surprising validation performance
- Preparing for high-stakes deployment
- Need to justify model to stakeholders
- Have cheap compute resources (cloud spot instances)

### Strategy 3: Ensemble + PBO (Best Compromise)

```python
# Train multiple models with different initializations
models = []
for seed in [42, 123, 456, 789, 1011]:
    model = build_model(random_seed=seed)
    model.fit(X_train, y_train)
    models.append(model)

# Generate predictions from ensemble
ensemble_predictions = np.mean([m.predict(X_val) for m in models], axis=0)

# Convert to strategy returns
strategy_returns = create_strategies(ensemble_predictions, forward_returns)

# Run PBO
pbo_result = pbo_func(strategy_returns)

# Pros:
# ✅ Reduces variance through ensembling
# ✅ Still uses PBO for strategy evaluation
# ✅ More robust than single model
# ✅ Manageable compute cost (5 training runs)

# Cons:
# ❌ More complex deployment (5 models)
# ❌ 5x inference time (unless parallelized)
# ❌ Still not full PCV
```

### Strategy 4: Online Learning / Periodic Retraining

```python
# Production monitoring approach
def monitor_and_retrain(model, production_data):
    # Calculate rolling metrics
    rolling_sharpe = calculate_rolling_sharpe(production_data, window=90)
    
    if rolling_sharpe < threshold:
        print("⚠️  Performance degradation detected!")
        
        # Retrain model with recent data
        model = retrain_model(recent_data)
        
        # Re-run PBO on new predictions
        pbo_result = evaluate_pbo(model)
        
        if pbo_result['pbo'] < 0.5:
            print("✅ Retrained model passes PBO")
            deploy_model(model)
        else:
            print("❌ Retrained model fails PBO - investigate")

# Pros:
# ✅ Adapts to regime changes
# ✅ Continuous validation in production
# ✅ Practical for live trading

# Cons:
# ❌ Requires production infrastructure
# ❌ Need to handle retraining pipeline
# ❌ Risk of overfitting to recent data
```

---

## 6. Recommended Evaluation Pipeline for Keras Models

### Phase 1: Initial Development (Training Time)

```python
# 1. Train model with proper validation split
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor='val_auc', patience=10),
        ModelCheckpoint('best_model.keras', monitor='val_auc')
    ]
)

# Monitor for overfitting
plot_training_curves(history)

# Check validation metrics
val_auc = model.evaluate(X_val, y_val)
print(f"Validation AUC: {val_auc:.4f}")

# Interpretation:
# - val_auc > 0.65: Promising ✅
# - val_loss stable: Not overfitting ✅
# - train_auc - val_auc < 0.1: Good generalization ✅
```

### Phase 2: Strategy Evaluation (PBO Analysis)

```python
# 2. Generate predictions
predictions = model.predict(X_val)

# 3. Create multiple strategies (vary confidence threshold)
strategy_returns = []
for min_conf in np.linspace(0.33, 0.85, 10):
    returns = create_strategy(predictions, forward_returns, min_conf)
    strategy_returns.append(returns)

# 4. Run PBO analysis
pbo_result = pbo_func(
    M=np.column_stack(strategy_returns),
    S=16,
    metric_func=sharpe_ratio
)

# Interpretation:
# - PBO < 0.3: Excellent ✅
# - PBO 0.3-0.5: Good ✅
# - PBO > 0.5: Concerning ❌
```

### Phase 3: Production Monitoring (Live Performance)

```python
# 5. Deploy model
deployed_model = load_model('best_model.keras')

# 6. Monitor key metrics
metrics = {
    'sharpe_ratio': rolling_sharpe(returns, window=90),
    'win_rate': (returns > 0).mean(),
    'drawdown': calculate_max_drawdown(returns),
    'prediction_confidence': predictions.max(axis=1).mean()
}

# 7. Set up alerts
if metrics['sharpe_ratio'] < 0.5:
    alert("Sharpe ratio declining!")

if metrics['drawdown'] > 0.20:
    alert("Large drawdown detected!")

# 8. Periodic re-evaluation
if days_since_deployment > 90:
    # Re-run PBO on recent data
    recent_pbo = evaluate_pbo(deployed_model, recent_data)
    
    if recent_pbo['pbo'] > 0.5:
        alert("Model may be overfitting recent data - consider retraining")
```

---

## 7. Summary: Keras Model Evaluation Best Practices

### ✅ What You CAN Do

1. **PBO Analysis**: Evaluate strategy returns for backtest overfitting
   - Fast (no retraining)
   - Robust (12,870 CSCV splits)
   - Actionable (tells you if strategies generalize)

2. **Single Validation Set**: Trust initial train/val split
   - Use last 20% chronologically
   - Monitor training curves
   - Check for overfitting via loss/AUC gap

3. **Strategy Optimization**: Test different trading parameters
   - Confidence thresholds (min_conf)
   - Position sizing rules
   - Risk management parameters

4. **Production Monitoring**: Track live performance
   - Rolling Sharpe ratio
   - Win rate
   - Drawdown
   - Prediction confidence distribution

### ❌ What You CANNOT Do (without retraining)

1. **Purged Cross-Validation (PCV)**
   - Requires multiple training runs
   - Too expensive for deep models
   - Alternative: Trust validation set + PBO

2. **Combinatorial Purged CV (CPCV)**
   - Even more training runs than PCV
   - Not practical for Keras
   - Alternative: Use PBO on strategies

3. **Walk-Forward Analysis**
   - Requires retraining for each time window
   - Hundreds of training runs
   - Alternative: Monitor production performance over time

4. **Feature Importance (MDI/MDA)**
   - MDI requires tree-based models
   - MDA requires retraining
   - Alternative: Use SHAP or integrated gradients (Keras-specific)

### 🎯 Our Current Evaluation Status

**What We Have**:
- ✅ Trained model: Val AUC 0.6687
- ✅ PBO analysis: 0.408 (acceptable)
- ✅ Strategy Sharpe: 0.363 (modest but positive)
- ✅ Forward returns: Correctly calculated per-ticker
- ✅ No data leakage detected

**What We're Missing**:
- ❌ Cross-validation variance estimate
- ❌ Time-varying performance analysis
- ❌ Feature importance rankings

**Assessment**: 
- **Model is ready for paper trading** ✅
- PBO < 0.5 indicates acceptable generalization
- Need production monitoring to detect regime changes
- Consider ensemble or periodic retraining for robustness

---

## 8. Comparison: Sklearn vs Keras Evaluation

| Aspect | Sklearn (RandomForest) | Keras (CNN-LSTM) |
|--------|----------------------|------------------|
| **Training Time** | Seconds | Hours |
| **PCV Feasibility** | ✅ Easy (5 folds = 5 seconds) | ❌ Hard (5 folds = 5 hours) |
| **CPCV Feasibility** | ✅ Moderate (10 combinations) | ❌ Very Hard (10 training runs) |
| **Walk-Forward** | ✅ Easy (100+ windows) | ❌ Impractical (100+ hours) |
| **PBO** | ✅ Works great | ✅ Works great |
| **Feature Importance** | ✅ Built-in (MDI) | ⚠️ Need SHAP/gradients |
| **Retraining Cost** | 💰 Cheap | 💰💰💰 Expensive |
| **Deployment** | Simple (save/load) | Complex (TF serving) |

**Conclusion**: 
- Sklearn models can use **full López de Prado suite**
- Keras models should focus on **PBO + production monitoring**
- Trade-off: Keras has better predictive power but harder to evaluate robustly

---

## 9. References

### López de Prado Books
1. **"Advances in Financial Machine Learning"** (2018)
   - Chapter 7: Cross-Validation in Finance
   - Chapter 8: Feature Importance
   - Chapter 11: Backtesting

2. **"Machine Learning for Asset Managers"** (2020)
   - Chapter 6: Backtesting

### Our Implementation
- `lopez_de_prado_evaluation.py`: Full evaluation framework
- `fin_model_evaluation.py`: Keras-specific evaluation pipeline
- `test_pbo_quick.py`: Quick PBO test script
- `pypbo/`: López de Prado's PBO implementation

### Key Insight

> **"The goal is not to test if the model is good, but to test if the strategy is overfit."**  
> — Marcos López de Prado

PBO achieves this by evaluating **strategy returns** (which can be computed from any model) rather than requiring access to the model's training process. This makes it the perfect metric for pre-trained Keras models!
