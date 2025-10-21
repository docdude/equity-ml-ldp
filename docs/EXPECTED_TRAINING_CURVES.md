# Expected Training Curves for Financial CNN-LSTM Model

## 📊 What to Expect in TensorBoard

Your model has **3 outputs**, so you'll see curves for each:
1. **Direction** (classification) - Primary metric
2. **Volatility** (regression) - Auxiliary
3. **Magnitude** (regression) - Auxiliary

---

## 🎯 **1. TOTAL LOSS (Combined)**

### Healthy Training Pattern:

```
Loss
3.0 |                    
2.5 | Train ●●●●●●●●●●●●●●●●●●●●●●●●●●
2.0 |       ●●●●            ●●●●●●●●●●●
1.5 |           ●●●●●●●●                
1.0 |                   ●●●●●●●●●●●●●●●●
0.5 |                                   ●●●●
0.0 |___________________________________________
    0    10   20   30   40   50   60   70   80
                        Epoch

2.5 |                    
2.0 | Val   ○○○○○○○○○○○○○○○○○○○○○○○○
1.5 |       ○○○○        ○○○○○○○○○○○○○
1.0 |           ○○○○○○○            ○○○○○
0.5 |                   ○○○○○○○○○○      ○○
0.0 |___________________________________________
    0    10   20   30   40   50   60   70   80
```

**What this shows:**
- **Phase 1 (0-20 epochs):** Rapid decrease - model learning basic patterns
- **Phase 2 (20-50 epochs):** Slower decrease - refining predictions
- **Phase 3 (50+ epochs):** Plateau or slight increase - approaching convergence
- **Gap between train/val:** Should be < 30% (otherwise overfitting)

**Expected Final Values:**
- Train loss: `0.4 - 0.6`
- Val loss: `0.5 - 0.8`
- **Gap:** Val should be 10-40% higher than train (normal for time series)

---

## 📉 **2. DIRECTION LOSS (Categorical Crossentropy)**

### Ideal Pattern:

```
Direction Loss
1.2 |                    
1.1 | Train ●●●●●●●●●●●●●●●●
1.0 |       ●●●●          ●●●●●●●●●●●●
0.9 |           ●●●●●●●●              ●●●●
0.8 |                   ●●●●●●●●●●        ●●
0.7 |                                      ●●●
    |___________________________________________
    0    10   20   30   40   50   60   70   80

1.2 |                    
1.1 | Val   ○○○○○○○○○○○○○○○
1.0 |       ○○○○      ○○○○○○○○○○○○○
0.9 |           ○○○○○              ○○○○○○
0.8 |                   ○○○○○○○○        ○○○
0.7 |___________________________________________
    0    10   20   30   40   50   60   70   80
```

**Interpretation:**
- **Perfect predictions:** Loss = 0.0 (impossible in finance)
- **Random guessing (3 classes):** Loss ≈ 1.099 (`-log(1/3)`)
- **Good model:** Loss = 0.7 - 0.9
- **Excellent model:** Loss < 0.7

**Your target:** Get below **0.85** for validation

---

## 📈 **3. DIRECTION ACCURACY**

### Expected Pattern:

```
Accuracy
100%|                    
 90%|                          ●●●●●●
 80%|                    ●●●●●●
 70%|              ●●●●●●
 60%|        ●●●●●●                  Train ●
 50%|  ●●●●●●                        Val   ○
 40%|        ○○○○○○○○○○○○○○○○○○○○
 33%|○○○○                            (random baseline)
    |___________________________________________
    0    10   20   30   40   50   60   70   80
```

**Baseline:** 33.3% (random guessing for 3 classes)

**Expected Final Accuracy:**
- Train: `60-75%`
- Val: `45-60%`

**Warning Signs:**
- Val stuck at 33%: Model not learning (check LR, features)
- Train >> Val (e.g., 80% vs 40%): Overfitting
- Both low (<40%): Underfitting or bad features

---

## 🎯 **4. DIRECTION AUC (PRIMARY METRIC)**

### Gold Standard Pattern:

```
AUC
1.0 |                                  ●●●
0.9 |                            ●●●●●●
0.8 |                      ●●●●●●
0.7 |                ●●●●●●              Train ●
0.6 |          ●●●●●●                    Val   ○
0.5 |    ●●●●●●
0.4 |  ●●                  ○○○○○○○○○○○○○
0.3 |○○○                   (random baseline = 0.5)
    |___________________________________________
    0    10   20   30   40   50   60   70   80
```

**THIS IS YOUR NORTH STAR METRIC!** 🌟

**Interpretation:**
- **0.50:** Random (coin flip)
- **0.50-0.60:** Weak signal (barely better than random)
- **0.60-0.70:** Moderate signal (tradeable with care)
- **0.70-0.80:** Strong signal (good model!)
- **0.80+:** Excellent (rare in finance, check for leakage!)

**Your Target:** Val AUC > **0.65**

**Realistic Expectations for Financial Data:**
- Train AUC: `0.72 - 0.80`
- Val AUC: `0.62 - 0.72`
- Gap: 0.05 - 0.15 is normal

---

## 📊 **5. VOLATILITY LOSS (MSE)**

### Expected Pattern:

```
Volatility MSE
0.010|                    
0.008| Train ●●●●●●●●●●
0.006|       ●●●●      ●●●●●●●●●●●●●●●
0.004|           ●●●●●●                ●●●●●
0.002|                                      ●●●
     |___________________________________________
     0    10   20   30   40   50   60   70   80

0.012|                    
0.010| Val   ○○○○○○○○○○○○○○○
0.008|       ○○○○    ○○○○○○○○○○○○○○○
0.006|           ○○○○              ○○○○○○○
     |___________________________________________
     0    10   20   30   40   50   60   70   80
```

**Interpretation:**
- Lower is better (squared error)
- Volatility typically 0.01 - 0.03 (1% - 3% daily)
- MSE should converge to ~0.002 - 0.008

**Expected Final Values:**
- Train MSE: `0.002 - 0.005`
- Val MSE: `0.003 - 0.008`

**What to look for:**
- Both decrease together: Good ✅
- Val plateaus while train decreases: Overfitting ⚠️
- Neither decreases: Wrong loss weight or bad features ❌

---

## 📊 **6. VOLATILITY MAE (Mean Absolute Error)**

### Expected Pattern:

```
Volatility MAE
0.08 |                    
0.07 | Train ●●●●●●●●●●●●
0.06 |       ●●●●      ●●●●●●●●●●●●●●
0.05 |           ●●●●●●              ●●●●
0.04 |                   ●●●●●●●●●●      ●●
0.03 |                                      ●●●
     |___________________________________________
     0    10   20   30   40   50   60   70   80
```

**More interpretable than MSE:**
- MAE = 0.02 means predictions are off by ±2% on average
- Expected: `0.03 - 0.05` (3-5% error)

**Target:**
- Train MAE: `0.025 - 0.040`
- Val MAE: `0.030 - 0.050`

---

## 📊 **7. MAGNITUDE LOSS (Huber)**

### Expected Pattern:

```
Magnitude Loss (Huber)
0.15 |                    
0.12 | Train ●●●●●●●●●●●●●●●●
0.09 |       ●●●●      ●●●●●●●●●●●●●●●
0.06 |           ●●●●●●              ●●●●●
0.03 |                   ●●●●●●●●●●      ●●●
     |___________________________________________
     0    10   20   30   40   50   60   70   80

0.18 |                    
0.15 | Val   ○○○○○○○○○○○○○○○○
0.12 |       ○○○○    ○○○○○○○○○○○○○○○
0.09 |           ○○○○              ○○○○○○
     |___________________________________________
     0    10   20   30   40   50   60   70   80
```

**Huber Loss combines MAE + MSE:**
- Robust to outliers
- Small errors: acts like MSE (penalizes quadratically)
- Large errors: acts like MAE (linear penalty)

**Expected Final Values:**
- Train: `0.015 - 0.030`
- Val: `0.020 - 0.040`

---

## 📊 **8. MAGNITUDE MAE**

### Expected Pattern:

```
Magnitude MAE
0.08 |                    
0.06 | Train ●●●●●●●●●●●●●●●●●●●●●●●●
0.04 |       ●●●●  ●●●●●●●●●●●●●●●●●●●●
0.02 |           ●●                  ●●●●●●●
     |___________________________________________
     0    10   20   30   40   50   60   70   80

0.10 |                    
0.08 | Val   ○○○○○○○○○○○○○○○○○○○○○○
0.06 |       ○○○○  ○○○○○○○○○○○○○○○○○○
0.04 |           ○○              ○○○○○○○○
     |___________________________________________
     0    10   20   30   40   50   60   70   80
```

**Interpretation:**
- Predicting absolute magnitude of price moves
- MAE = 0.03 means off by ±3% on magnitude
- Financial moves are hard to predict exactly!

**Expected:**
- Train MAE: `0.020 - 0.035`
- Val MAE: `0.025 - 0.045`

---

## 🚨 **WARNING PATTERNS TO WATCH FOR**

### 1. **Overfitting** (Most Common)

```
Loss
2.0 |
1.5 | Train ●●●●●●●●●●●●●●●●●
1.0 |       ●●●●      ●●●●●●●●●●●●●●●
0.5 |           ●●●●●●              ●●●●●
0.0 |___________________________________________

2.0 | Val   ○○○○○
1.5 |           ○○
1.0 |             ○○○○              ← Gap widening!
0.5 |                 ○○○○○○○○○○○○○ ← Getting worse!
    |___________________________________________
```

**Fix:**
- Reduce learning rate (0.001 → 0.0003)
- Increase dropout (0.3 → 0.4)
- Add more regularization
- Use smaller model

---

### 2. **Underfitting** (Not Learning)

```
Loss
2.0 | Train ●●●●●●●●●●●●●●●●●●●●●●●● ← Stuck!
1.5 | Val   ○○○○○○○○○○○○○○○○○○○○○○ ← Stuck!
1.0 |
    |___________________________________________
    Both losses stay high and flat
```

**Fix:**
- Increase learning rate (0.001 → 0.002)
- Reduce dropout (0.3 → 0.2)
- Increase model capacity
- Check if features are meaningful

---

### 3. **Divergence** (Exploding)

```
Loss
5.0 |                               ○
4.0 |                           ○
3.0 |                       ○       ← Loss increasing!
2.0 |                   ○
1.0 |       ●●●●    ○               ← Exploding!
0.0 |●●●●●●
    |___________________________________________
```

**Fix:**
- IMMEDIATELY reduce learning rate by 10x
- Check for NaN/Inf in data
- Gradient clipping might help
- Check loss weights aren't too large

---

### 4. **Train/Val Reversed** (Data Leakage!)

```
AUC
0.8 | Val   ○○○○○○○○○○○○○○○○○○○○○○ ← Val better?!
0.7 | Train ●●●●●●●●●●●●●●●●●●●●●●
    |___________________________________________
    🚨 VALIDATION BETTER THAN TRAIN = LEAKAGE!
```

**This means:**
- Forward-looking features in training data
- Label leakage
- Time series split is wrong
- **STOP and fix data pipeline!**

---

## 📊 **SUMMARY: HEALTHY vs UNHEALTHY**

### ✅ **Healthy Training:**
```
                 TRAIN    VAL
Direction AUC:   0.75    0.68   ← Gap is reasonable
Direction Loss:  0.75    0.85   ← Val slightly higher
Volatility MAE:  0.035   0.042  ← Close together
Magnitude MAE:   0.028   0.035  ← Good generalization
```

### ⚠️ **Overfitting:**
```
                 TRAIN    VAL
Direction AUC:   0.85    0.58   ← Huge gap!
Direction Loss:  0.45    1.20   ← Val much worse
Volatility MAE:  0.015   0.080  ← 5x difference!
Magnitude MAE:   0.012   0.065  ← Not generalizing
```

### ❌ **Not Learning:**
```
                 TRAIN    VAL
Direction AUC:   0.52    0.51   ← Barely above random
Direction Loss:  1.05    1.08   ← Near random (1.099)
Volatility MAE:  0.095   0.098  ← Both high
Magnitude MAE:   0.085   0.087  ← No improvement
```

---

## 🎯 **YOUR SUCCESS CRITERIA**

After training completes, you want to see:

1. **Val Direction AUC: > 0.65** ⭐ (Primary goal)
2. **Train-Val gap: < 0.15** (AUC difference)
3. **Val loss stable or decreasing** (not oscillating wildly)
4. **All losses converged** (plateaued for 15+ epochs)
5. **Early stopping triggered** (around epoch 60-100)

**If you achieve:**
- Val AUC **0.65-0.70:** ✅ Good model, proceed to backtest
- Val AUC **0.70-0.75:** 🌟 Excellent, rare for financial data
- Val AUC **0.75+:** 🚨 Check for data leakage!

---

## 📁 **Where to Find These Curves**

### Option 1: TensorBoard (Interactive)
```bash
tensorboard --logdir=run_financial_wavenet_v1/tensorboard_logs --port=6006
```
Then open: `http://localhost:6006`

### Option 2: CSV Log (Raw Data)
```bash
cat run_financial_wavenet_v1/training_log.csv
```

### Option 3: Python Plotting
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training log
log = pd.read_csv('run_financial_wavenet_v1/training_log.csv')

# Plot direction AUC
plt.figure(figsize=(10, 6))
plt.plot(log['epoch'], log['direction_auc'], label='Train AUC')
plt.plot(log['epoch'], log['val_direction_auc'], label='Val AUC')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random')
plt.axhline(y=0.65, color='g', linestyle='--', label='Target')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Direction AUC Over Time')
plt.grid(True)
plt.show()
```

---

## 🎓 **Key Takeaways**

1. **Direction AUC is king** - Focus on this metric above all
2. **Financial data is noisy** - Don't expect 90%+ accuracy
3. **Val AUC 0.65-0.70 is excellent** - That's edge in trading
4. **Small train-val gap is good** - Shows generalization
5. **Monitor for overfitting** - Most common failure mode
6. **Auxiliary tasks help** - Vol/magnitude improve direction prediction

Good luck! 🚀 Watch those curves and adjust hyperparameters based on what you see.
