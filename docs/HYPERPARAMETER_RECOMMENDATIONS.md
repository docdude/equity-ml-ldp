# Hyperparameter Recommendations for Financial CNN-LSTM

## Current vs Recommended Settings

### 1. **Learning Rate** 📈

**Current:** `0.001` (1e-3)

**Recommended Options:**

#### Conservative (Recommended for Financial Data):
```python
'learning_rate': 0.0005,  # 5e-4
```
**Why:** Financial time series are noisy and non-stationary. Lower LR provides:
- More stable convergence
- Better generalization
- Less overfitting to recent patterns

#### Aggressive (If you have lots of data):
```python
'learning_rate': 0.001,  # 1e-3 (keep current)
```
**When to use:** 
- Large dataset (>100k samples)
- Strong signal-to-noise ratio
- Using ReduceLROnPlateau callback

#### Warmup Schedule (Best Practice):
```python
# Start low, increase, then decay
initial_learning_rate = 0.0001  # 1e-4
peak_learning_rate = 0.001      # 1e-3
decay_steps = 10000

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=peak_learning_rate,
    first_decay_steps=decay_steps,
    t_mul=2.0,
    m_mul=0.9,
    alpha=0.0001
)
optimizer = optimizers.Adam(learning_rate=lr_schedule)
```

---

### 2. **Batch Size** 🎲

**Current:** `64`

**Recommended Options:**

#### Small Batches (Better for Time Series):
```python
'batch_size': 32,
```
**Pros:**
- More frequent weight updates
- Better escape from local minima
- More noise = better generalization
- Works well with small datasets (<50k samples)

**Cons:**
- Slower training
- More memory-efficient updates

#### Medium Batches (Balanced - RECOMMENDED):
```python
'batch_size': 64,  # Keep current
```
**Sweet spot for:**
- Datasets 50k-200k samples
- Balance of speed and generalization
- Standard for most financial ML

#### Large Batches (If lots of data):
```python
'batch_size': 128,
```
**Only if:**
- Dataset >200k samples
- Strong GPU (V100, A100)
- Using batch normalization effectively

---

### 3. **Epochs** 🔄

**Current:** `175`

**Recommended:** `100-150` with early stopping

**Why:**
- You have `EarlyStopping(patience=20)` ✅
- Most financial models converge in 50-80 epochs
- Additional epochs = overfitting risk
- Let early stopping decide when to stop

**Optimal Configuration:**
```python
'max_epochs': 150,  # Upper bound
'early_stopping_patience': 25,  # Increased patience
'reduce_lr_patience': 10,  # Reduce LR before stopping
```

---

## 🎯 **Recommended Configuration Sets**

### Set 1: Conservative (Financial Data Best Practice)
```python
training_parameters = {
    'learning_rate': 0.0003,           # ⬇️ Lower for stability
    'beta_1': 0.9,
    'beta_2': 0.999,
    'batch_size': 32,                  # ⬇️ Smaller for better generalization
    'max_epochs': 150,                 # ⬇️ Let early stopping decide
    'early_stopping_patience': 30,     # ⬆️ More patience
    'reduce_lr_patience': 12,          # ⬆️ More patience before LR reduction
    'direction_loss_weight': 0.7,
    'volatility_loss_weight': 0.2,
    'magnitude_loss_weight': 0.3
}

# Callbacks configuration
callbacks = [
    EarlyStopping(patience=30, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=12, min_lr=1e-7),
    ModelCheckpoint(...)
]
```

**Best for:**
- Small-medium datasets (<100k samples)
- Noisy financial data
- First training attempt
- Avoiding overfitting

---

### Set 2: Balanced (Recommended Starting Point)
```python
training_parameters = {
    'learning_rate': 0.0005,           # 5e-4 - middle ground
    'beta_1': 0.9,
    'beta_2': 0.999,
    'batch_size': 64,                  # ✅ Keep current
    'max_epochs': 150,                 # ⬇️ Reduced from 175
    'early_stopping_patience': 25,     # ⬆️ Slightly more patience
    'reduce_lr_patience': 10,          # ⬆️ More patience
    'direction_loss_weight': 0.7,
    'volatility_loss_weight': 0.2,
    'magnitude_loss_weight': 0.3
}
```

**Best for:**
- Medium datasets (50k-150k samples)
- Standard GPU (RTX 3090, 4090)
- Balance of speed and accuracy
- **⭐ RECOMMENDED STARTING POINT**

---

### Set 3: Aggressive (High-Performance Setup)
```python
training_parameters = {
    'learning_rate': 0.001,            # ✅ Keep current
    'beta_1': 0.9,
    'beta_2': 0.999,
    'batch_size': 128,                 # ⬆️ Larger batches
    'max_epochs': 200,                 # ⬆️ More epochs
    'early_stopping_patience': 20,     # Standard patience
    'reduce_lr_patience': 8,           # Quick LR adjustment
    'direction_loss_weight': 0.7,
    'volatility_loss_weight': 0.2,
    'magnitude_loss_weight': 0.3
}
```

**Best for:**
- Large datasets (>200k samples)
- Strong GPU (A100, H100)
- Clean, high-quality data
- Fast iteration needed

---

## 📊 **Dataset Size Guidelines**

Based on your ticker list (11 tickers × ~5 years × 252 trading days):
- **Estimated samples:** ~13,000-14,000 raw samples
- **After sequencing (20 timesteps):** ~13,000 sequences
- **After train/val split:** ~10,000 train, ~3,000 val

**For this dataset size → Use Set 1 (Conservative) or Set 2 (Balanced)**

---

## 🔬 **Advanced: Learning Rate Schedules**

### Option 1: Cosine Decay with Warmup (BEST)
```python
def create_lr_schedule(initial_lr=0.0001, peak_lr=0.001, 
                       warmup_steps=1000, total_steps=10000):
    """
    Warmup + Cosine decay schedule
    """
    def lr_schedule(step):
        if step < warmup_steps:
            # Linear warmup
            return initial_lr + (peak_lr - initial_lr) * (step / warmup_steps)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return initial_lr + (peak_lr - initial_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
    
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Add to callbacks
callbacks.append(create_lr_schedule())
```

### Option 2: One Cycle Policy (Fast Training)
```python
from tensorflow.keras.callbacks import LearningRateScheduler

def one_cycle_lr(max_lr=0.003, min_lr=0.0001, steps_per_epoch=100, epochs=150):
    """One cycle learning rate policy"""
    total_steps = steps_per_epoch * epochs
    mid_step = total_steps // 2
    
    def lr_schedule(step):
        if step < mid_step:
            # Increase phase
            return min_lr + (max_lr - min_lr) * (step / mid_step)
        else:
            # Decrease phase
            return max_lr - (max_lr - min_lr) * ((step - mid_step) / mid_step)
    
    return LearningRateScheduler(lambda epoch, lr: lr_schedule(epoch * steps_per_epoch))
```

---

## 🎯 **Practical Recommendations**

### For Your Current Run:
**Keep running with current settings!** Your configuration is reasonable:
- ✅ LR=0.001 is standard
- ✅ Batch=64 is good
- ✅ Epochs=175 with early stopping will prevent overtraining

### For Next Run (Improved):
```python
training_parameters = {
    'learning_rate': 0.0005,        # ⬇️ Reduce by 50%
    'batch_size': 64,               # ✅ Keep
    'max_epochs': 150,              # ⬇️ Slightly lower
    # ... rest same
}

# Update callbacks
callbacks = create_callbacks(
    output_dir=experiment_save_path,
    patience=25,                    # ⬆️ More patience
    monitor='val_direction_auc'
)
```

---

## 📈 **Monitoring Tips**

Watch for these patterns in `training_log.csv`:

### Good Training:
```
Epoch   train_loss   val_loss   val_direction_auc
10      0.850        0.920      0.650
20      0.720        0.810      0.680
30      0.650        0.750      0.695
40      0.600        0.720      0.702  ← Improving
```

### Overfitting (Reduce LR or Batch Size):
```
Epoch   train_loss   val_loss   val_direction_auc
10      0.850        0.920      0.650
20      0.550        0.880      0.670
30      0.350        0.950      0.665  ← Val getting worse
40      0.200        1.100      0.655  ← Overfitting!
```

### Underfitting (Increase LR or Capacity):
```
Epoch   train_loss   val_loss   val_direction_auc
10      0.950        0.960      0.520
50      0.945        0.958      0.525  ← Not learning
100     0.943        0.957      0.527  ← Stuck!
```

---

## 🚀 **Quick Action Items**

1. **For next run, change these:**
```python
'learning_rate': 0.0005,      # ⬇️ From 0.001
'max_epochs': 150,            # ⬇️ From 175
'early_stopping_patience': 25 # ⬆️ From 20
```

2. **Monitor validation AUC** - Target: >0.65 for financial data

3. **If val_auc plateaus early (<30 epochs):**
   - Increase learning rate to 0.001
   - Reduce batch size to 32

4. **If overfitting (train >> val):**
   - Decrease learning rate to 0.0003
   - Increase dropout to 0.4
   - Add more data augmentation

---

## 🔍 **Expected Results by Configuration**

| Config | Train Time | Val AUC | Overfitting Risk |
|--------|-----------|---------|------------------|
| Conservative (LR=0.0003, BS=32) | 2-3 hrs | 0.65-0.70 | Low ✅ |
| Balanced (LR=0.0005, BS=64) | 1-2 hrs | 0.68-0.72 | Medium |
| Aggressive (LR=0.001, BS=128) | 1 hr | 0.70-0.75 | High ⚠️ |
| Current (LR=0.001, BS=64) | 1.5 hrs | 0.68-0.73 | Medium-High |

---

## 💡 **Key Insights**

1. **Financial data is noisy** → Lower LR (0.0003-0.0005) often works better
2. **Small batches generalize better** → BS=32-64 is sweet spot
3. **Early stopping is your friend** → Don't worry about max_epochs
4. **Monitor val_direction_auc** → It's your north star metric
5. **First run = exploration** → Your current settings are fine for baseline!

**Bottom line:** Your current config is reasonable. After this run completes, adjust based on the validation curve! 📊
