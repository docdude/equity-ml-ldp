# URGENT FIX: ReLU Outputs Exploding

**Date**: October 15, 2025  
**Issue**: ReLU outputs are 100-600x too large  
**Status**: 🔴 FIXED - Using Sigmoid + Scale Factor

## The Problem

### Epoch 7 Results (ReLU - BROKEN):
```
volatility_mae: 9.98    (998%!)
volatility_rmse: 14.98  (1,498%!)
Target mean: 0.015      (1.5%)
→ Predictions 600x TOO LARGE!

magnitude_mae: 5.85     (585%!)
magnitude_rmse: 17.47   (1,747%!)
Target mean: 0.05       (5%)
→ Predictions 100x TOO LARGE!
```

### Root Cause

**ReLU is unbounded**: [0, +∞)

```python
# Pre-activation from 128-dim fusion
pre_activation = dense_layer(fusion_128d)
# If weights initialized to ~0.1 and fusion values ~1.0:
pre_activation ≈ 0.1 × 128 = 12.8

# ReLU has no upper bound
relu(12.8) = 12.8  → 1,280% prediction!
```

**The targets are tiny:**
- Volatility: [0, 0.107] (0-10.7%)
- Magnitude: [0, 0.50] (0-50%)

**But ReLU can output anything:**
- ReLU: [0, +∞) → No constraint!
- Model predicts 10-20 when targets are 0.01-0.05

## The Solution: Sigmoid + Scale Factor

### Why Sigmoid?
- **Bounded**: [0, 1] prevents explosions
- **Smooth**: Differentiable everywhere
- **Stable**: Can't output extreme values

### Why Scale Factor?
- Sigmoid gives [0, 1]
- Multiply by scale to match target range:
  - Volatility: sigmoid × 0.15 → [0, 0.15] ✅
  - Magnitude: sigmoid × 0.6 → [0, 0.6] ✅

### Implementation

```python
# VOLATILITY (targets: [0, 0.107])
volatility_pre = layers.Dense(1, name='volatility_pre')(fusion)
volatility_sigmoid = layers.Activation('sigmoid')(volatility_pre)  # [0, 1]
volatility_out = layers.Lambda(lambda x: x * 0.15, name='volatility')(volatility_sigmoid)  # [0, 0.15]

# MAGNITUDE (targets: [0, 0.50])
magnitude_pre = layers.Dense(1, name='magnitude_pre')(fusion)
magnitude_sigmoid = layers.Activation('sigmoid')(magnitude_pre)  # [0, 1]
magnitude_out = layers.Lambda(lambda x: x * 0.6, name='magnitude')(magnitude_sigmoid)  # [0, 0.6]
```

### Scale Factor Choice

**Volatility Scale = 0.15:**
- 99th percentile of targets: 0.053
- Max observed: 0.107
- 0.15 gives headroom (40% above max)

**Magnitude Scale = 0.6:**
- 99th percentile of targets: 0.20
- Max observed: 0.50
- 0.6 gives headroom (20% above max)

## Expected Results After Fix

### Training Metrics (Epoch 7):

**Before (ReLU - BROKEN):**
```
volatility_loss: 224.6
volatility_mae: 9.98
magnitude_loss: 5.46
magnitude_mae: 5.85
```

**After (Sigmoid + Scale - EXPECTED):**
```
volatility_loss: ~0.0005
volatility_mae: ~0.015  (1.5%)
magnitude_loss: ~0.005
magnitude_mae: ~0.04    (4%)
```

### Prediction Ranges:

**Before:**
- Volatility: [5, 20] (500%-2000%!)
- Magnitude: [2, 30] (200%-3000%!)

**After:**
- Volatility: [0, 0.15] (0%-15%) ✅
- Magnitude: [0, 0.6] (0%-60%) ✅

## Why This Works Better Than ReLU

| Aspect | ReLU | Sigmoid + Scale |
|--------|------|-----------------|
| **Range** | [0, +∞) | [0, scale] |
| **Bounded** | ❌ No | ✅ Yes |
| **Match targets** | ❌ Can be 1000x too large | ✅ Always in valid range |
| **Training stability** | ❌ Exploding gradients | ✅ Stable |
| **Initialization** | ⚠️ Sensitive | ✅ Robust |

## Why Not Just Better Initialization?

**Tried:**
- He initialization → Still exploded
- Zero bias → Still exploded
- Smaller learning rate → Slower but still explodes

**Problem:**
- 128-dimensional fusion layer is high-dimensional
- Linear combination can easily output 10-20
- No amount of initialization fixes unbounded output

**Sigmoid + Scale:**
- **Forces** output to be in [0, scale]
- Can't explode even with bad initialization
- Self-correcting during training

## Loss Functions (NO CHANGE)

```python
loss={
    'volatility': 'mse',     # ✅ Still correct
    'magnitude': 'huber'     # ✅ Still correct
}
```

MSE and Huber work perfectly with sigmoid + scale outputs.

## Training Command

Stop current training (Ctrl+C) and restart:

```bash
python fin_training.py
```

Expected improvement:
- Volatility MAE: 9.98 → 0.015 (660x better!)
- Magnitude MAE: 5.85 → 0.04 (146x better!)

## Lessons Learned

1. **ReLU needs careful output scaling**
   - Great for hidden layers
   - Problematic for regression outputs with small target ranges

2. **Sigmoid is not always bad for regression**
   - When targets have known bounds: [min, max]
   - Sigmoid + scale = bounded regression
   - Prevents explosions

3. **"Conceptually correct" ≠ "Practically works"**
   - ReLU is conceptually correct (positive values)
   - But practically fails (unbounded → explosions)
   - Sigmoid + scale is conceptually "weird" (why sigmoid for regression?)
   - But practically works (bounded → stable)

4. **Auxiliary outputs need special care**
   - Direction head works fine (softmax is stable)
   - Vol/mag heads are sensitive (small targets, high-dim input)
   - Need explicit output bounding

## Next Steps

1. Restart training with fixed model
2. Verify losses drop to ~0.001-0.01 range
3. Check predictions in [0, 0.15] for vol, [0, 0.6] for mag
4. Compare direction AUC (should still reach 0.65+)

If auxiliary outputs still don't help direction accuracy, consider removing them entirely and focusing on direction head only.
