# Training Run #3 Plan - Fixed Volatility & Magnitude Outputs

**Date**: October 15, 2025  
**Previous Run**: Run #2 with broken vol/mag (saturated at 1.0)  
**This Run**: Run #3 with ReLU + He initialization

## Changes Applied ✅

### 1. Fixed Model Architecture (`fin_model.py`)

**Before (BROKEN):**
```python
volatility_out = layers.Dense(1, activation='sigmoid', name='volatility')(fusion)
magnitude_out = layers.Dense(1, activation='linear', name='magnitude')(fusion)
```

**After (FIXED):**
```python
volatility_out = layers.Dense(
    1, 
    activation='relu',
    kernel_initializer='he_normal',
    name='volatility'
)(fusion)

magnitude_out = layers.Dense(
    1, 
    activation='relu',
    kernel_initializer='he_normal',
    name='magnitude'
)(fusion)
```

### 2. Why This Fix Works

**Problem Summary:**
- Sigmoid outputs [0, 1], but volatility targets are [0, 0.107]
- MSE loss: (1.0 - 0.02)² = 0.96 per sample → massive loss
- Model learned to saturate at 1.0 (100% of predictions = 0.9999+)
- Magnitude predictions were 100%+ when reality is 5%

**ReLU Solution:**
- ✅ Outputs [0, +∞) matching volatility/magnitude domain
- ✅ MSE loss now makes sense on natural scale
- ✅ He initialization prevents dying neurons
- ✅ No post-processing needed (strictly non-negative)

**Why Not LeakyReLU?**
- Training is short (15-20 epochs)
- Early stopping on val_direction_auc (not vol/mag)
- Vol/mag are auxiliary outputs (low priority)
- Dying ReLU risk is negligible in this setup
- ReLU + He init is simpler and sufficient

## Expected Improvements

### Training Metrics

**Run #2 (Broken):**
```
Epoch 19/500
   direction_loss: 0.5894
   volatility_loss: 398.3452    ← MASSIVE!
   magnitude_loss: 87.2134      ← MASSIVE!
   
   val_direction_auc: 0.6516 ✅
   val_volatility_mae: 631%     ← Meaningless
   val_magnitude_mae: 362%      ← Meaningless
```

**Run #3 (Expected):**
```
Epoch ~15-20/500
   direction_loss: ~0.58        (similar)
   volatility_loss: ~0.002      ← Fixed! (100x smaller)
   magnitude_loss: ~0.02        ← Fixed! (1000x smaller)
   
   val_direction_auc: 0.65-0.66 (should be similar or better)
   val_volatility_mae: ~0.005   ← Interpretable! (0.5%)
   val_magnitude_mae: ~0.02     ← Interpretable! (2%)
```

### Prediction Quality

**Run #2 (Broken):**
```
Volatility predictions: [0.9999, 1.0000]  (saturated, useless)
Magnitude predictions:  [0.39, 3.93]      (39%-393%, absurd)
```

**Run #3 (Expected):**
```
Volatility predictions: [0.005, 0.08]     (0.5%-8%, realistic!)
Magnitude predictions:  [0.01, 0.30]      (1%-30%, matches reality!)
```

## Verification Steps

After training completes, check:

1. **Volatility predictions NOT saturated:**
   ```python
   import ast
   df = pd.read_csv('predictions.csv')
   vol_preds = [ast.literal_eval(row['volatility'])[0] for _, row in df.iterrows()]
   print(f"Max: {max(vol_preds):.4f}")  # Should be < 0.15, not 1.0
   print(f"% near 1.0: {sum(v > 0.999 for v in vol_preds) / len(vol_preds) * 100:.1f}%")  # Should be 0%
   ```

2. **Magnitude predictions reasonable:**
   ```python
   mag_preds = [ast.literal_eval(row['magnitude'])[0] for _, row in df.iterrows()]
   print(f"Mean: {np.mean(mag_preds):.4f}")  # Should be ~0.05, not 1.22
   print(f"Max: {max(mag_preds):.4f}")       # Should be < 0.50, not 3.93
   ```

3. **Training losses small:**
   ```python
   # Check final epoch logs
   # volatility_loss should be < 0.01
   # magnitude_loss should be < 0.05
   ```

4. **Direction AUC maintained:**
   ```python
   # val_direction_auc should still be >= 0.65
   # If better (0.66-0.67), auxiliary outputs are helping!
   ```

## Re-run PBO Test

After training, re-run PBO with corrected model:

```bash
python test_pbo_quick.py
```

**Expected outcome:**
- PBO should remain low (< 0.3)
- Sharpe ratio might improve slightly if vol/mag help direction
- Strategy returns should be similar (still using direction only)

## Training Command

```bash
python fin_training.py
```

This will:
1. Load balanced_config from training_configs.py
2. Build model with fixed ReLU outputs
3. Train for up to 500 epochs (early stop ~15-20)
4. Save best model based on val_direction_auc
5. Generate predictions.csv for PBO analysis

## Success Criteria

✅ **Must Have:**
- [ ] Volatility predictions in range [0, 0.15]
- [ ] Magnitude predictions in range [0, 0.50]
- [ ] Volatility loss < 0.01
- [ ] Magnitude loss < 0.05
- [ ] val_direction_auc >= 0.65

🎯 **Nice to Have:**
- [ ] val_direction_auc > 0.66 (auxiliary outputs helping)
- [ ] PBO < 0.2 (maintained or improved)
- [ ] Sharpe ratio > 0.5 (confidence threshold optimization)

## Rollback Plan

If Run #3 performs worse:
- [ ] Check if direction AUC dropped significantly
- [ ] Verify loss weights are still (1.0, 0.3, 0.3)
- [ ] Try reducing vol/mag loss weights to (1.0, 0.1, 0.1)
- [ ] Consider removing vol/mag outputs entirely if they hurt

## Next Steps After Run #3

1. **If successful**: Document improved architecture, run full López de Prado evaluation
2. **If direction AUC improved**: Auxiliary outputs are valuable, keep them
3. **If direction AUC same**: Auxiliary outputs not helping much, consider simplifying
4. **If direction AUC worse**: Re-evaluate loss weights or remove auxiliary outputs

Ready to train? Run:
```bash
python fin_training.py
```
