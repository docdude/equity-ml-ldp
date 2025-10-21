# Meta-Labeling vs Triple Barrier: Critical Conceptual Gap

## 🚨 CRITICAL ISSUE: We're NOT using meta-labeling correctly!

After reviewing the notebooks and our implementation, there's a fundamental misunderstanding of how meta-labeling should work in López de Prado's framework.

---

## 1. What is Meta-Labeling? (The Correct Approach)

### The Two-Stage Process:

**Stage 1: PRIMARY MODEL** (predicts DIRECTION)
- An exogenous/fundamental model predicts BUY or SELL signals
- Examples: Technical indicator crossover, fundamental signal, sentiment signal
- This is the `side_prediction` parameter in `get_events()`
- Output: `side = +1` (buy) or `side = -1` (sell)

**Stage 2: SECONDARY MODEL (META-LABELING)** (predicts SIZE)
- ML model learns to predict: "Should we take this bet or not?"
- Binary classification: `{0, 1}` where:
  - `1` = "YES, take this bet" (with confidence → bet size)
  - `0` = "NO, skip this bet"
- This filters out false positives from the primary model
- Uses `meta_labeling()` function with `side` column present

### The Key Insight:
```
Primary Model: "I think we should BUY" (direction)
Secondary Model: "YES, I agree, bet 50% of kelly" (size/confidence)
                OR
                "NO, this looks like a false positive" (filter out)
```

---

## 2. What Are We Actually Doing? (Our Current Approach)

### Our Current Process:

```python
# Step 1: Create triple barrier events
events = get_events(
    close=df['Close'],
    t_events=df.index,
    pt_sl=[2, 1],
    target=volatility,
    min_ret=0.005,
    side_prediction=None  # ❌ NO PRIMARY MODEL!
)

# Step 2: Apply meta_labeling
labels = meta_labeling(events, df['Close'])
# Returns: {-1, 0, 1} for DOWN, TIMEOUT, UP
```

### What This Actually Does:

When `side_prediction=None`:
1. `get_events()` assumes `side = +1` (all long positions)
2. Triple barriers look for TP/SL/vertical hits
3. `meta_labeling()` returns **3-class labels**: {-1, 0, 1}
   - `1` = UP/TP hit (profitable long)
   - `-1` = DOWN/SL hit (loss on long)
   - `0` = TIMEOUT/vertical (neither TP nor SL hit)

**This is NOT meta-labeling! This is standard triple-barrier labeling!**

---

## 3. The Notebooks Show REAL Meta-Labeling

### From `04MetaLabeling.ipynb`:

```python
# STEP 1: Create initial events (no side)
triple_barrier_events = get_events(
    close=data['Close'],
    t_events=data.index[2:],
    pt_sl=[2, 1],
    target=volatility,
    side_prediction=None  # Initial pass
)

# STEP 2: Get initial labels
labels = meta_labeling(triple_barrier_events, data['Close'])
# Returns {-1, 1} for this dataset

# STEP 3: ✅ ADD SIDE PREDICTIONS (This is the key!)
triple_barrier_events['side'] = labels['bin']
# Now events has a 'side' column with primary model predictions

# STEP 4: ✅ APPLY META-LABELING (Second pass)
meta_labels = meta_labeling(
    triple_barrier_events,  # Now with 'side' column!
    data['Close']
)
# Returns {0, 1} - binary classification
# 1 = take the bet, 0 = skip the bet
```

### Key Difference:
```python
# Without 'side' column (our approach):
labels['bin'] values: {-1, 0, 1}  # 3-class: direction prediction
Purpose: Predict UP, DOWN, or NEUTRAL

# With 'side' column (true meta-labeling):
meta_labels['bin'] values: {0, 1}  # 2-class: bet sizing
Purpose: Filter false positives, determine bet size
```

---

## 4. How López de Prado Uses This

### The Complete Framework:

```python
# STAGE 1: PRIMARY MODEL (can be anything)
# - Moving average crossover
# - RSI overbought/oversold
# - Fundamental signal
# - Another ML model
# → Generates side_prediction Series: +1 or -1

# STAGE 2: TRIPLE BARRIERS (with side)
events = get_events(
    close=prices,
    t_events=signal_dates,
    pt_sl=[2, 2],  # Symmetric when using side
    target=volatility,
    side_prediction=primary_signals  # ✅ From primary model
)

# STAGE 3: META-LABELING
meta_labels = meta_labeling(events, prices)
# Now it returns {0, 1}:
# - 1 if the bet was correct (hit TP)
# - 0 if the bet was wrong (hit SL) or timed out

# STAGE 4: TRAIN META-MODEL
# Features: market conditions, volatility, trend strength, etc.
# Target: meta_labels['bin']  # {0, 1}
# Prediction: "Should we take this primary model's signal?"
# Confidence: Used for bet sizing (Kelly criterion)
```

---

## 5. What Our WaveNet is Actually Learning

### Current Reality:

```python
# We're training WaveNet to predict:
X_seq → y = {-1, 0, 1}

# This means:
# -1: "Price will go DOWN (hit stop-loss)"
#  0: "Price will TIMEOUT (hit vertical barrier)"
#  1: "Price will go UP (hit take-profit)"

# This is DIRECTIONAL prediction, NOT meta-labeling!
```

### Why This Causes 100% NEUTRAL:
- Model sees 3 balanced classes (~33% each)
- With low signal-to-noise ratio in financial data
- Model learns: "Can't predict direction reliably"
- Output: Predict neutral (0) for everything
- Result: No actionable trades

---

## 6. What We SHOULD Be Doing (Two Options)

### Option A: True Meta-Labeling (López de Prado's Way)

```python
# 1. Build/use a PRIMARY MODEL
primary_signals = create_primary_model(df)  # Returns +1/-1

# 2. Create events WITH side prediction
events = get_events(
    close=df['Close'],
    t_events=signal_dates,
    pt_sl=[2, 2],  # Symmetric
    target=volatility,
    side_prediction=primary_signals  # ✅ Key!
)

# 3. Get meta-labels
meta_labels = meta_labeling(events, df['Close'])
# Returns {0, 1}

# 4. Train WaveNet
X_seq → y = {0, 1}  # "Take bet" or "Skip bet"
# Model output used for: bet sizing, filtering false positives
```

**Advantages:**
- Simpler 2-class problem
- Separates direction (primary) from sizing (secondary)
- Can use interpretable primary model + ML for refinement
- Reduces overfitting (not learning direction)

### Option B: Standard Triple Barrier (What we're actually doing)

```python
# 1. Create events (no primary model)
events = get_events(
    close=df['Close'],
    t_events=df.index,
    pt_sl=[2, 1],  # Asymmetric
    target=volatility,
    side_prediction=None  # No primary model
)

# 2. Get labels
labels = meta_labeling(events, df['Close'])
# Returns {-1, 0, 1} or {-1, 1}

# 3. Train WaveNet for DIRECTION
X_seq → y = {-1, 0, 1}  # DOWN, NEUTRAL, UP
# Model predicts: direction of next move
```

**Challenges:**
- 3-class problem is harder
- Low signal-to-noise ratio
- Needs strong features
- Risk of 100% neutral predictions

---

## 7. Why We're Confused

### The Naming is Misleading:

```python
# Function name: meta_labeling()
# But behavior depends on 'side' column:

# WITHOUT 'side' column:
meta_labeling(events, prices)
# → Returns {-1, 0, 1}: DIRECTION labels
# → This is NOT meta-labeling!
# → This is standard triple-barrier labeling

# WITH 'side' column:
events['side'] = primary_signals
meta_labeling(events, prices)
# → Returns {0, 1}: BET SIZING labels
# → This IS meta-labeling!
```

### The Code Shows This:

```python
# From labeling.py:
def meta_labeling(triple_barrier_events, close):
    # ...
    if 'side' in events_:
        out_df['ret'] = out_df['ret'] * events_['side']  # ✅ Signed return
    
    out_df = barrier_touched(out_df, triple_barrier_events)
    
    if 'side' in events_:
        out_df.loc[out_df['ret'] <= 0, 'bin'] = 0  # ✅ Binary: 0 or 1
    # ...
```

---

## 8. Recommendations

### Immediate Decision Required:

**A. Continue with Standard Triple Barrier (Simpler)**
```python
# Accept we're doing direction prediction
# Focus on fixing 100% neutral issue:
# 1. Better features
# 2. Heatmap augmentation (our swimming method!)
# 3. Class balancing
# 4. Threshold tuning

# Keep current approach:
side_prediction=None  # No primary model
labels: {-1, 0, 1}    # Direction prediction
```

**B. Switch to True Meta-Labeling (More Sophisticated)**
```python
# Build complete 2-stage system:
# 1. Create primary model (technical indicators, fundamentals)
# 2. Use WaveNet as meta-model (filters false positives)
# 3. Combine for bet sizing

# Changes needed:
side_prediction=primary_signals  # Add primary model
labels: {0, 1}                   # Bet sizing prediction
```

### My Recommendation: **Start with A, evolve to B**

**Phase 1 (Now):** Fix standard triple-barrier approach
- Focus on heatmap augmentation (your swimming method!)
- This addresses the 100% neutral problem
- Get baseline working first

**Phase 2 (Later):** Add meta-labeling
- Once we have working direction prediction
- Use those predictions as primary model
- Train second WaveNet for bet sizing
- This is the full López de Prado framework

---

## 9. Action Items

### Understand Current State:
- [ ] We're NOT using meta-labeling (despite calling the function)
- [ ] We're doing 3-class direction prediction
- [ ] This explains why we see {-1, 0, 1} labels everywhere

### For Heatmap Approach:
- [ ] Keep `side_prediction=None` (standard approach)
- [ ] Keep {-1, 0, 1} labels (direction prediction)
- [ ] Apply heatmap augmentation to THESE labels
- [ ] This should fix 100% neutral issue

### For Future Meta-Labeling:
- [ ] Build primary model (separate project)
- [ ] Generate `side_prediction` Series
- [ ] Pass to `get_events(side_prediction=signals)`
- [ ] Train WaveNet on {0, 1} meta-labels
- [ ] Use probabilities for bet sizing

---

## 10. Key Takeaways

1. **Meta-labeling is a 2-stage process:**
   - Primary model: predicts DIRECTION
   - Secondary model: predicts SIZE/confidence

2. **We're currently doing:**
   - Single-stage direction prediction
   - No primary model
   - Standard triple-barrier labels

3. **The function `meta_labeling()` is misleading:**
   - Without 'side': returns direction labels {-1, 0, 1}
   - With 'side': returns sizing labels {0, 1}
   - Same function, different behavior!

4. **Our 100% neutral issue is from:**
   - 3-class direction prediction problem
   - Not from incorrect meta-labeling implementation
   - Because we're not actually using meta-labeling!

5. **The heatmap approach is still valid:**
   - Works for direction prediction
   - Just need to augment BEFORE sequencing
   - Should reduce neutral bias

6. **To use REAL meta-labeling:**
   - Need to build primary model first
   - Then train WaveNet as secondary filter
   - This is an architectural change, not a bug fix

---

## Conclusion

**We haven't been using meta-labeling at all!** We've been using standard triple-barrier labeling, which returns direction labels. The notebooks show how REAL meta-labeling works: you need a primary model that provides `side_prediction`, then the secondary model learns to filter and size bets.

This doesn't invalidate our work - we just need to be clear about what we're actually doing: **3-class direction prediction**, not **2-class bet sizing (meta-labeling)**.

For now, let's fix the direction prediction with heatmap augmentation. Later, we can add true meta-labeling as a second stage.
