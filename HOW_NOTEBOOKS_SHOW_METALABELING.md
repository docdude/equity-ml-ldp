# How the Notebooks Demonstrate True Meta-Labeling

## Overview

The notebooks (`04MetaLabeling.ipynb` and `03Boosting.ipynb`) show the CORRECT two-stage meta-labeling process that we're NOT currently using in our WaveNet training.

---

## 04MetaLabeling.ipynb - The Key Steps

### Step 1: Initial Triple Barrier (WITHOUT side prediction)

```python
triple_barrier_events = get_events(
    close = data['Close'],
    t_events = data.index[2:],
    pt_sl = [2, 1],
    target = volatility,
    min_ret = 0.01,
    num_threads = 1,
    vertical_barrier_times = vertical_barrier,
    side_prediction = None  # ← NO PRIMARY MODEL YET
)
```

**Result:** Events DataFrame WITHOUT 'side' column

### Step 2: Initial Labels (Acts as Primary Model)

```python
labels = meta_labeling(
    triple_barrier_events,
    data['Close']
)
```

**Result:** Labels with `bin` column = {-1, 1}
- These become the "primary model predictions"
- Represents: "I think we should go LONG (1) or SHORT (-1)"

### Step 3: **THE CRITICAL STEP** - Add Side Predictions

```python
# This is where meta-labeling actually starts!
triple_barrier_events['side'] = labels['bin']
```

**Result:** Now events DataFrame HAS 'side' column
- This is the primary model's directional signal
- Each row has: "Primary model says BUY (+1) or SELL (-1)"

### Step 4: Apply Meta-Labeling (Second Pass)

```python
meta_labels = meta_labeling(
    triple_barrier_events,  # NOW with 'side' column!
    data['Close']
)
```

**Result:** Meta-labels with `bin` column = {0, 1}
- `0` = "Don't take this bet" (primary was wrong or timed out)
- `1` = "Take this bet" (primary was correct)

### Step 5: Filter to Only Positive Bets

```python
# Only train on rows where meta_label = 1
positive_bets = meta_labels[meta_labels['bin'] == 1]
```

**This is the key!** The notebook shows:
- Gray shaded areas = duration of the bet
- Red stars = where `meta_labels['bin'] == 1` (take the bet)
- Model learns: "When should I trust the primary signal?"

---

## The Critical Code Section from Notebook

```python
# Cell: Apply initial labeling
labels = meta_labeling(
    triple_barrier_events,
    data['Close']
)
labels.head()  # Shows {-1, 1} labels

# Cell: ADD SIDE TO EVENTS (this is the key!)
triple_barrier_events['side'] = labels['bin']
triple_barrier_events.head()  # Now has 'side' column

# Cell: Apply meta-labeling
meta_labels = meta_labeling(
    triple_barrier_events,  # with side
    data['Close']
)
meta_labels.head()  # Shows {0, 1} labels
```

**The markdown cell explains:**
> "Possible values ​​for `meta_labels['bin']` generated here are {0, 1} as opposed to {-1, 0, 1}. Machine Learning algorithms are trained solely to decide whether to place a bet or not. If the prediction label is 1, the probability of the secondary model is used to determine the size of the bet, and the side of the position (buy or sell) is already determined by the primary model."

---

## 03Boosting.ipynb - Shows the Same Pattern

### Creating Meta-Labels for RandomForest

```python
# Step 1: Get events (without side)
triple_barrier_event = pd.read_parquet('AAPL_triple_barrier_events.parquet')

# Step 2: Initial labels (primary model)
labels = meta_labeling(
    triple_barrier_event,
    feature_matrix['Close']
)

# Step 3: Add side to events
triple_barrier_event['side'] = labels['bin']

# Step 4: Meta-labeling (secondary model)
meta_labels = meta_labeling(
    triple_barrier_event,  # with side
    feature_matrix['Close']
)

# Step 5: Use meta-labels for training
feature_matrix['side'] = triple_barrier_event['side'].copy()
feature_matrix['label'] = meta_labels['bin'].copy()  # {0, 1}

# Step 6: Filter to only where side != 0
matrix = feature_matrix[feature_matrix['side'] != 0]
```

**Key insight from this notebook:**
- They filter out neutral positions (`side != 0`)
- Train only on directional bets (long or short)
- Model learns to predict `label` = {0, 1}:
  - `0` = Skip this bet (don't trust primary signal)
  - `1` = Take this bet (trust primary signal)

---

## What We're Actually Doing (Our Code)

### In `fin_load_and_sequence.py` (line 160-172):

```python
# Step 1: Create events
events = get_events(
    close=df['Close'],
    t_events=t_events,
    pt_sl=pt_sl,
    target=volatility,
    min_ret=min_ret,
    num_threads=num_threads,
    vertical_barrier_times=vertical_barriers,
    side_prediction=None  # ❌ NO PRIMARY MODEL
)

# Step 2: Apply "meta_labeling" (misleading name!)
labels = meta_labeling(events, df['Close'])

# ❌ WE STOP HERE!
# ❌ We never add 'side' to events
# ❌ We never call meta_labeling a second time
# ❌ We train on {-1, 0, 1} direction labels
```

**What we're missing:**
```python
# ✅ Should do this (if using true meta-labeling):
events['side'] = labels['bin']  # Add primary predictions
meta_labels = meta_labeling(events, df['Close'])  # Get {0, 1} labels
# Now train on meta_labels['bin'], not labels['bin']!
```

---

## The Visualization from Notebook

```python
fig, ax = plt.subplots(figsize = (8, 5))

# For each bet where meta_label = 1:
for idx in meta_labels[meta_labels['bin'] == 1]['2023':].index :
    # Gray box from entry to exit (t1)
    ax.axvspan(
        idx,
        triple_barrier_events.loc[idx]['t1'],
        color = 'lightgray',
        alpha = 0.3
    )

# Red star at entry point
ax.scatter(
    meta_labels[meta_labels['bin'] == 1]['2023':].index,
    data.loc[meta_labels[meta_labels['bin'] == 1].index]['2023':]['Close'],
    marker = '*',
    color = 'red',
    s = 10,
    label = 'meta label'
)
```

**This shows:**
- Only trades where `meta_labels['bin'] == 1` (confident bets)
- Duration of each trade (gray shading)
- Entry points (red stars)
- **Not all possible signals - only filtered high-confidence ones!**

---

## Why This Matters for Our WaveNet

### Current Approach (What We Do):
```
Input: X_seq (features)
Output: y = {-1, 0, 1}
Model predicts: DIRECTION (up/down/neutral)
Problem: 3-class, low signal/noise → 100% neutral
```

### True Meta-Labeling (What Notebooks Show):
```
Stage 1 (Primary Model):
  - Technical indicators
  - Fundamental signals
  - Or another ML model
  - Output: side = {-1, 1}

Stage 2 (Secondary Model - WaveNet):
  Input: X_seq (features) + primary_signal (side)
  Output: y = {0, 1}
  Model predicts: CONFIDENCE (take bet or skip)
  Advantage: 2-class, easier to learn
```

---

## How to Implement True Meta-Labeling

### Option 1: Use Initial Labels as Primary

```python
# In fin_load_and_sequence.py:

# First pass: Get direction labels
events = get_events(..., side_prediction=None)
initial_labels = meta_labeling(events, prices)

# Second pass: Add side and get meta-labels
events['side'] = initial_labels['bin']  # ← Add this!
meta_labels = meta_labeling(events, prices)  # ← Call again!

# Now use meta_labels['bin'] for training
# This is what the notebooks do!
```

### Option 2: Use External Primary Model

```python
# Create primary model (e.g., MA crossover, RSI, etc.)
primary_signals = create_primary_model(df)  # {-1, 1}

# Use primary model in get_events
events = get_events(
    ...,
    side_prediction=primary_signals  # ← Use primary model
)

# Get meta-labels
meta_labels = meta_labeling(events, prices)
# Returns {0, 1}

# Train WaveNet on meta-labels
```

---

## Key Differences Summary

| Aspect | Our Current Code | Notebook Approach |
|--------|------------------|-------------------|
| **Number of `meta_labeling()` calls** | 1 | 2 |
| **'side' column in events** | No | Yes (added after first call) |
| **Label values** | {-1, 0, 1} | {0, 1} (second call) |
| **Model learns** | Direction | Bet sizing/filtering |
| **Primary model** | None | Initial labels (or external) |
| **Use case** | End-to-end direction prediction | Filter and size bets |

---

## Conclusion

The notebooks demonstrate **TWO-STAGE** meta-labeling:

1. **First Stage:** Get initial labels (acts as primary model)
   ```python
   labels = meta_labeling(events_without_side, prices)
   # Returns {-1, 0, 1} or {-1, 1}
   ```

2. **Second Stage:** Add side, call meta_labeling again
   ```python
   events['side'] = labels['bin']
   meta_labels = meta_labeling(events_with_side, prices)
   # Returns {0, 1}
   ```

3. **Train ML Model:** On meta-labels (not initial labels!)
   ```python
   # Only on confident bets
   train_data = meta_labels[meta_labels['bin'] == 1]
   ```

**We're only doing Step 1!** We're not using meta-labeling at all - we're using standard triple-barrier labeling, which is fine, but we need to understand what we're actually doing.

For our WaveNet:
- **Keep current approach** (direction prediction) and fix with heatmaps
- **OR switch to true meta-labeling** (2-stage process shown in notebooks)

The heatmap approach will work with either - it's about improving the learning process, not changing the fundamental task.
