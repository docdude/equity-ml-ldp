# Walk-Forward Implementation Comparison

## Working Implementation (evaluate_lopez_de_prado.py)

### ✅ Correct Embargo Logic

```python
# Expanding window for training
train_dates_all = np.concatenate(chunks[:i])
test_dates = chunks[i]

# Apply embargo between train and test
if embargo_days > 0 and len(train_dates_all) > embargo_days:
    # Remove last embargo_days from training
    train_dates = train_dates_all[:-embargo_days]
    embargo_dates = train_dates_all[-embargo_days:]
    
    print(f"   Train period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"   Embargo gap:  {embargo_dates[0]} to {embargo_dates[-1]} ({embargo_days} days) [EXCLUDED]")
    print(f"   Test period:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
```

**Key Points**:
1. **Removes last embargo_days FROM train** - doesn't create a gap before test
2. **Train goes right up to embargo boundary** - no wasted data
3. **Embargo is the gap between train end and test start** - this prevents leakage

### Why This Works

```
Timeline:
[Day 1 -------- Day 240] [Day 241-245] [Day 246 -------- Day 270]
     TRAINING               EMBARGO         TEST
```

**The Logic**:
- Train uses days 1-240
- Embargo removes days 241-245 (last 5 days of training window)
- Test uses days 246-270
- **No gap before test starts** - test immediately follows embargo
- **No data wasted** - embargo comes from end of training

## Our Broken Implementation (lopez_de_prado_evaluation.py)

### ❌ Incorrect Logic

```python
train_end = start_idx - embargo_size      # Creates gap!
embargo_start = train_end
embargo_end = start_idx
test_start = start_idx
```

**Problem**:
```
Timeline with start_idx=252, embargo_size=5:
[Day 0 -------- Day 246] [Day 247-251] [Day 252 -------- Day 272]
     TRAINING               "EMBARGO"       TEST
```

- Train uses 0-246 (stops at 247)
- "Embargo" is 247-251 (not from training, just a gap!)
- Test uses 252-272
- **Gap exists** - days 247-251 are unused
- **Not a real embargo** - just wasting data

## The Fix

### Option 1: Simple Fix (Match Working Implementation)

```python
def walk_forward_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                         window_size: int = 252, step_size: int = 21,
                         sample_weights: Optional[np.ndarray] = None,
                         expanding: bool = False,
                         save_predictions: bool = True) -> Dict:
    
    # ... setup ...
    
    while start_idx + step_size < len(X):  # Remove embargo_size from condition
        period += 1
        
        # Define train and test periods
        if expanding:
            train_start = 0
        else:
            train_start = start_idx - window_size
        
        # NEW: Train goes right up to start_idx
        train_end_all = start_idx
        
        # NEW: Remove embargo from END of training
        if embargo_size > 0:
            train_end = train_end_all - embargo_size
            embargo_start = train_end
            embargo_end = train_end_all
        else:
            train_end = train_end_all
            embargo_start = embargo_end = train_end_all
        
        test_start = start_idx
        test_end = min(start_idx + step_size, len(X))
        
        # Rest stays the same...
```

### Option 2: Proper Label-Time-Based Purging (Best)

```python
def walk_forward_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                         pred_times: pd.Index,  # ADD THIS
                         window_size: int = 252, step_size: int = 21,
                         sample_weights: Optional[np.ndarray] = None,
                         expanding: bool = False,
                         save_predictions: bool = True) -> Dict:
    
    # Get label determination times
    train_times = self.get_train_times(X, pred_times)
    
    while start_idx + step_size < len(X):
        period += 1
        
        # Define ranges WITHOUT artificial gap
        if expanding:
            train_start = 0
        else:
            train_start = start_idx - window_size
        
        train_end = start_idx  # No gap!
        test_start = start_idx
        test_end = min(start_idx + step_size, len(X))
        
        # Get ALL potential training indices
        train_idx = range(train_start, train_end)
        test_idx = range(test_start, test_end)
        
        # Apply PROPER embargo based on label determination times
        test_times = pred_times[test_idx]
        t0_test = test_times.min()
        t1_test = test_times.max()
        embargo_time = (t1_test - t0_test) * self.embargo_pct
        
        # Remove training samples with overlapping label times
        train_times_subset = train_times[train_idx]
        valid_train_mask = (train_times_subset < t0_test - embargo_time) | \
                          (train_times_subset > t1_test + embargo_time)
        purged_train_idx = train_idx[valid_train_mask]
        
        # Report what got purged
        n_purged = len(train_idx) - len(purged_train_idx)
        print(f"   Train: {train_start} to {train_end-1} "
              f"({len(train_idx)} samples, {n_purged} purged by embargo)")
        print(f"   Test:  {test_start} to {test_end-1} ({len(test_idx)} samples)")
        
        # Train on purged data
        X_train = X.iloc[purged_train_idx]
        y_train = y.iloc[purged_train_idx]
        # ... rest of training ...
```

## Summary of Changes Needed

### 1. Walk-Forward Fix (Immediate)
- Remove `embargo_size` from the condition
- Train goes right up to test start
- Remove embargo from END of training, not create gap before test

### 2. Forward Returns Removal (Next)
- Delete lines 108-112 in `fin_feature_preprocessing.py`
- Remove `forward_return_1d`, `forward_return_3d`, `forward_return_5d`

### 3. PBO Fix (After above)
- Remove random data generation
- Implement proper CSCV-based PBO
- Use real strategy returns from walk-forward predictions
- OR: Disable PBO until we have multiple strategies to test
