# PBO Input Format Standardization

## Summary

**Fixed:** `probability_backtest_overfitting()` now accepts strategy returns in the standard `(observations x strategies)` format, eliminating the need for transpose transformations.

## Changes Made

### 1. **lopez_de_prado_evaluation.py**

**Before:**
```python
def probability_backtest_overfitting(self, strategy_returns: np.ndarray, ...):
    """
    Args:
        strategy_returns: Matrix (N_strategies x T_observations)
            Each row is returns for one strategy configuration
    """
    # Convert list to array
    if isinstance(strategy_returns, list):
        strategy_returns = np.array(strategy_returns)
    
    # Transpose: pypbo expects (observations x strategies)
    M = strategy_returns.T
```

**After:**
```python
def probability_backtest_overfitting(self, strategy_returns: np.ndarray, ...):
    """
    Args:
        strategy_returns: Matrix (T_observations x N_strategies)
            Already in correct format for pypbo (observations x strategies)
    """
    # Validate input shape
    if strategy_returns.ndim != 2:
        raise ValueError(f"strategy_returns must be 2D array")
    
    n_observations, n_strategies = strategy_returns.shape
    print(f"   Input shape: ({n_observations} observations, "
          f"{n_strategies} strategies)")
    
    # Use directly - already in correct format
    M = strategy_returns
```

### 2. **fin_training.py**

**Before:**
```python
strategy_returns = []
for i in range(n_strategies):
    returns = np.random.normal(...)
    strategy_returns.append(returns)

# Pass list (gets converted to (strategies, obs) then transposed)
results = evaluator.comprehensive_evaluation(
    strategy_returns=strategy_returns  # List
)
```

**After:**
```python
strategy_returns = []
for i in range(n_strategies):
    returns = np.random.normal(...)
    strategy_returns.append(returns)

# Convert to DataFrame and transpose
returns_df = pd.DataFrame(strategy_returns).T  # (obs x strategies)

# Pass numpy array in correct format
results = evaluator.comprehensive_evaluation(
    strategy_returns=returns_df.values  # (obs x strategies)
)
```

### 3. **demo_evaluation()**

**Before:**
```python
strategy_returns = []
for i in range(n_strategies):
    returns = np.random.normal(...)
    strategy_returns.append(returns)

# Pass list
evaluator.comprehensive_evaluation(
    strategy_returns=strategy_returns
)
```

**After:**
```python
strategy_returns_list = []
for i in range(n_strategies):
    returns = np.random.normal(...)
    strategy_returns_list.append(returns)

# Convert to DataFrame then numpy array (obs x strategies)
strategy_returns_df = pd.DataFrame(strategy_returns_list).T
strategy_returns = strategy_returns_df.values

print(f"Strategy returns shape: {strategy_returns.shape}")
print(f"Format: (observations={strategy_returns.shape[0]}, "
      f"strategies={strategy_returns.shape[1]})")

evaluator.comprehensive_evaluation(
    strategy_returns=strategy_returns
)
```

### 4. **test_pbo.py**

Updated all test functions to transpose after list→array conversion:

```python
# Create list of strategies
strategy_returns = []
for i in range(n_strategies):
    returns = np.random.normal(...)
    strategy_returns.append(returns)

# Convert to numpy and transpose: (strategies, obs) -> (obs, strategies)
strategy_returns = np.array(strategy_returns).T

print(f"Format: ({n_observations} observations, {n_strategies} strategies)")
```

## Benefits

1. **Eliminates Confusion:** No more wondering if you need to transpose or not
2. **Matches pypbo:** Direct compatibility with pypbo library expectations
3. **Matches Notebook:** Aligns with how pypbo.ipynb uses the data
4. **Clearer API:** Docstring explicitly states expected format
5. **Better Validation:** Explicit shape checking with helpful error messages
6. **Metric Flexibility:** Metric function is defined inside the PBO method (can be customized later)

## Migration Path

For existing code that passes `(strategies x observations)`:

```python
# OLD CODE:
strategy_returns = np.array([...])  # Shape: (20, 252)
pbo_result = evaluator.probability_backtest_overfitting(strategy_returns)

# NEW CODE:
strategy_returns = np.array([...])  # Shape: (20, 252)
strategy_returns = strategy_returns.T  # Transpose to (252, 20)
pbo_result = evaluator.probability_backtest_overfitting(strategy_returns)
```

Or convert from list:

```python
# If you have a list of strategy returns:
strategy_returns_list = [...]  # List of 20 arrays, each (252,)

# Convert to DataFrame and transpose
import pandas as pd
returns_df = pd.DataFrame(strategy_returns_list).T  # (252 obs, 20 strat)
pbo_result = evaluator.probability_backtest_overfitting(returns_df.values)
```

## Testing

Run the updated test suite:

```bash
python test_pbo.py
```

All tests now pass with the standardized format:
- ✅ Test 1: Basic PBO with simulated strategies
- ✅ Test 2: High PBO with overfit strategies  
- ✅ Test 3: Low PBO with robust strategies
- ✅ Test 4: List input compatibility
- ✅ Test 5: Single strategy edge case

## Verification

The notebook cell at the end of `pypbo.ipynb` verifies that the integrated function matches pypbo results:

```python
# Using lopez_de_prado_evaluation.py
pbo_result_integrated = evaluator.probability_backtest_overfitting(
    strategy_returns=returns_df.values  # (obs x strategies)
)

# Should match pypbo library result
assert abs(pbo_result_integrated['pbo'] - pbo_result.pbo) < 0.001
```

## Future Considerations

The metric function is now internal to `probability_backtest_overfitting()`:

```python
def metric_func(returns):
    """Calculate Sharpe ratio for each strategy"""
    return perf.sharpe_iid(returns, bench=0, factor=1, log=False)
```

To use a different metric (e.g., Sortino, Calmar), you could:
1. Add a `metric` parameter to the function
2. Pass a custom metric function
3. Create a subclass with different metrics

For now, Sharpe ratio is the standard and appropriate metric for PBO analysis.
