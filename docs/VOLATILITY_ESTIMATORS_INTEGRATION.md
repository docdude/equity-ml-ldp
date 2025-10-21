# Volatility Estimators Integration Plan

## Analysis of Professional Repository

### Available Estimators

From https://github.com/jasonstrimpel/volatility-trading, the following estimators are available:

1. **GarmanKlass** ✅ Already implemented
2. **HodgesTompkins** ⏳ Not yet implemented
3. **Kurtosis** ⏳ Not yet implemented (statistical feature)
4. **Parkinson** ✅ Already implemented
5. **Raw** (Close-to-Close) ✅ Already implemented
6. **RogersSatchell** ⏳ Not yet implemented
7. **Skew** ⏳ Not yet implemented (statistical feature)
8. **YangZhang** ✅ Already implemented

### Key Architecture from volest.py

The `VolatilityEstimator` class provides:
- `_get_estimator(window, price_data, clean=True)` - Core method that calls specific estimators
- Multiple analysis methods (cones, rolling_quantiles, etc.) - **We don't need these**
- Plotting functions - **We don't need these**

**What we need**: Just the core estimator calculation methods from `volatility/models/*.py`

## Recommended Integration Strategy

### Option 1: Direct Integration (RECOMMENDED)
Copy the estimator calculation logic directly into our `fin_feature_preprocessing.py` methods.

**Advantages:**
- No external dependencies
- Full control over implementation
- Already matches our pattern (we've done this for Yang-Zhang, Parkinson, Garman-Klass)
- No plotting/analysis overhead

**Disadvantages:**
- Need to maintain code ourselves
- But we're already doing this for 3 estimators

### Option 2: Import as Dependency
Add `volatility-trading` as a pip dependency and import estimators.

**Advantages:**
- Get updates from upstream
- Less code to maintain

**Disadvantages:**
- External dependency to manage
- Includes plotting libraries (matplotlib, statsmodels) we don't need
- Less control over implementation details
- GPL-3.0 license (must consider compatibility with our license)

## Implementation Plan

### RECOMMENDED: Option 1 - Direct Integration

**Step 1: Add Missing Estimators**

Add the following to `fin_feature_preprocessing.py`:

#### 1. Rogers-Satchell Volatility
```python
def _rogers_satchell_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252):
    """
    Rogers-Satchell volatility estimator
    Based on: https://github.com/jasonstrimpel/volatility-trading
    """
    import math
    
    log_ho = (df['high'] / df['open']).apply(np.log)
    log_lo = (df['low'] / df['open']).apply(np.log)
    log_co = (df['close'] / df['open']).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return (trading_periods * v.mean())**0.5
    
    result = rs.rolling(window=window, center=False).apply(func=f)
    
    return result
```

#### 2. Hodges-Tompkins Volatility
```python
def _hodges_tompkins_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252):
    """
    Hodges-Tompkins volatility estimator with bias correction
    Based on: https://github.com/jasonstrimpel/volatility-trading
    """
    import math
    
    log_return = (df['close'] / df['close'].shift(1)).apply(np.log)

    vol = log_return.rolling(window=window, center=False).std() * math.sqrt(trading_periods)

    h = window
    n = (log_return.count() - h) + 1

    adj_factor = 1.0 / (1.0 - (h / n) + ((h**2 - 1) / (3 * n**2)))

    result = vol * adj_factor
    
    return result
```

**Step 2: Add to Feature Creation**

In `create_all_features()`, add to the VOLATILITY GROUP:

```python
if config['volatility']:
    print("  → Volatility (Yang-Zhang, Parkinson, Garman-Klass, Rogers-Satchell, Hodges-Tompkins)")
    
    # Existing estimators
    features['volatility_yz_10'] = self._yang_zhang_volatility(df, 10)
    features['volatility_yz_20'] = self._yang_zhang_volatility(df, 20)
    features['volatility_yz_60'] = self._yang_zhang_volatility(df, 60)
    
    features['volatility_parkinson_10'] = self._parkinson_volatility(df, 10)
    features['volatility_parkinson_20'] = self._parkinson_volatility(df, 20)
    
    features['volatility_gk_20'] = self._garman_klass_volatility(df, 20)
    
    features['volatility_cc_20'] = self._close_to_close_volatility(df, 20)
    features['volatility_cc_60'] = self._close_to_close_volatility(df, 60)
    
    # NEW estimators
    features['volatility_rs_20'] = self._rogers_satchell_volatility(df, 20)
    features['volatility_rs_60'] = self._rogers_satchell_volatility(df, 60)
    
    features['volatility_ht_20'] = self._hodges_tompkins_volatility(df, 20)
    features['volatility_ht_60'] = self._hodges_tompkins_volatility(df, 60)
    
    # Volatility ratios (can add more comparisons)
    features['vol_ratio_short_long'] = (
        features['volatility_yz_10'] / (features['volatility_yz_60'] + 1e-8)
    )
    
    # NEW: Rogers-Satchell vs Yang-Zhang ratio
    features['vol_ratio_rs_yz'] = (
        features['volatility_rs_20'] / (features['volatility_yz_20'] + 1e-8)
    )
    
    # Volatility of volatility
    features['vol_of_vol'] = features['volatility_yz_20'].rolling(20).std()
    
    # Realized volatility components (upside vs downside)
    features['realized_vol_positive'] = (
        returns.clip(lower=0).rolling(20).std()
    )
    features['realized_vol_negative'] = (
        returns.clip(upper=0).rolling(20).std()
    )
```

### Step 3: Statistical Features (Skew/Kurtosis)

These are already available in pandas but can be added to the STATISTICAL GROUP:

```python
if config['statistical']:
    print("  → Statistical properties")
    
    # Existing
    features['serial_corr_1'] = returns.rolling(20).apply(
        lambda x: x.autocorr(1) if len(x) > 1 else 0
    )
    
    features['skewness_20'] = returns.rolling(20).skew()
    features['skewness_60'] = returns.rolling(60).skew()
    features['kurtosis_20'] = returns.rolling(20).kurt()
    features['kurtosis_60'] = returns.rolling(60).kurt()
    
    # Can add log-return based versions to match professional repo
    log_returns = (df['close'] / df['close'].shift(1)).apply(np.log)
    features['log_return_skew_20'] = log_returns.rolling(20).skew()
    features['log_return_kurt_20'] = log_returns.rolling(20).kurt()
```

## Feature Count Impact

### Current Minimal Preset: 69 features
- Returns: 7 features
- Volatility: 11 features
- Volume: 13 features
- Momentum: 15 features
- Trend: 10 features
- Bollinger: 5 features
- Price Position: 8 features

### After Adding New Volatility Estimators: +6 features = 75 features
- volatility_rs_20
- volatility_rs_60
- volatility_ht_20
- volatility_ht_60
- vol_ratio_rs_yz
- (existing features remain)

## Benefits of Additional Estimators

### Rogers-Satchell Volatility
- **Advantage**: Assumes zero drift (no directional bias)
- **Use Case**: Better for intraday analysis, options pricing
- **Formula**: Uses only intraday high-low-open-close, ignores overnight gaps

### Hodges-Tompkins Volatility
- **Advantage**: Bias-corrected close-to-close estimator
- **Use Case**: Better for small sample sizes
- **Formula**: Adjusts standard deviation for overlapping samples

### Why Add These?

1. **Diversity**: Different estimators capture different aspects of volatility
2. **Robustness**: Ensemble of estimators is more robust than single estimator
3. **Model Learning**: ML model can learn which estimator is most predictive in different regimes
4. **Relative Ratios**: Ratios between estimators (like `vol_ratio_rs_yz`) can signal market conditions

## Testing Strategy

After implementation:

1. **Test all tickers** - Ensure no extreme values
2. **Check correlation** - Verify estimators are not perfectly correlated
3. **Validate ranges** - All volatility features should be in reasonable range
4. **Retrain model** - Test if additional features improve AUC/PBO

## License Consideration

The professional repo is GPL-3.0. Since we're:
- Copying the calculation formulas (mathematical facts, not copyrightable)
- Not importing their code as a dependency
- Citing the source in comments
- The formulas are from Euan Sinclair's book (academic/public knowledge)

This is similar to implementing a published algorithm from a paper - legally acceptable.

## Recommendation

**Implement Rogers-Satchell and Hodges-Tompkins immediately** since:
1. They're well-established estimators
2. Simple to implement (~15 lines each)
3. Professional reference implementation available
4. Will complement existing estimators
5. Minimal additional computation cost
6. Will increase from 69 → 75 features (reasonable)

Skip Skew/Kurtosis for now since we already have those in the statistical group.
