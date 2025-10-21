# Feature Configuration System

A flexible system for selecting which features to engineer in the financial ML pipeline.

## Overview

The feature configuration system organizes **108 features** into **12 logical groups** with toggle switches for easy control. This allows you to:

- **Start with minimal features** for faster experimentation
- **Scale up gradually** as you understand your data
- **Reduce overfitting risk** by limiting feature count
- **Improve training speed** with selective feature engineering
- **Create custom presets** for different use cases

## Quick Start

### Method 1: Use a Preset (Recommended)

```python
from fin_feature_preprocessing import EnhancedFinancialFeatures
import pandas as pd

# Load data
df = pd.read_parquet('data_raw/AAPL.parquet')

# Create feature engineer with preset
fe = EnhancedFinancialFeatures(feature_preset='balanced')

# Show configuration
fe.print_config()

# Generate features
features = fe.create_all_features(df)
print(f"Created {len(features.columns)} features")
```

### Method 2: Custom Configuration

```python
from feature_config import FeatureConfig
from fin_feature_preprocessing import EnhancedFinancialFeatures

# Start with minimal preset and add features
config = FeatureConfig.create_custom(
    'minimal',
    entropy=True,          # Add entropy features
    microstructure=True    # Add microstructure features
)

fe = EnhancedFinancialFeatures(feature_config=config)
features = fe.create_all_features(df)
```

### Method 3: Manual Configuration

```python
# Fully custom configuration
manual_config = {
    'returns': True,
    'volatility': True,
    'momentum': True,
    'trend': False,
    'volume': True,
    'bollinger': False,
    'price_position': True,
    'microstructure': False,
    'statistical': False,
    'entropy': False,
    'regime': False,
    'risk_adjusted': False
}

fe = EnhancedFinancialFeatures(feature_config=manual_config)
```

## Available Presets

### 1. **minimal** (~72 features)
- **Use Case**: Initial experiments, quick iterations, baseline models
- **Features**: Core returns, volatility, momentum, trend, volume, bollinger, price position
- **Training Time**: Fast
- **Recommendation**: Start here for first runs

### 2. **balanced** (~98 features) ⭐ RECOMMENDED
- **Use Case**: Production models, good signal-to-noise ratio
- **Features**: All minimal + statistical, regime, risk-adjusted
- **Training Time**: Medium
- **Recommendation**: Best balance for most use cases

### 3. **comprehensive** (108 features)
- **Use Case**: Research, feature importance analysis, experimentation
- **Features**: All features enabled
- **Training Time**: Slowest
- **Recommendation**: Use for research and feature selection studies

### 4. **lightweight** (~56 features)
- **Use Case**: High-frequency trading, low-latency requirements
- **Features**: Only fastest-computing features
- **Training Time**: Very fast
- **Recommendation**: When speed is critical

### 5. **technical_only** (~72 features)
- **Use Case**: Traditional technical analysis approach
- **Features**: Standard technical indicators only
- **Training Time**: Fast
- **Recommendation**: Baseline technical strategy

### 6. **quant_research** (108 features)
- **Use Case**: Academic research, strategy development
- **Features**: All features including advanced metrics
- **Training Time**: Slowest
- **Recommendation**: For thorough research projects

## Feature Groups

### Critical (22 features)

| Group | Count | Description | Importance |
|-------|-------|-------------|------------|
| **returns** | 10 | Log returns, forward returns, acceleration | CRITICAL |
| **volatility** | 12 | Yang-Zhang, Parkinson, Garman-Klass, realized vol | CRITICAL |

### High Priority (49 features)

| Group | Count | Description | Importance |
|-------|-------|-------------|------------|
| **momentum** | 17 | RSI, MACD, Stochastic, Williams %R, CCI | HIGH |
| **trend** | 10 | ADX, ROC, SAR, Aroon, moving averages | HIGH |
| **volume** | 13 | OBV, VWAP, A/D, CMF | HIGH |

### Medium Priority (37 features)

| Group | Count | Description | Importance |
|-------|-------|-------------|------------|
| **bollinger** | 5 | Bollinger Bands and position indicators | MEDIUM |
| **price_position** | 8 | Distance from MA, highs/lows | MEDIUM |
| **microstructure** | 10 | Spreads, liquidity, order flow, VPIN | MEDIUM |
| **statistical** | 8 | Skewness, kurtosis, z-scores | MEDIUM |
| **regime** | 8 | Volatility regime, market state | MEDIUM |
| **risk_adjusted** | 8 | Sharpe, Sortino, Calmar ratios | MEDIUM |

### Low Priority (4 features)

| Group | Count | Description | Importance |
|-------|-------|-------------|------------|
| **entropy** | 4 | Shannon entropy, LZ complexity, variance ratio | LOW |

## Usage in Training Script

```python
# In fin_training.py

from fin_feature_preprocessing import EnhancedFinancialFeatures

# Option 1: Use balanced preset (recommended for first run)
feature_engineer = EnhancedFinancialFeatures(feature_preset='balanced')

# Option 2: Start minimal and scale up
feature_engineer = EnhancedFinancialFeatures(feature_preset='minimal')

# Option 3: Full features for research
feature_engineer = EnhancedFinancialFeatures(feature_preset='comprehensive')

# Process data
for ticker in tickers:
    df = pd.read_parquet(f'data_raw/{ticker}.parquet')
    features = feature_engineer.create_all_features(df)
```

## Utility Functions

### List All Presets

```python
from feature_config import FeatureConfig

FeatureConfig.list_presets()
```

### Describe a Feature Group

```python
FeatureConfig.describe_group('momentum')
```

### Print Current Configuration

```python
fe = EnhancedFinancialFeatures(feature_preset='balanced')
fe.print_config()
```

### Get Configuration Info

```python
info = fe.get_enabled_features_info()
print(f"Enabled groups: {info['enabled_groups']}")
print(f"Feature count: {info['estimated_feature_count']}")
```

## Recommendations by Use Case

### First Training Run
```python
feature_engineer = EnhancedFinancialFeatures(feature_preset='minimal')
```
- Fastest iteration
- Lower overfitting risk
- Good baseline

### Production Deployment
```python
feature_engineer = EnhancedFinancialFeatures(feature_preset='balanced')
```
- Proven feature set
- Good performance/complexity tradeoff
- Excludes noisy features (microstructure, entropy)

### Research & Feature Selection
```python
feature_engineer = EnhancedFinancialFeatures(feature_preset='comprehensive')
```
- All features for analysis
- Run feature importance studies
- Identify best features for your data

### High-Frequency Trading
```python
feature_engineer = EnhancedFinancialFeatures(feature_preset='lightweight')
```
- Fast computation
- Low latency
- Essential features only

## Demo Script

Run the demo to see all options:

```bash
python demo_feature_config.py
```

## Next Steps

1. **Start with `minimal`** for your first training run
2. **Monitor feature importance** in your model
3. **Gradually add feature groups** that show value
4. **Create custom preset** based on your findings
5. **Document your optimal config** for production

## Files

- `feature_config.py` - Configuration system and presets
- `fin_feature_preprocessing.py` - Feature engineering (updated to use config)
- `demo_feature_config.py` - Demo script showing usage
- `README_FEATURE_CONFIG.md` - This file
