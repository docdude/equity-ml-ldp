# Equity ML with López de Prado Methods

**Advanced machine learning framework for equity trading using López de Prado's methodologies**

A comprehensive implementation of quantitative trading strategies using López de Prado's *Advances in Financial Machine Learning* techniques, including triple-barrier labeling, sample weighting, feature importance analysis, and meta-labeling.

## 🎯 Features

### Core Functionality
- **Triple Barrier Labeling** - Dynamic volatility-based barriers with MLFinLab integration
- **Sample Weighting** - López de Prado uniqueness and return attribution weights
- **Feature Engineering** - 100+ technical indicators and microstructure features
- **Market Context** - Configurable market features (SPY, VIX, treasuries, gold, forex)
- **WaveNet Architecture** - Temporal convolutional network for sequence prediction
- **Feature Importance** - MDI, MDA, and orthogonalization analysis
- **Meta-Labeling** - Two-stage prediction for bet sizing
- **Kelly Positioning** - Bayesian Kelly criterion for optimal position sizing
- **PBO Analysis** - Probability of backtest overfitting validation

### Recent Additions
- ✅ Configurable market feature selection
- ✅ Feature importance analysis notebook
- ✅ Comprehensive meta-labeling documentation
- ✅ Kelly positioning integration
- ✅ Heatmap-based training pipeline

## 📁 Project Structure

```
equity-ml-ldp/
├── fin_training_ldp.py           # Main training pipeline with LdP methods
├── fin_training_heatmap.py       # Heatmap-augmented training
├── fin_load_and_sequence.py      # Feature loading and sequencing
├── fin_feature_preprocessing.py  # Feature engineering (100+ features)
├── fin_market_context.py         # Market context feature creation
├── fin_model.py                  # WaveNet model architecture
├── fin_model_evaluation.py       # Model evaluation and metrics
├── feature_config.py             # Feature configuration system
├── equity_feature_importance_analysis.ipynb  # Feature importance notebook
├── requirements.txt              # Python dependencies
│
├── MLFinance/                    # MLFinLab integration
│   └── FinancialMachineLearning/
│       ├── labeling/             # Triple barrier methods
│       ├── feature_importance/   # MDI, MDA, orthogonalization
│       ├── cross_validation/     # Purged K-Fold CV
│       ├── sample_weights/       # Uniqueness and attribution
│       └── bet_sizing/           # Kelly criterion
│
├── kelly_positioning/            # Kelly positioning system
│   ├── bayesian_kelly.py        # Bayesian Kelly implementation
│   ├── position_sizing_kelly.py # Enhanced Kelly with ML
│   └── kelly_backtest_integration.py
│
├── docs/                         # Documentation
│   ├── METALABELING_EXPLANATION.md
│   ├── KELLY_VS_METALABELING_ANALYSIS.md
│   ├── FIN_TRAINING_LDP_GUIDE.md
│   └── WAVENET_V2_USAGE.md
│
├── artifacts/                    # Model outputs and results
├── cache/                        # Cached sequences
├── data_raw/                     # Raw market data
├── models/                       # Saved models
└── config/                       # Configuration files
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/equity-ml-ldp.git
cd equity-ml-ldp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Train Model with Triple Barrier Labeling

```python
python fin_training_ldp.py \
    --tickers AAPL NVDA MSFT \
    --epochs 50 \
    --market-features spy vix
```

#### 2. Feature Importance Analysis

```bash
jupyter notebook equity_feature_importance_analysis.ipynb
```

#### 3. Test Market Feature Selection

```python
python test_market_selection.py
```

## 📊 Key Components

### Triple Barrier Labeling

```python
from fin_load_and_sequence import load_or_generate_sequences

X, y, dates, ldp_weights, returns = load_or_generate_sequences(
    tickers=['AAPL', 'NVDA'],
    barrier_params={
        'pt_sl': [2, 1],      # Asymmetric barriers
        'min_ret': 0.005,     # 0.5% minimum return
        'num_days': 7,        # 7-day max holding
        'lookback': 60        # 60-day volatility
    },
    market_features=['spy', 'vix'],  # Select market features
    use_cache=True
)
```

### Feature Configuration

```python
from feature_config import FeatureConfig

# Use preset configuration
config = FeatureConfig.get_preset('wavenet_optimized')

# Or customize
config = FeatureConfig.get_preset('balanced')
config['entropy'] = True      # Enable entropy features
config['microstructure'] = True  # Enable microstructure
```

### Market Feature Selection

```python
# Options: None, ['spy', 'vix'], or all 6 features
market_features = ['spy', 'vix']  # Minimal market context

# Available features:
# - 'spy': S&P 500 returns
# - 'vix': Volatility index
# - 'fvx': 5-year treasury yields
# - 'tyx': 30-year treasury yields
# - 'gold': Gold returns
# - 'jpyx': JPY/USD forex
```

## 📈 Model Architecture

### WaveNet Temporal Convolutional Network

- **Dilated convolutions** for exponential receptive field growth
- **Residual connections** for gradient flow
- **Causal padding** to prevent look-ahead bias
- **Spatial dropout** for regularization
- **López de Prado sample weighting** during training

### Feature Groups (108 total features)

| Group | Count | Description |
|-------|-------|-------------|
| Returns | 10 | Log returns, forward returns, acceleration |
| Volatility | 12 | Yang-Zhang, Parkinson, GK, realized vol |
| Momentum | 17 | RSI, MACD, Stochastic, Williams %R |
| Trend | 10 | ADX, ROC, SAR, Aroon, moving averages |
| Volume | 13 | OBV, VWAP, A/D, CMF, volume indicators |
| Bollinger | 5 | Bands, position, width indicators |
| Price Position | 8 | Distance from highs/lows, price levels |
| Microstructure | 10 | Spreads, liquidity, order flow, VPIN |
| Statistical | 8 | Skewness, kurtosis, z-scores |
| Entropy | 4 | Shannon entropy, LZ complexity |
| Regime | 8 | Volatility regime, market state |
| Risk-Adjusted | 8 | Sharpe, Sortino, Calmar ratios |

## 🔬 Advanced Features

### Meta-Labeling

Two-stage prediction framework:

```python
# Stage 1: Primary model (LightGBM) predicts direction
lgbm_signals = load_lightgbm_predictions()  # {-1, +1}

# Stage 2: WaveNet predicts confidence
events = get_events(..., side_prediction=lgbm_signals)
meta_labels = meta_labeling(events, prices)  # {0, 1}
wavenet.fit(X_features, meta_labels['bin'])
```

### Kelly Position Sizing

```python
from kelly_positioning.position_sizing_kelly import EnhancedKellyPositionSizer

kelly_sizer = EnhancedKellyPositionSizer(
    fractional_kelly=0.25,  # LdP recommendation
    max_leverage=1.0
)

position_size = kelly_sizer.calculate_position_size(
    returns=historical_returns,
    ml_score=confidence_from_wavenet,  # Connect ML prediction
    volatility=current_volatility
)
```

### Feature Importance (MDI/MDA)

```python
from FinancialMachineLearning.feature_importance.importance import (
    mean_decrease_impurity,
    mean_decrease_accuracy
)

# In-sample importance
mdi = mean_decrease_impurity(forest, feature_names)

# Out-of-sample importance with purged CV
mda = mean_decrease_accuracy(
    forest, X, y, cv_gen,
    sample_weight=ldp_weights,
    scoring=accuracy_score
)
```

## 📚 Documentation

- [Meta-Labeling Explanation](METALABELING_EXPLANATION.md) - Complete theory and implementation
- [Kelly vs Meta-Labeling](KELLY_VS_METALABELING_ANALYSIS.md) - Understanding the relationship
- [Training Guide](docs/FIN_TRAINING_LDP_GUIDE.md) - Detailed training documentation
- [WaveNet Usage](WAVENET_V2_USAGE.md) - Model architecture and configuration
- [Market Features](MARKET_FEATURES_ADDED.md) - Market context integration

## 🧪 Testing

```bash
# Test market feature selection
python test_market_selection.py

# Test feature engineering
python test_features.py

# Test barrier labeling
python test_barriers_visualization.ipynb

# Test PBO analysis
python test_pbo.py
```

## 📊 Results

### Performance Metrics

- **PBO Score**: 30.19% (good - below 50% threshold)
- **Sharpe Ratio Range**: 0.86 - 1.26 (profitable strategies)
- **Label Distribution**: Balanced after triple-barrier implementation
- **Feature Importance**: 108 features → ~20 robust features identified

### Key Findings

1. **Market features impact**: Minimal market context (SPY + VIX) often outperforms all 6 features
2. **Sample weighting**: López de Prado weights significantly improve generalization
3. **Triple barriers**: Dynamic volatility-based barriers improve label alignment
4. **Feature correlation**: High multicollinearity requires orthogonalization

## 🔧 Configuration

### Barrier Parameters

```yaml
barrier_params:
  pt_sl: [2, 1]        # Profit-taking: 2×σ, Stop-loss: 1×σ
  min_ret: 0.005       # 0.5% minimum return threshold
  num_days: 7          # 7-day maximum holding period
  lookback: 60         # 60-day volatility lookback
```

### Model Hyperparameters

```yaml
model:
  seq_len: 20          # Sequence length
  filters: 32          # Convolutional filters
  kernel_size: 3       # Kernel size
  dilation_rates: [1, 2, 4, 8, 16]
  dropout: 0.2         # Spatial dropout
  learning_rate: 0.001
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Kelly, J. L. (1956). "A New Interpretation of Information Rate"

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MLFinLab for López de Prado implementations
- Hudson & Thames for quantitative finance tools
- The quantitative finance community for continuous research

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Create an issue](https://github.com/yourusername/equity-ml-ldp/issues)
- Email: your.email@example.com

---

**Note**: This is a research and educational project. Not financial advice. Trade at your own risk.
