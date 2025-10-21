"""
Enhanced Position Sizing Module with Kelly Criterion
Incorporates principles from Marcos Lopez de Prado and Bailey's research
Designed for ML-driven equity trading with proper OOS validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

@dataclass
class KellyConfig:
    """Configuration for Kelly position sizing"""
    method: str = 'bayesian'  # 'classical', 'bayesian', 'empirical', 'combinatorial'
    kelly_fraction: float = 0.25  # Fractional Kelly (Lopez de Prado recommends 0.25)
    
    # Bayesian parameters
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    decay_factor: float = 0.94  # For exponential weighting of observations
    
    # Risk constraints
    max_position_size: float = 0.25  # Maximum 25% per position
    min_position_size: float = 0.01  # Minimum 1% per position
    max_leverage: float = 1.0  # No leverage by default
    
    # Sample size adjustments (Lopez de Prado)
    min_sample_size: int = 100  # Minimum observations for Kelly
    confidence_level: float = 0.95  # For confidence intervals
    
    # Sharpe ratio constraints
    min_sharpe_threshold: float = 0.5
    
    # Detrending (Bailey & Lopez de Prado)
    detrend_returns: bool = True
    
    # Strategy specific
    use_ml_signals: bool = True
    ml_confidence_weight: bool = True  # Weight by ML confidence score
    
    # Regime awareness
    regime_adjust: bool = True
    vol_lookback: int = 60

@dataclass
class PositionSizeResult:
    """Results from position sizing calculation"""
    position_size: float
    kelly_raw: float
    kelly_adjusted: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    regime: str
    method_used: str
    adjustments_applied: List[str]

class EnhancedKellyPositionSizer:
    """
    Advanced position sizing using Kelly Criterion with ML integration
    Implements Lopez de Prado and Bailey's improvements
    """
    
    def __init__(self, config: KellyConfig = None):
        self.config = config or KellyConfig()
        self.returns_history = []
        self.ml_scores_history = []
        self.position_history = []
        self.performance_history = []
        
        # Bayesian state
        self.alpha = self.config.alpha_prior
        self.beta = self.config.beta_prior
        
    def calculate_position_size(
        self,
        returns: pd.Series,
        ml_score: float = None,
        current_price: float = None,
        features: pd.DataFrame = None,
        method_override: str = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using enhanced Kelly criterion
        
        Args:
            returns: Historical returns series
            ml_score: ML model confidence score (0-1)
            current_price: Current asset price
            features: Additional features for regime detection
            method_override: Override configured method
            
        Returns:
            PositionSizeResult with position sizing details
        """
        method = method_override or self.config.method
        
        # Validate inputs
        if len(returns) < self.config.min_sample_size:
            return self._safe_position_size(
                reason=f"Insufficient samples: {len(returns)} < {self.config.min_sample_size}"
            )
        
        # Preprocess returns (Lopez de Prado's detrending)
        processed_returns = self._preprocess_returns(returns)
        
        # Calculate base Kelly
        if method == 'bayesian':
            kelly_result = self._bayesian_kelly(processed_returns, ml_score)
        elif method == 'empirical':
            kelly_result = self._empirical_kelly(processed_returns)
        elif method == 'combinatorial':
            kelly_result = self._combinatorial_kelly(processed_returns)
        else:  # classical
            kelly_result = self._classical_kelly(processed_returns)
        
        # Apply adjustments
        adjustments = []
        kelly_adjusted = kelly_result['kelly']
        
        # 1. Sample size adjustment (Bailey & Lopez de Prado)
        sample_adjustment = self._sample_size_adjustment(
            len(processed_returns),
            kelly_result['sharpe']
        )
        kelly_adjusted *= sample_adjustment
        if sample_adjustment < 1.0:
            adjustments.append(f"sample_size_{sample_adjustment:.2f}")
        
        # 2. Confidence interval adjustment
        ci_lower, ci_upper = self._calculate_confidence_interval(
            processed_returns,
            kelly_result
        )
        if kelly_adjusted > ci_upper:
            kelly_adjusted = ci_upper
            adjustments.append("ci_capped")
        
        # 3. ML signal adjustment
        if self.config.use_ml_signals and ml_score is not None:
            ml_adjustment = self._ml_signal_adjustment(ml_score)
            kelly_adjusted *= ml_adjustment
            if ml_adjustment != 1.0:
                adjustments.append(f"ml_signal_{ml_adjustment:.2f}")
        
        # 4. Regime adjustment
        regime = 'normal'
        if self.config.regime_adjust and features is not None:
            regime = self._detect_regime(returns, features)
            regime_mult = self._regime_adjustment(regime)
            kelly_adjusted *= regime_mult
            if regime_mult != 1.0:
                adjustments.append(f"regime_{regime}_{regime_mult:.2f}")
        
        # 5. Apply fractional Kelly
        kelly_final = kelly_adjusted * self.config.kelly_fraction
        
        # 6. Apply position limits
        position_size = np.clip(
            kelly_final,
            self.config.min_position_size,
            self.config.max_position_size
        )
        
        if position_size != kelly_final:
            adjustments.append(f"clipped_{position_size:.3f}")
        
        return PositionSizeResult(
            position_size=position_size,
            kelly_raw=kelly_result['kelly'],
            kelly_adjusted=kelly_adjusted,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(processed_returns),
            sharpe_ratio=kelly_result['sharpe'],
            win_rate=kelly_result['win_rate'],
            avg_win=kelly_result['avg_win'],
            avg_loss=kelly_result['avg_loss'],
            regime=regime,
            method_used=method,
            adjustments_applied=adjustments
        )
    
    def _preprocess_returns(self, returns: pd.Series) -> pd.Series:
        """Preprocess returns following Lopez de Prado's recommendations"""
        if self.config.detrend_returns:
            # Remove linear trend
            x = np.arange(len(returns))
            slope, intercept = np.polyfit(x, returns, 1)
            detrended = returns - (slope * x + intercept)
            return detrended
        return returns
    
    def _classical_kelly(self, returns: pd.Series) -> Dict:
        """Classical Kelly criterion calculation"""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return {
                'kelly': 0.0,
                'win_rate': len(wins) / len(returns),
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'sharpe': 0.0
            }
        
        p = len(wins) / len(returns)  # Win probability
        q = 1 - p  # Loss probability
        b = wins.mean() / abs(losses.mean())  # Win/loss ratio
        
        # Kelly formula: f = (p*b - q) / b
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Calculate Sharpe
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'kelly': max(0, kelly),
            'win_rate': p,
            'avg_win': wins.mean(),
            'avg_loss': abs(losses.mean()),
            'sharpe': sharpe
        }
    
    def _bayesian_kelly(self, returns: pd.Series, ml_score: float = None) -> Dict:
        """
        Bayesian Kelly with conjugate priors
        Inspired by your existing Bayesian approach
        """
        # Update beliefs with exponential decay
        for i, ret in enumerate(returns):
            weight = self.config.decay_factor ** (len(returns) - i - 1)
            if ret > 0:
                self.alpha += weight
            else:
                self.beta += weight
        
        # Current win probability (posterior mean)
        p = self.alpha / (self.alpha + self.beta)
        
        # If ML score available, blend with Bayesian estimate
        if ml_score is not None and self.config.ml_confidence_weight:
            # Weighted average based on sample size
            weight = min(len(returns) / 500, 0.8)  # Max 80% weight on historical
            p = weight * p + (1 - weight) * ml_score
        
        # Calculate win/loss statistics
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return self._classical_kelly(returns)  # Fallback
        
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        # Bayesian Kelly formula
        q = 1 - p
        kelly = (p * avg_win - q * avg_loss) / (avg_win * avg_loss) if avg_win * avg_loss > 0 else 0
        
        # Calculate Sharpe
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'kelly': max(0, kelly),
            'win_rate': p,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe': sharpe
        }
    
    def _empirical_kelly(self, returns: pd.Series) -> Dict:
        """
        Empirical Kelly using maximum likelihood estimation
        As recommended by Lopez de Prado for small samples
        """
        def negative_log_likelihood(f):
            """Negative log-likelihood for optimization"""
            portfolio_returns = returns * f
            log_returns = np.log(1 + portfolio_returns)
            return -log_returns.mean()
        
        # Find optimal f using MLE
        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(0, 1),
            method='bounded'
        )
        
        kelly = result.x if result.success else 0.0
        
        # Calculate statistics
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        return {
            'kelly': kelly,
            'win_rate': len(wins) / len(returns) if len(returns) > 0 else 0,
            'avg_win': wins.mean() if len(wins) > 0 else 0,
            'avg_loss': abs(losses.mean()) if len(losses) > 0 else 0,
            'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        }
    
    def _combinatorial_kelly(self, returns: pd.Series) -> Dict:
        """
        Combinatorial purged cross-validation Kelly (Lopez de Prado)
        More robust for non-IID returns
        """
        n_splits = min(5, len(returns) // 20)
        kelly_estimates = []
        
        for i in range(n_splits):
            # Create train/test split with purging
            test_start = i * len(returns) // n_splits
            test_end = (i + 1) * len(returns) // n_splits
            purge_size = 5  # Purge 5 observations around test set
            
            train_idx = np.concatenate([
                np.arange(0, max(0, test_start - purge_size)),
                np.arange(min(len(returns), test_end + purge_size), len(returns))
            ])
            
            if len(train_idx) > 10:
                train_returns = returns.iloc[train_idx]
                kelly_fold = self._classical_kelly(train_returns)
                kelly_estimates.append(kelly_fold['kelly'])
        
        if kelly_estimates:
            # Use median for robustness
            kelly = np.median(kelly_estimates)
        else:
            kelly = 0.0
        
        # Calculate overall statistics
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        return {
            'kelly': kelly,
            'win_rate': len(wins) / len(returns) if len(returns) > 0 else 0,
            'avg_win': wins.mean() if len(wins) > 0 else 0,
            'avg_loss': abs(losses.mean()) if len(losses) > 0 else 0,
            'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        }
    
    def _sample_size_adjustment(self, n_samples: int, sharpe: float) -> float:
        """
        Adjust Kelly based on sample size (Bailey & Lopez de Prado)
        Small samples tend to overestimate Kelly
        """
        # Minimum samples for full Kelly
        min_samples_full = 252 * 3  # 3 years of daily data
        
        # Linear scaling up to min_samples_full
        sample_factor = min(1.0, n_samples / min_samples_full)
        
        # Additional Sharpe-based adjustment
        # Low Sharpe ratios need more samples
        sharpe_factor = 1.0
        if sharpe < self.config.min_sharpe_threshold:
            sharpe_factor = sharpe / self.config.min_sharpe_threshold
        
        return sample_factor * sharpe_factor
    
    def _calculate_confidence_interval(
        self,
        returns: pd.Series,
        kelly_result: Dict
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for Kelly estimate
        Using bootstrap or analytical methods
        """
        n_bootstrap = 1000
        kelly_samples = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_returns = returns.sample(len(returns), replace=True)
            kelly_boot = self._classical_kelly(sample_returns)
            kelly_samples.append(kelly_boot['kelly'])
        
        # Calculate confidence interval
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(kelly_samples, alpha/2 * 100)
        ci_upper = np.percentile(kelly_samples, (1 - alpha/2) * 100)
        
        return (max(0, ci_lower), ci_upper)
    
    def _ml_signal_adjustment(self, ml_score: float) -> float:
        """
        Adjust Kelly based on ML model confidence
        Higher confidence -> closer to full Kelly
        """
        if ml_score < 0.5:
            # Bearish signal - reduce or invert
            return 0.0
        elif ml_score < 0.55:
            # Weak signal
            return 0.5
        elif ml_score < 0.65:
            # Moderate signal
            return 0.75
        elif ml_score < 0.75:
            # Strong signal
            return 1.0
        else:
            # Very strong signal - can go slightly above 1
            return min(1.2, 1 + (ml_score - 0.75))
    
    def _detect_regime(self, returns: pd.Series, features: pd.DataFrame) -> str:
        """
        Detect market regime for position sizing adjustment
        """
        # Calculate recent volatility
        recent_vol = returns.tail(20).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        
        # Volatility regime
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Trend regime (if available in features)
        trend_score = 0
        if 'mkt_trend_20_50' in features.columns:
            trend_score = features['mkt_trend_20_50'].iloc[-1]
        
        # Classify regime
        if vol_ratio > 1.5:
            return 'high_volatility'
        elif vol_ratio < 0.7:
            return 'low_volatility'
        elif trend_score > 0.7:
            return 'strong_trend'
        elif trend_score < 0.3:
            return 'mean_reverting'
        else:
            return 'normal'
    
    def _regime_adjustment(self, regime: str) -> float:
        """
        Adjust Kelly fraction based on market regime
        """
        adjustments = {
            'high_volatility': 0.5,    # Reduce in high vol
            'low_volatility': 1.2,     # Increase in low vol
            'strong_trend': 1.1,       # Slight increase in trends
            'mean_reverting': 0.8,     # Reduce in choppy markets
            'normal': 1.0
        }
        return adjustments.get(regime, 1.0)
    
    def _safe_position_size(self, reason: str = "") -> PositionSizeResult:
        """Return minimal safe position when Kelly cannot be calculated"""
        return PositionSizeResult(
            position_size=self.config.min_position_size,
            kelly_raw=0.0,
            kelly_adjusted=0.0,
            confidence_interval=(0.0, 0.0),
            sample_size=0,
            sharpe_ratio=0.0,
            win_rate=0.5,
            avg_win=0.0,
            avg_loss=0.0,
            regime='unknown',
            method_used='safe_minimum',
            adjustments_applied=[f"safe_mode: {reason}"]
        )
    
    def calculate_portfolio_allocation(
        self,
        signals: pd.DataFrame,
        returns_dict: Dict[str, pd.Series],
        features_dict: Dict[str, pd.DataFrame] = None,
        capital: float = 100000
    ) -> pd.DataFrame:
        """
        Calculate position sizes for multiple signals
        
        Args:
            signals: DataFrame with columns ['ticker', 'ml_score', 'date']
            returns_dict: Dictionary of returns series by ticker
            features_dict: Optional dictionary of features by ticker
            capital: Total capital to allocate
            
        Returns:
            DataFrame with position sizing for each signal
        """
        allocations = []
        
        for _, signal in signals.iterrows():
            ticker = signal['ticker']
            ml_score = signal.get('ml_score', None)
            
            if ticker not in returns_dict:
                continue
            
            returns = returns_dict[ticker]
            features = features_dict.get(ticker) if features_dict else None
            
            # Calculate position size
            result = self.calculate_position_size(
                returns=returns,
                ml_score=ml_score,
                features=features
            )
            
            allocation = {
                'ticker': ticker,
                'date': signal.get('date'),
                'ml_score': ml_score,
                'position_pct': result.position_size,
                'position_dollars': capital * result.position_size,
                'kelly_raw': result.kelly_raw,
                'kelly_adjusted': result.kelly_adjusted,
                'confidence_lower': result.confidence_interval[0],
                'confidence_upper': result.confidence_interval[1],
                'sharpe': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'regime': result.regime,
                'adjustments': ','.join(result.adjustments_applied)
            }
            allocations.append(allocation)
        
        allocations_df = pd.DataFrame(allocations)
        
        # Normalize if total allocation exceeds 100%
        total_allocation = allocations_df['position_pct'].sum()
        if total_allocation > 1.0:
            allocations_df['position_pct'] *= 1.0 / total_allocation
            allocations_df['position_dollars'] = capital * allocations_df['position_pct']
        
        return allocations_df

class KellyBacktestIntegration:
    """
    Integration module for Kelly position sizing with your backtest framework
    Designed for 10yr training, 1yr OOS signals, 6-12mo backtest
    """
    
    def __init__(self, config: KellyConfig = None):
        self.config = config or KellyConfig()
        self.position_sizer = EnhancedKellyPositionSizer(config)
    
    def prepare_backtest_data(
        self,
        train_data: pd.DataFrame,      # 10 years training
        oos_data: pd.DataFrame,         # 1 year OOS
        ml_predictions: pd.DataFrame,  # ML signals on OOS
        backtest_start: str,           # Start of backtest period
        backtest_end: str              # End of backtest period
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for backtesting with Kelly position sizing
        
        Returns:
            Tuple of (signals_with_sizing, backtest_data)
        """
        # Filter to backtest period
        backtest_mask = (ml_predictions['date'] >= backtest_start) & \
                       (ml_predictions['date'] <= backtest_end)
        backtest_signals = ml_predictions[backtest_mask].copy()
        
        # Calculate returns for each ticker using training + OOS history
        returns_dict = {}
        features_dict = {}
        
        for ticker in backtest_signals['ticker'].unique():
            # Combine train and OOS data for this ticker
            ticker_train = train_data[train_data['ticker'] == ticker]
            ticker_oos = oos_data[oos_data['ticker'] == ticker]
            ticker_data = pd.concat([ticker_train, ticker_oos])
            
            # Calculate returns
            ticker_data = ticker_data.sort_values('date')
            ticker_data['returns'] = ticker_data['close'].pct_change()
            
            # Store returns up to each signal date
            returns_dict[ticker] = ticker_data.set_index('date')['returns']
            
            # Store features if available
            feature_cols = [col for col in ticker_data.columns 
                          if col not in ['date', 'ticker', 'open', 'high', 
                                       'low', 'close', 'volume', 'returns']]
            if feature_cols:
                features_dict[ticker] = ticker_data.set_index('date')[feature_cols]
        
        # Calculate position sizes for each signal
        allocations = []
        
        for date in backtest_signals['date'].unique():
            date_signals = backtest_signals[backtest_signals['date'] == date]
            
            # Get returns up to this date for each ticker
            date_returns_dict = {}
            date_features_dict = {}
            
            for ticker in date_signals['ticker'].unique():
                if ticker in returns_dict:
                    # Use data up to current date (no lookahead)
                    historical_returns = returns_dict[ticker][
                        returns_dict[ticker].index < date
                    ].tail(252 * 3)  # Use up to 3 years
                    
                    date_returns_dict[ticker] = historical_returns
                    
                    if ticker in features_dict:
                        date_features_dict[ticker] = features_dict[ticker][
                            features_dict[ticker].index <= date
                        ].tail(1)
            
            # Calculate allocations for this date
            if date_returns_dict:
                date_allocations = self.position_sizer.calculate_portfolio_allocation(
                    signals=date_signals,
                    returns_dict=date_returns_dict,
                    features_dict=date_features_dict if date_features_dict else None
                )
                date_allocations['date'] = date
                allocations.append(date_allocations)
        
        if allocations:
            all_allocations = pd.concat(allocations, ignore_index=True)
        else:
            all_allocations = pd.DataFrame()
        
        return all_allocations, backtest_signals
    
    def create_backtrader_strategy(self):
        """
        Create a Backtrader strategy that uses Kelly position sizing
        """
        import backtrader as bt
        
        class KellyStrategy(bt.Strategy):
            params = (
                ('kelly_config', None),
                ('position_data', None),
                ('printlog', False),
            )
            
            def __init__(self):
                self.position_sizer = EnhancedKellyPositionSizer(
                    self.params.kelly_config
                )
                self.position_data = self.params.position_data
                self.current_positions = {}
            
            def next(self):
                current_date = self.datas[0].datetime.date(0)
                
                # Get today's signals
                today_positions = self.position_data[
                    self.position_data['date'] == current_date
                ]
                
                if not today_positions.empty:
                    # Close positions not in today's signals
                    for data in self.datas:
                        if self.getposition(data).size != 0:
                            if data._name not in today_positions['ticker'].values:
                                self.close(data=data)
                    
                    # Open/adjust positions based on Kelly sizing
                    for _, position in today_positions.iterrows():
                        ticker = position['ticker']
                        size_pct = position['position_pct']
                        
                        # Find the data feed for this ticker
                        for data in self.datas:
                            if data._name == ticker:
                                target_value = self.broker.getvalue() * size_pct
                                target_size = int(target_value / data.close[0])
                                
                                current_size = self.getposition(data).size
                                
                                if target_size != current_size:
                                    self.order_target_size(data=data, target=target_size)
                                
                                if self.params.printlog:
                                    print(f'{current_date}: {ticker} Kelly size: {size_pct:.2%}')
                                break
        
        return KellyStrategy
    
    def analyze_position_sizes(self, allocations: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the distribution and characteristics of Kelly position sizes
        """
        analysis = {
            'summary_statistics': {
                'mean_position_size': allocations['position_pct'].mean(),
                'std_position_size': allocations['position_pct'].std(),
                'median_position_size': allocations['position_pct'].median(),
                'max_position_size': allocations['position_pct'].max(),
                'min_position_size': allocations['position_pct'].min(),
            },
            'kelly_statistics': {
                'mean_kelly_raw': allocations['kelly_raw'].mean(),
                'mean_kelly_adjusted': allocations['kelly_adjusted'].mean(),
                'kelly_reduction_factor': allocations['kelly_adjusted'].mean() / 
                                         (allocations['kelly_raw'].mean() + 1e-10),
            },
            'risk_metrics': {
                'mean_sharpe': allocations['sharpe'].mean(),
                'mean_win_rate': allocations['win_rate'].mean(),
                'positions_at_max': (allocations['position_pct'] >= 
                                   self.config.max_position_size).mean(),
                'positions_at_min': (allocations['position_pct'] <= 
                                   self.config.min_position_size * 1.1).mean(),
            },
            'regime_distribution': allocations['regime'].value_counts().to_dict(),
            'adjustment_frequency': {
                adj: allocations['adjustments'].str.contains(adj).mean()
                for adj in ['sample_size', 'ci_capped', 'ml_signal', 'regime', 'clipped']
            }
        }
        
        return analysis

# Example usage function
def integrate_kelly_with_ml_pipeline(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    ml_predictions: pd.DataFrame,
    config_override: Dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Example integration with your ML pipeline
    
    Args:
        train_data: 10 years of training data
        test_data: 1 year of OOS data
        ml_predictions: ML model predictions on OOS data
        config_override: Optional config overrides
        
    Returns:
        Tuple of (position_sized_signals, analysis_results)
    """
    # Configure Kelly
    kelly_config = KellyConfig()
    if config_override:
        for key, value in config_override.items():
            setattr(kelly_config, key, value)
    
    # Initialize integration
    kelly_integration = KellyBacktestIntegration(kelly_config)
    
    # Define backtest period (last 6 months of OOS)
    backtest_end = test_data['date'].max()
    backtest_start = backtest_end - pd.Timedelta(days=180)
    
    # Prepare data with Kelly sizing
    allocations, signals = kelly_integration.prepare_backtest_data(
        train_data=train_data,
        oos_data=test_data,
        ml_predictions=ml_predictions,
        backtest_start=str(backtest_start),
        backtest_end=str(backtest_end)
    )
    
    # Analyze position sizes
    analysis = kelly_integration.analyze_position_sizes(allocations)
    
    print("Kelly Position Sizing Summary:")
    print(f"Average position size: {analysis['summary_statistics']['mean_position_size']:.2%}")
    print(f"Position size std dev: {analysis['summary_statistics']['std_position_size']:.2%}")
    print(f"Average Sharpe ratio: {analysis['risk_metrics']['mean_sharpe']:.2f}")
    print(f"Average win rate: {analysis['risk_metrics']['mean_win_rate']:.2%}")
    
    return allocations, analysis
