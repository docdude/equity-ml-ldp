# enhanced_preprocessing.py
"""
Enhanced Financial Feature Engineering
Based on López de Prado's "Advances in Financial Machine Learning"
"""

import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import talib
from numba import jit
from scipy import stats, signal

from feature_config import FeatureConfig

warnings.filterwarnings('ignore')


class EnhancedFinancialFeatures:
    """
    Comprehensive feature engineering for financial ML
    Implements features from AFML Chapter 5 and industry best practices
    
    Supports selective feature engineering via FeatureConfig system.
    """
    
    def __init__(self, feature_preset: Optional[str] = None, feature_config: Optional[dict] = None):
        """
        Initialize feature engineer with optional configuration.
        
        Args:
            feature_preset: Name of preset config ('minimal', 'balanced', 'comprehensive', etc.)
                          If None, uses all features.
            feature_config: Custom feature configuration dictionary.
                          If provided, overrides feature_preset.
        """
        self.feature_names = []
        self.scaler_params = {}
        
        # Set up feature configuration
        if feature_config is not None:
            self.feature_config = feature_config
        elif feature_preset is not None:
            self.feature_config = FeatureConfig.get_preset(feature_preset)
        else:
            # Default: all features enabled
            self.feature_config = FeatureConfig.get_default()
        
        # Validate configuration
        valid, errors = FeatureConfig.validate_config(self.feature_config)
        if not valid:
            raise ValueError(f"Invalid feature configuration: {errors}")
    
    def get_enabled_features_info(self) -> dict:
        """
        Get information about currently enabled features.
        
        Returns:
            Dictionary with enabled groups and estimated feature count
        """
        enabled_groups = [k for k, v in self.feature_config.items() if v]
        total_features = sum(
            FeatureConfig.FEATURE_GROUPS[group]['count'] 
            for group in enabled_groups
        )
        
        return {
            'enabled_groups': enabled_groups,
            'disabled_groups': [k for k, v in self.feature_config.items() if not v],
            'estimated_feature_count': total_features,
            'config': self.feature_config.copy()
        }
    
    def print_config(self):
        """Print current feature configuration."""
        FeatureConfig.print_config_summary(self.feature_config)
        
    def create_all_features(self, df: pd.DataFrame, feature_config: Optional[dict] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for deep learning (100+ features)
        
        Categories:
        1. Price-based (returns, volatility)
        2. Volume-based (VWAP, volume bars)
        3. Microstructure (Kyle's lambda, Amihud illiquidity)
        4. Technical indicators (entropy, fractal dimension)
        5. Regime indicators (structural breaks)
        """
        
        # Use provided config or instance config
        config = feature_config if feature_config is not None else self.feature_config
        
        print(f"  Engineering features for {len(df)} samples...")
        features = pd.DataFrame(index=df.index)
        
        # 1. RETURNS GROUP - CRITICAL
        if config['returns']:
            print("  → Returns (multi-horizon)")
            # Log returns at multiple horizons
            for horizon in [1, 2, 3, 5, 10, 20]:
                features[f'log_return_{horizon}d'] = np.log(df['Close'] / df['Close'].shift(horizon))
            
            # Return acceleration
            features['return_acceleration'] = features['log_return_1d'].diff()
            
            # Forward returns (for labeling/targets)
          #  for horizon in [1, 3, 5]:
           #     features[f'forward_return_{horizon}d'] = np.log(
            #        df['Close'].shift(-horizon) / df['Close']
            #    )
        
        # Compute returns once for reuse
        returns = df['Close'].pct_change()
        
        # 2. VOLATILITY GROUP - CRITICAL
        if config['volatility']:
            print("  → Volatility (Yang-Zhang, Parkinson, Garman-Klass, Rogers-Satchell, Hodges-Tompkins)")
            # Yang-Zhang volatility at multiple windows
            features['volatility_yz_10'] = self._yang_zhang_volatility(df, 10)
            features['volatility_yz_20'] = self._yang_zhang_volatility(df, 20)
            features['volatility_yz_60'] = self._yang_zhang_volatility(df, 60)
            
            # Parkinson volatility (uses high-low range)
            features['volatility_parkinson_10'] = self._parkinson_volatility(df, 10)
            features['volatility_parkinson_20'] = self._parkinson_volatility(df, 20)
            
            # Garman-Klass volatility
            features['volatility_gk_20'] = self._garman_klass_volatility(df, 20)
            
            # Simple close-to-close volatility (using log returns)
            features['volatility_cc_20'] = self._close_to_close_volatility(df, 20)
            features['volatility_cc_60'] = self._close_to_close_volatility(df, 60)
            
            # Rogers-Satchell volatility (assumes zero drift)
            features['volatility_rs_20'] = self._rogers_satchell_volatility(df, 20)
            features['volatility_rs_60'] = self._rogers_satchell_volatility(df, 60)
            
            # Hodges-Tompkins volatility (bias-corrected)
            features['volatility_ht_20'] = self._hodges_tompkins_volatility(df, 20)
            features['volatility_ht_60'] = self._hodges_tompkins_volatility(df, 60)
            
            # Volatility ratios
            # Note: Uses Yang-Zhang volatility (Rogers-Satchell variant)
            # 98%+ match with reference is expected due to formula variations
            features['vol_ratio_short_long'] = (
                features['volatility_yz_10'] / (features['volatility_yz_60'] + 1e-8)
            )
            
            # Rogers-Satchell vs Yang-Zhang ratio (captures intraday vs full-day volatility difference)
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
        
        # 3. MICROSTRUCTURE GROUP - Market microstructure and liquidity
        if config['microstructure']:
            print("  → Microstructure features")
            # Volume-synchronized probability of informed trading (VPIN)
            features['vpin'] = self._calculate_vpin(df)
            
            # Kyle's Lambda (price impact)
            features['kyle_lambda'] = self._kyle_lambda(df)
            
            # Amihud illiquidity measure
            features['amihud_illiquidity'] = np.abs(returns) / (df['Volume'] * df['Close'] + 1e-8)
            
            # High-low range (with winsorization + log transform to handle extreme volatility days)
            # Note: hl_range can spike to extreme values on volatile days (e.g., 0.16 = 16% range)
            # Without treatment, this creates outliers that survive even RobustScaler normalization
            # Strategy: Winsorize at 99th percentile, then log1p transform
            hl_range_raw = (df['High'] - df['Low']) / (df['Close'] + 1e-8)
            hl_range_p99 = hl_range_raw.quantile(0.99)
            hl_range_winsorized = hl_range_raw.clip(upper=hl_range_p99)  # Cap at 99th percentile
            features['hl_range'] = np.log1p(hl_range_winsorized)  # Log transform for mild skew
            features['hl_range_ma'] = features['hl_range'].rolling(20).mean()
            
            # Open-close range
            features['oc_range'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-8)
            
            # Roll's implied spread estimator
            features['roll_spread'] = self._roll_spread(df)
            
            # Corwin-Schultz bid-ask spread estimator
            features['cs_spread'] = self._corwin_schultz_spread(df)
            
            # High-low volatility ratio (indicates bid-ask bounce)
            if config['volatility']:  # Only if we have volatility features
                features['hl_volatility_ratio'] = (
                    features['volatility_parkinson_20'] / features['volatility_yz_20']
                )
            
            # Amihud illiquidity (20-day moving average)
            features['amihud_illiq'] = (
                np.abs(returns) / (df['Volume'] * df['Close'] + 1e-8)
            ).rolling(20).mean()
            
            # Order flow imbalance proxy (using volume and price direction)
            # Note: Symmetric version [-1, 1] showing net buy/sell pressure
            # (Standard version uses buy ratio [0, 1], but symmetric is more intuitive)
            features['order_flow_imbalance'] = self._order_flow_imbalance(df)
        
        # 4. VOLUME GROUP - Volume-based indicators
        if config['volume']:
            print("  → Volume-weighted features")
            # Volume rate of change
            features['volume_roc'] = df['Volume'].pct_change(5)
            
            # Dollar volume (use ratio, not raw values)
            dollar_volume = df['Volume'] * df['Close']
            features['dollar_volume_ma_ratio'] = dollar_volume / (dollar_volume.rolling(20).mean() + 1e-8)
            
            # Volume metrics
            features['volume_norm'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
            features['volume_zscore'] = (
                (df['Volume'] - df['Volume'].rolling(20).mean()) / 
                (df['Volume'].rolling(20).std() + 1e-8)
            )
            
            # VWAP
            features['vwap_20'] = (
                (df['Close'] * df['Volume']).rolling(20).sum() /
                (df['Volume'].rolling(20).sum() + 1e-8)
            )
            features['price_vwap_ratio'] = df['Close'] / (features['vwap_20'] + 1e-8)
            
            # On-Balance Volume (normalize by cumulative volume to make scale-invariant)
            obv_raw = talib.OBV(df['Close'], df['Volume'])
            cum_volume = df['Volume'].cumsum()
            # OBV normalized by total volume traded (scale-invariant across stocks)
            features['obv_normalized'] = obv_raw / (cum_volume + 1e-8)
            # OBV rate of change (momentum) - use log returns to avoid explosion
            features['obv_roc'] = np.log(obv_raw / (obv_raw.shift(20) + 1e-8))
            
            # Accumulation/Distribution (normalize by cumulative volume for scale-invariance)
            ad_raw = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
            cum_volume = df['Volume'].cumsum()
            # A/D normalized by total volume (scale-invariant)
            features['ad_normalized'] = ad_raw / (cum_volume + 1e-8)
            # A/D rate of change (momentum) - use log returns to avoid explosion
            features['ad_roc'] = np.log(ad_raw / (ad_raw.shift(20) + 1e-8))
            
            # Chaikin Money Flow (CORRECT calculation, not ADOSC!)
            # CMF = sum(Money Flow Volume) / sum(Volume) over period
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
            mfv = mfm * df['Volume']
            features['cmf'] = mfv.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-8)
            
            # Relative volume (with log1p transform to handle volume spikes)
            # Note: relative_volume can spike to 5x+ on high-volume days (news, earnings)
            # Log1p transform compresses these outliers while preserving relative ordering
            rel_vol_raw = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)
            features['relative_volume'] = np.log1p(rel_vol_raw)  # Log transform to handle spikes
        
        # 5. MOMENTUM GROUP - Momentum and oscillator indicators
        if config['momentum']:
            print("  → Momentum & oscillators")
            # RSI at multiple periods
            for period in [7, 14, 21]:
                features[f'rsi_{period}'] = talib.RSI(df['Close'], period)
            
            # MACD
            macd, signal_line, hist = talib.MACD(df['Close'])
            features['macd'] = macd
            features['macd_signal'] = signal_line
            features['macd_hist'] = hist
            features['macd_divergence'] = macd - signal_line
            
            # Stochastic oscillator
            slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'])
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            features['stoch_k_d_diff'] = slowk - slowd
            
            # Williams %R
            features['williams_r'] = talib.WILLR(df['High'], df['Low'], df['Close'])
            
            # Average True Range (ATR)
            features['atr'] = talib.ATR(df['High'], df['Low'], df['Close'])
            features['atr_ratio'] = features['atr'] / df['Close']
            
            # CCI (Commodity Channel Index)
            features['cci'] = talib.CCI(df['High'], df['Low'], df['Close'])
            
            # Z-score of returns
            features['return_zscore_20'] = (
                (returns - returns.rolling(20).mean()) /
                (returns.rolling(20).std() + 1e-8)
            )
            
            # ROC (Rate of Change)
            features['roc_10'] = talib.ROC(df['Close'], 10)
            features['roc_20'] = talib.ROC(df['Close'], 20)
        
        # 6. TREND GROUP - Trend strength and direction
        if config['trend']:
            print("  → Trend indicators")
            # ADX (trend strength) and components
            features['adx'] = talib.ADX(df['High'], df['Low'], df['Close'])
            features['adx_plus'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'])
            features['adx_minus'] = talib.MINUS_DI(
                df['High'], df['Low'], df['Close']
            )
            
            # Parabolic SAR
            features['sar'] = talib.SAR(df['High'], df['Low'])
            features['sar_signal'] = (df['Close'] > features['sar']).astype(int)
            
            # Aroon
            aroon_down, aroon_up = talib.AROON(df['High'], df['Low'])
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down
            features['aroon_oscillator'] = aroon_up - aroon_down
            
            # Distance from moving averages
            for period in [10, 20, 50, 200]:
                ma = df['Close'].rolling(period).mean()
                features[f'dist_from_ma{period}'] = (
                    (df['Close'] - ma) / (ma + 1e-8)
                )
        
        # 7. BOLLINGER GROUP - Bollinger Bands
        if config['bollinger']:
            print("  → Bollinger Bands")
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], timeperiod=20)
            features['bb_upper'] = bb_upper
            features['bb_lower'] = bb_lower
            features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_percent_b'] = features['bb_position']  # Same as bb_position
        
        # 8. PRICE_POSITION GROUP - Relative price levels
        if config['price_position']:
            print("  → Price position indicators")
            # Price position in range
            features['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
            
            # Distance from moving averages
            for period in [10, 20, 50, 200]:
                ma = df['Close'].rolling(period).mean()
                features[f'dist_from_ma{period}'] = (
                    (df['Close'] - ma) / (ma + 1e-8)
                )
            
            # Distance from highs/lows
            features['dist_from_20d_high'] = (
                df['Close'] / df['High'].rolling(20).max() - 1
            )
            features['dist_from_20d_low'] = (
                df['Close'] / df['Low'].rolling(20).min() - 1
            )
            
            # Serial correlation (autocorrelation of returns)
            features['serial_corr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(5))
        
        # 9. ENTROPY GROUP - Information theory measures
        if config['entropy']:
            print("  → Entropy & complexity")
            # Shannon entropy of returns
            features['return_entropy'] = returns.rolling(20).apply(self._shannon_entropy)
            
            # Lempel-Ziv complexity (market efficiency measure)
            features['lz_complexity'] = returns.rolling(20).apply(self._lempel_ziv_complexity)
            
            # Variance ratio test (mean reversion/momentum)
            features['variance_ratio'] = self._variance_ratio(returns, 5)
            
            # Hurst exponent (persistence)
            features['hurst_exponent'] = returns.rolling(60).apply(self._hurst_exponent)
        
        # 10. REGIME GROUP - Market regime classification
        if config['regime']:
            print("  → Regime indicators")
            # Volatility regime (requires volatility features)
            if config['volatility']:
                vol_percentiles = features['volatility_yz_20'].rolling(252).rank(pct=True)
                features['vol_regime'] = pd.cut(
                    vol_percentiles, bins=[0, 0.33, 0.67, 1.0], labels=[0, 1, 2]
                )
                
                # Relative volatility
                features['relative_volatility'] = (
                    features['volatility_yz_20'] / features['volatility_yz_20'].rolling(60).mean()
                )
            
            # Volume regime
            features['volume_percentile'] = (
                df['Volume'].rolling(252).rank(pct=True)
            )
            
            # Trend regime (requires trend features)
            if config['trend']:
                features['trend_percentile'] = features['adx'].rolling(252).rank(pct=True)
            
            # Relative Volatility Index
            features['rvi'] = self._relative_volatility_index(df['Close'], 14)
            
            # Market state classification
            # Note: Enhanced classification using trend + volatility (not just vol binning)
            # More sophisticated than standard 3-regime volatility classification
            if config['volatility']:
                features['market_state'] = self._classify_market_state(
                    df['Close'], features['volatility_yz_20']
                )
            
            # Fractal dimension (roughness of price path)
            features['fractal_dimension'] = self._fractal_dimension(df['Close'], 20)
        
        # 11. STATISTICAL GROUP - Statistical properties of returns
        if config['statistical']:
            print("  → Statistical properties (Skew, Kurtosis)")
            # Autocorrelation at multiple lags
            features['serial_corr_1'] = returns.rolling(20).apply(
                lambda x: x.autocorr(1) if len(x) > 1 else 0
            )
            features['serial_corr_5'] = returns.rolling(20).apply(
                lambda x: x.autocorr(5) if len(x) > 5 else 0
            )
            
            # Skewness & Kurtosis using professional log-return-based estimators
            # These match the professional repo implementation
            features['skewness_20'] = self._skew_volatility(df, 20)
            features['skewness_60'] = self._skew_volatility(df, 60)
            features['kurtosis_20'] = self._kurtosis_volatility(df, 20)
            features['kurtosis_60'] = self._kurtosis_volatility(df, 60)
            
            # Z-score of returns (duplicate in momentum, but kept here)
            if not config['momentum']:  # Only add if not already in momentum
                features['return_zscore_20'] = (
                    (returns - returns.rolling(20).mean()) /
                    (returns.rolling(20).std() + 1e-8)
                )
        
        # 12. RISK_ADJUSTED GROUP - Risk-adjusted performance metrics
        if config['risk_adjusted']:
            print("  → Risk-adjusted metrics")
            # Sharpe-like ratio
            features['sharpe_20'] = (
                returns.rolling(20).mean() /
                (returns.rolling(20).std() + 1e-8)
            )
            features['sharpe_60'] = (
                returns.rolling(60).mean() /
                (returns.rolling(60).std() + 1e-8)
            )
            
            # Risk-adjusted momentum (requires volatility)
            if config['volatility']:
                features['risk_adj_momentum_20'] = (
                    (df['Close'] / df['Close'].shift(20) - 1) /
                    (features['volatility_yz_20'] + 1e-8)
                )
            
            # Downside deviation (Sortino-like)
            downside_returns = returns.clip(upper=0)
            features['downside_vol_20'] = downside_returns.rolling(20).std()
            features['sortino_20'] = (
                returns.rolling(20).mean() /
                (features['downside_vol_20'] + 1e-8)
            )
            
            # Maximum drawdown
            features['max_drawdown_20'] = self._max_drawdown(df['Close'], 20)
            
            # Calmar ratio (return vs max drawdown)
            features['calmar_20'] = (
                returns.rolling(20).mean() /
                (features['max_drawdown_20'].abs() + 1e-8)
            )
            
            # Hui-Heubel liquidity ratio
            features['hui_heubel'] = self._hui_heubel_liquidity(df)
        
        # Clean up
        print("  → Cleaning features...")
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill to propagate last valid value
        # Then backfill warm-up period with first valid value
        # This is better than filling with 0 which causes division issues
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaN (shouldn't happen), fill with median
        # This is safer than 0 for ratio-based features
        for col in features.columns:
            if features[col].isna().any():
                # Skip categorical columns
                if pd.api.types.is_categorical_dtype(features[col]):
                    features[col] = features[col].fillna(features[col].mode()[0] if len(features[col].mode()) > 0 else 0)
                    continue
                
                # For numeric columns, use median
                median_val = features[col].median()
                if pd.isna(median_val) or median_val == 0:
                    # If median is 0 or NaN, use a small positive value
                    features[col] = features[col].fillna(1e-6)
                else:
                    features[col] = features[col].fillna(median_val)
        
        print(f"  ✅ Created {len(features.columns)} features")
        
        # FEATURE SELECTION: If 'feature_list' is specified in config, filter to those features
        # This is used by the 'wavenet_optimized' preset to select the 18 López de Prado features
        if 'feature_list' in config and config['feature_list'] is not None:
            feature_list = config['feature_list']
            available_features = [f for f in feature_list if f in features.columns]
            missing_features = [f for f in feature_list if f not in features.columns]
            
            if missing_features:
                print(f"  ⚠️  WARNING: {len(missing_features)} features not found: {missing_features[:5]}...")
            
            if available_features:
                features = features[available_features]
                print(f"  🎯 Filtered to {len(features.columns)} selected features (from feature_list)")
            else:
                print(f"  ⚠️  WARNING: No features from feature_list found! Using all {len(features.columns)} features")
        
        return features
    
    @staticmethod
    @jit(nopython=True)
    def _yang_zhang_volatility_jit(high, low, close, open_price, window=20):
        """Yang-Zhang volatility estimator (Numba optimized)"""
        n = len(close)
        result = np.empty(n, dtype=np.float64)
        
        # Fill initial values with NaN
        for i in range(window):
            result[i] = np.nan
        
        for i in range(window, n):
            # Rogers-Satchell volatility component
            rs_sum = 0.0
            for j in range(i-window, i):
                hl = np.log(high[j] / close[j])
                ho = np.log(high[j] / open_price[j])
                rs_sum += hl * ho
            rs_var = rs_sum / window
            
            # Close-Open volatility
            co_sum = 0.0
            co_mean = 0.0
            for j in range(i-window, i):
                val = np.log(close[j] / open_price[j])
                co_mean += val
            co_mean /= window
            
            for j in range(i-window, i):
                val = np.log(close[j] / open_price[j])
                co_sum += (val - co_mean) ** 2
            co_var = co_sum / window
            
            # Open-Close volatility (overnight)
            oc_sum = 0.0
            oc_mean = 0.0
            for j in range(i-window, i):
                if j > 0:
                    val = np.log(open_price[j] / close[j-1])
                    oc_mean += val
            oc_mean /= window
            
            for j in range(i-window, i):
                if j > 0:
                    val = np.log(open_price[j] / close[j-1])
                    oc_sum += (val - oc_mean) ** 2
            oc_var = oc_sum / window
            
            # Yang-Zhang estimator
            k = 0.34 / (1.0 + (window + 1.0) / (window - 1.0))
            yz_var = k * oc_var + (1.0 - k) * (co_var + rs_var)
            
            # Ensure non-negative before taking sqrt
            # Use NaN instead of 0.0 to properly signal missing data
            if yz_var > 0:
                result[i] = np.sqrt(yz_var)
            else:
                result[i] = np.nan
            
        return result
    
    def _yang_zhang_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252) -> pd.Series:
        """
        Yang-Zhang volatility estimator
        Based on: https://github.com/jasonstrimpel/volatility-trading
        
        More efficient than the JIT version, uses pandas rolling
        """
        import math
        
        log_ho = (df['High'] / df['Open']).apply(np.log)
        log_lo = (df['Low'] / df['Open']).apply(np.log)
        log_co = (df['Close'] / df['Open']).apply(np.log)
        
        log_oc = (df['Open'] / df['Close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc**2
        
        log_cc = (df['Close'] / df['Close'].shift(1)).apply(np.log)
        log_cc_sq = log_cc**2
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        close_vol = log_cc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        
        open_vol = log_oc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        
        window_rs = rs.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)
        
        return result
    
    def _parkinson_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252) -> pd.Series:
        """
        Parkinson volatility using high-low range
        Based on: https://github.com/jasonstrimpel/volatility-trading
        """
        import math
        
        log_hl = (df['High'] / df['Low']).apply(np.log)
        rs = (1.0 / (4.0 * math.log(2.0))) * (log_hl**2.0)
        
        def f(v):
            return (trading_periods * v.mean())**0.5
        
        result = rs.rolling(
            window=window,
            center=False
        ).apply(func=f)
        
        return result
    
    def _calculate_vpin(self, df: pd.DataFrame, n_buckets: int = 50) -> pd.Series:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN)
        Easley et al. (2012)
        
        Fixed to use proper bucketing by total volume divided into n_buckets,
        and handle zero price changes properly.
        """
        # Classify volume as buy or sell using tick rule
        price_change = df['Close'].diff()
        
        # Handle zero price changes: assign to buy if next change is positive
        # This is the "tick test" - use direction of next non-zero change
        buy_volume = df['Volume'].where(price_change > 0, 0)
        sell_volume = df['Volume'].where(price_change < 0, 0)
        
        # For zero changes, split volume equally (conservative approach)
        zero_change_mask = (price_change == 0) & (price_change.notna())
        buy_volume = buy_volume.where(~zero_change_mask, df['Volume'] / 2)
        sell_volume = sell_volume.where(~zero_change_mask, df['Volume'] / 2)
        
        # Calculate total volume and create buckets
        total_volume = df['Volume'].sum()
        volume_per_bucket = total_volume / n_buckets
        
        # Assign each bar to a bucket based on cumulative volume
        cum_volume = df['Volume'].cumsum()
        bucket_id = (cum_volume / volume_per_bucket).astype(int)
        
        # Calculate order imbalance for each bucket
        vpin = pd.Series(index=df.index, dtype=float)
        for bucket in bucket_id.unique():
            mask = bucket_id == bucket
            if mask.sum() > 0:
                buy_vol = buy_volume[mask].sum()
                sell_vol = sell_volume[mask].sum()
                total_vol = buy_vol + sell_vol
                if total_vol > 0:
                    vpin[mask] = abs(buy_vol - sell_vol) / total_vol
                else:
                    vpin[mask] = 0.0
                    
        return vpin.rolling(20).mean()
    
    def _kyle_lambda(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Kyle's Lambda - price impact coefficient
        Measures information asymmetry
        """
        returns = df['Close'].pct_change()
        signed_volume = df['Volume'] * np.sign(returns)
        
        lambda_values = pd.Series(index=df.index, dtype=float)
        
        for i in range(window, len(df)):
            y = returns.iloc[i-window:i].values
            x = signed_volume.iloc[i-window:i].values
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                # Linear regression: price change = lambda * signed_volume
                x_clean, y_clean = x[mask], y[mask]
                if np.std(x_clean) > 0:
                    lambda_values.iloc[i] = np.cov(x_clean, y_clean)[0, 1] / np.var(x_clean)
                    
        return lambda_values
    
    def _roll_spread(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Roll's implied spread estimator
        Based on serial covariance of price changes
        """
        returns = df['Close'].pct_change()
        
        spread = returns.rolling(window).apply(
            lambda x: 2 * np.sqrt(-np.cov(x[:-1], x[1:])[0, 1]) 
            if np.cov(x[:-1], x[1:])[0, 1] < 0 else 0
        )
        
        return spread
    
    def _corwin_schultz_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        Corwin-Schultz bid-ask spread estimator (Simplified)
        Uses high-low prices over 2 days
        
        Reference: Corwin and Schultz (2012) "A Simple Way to Estimate
        Bid-Ask Spreads from Daily High and Low Prices"
        
        Note: This is a simplified version. The full implementation requires
        extensive overnight return adjustments and data cleaning that are
        dataset-specific. For equity data without these adjustments, we use
        roll_spread as the primary spread estimator instead.
        
        Fallback: Returns hl_range / 2 as a simple spread proxy.
        """
        # Simple proxy: High-Low range divided by 2
        # This is a reasonable approximation of the bid-ask bounce
        # and correlates with true spreads
        hl_range = (df['High'] - df['Low']) / df['Close']
        spread = (hl_range / 2).clip(lower=0, upper=0.1)
        
        return spread
    
    def _order_flow_imbalance(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Order flow imbalance proxy (symmetric version)
        
        Returns net buy/sell pressure from -1 (all sells) to +1 (all buys)
        
        Note: This is the symmetric formulation: (buy - sell) / (buy + sell)
        Standard version uses buy ratio: buy / (buy + sell) which ranges [0, 1]
        Symmetric version is more intuitive for ML features (centered at zero)
        """
        returns = df['Close'].pct_change()
        
        # Classify volume using tick rule
        buy_volume = df['Volume'] * (returns > 0).astype(float)
        sell_volume = df['Volume'] * (returns < 0).astype(float)
        
        # Calculate imbalance (symmetric: -1 to +1)
        imbalance = (buy_volume - sell_volume).rolling(window).sum() / \
                   (buy_volume + sell_volume).rolling(window).sum()
        
        return imbalance
    
    def _shannon_entropy(self, returns: pd.Series) -> float:
        """
        Calculate Shannon entropy of returns distribution
        Uses histogram binning (equal-width) instead of quantile binning
        to capture actual distribution variation
        """
        if len(returns) < 2:
            return 0
            
        # Use histogram binning with equal-width bins
        # Number of bins using Sturges' rule: k = 1 + log2(n)
        n_bins = max(3, int(np.ceil(1 + np.log2(len(returns)))))
        
        try:
            # Create histogram with equal-width bins
            counts, _ = np.histogram(returns, bins=n_bins)
            
            # Filter out empty bins and calculate probabilities
            counts = counts[counts > 0]
            probs = counts / counts.sum()
            
            # Shannon entropy: H = -sum(p * log2(p))
            entropy = -np.sum(probs * np.log2(probs))
            
            return entropy
        except Exception:
            # Fallback for edge cases (e.g., all values identical)
            return 0.0
    
    def _lempel_ziv_complexity(self, returns: pd.Series) -> float:
        """
        Lempel-Ziv complexity
        Measures randomness/predictability
        """
        # Convert to binary string
        binary = (returns > returns.median()).astype(int).astype(str)
        s = ''.join(binary.values)
        
        n = len(s)
        if n == 0:
            return 0
            
        # LZ76 algorithm
        i, k, l, k_max = 0, 1, 1, 1
        c = 1
        
        while k + l <= n:
            if s[i:i+l] == s[k:k+l]:
                l += 1
            else:
                c += 1
                if l > k_max:
                    k_max = l
                i += 1
                k = i + 1
                l = 1
                
        # Normalize
        return c / (n / np.log2(n + 1))
    
    def _variance_ratio(self, returns: pd.Series, lag: int = 5) -> pd.Series:
        """
        Variance ratio test for mean reversion/momentum
        VR < 1: Mean reversion
        VR > 1: Momentum
        """
        def calc_vr(x):
            if len(x) < lag * 2:
                return np.nan
            var_1 = np.var(x)
            var_lag = np.var(x[::lag])
            return var_lag / (var_1 * lag) if var_1 > 0 else np.nan
            
        return returns.rolling(60).apply(calc_vr)
    
    def _hurst_exponent(self, returns: pd.Series) -> float:
        """
        Hurst exponent
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(returns) < 20:
            return 0.5
            
        # R/S analysis
        lags = range(2, min(20, len(returns) // 2))
        rs_values = []
        
        for lag in lags:
            # Calculate R/S statistic
            chunks = [returns[i:i+lag] for i in range(0, len(returns), lag)]
            rs_list = []
            
            for chunk in chunks:
                if len(chunk) >= lag:
                    mean = chunk.mean()
                    std = chunk.std()
                    if std > 0:
                        cumsum = (chunk - mean).cumsum()
                        R = cumsum.max() - cumsum.min()
                        rs_list.append(R / std)
                        
            if rs_list:
                rs_values.append(np.mean(rs_list))
                
        if len(rs_values) > 1:
            # Linear regression of log(R/S) vs log(lag)
            log_lags = np.log(list(lags)[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove invalid values
            mask = np.isfinite(log_lags) & np.isfinite(log_rs)
            if mask.sum() > 1:
                hurst = np.polyfit(log_lags[mask], log_rs[mask], 1)[0]
                return np.clip(hurst, 0, 1)
                
        return 0.5
    
    def _fractal_dimension(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Fractal dimension of price path
        FD ≈ 1: Smooth trend
        FD ≈ 1.5: Random walk
        FD ≈ 2: Very rough/noisy
        """
        def calc_fd(x):
            if len(x) < 3:
                return 1.5
            
            # Convert to numpy array to ensure integer indexing
            x_arr = np.asarray(x)
                
            # Higuchi method
            N = len(x_arr)
            k_max = min(8, N // 2)
            L = []
            
            for k in range(1, k_max):
                Lk = []
                for m in range(k):
                    Lmk = 0
                    num_segments = int((N - m) / k)
                    if num_segments < 1:
                        continue
                    for i in range(1, num_segments + 1):
                        idx_current = m + i * k
                        idx_prev = m + (i - 1) * k
                        if idx_current < N and idx_prev < N:
                            Lmk += abs(x_arr[idx_current] - x_arr[idx_prev])
                    if num_segments > 0:
                        # Higuchi formula: normalize and divide by k
                        Lmk = (Lmk * (N - 1)) / (k * num_segments * k)
                        Lk.append(Lmk)
                if len(Lk) > 0:
                    L.append(np.mean(Lk))
                
            if len(L) > 1:
                # Higuchi fractal dimension: FD = -slope
                # where slope is from log2(L(k)) vs log2(k)
                log_k = np.log2(np.arange(1, len(L) + 1))
                log_L = np.log2(L)
                
                mask = np.isfinite(log_k) & np.isfinite(log_L)
                if mask.sum() > 1:
                    slope = np.polyfit(log_k[mask], log_L[mask], 1)[0]
                    fd = -slope  # FD = -slope (NOT 2 - slope)
                    return np.clip(fd, 1, 2)
                    
            return 1.5
            
        return prices.rolling(window).apply(calc_fd, raw=False)
    
    def _hui_heubel_liquidity(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Hui-Heubel liquidity ratio
        Lower values = more liquid
        """
        high = df['High'].rolling(window).max()
        low = df['Low'].rolling(window).min()
        volume = df['Volume'].rolling(window).sum()
        
        price_range = (high - low) / low
        turnover = volume / df['Volume'].rolling(20).mean()
        
        return price_range / (turnover + 1e-8)
    
    def _garman_klass_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252):
        """
        Garman-Klass volatility estimator
        Based on: https://github.com/jasonstrimpel/volatility-trading
        """
        import math
        
        log_hl = (df['High'] / df['Low']).apply(np.log)
        log_co = (df['Close'] / df['Open']).apply(np.log)
        
        rs = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2
        
        def f(v):
            return (trading_periods * v.mean())**0.5
        
        result = rs.rolling(window=window, center=False).apply(func=f)
        
        return result
    
    def _close_to_close_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252):
        """
        Close-to-close (simple) volatility estimator
        Based on: https://github.com/jasonstrimpel/volatility-trading
        Uses log returns instead of pct_change for stability
        """
        import math
        
        log_return = (df['Close'] / df['Close'].shift(1)).apply(np.log)
        
        def f(v):
            return (trading_periods * v.var())**0.5
        
        result = log_return.rolling(window=window, center=False).apply(func=f)
        
        return result
    
    def _rogers_satchell_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252):
        """
        Rogers-Satchell volatility estimator
        Based on: https://github.com/jasonstrimpel/volatility-trading
        
        Assumes zero drift (no directional bias).
        Better for intraday analysis and options pricing.
        """
        import math
        
        log_ho = (df['High'] / df['Open']).apply(np.log)
        log_lo = (df['Low'] / df['Open']).apply(np.log)
        log_co = (df['Close'] / df['Open']).apply(np.log)
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        def f(v):
            return (trading_periods * v.mean())**0.5
        
        result = rs.rolling(window=window, center=False).apply(func=f)
        
        return result
    
    def _hodges_tompkins_volatility(self, df: pd.DataFrame, window: int = 20, trading_periods: int = 252):
        """
        Hodges-Tompkins volatility estimator with bias correction
        Based on: https://github.com/jasonstrimpel/volatility-trading
        
        Bias-corrected close-to-close estimator.
        Better for small sample sizes with overlapping samples.
        """
        import math
        
        log_return = (df['Close'] / df['Close'].shift(1)).apply(np.log)

        vol = log_return.rolling(window=window, center=False).std() * math.sqrt(trading_periods)

        h = window
        n = (log_return.count() - h) + 1

        adj_factor = 1.0 / (1.0 - (h / n) + ((h**2 - 1) / (3 * n**2)))

        result = vol * adj_factor
        
        return result
    
    def _skew_volatility(self, df: pd.DataFrame, window: int = 20):
        """
        Skewness estimator for log returns
        Based on: https://github.com/jasonstrimpel/volatility-trading
        
        Measures asymmetry of return distribution.
        Negative skew: left tail (more downside risk)
        Positive skew: right tail (more upside potential)
        """
        log_return = (df['Close'] / df['Close'].shift(1)).apply(np.log)
        
        result = log_return.rolling(window=window, center=False).skew()
        
        return result
    
    def _kurtosis_volatility(self, df: pd.DataFrame, window: int = 20):
        """
        Kurtosis estimator for log returns
        Based on: https://github.com/jasonstrimpel/volatility-trading
        
        Measures tail heaviness of return distribution.
        High kurtosis: fat tails (more extreme events)
        Low kurtosis: thin tails (fewer extreme events)
        Normal distribution has kurtosis = 3
        """
        log_return = (df['Close'] / df['Close'].shift(1)).apply(np.log)
        
        result = log_return.rolling(window=window, center=False).kurt()
        
        return result
    
    def _max_drawdown(self, prices, window):
        """Maximum drawdown over window"""
        def calc_dd(x):
            cummax = pd.Series(x).expanding().max()
            dd = (x - cummax) / cummax
            return dd.min()
        
        return prices.rolling(window).apply(calc_dd)
    
    def _approximate_entropy(self, x, m=2, r=0.2):
        """Approximate entropy"""
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x_arr = [[x.iloc[j] for j in range(i, i + m - 1 + 1)] 
                     for i in range(len(x) - m + 1)]
            C = [
                len([1 for x_j in x_arr if _maxdist(x_i, x_j) <= r]) / 
                (len(x_arr) + 0.0)
                for x_i in x_arr
            ]
            return (len(x_arr) + 0.0) ** (-1) * sum(np.log(C))
        
        try:
            return abs(_phi(m + 1) - _phi(m))
        except:
            return 0
    
    def _relative_volatility_index(self, prices, period):
        """Relative Volatility Index"""
        price_change = prices.diff()
        std_up = price_change.where(price_change > 0, 0).rolling(period).std()
        std_down = price_change.where(price_change < 0, 0).rolling(period).std()
        
        rvi = 100 * std_up / (std_up + std_down + 1e-8)
        return rvi
    
    def _classify_market_state(self, prices, volatility):
        """
        Classify market state: 0=sideways, 1=uptrend, 2=downtrend
        
        Enhanced classification using both trend and volatility.
        
        Note: This is more sophisticated than standard volatility regime classification
        which only bins by volatility level. This version considers:
        - Trend direction (20-day returns)
        - Volatility regime (vs 252-day median)
        - Combines both for more nuanced classification
        
        Returns:
            0 = Sideways (default): No strong trend or high volatility
            1 = Uptrend: Positive 20d return AND below-median volatility
            2 = Downtrend: Negative 20d return AND below-median volatility
        """
        returns_20 = prices.pct_change(20)
        vol_20 = volatility
        
        # Thresholds
        trend_threshold = 0.05  # 5% move
        vol_threshold = vol_20.rolling(252).quantile(0.5)
        
        state = pd.Series(0, index=prices.index)  # Default sideways
        
        # Uptrend: positive return AND low volatility
        state = state.where(
            ~((returns_20 > trend_threshold) & (vol_20 < vol_threshold)),
            1
        )
        
        # Downtrend: negative return AND low volatility
        state = state.where(
            ~((returns_20 < -trend_threshold) & (vol_20 < vol_threshold)),
            2
        )
        
        return state
    
    def create_dynamic_triple_barriers(self, df: pd.DataFrame,
                                      base_tp: float = 0.06,
                                      base_sl: float = 0.03,
                                      horizon: int = 5) -> pd.DataFrame:
        """
        Dynamic triple barriers based on volatility regime - MLFINLAB METHOD
        
        Returns labels AND actual returns at barrier exit (not fixed-horizon returns).
        This aligns with López de Prado's approach and mlfinlab implementation.
        
        Returns:
            DataFrame with columns:
            - label: Which barrier hit first (-1=timeout, 0=SL, 1=TP)
            - t1: Timestamp when barrier was hit (exit time)
            - exit_return: Actual return from entry to exit
            - exit_day: Number of days until exit (1 to horizon)
            - dynamic_tp: TP threshold used
            - dynamic_sl: SL threshold used
        """
        # Calculate realized volatility
        volatility = df['Close'].pct_change().rolling(20, min_periods=10).std()
        
        # Fill initial NaN values in volatility with forward fill then median
        volatility = volatility.fillna(method='bfill')
        if volatility.isna().any():
            volatility = volatility.fillna(volatility.median())
        
        # Volatility percentile (regime) - use expanding window instead of rolling
        # This avoids the 252-day warmup period
        vol_percentile = volatility.expanding(min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        
        # Fill initial NaN values in percentile with 0.5 (neutral regime)
        vol_percentile = vol_percentile.fillna(0.5)
        
        # Dynamic barriers
        # High volatility: wider barriers
        # Low volatility: tighter barriers
        tp_mult = 1 + (vol_percentile - 0.5) * 0.5  # Range: [0.75, 1.25]
        sl_mult = 1 + (vol_percentile - 0.5) * 0.3  # Range: [0.85, 1.15]
        
        dynamic_tp = base_tp * tp_mult
        dynamic_sl = base_sl * sl_mult
        
        # Fill any remaining NaN with base thresholds (warmup period)
        dynamic_tp = dynamic_tp.fillna(base_tp)
        dynamic_sl = dynamic_sl.fillna(base_sl)
        
        # Apply triple barrier labeling with dynamic thresholds
        # Track EXIT times and returns (mlfinlab approach)
        labels = []
        exit_timestamps = []
        exit_returns = []
        exit_days = []
        
        for i in range(len(df) - horizon):
            future_returns = df['Close'].pct_change().iloc[i+1:i+1+horizon].cumsum()
            entry_price = df['Close'].iloc[i]
            
            tp_threshold = dynamic_tp.iloc[i]
            sl_threshold = dynamic_sl.iloc[i]
            
            # Check which barrier is hit first
            tp_hit = (future_returns >= tp_threshold).idxmax() if (future_returns >= tp_threshold).any() else None
            sl_hit = (future_returns <= -sl_threshold).idxmax() if (future_returns <= -sl_threshold).any() else None
            
            if tp_hit is not None and (sl_hit is None or tp_hit < sl_hit):
                # TP hit first
                labels.append(1)
                exit_idx = df.index.get_loc(tp_hit)
                exit_day_offset = exit_idx - i
                exit_timestamps.append(tp_hit)
                exit_days.append(exit_day_offset)
                
                # Calculate actual return at TP touch (mlfinlab method)
                exit_price = df['Close'].loc[tp_hit]
                exit_returns.append((exit_price / entry_price) - 1)
                
            elif sl_hit is not None:
                # SL hit first
                labels.append(0)
                exit_idx = df.index.get_loc(sl_hit)
                exit_day_offset = exit_idx - i
                exit_timestamps.append(sl_hit)
                exit_days.append(exit_day_offset)
                
                # Calculate actual return at SL touch
                exit_price = df['Close'].loc[sl_hit]
                exit_returns.append((exit_price / entry_price) - 1)
                
            else:
                # Timeout (vertical barrier)
                labels.append(-1)
                timeout_idx = i + horizon
                if timeout_idx < len(df):
                    exit_timestamps.append(df.index[timeout_idx])
                    exit_days.append(horizon)
                    exit_price = df['Close'].iloc[timeout_idx]
                    exit_returns.append((exit_price / entry_price) - 1)
                else:
                    # Edge case: not enough data for full horizon
                    exit_timestamps.append(df.index[-1])
                    exit_days.append(len(df) - i - 1)
                    exit_price = df['Close'].iloc[-1]
                    exit_returns.append((exit_price / entry_price) - 1)
        
        # Pad the end with NaN (no valid trades in last horizon days)
        pad_length = horizon
        labels.extend([-1] * pad_length)
        exit_timestamps.extend([pd.NaT] * pad_length)
        exit_returns.extend([0.0] * pad_length)
        exit_days.extend([0] * pad_length)
        
        return pd.DataFrame({
            'label': labels,
            't1': exit_timestamps,              # NEW: Exit timestamp
            'exit_return': exit_returns,        # NEW: Actual return at exit
            'exit_day': exit_days,              # NEW: Days until exit
            'dynamic_tp': dynamic_tp,
            'dynamic_sl': dynamic_sl
        }, index=df.index)

