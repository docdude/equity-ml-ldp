"""
Feature Configuration System for Financial ML Pipeline

This module provides a flexible configuration system for selective feature engineering.
Features are organized into logical groups with toggle switches for easy control.

Usage Examples:
    # Use a preset configuration
    config = FeatureConfig.get_preset('minimal')
    
    # Create custom configuration
    config = FeatureConfig.get_preset('balanced')
    config['entropy'] = True  # Enable entropy features
    
    # Use in feature engineering
    feature_engineer = EnhancedFinancialFeatures()
    features = feature_engineer.create_all_features(df, feature_config=config)
"""

class FeatureConfig:
    """
    Configuration class for selective feature engineering.
    
    Organizes 108 features into 12 logical groups with toggle switches.
    Provides preset configurations for different use cases.
    """
    
    # Default feature group toggles
    DEFAULT_CONFIG = {
        # CORE FEATURES (22 features) - Foundation for most models
        'returns': True,              # 10 features - log returns, forward returns, acceleration
        'volatility': True,            # 12 features - Yang-Zhang, Parkinson, Garman-Klass, realized vol
        
        # TECHNICAL INDICATORS (49 features) - Standard technical analysis
        'momentum': True,              # 17 features - RSI, MACD, Stochastic, Williams %R, CCI
        'trend': True,                 # 10 features - ADX, ROC, SAR, Aroon, moving averages
        'volume': True,                # 13 features - OBV, VWAP, Accumulation/Distribution, CMF
        'bollinger': True,             # 5 features - Bollinger Bands and position indicators
        'price_position': True,        # 8 features - Distance from highs/lows, price levels
        
        # ADVANCED FEATURES (37 features) - Sophisticated metrics
        'microstructure': True,        # 10 features - Spreads, liquidity, order flow, VPIN
        'statistical': True,           # 8 features - Skewness, kurtosis, z-scores
        'entropy': True,               # 4 features - Shannon entropy, LZ complexity, variance ratio
        'regime': True,                # 8 features - Volatility regime, market state classification
        'risk_adjusted': True,         # 8 features - Sharpe, Sortino, Calmar ratios
    }
    
    # Feature group details with descriptions and feature counts
    FEATURE_GROUPS = {
        'returns': {
            'count': 10,
            'description': 'Return-based features',
            'features': [
                'log_return_1d', 'log_return_2d', 'log_return_3d', 'log_return_5d',
                'log_return_10d', 'log_return_20d',
                'forward_return_1d', 'forward_return_3d', 'forward_return_5d',
                'return_acceleration'
            ],
            'importance': 'CRITICAL',
            'computation': 'Fast'
        },
        'volatility': {
            'count': 12,
            'description': 'Volatility measures using multiple estimators',
            'features': [
                'volatility_yz_10', 'volatility_yz_20', 'volatility_yz_60',
                'volatility_parkinson_10', 'volatility_parkinson_20',
                'volatility_gk_20', 'volatility_cc_20', 'volatility_cc_60',
                'vol_ratio_short_long', 'vol_of_vol',
                'realized_vol_positive', 'realized_vol_negative'
            ],
            'importance': 'CRITICAL',
            'computation': 'Medium (Numba JIT optimized)'
        },
        'momentum': {
            'count': 17,
            'description': 'Momentum and oscillator indicators',
            'features': [
                'rsi_7', 'rsi_14', 'rsi_21',
                'macd', 'macd_signal', 'macd_hist', 'macd_divergence',
                'stoch_k', 'stoch_d', 'stoch_k_d_diff',
                'williams_r', 'atr', 'atr_ratio',
                'roc_10', 'roc_20', 'cci', 'return_zscore_20'
            ],
            'importance': 'HIGH',
            'computation': 'Fast (TA-Lib)'
        },
        'trend': {
            'count': 10,
            'description': 'Trend strength and direction indicators',
            'features': [
                'adx', 'adx_plus', 'adx_minus',
                'sar', 'sar_signal',
                'aroon_up', 'aroon_down', 'aroon_oscillator',
                'dist_from_ma10', 'dist_from_ma20'
            ],
            'importance': 'HIGH',
            'computation': 'Fast (TA-Lib)'
        },
        'volume': {
            'count': 13,
            'description': 'Volume-based indicators and analysis',
            'features': [
                'volume_norm', 'volume_std', 'volume_roc',
                'dollar_volume', 'dollar_volume_ma_ratio',
                'vwap_20', 'price_vwap_ratio',
                'obv', 'obv_ma_20',
                'ad_line', 'ad_normalized', 'cmf',
                'relative_volume'
            ],
            'importance': 'HIGH',
            'computation': 'Fast'
        },
        'bollinger': {
            'count': 5,
            'description': 'Bollinger Bands and position indicators',
            'features': [
                'bb_upper', 'bb_lower', 'bb_position', 'bb_width', 'bb_percent_b'
            ],
            'importance': 'MEDIUM',
            'computation': 'Fast (TA-Lib)'
        },
        'price_position': {
            'count': 8,
            'description': 'Relative price position and levels',
            'features': [
                'price_position',
                'dist_from_ma10', 'dist_from_ma20',
                'dist_from_ma50', 'dist_from_ma200',
                'dist_from_20d_high', 'dist_from_20d_low',
                'serial_corr_5'
            ],
            'importance': 'MEDIUM',
            'computation': 'Fast'
        },
        'microstructure': {
            'count': 10,
            'description': 'Market microstructure and liquidity metrics',
            'features': [
                'hl_range', 'hl_range_ma', 'oc_range',
                'roll_spread', 'cs_spread', 'hl_volatility_ratio',
                'amihud_illiq', 'amihud_illiquidity',
                'order_flow_imbalance', 'vpin', 'kyle_lambda'
            ],
            'importance': 'MEDIUM',
            'computation': 'Medium (Some complex calculations)'
        },
        'statistical': {
            'count': 6,
            'description': 'Statistical properties of returns',
            'features': [
                'serial_corr_1',
                'skewness_20', 'skewness_60',
                'kurtosis_20', 'kurtosis_60',
                'return_zscore_20'
            ],
            'importance': 'MEDIUM',
            'computation': 'Fast'
        },
        'entropy': {
            'count': 4,
            'description': 'Information theory and complexity measures',
            'features': [
                'return_entropy', 'lz_complexity',
                'variance_ratio', 'hurst_exponent'
            ],
            'importance': 'LOW',
            'computation': 'Slow (Rolling window calculations)'
        },
        'regime': {
            'count': 6,
            'description': 'Market regime classification and indicators',
            'features': [
                'vol_regime', 'volume_percentile', 'trend_percentile',
                'rvi', 'market_state', 'fractal_dimension'
            ],
            'importance': 'MEDIUM',
            'computation': 'Medium'
        },
        'risk_adjusted': {
            'count': 8,
            'description': 'Risk-adjusted performance metrics',
            'features': [
                'sharpe_20', 'sharpe_60',
                'risk_adj_momentum_20',
                'downside_vol_20', 'sortino_20',
                'max_drawdown_20', 'calmar_20',
                'hui_heubel'
            ],
            'importance': 'MEDIUM',
            'computation': 'Fast'
        }
    }
    
    # Preset configurations for common use cases
    PRESETS = {
        'minimal': {
            'description': 'Core features for fast training (~50 features)',
            'use_case': 'Initial experiments, quick iterations, baseline models',
            'config': {
                'returns': True,
                'volatility': True,
                'momentum': True,
                'trend': True,
                'volume': True,
                'bollinger': True,
                'price_position': True,
                'microstructure': False,
                'statistical': False,
                'entropy': False,
                'regime': False,
                'risk_adjusted': False
            },
            'expected_features': 72
        },
        
        'balanced': {
            'description': 'Balanced feature set for production (~85 features)',
            'use_case': 'Production models, good signal-to-noise ratio',
            'config': {
                'returns': True,
                'volatility': True,
                'momentum': True,
                'trend': True,
                'volume': True,
                'bollinger': True,
                'price_position': True,
                'microstructure': False,  # Often noisy in daily data
                'statistical': True,
                'entropy': False,  # Slow to compute, marginal value
                'regime': True,
                'risk_adjusted': True
            },
            'expected_features': 98
        },
        
        'comprehensive': {
            'description': 'All features for research (~108 features)',
            'use_case': 'Research, feature importance analysis, experimentation',
            'config': {k: True for k in DEFAULT_CONFIG.keys()},
            'expected_features': 108
        },
        
        'lightweight': {
            'description': 'Minimum features for low-latency applications (~40 features)',
            'use_case': 'High-frequency trading, low-latency requirements',
            'config': {
                'returns': True,
                'volatility': True,
                'momentum': True,
                'trend': False,
                'volume': True,
                'bollinger': True,
                'price_position': True,
                'microstructure': False,
                'statistical': False,
                'entropy': False,
                'regime': False,
                'risk_adjusted': False
            },
            'expected_features': 56
        },
        
        'technical_only': {
            'description': 'Pure technical indicators (~50 features)',
            'use_case': 'Traditional technical analysis approach',
            'config': {
                'returns': True,
                'volatility': True,
                'momentum': True,
                'trend': True,
                'volume': True,
                'bollinger': True,
                'price_position': True,
                'microstructure': False,
                'statistical': False,
                'entropy': False,
                'regime': False,
                'risk_adjusted': False
            },
            'expected_features': 72
        },
        
        'quant_research': {
            'description': 'Research-focused features (~95 features)',
            'use_case': 'Academic research, strategy development',
            'config': {
                'returns': True,
                'volatility': True,
                'momentum': True,
                'trend': True,
                'volume': True,
                'bollinger': True,
                'price_position': True,
                'microstructure': True,
                'statistical': True,
                'entropy': True,
                'regime': True,
                'risk_adjusted': True
            },
            'expected_features': 108
        },
        
        'wavenet_optimized': {
            'description': 'López de Prado optimized features for WaveNet (~18 features)',
            'use_case': 'WaveNet training - features selected by MDI/MDA/SFI/Orthogonal analysis',
            'config': {
                'returns': False,          # Not in top performers
                'volatility': True,        # 6 features: parkinson_10, yz_20, cc_60, vol_of_vol, rs_60, bb_width
                'momentum': True,          # 2 features: rsi_14, atr_ratio
                'trend': False,            # Not in top performers
                'volume': True,            # 3 features: volume_percentile, relative_volume, price_vwap_ratio
                'bollinger': True,         # 1 feature: bb_width (already included via volatility)
                'price_position': True,    # 3 features: dist_from_ma20, dist_from_20d_high, price_position
                'microstructure': True,    # 3 features: hl_range, hl_range_ma, roll_spread
                'statistical': False,      # Not in top performers
                'entropy': False,          # Not in top performers
                'regime': True,            # 2 features: market_state, vol_regime
                'risk_adjusted': False     # Not in top performers
            },
            'expected_features': 18,
            'notes': 'Based on comprehensive López de Prado evaluation (Oct 2025). See docs/WAVENET_FEATURE_SELECTION_RESULTS.md',
            'feature_list': [
                # Core Volatility (6)
                'volatility_parkinson_10', 'volatility_yz_20',
                'volatility_rs_20', 'volatility_cc_60', 'vol_of_vol',
                'volatility_rs_60', 'bb_width',
                # Microstructure (4)
                'hl_range', 'hl_range_ma', 'atr_ratio', 'roll_spread',
                # Price Position (3)
                'dist_from_ma20', 'price_vwap_ratio', 'dist_from_20d_high',
                # Volume (2)
                'volume_percentile', 'relative_volume',
                # Regime (2)
                'market_state', 'vol_regime',
                # Momentum (1)
                'rsi_14'
            ],
            'selection_criteria': {
                'mdi_top10': ['volatility_parkinson_10', 'atr_ratio', 'hl_range_ma', 'hl_range', 
                             'volatility_parkinson_20', 'volatility_yz_60', 'volatility_yz_20'],
                'mda_top5': ['hl_range', 'dist_from_ma20', 'volatility_yz_20', 
                            'volatility_parkinson_20', 'price_vwap_ratio'],
                'sfi_top5': ['atr_ratio', 'hl_range_ma', 'volatility_parkinson_10', 
                            'volatility_gk_20', 'volatility_rs_20'],
                'orthogonal_top5': ['vol_of_vol', 'volatility_yz_60', 'bb_width', 
                                   'volatility_rs_60', 'volume_percentile']
            },
            'timestep_importance': 't19 (most recent) - All top features from final timestep',
            'pbo_score': 0.000,  # Very low overfitting risk
            'purged_cv_auc': 0.6918,
            'walkforward_auc': 0.6742
        },
        
        'wavenet_optimized_v2': {
            'description': 'Consensus top 40 features from comprehensive importance analysis (Oct 2025)',
            'use_case': 'WaveNet training - balanced feature set combining MDI, MDA original, and MDA orthogonal',
            'config': {
                'returns': True,           # log_return_2d, 3d, 5d, 10d
                'volatility': True,        # volatility_ht_60
                'momentum': True,          # macd*, cci, rsi_7, stoch_k_d_diff
                'trend': True,             # sar, adx_plus
                'volume': True,            # volume_zscore, volume_norm, etc
                'bollinger': True,         # bb_lower, bb_position, etc
                'price_position': True,    # dist_from_ma*, serial_corr_*
                'microstructure': True,    # hl_volatility_ratio, amihud_illiq
                'statistical': True,       # kurtosis_60, skewness_*, serial_corr_1
                'entropy': True,           # variance_ratio ONLY
                'regime': True,            # relative_volatility ONLY
                'risk_adjusted': True      # risk_adj_momentum_20, max_drawdown_20
            },
            'expected_features': 42,
            'notes': 'Generated from equity_feature_importance_analysis.ipynb (Oct 21, 2025). Updated to include top 20 from Orthogonal MDA analysis. 70% overlap with notebook recommendations. Market features GC=F and ^FVX excluded (added separately via MARKET_FEATURES config).',
            'feature_list': [
                # Top performers from Orthogonal MDA (notebook's TOP 20)
                'serial_corr_1',           # #1 Ortho MDA: 0.01881
                'hl_volatility_ratio',     # #2 Ortho MDA: 0.01813
                'macd_hist',               # #3 Ortho MDA: 0.01756
                'macd_divergence',         # #4 Ortho MDA: 0.01756
                'macd_signal',             # #5 Ortho MDA: 0.01713
                'amihud_illiq',            # #6 Ortho MDA: 0.01703
                'roll_spread',             # #7 Ortho MDA: 0.01648 (ADDED)
                'macd',                    # #9 Ortho MDA: 0.01580
                'relative_volatility',     # #10 Ortho MDA: 0.01565
                'skewness_20',             # #11 Ortho MDA: 0.01543
                'sar',                     # #12 Ortho MDA: 0.01494
                'bb_upper',                # #13 Ortho MDA: 0.01494
                'hurst_exponent',          # #15 Ortho MDA: 0.01487 (ADDED)
                'vwap_20',                 # #16 Ortho MDA: 0.01476
                'ad_roc',                  # #17 Ortho MDA: 0.01455
                'bb_lower',                # #18 Ortho MDA: 0.01431
                'return_entropy',          # #19 Ortho MDA: 0.01430 (ADDED)
                'adx',                     # #20 Ortho MDA: 0.01407 (ADDED)
                # Additional high-value features from previous consensus
                'kurtosis_60', 'dist_from_ma50', 'dollar_volume_ma_ratio',
                'volume_zscore', 'bb_position', 'volatility_ht_60', 'cci',
                'dist_from_ma50', 'relative_volume', 'volume_norm',
                'dist_from_ma10', 'log_return_2d', 'log_return_3d',
                'log_return_5d', 'rsi_7', 'stoch_k_d_diff', 'serial_corr_5',
                'skewness_60', 'bb_percent_b', 'variance_ratio',
                'log_return_10d', 'risk_adj_momentum_20', 'max_drawdown_20',
                'adx_plus', 'vpin', 'price_vwap_ratio', 'realized_vol_positive'
            ],
            'selection_criteria': {
                'methodology': 'Consensus ranking across three methods',
                'mda_original_top10': [
                    'volume_zscore', 'bb_lower', 'serial_corr_5',
                    'aroon_up', 'kurtosis_60', 'bb_position',
                    'dollar_volume_ma_ratio', 'dist_from_ma50',
                    'dist_from_ma20', 'log_return_5d'
                ],
                'mda_orthogonal_top10': [
                    'serial_corr_1', 'hl_volatility_ratio', 'macd_hist',
                    'macd_divergence', 'macd_signal', 'amihud_illiq',
                    'roll_spread', '^TYX', 'macd', 'relative_volatility'
                ],
                'mdi_top10': [
                    'adx_plus', 'log_return_2d', 'volatility_ht_60',
                    'volume_percentile', 'kurtosis_60', 'volume_zscore',
                    'risk_adj_momentum_20', 'dollar_volume_ma_ratio',
                    'atr_20', 'max_drawdown_20'
                ],
                'market_features_note': (
                    'GC=F (rank 13/21) and ^FVX (rank 14 ortho) excluded '
                    'from feature_list. Add via MARKET_FEATURES config. '
                    '^GSPC/^VIX/JPY=X redundant with stock features.'
                )
            },
            'model_performance': {
                'rf_oob_score': 0.4613,
                'rf_oos_score': 0.4344,
                'rf_oos_std': 0.0394,
                'overfitting_gap': 0.027,
                'cv_method': '5-fold PurgedKFold with 1% embargo',
                'sample_weights': 'López de Prado uniqueness weights'
            },
            'analysis_date': 'October 2025',
            'analysis_notebook': 'equity_feature_importance_analysis.ipynb',
            'tickers': ['AAPL', 'NVDA', 'TSLA', 'AMZN', 'AVGO'],
            'samples': 2636
        },
        
        'wavenet_optimized_v3': {
            'description': 'Systematic feature selection from MDI/MDA/Orthogonal analysis (Oct 2025)',
            'use_case': 'WaveNet training - Method 3 (Hybrid) + Top Quadrant 1 (Robust) features',
            'config': {
                'returns': True,           # log_return_2d, 5d, return_acceleration
                'volatility': True,        # vol_of_vol, vol_ratio_short_long
                'momentum': True,          # cci, macd_signal, roc_10, stoch_*
                'trend': True,             # aroon_down, sar_signal, dist_from_ma200
                'volume': True,            # volume_percentile, volume_zscore, volume_norm, etc
                'bollinger': True,         # bb_position, bb_percent_b
                'price_position': False,   # Not in top selections
                'microstructure': True,    # vpin, oc_range, order_flow_imbalance
                'statistical': True,       # kurtosis_60, skewness_60, serial_corr_5
                'entropy': True,           # variance_ratio ONLY
                'regime': True,            # trend_percentile
                'risk_adjusted': True      # sharpe_60, sortino_20
            },
            'expected_features': 32,
            'notes': (
                'Generated from systematic_feature_selection.py (Oct 21, 2025). '
                'Uses Method 3 (Hybrid): Top 50 Ortho MDA excluding Quadrant 4, '
                'plus Top 5 Quadrant 1 (Robust) features. '
                'Removes dead weight (negative MDA features). '
                'Includes VPIN (now fixed and Quadrant 1 rank #5). '
                'Excludes market features - add separately if needed via MARKET_FEATURES config.'
            ),
            'feature_list': [
                # VOLUME (6) - Strong predictive power for liquidity/participation
                'volume_percentile',       # Ortho: 0.0353, Quadrant 1, Top unique info
                'volume_zscore',           # Quadrant 1 rank #1 (MDI: 0.0104, MDA: 0.0132)
                'volume_norm',             # Quadrant 1 rank #3 (MDI: 0.0094, MDA: 0.0125)
                'volume_roc',              # Ortho: 0.0197, momentum of volume changes
                'obv_normalized',          # Ortho: 0.0159, cumulative volume indicator
                'ad_roc',                  # Ortho: 0.0151, accumulation/distribution momentum
                
                # MOMENTUM (5) - Oscillators and trend-following indicators
                'cci',                     # Quadrant 1 rank #2 (MDI: 0.0102, MDA: 0.0132)
                'stoch_k_d_diff',          # Ortho: 0.0262, stochastic divergence
                'stoch_d',                 # Ortho: 0.0099, stochastic signal line
                'macd_signal',             # Ortho: 0.0092, MACD signal crosses
                'roc_10',                  # Ortho: 0.0087, 10-day rate of change
                
                # TREND (3) - Direction and strength indicators
                'sar_signal',              # Ortho: 0.0304, parabolic SAR signals
                'aroon_down',              # Ortho: 0.0229, downtrend strength
                'dist_from_ma200',         # Ortho: 0.0155, long-term trend position
                
                # VOLATILITY (2) - Vol dynamics and regime
                'vol_of_vol',              # Ortho: 0.0212, volatility instability
                'vol_ratio_short_long',    # Ortho: 0.0142, vol regime changes
                
                # BOLLINGER (2) - Price position relative to bands
                'bb_position',             # Quadrant 1 rank #4 (MDI: 0.0099, MDA: 0.0122)
                'bb_percent_b',            # Quadrant 1 rank #5 (MDI: 0.0094, MDA: 0.0113)
                
                # RETURNS (3) - Multi-timeframe return signals
                'log_return_2d',           # Ortho: 0.0186, 2-day momentum
                'log_return_5d',           # Ortho: 0.0132, weekly momentum
                'return_acceleration',     # Ortho: 0.0203, momentum of momentum
                
                # STATISTICAL (3) - Distribution properties
                'skewness_60',             # Ortho: 0.0290, tail risk indicator
                'kurtosis_60',             # Ortho: 0.0183, extreme event likelihood
                'serial_corr_5',           # Ortho: 0.0161, short-term autocorrelation
                
                # MICROSTRUCTURE (3) - Intraday price dynamics
                'vpin',                    # Quadrant 1 rank #5 (MDI: 0.0107, MDA: 0.0119) - FIXED!
                'oc_range',                # Ortho: 0.0116, open-close range
                'order_flow_imbalance',    # Ortho: 0.0153, buy/sell pressure
                
                # REGIME (1) - Market phase classification
                'trend_percentile',        # Ortho: 0.0155, trend strength percentile
                
                # RISK (2) - Risk-adjusted metrics
                'sharpe_60',               # Ortho: 0.0095, 60-day Sharpe ratio
                'sortino_20',              # Ortho: 0.0104, downside risk-adjusted
                
                # OTHER (2) - Hybrid indicators
                'cmf',                     # Ortho: 0.0318, Chaikin Money Flow
                'variance_ratio',          # Ortho: 0.0125, random walk test
                
            ],
            'selection_criteria': {
                'methodology': 'Method 3 (Hybrid) + Top 5 Quadrant 1',
                'method_3_logic': 'Top 50 Ortho MDA, exclude Quadrant 4 (Low MDI + Low MDA)',
                'quadrant_1_boost': 'Added top 5 Robust features (High MDI + High MDA)',
                'key_insight': 'Ortho MDA measures UNIQUE information after removing correlations',
                'quadrant_1_top5': [
                    'volume_zscore (MDI: 0.0104, MDA: 0.0132)',
                    'cci (MDI: 0.0102, MDA: 0.0132)',
                    'volume_norm (MDI: 0.0094, MDA: 0.0125)',
                    'bb_position (MDI: 0.0099, MDA: 0.0122)',
                    'vpin (MDI: 0.0107, MDA: 0.0119) - FIXED implementation!'
                ],
                'removed_dead_weight': [
                    'downside_vol_20 (MDA: -0.0065)',
                    'volatility_cc_20 (MDA: -0.0047)',
                    'log_return_1d (MDA: -0.0041)',
                    'ad_normalized (MDA: -0.0039)',
                    'amihud_illiquidity (MDA: -0.0023)',
                    'volatility_rs_60 (MDA: -0.0019)',
                    'market_state (MDA: -0.0014)',
                    'williams_r (MDA: -0.0003)'
                ],
                'vpin_status': 'ADDED - Now working after fixing volume bucketing bug'
            },
            'model_performance': {
                'target_sample_size': '500-1000',
                'feature_count_rationale': 'sqrt(n_samples) ≈ 30-35 features optimal',
                'expected_benefit': 'Reduced overfitting, better generalization',
                'validation_method': 'PurgedKFold with embargo, López de Prado weights'
            },
            'analysis_date': 'October 21, 2025',
            'analysis_script': 'systematic_feature_selection.py',
            'reference_docs': 'FEATURE_SELECTION_METHODOLOGY.md',
            'tickers': ['AAPL', 'NVDA', 'TSLA', 'AMZN', 'AVGO'],
            'samples': 2636,
            'core_improvements': [
                '✅ VPIN now included (was broken, now Quadrant 1 rank #5)',
                '✅ Removed 8 features with negative MDA (dead weight)',
                '✅ Balanced across 10 feature categories',
                '✅ Method 3 (Hybrid): Best balance of unique info + validation',
                '✅ 35 features optimal for sample size ~500-1000'
            ]
        },
        
        'wavenet_optimized_min': {
            'description': 'Minimal 10-feature set for fast training and low-latency deployment',
            'use_case': 'Production: Low-latency inference, quick experimentation, baseline comparison',
            'config': {
                'returns': True,           # log_return_3d
                'volatility': True,        # realized_vol_positive
                'momentum': True,          # cci
                'trend': True,             # dist_from_ma50
                'volume': True,            # volume_zscore, volume_norm
                'bollinger': True,         # bb_position, bb_percent_b
                'price_position': False,   # Not needed
                'microstructure': True,    # vpin
                'statistical': True,       # kurtosis_20
                'entropy': False,          # Not needed
                'regime': False,           # Not needed
                'risk_adjusted': False     # Not needed
            },
            'expected_features': 10,
            'notes': (
                'Minimal feature set for production deployment. '
                'Selected from Quadrant 1 (Robust: High MDI + High MDA). '
                'All features have proven predictive power with low correlation. '
                'Ideal for: Fast training, low-latency inference, baseline comparison. '
                '8 feature categories represented for signal diversity.'
            ),
            'feature_list': [
                # VOLUME (2) - Participation and liquidity signals
                'volume_zscore',           # Quadrant 1 rank #1 (MDI: 0.0104, MDA: 0.0132)
                'volume_norm',             # Quadrant 1 rank #3 (MDI: 0.0094, MDA: 0.0125)
                
                # MOMENTUM (1) - Oscillator signal
                'cci',                     # Quadrant 1 rank #2 (MDI: 0.0102, MDA: 0.0132)
                
                # BOLLINGER (2) - Price position indicators
                'bb_position',             # Quadrant 1 rank #4 (MDI: 0.0099, MDA: 0.0122)
                'bb_percent_b',            # Quadrant 1 rank #6 (MDI: 0.0094, MDA: 0.0113)
                
                # MICROSTRUCTURE (1) - Order flow toxicity
                'vpin',                    # Quadrant 1 rank #5 (MDI: 0.0107, MDA: 0.0119)
                
                # VOLATILITY (1) - Upside volatility
                'realized_vol_positive',   # Quadrant 1 rank #11 (MDI: 0.0099, MDA: 0.0084)
                
                # RETURNS (1) - Short-term momentum
                'log_return_3d',           # Quadrant 1 rank #8 (MDI: 0.0093, MDA: 0.0107)
                
                # STATISTICAL (1) - Tail risk
                'kurtosis_20',             # Quadrant 1 rank #12 (MDI: 0.0092, MDA: 0.0083)
                
                # TREND (1) - Medium-term trend
                'dist_from_ma50',          # Quadrant 1 rank #13 (MDI: 0.0094, MDA: 0.0083)
            ],
            'selection_criteria': {
                'methodology': 'Top 10 from Quadrant 1 (Robust) with category diversity',
                'quadrant_1_definition': 'High MDI + High MDA = Proven predictive power',
                'diversity_constraint': 'Represent 8 different feature categories',
                'min_mda': 0.0083,  # All features above this threshold
                'min_mdi': 0.0092,  # All features above this threshold
            },
            'model_performance': {
                'expected_speed': '3-5x faster training vs v3 (32 features)',
                'inference_latency': '<1ms per prediction',
                'sample_efficiency': 'Optimal for small datasets (<500 samples)',
                'use_case': 'Production baseline, quick iterations, low-latency trading'
            },
            'analysis_date': 'October 21, 2025',
            'analysis_source': 'systematic_feature_selection.py (Quadrant 1 analysis)',
            'comparison_to_v3': {
                'features': '10 vs 32 (69% reduction)',
                'training_speed': '3-5x faster',
                'categories': '8 vs 11 (focused)',
                'signal_quality': 'All Quadrant 1 (robust)',
                'expected_performance': '85-90% of v3 accuracy with 3x speed'
            },
            'category_breakdown': {
                'volume': 2,
                'momentum': 1,
                'bollinger': 2,
                'microstructure': 1,
                'volatility': 1,
                'returns': 1,
                'statistical': 1,
                'trend': 1
            },
            'triple_barrier_alignment': (
                'Perfect for triple barrier prediction: '
                '2 volume signals (participation), '
                '2 bollinger bands (price position), '
                '1 momentum oscillator, '
                '1 microstructure (toxicity), '
                '1 volatility (regime), '
                '1 return (direction), '
                '1 statistical (tail risk), '
                '1 trend (context). '
                'All 8 categories contribute unique information for barrier prediction.'
            )
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> dict:
        """
        Get a preset configuration by name.
        
        Args:
            preset_name: Name of preset ('minimal', 'balanced', 'comprehensive', etc.')
            
        Returns:
            Dictionary of feature group toggles and optional feature_list
            
        Raises:
            ValueError: If preset_name is not found
        """
        if preset_name not in cls.PRESETS:
            available = ', '.join(cls.PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Available presets: {available}"
            )
        
        preset = cls.PRESETS[preset_name]
        config = preset['config'].copy()
        
        # Include feature_list if present (used by wavenet_optimized)
        if 'feature_list' in preset:
            config['feature_list'] = preset['feature_list']
        
        return config
    
    @classmethod
    def get_default(cls) -> dict:
        """Get the default configuration (all features enabled)."""
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def list_presets(cls) -> None:
        """Print all available presets with descriptions."""
        print("=" * 80)
        print("AVAILABLE FEATURE PRESETS")
        print("=" * 80)
        
        for name, preset in cls.PRESETS.items():
            print(f"\n📋 {name.upper()}")
            print(f"   Description: {preset['description']}")
            print(f"   Use Case: {preset['use_case']}")
            print(f"   Expected Features: ~{preset['expected_features']}")
            
            # Count enabled groups
            enabled = sum(1 for v in preset['config'].values() if v)
            total = len(preset['config'])
            print(f"   Feature Groups: {enabled}/{total} enabled")
    
    @classmethod
    def describe_group(cls, group_name: str) -> None:
        """
        Print detailed information about a feature group.
        
        Args:
            group_name: Name of the feature group to describe
        """
        if group_name not in cls.FEATURE_GROUPS:
            available = ', '.join(cls.FEATURE_GROUPS.keys())
            raise ValueError(
                f"Unknown group '{group_name}'. "
                f"Available groups: {available}"
            )
        
        group = cls.FEATURE_GROUPS[group_name]
        
        print("=" * 80)
        print(f"FEATURE GROUP: {group_name.upper()}")
        print("=" * 80)
        print(f"Description: {group['description']}")
        print(f"Feature Count: {group['count']}")
        print(f"Importance: {group['importance']}")
        print(f"Computation: {group['computation']}")
        print(f"\nFeatures:")
        for i, feature in enumerate(group['features'], 1):
            print(f"  {i:2d}. {feature}")
    
    @classmethod
    def print_config_summary(cls, config: dict) -> None:
        """
        Print a summary of a configuration.
        
        Args:
            config: Configuration dictionary to summarize
        """
        print("=" * 80)
        print("FEATURE CONFIGURATION SUMMARY")
        print("=" * 80)
        
        # Optional keys that are not feature groups
        optional_keys = {'feature_list'}
        
        # Filter out optional keys when counting groups
        feature_groups = {k: v for k, v in config.items() if k not in optional_keys}
        enabled_groups = [k for k, v in feature_groups.items() if v]
        disabled_groups = [k for k, v in feature_groups.items() if not v]
        
        total_features = sum(
            cls.FEATURE_GROUPS[group]['count'] 
            for group in enabled_groups
        )
        
        # If feature_list is present, show that instead
        if 'feature_list' in config and config['feature_list']:
            actual_feature_count = len(config['feature_list'])
            print(f"\nEnabled Groups: {len(enabled_groups)}/{len(feature_groups)}")
            print(f"🎯 Selected Features: {actual_feature_count} (from feature_list)")
            print(f"   (Groups enabled would provide ~{total_features} features)")
        else:
            print(f"\nEnabled Groups: {len(enabled_groups)}/{len(feature_groups)}")
            print(f"Estimated Total Features: ~{total_features}")
        
        print("\n✅ ENABLED GROUPS:")
        for group in enabled_groups:
            info = cls.FEATURE_GROUPS[group]
            print(f"   • {group:20s} ({info['count']:3d} features) - {info['importance']:8s} - {info['description']}")
        
        if disabled_groups:
            print("\n❌ DISABLED GROUPS:")
            for group in disabled_groups:
                info = cls.FEATURE_GROUPS[group]
                print(f"   • {group:20s} ({info['count']:3d} features) - {info['description']}")
    
    @classmethod
    def create_custom(cls, base_preset: str = 'balanced', **overrides) -> dict:
        """
        Create a custom configuration based on a preset with overrides.
        
        Args:
            base_preset: Name of preset to use as base
            **overrides: Feature group toggles to override (e.g., entropy=True)
            
        Returns:
            Custom configuration dictionary
            
        Example:
            config = FeatureConfig.create_custom('minimal', entropy=True, regime=True)
        """
        config = cls.get_preset(base_preset)
        
        for key, value in overrides.items():
            if key not in config:
                raise ValueError(f"Unknown feature group: {key}")
            config[key] = value
        
        return config
    
    @classmethod
    def validate_config(cls, config: dict) -> tuple[bool, list]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Optional keys that are not feature groups
        optional_keys = {'feature_list'}
        
        # Check all required groups are present
        for group in cls.DEFAULT_CONFIG.keys():
            if group not in config:
                errors.append(f"Missing required group: {group}")
        
        # Check for unknown groups (excluding optional keys)
        for group in config.keys():
            if group not in cls.DEFAULT_CONFIG and group not in optional_keys:
                errors.append(f"Unknown feature group: {group}")
        
        # Check all feature group values are boolean (excluding optional keys)
        for group, value in config.items():
            if group not in optional_keys and not isinstance(value, bool):
                errors.append(f"Group '{group}' must be boolean, got {type(value)}")
        
        # Warn if no features are enabled
        feature_groups = {k: v for k, v in config.items() if k not in optional_keys}
        if all(not v for v in feature_groups.values()):
            errors.append("Warning: No feature groups are enabled!")
        
        return len(errors) == 0, errors


# Convenience function for quick access
def get_feature_config(preset: str = 'balanced') -> dict:
    """
    Quick access function to get a feature configuration.
    
    Args:
        preset: Name of preset configuration
        
    Returns:
        Configuration dictionary
    """
    return FeatureConfig.get_preset(preset)


if __name__ == "__main__":
    # Demo the feature configuration system
    print("\n" + "=" * 80)
    print("FEATURE CONFIGURATION SYSTEM DEMO")
    print("=" * 80)
    
    # Show all presets
    FeatureConfig.list_presets()
    
    # Show detailed info for a specific group
    print("\n")
    FeatureConfig.describe_group('momentum')
    
    # Create and display configurations
    print("\n")
    config = FeatureConfig.get_preset('balanced')
    FeatureConfig.print_config_summary(config)
    
    # Custom configuration example
    print("\n")
    custom = FeatureConfig.create_custom('minimal', entropy=True, regime=True)
    print("\n📝 CUSTOM CONFIGURATION (minimal + entropy + regime):")
    FeatureConfig.print_config_summary(custom)
    
    # Validation example
    print("\n" + "=" * 80)
    print("VALIDATION EXAMPLE")
    print("=" * 80)
    valid, errors = FeatureConfig.validate_config(config)
    if valid:
        print("✅ Configuration is valid!")
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
