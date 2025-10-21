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
            'expected_features': 38,
            'notes': 'Generated from equity_feature_importance_analysis.ipynb. Ranked by consensus across MDI, MDA, and orthogonal MDA. Market features GC=F and ^FVX excluded (added separately via MARKET_FEATURES config).',
            'feature_list': [
                # Consensus 3/3 (appears in all three importance methods)
                'kurtosis_60',
                # Consensus 2/3 (appears in two of three methods)
                'macd_signal', 'dist_from_ma50', 'macd', 'macd_hist',
                'dollar_volume_ma_ratio', 'bb_lower', 'volume_zscore',
                'bb_position',
                # Consensus 1/3 but high average rank
                'volatility_ht_60', 'cci', 'dist_from_ma20', 'relative_volume',
                'volume_norm', 'sar', 'dist_from_ma10', 'serial_corr_1',
                'log_return_2d', 'skewness_20', 'vwap_20', 'bb_upper',
                'macd_divergence', 'log_return_3d', 'log_return_5d', 'rsi_7',
                'stoch_k_d_diff', 'hl_volatility_ratio', 'ad_roc',
                'relative_volatility', 'serial_corr_5', 'skewness_60',
                'bb_percent_b', 'variance_ratio', 'amihud_illiq',
                'log_return_10d', 'risk_adj_momentum_20', 'max_drawdown_20',
                'adx_plus'
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
