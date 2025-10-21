"""
Kelly Position Sizing Integration for Backtest Framework
Integrates with bt_runner, bt_threshold_sim, and strategy_from_signals
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import yaml
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import your Kelly module
from position_sizing_kelly import (
    KellyConfig,
    EnhancedKellyPositionSizer,
    PositionSizeResult
)

# Import backtest components
from backtest.engines.strategy_from_signals import SignalData


class KellySignalStrategy(bt.Strategy):
    """
    Enhanced strategy that uses Kelly Criterion for position sizing
    Replaces equal_weight and score_weighted with Kelly-based sizing
    """
    params = dict(
        # Entry/Exit thresholds
        entry_threshold=0.55,
        exit_threshold=0.50,
        hold_days=5,
        max_positions=10,
        
        # Kelly configuration
        use_kelly=True,
        kelly_method='bayesian',  # 'classical', 'bayesian', 'empirical', 'combinatorial'
        kelly_fraction=0.25,      # Fractional Kelly (Lopez de Prado recommends 0.25)
        
        # Kelly constraints
        kelly_min_position=0.02,  # Minimum 2% position
        kelly_max_position=0.20,  # Maximum 20% position
        kelly_min_samples=100,     # Minimum samples for Kelly calculation
        kelly_confidence=0.95,     # Confidence level for Kelly bounds
        
        # Risk parameters
        use_atr_stop=False,
        atr_period=14,
        atr_mult=0.0,
        
        # Fallback sizing (when Kelly unavailable)
        fallback_cash_per_position=0.10,
        
        # ML integration
        use_ml_confidence=True,    # Use ML scores for Kelly adjustment
        regime_adjust=True,        # Adjust for market regime
        
        # Commission
        commission_perc=0.0005,    # 5 bps
    )
    
    def __init__(self):
        # Register commission
        self.broker.setcommission(commission=self.p.commission_perc)
        
        # Initialize Kelly position sizer
        kelly_config = KellyConfig(
            method=self.p.kelly_method,
            kelly_fraction=self.p.kelly_fraction,
            min_position_size=self.p.kelly_min_position,
            max_position_size=self.p.kelly_max_position,
            min_sample_size=self.p.kelly_min_samples,
            confidence_level=self.p.kelly_confidence,
            use_ml_signals=self.p.use_ml_confidence,
            regime_adjust=self.p.regime_adjust,
        )
        self.kelly_sizer = EnhancedKellyPositionSizer(kelly_config)
        
        # Track signals and positions
        self.signals = {d: d.signal for d in self.datas}
        self.vol_scaling = {d: getattr(d, 'vol_scaling', None) for d in self.datas}
        self.holds = {d: 0 for d in self.datas}
        self.entry_price = {d: None for d in self.datas}
        
        # Kelly calculation cache
        self.kelly_cache = {}
        self.returns_history = {d._name: [] for d in self.datas}
        
        # Logging
        self.trades = []
        self.position_history = []
        self.kelly_history = []
        
        # ATR for stops if enabled
        if self.p.use_atr_stop and self.p.atr_mult > 0:
            self.atr = {d: bt.ind.ATR(d, period=self.p.atr_period) for d in self.datas}
        else:
            self.atr = {d: None for d in self.datas}
    
    def _update_returns_history(self):
        """Update returns history for Kelly calculations"""
        for d in self.datas:
            if len(d) > 1:
                ret = (d.close[0] - d.close[-1]) / d.close[-1]
                self.returns_history[d._name].append(ret)
                
                # Keep only recent history (3 years max)
                max_history = 252 * 3
                if len(self.returns_history[d._name]) > max_history:
                    self.returns_history[d._name] = self.returns_history[d._name][-max_history:]
    
    def _calculate_kelly_size(self, data, ml_score, broker_value):
        """Calculate position size using Kelly Criterion"""
        ticker = data._name
        
        # Get returns history
        returns = pd.Series(self.returns_history[ticker])
        
        # Need minimum samples
        if len(returns) < self.p.kelly_min_samples:
            # Fallback to minimum position
            return self._fallback_size(data, broker_value, reason="insufficient_history")
        
        # Prepare features if available (for regime detection)
        features = None
        if self.p.regime_adjust:
            # Could extract technical indicators here
            # For now, we'll pass None and use returns-based regime
            pass
        
        # Calculate Kelly position size
        try:
            kelly_result = self.kelly_sizer.calculate_position_size(
                returns=returns,
                ml_score=ml_score if self.p.use_ml_confidence else None,
                current_price=float(data.close[0]),
                features=features
            )
            
            # Log Kelly calculation
            self.kelly_history.append({
                'date': bt.num2date(self.datas[0].datetime[0]).strftime('%Y-%m-%d'),
                'ticker': ticker,
                'kelly_raw': kelly_result.kelly_raw,
                'kelly_adjusted': kelly_result.kelly_adjusted,
                'position_size': kelly_result.position_size,
                'sharpe': kelly_result.sharpe_ratio,
                'win_rate': kelly_result.win_rate,
                'regime': kelly_result.regime,
                'adjustments': ','.join(kelly_result.adjustments_applied)
            })
            
            # Convert percentage to shares
            position_value = broker_value * kelly_result.position_size
            shares = int(position_value / float(data.close[0]))
            
            return shares
            
        except Exception as e:
            print(f"Kelly calculation failed for {ticker}: {e}")
            return self._fallback_size(data, broker_value, reason=f"kelly_error: {e}")
    
    def _fallback_size(self, data, broker_value, reason=""):
        """Fallback sizing when Kelly unavailable"""
        target_cash = broker_value * self.p.fallback_cash_per_position
        shares = int(target_cash / float(data.close[0]))
        
        if reason:
            print(f"Using fallback sizing for {data._name}: {reason}")
        
        return shares
    
    def _open_positions(self):
        """Count open positions"""
        return sum(1 for d in self.datas if self.getposition(d).size != 0)
    
    def _record_trade(self, dt, data, action, reason, size=0, price=np.nan, score=np.nan, kelly_pct=np.nan):
        """Record trade with Kelly information"""
        pnl = np.nan
        if action == 'EXIT' and self.entry_price[data] is not None and not pd.isna(price):
            entry_p = self.entry_price[data]
            exit_p = float(price)
            pnl = (exit_p - entry_p) * abs(size)
        
        self.trades.append({
            'dt': bt.num2date(dt).strftime('%Y-%m-%d'),
            'symbol': data._name,
            'action': action,
            'size': int(size),
            'price': float(price) if not pd.isna(price) else np.nan,
            'score': float(score) if not pd.isna(score) else np.nan,
            'kelly_pct': float(kelly_pct) if not pd.isna(kelly_pct) else np.nan,
            'reason': reason,
            'pnl': float(pnl) if not pd.isna(pnl) else np.nan,
        })
    
    def next(self):
        """Main strategy logic with Kelly position sizing"""
        dt = self.datas[0].datetime[0]
        broker_val = self.broker.getvalue()
        
        # Update returns history
        self._update_returns_history()
        
        # 1) Exit logic
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size != 0:
                self.holds[d] += 1
                reason = None
                sc = float(self.signals[d][0])
                
                # Exit conditions
                if sc <= self.p.exit_threshold:
                    reason = f'score_below_exit({sc:.3f}<{self.p.exit_threshold})'
                elif self.holds[d] >= self.p.hold_days:
                    reason = f'time_exit({self.holds[d]}days)'
                elif self.p.use_atr_stop and self.atr[d] and self.entry_price[d]:
                    stop_level = self.entry_price[d] - self.p.atr_mult * self.atr[d][0]
                    if d.close[0] < stop_level:
                        reason = f'atr_stop({d.close[0]:.2f}<{stop_level:.2f})'
                
                if reason:
                    self.close(data=d)
                    self._record_trade(dt, d, 'EXIT', reason, 
                                     size=pos.size, 
                                     price=d.close[0], 
                                     score=sc)
                    self.holds[d] = 0
                    self.entry_price[d] = None
        
        # 2) Entry logic with Kelly sizing
        free_slots = max(self.p.max_positions - self._open_positions(), 0)
        if free_slots <= 0:
            return
        
        # Get candidates above entry threshold
        candidates = []
        for d in self.datas:
            if self.getposition(d).size != 0:
                continue
            score = float(self.signals[d][0])
            if score >= self.p.entry_threshold:
                candidates.append((d, score))
        
        if not candidates:
            return
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate Kelly sizes for all candidates
        sized_candidates = []
        total_allocation = 0.0
        
        for d, score in candidates[:free_slots]:
            if self.p.use_kelly:
                # Calculate Kelly position size
                size = self._calculate_kelly_size(d, score, broker_val)
                
                # Get Kelly percentage for logging
                if self.kelly_history:
                    kelly_pct = self.kelly_history[-1]['position_size']
                else:
                    kelly_pct = self.p.fallback_cash_per_position
            else:
                # Fallback to simple sizing
                size = self._fallback_size(d, broker_val, reason="kelly_disabled")
                kelly_pct = self.p.fallback_cash_per_position
            
            # Apply volatility scaling if available
            if self.vol_scaling[d]:
                vol_scale = float(self.vol_scaling[d][0])
                size = int(size * vol_scale)
            
            if size > 0:
                sized_candidates.append((d, score, size, kelly_pct))
                position_value = size * float(d.close[0])
                total_allocation += position_value / broker_val
        
        # Normalize if over-allocated (optional)
        if total_allocation > 1.0:
            print(f"Over-allocated: {total_allocation:.1%}, normalizing...")
            normalization_factor = 0.95 / total_allocation  # Target 95% allocation
            sized_candidates = [
                (d, score, int(size * normalization_factor), kelly_pct)
                for d, score, size, kelly_pct in sized_candidates
            ]
        
        # Execute orders
        for d, score, size, kelly_pct in sized_candidates:
            if size > 0:
                self.buy(data=d, size=size)
                self.entry_price[d] = float(d.close[0])
                self.holds[d] = 0
                self._record_trade(dt, d, 'ENTRY', 
                                 f'kelly_{kelly_pct:.1%}',
                                 size=size,
                                 price=d.close[0],
                                 score=score,
                                 kelly_pct=kelly_pct)
        
        # Track daily position count
        self.position_history.append({
            'date': bt.num2date(dt).strftime('%Y-%m-%d'),
            'open_positions': self._open_positions(),
            'total_allocation': sum(
                self.getposition(d).size * d.close[0] / broker_val
                for d in self.datas if self.getposition(d).size != 0
            )
        })
    
    def stop(self):
        """Save results for analysis"""
        self._result_trades = pd.DataFrame(self.trades)
        self._result_positions = pd.DataFrame(self.position_history)
        self._result_kelly = pd.DataFrame(self.kelly_history)


def add_kelly_to_bt_runner(args_namespace):
    """
    Modify bt_runner arguments to use Kelly sizing
    
    Usage in bt_runner.py:
        from kelly_backtest_integration import add_kelly_to_bt_runner
        args = add_kelly_to_bt_runner(args)
    """
    # Add Kelly-specific arguments
    setattr(args_namespace, 'use_kelly', True)
    setattr(args_namespace, 'kelly_method', 'bayesian')
    setattr(args_namespace, 'kelly_fraction', 0.25)
    setattr(args_namespace, 'kelly_min_position', 0.02)
    setattr(args_namespace, 'kelly_max_position', 0.20)
    
    # Override weighting flags
    setattr(args_namespace, 'equal_weight', False)
    setattr(args_namespace, 'score_weighted', False)
    
    return args_namespace


def create_kelly_cerebro(config_path: str,
                        start_date: str,
                        end_date: str,
                        initial_cash: float = 100000,
                        kelly_params: Dict = None) -> bt.Cerebro:
    """
    Create a Cerebro instance configured for Kelly position sizing
    
    Args:
        config_path: Path to ML config file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_cash: Starting capital
        kelly_params: Override Kelly parameters
        
    Returns:
        Configured Cerebro instance
    """
    cerebro = bt.Cerebro()
    
    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0005)  # 5 bps
    
    # Load data with signals (using your existing loader)
    from backtest.engines.bt_threshold_sim import load_panel_with_signals
    
    panel = load_panel_with_signals(
        config_path=config_path,
        start=start_date,
        end=end_date,
        enable_vol_targeting=False  # Kelly handles its own scaling
    )
    
    # Add data feeds
    for ticker, df in panel.items():
        data_feed = SignalData(
            dataname=df,
            name=ticker
        )
        cerebro.adddata(data_feed)
    
    # Configure Kelly strategy parameters
    strategy_params = {
        'entry_threshold': 0.55,
        'exit_threshold': 0.50,
        'hold_days': 5,
        'max_positions': 10,
        'use_kelly': True,
        'kelly_method': 'bayesian',
        'kelly_fraction': 0.25,
        'kelly_min_position': 0.02,
        'kelly_max_position': 0.20,
        'use_ml_confidence': True,
        'regime_adjust': True,
    }
    
    # Override with custom params
    if kelly_params:
        strategy_params.update(kelly_params)
    
    # Add strategy
    cerebro.addstrategy(KellySignalStrategy, **strategy_params)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                       timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timeret',
                       timeframe=bt.TimeFrame.Days)
    
    return cerebro


def run_kelly_backtest_comparison(config_path: str,
                                 start_date: str,
                                 end_date: str,
                                 initial_cash: float = 100000) -> pd.DataFrame:
    """
    Run comparison of different position sizing methods
    
    Returns DataFrame comparing:
    - Equal weight
    - Score weighted  
    - Kelly (classical)
    - Kelly (Bayesian)
    - Kelly (empirical)
    """
    results = []
    
    # Test configurations
    test_configs = [
        {'name': 'Equal_Weight', 'params': {
            'use_kelly': False,
            'fallback_cash_per_position': 0.10
        }},
        {'name': 'Score_Weighted', 'params': {
            'use_kelly': False,
            'fallback_cash_per_position': 0.10,
            # Would need to implement score weighting in fallback
        }},
        {'name': 'Kelly_Classical', 'params': {
            'use_kelly': True,
            'kelly_method': 'classical',
            'kelly_fraction': 0.25
        }},
        {'name': 'Kelly_Bayesian', 'params': {
            'use_kelly': True,
            'kelly_method': 'bayesian',
            'kelly_fraction': 0.25
        }},
        {'name': 'Kelly_Empirical', 'params': {
            'use_kelly': True,
            'kelly_method': 'empirical',
            'kelly_fraction': 0.25
        }},
        {'name': 'Kelly_Combinatorial', 'params': {
            'use_kelly': True,
            'kelly_method': 'combinatorial',
            'kelly_fraction': 0.25
        }},
    ]
    
    for config in test_configs:
        print(f"\nRunning {config['name']}...")
        
        try:
            # Create and run backtest
            cerebro = create_kelly_cerebro(
                config_path=config_path,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                kelly_params=config['params']
            )
            
            run_results = cerebro.run()
            strat = run_results[0]
            
            # Extract metrics
            final_value = cerebro.broker.getvalue()
            total_return = (final_value / initial_cash - 1) * 100
            
            # Get analyzers
            ta = strat.analyzers.ta.get_analysis()
            sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            dd = strat.analyzers.dd.get_analysis()
            max_dd = abs(dd.max.drawdown) if dd.max.drawdown else 0
            
            # Get trade stats
            total_trades = ta.total.total if hasattr(ta.total, 'total') else 0
            won = ta.won.total if hasattr(ta.won, 'total') else 0
            lost = ta.lost.total if hasattr(ta.lost, 'total') else 0
            win_rate = (won / (won + lost) * 100) if (won + lost) > 0 else 0
            
            # Kelly-specific metrics
            kelly_stats = {}
            if hasattr(strat, '_result_kelly') and not strat._result_kelly.empty:
                kelly_df = strat._result_kelly
                kelly_stats = {
                    'avg_kelly_raw': kelly_df['kelly_raw'].mean(),
                    'avg_kelly_adjusted': kelly_df['kelly_adjusted'].mean(),
                    'avg_position_size': kelly_df['position_size'].mean(),
                    'avg_sharpe': kelly_df['sharpe'].mean(),
                }
            
            results.append({
                'Method': config['name'],
                'Final_Value': final_value,
                'Total_Return_%': total_return,
                'Sharpe_Ratio': sharpe,
                'Max_Drawdown_%': max_dd,
                'Total_Trades': total_trades,
                'Win_Rate_%': win_rate,
                **kelly_stats
            })
            
        except Exception as e:
            print(f"Error running {config['name']}: {e}")
            results.append({
                'Method': config['name'],
                'Error': str(e)
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    return comparison_df


# Integration with bt_threshold_sim
def enhance_bt_threshold_with_kelly(run_backtest_func):
    """
    Decorator to add Kelly sizing to bt_threshold_sim's run_backtest
    
    Usage:
        from kelly_backtest_integration import enhance_bt_threshold_with_kelly
        run_backtest = enhance_bt_threshold_with_kelly(run_backtest)
    """
    def wrapper(*args, **kwargs):
        # Check if Kelly is requested
        use_kelly = kwargs.pop('use_kelly', False)
        
        if use_kelly:
            print("Using Kelly position sizing...")
            # Modify kwargs to use Kelly strategy
            kwargs['score_weighted'] = False  # Disable score weighting
            kwargs['equal_weight'] = False    # Disable equal weighting
            
            # Add Kelly parameters
            kwargs['kelly_method'] = kwargs.get('kelly_method', 'bayesian')
            kwargs['kelly_fraction'] = kwargs.get('kelly_fraction', 0.25)
            
        return run_backtest_func(*args, **kwargs)
    
    return wrapper


# Example usage script
if __name__ == "__main__":
    # Example: Run comparison of sizing methods
    comparison = run_kelly_backtest_comparison(
        config_path='config/main.yaml',
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_cash=100000
    )
    
    print("\n" + "="*60)
    print("POSITION SIZING COMPARISON")
    print("="*60)
    print(comparison.to_string())
    
    # Save results
    comparison.to_csv('kelly_comparison_results.csv', index=False)
    print(f"\nResults saved to kelly_comparison_results.csv")
