# In bt_runner.py, add to argparse section:
ap.add_argument("--use_kelly", action="store_true", 
                help="Use Kelly Criterion for position sizing")
ap.add_argument("--kelly_method", type=str, default="bayesian",
                choices=["classical", "bayesian", "empirical", "combinatorial"],
                help="Kelly calculation method")
ap.add_argument("--kelly_fraction", type=float, default=0.25,
                help="Fraction of Kelly to use (default: 0.25)")

# In the strategy selection section, add:
elif args.use_kelly:
    from kelly_backtest_integration import KellySignalStrategy
    strat_kwargs.update({
        "entry_threshold": args.entry_threshold,
        "exit_threshold": args.exit_threshold,
        "hold_days": args.hold_days,
        "max_positions": args.max_positions,
        "use_kelly": True,
        "kelly_method": args.kelly_method,
        "kelly_fraction": args.kelly_fraction,
    })
    cerebro.addstrategy(KellySignalStrategy, **strat_kwargs)



# In ScoreThresholdStrategy class, modify the sizing section:
if self.p.use_kelly:
    from kelly_backtest_integration import EnhancedKellyPositionSizer, KellyConfig
    
    # Initialize Kelly sizer if not already done
    if not hasattr(self, 'kelly_sizer'):
        kelly_config = KellyConfig(
            method=self.p.kelly_method,
            kelly_fraction=self.p.kelly_fraction,
        )
        self.kelly_sizer = EnhancedKellyPositionSizer(kelly_config)
    
    # Calculate Kelly size
    # ... (use the Kelly calculation logic from KellySignalStrategy)


# Standard equal weight backtest
python backtest/engines/bt_runner.py \
    --config config/main.yaml \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --equal_weight

# Kelly (Bayesian) backtest
python backtest/engines/bt_runner.py \
    --config config/main.yaml \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --use_kelly \
    --kelly_method bayesian \
    --kelly_fraction 0.25

# Kelly with all parameters
python backtest/engines/bt_runner.py \
    --config config/main.yaml \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --use_kelly \
    --kelly_method bayesian \
    --kelly_fraction 0.25 \
    --entry_threshold 0.55 \
    --exit_threshold 0.50 \
    --max_positions 10 \
    --hold_days 5
