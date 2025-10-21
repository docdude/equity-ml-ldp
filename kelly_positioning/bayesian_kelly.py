"""Bayesian Kelly Trader for Power Markets.

This module implements a Bayesian approach to the Kelly Criterion for position
sizing in power market futures trading. It continuously updates probability
estimates using Bayesian inference and computes optimal position sizes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceData:
    """
    Immutable container for price data.
    
    Attributes:
        prices: 
            Array of price values.
        dates: 
            Optional array of corresponding dates.
        symbol: 
            Optional identifier for the price series.
    """
    prices: np.ndarray
    dates: Optional[np.ndarray] = None
    symbol: Optional[str] = None


@dataclass(frozen=True)
class TradeResult:
    """
    Immutable container for a single trade result.
    
    Attributes:
        entry_price: 
            Price at trade entry.
        exit_price: 
            Price at trade exit.
        position_size: 
            Fraction of capital allocated to the trade.
        gain_loss: 
            Realized profit/loss amount.
    """
    entry_price: float
    exit_price: float
    position_size: float
    gain_loss: float


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Immutable container for trading performance metrics.
    
    Attributes:
        final_capital: 
            Ending capital amount.
        total_return_pct: 
            Total return percentage.
        sharpe_ratio: 
            Annualized Sharpe ratio.
        max_drawdown_pct: 
            Maximum drawdown percentage.
        win_rate_pct: 
            Percentage of winning trades.
        trades_count: 
            Total number of trades.
    """
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades_count: int


@dataclass
class BayesianKellyTrader:
    """
    Implements a Bayesian approach to Kelly Criterion for position sizing.
    
    Uses Bayesian updating of Beta distribution parameters to dynamically
    estimate win probability, and applies the Kelly formula to calculate
    optimal position sizes.
    
    Attributes:
        alpha: 
            Beta distribution parameter for "up" moves.
        beta: 
            Beta distribution parameter for "down" moves.
        expected_gain: 
            Expected percentage gain on winning trades.
        expected_loss: 
            Expected percentage loss on losing trades.
        kelly_fraction: 
            Fraction of full Kelly to use (conservatism factor).
        probability_history: 
            History of win probability estimates.
        kelly_bet_history: 
            History of calculated Kelly fractions.
        price_data: 
            Processed price data if available.
    """
    alpha: float = 1.0
    beta: float = 1.0
    expected_gain: float = 0.015
    expected_loss: float = 0.015
    kelly_fraction: float = 0.50
    probability_history: List[float] = field(default_factory=list)
    kelly_bet_history: List[float] = field(default_factory=list)
    price_data: Optional[PriceData] = None
    
    def __post_init__(self) -> None:
        """Validate initialization parameters."""
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Alpha and beta must be positive")
        if self.expected_gain <= 0:
            raise ValueError("Expected gain must be positive")
        if self.expected_loss <= 0:
            raise ValueError("Expected loss must be positive")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("Kelly fraction must be between 0 and 1")
    
    def update_belief(self, price_up: bool) -> None:
        """
        Update the Bayesian belief based on price movement.
        
        Args:
            price_up: True if price moved up, False if down.
        """
        if price_up:
            self.alpha += 1
        else:
            self.beta += 1
            
    def get_win_probability(self) -> float:
        """
        Calculate the current win probability from the Beta distribution.
        
        Returns:
            Probability of price moving up (mean of Beta distribution).
        """
        return self.alpha / (self.alpha + self.beta)
    
    def calculate_kelly_fraction(self) -> float:
        """
        Calculate the optimal Kelly position size.
        
        Returns:
            Recommended position size as fraction of capital (0 to 1).
        """
        p: float = self.get_win_probability()
        q: float = 1 - p
        b: float = self.expected_gain
        loss: float = self.expected_loss
        
        try:
            # Original Kelly formula
            kelly: float = (p * b - q * loss) / (b * loss)
            
            # Apply fractional Kelly to avoid overbetting
            kelly: float = np.clip(kelly, -1.0, 1.0) * self.kelly_fraction
            
            return kelly
        except ZeroDivisionError:
            logger.warning("Division by zero in Kelly calculation")
            return 0.0
    
    def process_data(self, price_data: PriceData) -> None:
        """
        Process a series of price data and update beliefs.
        
        Args:
            price_data: Container with price data to process.
            
        Raises:
            ValueError: If price data contains fewer than 2 observations.
        """
        if len(price_data.prices) < 2:
            raise ValueError("Price data must contain at least 2 observations")
        
        # Store the price data
        self.price_data = price_data
        prices: np.ndarray = price_data.prices
        
        # Reset histories
        self.probability_history = []
        self.kelly_bet_history = []
        
        try:
            # Calculate returns
            returns: np.ndarray = np.diff(prices) / prices[:-1]
            
            # Process each day's return
            for ret in returns:
                # Store current probability and Kelly bet before update
                self.probability_history.append(self.get_win_probability())
                self.kelly_bet_history.append(self.calculate_kelly_fraction())
                
                # Update belief based on price movement
                self.update_belief(ret > 0)
                
        except Exception as e:
            logger.error(f"Error processing price data: {e}")
            raise
    
    def plot_results(self, rolling_window: int = 20) -> None:
        """
        Plot the results of the Bayesian updating process.
        
        Args:
            rolling_window: Window size for the rolling frequency estimate.
            
        Raises:
            ValueError: If no price data has been processed.
        """
        if self.price_data is None:
            raise ValueError("No price data has been processed")
            
        prices: np.ndarray = self.price_data.prices
        returns: np.ndarray = np.diff(prices) / prices[:-1]
        
        # Calculate rolling frequency estimate
        rolling_up_prob: np.ndarray = pd.Series(returns > 0).rolling(
            rolling_window
        ).mean()
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Win Probability Estimates
        plt.subplot(3, 1, 1)
        plt.plot(
            self.probability_history, 
            label='Bayesian Estimate', 
            color='blue', 
            linewidth=2
        )
        plt.plot(rolling_up_prob, 
            label=f'{rolling_window}-day Rolling', 
            color='red', 
            linestyle='--'
        )
        plt.title('Win Probability Estimates')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Kelly Betting Fraction
        plt.subplot(3, 1, 2)
        plt.plot(
            self.kelly_bet_history, 
            label='Kelly Fraction', 
            color='green', 
            linewidth=2
        )
        plt.title('Kelly Position Sizing')
        plt.ylabel('Fraction of Capital')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Price History
        plt.subplot(3, 1, 3)
        plt.plot(prices, label='Price', color='black')
        plt.title('Price History')
        plt.ylabel('Price')
        plt.xlabel('Trading Days')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def simulate_trading(
        self, 
        initial_capital: float = 10000.0
    ) -> PerformanceMetrics:
        """
        Simulate trading performance using the Bayesian Kelly approach.
        
        Args:
            initial_capital: Initial capital to start trading with.
            
        Returns:
            PerformanceMetrics object with simulation results.
            
        Raises:
            ValueError: 
                If no price data has been processed or probability
                history is insufficient.
        """
        if self.price_data is None:
            raise ValueError("No price data has been processed")
            
        if len(self.probability_history) < 1:
            raise ValueError("Insufficient probability history")
            
        prices: np.ndarray = self.price_data.prices
        returns: np.ndarray = np.diff(prices) / prices[:-1]
        
        # Initialize capital and tracking
        capital: float = initial_capital
        capital_history: List[float] = [capital]
        position_sizes: List[float] = []
        
        # Run trading simulation
        for i, ret in enumerate(returns[:-1]):  # Use all but last return
            # Use probability and Kelly fraction for next trade
            kelly = self.kelly_bet_history[i+1]
            
            # Calculate position size
            position_size: float = capital * kelly
            position_sizes.append(position_size)
            
            # Update capital based on return
            capital *= (1 + kelly * returns[i+1])
            capital_history.append(capital)
        
        # Calculate performance metrics
        returns_series = pd.Series(np.diff(capital_history) / 
                                  capital_history[:-1])
        
        try:
            metrics: PerformanceMetrics = PerformanceMetrics(
                final_capital=capital,
                total_return_pct=(capital / initial_capital - 1) * 100,
                sharpe_ratio=(returns_series.mean() / returns_series.std() * 
                             np.sqrt(252)) if returns_series.std() > 0 else 0,
                max_drawdown_pct=(1 - pd.Series(capital_history) / 
                                pd.Series(capital_history).cummax()).max() * 100,
                win_rate_pct=(returns_series > 0).mean() * 100,
                trades_count=len(returns_series)
            )
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
        
        # Create performance plot
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Equity Curve
        plt.subplot(2, 1, 1)
        plt.plot(capital_history, label='Equity Curve', 
                 color='blue', linewidth=2)
        plt.title('Trading Simulation: Bayesian Kelly Strategy')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Position Sizes
        plt.subplot(2, 1, 2)
        plt.bar(
            range(len(position_sizes)), 
            position_sizes,
            color='green', 
            alpha=0.7
        )
        plt.title('Position Sizes')
        plt.ylabel('Position Size ($)')
        plt.xlabel('Trading Days')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return metrics


def generate_power_futures_prices(
    T: int = 90,                  # Days until contract delivery
    F0: float = 100.0,            # Initial futures price
    mu_final: float = 90.0,       # Final expected spot/futures price at delivery
    kappa: float = 0.3,           # Mean reversion speed
    sigma_initial: float = 0.20,  # Initial volatility
    sigma_final: float = 0.05,    # Final volatility (lower closer to delivery)
    jump_lambda: float = 0.05,    # Probability of jump per day
    jump_mu: float = 0.02,        # Mean jump size (2% upwards on average)
    jump_sigma: float = 0.03,     # Std. dev. of jump size
    dt: float = 1/252,            # Daily increments (trading days)
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generate synthetic power futures prices using mean-reverting jump 
    diffusion.
    
    Implements a stochastic process model for power futures with:
    - Mean reversion towards a time-dependent equilibrium level
    - Time-dependent volatility that decreases as delivery approaches
    - Jump component to simulate price spikes
    
    Args:
        T: 
            Days until contract delivery.
        F0: 
            Initial futures price.
        mu_final: 
            Final expected spot/futures price at delivery.
        kappa: 
            Mean reversion speed.
        sigma_initial: 
            Initial volatility.
        sigma_final: 
            Final volatility (lower closer to delivery).
        jump_lambda: 
            Probability of jump per day.
        jump_mu: 
            Mean jump size (2% upwards on average).
        jump_sigma: 
            Standard deviation of jump size.
        dt: 
            Daily increments (trading days).
        seed: 
            Random seed for reproducibility.
        
    Returns:
        Array of simulated daily futures prices.
    """
    if seed is not None:
        np.random.seed(seed)
        
    days: int = int(T)
    prices: np.ndarray = np.zeros(days)
    prices[0] = F0
    
    for t in range(1, days):
        # Time-dependent mean and volatility
        time_factor: float = np.exp(-0.05 * t)  # Exponential decay factor
        
        # Mean converges toward mu_final as t approaches T
        mu_t = mu_final + (F0 - mu_final) * time_factor
        
        # Volatility decreases as t approaches T
        sigma_t = sigma_final + (sigma_initial - sigma_final) * time_factor
        
        # Mean reversion drift
        drift = kappa * (mu_t - prices[t-1]) * dt
        
        # Continuous volatility component
        diffusion = sigma_t * np.sqrt(dt) * np.random.normal()
        
        # Jump component (Poisson arrival)
        jump = 0.0
        if np.random.rand() < jump_lambda * dt:
            # Log-normal jump size
            jump_size = np.random.lognormal(
                mean=jump_mu - 0.5 * jump_sigma**2, 
                sigma=jump_sigma
            ) - 1.0
            jump = prices[t-1] * jump_size
        
        # Update price
        prices[t] = prices[t-1] + drift + prices[t-1] * diffusion + jump
        
        # Floor the price at 1.0
        prices[t] = max(1.0, prices[t])
    
    return prices


def run_demo(
    show_plots: bool = True,
    alpha_prior: float = 3.0,
    beta_prior: float = 2.0,
    expected_gain: float = 0.01,
    expected_loss: float = 0.015,
    kelly_fraction: float = 0.5,
    initial_capital: float = 10000.0,
    price_params: Optional[Dict[str, Any]] = None
) -> Tuple[BayesianKellyTrader, PerformanceMetrics]:
    """
    Run a demonstration of the Bayesian Kelly trading approach.
    
    Args:
        show_plots: 
            Whether to display the visualization plots.
        alpha_prior: 
            Prior parameter for "up" moves.
        beta_prior: 
            Prior parameter for "down" moves.
        expected_gain: 
            Expected percentage gain on winning trades.
        expected_loss: 
            Expected percentage loss on losing trades.
        kelly_fraction: 
            Fraction of full Kelly to use.
        initial_capital: 
            Initial capital for simulation.
        price_params: 
            Optional parameters for price generation.
        
    Returns:
        Tuple of (trader instance, performance metrics).
    """
    try:
        # Initialize the trader
        trader: BayesianKellyTrader = BayesianKellyTrader(
            alpha=alpha_prior,
            beta=beta_prior,
            expected_gain=expected_gain,
            expected_loss=expected_loss,
            kelly_fraction=kelly_fraction
        )
        
        # Generate power futures prices
        params: Dict = price_params or {}
        prices: np.ndarray = generate_power_futures_prices(**params)
        
        # Create a price data object
        price_data: PriceData = PriceData(
            prices=prices,
            symbol="ERCOT_NORTH_AUG"
        )
        
        # Process the data
        trader.process_data(price_data)
        
        # Plot the results if requested
        if show_plots:
            trader.plot_results(rolling_window=20)
        
        # Simulate trading performance
        performance = trader.simulate_trading(initial_capital=initial_capital)
        
        # Print performance metrics
        print("\nPerformance Metrics:")
        for field_name, field_value in performance.__dict__.items():
            print(f"{field_name.replace('_', ' ').title()}: {field_value:.2f}")
            
        return trader, performance
        
    except Exception as e:
        logger.error(f"Error running demonstration: {e}")
        raise


if __name__ == "__main__":
    # Example usage with default parameters
    trader, metrics = run_demo(
        alpha_prior=1.0,              # More bearish prior (20% up probability)
        beta_prior=4.0,
        expected_gain=0.015,
        expected_loss=0.015,
        kelly_fraction=0.25,          # Quarter-Kelly for safety
        price_params={
            'T': 90,                  # 90 days until delivery
            'F0': 125.0,              # Initial price $125/MWh
            'mu_final': 90.0,         # Expected to rise slightly
            'kappa': 0.5,             # Mean reversion speed
            'sigma_initial': 0.35,    # Higher initial volatility
            'sigma_final': 0.15,      # Lower volatility near delivery
            'jump_lambda': 0.03,      # Higher jump frequency for power
            'jump_mu': 0.03,          # Larger mean jump size
            'jump_sigma': 0.05,       # Larger jump volatility
            'seed': 42                # For reproducibility
        }
    )