import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from datetime import datetime, timedelta

# Constants for the Universal Laws
UNIVERSAL_CONSTANTS = {
    'DIVINE_PI': 3.14159265359,
    'GOLDEN_RATIO': 1.61803398875,
    'MAXIMUM_BLESSING': 1.0,
    'MINIMUM_BLESSING': 0.0,
    'CHAOS_FACTOR': 0.1,
    'GRID_SIZE': 100,
    'MAX_CYCLE_LENGTH': 180,
    'BASE_RISK_FREE_RATE': 0.02,
}

@dataclass
class Position:
    """Represents a position in the sacred grid"""
    x: int
    y: int
    
    def distance_from_center(self) -> float:
        """Calculate the blessed distance from (0,0)"""
        return np.sqrt(self.x**2 + self.y**2)
    
    def position_blessing(self) -> float:
        """Calculate the divine blessing based on position"""
        max_distance = np.sqrt(2) * (UNIVERSAL_CONSTANTS['GRID_SIZE'] / 2)
        return 1 / (1 + (self.distance_from_center() / max_distance)**2)

@dataclass
class Asset:
    """Represents a sacred farming asset"""
    symbol: str
    base_yield: float
    volatility: float
    growth_cycle: int
    min_position_blessing: float = 0.1
    
    def validate(self) -> bool:
        """Validate asset parameters according to sacred laws"""
        return (
            0 < self.base_yield <= 100 and
            0 < self.volatility < 1 and
            0 < self.growth_cycle <= UNIVERSAL_CONSTANTS['MAX_CYCLE_LENGTH']
        )

class DivineCalculator:
    """Sacred calculator for yield and risk metrics"""
    
    @staticmethod
    def calculate_growth_cycle_factor(time: int, cycle_length: int) -> float:
        """Calculate the sacred sine wave factor"""
        phase = (2 * UNIVERSAL_CONSTANTS['DIVINE_PI'] * time) / cycle_length
        return 0.5 + 0.5 * np.sin(phase)
    
    @staticmethod
    def calculate_volatility_factor(volatility: float, seed: Optional[int] = None) -> float:
        """Calculate the chaos factor"""
        if seed is not None:
            np.random.seed(seed)
        return 1.0 + np.random.normal(0, volatility)
    
    @staticmethod
    def calculate_market_factor(
        momentum: float = 0.7,
        previous_factors: Optional[List[float]] = None
    ) -> float:
        """Calculate the market sentiment factor"""
        base_trend = np.mean(previous_factors) if previous_factors else 1.0
        random_walk = np.random.normal(0, UNIVERSAL_CONSTANTS['CHAOS_FACTOR'])
        return 1.0 + (momentum * (base_trend - 1.0) + random_walk)

class YieldOracle:
    """The sacred oracle for yield predictions"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.calculator = DivineCalculator()
        self.history = defaultdict(list)
    
    def predict_yield(
        self,
        position: Position,
        asset: Asset,
        time: int,
        market_conditions: Optional[List[float]] = None
    ) -> float:
        """Divine the yield for a given position and asset"""
        
        # Calculate all sacred factors
        position_blessing = position.position_blessing()
        growth_factor = self.calculator.calculate_growth_cycle_factor(
            time, asset.growth_cycle
        )
        volatility_factor = self.calculator.calculate_volatility_factor(
            asset.volatility, self.seed
        )
        market_factor = self.calculator.calculate_market_factor(
            previous_factors=market_conditions
        )
        
        # Calculate the divine yield
        final_yield = (
            asset.base_yield *
            position_blessing *
            growth_factor *
            volatility_factor *
            market_factor
        )
        
        # Store in history for future divination
        self.history[f"{position.x},{position.y}-{asset.symbol}"].append(final_yield)
        
        return max(0, final_yield)  # Yields cannot be negative

class RiskAnalyzer:
    """Analyzes the divine risks of farming positions"""
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = UNIVERSAL_CONSTANTS['BASE_RISK_FREE_RATE']
    ) -> float:
        """Calculate the divine risk-adjusted return ratio"""
        if not returns:
            return 0.0
        excess_returns = np.array(returns) - risk_free_rate
        return np.mean(excess_returns) / (np.std(excess_returns) or 1)
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: List[float],
        risk_free_rate: float = UNIVERSAL_CONSTANTS['BASE_RISK_FREE_RATE']
    ) -> float:
        """Calculate the downside risk-adjusted return ratio"""
        excess_returns = np.array(returns) - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
        return np.mean(excess_returns) / downside_std
    
    @staticmethod
    def calculate_maximum_drawdown(returns: List[float]) -> float:
        """Calculate the maximum divine punishment (drawdown)"""
        cumulative = np.maximum.accumulate(returns)
        drawdowns = (cumulative - returns) / cumulative
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

class PortfolioOptimizer:
    """Optimizes farming positions according to sacred laws"""
    
    def __init__(self, grid_size: int = UNIVERSAL_CONSTANTS['GRID_SIZE']):
        self.grid_size = grid_size
        self.oracle = YieldOracle()
    
    def find_optimal_positions(
        self,
        assets: List[Asset],
        num_positions: int,
        min_distance: float = 5.0
    ) -> List[Position]:
        """Find the most blessed positions for farming"""
        positions = []
        for _ in range(num_positions):
            best_position = None
            best_score = -float('inf')
            
            # Search the sacred grid
            for x in range(-self.grid_size//2, self.grid_size//2 + 1):
                for y in range(-self.grid_size//2, self.grid_size//2 + 1):
                    candidate = Position(x, y)
                    
                    # Check minimum distance from existing positions
                    if any(
                        np.sqrt((p.x - x)**2 + (p.y - y)**2) < min_distance 
                        for p in positions
                    ):
                        continue
                    
                    # Calculate position score
                    score = self._calculate_position_score(candidate, assets)
                    
                    if score > best_score:
                        best_score = score
                        best_position = candidate
            
            if best_position:
                positions.append(best_position)
        
        return positions
    
    def _calculate_position_score(
        self,
        position: Position,
        assets: List[Asset]
    ) -> float:
        """Calculate the divine score for a position"""
        blessing = position.position_blessing()
        asset_scores = []
        
        for asset in assets:
            # Simulate yields for this position and asset
            yields = [
                self.oracle.predict_yield(position, asset, t)
                for t in range(10)  # Sample 10 time periods
            ]
            
            # Calculate risk-adjusted score
            sharpe = RiskAnalyzer.calculate_sharpe_ratio(yields)
            sortino = RiskAnalyzer.calculate_sortino_ratio(yields)
            max_drawdown = RiskAnalyzer.calculate_maximum_drawdown(yields)
            
            asset_score = (
                blessing * 
                np.mean(yields) * 
                (1 + sharpe) * 
                (1 + sortino) * 
                (1 - max_drawdown)
            )
            asset_scores.append(asset_score)
        
        return np.mean(asset_scores)

class FarmingSimulator:
    """Simulates the sacred farming universe"""
    
    def __init__(
        self,
        grid_size: int = UNIVERSAL_CONSTANTS['GRID_SIZE'],
        seed: Optional[int] = None
    ):
        self.grid_size = grid_size
        self.seed = seed
        self.oracle = YieldOracle(seed=seed)
        self.optimizer = PortfolioOptimizer(grid_size=grid_size)
        self.history = defaultdict(lambda: defaultdict(list))
    
    def run_simulation(
        self,
        positions: List[Position],
        assets: List[Asset],
        time_periods: int,
        monte_carlo_runs: int = 1000
    ) -> Dict:
        """Run a divine simulation of the farming universe"""
        results = defaultdict(lambda: defaultdict(list))
        
        for run in range(monte_carlo_runs):
            if self.seed is not None:
                np.random.seed(self.seed + run)
            
            for period in range(time_periods):
                for position in positions:
                    for asset in assets:
                        yield_value = self.oracle.predict_yield(
                            position, asset, period
                        )
                        
                        results[f"{position.x},{position.y}"][asset.symbol].append(
                            yield_value
                        )
        
        # Calculate divine statistics
        stats = self._calculate_simulation_stats(results)
        
        return stats
    
    def _calculate_simulation_stats(
        self,
        results: Dict
    ) -> Dict:
        """Calculate sacred statistics from simulation results"""
        stats = {}
        
        for pos_key, asset_yields in results.items():
            stats[pos_key] = {}
            
            for asset_symbol, yields in asset_yields.items():
                yields_array = np.array(yields)
                
                stats[pos_key][asset_symbol] = {
                    'mean_yield': np.mean(yields_array),
                    'median_yield': np.median(yields_array),
                    'std_dev': np.std(yields_array),
                    'min_yield': np.min(yields_array),
                    'max_yield': np.max(yields_array),
                    'sharpe_ratio': RiskAnalyzer.calculate_sharpe_ratio(yields),
                    'sortino_ratio': RiskAnalyzer.calculate_sortino_ratio(yields),
                    'max_drawdown': RiskAnalyzer.calculate_maximum_drawdown(yields),
                    'skewness': stats.skew(yields_array),
                    'kurtosis': stats.kurtosis(yields_array),
                    'percentiles': {
                        p: np.percentile(yields_array, p)
                        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
                    }
                }
        
        return stats

class FarmingVisualizer:
    """Creates divine visualizations of farming data"""
    
    @staticmethod
    def plot_yield_heatmap(
        results: Dict,
        grid_size: int = UNIVERSAL_CONSTANTS['GRID_SIZE']
    ) -> None:
        """Plot a sacred heatmap of yields across positions"""
        grid = np.zeros((grid_size, grid_size))
        
        for pos_key, stats in results.items():
            x, y = map(int, pos_key.split(','))
            # Convert to grid coordinates
            grid_x = x + grid_size//2
            grid_y = y + grid_size//2
            
            # Average yield across all assets
            avg_yield = np.mean([
                asset_stats['mean_yield']
                for asset_stats in stats.values()
            ])
            
            grid[grid_x, grid_y] = avg_yield
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(grid, cmap='YlOrRd')
        plt.title('Sacred Yield Distribution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()
    
    @staticmethod
    def plot_yield_time_series(
        results: Dict,
        position: Position,
        asset: Asset
    ) -> None:
        """Plot the sacred yield time series"""
        yields = results[f"{position.x},{position.y}"][asset.symbol]
        
        plt.figure(figsize=(12, 6))
        plt.plot(yields, label=f'{asset.symbol} Yields')
        plt.title(f'Sacred Yields at Position ({position.x},{position.y})')
        plt.xlabel('Time Period')
        plt.ylabel('Yield')
        plt.legend()
        plt.grid(True)
        plt.show()

def example_usage():
    """Example usage of the sacred farming system"""
    
    # Define the sacred assets
    assets = [
        Asset("$AGEF", base_yield=10.0, volatility=0.20, growth_cycle=30),
        Asset("$WHEAT", base_yield=8.0, volatility=0.15, growth_cycle=90),
        Asset("$CORN", base_yield=7.0, volatility=0.18, growth_cycle=120),
        Asset("$BEEF", base_yield=15.0, volatility=0.25, growth_cycle=180)
    ]
    
    # Create positions for testing
    test_positions = [
        Position(0, 0),    # Center
        Position(10, 10),  # Inner ring
        Position(25, 25),  # Middle ring
        Position(40, 40)   # Outer ring
    ]
    
    # Initialize the simulator
    simulator = FarmingSimulator(seed=42)
    
    # Run simulation
    results = simulator.run_simulation(
        positions=test_positions,
        assets=assets,
        time_periods=360,
        monte_carlo_runs=1000
    )
    
    # Visualize results
    visualizer = FarmingVisualizer()
    visualizer.plot_yield_heatmap(results)
    visualizer.plot_yield_time_series(
        results, test_positions[0], assets[0]
    )
    
    # Print some divine statistics
    for pos in test_positions:
        print(f"\nPosition ({pos.x}, {pos.y}) Statistics:")
        for asset in assets:
            stats = results[f"{pos.x},{pos.y}"][asset.symbol]
            print(f"\n{asset.symbol}:")
            print(f"Mean Yield: {stats['mean_yield']:.2f}%")
            print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {stats['max_drawdown']:.2f}%")

if __name__ == "__main__":
    example_usage()
