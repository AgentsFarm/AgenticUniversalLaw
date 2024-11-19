import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class FarmableAsset:
    """Represents a farmable asset with its base properties"""
    symbol: str
    base_yield: float  # Base yield per unit time
    volatility: float  # Price volatility factor
    growth_cycle: int  # Time units needed for full growth

class AgentsFarm:
    def __init__(self, grid_size: int = 100):
        """
        Initialize the farming universe with a grid of specified size
        grid_size: The size of the square grid (grid_size x grid_size)
        Center is at (0,0)
        """
        self.grid_size = grid_size
        self.center = (0, 0)
        
        # Initialize farmable assets with their properties
        self.assets = {
            "$AGEF": FarmableAsset("$AGEF", base_yield=10.0, volatility=0.2, growth_cycle=30),
            "$WHEAT": FarmableAsset("$WHEAT", base_yield=8.0, volatility=0.15, growth_cycle=90),
            "$CORN": FarmableAsset("$CORN", base_yield=7.0, volatility=0.18, growth_cycle=120),
            "$BEEF": FarmableAsset("$BEEF", base_yield=15.0, volatility=0.25, growth_cycle=180)
        }

    def calculate_distance_factor(self, position: Tuple[int, int]) -> float:
        """
        Calculate the distance-based yield multiplier
        Returns a value between 0 and 1, where 1 is at the center
        and decreases with distance following an inverse square law
        """
        x, y = position
        distance = np.sqrt(x**2 + y**2)
        max_distance = np.sqrt(2) * (self.grid_size / 2)
        
        # Inverse square law with normalization
        distance_factor = 1 / (1 + (distance / max_distance)**2)
        return distance_factor

    def calculate_interest_rate(self, position: Tuple[int, int], asset: str, time: int) -> float:
        """
        Calculate the interest rate (yield) for a given position and asset
        
        Parameters:
        position: (x, y) coordinates
        asset: Symbol of the farmable asset
        time: Current time unit
        
        Returns:
        Interest rate as a percentage
        """
        if asset not in self.assets:
            raise ValueError(f"Unknown asset: {asset}")
            
        asset_info = self.assets[asset]
        
        # Calculate base position-dependent yield
        distance_factor = self.calculate_distance_factor(position)
        
        # Time-based growth cycle factor (sinusoidal variation)
        growth_phase = (2 * np.pi * time) / asset_info.growth_cycle
        cycle_factor = 0.5 + 0.5 * np.sin(growth_phase)
        
        # Add some randomness based on asset volatility
        random_factor = 1.0 + np.random.normal(0, asset_info.volatility)
        
        # Combine all factors to get final interest rate
        interest_rate = (
            asset_info.base_yield *
            distance_factor *
            cycle_factor *
            max(0, random_factor)  # Ensure non-negative
        )
        
        return interest_rate

    def simulate_farm_yields(self, positions: List[Tuple[int, int]], 
                           assets: List[str], 
                           time_period: int) -> Dict:
        """
        Simulate yields for multiple positions and assets over time
        
        Parameters:
        positions: List of (x,y) coordinates
        assets: List of asset symbols to simulate
        time_period: Number of time units to simulate
        
        Returns:
        Dictionary with simulation results
        """
        results = {}
        
        for pos in positions:
            results[pos] = {}
            for asset in assets:
                yields = []
                for t in range(time_period):
                    rate = self.calculate_interest_rate(pos, asset, t)
                    yields.append(rate)
                results[pos][asset] = {
                    'mean_yield': np.mean(yields),
                    'max_yield': np.max(yields),
                    'min_yield': np.min(yields),
                    'yield_volatility': np.std(yields)
                }
                
        return results

def example_usage():
    # Create a new farm universe
    farm = AgentsFarm(grid_size=100)
    
    # Test positions at different distances from center
    test_positions = [(0,0), (10,10), (25,25), (40,40)]
    test_assets = ["$AGEF", "$WHEAT", "$CORN", "$BEEF"]
    
    # Simulate for 360 time units
    results = farm.simulate_farm_yields(test_positions, test_assets, 360)
    
    # Print results
    for pos in test_positions:
        print(f"\nPosition {pos}:")
        for asset in test_assets:
            print(f"{asset}:")
            for metric, value in results[pos][asset].items():
                print(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    example_usage()
