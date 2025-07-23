import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
import openpyxl
from pathlib import Path

class StressTester:
    """
    A class to calculate liquidity and market shock stress impacts on financial data.
    Supports both file-based input and direct dataframe input.
    """
    
    # Class constants to avoid hardcoding
    ANNUAL_LIQUIDITY_STRESS = "Annual Liquidity Stress (%)"
    LIQUIDITY_STRESS_IMPACT = "Liquidity Stress Impact (EUR M)"
    ANNUAL_MARKET_SHOCK = "Annual Market Shock (%)"
    MARKET_SHOCK_IMPACT = "Market Shock Impact (EUR M)"
    TOTAL_SHOCK_IMPACT = "Total Shock Impact (EUR M)"
    
    DAYS_PER_YEAR = 360
    GRAND_TOTAL_LABEL = "Grand Total"
    NET_GAP_PATTERN = ["grand total", "net gap", "total"]
    
    def __init__(self, 
                 liquidity_stress_rate: float, 
                 market_shock_rate: float,
                 file_path: Optional[str] = None,
                 sheet_name: Optional[str] = None,
                 pivot_dataframe: Optional[pd.DataFrame] = None,
                 liquidity_stress_function: Optional[callable] = None):
        """
        Initialize the StressTester.
        
        Parameters:
        - liquidity_stress_rate: Annual liquidity stress rate (as decimal, e.g., 0.05 for 5%)
        - market_shock_rate: Annual market shock stress rate (as decimal, e.g., 0.10 for 10%)
        - file_path: Path to Excel file (optional)
        - sheet_name: Sheet name in Excel file (optional, uses first sheet if not specified)
        - pivot_dataframe: Direct dataframe input (optional)
        - liquidity_stress_function: Custom function to generate liquidity stress ladder (optional)
        """
        self.liquidity_stress_rate = liquidity_stress_rate
        self.market_shock_rate = market_shock_rate
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.pivot_dataframe = pivot_dataframe
        self.liquidity_stress_function = liquidity_stress_function or self._default_liquidity_stress_function
        self.original_data = None
        self.net_gap_data = None
        self.tenor_days = None
        
    def _extract_tenor_days(self, tenor_columns: List[str]) -> Dict[str, int]:
        """
        Extract days from tenor column names.
        Assumes format like '1D', '7D', '1M', '3M', '1Y', etc.
        """
        tenor_days = {}
        
        for tenor in tenor_columns:
            tenor_str = str(tenor).upper().strip()
            
            if 'D' in tenor_str and not 'M' in tenor_str:
                # Days format (e.g., '1D', '7D')
                days = int(''.join(filter(str.isdigit, tenor_str)))
                tenor_days[tenor] = days
            elif 'W' in tenor_str:
                # Weeks format (e.g., '1W', '2W')
                weeks = int(''.join(filter(str.isdigit, tenor_str)))
                tenor_days[tenor] = weeks * 7
            elif 'M' in tenor_str and 'Y' not in tenor_str:
                # Months format (e.g., '1M', '3M', '6M')
                months = int(''.join(filter(str.isdigit, tenor_str)))
                tenor_days[tenor] = months * 30  # Approximate
            elif 'Y' in tenor_str:
                # Years format (e.g., '1Y', '2Y')
                years = int(''.join(filter(str.isdigit, tenor_str)))
                tenor_days[tenor] = years * 360
            else:
                # If format is unclear, try to extract number and assume days
                try:
                    days = int(''.join(filter(str.isdigit, tenor_str)))
                    tenor_days[tenor] = days if days > 0 else 1
                except:
                    tenor_days[tenor] = 1  # Default to 1 day
                    
        return tenor_days
    
    def _default_liquidity_stress_function(self, tenor_days: Dict[str, int]) -> Dict[str, float]:
        """
        Default liquidity stress function that returns the same stress rate for all tenors.
        Can be overridden with custom logic.
        
        Parameters:
        - tenor_days: Dictionary mapping tenor names to days
        
        Returns:
        - Dictionary mapping tenor names to stress rates (as decimals)
        """
        return {tenor: self.liquidity_stress_rate for tenor in tenor_days.keys()}
    
    def _generate_liquidity_stress_ladder(self, tenor_days: Dict[str, int]) -> Dict[str, float]:
        """
        Generate liquidity stress ladder using the configured function.
        
        Parameters:
        - tenor_days: Dictionary mapping tenor names to days
        
        Returns:
        - Dictionary mapping tenor names to stress rates (as decimals)
        """
        return self.liquidity_stress_function(tenor_days)
    
    def _load_data_from_file(self) -> pd.DataFrame:
        """Load data from Excel file."""
        if not self.file_path:
            raise ValueError("File path must be provided when loading from file.")
        
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Load the data
        if self.sheet_name:
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, index_col=[0, 1])
        else:
            df = pd.read_excel(self.file_path, index_col=[0, 1])
        
        self.original_data = df
        return df
    
    def _find_net_gap_row(self, df: pd.DataFrame) -> pd.Series:
        """
        Find the net gap row (Grand Total) from the dataframe.
        """
        # Look for Grand Total or similar patterns in the index
        for idx in df.index:
            # Check if it's a multi-index
            if isinstance(idx, tuple):
                row_text = ' '.join(str(x).lower() for x in idx)
            else:
                row_text = str(idx).lower()
            
            # Check if any of our patterns match
            if any(pattern in row_text for pattern in self.NET_GAP_PATTERN):
                return df.loc[idx]
        
        # If no Grand Total found, use the last row
        print("Warning: Grand Total row not found. Using the last row as net gap.")
        return df.iloc[-1]
    
    def _calculate_liquidity_stress_impact(self, net_gap_values: pd.Series, 
                                         tenor_days: Dict[str, int]) -> pd.Series:
        """
        Calculate liquidity stress impact using dynamic stress ladder.
        Formula: value * stress_rate_for_tenor * tenor_days / 360
        If result is negative, report as 0.
        """
        # Get dynamic stress ladder
        stress_ladder = self._generate_liquidity_stress_ladder(tenor_days)
        
        impact_values = {}
        
        for tenor in net_gap_values.index:
            value = net_gap_values[tenor]
            days = tenor_days.get(tenor, 1)
            stress_rate = stress_ladder.get(tenor, self.liquidity_stress_rate)
            
            # Calculate impact
            impact = value * stress_rate * days / self.DAYS_PER_YEAR
            
            # If negative, set to 0
            impact_values[tenor] = max(0, impact)
        
        return pd.Series(impact_values, index=net_gap_values.index)
    
    def _calculate_market_shock_impact(self, net_gap_values: pd.Series, 
                                     tenor_days: Dict[str, int]) -> pd.Series:
        """
        Calculate market shock stress impact.
        Formula: value * annual_stress * tenor_days / 360
        If result is negative, report as 0.
        """
        impact_values = {}
        
        for tenor in net_gap_values.index:
            value = net_gap_values[tenor]
            days = tenor_days.get(tenor, 1)
            
            # Calculate impact
            impact = value * self.market_shock_rate * days / self.DAYS_PER_YEAR
            
            # If negative, set to 0
            impact_values[tenor] = max(0, impact)
        
        return pd.Series(impact_values, index=net_gap_values.index)
    
    def calculate_stress_impacts(self) -> pd.DataFrame:
        """
        Calculate stress impacts and return results dataframe.
        """
        # Load data based on input method
        if self.pivot_dataframe is not None:
            df = self.pivot_dataframe.copy()
        else:
            df = self._load_data_from_file()
        
        # Extract net gap row
        net_gap = self._find_net_gap_row(df)
        self.net_gap_data = net_gap
        
        # Extract tenor days mapping
        self.tenor_days = self._extract_tenor_days(net_gap.index.tolist())
        
        # Calculate stress impacts
        liquidity_impact = self._calculate_liquidity_stress_impact(net_gap, self.tenor_days)
        market_shock_impact = self._calculate_market_shock_impact(net_gap, self.tenor_days)
        
        # Get the stress ladder for display purposes
        liquidity_stress_ladder = self._generate_liquidity_stress_ladder(self.tenor_days)
        
        # Calculate total impact
        total_impact = liquidity_impact + market_shock_impact
        
        # Create results dataframe with tenors as columns
        results_data = []
        
        # Row 1: Annual Liquidity Stress (dynamic ladder values)
        liquidity_stress_row = [liquidity_stress_ladder.get(tenor, self.liquidity_stress_rate) * 100 
                               for tenor in net_gap.index]
        results_data.append(liquidity_stress_row)
        
        # Row 2: Liquidity Stress Impact
        results_data.append(liquidity_impact.values)
        
        # Row 3: Annual Market Shock (same percentage for all tenors)
        market_shock_row = [self.market_shock_rate * 100] * len(net_gap)
        results_data.append(market_shock_row)
        
        # Row 4: Market Shock Impact
        results_data.append(market_shock_impact.values)
        
        # Row 5: Total Shock Impact
        results_data.append(total_impact.values)
        
        # Create DataFrame with proper structure
        results_df = pd.DataFrame(
            results_data,
            columns=net_gap.index,  # Tenor columns
            index=[
                self.ANNUAL_LIQUIDITY_STRESS,
                self.LIQUIDITY_STRESS_IMPACT,
                self.ANNUAL_MARKET_SHOCK,
                self.MARKET_SHOCK_IMPACT,
                self.TOTAL_SHOCK_IMPACT
            ]
        )
        
        return results_df
    
    def run_analysis(self, output_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete stress analysis.
        
        Parameters:
        - output_file_path: Path to save results (optional)
        
        Returns:
        - DataFrame with stress analysis results
        """
        results_df = self.calculate_stress_impacts()
        
        if self.file_path and output_file_path is None:
            # Append to original file
            self._append_to_original_file(results_df)
        elif output_file_path:
            # Save to specified file
            results_df.to_excel(output_file_path)
            print(f"Results saved to: {output_file_path}")
        
        return results_df
    
    def _append_to_original_file(self, results_df: pd.DataFrame):
        """
        Append results to the original Excel file.
        """
        if not self.file_path:
            raise ValueError("Original file path not available for appending.")
        
        # Load the workbook
        book = openpyxl.load_workbook(self.file_path)
        sheet_name = self.sheet_name if self.sheet_name else book.active.title
        
        # Use ExcelWriter with existing workbook
        with pd.ExcelWriter(self.file_path, engine='openpyxl', mode='a', 
                           if_sheet_exists='overlay') as writer:
            writer.book = book
            
            # Find the last row in the existing sheet
            worksheet = book[sheet_name]
            last_row = worksheet.max_row
            
            # Write results starting from the row after last data + 2 (for spacing)
            results_df.to_excel(writer, sheet_name=sheet_name, 
                               startrow=last_row + 2, header=True, index=True)
        
        print(f"Results appended to: {self.file_path}")

# Example usage and utility functions
def create_sample_data() -> pd.DataFrame:
    """
    Create sample multi-index dataframe for testing.
    """
    # Sample data structure
    currencies = ['USD', 'EUR', 'GBP']
    types = ['Assets', 'Liabilities']
    tenors = ['1D', '7D', '1M', '3M', '6M', '1Y', '2Y']
    
    # Create multi-index
    index = pd.MultiIndex.from_product([currencies, types], 
                                     names=['Currency', 'Type'])
    
    # Add Grand Total row
    grand_total_index = pd.MultiIndex.from_tuples([('Grand Total', '')], 
                                                names=['Currency', 'Type'])
    full_index = index.append(grand_total_index)
    
    # Generate sample data (in millions EUR)
    np.random.seed(42)
    data = np.random.randn(len(full_index), len(tenors)) * 100
    
    # Make Grand Total row sum of all other rows
    data[-1] = np.sum(data[:-1], axis=0)
    
    df = pd.DataFrame(data, index=full_index, columns=tenors)
    return df

def custom_liquidity_stress_function(tenor_days: Dict[str, int]) -> Dict[str, float]:
    """
    Example custom liquidity stress function.
    This could implement complex logic based on tenor characteristics.
    
    Parameters:
    - tenor_days: Dictionary mapping tenor names to days
    
    Returns:
    - Dictionary mapping tenor names to stress rates (as decimals)
    """
    stress_rates = {}
    
    for tenor, days in tenor_days.items():
        if days <= 7:  # Short-term: higher stress
            stress_rates[tenor] = 0.08  # 8%
        elif days <= 90:  # Medium-term: moderate stress
            stress_rates[tenor] = 0.05  # 5%
        elif days <= 365:  # Long-term: lower stress
            stress_rates[tenor] = 0.03  # 3%
        else:  # Very long-term: minimal stress
            stress_rates[tenor] = 0.01  # 1%
    
    return stress_rates

# Example usage
if __name__ == "__main__":
    # Example 1: Using with sample dataframe and default liquidity stress function
    print("Example 1: Using default liquidity stress function")
    sample_df = create_sample_data()
    
    stress_tester = StressTester(
        liquidity_stress_rate=0.05,  # 5% annual (used as default)
        market_shock_rate=0.10,      # 10% annual
        pivot_dataframe=sample_df
    )
    
    results = stress_tester.run_analysis()
    print("\nStress Analysis Results (Default Function):")
    print(results.round(2))
    
    # Example 2: Using with custom liquidity stress function
    print("\n" + "="*70)
    print("Example 2: Using custom liquidity stress function")
    
    stress_tester_custom = StressTester(
        liquidity_stress_rate=0.05,  # Used as fallback only
        market_shock_rate=0.10,      # 10% annual
        pivot_dataframe=sample_df,
        liquidity_stress_function=custom_liquidity_stress_function
    )
    
    results_custom = stress_tester_custom.run_analysis()
    print("\nStress Analysis Results (Custom Function):")
    print(results_custom.round(2))
    
    # Example 3: Using with file (commented out - uncomment when you have a file)
    """
    print("\n" + "="*70)
    print("Example 3: Using with Excel file and custom function")
    
    stress_tester_file = StressTester(
        liquidity_stress_rate=0.05,  # Fallback rate
        market_shock_rate=0.10,      # 10% annual
        file_path="your_data_file.xlsx",
        sheet_name="Sheet1",  # Optional
        liquidity_stress_function=custom_liquidity_stress_function
    )
    
    results_from_file = stress_tester_file.run_analysis()
    print("Results from file with custom function:")
    print(results_from_file.round(2))
    """
