import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TradeReconciliation:
    """
    A class for reconciling trade data between two different trading systems.
    Provides detailed analysis with pivot tables and break identification.
    """
    
    def __init__(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame, 
                 dataset1_name: str = "Dataset1", dataset2_name: str = "Dataset2"):
        """
        Initialize the reconciliation class with two datasets.
        
        Parameters:
        -----------
        dataset1 : pd.DataFrame
            First trading dataset
        dataset2 : pd.DataFrame
            Second trading dataset  
        dataset1_name : str
            Name for the first dataset (for reporting)
        dataset2_name : str
            Name for the second dataset (for reporting)
        """
        self.dataset1 = dataset1.copy()
        self.dataset2 = dataset2.copy()
        self.dataset1_name = dataset1_name
        self.dataset2_name = dataset2_name
        
        # Validate required columns
        required_cols = ['trade_id', 'booking_system', 'currency', 'notional', 'underlying_static']
        self._validate_columns(required_cols)
        
        # Store results
        self.pivot_results = {}
        self.break_results = {}
        
    def _validate_columns(self, required_cols: List[str]):
        """Validate that both datasets have required columns."""
        for col in required_cols:
            if col not in self.dataset1.columns:
                raise ValueError(f"Column '{col}' missing from {self.dataset1_name}")
            if col not in self.dataset2.columns:
                raise ValueError(f"Column '{col}' missing from {self.dataset2_name}")
    
    def _create_pivot_summary(self, df: pd.DataFrame, index_cols: List[str], 
                            dataset_name: str) -> pd.DataFrame:
        """Create pivot summary with sum of notional and count of trades."""
        pivot = df.groupby(index_cols).agg({
            'notional': ['sum', 'count']
        }).round(2)
        
        # Flatten column names
        pivot.columns = [f'{dataset_name}_{col[1]}' if col[1] else f'{dataset_name}_{col[0]}'
                        for col in pivot.columns]
        pivot.columns = [col.replace('sum', 'notional_sum').replace('count', 'trade_count') 
                        for col in pivot.columns]
        
        return pivot.reset_index()
    
    def _combine_and_calculate_differences(self, pivot1: pd.DataFrame, pivot2: pd.DataFrame,
                                         index_cols: List[str]) -> pd.DataFrame:
        """Combine pivots and calculate differences."""
        # Merge the two pivots
        combined = pd.merge(pivot1, pivot2, on=index_cols, how='outer').fillna(0)
        
        # Calculate differences
        notional_col1 = f'{self.dataset1_name}_notional_sum'
        notional_col2 = f'{self.dataset2_name}_notional_sum'
        count_col1 = f'{self.dataset1_name}_trade_count'
        count_col2 = f'{self.dataset2_name}_trade_count'
        
        combined['notional_difference'] = combined[notional_col1] - combined[notional_col2]
        combined['trade_count_difference'] = combined[count_col1] - combined[count_col2]
        
        # Round for display
        combined = combined.round(2)
        
        return combined
    
    def _identify_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify rows where there are differences (breaks)."""
        breaks = df[
            (df['notional_difference'] != 0) | 
            (df['trade_count_difference'] != 0)
        ].copy()
        return breaks
    
    def generate_currency_pivot(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate pivot by currency with metrics."""
        print("Generating currency pivot...")
        
        # Create pivots for each dataset
        pivot1 = self._create_pivot_summary(self.dataset1, ['currency'], self.dataset1_name)
        pivot2 = self._create_pivot_summary(self.dataset2, ['currency'], self.dataset2_name)
        
        # Combine and calculate differences
        combined = self._combine_and_calculate_differences(pivot1, pivot2, ['currency'])
        breaks = self._identify_breaks(combined)
        
        self.pivot_results['currency'] = combined
        self.break_results['currency'] = breaks
        
        return combined, breaks
    
    def generate_currency_counterparty_pivot(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate detailed pivot by currency x counterparty."""
        print("Generating currency x counterparty pivot...")
        
        # Assuming counterparty is in underlying_static - adjust if needed
        index_cols = ['currency', 'underlying_static']
        
        pivot1 = self._create_pivot_summary(self.dataset1, index_cols, self.dataset1_name)
        pivot2 = self._create_pivot_summary(self.dataset2, index_cols, self.dataset2_name)
        
        combined = self._combine_and_calculate_differences(pivot1, pivot2, index_cols)
        breaks = self._identify_breaks(combined)
        
        self.pivot_results['currency_counterparty'] = combined
        self.break_results['currency_counterparty'] = breaks
        
        return combined, breaks
    
    def generate_currency_system_counterparty_pivot(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate detailed pivot by currency x system x counterparty."""
        print("Generating currency x system x counterparty pivot...")
        
        index_cols = ['currency', 'booking_system', 'underlying_static']
        
        pivot1 = self._create_pivot_summary(self.dataset1, index_cols, self.dataset1_name)
        pivot2 = self._create_pivot_summary(self.dataset2, index_cols, self.dataset2_name)
        
        combined = self._combine_and_calculate_differences(pivot1, pivot2, index_cols)
        breaks = self._identify_breaks(combined)
        
        self.pivot_results['currency_system_counterparty'] = combined
        self.break_results['currency_system_counterparty'] = breaks
        
        return combined, breaks
    
    def generate_all_pivots(self) -> Dict[str, pd.DataFrame]:
        """Generate all required pivots."""
        print("Generating all pivot tables...")
        
        # Generate all pivots
        self.generate_currency_pivot()
        self.generate_currency_counterparty_pivot() 
        self.generate_currency_system_counterparty_pivot()
        
        return self.pivot_results
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get high-level summary statistics."""
        stats = []
        
        for dataset_name, dataset in [(self.dataset1_name, self.dataset1), 
                                    (self.dataset2_name, self.dataset2)]:
            stat = {
                'Dataset': dataset_name,
                'Total_Trades': len(dataset),
                'Total_Notional': dataset['notional'].sum(),
                'Unique_Currencies': dataset['currency'].nunique(),
                'Unique_Systems': dataset['booking_system'].nunique(),
                'Unique_Counterparties': dataset['underlying_static'].nunique()
            }
            stats.append(stat)
        
        # Add difference row
        diff_stat = {
            'Dataset': 'Difference',
            'Total_Trades': len(self.dataset1) - len(self.dataset2),
            'Total_Notional': self.dataset1['notional'].sum() - self.dataset2['notional'].sum(),
            'Unique_Currencies': self.dataset1['currency'].nunique() - self.dataset2['currency'].nunique(),
            'Unique_Systems': self.dataset1['booking_system'].nunique() - self.dataset2['booking_system'].nunique(),
            'Unique_Counterparties': self.dataset1['underlying_static'].nunique() - self.dataset2['underlying_static'].nunique()
        }
        stats.append(diff_stat)
        
        return pd.DataFrame(stats).round(2)
    
    def export_to_excel(self, filename: str = 'trade_reconciliation_report.xlsx'):
        """Export all results to an Excel file with multiple sheets."""
        print(f"Exporting results to {filename}...")
        
        # Generate all pivots if not already done
        if not self.pivot_results:
            self.generate_all_pivots()
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary statistics
            summary = self.get_summary_statistics()
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Original datasets
            self.dataset1.to_excel(writer, sheet_name=f'{self.dataset1_name}_Raw', index=False)
            self.dataset2.to_excel(writer, sheet_name=f'{self.dataset2_name}_Raw', index=False)
            
            # Pivot results
            sheet_names = {
                'currency': 'Currency_Pivot',
                'currency_counterparty': 'Currency_Counterparty_Pivot', 
                'currency_system_counterparty': 'Currency_System_Counterparty_Pivot'
            }
            
            for key, sheet_name in sheet_names.items():
                if key in self.pivot_results:
                    self.pivot_results[key].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Break results (filtered pivots showing only differences)
            break_sheet_names = {
                'currency': 'Currency_Breaks',
                'currency_counterparty': 'Currency_Counterparty_Breaks',
                'currency_system_counterparty': 'Currency_System_Counterparty_Breaks'
            }
            
            for key, sheet_name in break_sheet_names.items():
                if key in self.break_results and not self.break_results[key].empty:
                    self.break_results[key].to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Report exported successfully to {filename}")
    
    def display_results(self):
        """Display all results in a formatted way."""
        print("="*80)
        print("TRADE RECONCILIATION REPORT")
        print("="*80)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print("-"*50)
        summary = self.get_summary_statistics()
        print(summary.to_string(index=False))
        
        # Generate pivots if not done
        if not self.pivot_results:
            self.generate_all_pivots()
        
        # Display each pivot and its breaks
        pivot_titles = {
            'currency': 'CURRENCY PIVOT',
            'currency_counterparty': 'CURRENCY x COUNTERPARTY PIVOT',
            'currency_system_counterparty': 'CURRENCY x SYSTEM x COUNTERPARTY PIVOT'
        }
        
        for key, title in pivot_titles.items():
            if key in self.pivot_results:
                print(f"\n{title}:")
                print("-"*len(title))
                print(self.pivot_results[key].to_string(index=False))
                
                if key in self.break_results and not self.break_results[key].empty:
                    print(f"\n{title} - BREAKS ONLY:")
                    print("-"*(len(title) + 13))
                    print(self.break_results[key].to_string(index=False))
                else:
                    print(f"\nNo breaks found in {title.lower()}")


# Example usage and test function
def create_sample_data():
    """Create sample datasets for testing."""
    np.random.seed(42)
    
    currencies = ['USD', 'EUR', 'GBP', 'JPY']
    systems = ['SystemA', 'SystemB', 'SystemC']
    counterparties = ['CP001', 'CP002', 'CP003', 'CP004', 'CP005']
    
    # Dataset 1
    n1 = 100
    dataset1 = pd.DataFrame({
        'trade_id': [f'T{i:04d}' for i in range(1, n1+1)],
        'booking_system': np.random.choice(systems, n1),
        'currency': np.random.choice(currencies, n1),
        'notional': np.random.uniform(10000, 1000000, n1).round(2),
        'underlying_static': np.random.choice(counterparties, n1)
    })
    
    # Dataset 2 - similar but with some differences
    n2 = 95
    dataset2 = pd.DataFrame({
        'trade_id': [f'T{i:04d}' for i in range(1, n2+1)],
        'booking_system': np.random.choice(systems, n2),
        'currency': np.random.choice(currencies, n2),
        'notional': np.random.uniform(10000, 1000000, n2).round(2),
        'underlying_static': np.random.choice(counterparties, n2)
    })
    
    return dataset1, dataset2

def test_reconciliation():
    """Test the reconciliation class with sample data."""
    print("Creating sample data...")
    dataset1, dataset2 = create_sample_data()
    
    print("Initializing reconciliation...")
    reconciler = TradeReconciliation(dataset1, dataset2, "TradingSystem1", "TradingSystem2")
    
    print("Running reconciliation analysis...")
    reconciler.display_results()
    
    print("\nExporting to Excel...")
    reconciler.export_to_excel('sample_trade_reconciliation.xlsx')
    
    return reconciler

# Uncomment the line below to run the test
# test_reconciliation()