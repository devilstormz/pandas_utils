"""
Trade Reconciliation System Architecture
Compares trade dataframes, identifies breaks, and exports detailed analysis to Excel
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
import logging

# Global Configuration Enums and Constants
class StandardColumnNames:
    """Standardized column names used internally"""
    TRADE_ID = 'trade_id'
    BOOKING_SYSTEM = 'booking_system'
    COUNTERPARTY = 'cpty'
    NOTIONAL = 'notional'
    CURRENCY = 'currency'
    ASSET_LIABILITY = 'asset_liability'
    PRODUCT_TYPE = 'product_type'

class SourceColumnNames:
    """Column names in the automated report (source dataframe)"""
    TRADE_ID = 'TradeReference'
    BOOKING_SYSTEM = 'System'
    COUNTERPARTY = 'Counterparty'
    NOTIONAL = 'NotionalAmount'
    CURRENCY = 'Currency'
    ASSET_LIABILITY = 'AssetLiabilityFlag'
    PRODUCT_TYPE = 'ProductType'

class TargetColumnNames:
    """Column names in the manual spreadsheet (target dataframe)"""
    TRADE_ID = 'trade_ref'
    BOOKING_SYSTEM = 'booking_sys'
    COUNTERPARTY = 'counterparty_name'
    NOTIONAL = 'notional_value'
    CURRENCY = 'ccy'
    ASSET_LIABILITY = 'asset_liab'
    PRODUCT_TYPE = 'product'

class DataFrameType(Enum):
    """Source and target dataframe identifiers"""
    SOURCE = 'source'
    TARGET = 'target'

class BreakCategory(Enum):
    """Classification of reconciliation breaks"""
    NOTIONAL_BREAK = 'notional_break'
    COUNT_BREAK = 'count_break'
    MISSING_TRADE = 'missing_trade'
    COMBINED_BREAK = 'combined_break'

class MetricType(Enum):
    """Metrics for break assessment"""
    NOTIONAL_SUM = 'notional_sum'
    TRADE_COUNT = 'trade_count'

class PivotDimension(Enum):
    """Predefined pivot dimensions for analysis"""
    CCY_BOOKING_ASSETLIAB = 'currency_booking_assetliab'
    CCY_BOOKING_CPTY = 'currency_booking_cpty'
    BOOKING_TRADEID = 'booking_tradeid'

class TradeReconciliation:
    """
    Main reconciliation class for comparing trade dataframes and identifying breaks
    """
    
    def __init__(self, tolerance: float = 0.01, high_volume_currencies: List[str] = None):
        """
        Initialize the reconciliation engine
        
        Args:
            tolerance: Acceptable difference threshold for notional amounts
            high_volume_currencies: Currencies to separate into individual sheets (default: USD, EUR)
        """
        self.tolerance = tolerance
        self.high_volume_currencies = high_volume_currencies or ['USD', 'EUR']
        self.source_df = None
        self.target_df = None
        self.reconciliation_results = {}
        self.pivot_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        """
        Load source and target dataframes for reconciliation
        
        Args:
            source_df: Primary dataset (automated report)
            target_df: Comparison dataset (manual spreadsheet)
        """
        self._validate_dataframes(source_df, target_df)
        
        # Standardize column names
        self.source_df = self._standardize_columns(source_df.copy(), SourceColumnNames)
        self.target_df = self._standardize_columns(target_df.copy(), TargetColumnNames)
        
        self.logger.info(f"Loaded {len(source_df)} source trades and {len(target_df)} target trades")
    
    def _validate_dataframes(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        """Validate that required columns exist in both dataframes"""
        # Check source dataframe columns
        source_required = [
            SourceColumnNames.TRADE_ID, SourceColumnNames.BOOKING_SYSTEM, SourceColumnNames.COUNTERPARTY,
            SourceColumnNames.NOTIONAL, SourceColumnNames.CURRENCY, SourceColumnNames.ASSET_LIABILITY,
            SourceColumnNames.PRODUCT_TYPE
        ]
        
        # Check target dataframe columns
        target_required = [
            TargetColumnNames.TRADE_ID, TargetColumnNames.BOOKING_SYSTEM, TargetColumnNames.COUNTERPARTY,
            TargetColumnNames.NOTIONAL, TargetColumnNames.CURRENCY, TargetColumnNames.ASSET_LIABILITY,
            TargetColumnNames.PRODUCT_TYPE
        ]
        
        missing_source = [col for col in source_required if col not in source_df.columns]
        missing_target = [col for col in target_required if col not in target_df.columns]
        
        if missing_source:
            raise ValueError(f"Missing columns in source dataframe: {missing_source}")
        if missing_target:
            raise ValueError(f"Missing columns in target dataframe: {missing_target}")
    
    def _standardize_columns(self, df: pd.DataFrame, column_mapping) -> pd.DataFrame:
        """Standardize column names to internal format"""
        column_map = {
            column_mapping.TRADE_ID: StandardColumnNames.TRADE_ID,
            column_mapping.BOOKING_SYSTEM: StandardColumnNames.BOOKING_SYSTEM,
            column_mapping.COUNTERPARTY: StandardColumnNames.COUNTERPARTY,
            column_mapping.NOTIONAL: StandardColumnNames.NOTIONAL,
            column_mapping.CURRENCY: StandardColumnNames.CURRENCY,
            column_mapping.ASSET_LIABILITY: StandardColumnNames.ASSET_LIABILITY,
            column_mapping.PRODUCT_TYPE: StandardColumnNames.PRODUCT_TYPE
        }
        return df.rename(columns=column_map)
    
    def perform_reconciliation(self) -> Dict:
        """
        Execute complete reconciliation process
        
        Returns:
            Dictionary containing all reconciliation results
        """
        if self.source_df is None or self.target_df is None:
            raise ValueError("Data must be loaded before reconciliation")
        
        self.logger.info("Starting trade reconciliation process")
        
        # Perform reconciliation for each dimension
        self._reconcile_currency_booking_assetliab()
        self._reconcile_currency_booking_cpty()
        self._reconcile_missing_trades()
        
        # Generate pivot tables
        self._generate_pivot_tables()
        
        self.logger.info("Reconciliation process completed")
        return self.reconciliation_results
    
    def _reconcile_currency_booking_assetliab(self) -> None:
        """Reconcile by Currency x Booking System x Asset/Liability"""
        dimension_cols = [StandardColumnNames.CURRENCY, StandardColumnNames.BOOKING_SYSTEM, StandardColumnNames.ASSET_LIABILITY]
        
        source_agg = self._aggregate_data(self.source_df, dimension_cols)
        target_agg = self._aggregate_data(self.target_df, dimension_cols)
        
        breaks = self._identify_breaks(source_agg, target_agg, dimension_cols)
        self.reconciliation_results[PivotDimension.CCY_BOOKING_ASSETLIAB.value] = breaks
    
    def _reconcile_currency_booking_cpty(self) -> None:
        """Reconcile by Currency x Booking System x Counterparty"""
        dimension_cols = [StandardColumnNames.CURRENCY, StandardColumnNames.BOOKING_SYSTEM, StandardColumnNames.COUNTERPARTY]
        
        source_agg = self._aggregate_data(self.source_df, dimension_cols)
        target_agg = self._aggregate_data(self.target_df, dimension_cols)
        
        breaks = self._identify_breaks(source_agg, target_agg, dimension_cols)
        self.reconciliation_results[PivotDimension.CCY_BOOKING_CPTY.value] = breaks
    
    def _reconcile_missing_trades(self) -> None:
        """Identify missing trades by Booking System x Trade ID"""
        dimension_cols = [StandardColumnNames.BOOKING_SYSTEM, StandardColumnNames.TRADE_ID]
        
        source_trades = set(zip(self.source_df[StandardColumnNames.BOOKING_SYSTEM], 
                               self.source_df[StandardColumnNames.TRADE_ID]))
        target_trades = set(zip(self.target_df[StandardColumnNames.BOOKING_SYSTEM], 
                               self.target_df[StandardColumnNames.TRADE_ID]))
        
        missing_in_target = source_trades - target_trades
        missing_in_source = target_trades - source_trades
        
        missing_trades = {
            'missing_in_target': list(missing_in_target),
            'missing_in_source': list(missing_in_source),
            'total_missing': len(missing_in_target) + len(missing_in_source)
        }
        
        self.reconciliation_results[PivotDimension.BOOKING_TRADEID.value] = missing_trades
    
    def _aggregate_data(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Aggregate data by specified columns"""
        return df.groupby(group_cols).agg({
            StandardColumnNames.NOTIONAL: ['sum', 'count']
        }).reset_index()
    
    def _identify_breaks(self, source_agg: pd.DataFrame, target_agg: pd.DataFrame, 
                        dimension_cols: List[str]) -> Dict:
        """Identify breaks between source and target aggregations"""
        # Flatten column names after aggregation
        source_agg.columns = dimension_cols + ['source_notional_sum', 'source_count']
        target_agg.columns = dimension_cols + ['target_notional_sum', 'target_count']
        
        # Merge on dimension columns
        merged = pd.merge(source_agg, target_agg, on=dimension_cols, how='outer', 
                         suffixes=('_source', '_target')).fillna(0)
        
        # Calculate differences
        merged['notional_diff'] = merged['source_notional_sum'] - merged['target_notional_sum']
        merged['count_diff'] = merged['source_count'] - merged['target_count']
        
        # Classify breaks
        merged['break_category'] = merged.apply(self._classify_break, axis=1)
        
        # Filter only breaks (non-zero differences)
        breaks = merged[
            (abs(merged['notional_diff']) > self.tolerance) | 
            (merged['count_diff'] != 0)
        ].copy()
        
        return {
            'breaks_summary': breaks,
            'total_notional_diff': breaks['notional_diff'].sum(),
            'total_count_diff': breaks['count_diff'].sum(),
            'break_count': len(breaks)
        }
    
    def _classify_break(self, row) -> str:
        """Classify the type of break based on differences"""
        has_notional_break = abs(row['notional_diff']) > self.tolerance
        has_count_break = row['count_diff'] != 0
        
        if has_notional_break and has_count_break:
            return BreakCategory.COMBINED_BREAK.value
        elif has_notional_break:
            return BreakCategory.NOTIONAL_BREAK.value
        elif has_count_break:
            return BreakCategory.COUNT_BREAK.value
        else:
            return 'no_break'
    
    def _generate_pivot_tables(self) -> None:
        """Generate detailed pivot tables for analysis"""
        self.pivot_results = {
            'summary_stats': self._generate_summary_stats(),
            'break_analysis': self._generate_break_analysis(),
            'currency_breakdown': self._generate_currency_breakdown()
        }
    
    def _generate_summary_stats(self) -> Dict:
        """Generate high-level summary statistics"""
        return {
            'source_total_notional': self.source_df[StandardColumnNames.NOTIONAL].sum(),
            'target_total_notional': self.target_df[StandardColumnNames.NOTIONAL].sum(),
            'source_trade_count': len(self.source_df),
            'target_trade_count': len(self.target_df),
            'total_currencies': self.source_df[StandardColumnNames.CURRENCY].nunique(),
            'total_booking_systems': self.source_df[StandardColumnNames.BOOKING_SYSTEM].nunique()
        }
    
    def _generate_break_analysis(self) -> Dict:
        """Generate detailed break analysis"""
        break_analysis = {}
        for dimension, results in self.reconciliation_results.items():
            if 'breaks_summary' in results:
                df = results['breaks_summary']
                break_analysis[dimension] = {
                    'by_break_category': df.groupby('break_category').size().to_dict(),
                    'largest_breaks': df.nlargest(10, 'notional_diff').to_dict('records')
                }
        return break_analysis
    
    def _generate_currency_breakdown(self) -> Dict:
        """Generate currency-specific breakdowns"""
        currency_breakdown = {}
        for dimension in [PivotDimension.CCY_BOOKING_ASSETLIAB.value, PivotDimension.CCY_BOOKING_CPTY.value]:
            if dimension in self.reconciliation_results:
                df = self.reconciliation_results[dimension]['breaks_summary']
                currency_breakdown[dimension] = {}
                for currency in df[StandardColumnNames.CURRENCY].unique():
                    currency_df = df[df[StandardColumnNames.CURRENCY] == currency]
                    currency_breakdown[dimension][currency] = currency_df.to_dict('records')
        return currency_breakdown
    
    def export_to_excel(self, filename: str) -> None:
        """
        Export reconciliation results to Excel with multiple sheets
        
        Args:
            filename: Output Excel filename
        """
        if not self.reconciliation_results:
            raise ValueError("No reconciliation results to export. Run perform_reconciliation() first.")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            self._write_summary_sheet(writer)
            
            # Dimension analysis sheets
            self._write_dimension_sheets(writer)
            
            # High-volume currency sheets (USD/EUR breakdown)
            self._write_currency_specific_sheets(writer)
            
            # Missing trades sheet
            self._write_missing_trades_sheet(writer)
        
        self.logger.info(f"Reconciliation results exported to {filename}")
    
    def _write_summary_sheet(self, writer) -> None:
        """Write summary statistics to Excel"""
        summary_data = []
        for key, value in self.pivot_results['summary_stats'].items():
            summary_data.append({'Metric': key, 'Value': value})
        
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    def _write_dimension_sheets(self, writer) -> None:
        """Write dimension analysis sheets"""
        for dimension, results in self.reconciliation_results.items():
            if 'breaks_summary' in results and dimension != PivotDimension.BOOKING_TRADEID.value:
                sheet_name = dimension.replace('_', ' ').title()[:31]  # Excel sheet name limit
                results['breaks_summary'].to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _write_currency_specific_sheets(self, writer) -> None:
        """Write separate sheets for high-volume currencies"""
        ccy_cpty_dimension = PivotDimension.CCY_BOOKING_CPTY.value
        if ccy_cpty_dimension in self.reconciliation_results:
            df = self.reconciliation_results[ccy_cpty_dimension]['breaks_summary']
            
            for currency in self.high_volume_currencies:
                if currency in df[StandardColumnNames.CURRENCY].values:
                    currency_df = df[df[StandardColumnNames.CURRENCY] == currency]
                    sheet_name = f'{currency} Breaks'
                    currency_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _write_missing_trades_sheet(self, writer) -> None:
        """Write missing trades analysis"""
        missing_data = self.reconciliation_results[PivotDimension.BOOKING_TRADEID.value]
        
        # Create dataframes for missing trades
        missing_in_target_df = pd.DataFrame(missing_data['missing_in_target'], 
                                          columns=[StandardColumnNames.BOOKING_SYSTEM, StandardColumnNames.TRADE_ID])
        missing_in_source_df = pd.DataFrame(missing_data['missing_in_source'], 
                                          columns=[StandardColumnNames.BOOKING_SYSTEM, StandardColumnNames.TRADE_ID])
        
        # Write to different sections of the same sheet
        missing_in_target_df.to_excel(writer, sheet_name='Missing Trades', 
                                     startrow=0, index=False)
        missing_in_source_df.to_excel(writer, sheet_name='Missing Trades', 
                                     startrow=len(missing_in_target_df) + 3, index=False)

# Example Usage with Dummy Data
def create_dummy_data():
    """Create realistic dummy data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Common data for both datasets
    booking_systems = ['MUREX', 'SUMMIT', 'CALYPSO', 'WALL_STREET']
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD']
    counterparties = ['GOLDMAN_SACHS', 'JP_MORGAN', 'BARCLAYS', 'DEUTSCHE_BANK', 'CREDIT_SUISSE', 'UBS']
    asset_liability_flags = ['ASSET', 'LIABILITY']
    product_types = ['SWAP', 'BOND', 'FRA', 'OPTION', 'FUTURE']
    
    # Generate source data (automated report format)
    n_source_trades = 1000
    source_data = {
        SourceColumnNames.TRADE_ID: [f'TRD_{i:06d}' for i in range(n_source_trades)],
        SourceColumnNames.BOOKING_SYSTEM: np.random.choice(booking_systems, n_source_trades),
        SourceColumnNames.COUNTERPARTY: np.random.choice(counterparties, n_source_trades),
        SourceColumnNames.NOTIONAL: np.random.uniform(100000, 50000000, n_source_trades).round(2),
        SourceColumnNames.CURRENCY: np.random.choice(currencies, n_source_trades, 
                                                    p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05]),  # USD/EUR weighted
        SourceColumnNames.ASSET_LIABILITY: np.random.choice(asset_liability_flags, n_source_trades),
        SourceColumnNames.PRODUCT_TYPE: np.random.choice(product_types, n_source_trades)
    }
    
    # Generate target data (manual spreadsheet format) - slightly different to create breaks
    n_target_trades = 950  # Some missing trades
    
    # Take most trades from source but with some modifications
    target_indices = np.random.choice(n_source_trades, n_target_trades, replace=False)
    
    target_data = {
        TargetColumnNames.TRADE_ID: [source_data[SourceColumnNames.TRADE_ID][i] for i in target_indices],
        TargetColumnNames.BOOKING_SYSTEM: [source_data[SourceColumnNames.BOOKING_SYSTEM][i] for i in target_indices],
        TargetColumnNames.COUNTERPARTY: [source_data[SourceColumnNames.COUNTERPARTY][i] for i in target_indices],
        TargetColumnNames.NOTIONAL: [source_data[SourceColumnNames.NOTIONAL][i] for i in target_indices],
        TargetColumnNames.CURRENCY: [source_data[SourceColumnNames.CURRENCY][i] for i in target_indices],
        TargetColumnNames.ASSET_LIABILITY: [source_data[SourceColumnNames.ASSET_LIABILITY][i] for i in target_indices],
        TargetColumnNames.PRODUCT_TYPE: [source_data[SourceColumnNames.PRODUCT_TYPE][i] for i in target_indices]
    }
    
    # Introduce some deliberate breaks for testing
    break_indices = np.random.choice(len(target_data[TargetColumnNames.TRADE_ID]), 50, replace=False)
    
    # Create notional breaks
    for i in break_indices[:25]:
        target_data[TargetColumnNames.NOTIONAL][i] *= np.random.uniform(0.95, 1.05)  # 5% variance
    
    # Create some additional trades in target (missing in source)
    additional_trades = 20
    for i in range(additional_trades):
        target_data[TargetColumnNames.TRADE_ID].append(f'MAN_{i:03d}')
        target_data[TargetColumnNames.BOOKING_SYSTEM].append(np.random.choice(booking_systems))
        target_data[TargetColumnNames.COUNTERPARTY].append(np.random.choice(counterparties))
        target_data[TargetColumnNames.NOTIONAL].append(np.random.uniform(100000, 10000000))
        target_data[TargetColumnNames.CURRENCY].append(np.random.choice(currencies))
        target_data[TargetColumnNames.ASSET_LIABILITY].append(np.random.choice(asset_liability_flags))
        target_data[TargetColumnNames.PRODUCT_TYPE].append(np.random.choice(product_types))
    
    source_df = pd.DataFrame(source_data)
    target_df = pd.DataFrame(target_data)
    
    return source_df, target_df

def example_usage():
    """Example of how to use the TradeReconciliation class with dummy data"""
    
    # Create dummy data
    source_df, target_df = create_dummy_data()
    
    print("Source DataFrame Info:")
    print(f"Shape: {source_df.shape}")
    print(f"Columns: {list(source_df.columns)}")
    print("\nTarget DataFrame Info:")
    print(f"Shape: {target_df.shape}")
    print(f"Columns: {list(target_df.columns)}")
    
    # Initialize reconciliation engine
    reconciler = TradeReconciliation(tolerance=1000.0, high_volume_currencies=['USD', 'EUR'])
    
    # Load data
    reconciler.load_data(source_df, target_df)
    
    # Perform reconciliation
    results = reconciler.perform_reconciliation()
    
    # Print summary results
    print("\n=== RECONCILIATION SUMMARY ===")
    print("Summary Statistics:", reconciler.pivot_results['summary_stats'])
    
    for dimension, result in results.items():
        if 'breaks_summary' in result:
            print(f"\n{dimension.upper()} - Total breaks: {result['break_count']}")
            print(f"Total notional difference: {result['total_notional_diff']:,.2f}")
            print(f"Total count difference: {result['total_count_diff']}")
        elif dimension == 'booking_tradeid':
            print(f"\nMISSING TRADES - Total missing: {result['total_missing']}")
            print(f"Missing in target: {len(result['missing_in_target'])}")
            print(f"Missing in source: {len(result['missing_in_source'])}")
    
    # Export to Excel
    reconciler.export_to_excel('trade_reconciliation_test.xlsx')
    print("\nResults exported to 'trade_reconciliation_test.xlsx'")
    
    return reconciler, source_df, target_df

if __name__ == "__main__":
    example_usage()
