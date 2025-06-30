from enum import Enum
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime, timedelta


class StandardColumnNames(Enum):
    """Enum for standardized internal column names"""
    TRADE_ID = 'trade_id'
    COUNTERPARTY = 'counterparty'
    CPTY_CODE = 'cpty_code'
    CONTRA_ACCOUNT = 'contra_account'
    CURRENCY = 'currency'
    MARKET_PRICE = 'market_price'
    HQLA_STATUS = 'hqla_status'
    NOTIONAL = 'notional'
    MKT_VALUE_COLLATERAL = 'mkt_value_collateral'
    TRADE_COUNT = 'trade_count'


class Dataset1ColumnNames(Enum):
    """Enum for Dataset 1 column names (your actual column names)"""
    TRADE_ID = 'TradeReference'
    COUNTERPARTY = 'CounterpartyName'
    CPTY_CODE = 'CptyCode'
    CONTRA_ACCOUNT = 'ContraAccount'
    CURRENCY = 'TradeCurrency'
    MARKET_PRICE = 'MarketPrice'
    HQLA_STATUS = 'HQLAStatus'
    NOTIONAL = 'NotionalAmount'
    MKT_VALUE_COLLATERAL = 'CollateralMarketValue'


class Dataset2ColumnNames(Enum):
    """Enum for Dataset 2 column names (your actual column names)"""
    TRADE_ID = 'trade_ref'
    COUNTERPARTY = 'cpty_name'
    CPTY_CODE = 'cpty_cd'
    CONTRA_ACCOUNT = 'contra_acct'
    CURRENCY = 'ccy'
    MARKET_PRICE = 'mkt_price'
    HQLA_STATUS = 'hqla_flag'
    NOTIONAL = 'notional_amt'
    MKT_VALUE_COLLATERAL = 'collateral_mv'


class ReconciliationSummary(object):
    """Class to hold reconciliation summary results (Python 3.6 compatible)"""
    
    def __init__(self, currency, counterparty, dataset1_mv_sum, dataset2_mv_sum, 
                 mv_difference, dataset1_trade_count, dataset2_trade_count,
                 trade_count_difference, missing_in_dataset1, missing_in_dataset2, reasons):
        self.currency = currency
        self.counterparty = counterparty
        self.dataset1_mv_sum = dataset1_mv_sum
        self.dataset2_mv_sum = dataset2_mv_sum
        self.mv_difference = mv_difference
        self.dataset1_trade_count = dataset1_trade_count
        self.dataset2_trade_count = dataset2_trade_count
        self.trade_count_difference = trade_count_difference
        self.missing_in_dataset1 = missing_in_dataset1
        self.missing_in_dataset2 = missing_in_dataset2
        self.reasons = reasons


class SampleDataGenerator:
    """Generate sample HQLA trade data for testing"""
    
    @staticmethod
    def generate_dataset1(num_trades=1000):
        """Generate sample dataset 1 with specified column names"""
        random.seed(42)  # For reproducible results
        np.random.seed(42)
        
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
        counterparties = ['Goldman Sachs', 'JPMorgan', 'Deutsche Bank', 'UBS', 'Credit Suisse', 
                         'BNP Paribas', 'Barclays', 'HSBC', 'Citi', 'Morgan Stanley']
        hqla_statuses = ['Level1', 'Level2A', 'Level2B', 'NonHQLA']
        
        data = []
        for i in range(num_trades):
            # Some trades will have _CA suffix
            trade_id = f"TRD{i+1:06d}"
            if random.random() < 0.1:  # 10% chance of _CA suffix
                trade_id += "_CA"
            
            counterparty = random.choice(counterparties)
            cpty_code = counterparty.replace(' ', '').upper()[:10]
            
            # Introduce some "null" values for cpty_code (5% chance)
            if random.random() < 0.05:
                cpty_code = "null"
            
            currency = random.choice(currencies)
            notional = random.uniform(100000, 10000000)
            market_price = random.uniform(95, 105)
            collateral_mv = notional * (market_price / 100) * random.uniform(0.8, 1.2)
            
            data.append({
                Dataset1ColumnNames.TRADE_ID.value: trade_id,
                Dataset1ColumnNames.COUNTERPARTY.value: counterparty,
                Dataset1ColumnNames.CPTY_CODE.value: cpty_code,
                Dataset1ColumnNames.CONTRA_ACCOUNT.value: counterparty.replace(' ', '').upper()[:10],
                Dataset1ColumnNames.CURRENCY.value: currency,
                Dataset1ColumnNames.MARKET_PRICE.value: market_price,
                Dataset1ColumnNames.HQLA_STATUS.value: random.choice(hqla_statuses),
                Dataset1ColumnNames.NOTIONAL.value: notional,
                Dataset1ColumnNames.MKT_VALUE_COLLATERAL.value: collateral_mv
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_dataset2(num_trades=1000, overlap_ratio=0.8):
        """Generate sample dataset 2 with some overlap with dataset 1"""
        random.seed(43)  # Different seed for variation
        np.random.seed(43)
        
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF']
        counterparties = ['Goldman Sachs', 'JPMorgan', 'Deutsche Bank', 'UBS', 'Credit Suisse', 
                         'BNP Paribas', 'Barclays', 'HSBC', 'Citi', 'Morgan Stanley']
        hqla_statuses = ['Level1', 'Level2A', 'Level2B', 'NonHQLA']
        
        data = []
        overlap_count = int(num_trades * overlap_ratio)
        
        for i in range(num_trades):
            # Create some overlap with dataset1
            if i < overlap_count:
                trade_id = f"TRD{i+1:06d}"
                # Some matching trades might have slight differences
                if random.random() < 0.1:
                    trade_id += "_CA"
            else:
                # Unique trades in dataset2
                trade_id = f"TRD{i+500:06d}"
            
            counterparty = random.choice(counterparties)
            cpty_code = counterparty.replace(' ', '').upper()[:10]
            currency = random.choice(currencies)
            notional = random.uniform(100000, 10000000)
            market_price = random.uniform(95, 105)
            collateral_mv = notional * (market_price / 100) * random.uniform(0.8, 1.2)
            
            data.append({
                Dataset2ColumnNames.TRADE_ID.value: trade_id,
                Dataset2ColumnNames.COUNTERPARTY.value: counterparty,
                Dataset2ColumnNames.CPTY_CODE.value: cpty_code,
                Dataset2ColumnNames.CONTRA_ACCOUNT.value: counterparty.replace(' ', '').upper()[:10],
                Dataset2ColumnNames.CURRENCY.value: currency,
                Dataset2ColumnNames.MARKET_PRICE.value: market_price,
                Dataset2ColumnNames.HQLA_STATUS.value: random.choice(hqla_statuses),
                Dataset2ColumnNames.NOTIONAL.value: notional,
                Dataset2ColumnNames.MKT_VALUE_COLLATERAL.value: collateral_mv
            })
        
        return pd.DataFrame(data)


class HQLATradeReconciliator:
    """
    A comprehensive reconciliation class for HQLA trade datasets.
    Handles preprocessing, reconciliation, and analysis of trade data.
    """
    
    def __init__(self):
        self.dataset1 = None
        self.dataset2 = None
        self.reconciliation_results = []
        self.trade_id_analysis = {}
        
        # Column mapping dictionaries
        self.dataset1_column_mapping = {
            Dataset1ColumnNames.TRADE_ID.value: StandardColumnNames.TRADE_ID.value,
            Dataset1ColumnNames.COUNTERPARTY.value: StandardColumnNames.COUNTERPARTY.value,
            Dataset1ColumnNames.CPTY_CODE.value: StandardColumnNames.CPTY_CODE.value,
            Dataset1ColumnNames.CONTRA_ACCOUNT.value: StandardColumnNames.CONTRA_ACCOUNT.value,
            Dataset1ColumnNames.CURRENCY.value: StandardColumnNames.CURRENCY.value,
            Dataset1ColumnNames.MARKET_PRICE.value: StandardColumnNames.MARKET_PRICE.value,
            Dataset1ColumnNames.HQLA_STATUS.value: StandardColumnNames.HQLA_STATUS.value,
            Dataset1ColumnNames.NOTIONAL.value: StandardColumnNames.NOTIONAL.value,
            Dataset1ColumnNames.MKT_VALUE_COLLATERAL.value: StandardColumnNames.MKT_VALUE_COLLATERAL.value
        }
        
        self.dataset2_column_mapping = {
            Dataset2ColumnNames.TRADE_ID.value: StandardColumnNames.TRADE_ID.value,
            Dataset2ColumnNames.COUNTERPARTY.value: StandardColumnNames.COUNTERPARTY.value,
            Dataset2ColumnNames.CPTY_CODE.value: StandardColumnNames.CPTY_CODE.value,
            Dataset2ColumnNames.CONTRA_ACCOUNT.value: StandardColumnNames.CONTRA_ACCOUNT.value,
            Dataset2ColumnNames.CURRENCY.value: StandardColumnNames.CURRENCY.value,
            Dataset2ColumnNames.MARKET_PRICE.value: StandardColumnNames.MARKET_PRICE.value,
            Dataset2ColumnNames.HQLA_STATUS.value: StandardColumnNames.HQLA_STATUS.value,
            Dataset2ColumnNames.NOTIONAL.value: StandardColumnNames.NOTIONAL.value,
            Dataset2ColumnNames.MKT_VALUE_COLLATERAL.value: StandardColumnNames.MKT_VALUE_COLLATERAL.value
        }
    
    def _normalize_columns(self, df, column_mapping):
        """
        Normalize column names using the provided mapping.
        
        Args:
            df: DataFrame to normalize
            column_mapping: Dictionary mapping original to standard column names
            
        Returns:
            DataFrame with normalized column names
        """
        df_normalized = df.copy()
        
        # Rename columns that exist in the mapping
        available_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
        df_normalized = df_normalized.rename(columns=available_mappings)
        
        return df_normalized
    
    def preprocess_dataset1(self, df):
        """
        Preprocess dataset 1 with specific business rules.
        
        Args:
            df: Raw dataset 1 DataFrame
            
        Returns:
            Preprocessed DataFrame with normalized columns
        """
        # First normalize column names
        df_processed = self._normalize_columns(df, self.dataset1_column_mapping)
        
        # Handle "null" string values in cpty_code
        if StandardColumnNames.CPTY_CODE.value in df_processed.columns:
            null_mask = df_processed[StandardColumnNames.CPTY_CODE.value] == "null"
            if null_mask.any():
                print("Found {} rows with 'null' cpty_code, overriding with contra_account values".format(null_mask.sum()))
                df_processed.loc[null_mask, StandardColumnNames.CPTY_CODE.value] = df_processed.loc[null_mask, StandardColumnNames.CONTRA_ACCOUNT.value]
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            StandardColumnNames.MARKET_PRICE.value,
            StandardColumnNames.NOTIONAL.value,
            StandardColumnNames.MKT_VALUE_COLLATERAL.value
        ]
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed
    
    def preprocess_dataset2(self, df):
        """
        Preprocess dataset 2 (minimal processing required).
        
        Args:
            df: Raw dataset 2 DataFrame
            
        Returns:
            Preprocessed DataFrame with normalized columns
        """
        # First normalize column names
        df_processed = self._normalize_columns(df, self.dataset2_column_mapping)
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            StandardColumnNames.MARKET_PRICE.value,
            StandardColumnNames.NOTIONAL.value,
            StandardColumnNames.MKT_VALUE_COLLATERAL.value
        ]
        
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed
    
    def load_datasets(self, dataset1, dataset2):
        """
        Load and preprocess both datasets.
        
        Args:
            dataset1: Raw dataset 1
            dataset2: Raw dataset 2
        """
        print("Processing Dataset 1...")
        self.dataset1 = self.preprocess_dataset1(dataset1)
        
        print("Processing Dataset 2...")
        self.dataset2 = self.preprocess_dataset2(dataset2)
        
        print("Dataset 1: {} rows".format(len(self.dataset1)))
        print("Dataset 2: {} rows".format(len(self.dataset2)))
    
    def _normalize_trade_id(self, trade_id):
        """
        Normalize trade ID by removing _CA suffix if present.
        
        Args:
            trade_id: Original trade ID
            
        Returns:
            Normalized trade ID
        """
        if isinstance(trade_id, str) and trade_id.endswith('_CA'):
            return trade_id[:-3]  # Remove '_CA' suffix
        return str(trade_id)
    
    def perform_currency_counterparty_reconciliation(self):
        """
        Perform reconciliation by currency and counterparty.
        
        Returns:
            List of reconciliation summaries
        """
        if self.dataset1 is None or self.dataset2 is None:
            raise ValueError("Datasets must be loaded first using load_datasets()")
        
        # Group by currency and counterparty for both datasets
        group_cols = [StandardColumnNames.CURRENCY.value, StandardColumnNames.COUNTERPARTY.value]
        
        agg_dict = {
            StandardColumnNames.MKT_VALUE_COLLATERAL.value: 'sum',
            StandardColumnNames.TRADE_ID.value: 'count'
        }
        
        df1_grouped = self.dataset1.groupby(group_cols).agg(agg_dict).reset_index()
        df1_grouped.columns = [StandardColumnNames.CURRENCY.value, StandardColumnNames.COUNTERPARTY.value, 'mv_sum_ds1', 'trade_count_ds1']
        
        df2_grouped = self.dataset2.groupby(group_cols).agg(agg_dict).reset_index()
        df2_grouped.columns = [StandardColumnNames.CURRENCY.value, StandardColumnNames.COUNTERPARTY.value, 'mv_sum_ds2', 'trade_count_ds2']
        
        # Merge the grouped data
        merged = pd.merge(df1_grouped, df2_grouped, 
                         on=[StandardColumnNames.CURRENCY.value, StandardColumnNames.COUNTERPARTY.value], 
                         how='outer', suffixes=('_ds1', '_ds2'))
        
        # Fill NaN values with 0
        merged = merged.fillna(0)
        
        # Calculate differences
        merged['mv_difference'] = merged['mv_sum_ds1'] - merged['mv_sum_ds2']
        merged['trade_count_difference'] = merged['trade_count_ds1'] - merged['trade_count_ds2']
        
        results = []
        
        for _, row in merged.iterrows():
            currency = row[StandardColumnNames.CURRENCY.value]
            counterparty = row[StandardColumnNames.COUNTERPARTY.value]
            
            # Get trade IDs for this currency/counterparty combination
            mask1 = (self.dataset1[StandardColumnNames.CURRENCY.value] == currency) & \
                   (self.dataset1[StandardColumnNames.COUNTERPARTY.value] == counterparty)
            mask2 = (self.dataset2[StandardColumnNames.CURRENCY.value] == currency) & \
                   (self.dataset2[StandardColumnNames.COUNTERPARTY.value] == counterparty)
            
            trades1 = set(self.dataset1[mask1][StandardColumnNames.TRADE_ID.value].astype(str))
            trades2 = set(self.dataset2[mask2][StandardColumnNames.TRADE_ID.value].astype(str))
            
            # Normalize trade IDs for comparison
            trades1_normalized = set([self._normalize_trade_id(tid) for tid in trades1])
            trades2_normalized = set([self._normalize_trade_id(tid) for tid in trades2])
            
            missing_in_ds1 = list(trades2_normalized - trades1_normalized)
            missing_in_ds2 = list(trades1_normalized - trades2_normalized)
            
            # Determine reasons for differences
            reasons = self._analyze_reasons(currency, counterparty, row, missing_in_ds1, missing_in_ds2)
            
            summary = ReconciliationSummary(
                currency=currency,
                counterparty=counterparty,
                dataset1_mv_sum=row['mv_sum_ds1'],
                dataset2_mv_sum=row['mv_sum_ds2'],
                mv_difference=row['mv_difference'],
                dataset1_trade_count=int(row['trade_count_ds1']),
                dataset2_trade_count=int(row['trade_count_ds2']),
                trade_count_difference=int(row['trade_count_difference']),
                missing_in_dataset1=missing_in_ds1,
                missing_in_dataset2=missing_in_ds2,
                reasons=reasons
            )
            
            results.append(summary)
        
        self.reconciliation_results = results
        return results
    
    def _analyze_reasons(self, currency, counterparty, merged_row, missing_in_ds1, missing_in_ds2):
        """
        Analyze and determine reasons for differences.
        
        Args:
            currency: Currency being analyzed
            counterparty: Counterparty being analyzed
            merged_row: Merged row data
            missing_in_ds1: Trade IDs missing in dataset 1
            missing_in_ds2: Trade IDs missing in dataset 2
            
        Returns:
            List of reason strings
        """
        reasons = []
        
        # Check for market value differences
        if abs(merged_row['mv_difference']) > 0.01:  # Allowing for small rounding differences
            reasons.append("Market value difference: {:.2f}".format(merged_row['mv_difference']))
        
        # Check for trade count differences
        if merged_row['trade_count_difference'] != 0:
            reasons.append("Trade count difference: {}".format(merged_row['trade_count_difference']))
        
        # Check for missing trade IDs
        if missing_in_ds1:
            reasons.append("Trade IDs missing in Dataset 1: {} trades".format(len(missing_in_ds1)))
        
        if missing_in_ds2:
            reasons.append("Trade IDs missing in Dataset 2: {} trades".format(len(missing_in_ds2)))
        
        # If no specific reasons found but there are differences, add generic reason
        if not reasons and (abs(merged_row['mv_difference']) > 0 or merged_row['trade_count_difference'] != 0):
            reasons.append("Unknown difference detected")
        
        return reasons
    
    def generate_trade_id_comparison_report(self):
        """
        Generate detailed trade ID comparison report.
        
        Returns:
            DataFrame with trade ID comparison details
        """
        if self.dataset1 is None or self.dataset2 is None:
            raise ValueError("Datasets must be loaded first using load_datasets()")
        
        # Create normalized trade ID columns
        df1 = self.dataset1.copy()
        df2 = self.dataset2.copy()
        
        df1['trade_id_normalized'] = df1[StandardColumnNames.TRADE_ID.value].astype(str).apply(self._normalize_trade_id)
        df2['trade_id_normalized'] = df2[StandardColumnNames.TRADE_ID.value].astype(str).apply(self._normalize_trade_id)
        
        # Get key columns for analysis
        key_columns = [
            StandardColumnNames.TRADE_ID.value,
            StandardColumnNames.COUNTERPARTY.value,
            StandardColumnNames.CURRENCY.value,
            StandardColumnNames.MARKET_PRICE.value,
            StandardColumnNames.HQLA_STATUS.value,
            StandardColumnNames.NOTIONAL.value,
            StandardColumnNames.MKT_VALUE_COLLATERAL.value
        ]
        
        # Filter to available columns
        available_cols1 = [col for col in key_columns if col in df1.columns]
        available_cols2 = [col for col in key_columns if col in df2.columns]
        
        df1_subset = df1[available_cols1 + ['trade_id_normalized']].copy()
        df2_subset = df2[available_cols2 + ['trade_id_normalized']].copy()
        
        # Add dataset identifier
        df1_subset['dataset'] = 'Dataset1'
        df2_subset['dataset'] = 'Dataset2'
        
        # Combine datasets
        combined = pd.concat([df1_subset, df2_subset], ignore_index=True, sort=False)
        
        # Check which normalized trade IDs exist in which datasets
        trade_id_counts = combined.groupby('trade_id_normalized')['dataset'].apply(list).reset_index()
        trade_id_counts['exists_in_ds1'] = trade_id_counts['dataset'].apply(lambda x: 'Dataset1' in x)
        trade_id_counts['exists_in_ds2'] = trade_id_counts['dataset'].apply(lambda x: 'Dataset2' in x)
        trade_id_counts['status'] = trade_id_counts.apply(
            lambda row: 'Both' if row['exists_in_ds1'] and row['exists_in_ds2'] 
                       else 'Only_Dataset1' if row['exists_in_ds1'] 
                       else 'Only_Dataset2', axis=1
        )
        
        # Merge back with original data
        result = pd.merge(combined, trade_id_counts[['trade_id_normalized', 'status']], 
                         on='trade_id_normalized', how='left')
        
        self.trade_id_analysis = result
        return result
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report.
        
        Returns:
            Formatted summary report string
        """
        if not self.reconciliation_results:
            return "No reconciliation results available. Please run perform_currency_counterparty_reconciliation() first."
        
        report = []
        report.append("=" * 80)
        report.append("HQLA TRADE DATASET RECONCILIATION SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_currencies = len(set([r.currency for r in self.reconciliation_results]))
        total_counterparties = len(set([r.counterparty for r in self.reconciliation_results]))
        
        report.append("Total Currencies: {}".format(total_currencies))
        report.append("Total Counterparties: {}".format(total_counterparties))
        report.append("Total Currency-Counterparty Combinations: {}".format(len(self.reconciliation_results)))
        report.append("")
        
        # Net impact by currency
        report.append("NET IMPACT BY CURRENCY:")
        report.append("-" * 40)
        
        currency_impact = {}
        for result in self.reconciliation_results:
            if result.currency not in currency_impact:
                currency_impact[result.currency] = {
                    'mv_difference': 0,
                    'trade_count_difference': 0,
                    'combinations': 0
                }
            currency_impact[result.currency]['mv_difference'] += result.mv_difference
            currency_impact[result.currency]['trade_count_difference'] += result.trade_count_difference
            currency_impact[result.currency]['combinations'] += 1
        
        for currency, impact in currency_impact.items():
            report.append("{}:".format(currency))
            report.append("  Market Value Difference: {:,.2f}".format(impact['mv_difference']))
            report.append("  Trade Count Difference: {:,}".format(impact['trade_count_difference']))
            report.append("  Counterparty Combinations: {}".format(impact['combinations']))
            report.append("")
        
        # Detailed breakdown by currency and counterparty
        report.append("DETAILED BREAKDOWN:")
        report.append("-" * 40)
        
        for result in self.reconciliation_results:
            if (abs(result.mv_difference) > 0.01 or result.trade_count_difference != 0 or 
                result.missing_in_dataset1 or result.missing_in_dataset2):
                
                report.append("{} - {}:".format(result.currency, result.counterparty))
                report.append("  MV Difference: {:,.2f}".format(result.mv_difference))
                report.append("  Trade Count Difference: {}".format(result.trade_count_difference))
                
                if result.missing_in_dataset1:
                    report.append("  Missing in Dataset1: {} trades".format(len(result.missing_in_dataset1)))
                
                if result.missing_in_dataset2:
                    report.append("  Missing in Dataset2: {} trades".format(len(result.missing_in_dataset2)))
                
                if result.reasons:
                    report.append("  Reasons:")
                    for reason in result.reasons:
                        report.append("    - {}".format(reason))
                
                report.append("")
        
        return "\n".join(report)
    
    def export_results_to_excel(self, filename):
        """
        Export reconciliation results to Excel file.
        
        Args:
            filename: Output Excel filename
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for result in self.reconciliation_results:
                    summary_data.append({
                        'Currency': result.currency,
                        'Counterparty': result.counterparty,
                        'Dataset1_MV_Sum': result.dataset1_mv_sum,
                        'Dataset2_MV_Sum': result.dataset2_mv_sum,
                        'MV_Difference': result.mv_difference,
                        'Dataset1_Trade_Count': result.dataset1_trade_count,
                        'Dataset2_Trade_Count': result.dataset2_trade_count,
                        'Trade_Count_Difference': result.trade_count_difference,
                        'Missing_in_Dataset1': ';'.join(result.missing_in_dataset1),
                        'Missing_in_Dataset2': ';'.join(result.missing_in_dataset2),
                        'Reasons': ';'.join(result.reasons)
                    })
                
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Reconciliation_Summary', index=False)
                
                # Trade ID analysis sheet
                if hasattr(self, 'trade_id_analysis') and isinstance(self.trade_id_analysis, pd.DataFrame):
                    self.trade_id_analysis.to_excel(writer, sheet_name='Trade_ID_Analysis', index=False)
        except ImportError:
            print("openpyxl not available, saving as CSV instead")
            pd.DataFrame(summary_data).to_csv(filename.replace('.xlsx', '_summary.csv'), index=False)


# Example usage and testing
if __name__ == "__main__":
    print("Generating sample data...")
    
    # Generate sample datasets
    sample_generator = SampleDataGenerator()
    dataset1 = sample_generator.generate_dataset1(1000)
    dataset2 = sample_generator.generate_dataset2(1000, overlap_ratio=0.8)
    
    print("Dataset 1 columns:", list(dataset1.columns))
    print("Dataset 2 columns:", list(dataset2.columns))
    print("\nDataset 1 sample:")
    print(dataset1.head())
    print("\nDataset 2 sample:")
    print(dataset2.head())
    
    # Initialize reconciliator
    reconciliator = HQLATradeReconciliator()
    
    # Load datasets
    reconciliator.load_datasets(dataset1, dataset2)
    
    # Perform reconciliation
    print("\nPerforming reconciliation...")
    results = reconciliator.perform_currency_counterparty_reconciliation()
    
    # Generate trade ID comparison
    print("Generating trade ID comparison...")
    trade_comparison = reconciliator.generate_trade_id_comparison_report()
    
    # Generate summary report
    print("\nSUMMARY REPORT:")
    print("=" * 50)
    print(reconciliator.generate_summary_report())
    
    # Export to Excel (if openpyxl is available)
    try:
        reconciliator.export_results_to_excel('reconciliation_results.xlsx')
        print("\nResults exported to reconciliation_results.xlsx")
    except Exception as e:
        print("Could not export to Excel: {}".format(str(e)))
