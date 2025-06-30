import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple, List, Optional
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

class Dataset1Columns(Enum):
    """Column mapping for Dataset 1"""
    TRADE_ID = 'trade_ref'
    CPTY_CODE = 'counterparty_code'
    CPTY_NAME = 'counterparty_name'
    CONTRA_ACCOUNT = 'contra_acct'
    NOTIONAL = 'notional_amount'
    COLLATERAL_MV = 'collateral_market_value'
    CURRENCY = 'ccy'
    MATURITY_DATE = 'maturity_dt'

class Dataset2Columns(Enum):
    """Column mapping for Dataset 2"""
    TRADE_ID = 'trade_identifier'
    CPTY_CODE = 'cpty_cd'
    CPTY_NAME = 'cpty_nm'
    NOTIONAL = 'notional_val'
    COLLATERAL_MV = 'mkt_val_collateral'
    CURRENCY = 'currency_code'
    MATURITY_DATE = 'mat_date'

class BreakType(Enum):
    """Classification of reconciliation breaks"""
    MISSING_TRADE_DS1 = 'Missing in Dataset 1'
    MISSING_TRADE_DS2 = 'Missing in Dataset 2'
    MV_DIFFERENCE = 'Market Value Difference'
    COUNT_DIFFERENCE = 'Trade Count Difference'
    DATA_QUALITY = 'Data Quality Issue'

class HQLATradeReconciliator:
    """
    HQLA Trade Dataset Reconciliation Class
    Handles preprocessing, reconciliation, and break analysis
    """
    
    def __init__(self, dataset1_columns: Dataset1Columns, dataset2_columns: Dataset2Columns):
        self.ds1_cols = dataset1_columns
        self.ds2_cols = dataset2_columns
        self.processed_ds1 = None
        self.processed_ds2 = None
        self.currency_summary = None
        self.trade_reconciliation = None
        self.break_analysis = None
    
    def preprocess_dataset1(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Dataset 1:
        - Override null counterparty codes with contra account values
        """
        df_processed = df.copy()
        
        # Handle null counterparty codes
        null_mask = df_processed[self.ds1_cols.CPTY_CODE.value] == 'null'
        if null_mask.any():
            print(f"Found {null_mask.sum()} rows with 'null' counterparty codes")
            df_processed.loc[null_mask, self.ds1_cols.CPTY_CODE.value] = \
                df_processed.loc[null_mask, self.ds1_cols.CONTRA_ACCOUNT.value]
            print("Overridden with contra account values")
        
        self.processed_ds1 = df_processed
        return df_processed
    
    def preprocess_dataset2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Dataset 2:
        - No preprocessing required as per requirements
        """
        self.processed_ds2 = df.copy()
        return self.processed_ds2
    
    def currency_counterparty_reconciliation(self) -> pd.DataFrame:
        """
        Perform currency and counterparty level reconciliation
        Compare sum of MV and trade counts
        """
        if self.processed_ds1 is None or self.processed_ds2 is None:
            raise ValueError("Datasets must be preprocessed first")
        
        # Aggregate Dataset 1
        ds1_agg = self.processed_ds1.groupby([
            self.ds1_cols.CURRENCY.value, 
            self.ds1_cols.CPTY_CODE.value
        ]).agg({
            self.ds1_cols.COLLATERAL_MV.value: 'sum',
            self.ds1_cols.TRADE_ID.value: 'count'
        }).reset_index()
        
        ds1_agg.columns = ['currency', 'cpty_code', 'mv_sum_ds1', 'trade_count_ds1']
        
        # Aggregate Dataset 2
        ds2_agg = self.processed_ds2.groupby([
            self.ds2_cols.CURRENCY.value, 
            self.ds2_cols.CPTY_CODE.value
        ]).agg({
            self.ds2_cols.COLLATERAL_MV.value: 'sum',
            self.ds2_cols.TRADE_ID.value: 'count'
        }).reset_index()
        
        ds2_agg.columns = ['currency', 'cpty_code', 'mv_sum_ds2', 'trade_count_ds2']
        
        # Merge and calculate differences
        merged = pd.merge(ds1_agg, ds2_agg, on=['currency', 'cpty_code'], how='outer')
        merged = merged.fillna(0)
        
        merged['mv_difference'] = merged['mv_sum_ds1'] - merged['mv_sum_ds2']
        merged['count_difference'] = merged['trade_count_ds1'] - merged['trade_count_ds2']
        merged['mv_match'] = abs(merged['mv_difference']) < 0.01
        merged['count_match'] = merged['count_difference'] == 0
        
        self.currency_summary = merged
        return merged
    
    def trade_id_reconciliation(self) -> pd.DataFrame:
        """
        Perform trade ID level reconciliation
        Identify missing trades and include original trade details
        """
        if self.processed_ds1 is None or self.processed_ds2 is None:
            raise ValueError("Datasets must be preprocessed first")
        
        # Get trade IDs from both datasets
        ds1_trades = set(self.processed_ds1[self.ds1_cols.TRADE_ID.value])
        ds2_trades = set(self.processed_ds2[self.ds2_cols.TRADE_ID.value])
        
        all_trades = ds1_trades.union(ds2_trades)
        
        # Create reconciliation DataFrame
        recon_data = []
        
        for trade_id in all_trades:
            record = {'trade_id': trade_id}
            
            # Check presence in each dataset
            in_ds1 = trade_id in ds1_trades
            in_ds2 = trade_id in ds2_trades
            
            record['in_dataset1'] = in_ds1
            record['in_dataset2'] = in_ds2
            
            if in_ds1 and in_ds2:
                record['match_status'] = 'Both'
            elif in_ds1:
                record['match_status'] = 'DS1 Only'
            else:
                record['match_status'] = 'DS2 Only'
            
            recon_data.append(record)
        
        recon_df = pd.DataFrame(recon_data)
        
        # Add original trade details with suffixes
        ds1_details = self.processed_ds1.copy()
        ds1_details.columns = [col + '_1' if col != self.ds1_cols.TRADE_ID.value else 'trade_id' 
                              for col in ds1_details.columns]
        
        ds2_details = self.processed_ds2.copy()
        ds2_details.columns = [col + '_2' if col != self.ds2_cols.TRADE_ID.value else 'trade_id' 
                              for col in ds2_details.columns]
        
        # Merge trade details
        recon_with_details = pd.merge(recon_df, ds1_details, on='trade_id', how='left')
        recon_with_details = pd.merge(recon_with_details, ds2_details, on='trade_id', how='left')
        
        self.trade_reconciliation = recon_with_details
        return recon_with_details
    
    def classify_breaks(self) -> pd.DataFrame:
        """
        Classify and analyze the drivers of breaks
        """
        breaks = []
        
        # Currency/Counterparty level breaks
        if self.currency_summary is not None:
            for _, row in self.currency_summary.iterrows():
                if not row['mv_match']:
                    breaks.append({
                        'break_type': BreakType.MV_DIFFERENCE.value,
                        'currency': row['currency'],
                        'cpty_code': row['cpty_code'],
                        'difference_amount': row['mv_difference'],
                        'description': f"MV difference of {row['mv_difference']:,.2f}"
                    })
                
                if not row['count_match']:
                    breaks.append({
                        'break_type': BreakType.COUNT_DIFFERENCE.value,
                        'currency': row['currency'],
                        'cpty_code': row['cpty_code'],
                        'difference_amount': row['count_difference'],
                        'description': f"Count difference of {row['count_difference']}"
                    })
        
        # Trade ID level breaks
        if self.trade_reconciliation is not None:
            for _, row in self.trade_reconciliation.iterrows():
                if row['match_status'] == 'DS1 Only':
                    breaks.append({
                        'break_type': BreakType.MISSING_TRADE_DS2.value,
                        'trade_id': row['trade_id'],
                        'description': f"Trade {row['trade_id']} missing in Dataset 2"
                    })
                elif row['match_status'] == 'DS2 Only':
                    breaks.append({
                        'break_type': BreakType.MISSING_TRADE_DS1.value,
                        'trade_id': row['trade_id'],
                        'description': f"Trade {row['trade_id']} missing in Dataset 1"
                    })
        
        self.break_analysis = pd.DataFrame(breaks)
        return self.break_analysis
    
    def export_to_excel(self, filename: Optional[str] = None) -> str:
        """
        Export all reconciliation results to Excel with separate tabs
        USD and EUR breaks get their own tabs due to large trade populations
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"HQLA_Reconciliation_{timestamp}.xlsx"
        
        print(f"\nExporting reconciliation results to: {filename}")
        
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                sheets_created = 0
                
                # 1. Executive Summary Tab (always create this first)
                try:
                    self._create_executive_summary(writer)
                    sheets_created += 1
                except Exception as e:
                    print(f"Warning: Could not create Executive Summary: {e}")
                    # Create a basic summary sheet as fallback
                    basic_summary = pd.DataFrame([
                        ['Export Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        ['Status', 'Reconciliation Completed']
                    ], columns=['Metric', 'Value'])
                    basic_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)
                    sheets_created += 1
                
                # 2. Currency Summary Tab
                if self.currency_summary is not None and len(self.currency_summary) > 0:
                    try:
                        self.currency_summary.to_excel(writer, sheet_name='Currency_Summary', index=False)
                        print("✓ Currency Summary exported")
                        sheets_created += 1
                    except Exception as e:
                        print(f"Warning: Could not export Currency Summary: {e}")
                else:
                    # Create empty sheet with headers
                    empty_currency = pd.DataFrame(columns=['currency', 'cpty_code', 'mv_sum_ds1', 'mv_sum_ds2', 
                                                          'trade_count_ds1', 'trade_count_ds2', 'mv_difference', 
                                                          'count_difference', 'mv_match', 'count_match'])
                    empty_currency.to_excel(writer, sheet_name='Currency_Summary', index=False)
                    print("✓ Currency Summary exported (empty)")
                    sheets_created += 1
                
                # 3. Trade Reconciliation Tab
                if self.trade_reconciliation is not None and len(self.trade_reconciliation) > 0:
                    try:
                        self.trade_reconciliation.to_excel(writer, sheet_name='Trade_Reconciliation', index=False)
                        print("✓ Trade Reconciliation exported")
                        sheets_created += 1
                    except Exception as e:
                        print(f"Warning: Could not export Trade Reconciliation: {e}")
                else:
                    # Create empty sheet with headers
                    empty_trades = pd.DataFrame(columns=['trade_id', 'in_dataset1', 'in_dataset2', 'match_status'])
                    empty_trades.to_excel(writer, sheet_name='Trade_Reconciliation', index=False)
                    print("✓ Trade Reconciliation exported (empty)")
                    sheets_created += 1
                
                # 4. Overall Break Analysis Tab
                if self.break_analysis is not None and len(self.break_analysis) > 0:
                    try:
                        self.break_analysis.to_excel(writer, sheet_name='All_Breaks', index=False)
                        print("✓ All Breaks exported")
                        sheets_created += 1
                    except Exception as e:
                        print(f"Warning: Could not export All Breaks: {e}")
                else:
                    # Create empty sheet with headers
                    empty_breaks = pd.DataFrame(columns=['break_type', 'currency', 'cpty_code', 'trade_id', 
                                                        'difference_amount', 'description'])
                    empty_breaks.to_excel(writer, sheet_name='All_Breaks', index=False)
                    print("✓ All Breaks exported (empty)")
                    sheets_created += 1
                
                # 5. USD Breaks Tab
                try:
                    self._export_currency_breaks(writer, 'USD')
                    sheets_created += 1
                except Exception as e:
                    print(f"Warning: Could not export USD Breaks: {e}")
                    empty_usd = pd.DataFrame(columns=['break_type', 'currency', 'description'])
                    empty_usd.to_excel(writer, sheet_name='USD_Breaks', index=False)
                    sheets_created += 1
                
                # 6. EUR Breaks Tab
                try:
                    self._export_currency_breaks(writer, 'EUR')
                    sheets_created += 1
                except Exception as e:
                    print(f"Warning: Could not export EUR Breaks: {e}")
                    empty_eur = pd.DataFrame(columns=['break_type', 'currency', 'description'])
                    empty_eur.to_excel(writer, sheet_name='EUR_Breaks', index=False)
                    sheets_created += 1
                
                # 7. Other Currency Breaks Tab
                try:
                    self._export_other_currency_breaks(writer)
                    sheets_created += 1
                except Exception as e:
                    print(f"Warning: Could not export Other Currency Breaks: {e}")
                    empty_other = pd.DataFrame(columns=['break_type', 'currency', 'description'])
                    empty_other.to_excel(writer, sheet_name='Other_Currency_Breaks', index=False)
                    sheets_created += 1
                
                # 8. Data Quality Issues Tab
                try:
                    self._export_data_quality_issues(writer)
                    sheets_created += 1
                except Exception as e:
                    print(f"Warning: Could not export Data Quality Issues: {e}")
                    empty_dq = pd.DataFrame(columns=['Issue_Type', 'Dataset', 'Count', 'Description'])
                    empty_dq.to_excel(writer, sheet_name='Data_Quality_Issues', index=False)
                    sheets_created += 1
                
                # 9. Detailed Trade Analysis Tab
                try:
                    self._export_detailed_trade_analysis(writer)
                    sheets_created += 1
                except Exception as e:
                    print(f"Warning: Could not export Detailed Trade Analysis: {e}")
                    empty_detailed = pd.DataFrame(columns=['trade_id', 'match_status'])
                    empty_detailed.to_excel(writer, sheet_name='Detailed_Trade_Analysis', index=False)
                    sheets_created += 1
                
                if sheets_created == 0:
                    # Fallback: create at least one sheet
                    fallback_df = pd.DataFrame([['Export Error', 'No data available for export']], 
                                             columns=['Status', 'Message'])
                    fallback_df.to_excel(writer, sheet_name='Export_Status', index=False)
                    sheets_created += 1
                
                print(f"✓ Export completed: {filename} ({sheets_created} sheets created)")
                
        except Exception as e:
            print(f"Error during Excel export: {e}")
            # Create a simple fallback Excel file
            try:
                fallback_df = pd.DataFrame([
                    ['Export Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    ['Status', 'Export Error'],
                    ['Error', str(e)]
                ], columns=['Metric', 'Value'])
                fallback_df.to_excel(filename, sheet_name='Export_Error', index=False)
                print(f"✓ Fallback export completed: {filename}")
            except Exception as fallback_error:
                print(f"Could not create fallback file: {fallback_error}")
                raise
        
        return filename
    
    def _create_executive_summary(self, writer):
        """Create executive summary tab with key metrics"""
        try:
            summary_data = []
            
            # Overall statistics
            if self.currency_summary is not None and len(self.currency_summary) > 0:
                total_ccy_cpty_combos = len(self.currency_summary)
                mv_breaks = len(self.currency_summary[~self.currency_summary['mv_match']])
                count_breaks = len(self.currency_summary[~self.currency_summary['count_match']])
                
                summary_data.extend([
                    ['CURRENCY/COUNTERPARTY ANALYSIS', ''],
                    ['Total Currency/Counterparty Combinations', total_ccy_cpty_combos],
                    ['Market Value Breaks', mv_breaks],
                    ['Trade Count Breaks', count_breaks],
                    ['MV Break Rate %', f"{(mv_breaks/total_ccy_cpty_combos*100):.1f}%" if total_ccy_cpty_combos > 0 else "N/A"],
                    ['Count Break Rate %', f"{(count_breaks/total_ccy_cpty_combos*100):.1f}%" if total_ccy_cpty_combos > 0 else "N/A"]
                ])
            else:
                summary_data.extend([
                    ['CURRENCY/COUNTERPARTY ANALYSIS', ''],
                    ['Status', 'No currency summary data available']
                ])
            
            # Trade level statistics
            if self.trade_reconciliation is not None and len(self.trade_reconciliation) > 0:
                total_trades = len(self.trade_reconciliation)
                matched_trades = len(self.trade_reconciliation[self.trade_reconciliation['match_status'] == 'Both'])
                ds1_only = len(self.trade_reconciliation[self.trade_reconciliation['match_status'] == 'DS1 Only'])
                ds2_only = len(self.trade_reconciliation[self.trade_reconciliation['match_status'] == 'DS2 Only'])
                
                summary_data.extend([
                    ['', ''],  # Empty row
                    ['TRADE LEVEL ANALYSIS', ''],
                    ['Total Unique Trades', total_trades],
                    ['Matched Trades', matched_trades],
                    ['Dataset 1 Only', ds1_only],
                    ['Dataset 2 Only', ds2_only],
                    ['Match Rate %', f"{(matched_trades/total_trades*100):.1f}%" if total_trades > 0 else "N/A"]
                ])
            else:
                summary_data.extend([
                    ['', ''],
                    ['TRADE LEVEL ANALYSIS', ''],
                    ['Status', 'No trade reconciliation data available']
                ])
            
            # Currency breakdown
            if self.currency_summary is not None and len(self.currency_summary) > 0:
                try:
                    currency_stats = self.currency_summary.groupby('currency').agg({
                        'mv_sum_ds1': 'sum',
                        'mv_sum_ds2': 'sum',
                        'trade_count_ds1': 'sum',
                        'trade_count_ds2': 'sum'
                    }).reset_index()
                    
                    summary_data.extend([
                        ['', ''],
                        ['CURRENCY BREAKDOWN', ''],
                        ['Currency | DS1 MV | DS2 MV | DS1 Count | DS2 Count', '']
                    ])
                    
                    for _, row in currency_stats.iterrows():
                        summary_data.append([
                            f"{row['currency']} | {row['mv_sum_ds1']:,.0f} | {row['mv_sum_ds2']:,.0f} | {row['trade_count_ds1']:,.0f} | {row['trade_count_ds2']:,.0f}",
                            ''
                        ])
                except Exception as e:
                    summary_data.extend([
                        ['', ''],
                        ['CURRENCY BREAKDOWN', ''],
                        ['Error', f'Could not generate currency breakdown: {e}']
                    ])
            
            # Break analysis summary
            if self.break_analysis is not None and len(self.break_analysis) > 0:
                try:
                    break_summary = self.break_analysis['break_type'].value_counts()
                    summary_data.extend([
                        ['', ''],
                        ['BREAK TYPE SUMMARY', ''],
                    ])
                    
                    for break_type, count in break_summary.items():
                        summary_data.append([break_type, count])
                except Exception as e:
                    summary_data.extend([
                        ['', ''],
                        ['BREAK TYPE SUMMARY', ''],
                        ['Error', f'Could not generate break summary: {e}']
                    ])
            else:
                summary_data.extend([
                    ['', ''],
                    ['BREAK TYPE SUMMARY', ''],
                    ['Status', 'No breaks identified']
                ])
            
            # Always add timestamp
            summary_data.extend([
                ['', ''],
                ['EXPORT INFORMATION', ''],
                ['Export Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ['Report Generated By', 'HQLA Trade Reconciliator']
            ])
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
            print("✓ Executive Summary exported")
            
        except Exception as e:
            print(f"Error creating executive summary: {e}")
            # Create minimal summary
            minimal_summary = pd.DataFrame([
                ['Export Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ['Status', 'Reconciliation Completed'],
                ['Note', 'Detailed summary could not be generated']
            ], columns=['Metric', 'Value'])
            minimal_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)
            print("✓ Minimal Executive Summary exported")
    
    def _export_currency_breaks(self, writer, currency: str):
        """Export breaks for specific currency (USD/EUR)"""
        if self.break_analysis is None:
            return
        
        # Filter breaks for specific currency
        currency_breaks = self.break_analysis[
            self.break_analysis.get('currency', '') == currency
        ].copy()
        
        # Also include trade-level breaks for trades in this currency
        if self.trade_reconciliation is not None:
            trade_breaks = []
            for _, row in self.trade_reconciliation.iterrows():
                if row['match_status'] != 'Both':
                    # Check if trade involves this currency (from either dataset)
                    currency_match = False
                    
                    # Check currency from dataset 1 columns
                    ccy_col_1 = f"{self.ds1_cols.CURRENCY.value}_1"
                    if ccy_col_1 in row and row[ccy_col_1] == currency:
                        currency_match = True
                    
                    # Check currency from dataset 2 columns
                    ccy_col_2 = f"{self.ds2_cols.CURRENCY.value}_2"
                    if ccy_col_2 in row and row[ccy_col_2] == currency:
                        currency_match = True
                    
                    if currency_match:
                        break_type = BreakType.MISSING_TRADE_DS2.value if row['match_status'] == 'DS1 Only' else BreakType.MISSING_TRADE_DS1.value
                        trade_breaks.append({
                            'break_type': break_type,
                            'trade_id': row['trade_id'],
                            'currency': currency,
                            'description': f"Trade {row['trade_id']} missing in {'Dataset 2' if row['match_status'] == 'DS1 Only' else 'Dataset 1'}"
                        })
            
            if trade_breaks:
                trade_breaks_df = pd.DataFrame(trade_breaks)
                currency_breaks = pd.concat([currency_breaks, trade_breaks_df], ignore_index=True)
        
        if len(currency_breaks) > 0:
            currency_breaks.to_excel(writer, sheet_name=f'{currency}_Breaks', index=False)
            print(f"✓ {currency} Breaks exported ({len(currency_breaks)} breaks)")
        else:
            # Create empty sheet with headers
            empty_df = pd.DataFrame(columns=['break_type', 'currency', 'description'])
            empty_df.to_excel(writer, sheet_name=f'{currency}_Breaks', index=False)
            print(f"✓ {currency} Breaks exported (no breaks found)")
    
    def _export_other_currency_breaks(self, writer):
        """Export breaks for currencies other than USD/EUR"""
        if self.break_analysis is None:
            return
        
        other_currency_breaks = self.break_analysis[
            (~self.break_analysis.get('currency', '').isin(['USD', 'EUR'])) |
            (self.break_analysis.get('currency', '').isna())
        ].copy()
        
        if len(other_currency_breaks) > 0:
            other_currency_breaks.to_excel(writer, sheet_name='Other_Currency_Breaks', index=False)
            print(f"✓ Other Currency Breaks exported ({len(other_currency_breaks)} breaks)")
        else:
            # Create empty sheet
            empty_df = pd.DataFrame(columns=['break_type', 'currency', 'description'])
            empty_df.to_excel(writer, sheet_name='Other_Currency_Breaks', index=False)
            print("✓ Other Currency Breaks exported (no breaks found)")
    
    def _export_data_quality_issues(self, writer):
        """Export data quality issues identified during preprocessing"""
        dq_issues = []
        
        # Add preprocessing issues
        if self.processed_ds1 is not None:
            null_count = (self.processed_ds1[self.ds1_cols.CPTY_CODE.value] == 'null').sum()
            if null_count > 0:
                dq_issues.append({
                    'Issue_Type': 'Null Counterparty Codes',
                    'Dataset': 'Dataset 1',
                    'Count': null_count,
                    'Description': 'Counterparty codes with "null" values overridden with contra account'
                })
        
        # Add any other data quality checks here
        if self.currency_summary is not None:
            zero_mv_count = len(self.currency_summary[
                (self.currency_summary['mv_sum_ds1'] == 0) | 
                (self.currency_summary['mv_sum_ds2'] == 0)
            ])
            if zero_mv_count > 0:
                dq_issues.append({
                    'Issue_Type': 'Zero Market Values',
                    'Dataset': 'Both',
                    'Count': zero_mv_count,
                    'Description': 'Currency/Counterparty combinations with zero market values'
                })
        
        if dq_issues:
            dq_df = pd.DataFrame(dq_issues)
            dq_df.to_excel(writer, sheet_name='Data_Quality_Issues', index=False)
            print(f"✓ Data Quality Issues exported ({len(dq_issues)} issues)")
        else:
            # Create empty sheet
            empty_df = pd.DataFrame(columns=['Issue_Type', 'Dataset', 'Count', 'Description'])
            empty_df.to_excel(writer, sheet_name='Data_Quality_Issues', index=False)
            print("✓ Data Quality Issues exported (no issues found)")
    
    def _export_detailed_trade_analysis(self, writer):
        """Export detailed trade-by-trade analysis for matched trades"""
        if self.trade_reconciliation is None:
            return
        
        # Focus on matched trades and compare their attributes
        matched_trades = self.trade_reconciliation[
            self.trade_reconciliation['match_status'] == 'Both'
        ].copy()
        
        if len(matched_trades) > 0:
            # Add comparison columns for key fields
            analysis_cols = ['trade_id', 'match_status']
            
            # Add market value comparison
            mv_col_1 = f"{self.ds1_cols.COLLATERAL_MV.value}_1"
            mv_col_2 = f"{self.ds2_cols.COLLATERAL_MV.value}_2"
            
            if mv_col_1 in matched_trades.columns and mv_col_2 in matched_trades.columns:
                matched_trades['mv_difference'] = matched_trades[mv_col_1] - matched_trades[mv_col_2]
                matched_trades['mv_match'] = abs(matched_trades['mv_difference']) < 0.01
                analysis_cols.extend([mv_col_1, mv_col_2, 'mv_difference', 'mv_match'])
            
            # Add currency comparison
            ccy_col_1 = f"{self.ds1_cols.CURRENCY.value}_1"
            ccy_col_2 = f"{self.ds2_cols.CURRENCY.value}_2"
            
            if ccy_col_1 in matched_trades.columns and ccy_col_2 in matched_trades.columns:
                matched_trades['currency_match'] = matched_trades[ccy_col_1] == matched_trades[ccy_col_2]
                analysis_cols.extend([ccy_col_1, ccy_col_2, 'currency_match'])
            
            # Include all original columns
            all_cols = analysis_cols + [col for col in matched_trades.columns if col not in analysis_cols]
            detailed_analysis = matched_trades[all_cols]
            
            detailed_analysis.to_excel(writer, sheet_name='Detailed_Trade_Analysis', index=False)
            print(f"✓ Detailed Trade Analysis exported ({len(detailed_analysis)} trades)")
        else:
            # Create empty sheet
            empty_df = pd.DataFrame(columns=['trade_id', 'match_status'])
            empty_df.to_excel(writer, sheet_name='Detailed_Trade_Analysis', index=False)
            print("✓ Detailed Trade Analysis exported (no matched trades found)")
    
    def run_full_reconciliation(self, ds1: pd.DataFrame, ds2: pd.DataFrame, 
                               export_to_excel: bool = True, filename: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Run complete reconciliation process with optional Excel export
        """
        print("Starting HQLA Trade Reconciliation...")
        
        # Preprocessing
        print("\n1. Preprocessing datasets...")
        self.preprocess_dataset1(ds1)
        self.preprocess_dataset2(ds2)
        
        # Currency/Counterparty reconciliation
        print("\n2. Currency/Counterparty reconciliation...")
        currency_recon = self.currency_counterparty_reconciliation()
        
        # Trade ID reconciliation
        print("\n3. Trade ID reconciliation...")
        trade_recon = self.trade_id_reconciliation()
        
        # Break analysis
        print("\n4. Break classification...")
        breaks = self.classify_breaks()
        
        print(f"\nReconciliation Complete!")
        print(f"- Currency/Cpty combinations: {len(currency_recon)}")
        print(f"- Total trades analyzed: {len(trade_recon)}")
        print(f"- Total breaks identified: {len(breaks)}")
        
        # Export to Excel if requested
        if export_to_excel:
            print("\n5. Exporting to Excel...")
            excel_file = self.export_to_excel(filename)
            print(f"Excel export completed: {excel_file}")
        
        return {
            'currency_summary': currency_recon,
            'trade_reconciliation': trade_recon,
            'break_analysis': breaks
        }

# Demo with dummy data
def create_dummy_data():
    """Create dummy datasets for demonstration"""
    
    # Dataset 1 (with some null counterparty codes)
    dataset1_data = {
        'trade_ref': ['TRD001', 'TRD002', 'TRD003', 'TRD004', 'TRD005'],
        'counterparty_code': ['CPTY_A', 'null', 'CPTY_C', 'null', 'CPTY_A'],
        'counterparty_name': ['Bank A', 'Bank B', 'Bank C', 'Bank D', 'Bank A'],
        'contra_acct': ['ACC_A', 'CPTY_B', 'ACC_C', 'CPTY_D', 'ACC_A'],
        'notional_amount': [1000000, 2000000, 1500000, 800000, 1200000],
        'collateral_market_value': [950000, 1900000, 1450000, 780000, 1150000],
        'ccy': ['USD', 'EUR', 'USD', 'GBP', 'USD'],
        'maturity_dt': ['2025-12-31', '2025-11-30', '2026-01-15', '2025-10-31', '2026-02-28']
    }
    
    # Dataset 2 (some trades missing, some additional)
    dataset2_data = {
        'trade_identifier': ['TRD001', 'TRD002', 'TRD004', 'TRD006', 'TRD007'],
        'cpty_cd': ['CPTY_A', 'CPTY_B', 'CPTY_D', 'CPTY_A', 'CPTY_C'],
        'cpty_nm': ['Bank A', 'Bank B', 'Bank D', 'Bank A', 'Bank C'],
        'notional_val': [1000000, 2000000, 800000, 900000, 1100000],
        'mkt_val_collateral': [945000, 1905000, 785000, 885000, 1080000],  # Slight differences
        'currency_code': ['USD', 'EUR', 'GBP', 'USD', 'EUR'],
        'mat_date': ['2025-12-31', '2025-11-30', '2025-10-31', '2025-09-30', '2026-03-31']
    }
    
    return pd.DataFrame(dataset1_data), pd.DataFrame(dataset2_data)

# Example usage
if __name__ == "__main__":
    # Create dummy data
    ds1, ds2 = create_dummy_data()
    
    print("Dataset 1:")
    print(ds1)
    print("\nDataset 2:")
    print(ds2)
    
    # Initialize reconciliator
    reconciliator = HQLATradeReconciliator(Dataset1Columns, Dataset2Columns)
    
    # Run full reconciliation with Excel export
    results = reconciliator.run_full_reconciliation(ds1, ds2, export_to_excel=True)
    
    # You can also export separately if needed
    # reconciliator.export_to_excel("Custom_Filename.xlsx")
    
    print("\n" + "="*60)
    print("RECONCILIATION RESULTS")
    print("="*60)
    
    print("\nCurrency/Counterparty Summary:")
    print(results['currency_summary'])
    
    print("\nTrade Reconciliation (first 5 rows):")
    print(results['trade_reconciliation'].head())
    
    print("\nBreak Analysis:")
    print(results['break_analysis'])
