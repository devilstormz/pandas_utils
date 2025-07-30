from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Union
from collections import defaultdict
import pandas as pd
import numpy as np

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ProductType(Enum):
    """Product type classifications"""
    BOND = "Bond"
    CORP_BOND_FIX = "CorpBondFix"
    CORP_BOND_FLT = "CorpBondFlt"
    CASH = "cash"
    FUNDING_TICKET = "funding_ticket"
    REPO = "Repo"
    REVERSE_REPO = "ReverseRepo"
    BOND_BORROW = "BondBorrow"
    BOND_LEND = "BondLend"
    COLLATERAL_SWAP = "CollateralSwap"


class BookingSystem(Enum):
    """Booking system identifiers"""
    K_PLUS = "K+"
    MUREX = "Murex"
    SUMMIT = "Summit"
    OTHER = "Other"


class Currency(Enum):
    """Major currency codes"""
    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"


class CollateralType(Enum):
    """Collateral type classifications from BBG"""
    GOVERNMENT = "Government"
    CORPORATE = "Corporate"
    COVERED = "Covered"
    SUPRANATIONAL = "Supranational"
    AGENCY = "Agency"


class TradeDirection(Enum):
    """Trade direction indicators"""
    LONG = "Long"
    SHORT = "Short"
    BUY = "Buy"
    SELL = "Sell"


# Frozen sets for product type groupings
BOND_PRODUCTS = set({
    ProductType.BOND,
    ProductType.CORP_BOND_FIX,
    ProductType.CORP_BOND_FLT
})

CASH_RELEVANT_PRODUCTS = set({
    ProductType.CASH,
    ProductType.FUNDING_TICKET,
    ProductType.REPO,
    ProductType.REVERSE_REPO
})

SWAP_PRODUCTS = set({
    ProductType.COLLATERAL_SWAP
})


# ============================================================================
# CONFIGURATION CLASSES (Python 3.6 compatible)
# ============================================================================

class DataCleaningConfig(object):
    """Configuration for data cleaning operations"""
    
    def __init__(self, apply_tactical_fixes=True, fix_maturity_dates=True, 
                 create_standardized_maturity=True, apply_legacy_risk_engine_cleaner=True,
                 maturity_date_tolerance_days=30):
        self.apply_tactical_fixes = apply_tactical_fixes
        self.fix_maturity_dates = fix_maturity_dates
        self.create_standardized_maturity = create_standardized_maturity
        self.apply_legacy_risk_engine_cleaner = apply_legacy_risk_engine_cleaner
        self.maturity_date_tolerance_days = maturity_date_tolerance_days


class DerivedColumnsConfig(object):
    """Configuration for derived column creation"""
    
    def __init__(self, create_cash_ccy=True, create_security_ccy=True,
                 create_cash_fx=True, create_security_fx=True,
                 create_cash_amount=True, create_notional_amount=True,
                 create_mv_amounts=True, create_eur_equivalent_amounts=True,
                 base_currency=Currency.EUR):
        self.create_cash_ccy = create_cash_ccy
        self.create_security_ccy = create_security_ccy
        self.create_cash_fx = create_cash_fx
        self.create_security_fx = create_security_fx
        self.create_cash_amount = create_cash_amount
        self.create_notional_amount = create_notional_amount
        self.create_mv_amounts = create_mv_amounts
        self.create_eur_equivalent_amounts = create_eur_equivalent_amounts
        self.base_currency = base_currency


class EnrichmentsConfig(object):
    """Configuration for data enrichments"""
    
    def __init__(self, identify_collateral_swaps=True, add_is_covered_flag=True,
                 determine_collateral_direction=True, apply_hierarchy_ranking=True):
        self.identify_collateral_swaps = identify_collateral_swaps
        self.add_is_covered_flag = add_is_covered_flag
        self.determine_collateral_direction = determine_collateral_direction
        self.apply_hierarchy_ranking = apply_hierarchy_ranking


class ExcelOutputConfig(object):
    """Configuration for Excel output formatting"""
    
    def __init__(self, enable_output=True, output_file_prefix="trading_data_",
                 export_intermediate_steps=True, navy_blue_headers=True,
                 white_cell_fill=True, format_numbers=True, round_to_nearest=True,
                 negative_with_parentheses=True):
        self.enable_output = enable_output
        self.output_file_prefix = output_file_prefix
        self.export_intermediate_steps = export_intermediate_steps
        self.navy_blue_headers = navy_blue_headers
        self.white_cell_fill = white_cell_fill
        self.format_numbers = format_numbers
        self.round_to_nearest = round_to_nearest
        self.negative_with_parentheses = negative_with_parentheses


class ProcessingConfig(object):
    """Main processing configuration"""
    
    def __init__(self, data_cleaning=None, derived_columns=None, enrichments=None,
                 excel_output=None, validation_enabled=True, debug_mode=False):
        self.data_cleaning = data_cleaning or DataCleaningConfig()
        self.derived_columns = derived_columns or DerivedColumnsConfig()
        self.enrichments = enrichments or EnrichmentsConfig()
        self.excel_output = excel_output or ExcelOutputConfig()
        self.validation_enabled = validation_enabled
        self.debug_mode = debug_mode


# ============================================================================
# COLUMN MAPPING WRAPPER CLASSES
# ============================================================================

class ColumnMappings(object):
    """Centralized column name mappings"""
    
    def __init__(self):
        # Input columns
        self.product_type = "ProductType"
        self.booking_system = "BookingSystem"
        self.bond_maturity = "Bond.Maturity"
        self.leg_arg_maturity = "LegArg.Maturity"
        self.ccy = "CCY"
        self.bond_ccy = "Bond.CCY"
        self.leg_start_cash = "Leg.StartCash"
        self.leg_arg_notional = "LegArg.Notional"
        self.pv = "PV"
        self.bond_bbg_collateral_type = "Bond.BBG.CollateralType"
        self.trade_direction = "TradeDirection"
        
        # Output columns
        self.standardized_maturity = "StandardizedMaturity"
        self.cash_ccy = "CashCCY"
        self.security_ccy = "SecurityCCY"
        self.cash_fx = "CashFX"
        self.security_fx = "SecurityFX"
        self.cash_amount = "CashAmount"
        self.notional_amount = "NotionalAmount"
        self.mv_cash = "MVCash"
        self.mv_security = "MVSecurity"
        self.eur_eq_cash = "EUREquivalentCash"
        self.eur_eq_security = "EUREquivalentSecurity"
        self.is_covered = "IsCovered"
        self.collateral_direction = "CollateralDirection"
        self.swap_type = "SwapType"
        self.hierarchy_score = "HierarchyScore"


# ============================================================================
# RESULTS TRACKING CLASS
# ============================================================================

class ProcessingResults(object):
    """Class to track processing results and intermediate dataframes"""
    
    def __init__(self):
        self.dataframes = defaultdict(pd.DataFrame)
        self.step_metadata = {}
        self.processing_log = []
    
    def add_dataframe(self, step_name, df, metadata=None):
        """Add a dataframe result for a processing step"""
        self.dataframes[step_name] = df.copy()
        if metadata:
            self.step_metadata[step_name] = metadata
        self.processing_log.append({
            'step': step_name,
            'timestamp': pd.Timestamp.now(),
            'rows': len(df),
            'columns': len(df.columns)
        })
    
    def get_dataframe(self, step_name):
        """Get dataframe for a specific step"""
        return self.dataframes.get(step_name, pd.DataFrame())
    
    def get_final_result(self):
        """Get the final processed dataframe"""
        if self.processing_log:
            final_step = self.processing_log[-1]['step']
            return self.dataframes[final_step]
        return pd.DataFrame()
    
    def get_step_names(self):
        """Get list of all processing step names"""
        return list(self.dataframes.keys())


# ============================================================================
# EXCEL EXPORTER CLASS
# ============================================================================

class ExcelExporter(object):
    """Professional Excel exporter using pandas styler with xlsxwriter"""
    
    def __init__(self, config):
        self.config = config
        
    def export_results(self, results, output_filename):
        """Export processing results to Excel with professional formatting using pandas styler"""
        if not self.config.enable_output:
            return
        
        try:
            import xlsxwriter
        except ImportError:
            print("Warning: xlsxwriter not available. Using basic Excel export.")
            self._export_basic(results, output_filename)
            return
        
        # Create Excel writer with xlsxwriter engine
        with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define sheet titles
            sheet_titles = {
                '00_original_input': 'Original Input Data',
                '01_tactical_fixes': 'After Tactical Fixes',
                '02_maturity_processing': 'After Maturity Processing',
                '03_legacy_risk_engine': 'After Legacy Risk Engine',
                '04_derived_columns': 'After Derived Columns',
                '05_enrichments': 'After Enrichments',
                '06_final_result': 'Final Processed Result'
            }
            
            # Export final result
            final_df = results.get_final_result()
            if not final_df.empty:
                self._write_styled_worksheet(writer, final_df, 'Final_Result', 
                                           'Final Processed Trading Data')
            
            # Export intermediate steps if configured
            if self.config.export_intermediate_steps:
                for step_name in results.get_step_names():
                    df = results.get_dataframe(step_name)
                    if not df.empty:
                        sheet_name = step_name.replace(' ', '_')[:31]  # Excel sheet name limit
                        title = sheet_titles.get(step_name, step_name.replace('_', ' ').title())
                        self._write_styled_worksheet(writer, df, sheet_name, title)
            
            # Add summary sheet
            self._write_summary_sheet(writer, results)
            
            # Apply full sheet white fill to all worksheets
            self._apply_full_sheet_formatting(workbook)
        
        print("Excel file exported: {}".format(output_filename))
    
    def _write_styled_worksheet(self, writer, df, sheet_name, title):
        """Write a styled dataframe to worksheet using pandas styler"""
        
        # Prepare dataframe with proper formatting
        df_formatted = self._format_dataframe_values(df.copy())
        
        # Create styler with professional formatting
        styler = df_formatted.style
        
        # Apply header styling (navy blue background, white text)
        styler = styler.set_table_styles([
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', '#1F4E79'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('vertical-align', 'middle'),
                    ('font-size', '11pt'),
                    ('font-family', 'Calibri')
                ]
            },
            {
                'selector': 'tbody td',
                'props': [
                    ('background-color', 'white'),
                    ('text-align', 'left'),
                    ('vertical-align', 'middle'),
                    ('font-size', '10pt'),
                    ('font-family', 'Calibri')
                ]
            }
        ])
        
        # Apply number formatting to numeric columns
        for col in df_formatted.columns:
            if df_formatted[col].dtype in ['int64', 'float64']:
                if col.lower() in ['fx', 'rate', 'score', 'pv'] or any(x in col.lower() for x in ['fx', 'rate']):
                    # Decimal formatting for rates and FX
                    styler = styler.format({col: '{:,.4f}'})
                    styler = styler.set_properties(subset=[col], **{'text-align': 'right'})
                else:
                    # Integer formatting with commas
                    if self.config.negative_with_parentheses:
                        styler = styler.format({col: lambda x: '({:,.0f})'.format(abs(x)) if x < 0 else '{:,.0f}'.format(x)})
                    else:
                        styler = styler.format({col: '{:,.0f}'})
                    styler = styler.set_properties(subset=[col], **{'text-align': 'right'})
        
        # Write to Excel starting from row 2 to leave space for title
        styler.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
        
        # Get the worksheet and add title
        worksheet = writer.sheets[sheet_name]
        
        # Add title in cell A1
        title_format = writer.book.add_format({
            'bold': True,
            'font_size': 14,
            'font_color': '#1F4E79',
            'bg_color': 'white',
            'align': 'left',
            'valign': 'vcenter'
        })
        worksheet.write('A1', title, title_format)
        
        # Auto-adjust column widths
        for idx, col in enumerate(df_formatted.columns):
            # Calculate max width needed
            max_len = max(
                len(str(col)),  # Header length
                df_formatted[col].astype(str).str.len().max() if not df_formatted[col].empty else 0
            )
            # Set column width (minimum 10, maximum 50)
            worksheet.set_column(idx, idx, min(max(max_len + 2, 10), 50))
    
    def _format_dataframe_values(self, df):
        """Format dataframe values for better Excel presentation"""
        df_formatted = df.copy()
        
        # Round numeric values if configured
        if self.config.round_to_nearest:
            for col in df_formatted.columns:
                if df_formatted[col].dtype == 'float64':
                    # Don't round FX rates and scores
                    if not any(x in col.lower() for x in ['fx', 'rate', 'score', 'pv']):
                        df_formatted[col] = df_formatted[col].round(0)
        
        return df_formatted
    
    def _write_summary_sheet(self, writer, results):
        """Write processing summary sheet using pandas styler"""
        
        # Create summary dataframe
        summary_data = []
        for log_entry in results.processing_log:
            summary_data.append({
                'Processing Step': log_entry['step'].replace('_', ' ').title(),
                'Timestamp': log_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Rows': log_entry['rows'],
                'Columns': log_entry['columns']
            })
        
        if not summary_data:
            return
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create styler for summary
        styler = summary_df.style
        
        # Apply styling
        styler = styler.set_table_styles([
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', '#1F4E79'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('vertical-align', 'middle'),
                    ('font-size', '11pt'),
                    ('font-family', 'Calibri')
                ]
            },
            {
                'selector': 'tbody td',
                'props': [
                    ('background-color', 'white'),
                    ('text-align', 'left'),
                    ('vertical-align', 'middle'),
                    ('font-size', '10pt'),
                    ('font-family', 'Calibri')
                ]
            }
        ])
        
        # Format numeric columns
        styler = styler.format({'Rows': '{:,}', 'Columns': '{:,}'})
        styler = styler.set_properties(subset=['Rows', 'Columns'], **{'text-align': 'right'})
        
        # Write to Excel
        styler.to_excel(writer, sheet_name='Processing_Summary', index=False, startrow=2)
        
        # Add title
        worksheet = writer.sheets['Processing_Summary']
        title_format = writer.book.add_format({
            'bold': True,
            'font_size': 14,
            'font_color': '#1F4E79',
            'bg_color': 'white',
            'align': 'left',
            'valign': 'vcenter'
        })
        worksheet.write('A1', 'Processing Summary Report', title_format)
        
        # Auto-adjust column widths
        for idx, col in enumerate(summary_df.columns):
            max_len = max(len(str(col)), summary_df[col].astype(str).str.len().max())
            worksheet.set_column(idx, idx, min(max(max_len + 2, 15), 40))
    
    def _apply_full_sheet_formatting(self, workbook):
        """Apply white background to entire sheets"""
        # This creates a default cell format that will be applied to empty cells
        default_format = workbook.add_format({
            'bg_color': 'white',
            'font_family': 'Calibri',
            'font_size': 10
        })
        
        # Apply to all worksheets
        for worksheet in workbook.worksheets():
            # Set default row height
            worksheet.set_default_row(15)
            # Apply white background to a large range to ensure full sheet coverage
            worksheet.conditional_format('A1:ZZ10000', {
                'type': 'no_errors',
                'format': default_format
            })
    
    def _export_basic(self, results, output_filename):
        """Basic Excel export fallback without advanced formatting"""
        with pd.ExcelWriter(output_filename) as writer:
            # Export final result
            final_df = results.get_final_result()
            if not final_df.empty:
                final_df.to_excel(writer, sheet_name='Final_Result', index=False)
            
            # Export intermediate steps
            if self.config.export_intermediate_steps:
                for step_name in results.get_step_names():
                    df = results.get_dataframe(step_name)
                    sheet_name = step_name.replace(' ', '_')[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, config, column_mappings):
        self.config = config
        self.columns = column_mappings
        self._original_dataframe = pd.DataFrame()
        self._modified_dataframe = pd.DataFrame()
        self._processing_metadata = {}
    
    @property
    def original_dataframe(self):
        """Get the original input dataframe"""
        return self._original_dataframe.copy()
    
    @property
    def modified_dataframe(self):
        """Get the modified output dataframe"""
        return self._modified_dataframe.copy()
    
    @property
    def processing_metadata(self):
        """Get processing metadata"""
        return self._processing_metadata.copy()
    
    def _store_original(self, df):
        """Store the original dataframe"""
        self._original_dataframe = df.copy()
    
    def _store_modified(self, df, metadata=None):
        """Store the modified dataframe"""
        self._modified_dataframe = df.copy()
        if metadata:
            self._processing_metadata.update(metadata)
    
    @abstractmethod
    def process(self, df):
        """Process the dataframe and return the processed result"""
        pass
    
    def _validate_columns(self, df, required_columns):
        """Validate that required columns exist in the dataframe"""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError("Missing required columns: {}".format(missing_cols))


# ============================================================================
# DATA CLEANING PROCESSORS
# ============================================================================

class TacticalFixesProcessor(DataProcessor):
    """Processor for applying tactical data fixes"""
    
    def process(self, df):
        """Apply tactical fixes to the dataframe"""
        self._store_original(df)
        
        if not self.config.apply_tactical_fixes:
            self._store_modified(df)
            return df
        
        # Dummy tactical fixes implementation
        df_cleaned = df.copy()
        
        # Example tactical fixes
        df_cleaned = self._fix_null_product_types(df_cleaned)
        df_cleaned = self._fix_invalid_currencies(df_cleaned)
        df_cleaned = self._fix_negative_amounts(df_cleaned)
        
        metadata = {
            'fixes_applied': ['null_product_types', 'invalid_currencies', 'negative_amounts'],
            'rows_processed': len(df_cleaned)
        }
        
        self._store_modified(df_cleaned, metadata)
        return df_cleaned
    
    def _fix_null_product_types(self, df):
        """Fix null product types based on business rules"""
        # Dummy implementation
        mask = df[self.columns.product_type].isnull()
        df.loc[mask, self.columns.product_type] = ProductType.CASH.value
        return df
    
    def _fix_invalid_currencies(self, df):
        """Fix invalid currency codes"""
        valid_currencies = {c.value for c in Currency}
        
        if self.columns.ccy in df.columns:
            invalid_mask = ~df[self.columns.ccy].isin(valid_currencies)
            df.loc[invalid_mask, self.columns.ccy] = Currency.EUR.value
        
        return df
    
    def _fix_negative_amounts(self, df):
        """Fix negative amounts where they shouldn't be negative"""
        amount_columns = [self.columns.leg_start_cash, self.columns.leg_arg_notional]
        
        for col in amount_columns:
            if col in df.columns:
                df[col] = df[col].abs()
        
        return df


class MaturityDateProcessor(DataProcessor):
    """Processor for fixing and standardizing maturity dates"""
    
    def process(self, df):
        """Process maturity dates"""
        self._store_original(df)
        
        if not self.config.fix_maturity_dates:
            self._store_modified(df)
            return df
        
        df_processed = df.copy()
        
        # Fix bad maturity dates
        df_processed = self._fix_bad_maturity_dates(df_processed)
        
        # Create standardized maturity column
        if self.config.create_standardized_maturity:
            df_processed = self._create_standardized_maturity(df_processed)
        
        metadata = {
            'maturity_dates_fixed': True,
            'standardized_maturity_created': self.config.create_standardized_maturity
        }
        
        self._store_modified(df_processed, metadata)
        return df_processed
    
    def _fix_bad_maturity_dates(self, df):
        """Fix obviously bad maturity dates"""
        maturity_columns = [self.columns.bond_maturity, self.columns.leg_arg_maturity]
        
        for col in maturity_columns:
            if col in df.columns:
                # Convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Fix dates that are too far in the past or future
                today = pd.Timestamp.now()
                min_date = today - pd.Timedelta(days=365*5)  # 5 years ago
                max_date = today + pd.Timedelta(days=365*50)  # 50 years from now
                
                mask = (df[col] < min_date) | (df[col] > max_date)
                df.loc[mask, col] = pd.NaT
        
        return df
    
    def _create_standardized_maturity(self, df):
        """Create standardized maturity column based on product type rules"""
        conditions = [
            df[self.columns.product_type].isin([pt.value for pt in BOND_PRODUCTS]),
            (df[self.columns.product_type].isin([ProductType.CASH.value, ProductType.FUNDING_TICKET.value])) & 
            (df[self.columns.booking_system] == BookingSystem.K_PLUS.value)
        ]
        
        choices = [
            df[self.columns.bond_maturity],
            df[self.columns.leg_arg_maturity]
        ]
        
        df[self.columns.standardized_maturity] = np.select(conditions, choices, default=pd.NaT)
        
        return df


class LegacyRiskEngineProcessor(DataProcessor):
    """Processor for applying legacy risk engine cleaning"""
    
    def process(self, df):
        """Apply legacy risk engine cleaning logic"""
        self._store_original(df)
        
        if not self.config.apply_legacy_risk_engine_cleaner:
            self._store_modified(df)
            return df
        
        # Dummy implementation of legacy risk engine cleaner
        df_cleaned = df.copy()
        
        # Example legacy cleaning operations
        df_cleaned = self._normalize_product_types(df_cleaned)
        df_cleaned = self._clean_booking_system_codes(df_cleaned)
        
        metadata = {
            'legacy_risk_engine_applied': True,
            'normalization_steps': ['product_types', 'booking_systems']
        }
        
        self._store_modified(df_cleaned, metadata)
        return df_cleaned
    
    def _normalize_product_types(self, df):
        """Normalize product type values"""
        # Mapping dictionary for legacy product type values
        product_type_mapping = {
            'bond': ProductType.BOND.value,
            'corpbondfix': ProductType.CORP_BOND_FIX.value,
            'corpbondflt': ProductType.CORP_BOND_FLT.value,
            'cash': ProductType.CASH.value,
            'repo': ProductType.REPO.value,
            'reverserepo': ProductType.REVERSE_REPO.value
        }
        
        df[self.columns.product_type] = df[self.columns.product_type].str.lower().map(
            product_type_mapping
        ).fillna(df[self.columns.product_type])
        
        return df
    
    def _clean_booking_system_codes(self, df):
        """Clean and normalize booking system codes"""
        booking_system_mapping = {
            'k+': BookingSystem.K_PLUS.value,
            'kplus': BookingSystem.K_PLUS.value,
            'murex': BookingSystem.MUREX.value,
            'summit': BookingSystem.SUMMIT.value
        }
        
        df[self.columns.booking_system] = df[self.columns.booking_system].str.lower().map(
            booking_system_mapping
        ).fillna(BookingSystem.OTHER.value)
        
        return df


# ============================================================================
# DERIVED COLUMNS PROCESSORS
# ============================================================================

class DerivedColumnsProcessor(DataProcessor):
    """Processor for creating derived columns"""
    
    def __init__(self, config, column_mappings):
        super(DerivedColumnsProcessor, self).__init__(config, column_mappings)
        self.fx_api_client = DummyFXAPIClient()
    
    def process(self, df):
        """Create all derived columns"""
        self._store_original(df)
        
        df_derived = df.copy()
        created_columns = []
        
        if self.config.create_cash_ccy:
            df_derived = self._create_cash_ccy(df_derived)
            created_columns.append('cash_ccy')
        
        if self.config.create_security_ccy:
            df_derived = self._create_security_ccy(df_derived)
            created_columns.append('security_ccy')
        
        if self.config.create_cash_fx:
            df_derived = self._create_cash_fx(df_derived)
            created_columns.append('cash_fx')
        
        if self.config.create_security_fx:
            df_derived = self._create_security_fx(df_derived)
            created_columns.append('security_fx')
        
        if self.config.create_cash_amount:
            df_derived = self._create_cash_amount(df_derived)
            created_columns.append('cash_amount')
        
        if self.config.create_notional_amount:
            df_derived = self._create_notional_amount(df_derived)
            created_columns.append('notional_amount')
        
        if self.config.create_mv_amounts:
            df_derived = self._create_mv_amounts(df_derived)
            created_columns.extend(['mv_cash', 'mv_security'])
        
        if self.config.create_eur_equivalent_amounts:
            df_derived = self._create_eur_equivalent_amounts(df_derived)
            created_columns.extend(['eur_eq_cash', 'eur_eq_security'])
        
        metadata = {
            'derived_columns_created': created_columns,
            'base_currency': self.config.base_currency.value
        }
        
        self._store_modified(df_derived, metadata)
        return df_derived
    
    def _create_cash_ccy(self, df):
        """Create Cash CCY column"""
        condition = df[self.columns.product_type].isin([pt.value for pt in CASH_RELEVANT_PRODUCTS])
        df[self.columns.cash_ccy] = np.where(condition, df[self.columns.ccy], "")
        return df
    
    def _create_security_ccy(self, df):
        """Create Security CCY column"""
        condition = df[self.columns.product_type].isin([pt.value for pt in CASH_RELEVANT_PRODUCTS])
        df[self.columns.security_ccy] = np.where(condition, df[self.columns.bond_ccy], "")
        return df
    
    def _create_cash_fx(self, df):
        """Create Cash FX rates"""
        if self.columns.cash_ccy not in df.columns:
            return df
        
        fx_rates = self.fx_api_client.get_fx_rates(
            df[self.columns.cash_ccy].dropna().unique().tolist(),
            self.config.base_currency.value
        )
        
        df[self.columns.cash_fx] = df[self.columns.cash_ccy].map(fx_rates).fillna(1.0)
        return df
    
    def _create_security_fx(self, df):
        """Create Security FX rates"""
        if self.columns.security_ccy not in df.columns:
            return df
        
        fx_rates = self.fx_api_client.get_fx_rates(
            df[self.columns.security_ccy].dropna().unique().tolist(),
            self.config.base_currency.value
        )
        
        df[self.columns.security_fx] = df[self.columns.security_ccy].map(fx_rates).fillna(1.0)
        return df
    
    def _create_cash_amount(self, df):
        """Create Cash Amount column"""
        condition = df[self.columns.product_type].isin([pt.value for pt in CASH_RELEVANT_PRODUCTS])
        df[self.columns.cash_amount] = np.where(condition, df[self.columns.leg_start_cash], np.nan)
        return df
    
    def _create_notional_amount(self, df):
        """Create Notional Amount column"""
        condition = df[self.columns.product_type].isin([pt.value for pt in CASH_RELEVANT_PRODUCTS])
        df[self.columns.notional_amount] = np.where(condition, df[self.columns.leg_arg_notional], np.nan)
        return df
    
    def _create_mv_amounts(self, df):
        """Create MV Cash and MV Security columns"""
        # MV Cash
        if self.columns.cash_amount in df.columns:
            df[self.columns.mv_cash] = df[self.columns.cash_amount] * df[self.columns.pv]
        
        # MV Security (using notional amount as proxy)
        if self.columns.notional_amount in df.columns:
            df[self.columns.mv_security] = df[self.columns.notional_amount] * df[self.columns.pv]
        
        return df
    
    def _create_eur_equivalent_amounts(self, df):
        """Create EUR equivalent amounts"""
        # EUR Equivalent Cash
        if all(col in df.columns for col in [self.columns.mv_cash, self.columns.cash_fx]):
            df[self.columns.eur_eq_cash] = df[self.columns.mv_cash] * df[self.columns.cash_fx]
        
        # EUR Equivalent Security
        if all(col in df.columns for col in [self.columns.mv_security, self.columns.security_fx]):
            df[self.columns.eur_eq_security] = df[self.columns.mv_security] * df[self.columns.security_fx]
        
        return df


# ============================================================================
# ENRICHMENT PROCESSORS
# ============================================================================

class CollateralSwapProcessor(DataProcessor):
    """Processor for collateral swap identification and enrichment"""
    
    def __init__(self, config, column_mappings, swap_identifier=None):
        super(CollateralSwapProcessor, self).__init__(config, column_mappings)
        self.swap_identifier = swap_identifier or DummySwapIdentifier()
    
    def process(self, df):
        """Process collateral swap identification and enrichments"""
        self._store_original(df)
        
        df_enriched = df.copy()
        enrichments_applied = []
        
        if self.config.identify_collateral_swaps:
            df_enriched = self.swap_identifier.identify_swaps(df_enriched)
            enrichments_applied.append('collateral_swaps')
        
        if self.config.add_is_covered_flag:
            df_enriched = self._add_is_covered_flag(df_enriched)
            enrichments_applied.append('is_covered_flag')
        
        if self.config.determine_collateral_direction:
            df_enriched = self._determine_collateral_direction(df_enriched)
            enrichments_applied.append('collateral_direction')
        
        metadata = {
            'enrichments_applied': enrichments_applied,
            'swap_identification_enabled': self.config.identify_collateral_swaps
        }
        
        self._store_modified(df_enriched, metadata)
        return df_enriched
    
    def _add_is_covered_flag(self, df):
        """Add IsCovered flag based on BBG collateral type"""
        if self.columns.bond_bbg_collateral_type not in df.columns:
            df[self.columns.is_covered] = False
            return df
        
        df[self.columns.is_covered] = df[self.columns.bond_bbg_collateral_type] == CollateralType.COVERED.value
        return df
    
    def _determine_collateral_direction(self, df):
        """Determine collateral received/given based on product type and trade direction"""
        conditions = [
            (df[self.columns.product_type] == ProductType.REPO.value) & 
            (df[self.columns.trade_direction] == TradeDirection.LONG.value),
            
            (df[self.columns.product_type] == ProductType.REVERSE_REPO.value) & 
            (df[self.columns.trade_direction] == TradeDirection.LONG.value),
            
            (df[self.columns.product_type] == ProductType.BOND_BORROW.value),
            
            (df[self.columns.product_type] == ProductType.BOND_LEND.value),
            
            df[self.columns.product_type] == ProductType.COLLATERAL_SWAP.value
        ]
        
        choices = [
            "Given",      # Repo Long
            "Received",   # Reverse Repo Long  
            "Received",   # Bond Borrow
            "Given",      # Bond Lend
            "Swap"        # Collateral Swap
        ]
        
        df[self.columns.collateral_direction] = np.select(conditions, choices, default="Unknown")
        return df


# ============================================================================
# DUMMY HELPER CLASSES
# ============================================================================

class DummyFXAPIClient(object):
    """Dummy FX API client for demonstration"""
    
    def get_fx_rates(self, currencies, base_currency):
        """Get FX rates for currencies against base currency"""
        # Dummy FX rates - in real implementation, this would call an actual API
        dummy_rates = {
            "EUR": 1.0,
            "USD": 0.85,
            "GBP": 1.15,
            "JPY": 0.0075,
            "CHF": 0.95,
            "CAD": 0.65,
            "": 1.0  # For empty currency values
        }
        
        return {ccy: dummy_rates.get(ccy, 1.0) for ccy in currencies}


class DummySwapIdentifier(object):
    """Dummy swap identifier for demonstration"""
    
    def identify_swaps(self, df):
        """Identify and enrich collateral swaps"""
        # Dummy implementation - in real code, this would contain complex business logic
        df_enriched = df.copy()
        
        # Add dummy swap type based on some basic logic
        df_enriched["SwapType"] = np.where(
            df["ProductType"] == ProductType.COLLATERAL_SWAP.value,
            "QualityUpgrade",
            ""
        )
        
        # Add dummy hierarchy score
        df_enriched["HierarchyScore"] = np.where(
            df["ProductType"] == ProductType.COLLATERAL_SWAP.value,
            np.random.uniform(0.1, 1.0, len(df)),
            0.0
        )
        
        return df_enriched


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

class TradingDataProcessor(object):
    """Main processing pipeline for trading data"""
    
    def __init__(self, config=None, column_mappings=None):
        self.config = config or ProcessingConfig()
        self.columns = column_mappings or ColumnMappings()
        self.results = ProcessingResults()
        self.excel_exporter = ExcelExporter(self.config.excel_output)
        
        # Initialize processors
        self._init_processors()
    
    def _init_processors(self):
        """Initialize all processor instances"""
        # Data cleaning processors
        self.tactical_fixes_processor = TacticalFixesProcessor(
            self.config.data_cleaning, self.columns
        )
        self.maturity_processor = MaturityDateProcessor(
            self.config.data_cleaning, self.columns
        )
        self.legacy_risk_processor = LegacyRiskEngineProcessor(
            self.config.data_cleaning, self.columns
        )
        
        # Derived columns processor
        self.derived_columns_processor = DerivedColumnsProcessor(
            self.config.derived_columns, self.columns
        )
        
        # Enrichment processors
        self.collateral_swap_processor = CollateralSwapProcessor(
            self.config.enrichments, self.columns
        )
    
    def process(self, df):
        """Execute the complete processing pipeline"""
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        df_processed = df.copy()
        
        # Store original input
        self.results.add_dataframe('00_original_input', df_processed)
        
        # Step 1: Data Cleaning
        if self.config.debug_mode:
            print("Starting data cleaning...")
        
        # Tactical fixes
        df_processed = self.tactical_fixes_processor.process(df_processed)
        self.results.add_dataframe('01_tactical_fixes', df_processed, 
                                 self.tactical_fixes_processor.processing_metadata)
        
        # Maturity processing
        df_processed = self.maturity_processor.process(df_processed)
        self.results.add_dataframe('02_maturity_processing', df_processed,
                                 self.maturity_processor.processing_metadata)
        
        # Legacy risk engine
        df_processed = self.legacy_risk_processor.process(df_processed)
        self.results.add_dataframe('03_legacy_risk_engine', df_processed,
                                 self.legacy_risk_processor.processing_metadata)
        
        # Step 2: Derived Columns
        if self.config.debug_mode:
            print("Creating derived columns...")
        
        df_processed = self.derived_columns_processor.process(df_processed)
        self.results.add_dataframe('04_derived_columns', df_processed,
                                 self.derived_columns_processor.processing_metadata)
        
        # Step 3: Enrichments
        if self.config.debug_mode:
            print("Applying enrichments...")
        
        df_processed = self.collateral_swap_processor.process(df_processed)
        self.results.add_dataframe('05_enrichments', df_processed,
                                 self.collateral_swap_processor.processing_metadata)
        
        # Store final result
        self.results.add_dataframe('06_final_result', df_processed)
        
        # Step 4: Validation
        if self.config.validation_enabled:
            self._validate_output(df_processed)
        
        # Step 5: Excel Export
        if self.config.excel_output.enable_output:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_filename = "{}processed_{}.xlsx".format(
                self.config.excel_output.output_file_prefix, timestamp
            )
            self.excel_exporter.export_results(self.results, output_filename)
        
        if self.config.debug_mode:
            print("Processing completed successfully")
            print("Total steps processed: {}".format(len(self.results.processing_log)))
        
        return df_processed
    
    def _validate_output(self, df):
        """Validate the processed output"""
        # Basic validation checks
        if df.empty:
            raise ValueError("Processed dataframe is empty")
        
        # Check for critical derived columns
        expected_columns = []
        if self.config.derived_columns.create_cash_ccy:
            expected_columns.append(self.columns.cash_ccy)
        if self.config.derived_columns.create_security_ccy:
            expected_columns.append(self.columns.security_ccy)
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError("Missing expected derived columns: {}".format(missing_columns))
        
        if self.config.debug_mode:
            print("Validation passed - all expected columns present")
    
    def get_processing_results(self):
        """Get the complete processing results object"""
        return self.results
    
    def get_dataframe_at_step(self, step_name):
        """Get dataframe at a specific processing step"""
        return self.results.get_dataframe(step_name)
    
    def export_to_excel(self, output_filename=None):
        """Export results to Excel with custom filename"""
        if output_filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_filename = "{}processed_{}.xlsx".format(
                self.config.excel_output.output_file_prefix, timestamp
            )
        
        self.excel_exporter.export_results(self.results, output_filename)
        return output_filename


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Example of how to use the trading data processing framework"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'ProductType': ['Bond', 'cash', 'CorpBondFix', 'Repo'],
        'BookingSystem': ['K+', 'Murex', 'K+', 'Summit'],
        'Bond.Maturity': ['2025-12-31', '2024-06-30', '2026-03-15', '2024-12-31'],
        'LegArg.Maturity': ['2024-12-31', '2024-07-15', '2025-01-30', '2024-11-30'],
        'CCY': ['EUR', 'USD', 'EUR', 'GBP'],
        'Bond.CCY': ['EUR', 'USD', 'EUR', 'GBP'],
        'Leg.StartCash': [1000000, 500000, 2000000, 750000],
        'LegArg.Notional': [1000000, 500000, 2000000, 750000],
        'PV': [1.0, 0.98, 1.02, 0.99],
        'Bond.BBG.CollateralType': ['Government', 'Corporate', 'Covered', 'Government'],
        'TradeDirection': ['Long', 'Short', 'Long', 'Long']
    })
    
    # Create custom configuration
    config = ProcessingConfig(
        data_cleaning=DataCleaningConfig(
            apply_tactical_fixes=True,
            fix_maturity_dates=True,
            create_standardized_maturity=True
        ),
        derived_columns=DerivedColumnsConfig(
            create_cash_ccy=True,
            create_security_ccy=True,
            create_eur_equivalent_amounts=True
        ),
        enrichments=EnrichmentsConfig(
            add_is_covered_flag=True,
            determine_collateral_direction=True
        ),
        excel_output=ExcelOutputConfig(
            enable_output=True,
            export_intermediate_steps=True,
            navy_blue_headers=True,
            format_numbers=True,
            negative_with_parentheses=True
        ),
        debug_mode=True
    )
    
    # Initialize processor
    processor = TradingDataProcessor(config)
    
    # Process the data
    result = processor.process(sample_data)
    
    # Get processing results
    processing_results = processor.get_processing_results()
    
    # Access specific processing steps
    original_data = processor.get_dataframe_at_step('00_original_input')
    tactical_fixes_data = processor.get_dataframe_at_step('01_tactical_fixes')
    final_data = processor.get_dataframe_at_step('06_final_result')
    
    return result, processing_results


# ============================================================================
# ADVANCED USAGE EXAMPLES
# ============================================================================

def create_custom_processor_config():
    """Example of creating a custom configuration for specific use cases"""
    
    # Configuration for minimal processing (only essential fixes)
    minimal_config = ProcessingConfig(
        data_cleaning=DataCleaningConfig(
            apply_tactical_fixes=True,
            fix_maturity_dates=False,
            create_standardized_maturity=False,
            apply_legacy_risk_engine_cleaner=False
        ),
        derived_columns=DerivedColumnsConfig(
            create_cash_ccy=True,
            create_security_ccy=False,
            create_cash_fx=False,
            create_security_fx=False,
            create_mv_amounts=False,
            create_eur_equivalent_amounts=False
        ),
        enrichments=EnrichmentsConfig(
            identify_collateral_swaps=False,
            add_is_covered_flag=True,
            determine_collateral_direction=False
        ),
        excel_output=ExcelOutputConfig(
            enable_output=True,
            export_intermediate_steps=False
        )
    )
    
    # Configuration for full processing with custom Excel formatting
    full_config = ProcessingConfig(
        excel_output=ExcelOutputConfig(
            enable_output=True,
            output_file_prefix="credit_repo_full_",
            export_intermediate_steps=True,
            navy_blue_headers=True,
            white_cell_fill=True,
            format_numbers=True,
            round_to_nearest=True,
            negative_with_parentheses=True
        ),
        debug_mode=True
    )
    
    return minimal_config, full_config


def process_with_custom_styler():
    """Example of using custom pandas styler for specific formatting needs"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'ProductType': ['Bond', 'Repo', 'Cash'],
        'Amount': [1000000, -500000, 2000000],
        'FXRate': [1.0, 0.85, 1.15],
        'PV': [1.02, 0.98, 1.00]
    })
    
    # Create processor with specific Excel configuration
    config = ProcessingConfig(
        excel_output=ExcelOutputConfig(
            enable_output=True,
            output_file_prefix="styled_report_",
            navy_blue_headers=True,
            white_cell_fill=True,
            format_numbers=True,
            negative_with_parentheses=True
        )
    )
    
    processor = TradingDataProcessor(config)
    result = processor.process(sample_data)
    
    return result


if __name__ == "__main__":
    # Run example
    processed_data, results = example_usage()
    print("\nProcessed Data Shape: {}".format(processed_data.shape))
    print("\nProcessed Data Columns:")
    print(processed_data.columns.tolist())
    print("\nSample of processed data:")
    print(processed_data.head())
    
    print("\nProcessing Steps Completed:")
    for step in results.get_step_names():
        df = results.get_dataframe(step)
        print("  {}: {} rows, {} columns".format(step, len(df), len(df.columns)))
