import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import namedtuple
from abc import ABC, abstractmethod
import logging
from enum import Enum

# Configuration structures
FilterConfig = namedtuple('FilterConfig', [
    'remove_wash_books', 'remove_maturing_trades', 'remove_internal_interdesk',
    'remove_non_cash_prds', 'remove_pgi_xmts_notes'
])

EnrichmentConfig = namedtuple('EnrichmentConfig', [
    'simple_enrichments', 'bond_haircut_hqla', 'bond_pool_factors', 
    'tenor_bucketing', 'counterparty_mapping', 'xccy_swap_splitting',
    'bond_ratings', 'bond_issuer', 'collateral_swap', 'market_value'
])

ProcessingConfig = namedtuple('ProcessingConfig', [
    'filter_config', 'enrichment_config', 'non_relevant_prds', 'apply_tactical_fixes', 'column_mapping'
])

# Column name constants
class ColumnNames:
    """Central repository for column names to avoid hardcoded strings"""
    TRADE_ID = 'trade_id'
    BOOK_ID = 'book_id'
    MATURITY_DATE = 'maturity_date'
    TRADE_DATE = 'trade_date'
    PRODUCT_TYPE = 'product_type'
    COUNTERPARTY = 'counterparty'
    CURRENCY = 'currency'
    PAY_CURRENCY = 'pay_currency'
    RECEIVE_CURRENCY = 'receive_currency'
    NOTIONAL = 'notional'
    MARKET_VALUE = 'market_value'
    # Add more column names as needed
    
    @classmethod
    def validate_columns(cls, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, bool]:
        """Validate if required columns exist in dataframe"""
        validation = {}
        for col in required_columns:
            validation[col] = col in df.columns
        return validation
    
    @classmethod
    def get_missing_columns(cls, df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        """Get list of missing required columns"""
        return [col for col in required_columns if col not in df.columns]

# API Integration Classes
class DataLoader:
    """Handles loading data from various sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_pv_trade_data(self, **api_params) -> pd.DataFrame:
        """Load PV trade data via RiskDBLoad API"""
        # Placeholder for RiskDBLoad API call
        self.logger.info("Loading PV trade data from RiskDBLoad API")
        # TODO: Implement actual API call
        # return RiskDBLoad.get_trades(**api_params)
        return pd.DataFrame()  # Placeholder
    
    def load_magallan_trade_data(self, **api_params) -> pd.DataFrame:
        """Load Magallan trade data via TradeDbApi"""
        # Placeholder for TradeDbApi call
        self.logger.info("Loading Magallan trade data from TradeDbApi")
        # TODO: Implement actual API call
        # return TradeDbApi.get_trades(**api_params)
        return pd.DataFrame()  # Placeholder

# Base Classes
class BaseFilter(ABC):
    """Abstract base class for all filters"""
    
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to the dataframe"""
        pass

class BaseEnricher(ABC):
    """Abstract base class for all enrichers"""
    
    @abstractmethod
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply enrichment to the dataframe"""
        pass

# Filter Classes
class WashBooksFilter(BaseFilter):
    """Remove wash books from the dataset"""
    
    def __init__(self, wash_book_ids: frozenset):
        self.wash_book_ids = wash_book_ids
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.wash_book_ids or ColumnNames.BOOK_ID not in df.columns:
            return df
        mask = ~df[ColumnNames.BOOK_ID].isin(self.wash_book_ids)
        return df[mask].copy()

class MaturingTradesFilter(BaseFilter):
    """Remove trades maturing within specified days"""
    
    def __init__(self, days_threshold: int = 1):
        self.days_threshold = days_threshold
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if ColumnNames.MATURITY_DATE not in df.columns:
            return df
        
        cutoff_date = pd.Timestamp.now() + pd.Timedelta(days=self.days_threshold)
        mask = pd.to_datetime(df[ColumnNames.MATURITY_DATE]) > cutoff_date
        return df[mask].copy()

class InternalInterdeskFilter(BaseFilter):
    """Remove internal interdesk trades"""
    
    def __init__(self, internal_counterparties: frozenset):
        self.internal_counterparties = internal_counterparties
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.internal_counterparties or ColumnNames.COUNTERPARTY not in df.columns:
            return df
        mask = ~df[ColumnNames.COUNTERPARTY].isin(self.internal_counterparties)
        return df[mask].copy()

class NonCashPRDFilter(BaseFilter):
    """Remove non-cash relevant PRDs for funding gap analysis"""
    
    def __init__(self, non_cash_prds: frozenset):
        self.non_cash_prds = non_cash_prds
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.non_cash_prds or ColumnNames.PRODUCT_TYPE not in df.columns:
            return df
        mask = ~df[ColumnNames.PRODUCT_TYPE].isin(self.non_cash_prds)
        return df[mask].copy()

class PGIXmtsNotesFilter(BaseFilter):
    """Remove PGI/Xmts notes"""
    
    def __init__(self, pgi_xmts_identifiers: frozenset):
        self.pgi_xmts_identifiers = pgi_xmts_identifiers
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.pgi_xmts_identifiers or ColumnNames.PRODUCT_TYPE not in df.columns:
            return df
        # Assuming product type contains PGI/Xmts identifiers
        mask = ~df[ColumnNames.PRODUCT_TYPE].isin(self.pgi_xmts_identifiers)
        return df[mask].copy()

# Enrichment Classes
class SimpleEnricher(BaseEnricher):
    """Apply simple enrichments like extracting IDs and adding flags"""
    
    def __init__(self, enrichment_rules: Dict[str, Callable]):
        self.enrichment_rules = enrichment_rules
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enriched = df.copy()
        
        for column_name, rule_func in self.enrichment_rules.items():
            df_enriched[column_name] = rule_func(df_enriched)
        
        return df_enriched

class BondHaircutHQLAEnricher(BaseEnricher):
    """Enrich bond haircut and HQLA status"""
    
    def __init__(self, haircut_mapping: Dict, hqla_mapping: Dict):
        self.haircut_mapping = haircut_mapping
        self.hqla_mapping = hqla_mapping
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder implementation
        df_enriched = df.copy()
        # TODO: Implement haircut and HQLA enrichment logic
        df_enriched['bond_haircut'] = np.nan
        df_enriched['hqla_status'] = False
        return df_enriched

class BondPoolFactorEnricher(BaseEnricher):
    """Enrich bond pool factors and market price from file"""
    
    def __init__(self, pool_factor_file_path: str):
        self.pool_factor_file_path = pool_factor_file_path
        self.pool_factors = self._load_pool_factors()
    
    def _load_pool_factors(self) -> pd.DataFrame:
        # Placeholder for loading pool factors from file
        return pd.DataFrame()
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder implementation
        df_enriched = df.copy()
        # TODO: Implement pool factor enrichment logic
        return df_enriched

class TenorBucketingEnricher(BaseEnricher):
    """Apply tenor bucketing to trades"""
    
    def __init__(self, tenor_buckets: Dict[str, Tuple[int, int]]):
        self.tenor_buckets = tenor_buckets
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enriched = df.copy()
        
        if ColumnNames.MATURITY_DATE in df.columns:
            # Calculate days to maturity
            today = pd.Timestamp.now()
            df_enriched['days_to_maturity'] = (
                pd.to_datetime(df_enriched[ColumnNames.MATURITY_DATE]) - today
            ).dt.days
            
            # Apply tenor bucketing
            df_enriched['tenor_bucket'] = self._assign_tenor_buckets(
                df_enriched['days_to_maturity']
            )
        
        return df_enriched
    
    def _assign_tenor_buckets(self, days_to_maturity: pd.Series) -> pd.Series:
        """Assign tenor buckets based on days to maturity"""
        buckets = pd.Series('Other', index=days_to_maturity.index)
        
        for bucket_name, (min_days, max_days) in self.tenor_buckets.items():
            mask = (days_to_maturity >= min_days) & (days_to_maturity <= max_days)
            buckets[mask] = bucket_name
        
        return buckets

class CounterpartyMappingEnricher(BaseEnricher):
    """Map counterparty codes to desk-friendly names"""
    
    def __init__(self, counterparty_mapping: Dict[str, str]):
        self.counterparty_mapping = counterparty_mapping
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enriched = df.copy()
        
        df_enriched['counterparty_name'] = df_enriched[ColumnNames.COUNTERPARTY].map(
            self.counterparty_mapping
        ).fillna(df_enriched[ColumnNames.COUNTERPARTY])
        
        return df_enriched

class XCcySwapSplitter(BaseEnricher):
    """Split XCcy swaps into pay and receive legs"""
    
    XCCY_SWAP_PRODUCT_TYPES = frozenset(['XCCY_SWAP', 'FX_SWAP', 'CURRENCY_SWAP'])
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check if required columns exist
        required_cols = [ColumnNames.PRODUCT_TYPE, ColumnNames.PAY_CURRENCY, 
                        ColumnNames.RECEIVE_CURRENCY, ColumnNames.CURRENCY, 
                        ColumnNames.NOTIONAL, ColumnNames.TRADE_ID]
        missing_cols = ColumnNames.get_missing_columns(df, required_cols)
        
        if missing_cols:
            self.logger.warning(f"XCcy swap splitting skipped - missing columns: {missing_cols}")
            return df
        
        xccy_mask = df[ColumnNames.PRODUCT_TYPE].isin(self.XCCY_SWAP_PRODUCT_TYPES)
        
        if not xccy_mask.any():
            return df
        
        xccy_trades = df[xccy_mask].copy()
        non_xccy_trades = df[~xccy_mask].copy()
        
        # Create pay leg
        pay_leg = xccy_trades.copy()
        pay_leg[ColumnNames.CURRENCY] = xccy_trades[ColumnNames.PAY_CURRENCY]
        pay_leg[ColumnNames.NOTIONAL] = -xccy_trades[ColumnNames.NOTIONAL]  # Negative for pay
        pay_leg['leg_type'] = 'PAY'
        pay_leg[ColumnNames.TRADE_ID] = pay_leg[ColumnNames.TRADE_ID].astype(str) + '_PAY'
        
        # Create receive leg
        receive_leg = xccy_trades.copy()
        receive_leg[ColumnNames.CURRENCY] = xccy_trades[ColumnNames.RECEIVE_CURRENCY]
        receive_leg['leg_type'] = 'RECEIVE'
        receive_leg[ColumnNames.TRADE_ID] = receive_leg[ColumnNames.TRADE_ID].astype(str) + '_RCV'
        
        # Combine all trades
        result = pd.concat([non_xccy_trades, pay_leg, receive_leg], ignore_index=True)
        return result

class BondRatingsEnricher(BaseEnricher):
    """Enrich bond ratings information"""
    
    def __init__(self, ratings_source: str):
        self.ratings_source = ratings_source
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder implementation
        df_enriched = df.copy()
        # TODO: Implement bond ratings enrichment logic
        df_enriched['bond_rating'] = np.nan
        return df_enriched

class BondIssuerEnricher(BaseEnricher):
    """Enrich bond issuer information"""
    
    def __init__(self, issuer_mapping: Dict):
        self.issuer_mapping = issuer_mapping
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        # Placeholder implementation
        df_enriched = df.copy()
        # TODO: Implement bond issuer enrichment logic
        df_enriched['bond_issuer'] = np.nan
        return df_enriched

# Risk Engine Cleaning (Placeholder for your existing class)
class RiskEngineCleaningClass:
    """Placeholder for your existing risk engine cleaning class"""
    
    def __init__(self):
        pass
    
    def clean_and_enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply risk engine cleaning and enrichment"""
        # TODO: Integrate your existing risk engine cleaning class
        return df

# Collateral Swap Enrichment (Placeholder for your existing class)
class CollateralSwapEnricher(BaseEnricher):
    """Placeholder for your existing collateral swap enrichment class"""
    
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply collateral swap identification and enrichment"""
        # TODO: Integrate your existing collateral swap enrichment class
        return df

# Main Pipeline Class
class TradingDataPipeline:
    """Main pipeline for processing trading data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        self.risk_engine_cleaner = RiskEngineCleaningClass()
        
        # Apply column mapping if provided
        if self.config.column_mapping:
            self._apply_column_mapping()
        
        # Storage for dataframes at different pipeline stages
        self.dataframes = {
            'raw_data': None,
            'after_initial_filtering': None,
            'after_tactical_fixes': None,
            'after_risk_engine_cleaning': None,
            'before_filters': None,
            'after_filters': None,
            'before_enrichments': None,
            'final_enriched': None
        }
        
        # Initialize filters
        self._initialize_filters()
        
        # Initialize enrichers
        self._initialize_enrichers()
    
    def _apply_column_mapping(self):
        """Apply column name mapping to ColumnNames class"""
        for standard_name, actual_name in self.config.column_mapping.items():
            if hasattr(ColumnNames, standard_name):
                setattr(ColumnNames, standard_name, actual_name)
                self.logger.info(f"Mapped column {standard_name} -> {actual_name}")
    
    def _initialize_filters(self):
        """Initialize filter instances based on configuration"""
        self.filters = {}
        
        if self.config.filter_config.remove_wash_books:
            self.filters['wash_books'] = WashBooksFilter(frozenset())  # TODO: Pass actual wash book IDs
        
        if self.config.filter_config.remove_maturing_trades:
            self.filters['maturing_trades'] = MaturingTradesFilter()
        
        if self.config.filter_config.remove_internal_interdesk:
            self.filters['internal_interdesk'] = InternalInterdeskFilter(frozenset())  # TODO: Pass actual internal counterparties
        
        if self.config.filter_config.remove_non_cash_prds:
            self.filters['non_cash_prds'] = NonCashPRDFilter(frozenset())  # TODO: Pass actual non-cash PRDs
        
        if self.config.filter_config.remove_pgi_xmts_notes:
            self.filters['pgi_xmts'] = PGIXmtsNotesFilter(frozenset())  # TODO: Pass actual PGI/Xmts identifiers
    
    def _initialize_enrichers(self):
        """Initialize enricher instances based on configuration"""
        self.enrichers = {}
        
        if self.config.enrichment_config.simple_enrichments:
            self.enrichers['simple'] = SimpleEnricher({})  # TODO: Pass actual enrichment rules
        
        if self.config.enrichment_config.bond_haircut_hqla:
            self.enrichers['bond_haircut_hqla'] = BondHaircutHQLAEnricher({}, {})
        
        if self.config.enrichment_config.bond_pool_factors:
            self.enrichers['bond_pool_factors'] = BondPoolFactorEnricher("")  # TODO: Pass actual file path
        
        if self.config.enrichment_config.tenor_bucketing:
            tenor_buckets = {
                'ON': (0, 1),
                'TN': (1, 2),
                '1W': (2, 7),
                '1M': (7, 30),
                '3M': (30, 90),
                '6M': (90, 180),
                '1Y': (180, 365),
                'LongTerm': (365, 999999)
            }
            self.enrichers['tenor_bucketing'] = TenorBucketingEnricher(tenor_buckets)
        
        if self.config.enrichment_config.counterparty_mapping:
            self.enrichers['counterparty_mapping'] = CounterpartyMappingEnricher({})  # TODO: Pass actual mapping
        
        if self.config.enrichment_config.xccy_swap_splitting:
            xccy_splitter = XCcySwapSplitter()
            xccy_splitter.logger = self.logger  # Pass logger to enricher
            self.enrichers['xccy_swap_splitting'] = xccy_splitter
        
        if self.config.enrichment_config.bond_ratings:
            self.enrichers['bond_ratings'] = BondRatingsEnricher("")  # TODO: Pass actual ratings source
        
        if self.config.enrichment_config.bond_issuer:
            self.enrichers['bond_issuer'] = BondIssuerEnricher({})  # TODO: Pass actual issuer mapping
        
        if self.config.enrichment_config.collateral_swap:
            self.enrichers['collateral_swap'] = CollateralSwapEnricher()
        
        # Market value enricher will be added later based on your requirements
    
    def load_data(self, **api_params) -> pd.DataFrame:
        """Load and combine data from all sources"""
        self.logger.info("Starting data loading process")
        
        # Load PV trade data
        pv_data = self.data_loader.load_pv_trade_data(**api_params.get('pv_params', {}))
        
        # Load Magallan trade data
        magallan_data = self.data_loader.load_magallan_trade_data(**api_params.get('magallan_params', {}))
        
        # Combine datasets
        combined_data = pd.concat([pv_data, magallan_data], ignore_index=True)
        
        # Store raw data
        self.dataframes['raw_data'] = combined_data.copy()
        
        # Log available columns for debugging
        self.logger.info(f"Available columns: {list(combined_data.columns)}")
        self.logger.info(f"Loaded {len(combined_data)} total trades")
        
        return combined_data
    
    def apply_initial_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply mandatory initial filtering"""
        self.logger.info("Applying initial filtering")
        
        # Filter out non-relevant PRDs
        if self.config.non_relevant_prds and ColumnNames.PRODUCT_TYPE in df.columns:
            initial_count = len(df)
            mask = ~df[ColumnNames.PRODUCT_TYPE].isin(self.config.non_relevant_prds)
            df = df[mask].copy()
            self.logger.info(f"Filtered out {initial_count - len(df)} non-relevant PRD trades")
        elif self.config.non_relevant_prds:
            self.logger.warning(f"Column '{ColumnNames.PRODUCT_TYPE}' not found. Skipping PRD filtering.")
        
        # Store dataframe after initial filtering
        self.dataframes['after_initial_filtering'] = df.copy()
        
        return df
    
    def apply_tactical_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply tactical fixes to the dataframe"""
        if not self.config.apply_tactical_fixes:
            return df
        
        self.logger.info("Applying tactical fixes")
        # TODO: Integrate your existing tactical fixes function
        # df = your_tactical_fixes_function(df)
        
        # Store dataframe after tactical fixes
        self.dataframes['after_tactical_fixes'] = df.copy()
        
        return df
    
    def apply_risk_engine_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply risk engine cleaning"""
        self.logger.info("Applying risk engine cleaning")
        df = self.risk_engine_cleaner.clean_and_enrich(df)
        
        # Store dataframe after risk engine cleaning
        self.dataframes['after_risk_engine_cleaning'] = df.copy()
        
        return df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured filters"""
        self.logger.info("Applying filters")
        
        # Store dataframe before filters
        self.dataframes['before_filters'] = df.copy()
        
        for filter_name, filter_instance in self.filters.items():
            initial_count = len(df)
            df = filter_instance.apply(df)
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                self.logger.info(f"Filter '{filter_name}' removed {filtered_count} trades")
        
        # Store dataframe after filters
        self.dataframes['after_filters'] = df.copy()
        
        return df
    
    def apply_enrichments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured enrichments"""
        self.logger.info("Applying enrichments")
        
        # Store dataframe before enrichments
        self.dataframes['before_enrichments'] = df.copy()
        
        for enricher_name, enricher_instance in self.enrichers.items():
            self.logger.info(f"Applying enrichment: {enricher_name}")
            df = enricher_instance.enrich(df)
        
        # Store final enriched dataframe
        self.dataframes['final_enriched'] = df.copy()
        
        return df
    
    def export_to_excel(self, file_path: str, export_all_stages: bool = False) -> None:
        """
        Export processed data to Excel
        
        Args:
            file_path: Path for the Excel file
            export_all_stages: If True, export all pipeline stages to separate sheets
        """
        self.logger.info(f"Exporting data to Excel: {file_path}")
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            if export_all_stages:
                # Export all pipeline stages
                for stage_name, df in self.dataframes.items():
                    if df is not None:
                        # Clean sheet name (Excel has 31 char limit and invalid chars)
                        sheet_name = stage_name.replace('_', ' ').title()[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        self.logger.info(f"Exported {len(df)} rows to sheet '{sheet_name}'")
            else:
                # Export only final enriched data
                if self.dataframes['final_enriched'] is not None:
                    self.dataframes['final_enriched'].to_excel(
                        writer, sheet_name='Final Enriched Data', index=False
                    )
                    self.logger.info(f"Exported {len(self.dataframes['final_enriched'])} rows to final sheet")
                else:
                    self.logger.warning("No final enriched data available for export")
    
    def get_dataframe(self, stage: str) -> Optional[pd.DataFrame]:
        """
        Get dataframe from a specific pipeline stage
        
        Args:
            stage: Pipeline stage name (e.g., 'raw_data', 'before_filters', 'final_enriched')
            
        Returns:
            DataFrame from the specified stage or None if not available
        """
        return self.dataframes.get(stage)
    
    def get_pipeline_summary(self) -> Dict[str, int]:
        """
        Get summary of record counts at each pipeline stage
        
        Returns:
            Dictionary with stage names and record counts
        """
        summary = {}
        for stage_name, df in self.dataframes.items():
            summary[stage_name] = len(df) if df is not None else 0
        return summary
    def process(self, export_path: Optional[str] = None, export_all_stages: bool = False, **api_params) -> pd.DataFrame:
        """
        Execute the full processing pipeline
        
        Args:
            export_path: Optional path to export final data to Excel
            export_all_stages: If True and export_path provided, export all pipeline stages
            **api_params: Parameters for API calls
            
        Returns:
            Final processed DataFrame
        """
        self.logger.info("Starting trading data processing pipeline")
        
        # Step 1: Load data
        df = self.load_data(**api_params)
        
        # Step 2: Apply initial filtering
        df = self.apply_initial_filtering(df)
        
        # Step 3: Apply tactical fixes
        df = self.apply_tactical_fixes(df)
        
        # Step 4: Apply risk engine cleaning
        df = self.apply_risk_engine_cleaning(df)
        
        # Step 5: Apply filters
        df = self.apply_filters(df)
        
        # Step 6: Apply enrichments
        df = self.apply_enrichments(df)
        
        # Step 7: Export to Excel if requested
        if export_path:
            self.export_to_excel(export_path, export_all_stages)
        
        self.logger.info(f"Pipeline completed. Final dataset contains {len(df)} trades")
        
        # Log pipeline summary
        summary = self.get_pipeline_summary()
        self.logger.info("Pipeline Summary:")
        for stage, count in summary.items():
            self.logger.info(f"  {stage}: {count:,} records")
        
        return df

# Example usage and configuration
def create_sample_config() -> ProcessingConfig:
    """Create a sample configuration for the pipeline"""
    
    filter_config = FilterConfig(
        remove_wash_books=True,
        remove_maturing_trades=True,
        remove_internal_interdesk=True,
        remove_non_cash_prds=True,
        remove_pgi_xmts_notes=True
    )
    
    enrichment_config = EnrichmentConfig(
        simple_enrichments=True,
        bond_haircut_hqla=True,
        bond_pool_factors=True,
        tenor_bucketing=True,
        counterparty_mapping=False,
        xccy_swap_splitting=True,
        bond_ratings=True,
        bond_issuer=True,
        collateral_swap=True,
        market_value=True
    )
    
    non_relevant_prds = frozenset([
        'NON_RELEVANT_PRD_1',
        'NON_RELEVANT_PRD_2',
        # Add actual non-relevant PRDs here
    ])
    
    # Column mapping for actual data column names
    # Map from ColumnNames attribute to actual column name in your data
    column_mapping = {
        'PRODUCT_TYPE': 'Product',  # Example: if your column is 'Product' instead of 'product_type'
        'TRADE_ID': 'TradeID',
        'BOOK_ID': 'BookID',
        'COUNTERPARTY': 'Counterparty_Code',
        # Add more mappings as needed based on your actual column names
    }
    
    return ProcessingConfig(
        filter_config=filter_config,
        enrichment_config=enrichment_config,
        non_relevant_prds=non_relevant_prds,
        apply_tactical_fixes=True,
        column_mapping=column_mapping
    )

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = create_sample_config()
    
    # Initialize pipeline
    pipeline = TradingDataPipeline(config)
    
    # Process data
    api_params = {
        'pv_params': {'date_from': '2024-01-01', 'date_to': '2024-01-31'},
        'magallan_params': {'date_from': '2024-01-01', 'date_to': '2024-01-31'}
    }
    
    # Process data with Excel export
    processed_data = pipeline.process(
        export_path="trading_data_processed.xlsx",
        export_all_stages=True,  # Export all pipeline stages to separate sheets
        **api_params
    )
    print(f"Processing completed. Final dataset shape: {processed_data.shape}")
    
    # Access dataframes from specific pipeline stages
    raw_data = pipeline.get_dataframe('raw_data')
    before_filters = pipeline.get_dataframe('before_filters') 
    after_filters = pipeline.get_dataframe('after_filters')
    final_data = pipeline.get_dataframe('final_enriched')
    
    # Get pipeline summary
    summary = pipeline.get_pipeline_summary()
    print("Pipeline Summary:")
    for stage, count in summary.items():
        print(f"  {stage}: {count:,} records")
