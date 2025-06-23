from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
import pandas as pd
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Enumeration of supported dataset types"""
    TRADE_DATA = "trade_data"
    RISK_DATA = "risk_data"

class DatasetGenerator(ABC):
    """
    Abstract base class for dataset generators
    Defines the interface that all dataset generators must implement
    """
    
    def __init__(self, load_date: Optional[date] = None):
        """
        Initialize the dataset generator
        
        Args:
            load_date: Reference date for data loading (defaults to today)
        """
        self.load_date = load_date or date.today()
        self.raw_data = None
        self.cleaned_data = None
        self.enriched_data = None
        self._mapping_cache = {}
        
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load raw data from source"""
        pass
    
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parse, clean, and transform the loaded data"""
        pass
    
    def load_mapping_data(self, mapping_type: str) -> pd.DataFrame:
        """
        Load mapping data from wrapper functions
        Default implementation - override in subclasses for specific mappings
        
        Args:
            mapping_type: Type of mapping to load (e.g., 'counterparty', 'instrument', 'currency')
            
        Returns:
            DataFrame containing mapping data
        """
        # Default implementation - subclasses should override this
        logger.warning(f"No mapping loader implemented for {mapping_type}")
        return pd.DataFrame()
    
    def apply_mapping(self, data: pd.DataFrame, mapping_df: pd.DataFrame, 
                     source_col: str, target_col: str, mapping_key_col: str, 
                     mapping_value_col: str) -> pd.DataFrame:
        """
        Apply mapping to enrich dataset
        
        Args:
            data: Source dataset to enrich
            mapping_df: Mapping data
            source_col: Column in source data to map from
            target_col: New column name for mapped values
            mapping_key_col: Key column in mapping data
            mapping_value_col: Value column in mapping data
            
        Returns:
            Enriched dataset
        """
        if mapping_df.empty:
            logger.warning(f"Empty mapping data for {target_col}")
            data[target_col] = None
            return data
        
        # Create mapping dictionary
        mapping_dict = dict(zip(mapping_df[mapping_key_col], mapping_df[mapping_value_col]))
        
        # Apply mapping
        enriched_data = data.copy()
        enriched_data[target_col] = enriched_data[source_col].map(mapping_dict)
        
        # Log mapping statistics
        mapped_count = enriched_data[target_col].notna().sum()
        total_count = len(enriched_data)
        logger.info(f"Applied {target_col} mapping: {mapped_count}/{total_count} records mapped")
        
        return enriched_data
    
    def get_base_dataset(self) -> pd.DataFrame:
        """
        Get the base cleaned dataset without enrichments
        
        Returns:
            Base cleaned dataset
        """
        if self.cleaned_data is None:
            self.generate_dataset()
        return self.cleaned_data.copy()
    
    def get_enriched_dataset(self, enrichments: List[str] = None) -> pd.DataFrame:
        """
        Get enriched dataset with specified enrichments applied
        
        Args:
            enrichments: List of enrichment types to apply (if None, applies all available)
            
        Returns:
            Enriched dataset
        """
        base_data = self.get_base_dataset()
        
        if enrichments is None:
            enrichments = self.get_available_enrichments()
        
        enriched_data = base_data.copy()
        
        for enrichment in enrichments:
            try:
                enriched_data = self.apply_enrichment(enriched_data, enrichment)
            except Exception as e:
                logger.error(f"Failed to apply enrichment {enrichment}: {str(e)}")
        
        self.enriched_data = enriched_data
        return enriched_data
    
    def apply_enrichment(self, data: pd.DataFrame, enrichment_type: str) -> pd.DataFrame:
        """
        Apply specific enrichment to dataset
        Override in subclasses to implement specific enrichments
        
        Args:
            data: Dataset to enrich
            enrichment_type: Type of enrichment to apply
            
        Returns:
            Enriched dataset
        """
        logger.warning(f"Enrichment {enrichment_type} not implemented in {self.__class__.__name__}")
        return data
    
    def get_available_enrichments(self) -> List[str]:
        """
        Get list of available enrichments for this generator
        Override in subclasses
        
        Returns:
            List of available enrichment types
        """
        return []
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Main method to generate a cleaned dataset
        Orchestrates the load and process workflow
        """
        try:
            logger.info(f"Starting dataset generation for {self.__class__.__name__}")
            
            # Load raw data
            self.raw_data = self.load_data()
            logger.info(f"Loaded {len(self.raw_data)} raw records")
            
            # Process data (parse, clean, and transform)
            self.cleaned_data = self.process_data(self.raw_data)
            logger.info(f"Processed to final dataset: {len(self.cleaned_data)} records")
            
            return self.cleaned_data
            
        except Exception as e:
            logger.error(f"Error generating dataset: {str(e)}")
            raise

class TradeDatasetGenerator(DatasetGenerator):
    """
    Generator for trade booking datasets
    Handles trade-specific data loading, cleaning, and transformations
    """
    
    def __init__(self, load_date: Optional[date] = None, 
                 remove_matured: bool = True, 
                 remove_forward_starting: bool = True):
        """
        Initialize trade dataset generator
        
        Args:
            load_date: Reference date for data loading
            remove_matured: Whether to remove matured trades
            remove_forward_starting: Whether to remove forward starting trades
        """
        super().__init__(load_date)
        self.remove_matured = remove_matured
        self.remove_forward_starting = remove_forward_starting
    
    def load_data(self) -> pd.DataFrame:
        """
        Load trade data using wrapper function
        This method should interface with your existing database wrapper
        """
        # Placeholder for your existing wrapper function
        # Replace this with your actual database loading logic
        try:
            # Example structure - replace with your wrapper function call
            trade_data = self._load_trade_data_wrapper()
            return trade_data
        except Exception as e:
            logger.error(f"Failed to load trade data: {str(e)}")
            raise
    
    def _load_trade_data_wrapper(self) -> pd.DataFrame:
        """
        Wrapper function for loading trade data from database
        Replace this with your actual database wrapper function
        """
        # This is a placeholder - implement your actual database loading logic here
        # Example:
        # return your_db_wrapper.get_trades(as_of_date=self.load_date)
        
        # Mock data for demonstration
        mock_data = {
            'trade_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
            'trade_date': ['2024-01-15', '2024-02-10', '2024-03-05', '2024-01-20', '2024-02-28'],
            'maturity_date': ['2024-12-15', '2025-02-10', '2023-12-31', '2025-06-20', '2024-05-28'],
            'start_date': ['2024-01-15', '2024-02-10', '2024-03-05', '2024-12-01', '2024-02-28'],
            'notional': [1000000, 2500000, 500000, 1500000, 750000],
            'currency': ['USD', 'EUR', 'GBP', 'USD', 'JPY'],
            'instrument_type': ['IRS', 'FRA', 'BOND', 'IRS', 'FX_SWAP'],
            'counterparty': ['CP_A', 'CP_B', 'CP_C', 'CP_A', 'CP_D']
        }
        return pd.DataFrame(mock_data)
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process trade data: parse, clean, and apply transformations
        """
        df = data.copy()
        
        # Parse and clean data
        # Convert date columns
        date_columns = ['trade_date', 'maturity_date', 'start_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        
        # Clean numeric columns
        if 'notional' in df.columns:
            df['notional'] = pd.to_numeric(df['notional'], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['trade_id', 'notional'])
        
        # Standardize string columns
        string_columns = ['currency', 'instrument_type', 'counterparty']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
        
        # Apply transformations
        # Remove matured trades (maturity_date <= load_date)
        if self.remove_matured and 'maturity_date' in df.columns:
            initial_count = len(df)
            df = df[df['maturity_date'] > self.load_date]
            removed_count = initial_count - len(df)
            logger.info(f"Removed {removed_count} matured trades")
        
        # Remove forward starting trades (start_date > load_date)
        if self.remove_forward_starting and 'start_date' in df.columns:
            initial_count = len(df)
            df = df[df['start_date'] <= self.load_date]
            removed_count = initial_count - len(df)
            logger.info(f"Removed {removed_count} forward starting trades")
        
        # Add derived columns
        df['days_to_maturity'] = (df['maturity_date'] - self.load_date).dt.days
        df['load_date'] = self.load_date
        
        return df
    
    def load_mapping_data(self, mapping_type: str) -> pd.DataFrame:
        """
        Load mapping data for risk dataset enrichments
        
        Args:
            mapping_type: Type of mapping ('risk_class', 'currency', 'portfolio')
            
        Returns:
            Mapping DataFrame
        """
        # Check cache first
        if mapping_type in self._mapping_cache:
            return self._mapping_cache[mapping_type]
        
        mapping_df = pd.DataFrame()
        
        try:
            if mapping_type == 'risk_class':
                mapping_df = self._load_risk_class_mapping()
            elif mapping_type == 'currency':
                mapping_df = self._load_currency_mapping()
            elif mapping_type == 'portfolio':
                mapping_df = self._load_portfolio_mapping()
            else:
                logger.warning(f"Unknown mapping type: {mapping_type}")
            
            # Cache the mapping
            self._mapping_cache[mapping_type] = mapping_df
            
        except Exception as e:
            logger.error(f"Failed to load {mapping_type} mapping: {str(e)}")
        
        return mapping_df
    
    def _load_risk_class_mapping(self) -> pd.DataFrame:
        """
        Load risk class mapping and details
        Replace with your actual wrapper function
        """
        # Placeholder - replace with your actual wrapper function
        mock_mapping = {
            'risk_class_code': ['IR', 'CR', 'FX', 'EQ', 'CO'],
            'risk_class_name': ['Interest Rate', 'Credit', 'Foreign Exchange', 'Equity', 'Commodity'],
            'risk_category': ['Market Risk', 'Credit Risk', 'Market Risk', 'Market Risk', 'Market Risk'],
            'regulatory_capital_multiplier': [1.0, 1.5, 1.2, 1.3, 1.4]
        }
        return pd.DataFrame(mock_mapping)
    
    def _load_currency_mapping(self) -> pd.DataFrame:
        """
        Load currency mapping (shared with trade dataset)
        """
        # Reuse the same currency mapping as trade dataset
        mock_mapping = {
            'currency_code': ['USD', 'EUR', 'GBP', 'JPY', 'CHF'],
            'currency_name': ['US Dollar', 'Euro', 'British Pound', 'Japanese Yen', 'Swiss Franc'],
            'region': ['Americas', 'Europe', 'Europe', 'Asia', 'Europe'],
            'is_major': [True, True, True, True, False]
        }
        return pd.DataFrame(mock_mapping)
    
    def _load_portfolio_mapping(self) -> pd.DataFrame:
        """
        Load portfolio/book mapping based on trade_id pattern
        Replace with your actual wrapper function
        """
        # Placeholder - replace with your actual wrapper function
        mock_mapping = {
            'trade_id_pattern': ['T001', 'T002', 'T003', 'T004', 'T005'],
            'portfolio': ['Trading Book A', 'Trading Book B', 'Banking Book', 'Trading Book A', 'Trading Book C'],
            'desk': ['Rates', 'Credit', 'Credit', 'Rates', 'FX'],
            'business_unit': ['Fixed Income', 'Credit', 'Credit', 'Fixed Income', 'Markets']
        }
        return pd.DataFrame(mock_mapping)
    
    def apply_enrichment(self, data: pd.DataFrame, enrichment_type: str) -> pd.DataFrame:
        """
        Apply specific enrichment to risk dataset
        
        Args:
            data: Risk dataset to enrich
            enrichment_type: Type of enrichment ('risk_class', 'currency', 'portfolio')
            
        Returns:
            Enriched dataset
        """
        enriched_data = data.copy()
        
        if enrichment_type == 'risk_class':
            mapping_df = self.load_mapping_data('risk_class')
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='risk_class',
                target_col='risk_class_name',
                mapping_key_col='risk_class_code',
                mapping_value_col='risk_class_name'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='risk_class',
                target_col='risk_category',
                mapping_key_col='risk_class_code',
                mapping_value_col='risk_category'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='risk_class',
                target_col='capital_multiplier',
                mapping_key_col='risk_class_code',
                mapping_value_col='regulatory_capital_multiplier'
            )
            
        elif enrichment_type == 'currency':
            mapping_df = self.load_mapping_data('currency')
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='currency',
                target_col='currency_name',
                mapping_key_col='currency_code',
                mapping_value_col='currency_name'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='currency',
                target_col='currency_region',
                mapping_key_col='currency_code',
                mapping_value_col='region'
            )
            
        elif enrichment_type == 'portfolio':
            mapping_df = self.load_mapping_data('portfolio')
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='trade_id',
                target_col='portfolio',
                mapping_key_col='trade_id_pattern',
                mapping_value_col='portfolio'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='trade_id',
                target_col='desk',
                mapping_key_col='trade_id_pattern',
                mapping_value_col='desk'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='trade_id',
                target_col='business_unit',
                mapping_key_col='trade_id_pattern',
                mapping_value_col='business_unit'
            )
        
        return enriched_data
    
    def get_available_enrichments(self) -> List[str]:
        """
        Get available enrichments for risk dataset
        
        Returns:
            List of available enrichment types
        """
        return ['risk_class', 'currency', 'portfolio']
    
    def load_mapping_data(self, mapping_type: str) -> pd.DataFrame:
        """
        Load mapping data for trade dataset enrichments
        
        Args:
            mapping_type: Type of mapping ('counterparty', 'instrument', 'currency')
            
        Returns:
            Mapping DataFrame
        """
        # Check cache first
        if mapping_type in self._mapping_cache:
            return self._mapping_cache[mapping_type]
        
        mapping_df = pd.DataFrame()
        
        try:
            if mapping_type == 'counterparty':
                mapping_df = self._load_counterparty_mapping()
            elif mapping_type == 'instrument':
                mapping_df = self._load_instrument_mapping()
            elif mapping_type == 'currency':
                mapping_df = self._load_currency_mapping()
            else:
                logger.warning(f"Unknown mapping type: {mapping_type}")
            
            # Cache the mapping
            self._mapping_cache[mapping_type] = mapping_df
            
        except Exception as e:
            logger.error(f"Failed to load {mapping_type} mapping: {str(e)}")
        
        return mapping_df
    
    def _load_counterparty_mapping(self) -> pd.DataFrame:
        """
        Load counterparty code to name mapping
        Replace with your actual wrapper function
        """
        # Placeholder - replace with your actual wrapper function
        # Example: return your_db_wrapper.get_counterparty_mapping()
        
        mock_mapping = {
            'cpty_code': ['CP_A', 'CP_B', 'CP_C', 'CP_D'],
            'cpty_name': ['Goldman Sachs', 'JP Morgan', 'Morgan Stanley', 'Deutsche Bank'],
            'cpty_rating': ['A+', 'AA-', 'A', 'A-'],
            'cpty_jurisdiction': ['US', 'US', 'US', 'DE']
        }
        return pd.DataFrame(mock_mapping)
    
    def _load_instrument_mapping(self) -> pd.DataFrame:
        """
        Load instrument type mapping and details
        Replace with your actual wrapper function
        """
        # Placeholder - replace with your actual wrapper function
        mock_mapping = {
            'instrument_code': ['IRS', 'FRA', 'BOND', 'FX_SWAP', 'CDS'],
            'instrument_name': ['Interest Rate Swap', 'Forward Rate Agreement', 
                              'Bond', 'FX Swap', 'Credit Default Swap'],
            'asset_class': ['IR', 'IR', 'CR', 'FX', 'CR'],
            'clearing_eligible': [True, True, False, False, True]
        }
        return pd.DataFrame(mock_mapping)
    
    def _load_currency_mapping(self) -> pd.DataFrame:
        """
        Load currency mapping and details
        Replace with your actual wrapper function
        """
        # Placeholder - replace with your actual wrapper function
        mock_mapping = {
            'currency_code': ['USD', 'EUR', 'GBP', 'JPY', 'CHF'],
            'currency_name': ['US Dollar', 'Euro', 'British Pound', 'Japanese Yen', 'Swiss Franc'],
            'region': ['Americas', 'Europe', 'Europe', 'Asia', 'Europe'],
            'is_major': [True, True, True, True, False]
        }
        return pd.DataFrame(mock_mapping)
    
    def apply_enrichment(self, data: pd.DataFrame, enrichment_type: str) -> pd.DataFrame:
        """
        Apply specific enrichment to trade dataset
        
        Args:
            data: Trade dataset to enrich
            enrichment_type: Type of enrichment ('counterparty', 'instrument', 'currency')
            
        Returns:
            Enriched dataset
        """
        enriched_data = data.copy()
        
        if enrichment_type == 'counterparty':
            mapping_df = self.load_mapping_data('counterparty')
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='counterparty',
                target_col='counterparty_name',
                mapping_key_col='cpty_code',
                mapping_value_col='cpty_name'
            )
            # Add additional counterparty fields
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='counterparty',
                target_col='counterparty_rating',
                mapping_key_col='cpty_code',
                mapping_value_col='cpty_rating'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='counterparty',
                target_col='counterparty_jurisdiction',
                mapping_key_col='cpty_code',
                mapping_value_col='cpty_jurisdiction'
            )
            
        elif enrichment_type == 'instrument':
            mapping_df = self.load_mapping_data('instrument')
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='instrument_type',
                target_col='instrument_name',
                mapping_key_col='instrument_code',
                mapping_value_col='instrument_name'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='instrument_type',
                target_col='asset_class',
                mapping_key_col='instrument_code',
                mapping_value_col='asset_class'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='instrument_type',
                target_col='clearing_eligible',
                mapping_key_col='instrument_code',
                mapping_value_col='clearing_eligible'
            )
            
        elif enrichment_type == 'currency':
            mapping_df = self.load_mapping_data('currency')
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='currency',
                target_col='currency_name',
                mapping_key_col='currency_code',
                mapping_value_col='currency_name'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='currency',
                target_col='currency_region',
                mapping_key_col='currency_code',
                mapping_value_col='region'
            )
            enriched_data = self.apply_mapping(
                enriched_data, mapping_df,
                source_col='currency',
                target_col='is_major_currency',
                mapping_key_col='currency_code',
                mapping_value_col='is_major'
            )
        
        return enriched_data
    
    def get_available_enrichments(self) -> List[str]:
        """
        Get available enrichments for trade dataset
        
        Returns:
            List of available enrichment types
        """
        return ['counterparty', 'instrument', 'currency']

class RiskDatasetGenerator(DatasetGenerator):
    """
    Generator for risk datasets (PV/DV01)
    Handles risk-specific data loading, cleaning, and transformations
    """
    
    def __init__(self, load_date: Optional[date] = None, 
                 risk_metrics: List[str] = None):
        """
        Initialize risk dataset generator
        
        Args:
            load_date: Reference date for data loading
            risk_metrics: List of risk metrics to include (e.g., ['PV', 'DV01'])
        """
        super().__init__(load_date)
        self.risk_metrics = risk_metrics or ['PV', 'DV01']
    
    def load_data(self) -> pd.DataFrame:
        """
        Load risk data using wrapper function
        """
        try:
            risk_data = self._load_risk_data_wrapper()
            return risk_data
        except Exception as e:
            logger.error(f"Failed to load risk data: {str(e)}")
            raise
    
    def _load_risk_data_wrapper(self) -> pd.DataFrame:
        """
        Wrapper function for loading risk data from database
        Replace this with your actual database wrapper function
        """
        # This is a placeholder - implement your actual database loading logic here
        # Example:
        # return your_db_wrapper.get_risk_data(as_of_date=self.load_date, metrics=self.risk_metrics)
        
        # Mock data for demonstration
        mock_data = {
            'trade_id': ['T001', 'T002', 'T003', 'T004', 'T005'],
            'valuation_date': [self.load_date] * 5,
            'PV': [50000, -25000, 75000, 100000, -15000],
            'DV01': [150, 200, 100, 300, 50],
            'currency': ['USD', 'EUR', 'GBP', 'USD', 'JPY'],
            'risk_class': ['IR', 'IR', 'CR', 'IR', 'FX']
        }
        return pd.DataFrame(mock_data)
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process risk data: parse, clean, and apply transformations
        """
        df = data.copy()
        
        # Parse and clean data
        # Convert date columns
        if 'valuation_date' in df.columns:
            df['valuation_date'] = pd.to_datetime(df['valuation_date']).dt.date
        
        # Clean numeric risk metrics
        for metric in self.risk_metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        # Remove rows with missing critical data
        required_columns = ['trade_id'] + self.risk_metrics
        df = df.dropna(subset=required_columns)
        
        # Standardize string columns
        string_columns = ['currency', 'risk_class']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
        
        # Apply transformations
        # Convert PV to base currency (placeholder logic)
        if 'PV' in df.columns and 'currency' in df.columns:
            # Add FX conversion logic here if needed
            df['PV_base'] = df['PV']  # Placeholder - implement FX conversion
        
        # Calculate absolute risk measures
        if 'DV01' in df.columns:
            df['DV01_abs'] = df['DV01'].abs()
        
        # Add risk bucketing
        if 'PV' in df.columns:
            df['pv_bucket'] = pd.cut(df['PV'], 
                                   bins=[-float('inf'), -100000, -10000, 10000, 100000, float('inf')],
                                   labels=['Large Negative', 'Negative', 'Small', 'Positive', 'Large Positive'])
        
        df['load_date'] = self.load_date
        
        return df

class DatasetFactory:
    """
    Factory class for creating dataset generators
    Implements the Factory design pattern
    """
    
    _generators = {
        DatasetType.TRADE_DATA: TradeDatasetGenerator,
        DatasetType.RISK_DATA: RiskDatasetGenerator
    }
    
    @classmethod
    def create_generator(cls, dataset_type: DatasetType, **kwargs) -> DatasetGenerator:
        """
        Create a dataset generator based on the specified type
        
        Args:
            dataset_type: Type of dataset generator to create
            **kwargs: Additional arguments to pass to the generator constructor
            
        Returns:
            DatasetGenerator instance
        """
        if dataset_type not in cls._generators:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        generator_class = cls._generators[dataset_type]
        return generator_class(**kwargs)
    
    @classmethod
    def register_generator(cls, dataset_type: DatasetType, generator_class: type):
        """
        Register a new dataset generator type
        
        Args:
            dataset_type: Type identifier for the generator
            generator_class: Generator class to register
        """
        if not issubclass(generator_class, DatasetGenerator):
            raise ValueError("Generator class must inherit from DatasetGenerator")
        
        cls._generators[dataset_type] = generator_class

class DatasetManager:
    """
    High-level manager class for dataset operations
    Provides convenient methods for common dataset operations
    """
    
    def __init__(self, load_date: Optional[date] = None):
        """
        Initialize dataset manager
        
        Args:
            load_date: Default load date for all operations
        """
        self.load_date = load_date or date.today()
    
    def get_trade_data(self, enrichments: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        Get trade dataset (base or enriched)
        
        Args:
            enrichments: List of enrichments to apply (None for base dataset, 
                        [] for base dataset, ['counterparty', 'instrument'] for specific enrichments)
            **kwargs: Additional arguments for TradeDatasetGenerator
            
        Returns:
            Trade dataset (base or enriched)
        """
        kwargs.setdefault('load_date', self.load_date)
        generator = DatasetFactory.create_generator(DatasetType.TRADE_DATA, **kwargs)
        
        if enrichments is None:
            return generator.get_base_dataset()
        elif len(enrichments) == 0:
            return generator.get_base_dataset()
        else:
            return generator.get_enriched_dataset(enrichments)
    
    def get_risk_data(self, enrichments: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        Get risk dataset (base or enriched)
        
        Args:
            enrichments: List of enrichments to apply (None for base dataset,
                        [] for base dataset, ['risk_class', 'currency'] for specific enrichments)
            **kwargs: Additional arguments for RiskDatasetGenerator
            
        Returns:
            Risk dataset (base or enriched)
        """
        kwargs.setdefault('load_date', self.load_date)
        generator = DatasetFactory.create_generator(DatasetType.RISK_DATA, **kwargs)
        
        if enrichments is None:
            return generator.get_base_dataset()
        elif len(enrichments) == 0:
            return generator.get_base_dataset()
        else:
            return generator.get_enriched_dataset(enrichments)
    
    def get_combined_data(self, trade_enrichments: List[str] = None, 
                         risk_enrichments: List[str] = None, **kwargs) -> pd.DataFrame:
        """
        Get combined trade and risk dataset with optional enrichments
        
        Args:
            trade_enrichments: List of enrichments to apply to trade data
            risk_enrichments: List of enrichments to apply to risk data
            **kwargs: Additional arguments for generators
            
        Returns:
            Combined dataset with trade and risk data
        """
        trade_data = self.get_trade_data(enrichments=trade_enrichments, **kwargs)
        risk_data = self.get_risk_data(enrichments=risk_enrichments, **kwargs)
        
        # Merge on trade_id
        combined_data = pd.merge(trade_data, risk_data, on='trade_id', how='inner')
        
        return combined_data

# Example usage and testing
if __name__ == "__main__":
    # Initialize dataset manager
    manager = DatasetManager(load_date=date(2024, 6, 21))
    
    # Get base trade data (no enrichments)
    print("=== Base Trade Data ===")
    base_trade_df = manager.get_trade_data()
    print(base_trade_df.head())
    print(f"Base trade data shape: {base_trade_df.shape}")
    print(f"Columns: {list(base_trade_df.columns)}")
    
    # Get enriched trade data
    print("\n=== Enriched Trade Data ===")
    enriched_trade_df = manager.get_trade_data(enrichments=['counterparty', 'instrument'])
    print(enriched_trade_df.head())
    print(f"Enriched trade data shape: {enriched_trade_df.shape}")
    print(f"Columns: {list(enriched_trade_df.columns)}")
    
    # Get base risk data
    print("\n=== Base Risk Data ===")
    base_risk_df = manager.get_risk_data()
    print(base_risk_df.head())
    print(f"Base risk data shape: {base_risk_df.shape}")
    
    # Get enriched risk data
    print("\n=== Enriched Risk Data ===")  
    enriched_risk_df = manager.get_risk_data(enrichments=['risk_class', 'portfolio'])
    print(enriched_risk_df.head())
    print(f"Enriched risk data shape: {enriched_risk_df.shape}")
    print(f"Columns: {list(enriched_risk_df.columns)}")
    
    # Get combined enriched data
    print("\n=== Combined Enriched Data ===")
    combined_df = manager.get_combined_data(
        trade_enrichments=['counterparty', 'currency'],
        risk_enrichments=['risk_class']
    )
    print(combined_df.head())
    print(f"Combined data shape: {combined_df.shape}")
    
    # Example of using generator directly for more control
    print("\n=== Direct Generator Usage with Enrichments ===")
    trade_generator = DatasetFactory.create_generator(
        DatasetType.TRADE_DATA,
        load_date=date(2024, 6, 21),
        remove_matured=False
    )
    
    # Get base dataset
    base_dataset = trade_generator.get_base_dataset()
    print(f"Base dataset shape: {base_dataset.shape}")
    
    # Get available enrichments
    available_enrichments = trade_generator.get_available_enrichments()
    print(f"Available enrichments: {available_enrichments}")
    
    # Apply specific enrichments
    enriched_dataset = trade_generator.get_enriched_dataset(['counterparty'])
    print(f"Enriched dataset shape: {enriched_dataset.shape}")
    print("New columns after counterparty enrichment:")
    new_cols = set(enriched_dataset.columns) - set(base_dataset.columns)
    print(list(new_cols))
