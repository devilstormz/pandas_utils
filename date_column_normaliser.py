import pandas as pd
import numpy as np
from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Optional, Union, Any
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookingSystem(Enum):
    """Enumeration of supported booking systems"""
    MUREX = "MUREX"
    SUMMIT = "SUMMIT"
    CALYPSO = "CALYPSO"
    OPENLINK = "OPENLINK"
    GENERIC = "GENERIC"

class ProductType(Enum):
    """Enumeration of supported product types"""
    SWAP = "SWAP"
    FORWARD = "FORWARD"
    OPTION = "OPTION"
    BOND = "BOND"
    REPO = "REPO"
    GENERIC = "GENERIC"

class DateColumnConfig:
    """Configuration class for date column mappings"""
    
    # Standard output column names
    NORMALIZED_START_DATE = "normalized_start_date"
    NORMALIZED_MATURITY_DATE = "normalized_maturity_date"
    
    # Date format patterns (order matters - most specific first)
    DATE_FORMATS = [
        "%Y-%m-%d %H:%M:%S.%f",  # 2023-12-01 10:30:45.123456
        "%Y-%m-%d %H:%M:%S",     # 2023-12-01 10:30:45
        "%Y-%m-%d",              # 2023-12-01
        "%d/%m/%Y",              # 01/12/2023
        "%m/%d/%Y",              # 12/01/2023
        "%d-%m-%Y",              # 01-12-2023
        "%m-%d-%Y",              # 12-01-2023
        "%d.%m.%Y",              # 01.12.2023
        "%Y%m%d",                # 20231201
        "%d%b%Y",                # 01Dec2023
        "%d-%b-%Y",              # 01-Dec-2023
        "%b %d, %Y",             # Dec 01, 2023
    ]
    
    # Null value representations
    NULL_VALUES = {"?", "N/A", "NULL", "null", "", " ", "NaN", "nan", "NaT", "nat"}
    
    # Column name mappings by booking system and product type
    COLUMN_MAPPINGS = {
        BookingSystem.MUREX: {
            ProductType.SWAP: {
                "start_date_cols": ["start_date", "effective_date", "trade_date"],
                "maturity_date_cols": ["maturity_date", "end_date", "termination_date"]
            },
            ProductType.FORWARD: {
                "start_date_cols": ["value_date", "settlement_date", "start_date"],
                "maturity_date_cols": ["maturity_date", "delivery_date", "expiry_date"]
            },
            ProductType.OPTION: {
                "start_date_cols": ["trade_date", "premium_date"],
                "maturity_date_cols": ["expiry_date", "exercise_date", "maturity_date"]
            },
            ProductType.GENERIC: {
                "start_date_cols": ["start_date", "trade_date", "effective_date", "value_date"],
                "maturity_date_cols": ["maturity_date", "end_date", "expiry_date", "termination_date"]
            }
        },
        BookingSystem.SUMMIT: {
            ProductType.SWAP: {
                "start_date_cols": ["effective_dt", "start_dt", "trade_dt"],
                "maturity_date_cols": ["maturity_dt", "end_dt", "term_dt"]
            },
            ProductType.BOND: {
                "start_date_cols": ["issue_date", "settlement_dt"],
                "maturity_date_cols": ["maturity_dt", "redemption_dt"]
            },
            ProductType.GENERIC: {
                "start_date_cols": ["start_dt", "trade_dt", "effective_dt", "value_dt"],
                "maturity_date_cols": ["maturity_dt", "end_dt", "expiry_dt", "term_dt"]
            }
        },
        BookingSystem.CALYPSO: {
            ProductType.REPO: {
                "start_date_cols": ["start_date", "near_date"],
                "maturity_date_cols": ["end_date", "far_date", "maturity_date"]
            },
            ProductType.GENERIC: {
                "start_date_cols": ["start_date", "trade_date", "settlement_date"],
                "maturity_date_cols": ["maturity_date", "end_date", "expiry_date"]
            }
        },
        BookingSystem.GENERIC: {
            ProductType.GENERIC: {
                "start_date_cols": ["start_date", "trade_date", "effective_date", "value_date", "settlement_date"],
                "maturity_date_cols": ["maturity_date", "end_date", "expiry_date", "termination_date", "delivery_date"]
            }
        }
    }

class DateNormalizer:
    """
    A class to normalize date columns from various booking systems and product types.
    
    This class handles:
    - Multiple date formats across different systems
    - Various null value representations
    - System-specific column naming conventions
    - Comprehensive error handling and logging
    """
    
    def __init__(self, booking_system: BookingSystem = BookingSystem.GENERIC, 
                 product_type: ProductType = ProductType.GENERIC):
        """
        Initialize the DateNormalizer.
        
        Args:
            booking_system: The booking system enum
            product_type: The product type enum
        """
        self.booking_system = booking_system
        self.product_type = product_type
        self.config = DateColumnConfig()
        self.stats = {
            "processed_rows": 0,
            "start_date_nulls": 0,
            "maturity_date_nulls": 0,
            "parsing_errors": 0
        }
        
    def _is_null_value(self, value: Any) -> bool:
        """Check if a value represents null/missing data."""
        if pd.isna(value) or value is None:
            return True
        if isinstance(value, str) and value.strip() in self.config.NULL_VALUES:
            return True
        return False
    
    def _parse_date(self, date_value: Any) -> Optional[pd.Timestamp]:
        """
        Parse a date value using multiple format attempts.
        
        Args:
            date_value: The date value to parse
            
        Returns:
            Parsed timestamp or None if parsing fails
        """
        if self._is_null_value(date_value):
            return None
            
        # If already a datetime/timestamp, return as timestamp
        if isinstance(date_value, (datetime, date, pd.Timestamp)):
            return pd.Timestamp(date_value)
        
        # Convert to string for parsing
        date_str = str(date_value).strip()
        
        # Try each format
        for fmt in self.config.DATE_FORMATS:
            try:
                return pd.Timestamp(datetime.strptime(date_str, fmt))
            except ValueError:
                continue
        
        # Try pandas to_datetime as fallback
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse date: {date_value}")
            return None
    
    def _find_date_column(self, df: pd.DataFrame, potential_columns: List[str]) -> Optional[str]:
        """
        Find the first existing column from a list of potential column names.
        
        Args:
            df: The dataframe to search
            potential_columns: List of potential column names
            
        Returns:
            The first matching column name or None
        """
        df_columns_lower = [col.lower() for col in df.columns]
        
        for col in potential_columns:
            if col.lower() in df_columns_lower:
                # Return the original column name with correct case
                return df.columns[df_columns_lower.index(col.lower())]
        return None
    
    def _get_column_mappings(self) -> Dict[str, List[str]]:
        """Get column mappings for the current booking system and product type."""
        try:
            return self.config.COLUMN_MAPPINGS[self.booking_system][self.product_type]
        except KeyError:
            logger.warning(f"No specific mapping for {self.booking_system}/{self.product_type}, using generic")
            return self.config.COLUMN_MAPPINGS[BookingSystem.GENERIC][ProductType.GENERIC]
    
    def normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize date columns in the dataframe.
        
        Args:
            df: Input dataframe with trade data
            
        Returns:
            Dataframe with normalized date columns added
            
        Raises:
            ValueError: If input dataframe is empty or no date columns found
        """
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Get column mappings
        mappings = self._get_column_mappings()
        
        # Find start date column
        start_col = self._find_date_column(df, mappings["start_date_cols"])
        if start_col is None:
            logger.warning("No start date column found, creating with NaT values")
            result_df[self.config.NORMALIZED_START_DATE] = pd.NaT
            self.stats["start_date_nulls"] = len(df)
        else:
            logger.info(f"Using '{start_col}' as start date column")
            result_df[self.config.NORMALIZED_START_DATE] = df[start_col].apply(self._parse_date)
            self.stats["start_date_nulls"] = result_df[self.config.NORMALIZED_START_DATE].isna().sum()
        
        # Find maturity date column
        maturity_col = self._find_date_column(df, mappings["maturity_date_cols"])
        if maturity_col is None:
            logger.warning("No maturity date column found, creating with NaT values")
            result_df[self.config.NORMALIZED_MATURITY_DATE] = pd.NaT
            self.stats["maturity_date_nulls"] = len(df)
        else:
            logger.info(f"Using '{maturity_col}' as maturity date column")
            result_df[self.config.NORMALIZED_MATURITY_DATE] = df[maturity_col].apply(self._parse_date)
            self.stats["maturity_date_nulls"] = result_df[self.config.NORMALIZED_MATURITY_DATE].isna().sum()
        
        self.stats["processed_rows"] = len(df)
        self.stats["parsing_errors"] = (
            self.stats["start_date_nulls"] + self.stats["maturity_date_nulls"]
        )
        
        self._log_statistics()
        return result_df
    
    def _log_statistics(self):
        """Log processing statistics."""
        logger.info(f"Processing complete:")
        logger.info(f"  Rows processed: {self.stats['processed_rows']}")
        logger.info(f"  Start date nulls: {self.stats['start_date_nulls']}")
        logger.info(f"  Maturity date nulls: {self.stats['maturity_date_nulls']}")
        logger.info(f"  Total parsing issues: {self.stats['parsing_errors']}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Return processing statistics."""
        return self.stats.copy()

# Example usage and testing
def create_sample_data() -> pd.DataFrame:
    """Create realistic sample trading data with various date formats and null values."""
    
    sample_data = {
        'trade_id': ['TRD001', 'TRD002', 'TRD003', 'TRD004', 'TRD005', 'TRD006', 'TRD007', 'TRD008'],
        'booking_system': ['MUREX', 'SUMMIT', 'CALYPSO', 'MUREX', 'SUMMIT', 'CALYPSO', 'MUREX', 'GENERIC'],
        'product_type': ['SWAP', 'BOND', 'REPO', 'FORWARD', 'SWAP', 'REPO', 'OPTION', 'SWAP'],
        'start_date': ['2023-12-01', '01/12/2023', '2023-12-01 10:30:00', '20231201', '?', None, '01Dec2023', 'N/A'],
        'maturity_date': ['2028-12-01', '01/12/2028', '2024-06-01 15:45:00', '20241201', 'null', np.nan, '01Dec2024', ''],
        'effective_date': ['2023-12-01', None, '?', '2023-12-01', '01-12-2023', '2023.12.01', 'Dec 01, 2023', '2023-12-01'],
        'end_date': ['2028-12-01', '01/12/2028', 'NaN', '2024-12-01', '01-12-2028', '2024.06.01', 'Dec 01, 2024', '2025-12-01'],
        'notional': [1000000, 500000, 750000, 2000000, 1500000, 300000, 1200000, 800000],
        'currency': ['USD', 'EUR', 'GBP', 'USD', 'EUR', 'GBP', 'USD', 'EUR']
    }
    
    return pd.DataFrame(sample_data)

# Example usage
if __name__ == "__main__":
    # Create sample data
    df = create_sample_data()
    print("Original Data:")
    print(df[['trade_id', 'booking_system', 'product_type', 'start_date', 'maturity_date']].head())
    print("\n")
    
    # Test with different booking systems
    systems_to_test = [
        (BookingSystem.MUREX, ProductType.SWAP),
        (BookingSystem.SUMMIT, ProductType.BOND),
        (BookingSystem.CALYPSO, ProductType.REPO),
        (BookingSystem.GENERIC, ProductType.GENERIC)
    ]
    
    for booking_system, product_type in systems_to_test:
        print(f"\n--- Testing {booking_system.value} / {product_type.value} ---")
        
        # Initialize normalizer
        normalizer = DateNormalizer(booking_system, product_type)
        
        # Normalize dates
        try:
            normalized_df = normalizer.normalize_dates(df)
            
            # Display results
            result_cols = ['trade_id', 'normalized_start_date', 'normalized_maturity_date']
            print(normalized_df[result_cols].head())
            
            # Show statistics
            stats = normalizer.get_statistics()
            print(f"Statistics: {stats}")
            
        except Exception as e:
            print(f"Error processing: {e}")
    
    print("\n--- Testing Error Handling ---")
    
    # Test with empty dataframe
    try:
        empty_normalizer = DateNormalizer()
        empty_df = pd.DataFrame()
        empty_normalizer.normalize_dates(empty_df)
    except ValueError as e:
        print(f"Empty dataframe error handled correctly: {e}")
    
    # Test with dataframe with no date columns
    no_date_df = pd.DataFrame({
        'trade_id': ['TRD001', 'TRD002'],
        'notional': [1000000, 500000],
        'currency': ['USD', 'EUR']
    })
    
    no_date_normalizer = DateNormalizer(BookingSystem.GENERIC, ProductType.GENERIC)
    result = no_date_normalizer.normalize_dates(no_date_df)
    print(f"\nNo date columns - result shape: {result.shape}")
    print(f"Added columns: {[col for col in result.columns if col not in no_date_df.columns]}")
