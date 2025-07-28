"""
Trade Data Enrichment Framework

A comprehensive framework for enriching trade datasets with reference data.
Supports multiple enrichment types with factory pattern and automatic registration.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import namedtuple
import pandas as pd
import numpy as np
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class EnrichmentError(Exception):
    """Base exception for enrichment operations"""
    pass


class EnrichmentDataError(EnrichmentError):
    """Exception raised when enrichment data is invalid or missing"""
    pass


class EnrichmentProcessingError(EnrichmentError):
    """Exception raised during enrichment processing"""
    pass


class EnrichmentConfigurationError(EnrichmentError):
    """Exception raised for configuration issues"""
    pass


class ReferenceDataError(EnrichmentError):
    """Exception raised when reference data cannot be loaded"""
    pass


class RatingCalculationError(EnrichmentError):
    """Exception raised during rating calculations"""
    pass

# =============================================================================
# REPORT AND ENRICHMENT CATEGORIES
# =============================================================================

class ReportType(Enum):
    """Types of reports that can be generated"""
    DATA_EXTRACT = "DATA_EXTRACT"
    CREDIT_RISK = "CREDIT_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    MARKET_RISK = "MARKET_RISK"
    REGULATORY = "REGULATORY"
    PORTFOLIO_ANALYTICS = "PORTFOLIO_ANALYTICS"
    TRADING_ANALYTICS = "TRADING_ANALYTICS"
    COMPLIANCE = "COMPLIANCE"


class EnrichmentCategory(Enum):
    """Categories of enrichments for grouping"""
    CREDIT = "CREDIT"
    LIQUIDITY = "LIQUIDITY" 
    CLASSIFICATION = "CLASSIFICATION"
    COUNTERPARTY = "COUNTERPARTY"
    PRICING = "PRICING"
    REGULATORY = "REGULATORY"
    RISK = "RISK"


# Named tuple for enrichment metadata (simplified)
EnrichmentMetadata = namedtuple('EnrichmentMetadata', [
    'name', 'category', 'enabled', 'required_for_reports', 
    'description', 'dependencies'
])


# =============================================================================
# ENUMS AND NAMEDTUPLES
# =============================================================================

class RatingScale(Enum):
    """Normalized rating scale for all agencies"""
    AAA = 1
    AA_PLUS = 2
    AA = 3
    AA_MINUS = 4
    A_PLUS = 5
    A = 6
    A_MINUS = 7
    BBB_PLUS = 8
    BBB = 9
    BBB_MINUS = 10
    BB_PLUS = 11
    BB = 12
    BB_MINUS = 13
    B_PLUS = 14
    B = 15
    B_MINUS = 16
    CCC_PLUS = 17
    CCC = 18
    CCC_MINUS = 19
    CC = 20
    C = 21
    D = 22
    NR = 999  # Not Rated


class RatingAgency(Enum):
    """Rating agencies enumeration"""
    SP = "S&P"
    MOODY = "Moody's"
    FITCH = "Fitch"


class EnrichmentStatus(Enum):
    """Status of enrichment operation"""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NO_DATA = "NO_DATA"


# Named tuples for structured data
EnrichmentResult = namedtuple('EnrichmentResult', ['status', 'message', 'data'])
RatingInfo = namedtuple('RatingInfo', ['rating', 'agency', 'scale_value'])
RatingSummary = namedtuple('RatingSummary', [
    'average_rating', 'best_rating', 'worst_rating', 
    'best_agency', 'worst_agency', 'rating_count'
])

# =============================================================================
# RATING MAPPERS
# =============================================================================

class RatingMapper:
    """Maps agency-specific ratings to normalized scale"""
    
    def __init__(self):
        self._sp_mapping = {
            'AAA': RatingScale.AAA,
            'AA+': RatingScale.AA_PLUS,
            'AA': RatingScale.AA,
            'AA-': RatingScale.AA_MINUS,
            'A+': RatingScale.A_PLUS,
            'A': RatingScale.A,
            'A-': RatingScale.A_MINUS,
            'BBB+': RatingScale.BBB_PLUS,
            'BBB': RatingScale.BBB,
            'BBB-': RatingScale.BBB_MINUS,
            'BB+': RatingScale.BB_PLUS,
            'BB': RatingScale.BB,
            'BB-': RatingScale.BB_MINUS,
            'B+': RatingScale.B_PLUS,
            'B': RatingScale.B,
            'B-': RatingScale.B_MINUS,
            'CCC+': RatingScale.CCC_PLUS,
            'CCC': RatingScale.CCC,
            'CCC-': RatingScale.CCC_MINUS,
            'CC': RatingScale.CC,
            'C': RatingScale.C,
            'D': RatingScale.D,
            'NR': RatingScale.NR
        }
        
        self._moody_mapping = {
            'Aaa': RatingScale.AAA,
            'Aa1': RatingScale.AA_PLUS,
            'Aa2': RatingScale.AA,
            'Aa3': RatingScale.AA_MINUS,
            'A1': RatingScale.A_PLUS,
            'A2': RatingScale.A,
            'A3': RatingScale.A_MINUS,
            'Baa1': RatingScale.BBB_PLUS,
            'Baa2': RatingScale.BBB,
            'Baa3': RatingScale.BBB_MINUS,
            'Ba1': RatingScale.BB_PLUS,
            'Ba2': RatingScale.BB,
            'Ba3': RatingScale.BB_MINUS,
            'B1': RatingScale.B_PLUS,
            'B2': RatingScale.B,
            'B3': RatingScale.B_MINUS,
            'Caa1': RatingScale.CCC_PLUS,
            'Caa2': RatingScale.CCC,
            'Caa3': RatingScale.CCC_MINUS,
            'Ca': RatingScale.CC,
            'C': RatingScale.C,
            'D': RatingScale.D,
            'NR': RatingScale.NR
        }
        
        self._fitch_mapping = {
            'AAA': RatingScale.AAA,
            'AA+': RatingScale.AA_PLUS,
            'AA': RatingScale.AA,
            'AA-': RatingScale.AA_MINUS,
            'A+': RatingScale.A_PLUS,
            'A': RatingScale.A,
            'A-': RatingScale.A_MINUS,
            'BBB+': RatingScale.BBB_PLUS,
            'BBB': RatingScale.BBB,
            'BBB-': RatingScale.BBB_MINUS,
            'BB+': RatingScale.BB_PLUS,
            'BB': RatingScale.BB,
            'BB-': RatingScale.BB_MINUS,
            'B+': RatingScale.B_PLUS,
            'B': RatingScale.B,
            'B-': RatingScale.B_MINUS,
            'CCC+': RatingScale.CCC_PLUS,
            'CCC': RatingScale.CCC,
            'CCC-': RatingScale.CCC_MINUS,
            'CC': RatingScale.CC,
            'C': RatingScale.C,
            'D': RatingScale.D,
            'NR': RatingScale.NR
        }
        
        self._agency_mappings = {
            RatingAgency.SP: self._sp_mapping,
            RatingAgency.MOODY: self._moody_mapping,
            RatingAgency.FITCH: self._fitch_mapping
        }
    
    def normalize_rating(self, rating: str, agency: RatingAgency) -> RatingScale:
        """Convert agency rating to normalized scale"""
        if not rating or pd.isna(rating):
            return RatingScale.NR
            
        mapping = self._agency_mappings.get(agency, {})
        return mapping.get(rating, RatingScale.NR)
    
    def get_rating_hierarchy_value(self, rating_scale: RatingScale) -> int:
        """Get numerical value for rating comparison (lower is better)"""
        return rating_scale.value

# Decorator function for easy registration
def register_enrichment(enrichment_class=None, **kwargs):
    """
    Decorator to register enrichment classes with the factory.
    
    Usage:
        @register_enrichment  # Uses defaults
        @register_enrichment(enabled=True, category=EnrichmentCategory.CREDIT)
        @register_enrichment(required_for_reports=[ReportType.CREDIT_RISK])
    """
    return EnrichmentFactory.register(enrichment_class, **kwargs)


class AbstractEnrichment(ABC):
    """Abstract base class for all enrichment operations"""
    
    def __init__(self, name: str):
        self.name = name
        self.enrichment_log = {}
        self._raw_dataframe = None
        self._enriched_dataframe = None
        self._reference_data = {}
        self._last_enrichment_timestamp = None
        
    @property
    def raw_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the raw input dataframe"""
        return self._raw_dataframe
    
    @property
    def enriched_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the enriched dataframe"""
        return self._enriched_dataframe
    
    @property
    def reference_data(self) -> Dict[str, Any]:
        """Get the loaded reference data"""
        return self._reference_data.copy()  # Return copy to prevent external modification
    
    @property
    def has_reference_data(self) -> bool:
        """Check if reference data is loaded"""
        return len(self._reference_data) > 0
    
    @property
    def enrichment_count(self) -> int:
        """Get count of successful enrichments"""
        success_count = 0
        for logs in self.enrichment_log.values():
            success_count += sum(1 for log in logs if log.get('status') == EnrichmentStatus.SUCCESS.value)
        return success_count
    
    @property
    def last_enrichment_timestamp(self) -> Optional[pd.Timestamp]:
        """Get timestamp of last enrichment operation"""
        return self._last_enrichment_timestamp
    
    def clear_data(self):
        """Clear all stored dataframes and reference data"""
        self._raw_dataframe = None
        self._enriched_dataframe = None
        self._reference_data.clear()
        self.enrichment_log.clear()
        self._last_enrichment_timestamp = None
        logger.info(f"Cleared all data for {self.name}")
    
    @abstractmethod
    def load_reference_data(self, keys: List[str]) -> Dict[str, Any]:
        """
        Load reference data from database using the provided keys.
        
        Args:
            keys: List of identifiers (ISIN, Bond ID, etc.)
            
        Returns:
            Dictionary mapping keys to reference data
            
        Raises:
            ReferenceDataError: If reference data cannot be loaded
        """
        pass
    
    @abstractmethod
    def apply_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply enrichment to the dataframe.
        
        Args:
            df: Input dataframe containing trade data
            
        Returns:
            Enriched dataframe with additional columns
            
        Raises:
            EnrichmentProcessingError: If enrichment processing fails
        """
        pass
    
    def _store_dataframes(self, raw_df: pd.DataFrame, enriched_df: pd.DataFrame):
        """Store raw and enriched dataframes"""
        self._raw_dataframe = raw_df.copy()
        self._enriched_dataframe = enriched_df.copy()
        self._last_enrichment_timestamp = pd.Timestamp.now()
    
    def _store_reference_data(self, ref_data: Dict[str, Any]):
        """Store reference data"""
        self._reference_data.update(ref_data)
        logger.info(f"Stored reference data for {len(ref_data)} keys in {self.name}")
        
    def _log_enrichment(self, key: str, status: EnrichmentStatus, message: str = ""):
        """Log enrichment operation per key"""
        if key not in self.enrichment_log:
            self.enrichment_log[key] = []
        
        self.enrichment_log[key].append({
            'enrichment': self.name,
            'status': status.value,
            'message': message,
            'timestamp': pd.Timestamp.now()
        })
        
        logger.info(f"Enrichment {self.name} - {key}: {status.value} - {message}")
        
    def get_enrichment_stats(self) -> Dict[str, int]:
        """Get statistics about enrichment operations"""
        stats = {
            'total_keys': len(self.enrichment_log),
            'successful': 0,
            'failed': 0,
            'partial': 0,
            'no_data': 0
        }
        
        for logs in self.enrichment_log.values():
            for log in logs:
                status = log.get('status')
                if status == EnrichmentStatus.SUCCESS.value:
                    stats['successful'] += 1
                elif status == EnrichmentStatus.FAILED.value:
                    stats['failed'] += 1
                elif status == EnrichmentStatus.PARTIAL.value:
                    stats['partial'] += 1
                elif status == EnrichmentStatus.NO_DATA.value:
                    stats['no_data'] += 1
        
        return stats



# =============================================================================
# ENRICHMENT FACTORY
# =============================================================================

class EnrichmentFactory:
    """Factory for auto-registering and creating enrichment classes with report-based filtering"""
    
    _enrichments = {}
    _enrichment_metadata = {}
    
    @classmethod
    def register(cls, 
                 enrichment_class=None,
                 *,
                 enabled: bool = True,
                 category: EnrichmentCategory = EnrichmentCategory.CLASSIFICATION,
                 required_for_reports: List[ReportType] = None,
                 description: str = "",
                 dependencies: List[str] = None):
        """
        Enhanced decorator to register enrichment classes with metadata.
        
        Args:
            enrichment_class: The enrichment class to register
            enabled: Whether this enrichment is complete and enabled
            category: Category of the enrichment
            required_for_reports: List of reports that require this enrichment
            description: Description of what this enrichment does
            dependencies: List of other enrichment names this depends on
        """
        def decorator(cls_to_register):
            if not issubclass(cls_to_register, AbstractEnrichment):
                raise EnrichmentConfigurationError(
                    f"Class {cls_to_register.__name__} must inherit from AbstractEnrichment"
                )
            
            # Set defaults
            if required_for_reports is None:
                reports = [ReportType.DATA_EXTRACT]  # Default to data extract only
            else:
                reports = required_for_reports
            
            if dependencies is None:
                deps = []
            else:
                deps = dependencies
            
            # Store class and metadata
            cls._enrichments[cls_to_register.__name__] = cls_to_register
            cls._enrichment_metadata[cls_to_register.__name__] = EnrichmentMetadata(
                name=cls_to_register.__name__,
                category=category,
                enabled=enabled,
                required_for_reports=reports,
                description=description or f"Enrichment for {cls_to_register.__name__}",
                dependencies=deps
            )
            
            status = "ENABLED" if enabled else "DISABLED"
            logger.info(f"Registered enrichment class: {cls_to_register.__name__} [{status}] "
                       f"- Category: {category.value}, Reports: {[r.value for r in reports]}")
            
            return cls_to_register
        
        # Support both @register_enrichment and @register_enrichment(params)
        if enrichment_class is None:
            return decorator
        else:
            return decorator(enrichment_class)
    
    @classmethod
    def create_enrichment(cls, name: str) -> AbstractEnrichment:
        """Create enrichment instance by name (only if enabled)"""
        if name not in cls._enrichments:
            available = ', '.join(cls.get_available_enrichments())
            raise EnrichmentConfigurationError(
                f"Unknown enrichment: {name}. Available: {available}"
            )
        
        metadata = cls._enrichment_metadata[name]
        if not metadata.enabled:
            raise EnrichmentConfigurationError(
                f"Enrichment {name} is disabled and cannot be created"
            )
        
        return cls._enrichments[name]()
    
    @classmethod
    def get_available_enrichments(cls, enabled_only: bool = True) -> List[str]:
        """Get list of available enrichment names"""
        if enabled_only:
            return [name for name, metadata in cls._enrichment_metadata.items() 
                   if metadata.enabled]
        else:
            return list(cls._enrichments.keys())
    
    @classmethod
    def get_enrichments_for_report(cls, report_type: ReportType, 
                                  enabled_only: bool = True) -> List[str]:
        """Get enrichments required for a specific report type"""
        matching_enrichments = []
        
        for name, metadata in cls._enrichment_metadata.items():
            if enabled_only and not metadata.enabled:
                continue
                
            if report_type in metadata.required_for_reports:
                matching_enrichments.append(name)
        
        return matching_enrichments
    
    @classmethod
    def get_enrichments_by_category(cls, category: EnrichmentCategory,
                                   enabled_only: bool = True) -> List[str]:
        """Get enrichments by category"""
        matching_enrichments = []
        
        for name, metadata in cls._enrichment_metadata.items():
            if enabled_only and not metadata.enabled:
                continue
                
            if metadata.category == category:
                matching_enrichments.append(name)
        
        return matching_enrichments
    
    @classmethod
    def create_enrichments_for_report(cls, report_type: ReportType) -> List[AbstractEnrichment]:
        """Create all enrichment instances required for a specific report"""
        enrichment_names = cls.get_enrichments_for_report(report_type)
        
        # Resolve dependencies
        resolved_names = cls._resolve_dependencies(enrichment_names)
        
        return [cls.create_enrichment(name) for name in resolved_names]
    
    @classmethod
    def create_all_enrichments(cls, enabled_only: bool = True) -> List[AbstractEnrichment]:
        """Create instances of all registered enrichments"""
        enrichment_names = cls.get_available_enrichments(enabled_only)
        resolved_names = cls._resolve_dependencies(enrichment_names)
        return [cls.create_enrichment(name) for name in resolved_names]
    
    @classmethod
    def get_enrichment_metadata(cls, name: str) -> Optional[EnrichmentMetadata]:
        """Get metadata for a specific enrichment"""
        return cls._enrichment_metadata.get(name)
    
    @classmethod
    def get_all_metadata(cls) -> Dict[str, EnrichmentMetadata]:
        """Get metadata for all enrichments"""
        return cls._enrichment_metadata.copy()
    
    @classmethod
    def get_report_summary(cls) -> pd.DataFrame:
        """Get summary of which enrichments are required for each report type"""
        summary_data = []
        
        for report_type in ReportType:
            enrichments = cls.get_enrichments_for_report(report_type)
            for enrichment_name in enrichments:
                metadata = cls._enrichment_metadata[enrichment_name]
                summary_data.append({
                    'Report_Type': report_type.value,
                    'Enrichment': enrichment_name,
                    'Category': metadata.category.value,
                    'Enabled': metadata.enabled,
                    'Description': metadata.description
                })
        
        return pd.DataFrame(summary_data)
    
    @classmethod
    def _resolve_dependencies(cls, enrichment_names: List[str]) -> List[str]:
        """Resolve dependencies and return enrichments in correct order"""
        resolved = []
        remaining = enrichment_names.copy()
        
        # Simple dependency resolution (topological sort would be better for complex deps)
        max_iterations = len(enrichment_names) * 2  # Prevent infinite loops
        iteration = 0
        
        while remaining and iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for name in remaining.copy():
                metadata = cls._enrichment_metadata.get(name)
                if not metadata:
                    remaining.remove(name)
                    continue
                
                # Check if all dependencies are resolved
                deps_satisfied = True
                for dep in metadata.dependencies:
                    if dep not in resolved and dep in remaining:
                        deps_satisfied = False
                        break
                
                if deps_satisfied:
                    resolved.append(name)
                    remaining.remove(name)
                    made_progress = True
            
            if not made_progress:
                # Add remaining items (circular dependencies or missing deps)
                logger.warning(f"Could not resolve dependencies for: {remaining}")
                resolved.extend(remaining)
                break
        
        return resolved
    
    @classmethod
    def validate_dependencies(cls) -> Dict[str, List[str]]:
        """Validate all dependencies and return any issues"""
        issues = {}
        
        for name, metadata in cls._enrichment_metadata.items():
            enrichment_issues = []
            
            for dep in metadata.dependencies:
                if dep not in cls._enrichments:
                    enrichment_issues.append(f"Missing dependency: {dep}")
                elif not cls._enrichment_metadata[dep].enabled:
                    enrichment_issues.append(f"Dependency {dep} is disabled")
            
            if enrichment_issues:
                issues[name] = enrichment_issues
        
        return issues









# =============================================================================
# BOND RATING ENRICHMENT
# =============================================================================

@register_enrichment(
    enabled=True,
    category=EnrichmentCategory.CREDIT,
    required_for_reports=[
        ReportType.DATA_EXTRACT,
        ReportType.CREDIT_RISK,
        ReportType.REGULATORY,
        ReportType.PORTFOLIO_ANALYTICS
    ],
    description="Provides comprehensive credit ratings from S&P, Moody's, and Fitch with statistical analysis"
)
class BondRatingEnrichment(AbstractEnrichment):
    """Enrichment class for bond credit ratings"""
    
    def __init__(self):
        super().__init__("BondRatingEnrichment")
        self.rating_mapper = RatingMapper()
    
    def load_reference_data(self, keys: List[str]) -> Dict[str, Any]:
        """
        Load bond rating data from RefData database.
        
        Args:
            keys: List of ISINs or Bond IDs
            
        Returns:
            Dictionary with rating data per key
            
        Raises:
            ReferenceDataError: If rating data cannot be loaded
        """
        try:
            # Placeholder for actual database call
            # In real implementation, this would call your RefData wrapper functions
            logger.info(f"Loading rating data for {len(keys)} instruments")
            
            # Mock data structure that would come from your database
            rating_data = {}
            for key in keys:
                # This would be replaced with actual database call
                rating_data[key] = {
                    RatingAgency.SP.value: 'A+',
                    RatingAgency.MOODY.value: 'A1',
                    RatingAgency.FITCH.value: 'A+'
                }
            
            # Store the loaded reference data
            self._store_reference_data(rating_data)
            return rating_data
            
        except Exception as e:
            raise ReferenceDataError(f"Failed to load rating data: {str(e)}")
    
    def get_ratings_for_isins(self, isins: List[str], 
                             return_format: str = 'dataframe') -> Union[pd.DataFrame, Dict]:
        """
        Get comprehensive rating information for list of ISINs.
        
        Args:
            isins: List of ISIN identifiers
            return_format: 'dataframe' or 'dictionary'
            
        Returns:
            Rating information in requested format
            
        Raises:
            EnrichmentDataError: If ISIN list is invalid
            RatingCalculationError: If rating calculations fail
        """
        if not isins:
            raise EnrichmentDataError("ISIN list cannot be empty")
        
        try:
            rating_data = self.load_reference_data(isins)
            results = {}
            
            for isin in isins:
                if isin not in rating_data:
                    self._log_enrichment(isin, EnrichmentStatus.NO_DATA, "No rating data found")
                    continue
                    
                ratings = rating_data[isin]
                rating_summary = self._calculate_rating_summary(ratings)
                
                results[isin] = {
                    'isin': isin,
                    'sp_rating': ratings.get(RatingAgency.SP.value),
                    'moody_rating': ratings.get(RatingAgency.MOODY.value),
                    'fitch_rating': ratings.get(RatingAgency.FITCH.value),
                    'average_rating': rating_summary.average_rating.name if rating_summary.average_rating else None,
                    'best_rating': rating_summary.best_rating.name if rating_summary.best_rating else None,
                    'worst_rating': rating_summary.worst_rating.name if rating_summary.worst_rating else None,
                    'best_agency': rating_summary.best_agency.value if rating_summary.best_agency else None,
                    'worst_agency': rating_summary.worst_agency.value if rating_summary.worst_agency else None,
                    'rating_count': rating_summary.rating_count
                }
                
                self._log_enrichment(isin, EnrichmentStatus.SUCCESS, f"Processed {rating_summary.rating_count} ratings")
            
            if return_format.lower() == 'dataframe':
                return pd.DataFrame.from_dict(results, orient='index').reset_index(drop=True)
            else:
                return results
                
        except Exception as e:
            if isinstance(e, (EnrichmentDataError, ReferenceDataError)):
                raise
            raise RatingCalculationError(f"Failed to calculate ratings: {str(e)}")
    
    def _calculate_rating_summary(self, ratings: Dict[str, str]) -> RatingSummary:
        """Calculate rating statistics from agency ratings"""
        try:
            valid_ratings = []
            
            for agency_name, rating_str in ratings.items():
                if not rating_str or pd.isna(rating_str) or rating_str == 'NR':
                    continue
                    
                # Find matching agency enum
                agency = None
                for ag in RatingAgency:
                    if ag.value == agency_name:
                        agency = ag
                        break
                
                if agency:
                    normalized_rating = self.rating_mapper.normalize_rating(rating_str, agency)
                    if normalized_rating != RatingScale.NR:
                        valid_ratings.append(RatingInfo(
                            rating=normalized_rating,
                            agency=agency,
                            scale_value=self.rating_mapper.get_rating_hierarchy_value(normalized_rating)
                        ))
            
            if not valid_ratings:
                return RatingSummary(None, None, None, None, None, 0)
            
            # Calculate statistics
            scale_values = [r.scale_value for r in valid_ratings]
            avg_value = np.mean(scale_values)
            
            # Find closest rating to average
            avg_rating = min(valid_ratings, key=lambda x: abs(x.scale_value - avg_value)).rating
            
            # Best rating (lowest numerical value)
            best_rating_info = min(valid_ratings, key=lambda x: x.scale_value)
            
            # Worst rating (highest numerical value)
            worst_rating_info = max(valid_ratings, key=lambda x: x.scale_value)
            
            return RatingSummary(
                average_rating=avg_rating,
                best_rating=best_rating_info.rating,
                worst_rating=worst_rating_info.rating,
                best_agency=best_rating_info.agency,
                worst_agency=worst_rating_info.agency,
                rating_count=len(valid_ratings)
            )
            
        except Exception as e:
            raise RatingCalculationError(f"Failed to calculate rating summary: {str(e)}")
    
    def apply_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rating enrichment to trade dataframe"""
        try:
            if df is None or df.empty:
                raise EnrichmentDataError("Input dataframe is empty or None")
                
            if 'ISIN' not in df.columns:
                raise EnrichmentDataError("ISIN column not found in dataframe")
            
            # Store raw dataframe
            raw_df = df.copy()
            
            isins = df['ISIN'].dropna().unique().tolist()
            if not isins:
                logger.warning("No valid ISINs found in dataframe")
                self._store_dataframes(raw_df, df)
                return df
            
            rating_df = self.get_ratings_for_isins(isins, return_format='dataframe')
            
            # Merge with original dataframe
            enriched_df = df.merge(rating_df, left_on='ISIN', right_on='isin', how='left')
            
            # Store both dataframes
            self._store_dataframes(raw_df, enriched_df)
            
            return enriched_df
            
        except Exception as e:
            if isinstance(e, (EnrichmentDataError, RatingCalculationError, ReferenceDataError)):
                raise
            raise EnrichmentProcessingError(f"Failed to apply rating enrichment: {str(e)}")

# =============================================================================
# OTHER ENRICHMENT CLASSES (STUBS)
# =============================================================================

@register_enrichment(
    enabled=True,
    category=EnrichmentCategory.CLASSIFICATION,
    required_for_reports=[
        ReportType.DATA_EXTRACT,
        ReportType.REGULATORY,
        ReportType.PORTFOLIO_ANALYTICS,
        ReportType.COMPLIANCE
    ],
    description="Enriches trades with asset classification data including sector, industry, and security type"
)
class AssetClassificationEnrichment(AbstractEnrichment):
    """Enrichment for asset classification data"""
    
    def __init__(self):
        super().__init__("AssetClassificationEnrichment")
    
    def load_reference_data(self, keys: List[str]) -> Dict[str, Any]:
        try:
            logger.info(f"Loading asset classification data for {len(keys)} instruments")
            # Implementation would go here
            classification_data = {}
            self._store_reference_data(classification_data)
            return classification_data
        except Exception as e:
            raise ReferenceDataError(f"Failed to load asset classification data: {str(e)}")
    
    def apply_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df is None or df.empty:
                raise EnrichmentDataError("Input dataframe is empty or None")
            
            raw_df = df.copy()
            # Implementation would go here
            enriched_df = df.copy()
            
            self._store_dataframes(raw_df, enriched_df)
            return enriched_df
        except Exception as e:
            if isinstance(e, EnrichmentDataError):
                raise
            raise EnrichmentProcessingError(f"Failed to apply asset classification enrichment: {str(e)}")


@register_enrichment(
    enabled=True,
    category=EnrichmentCategory.COUNTERPARTY,
    required_for_reports=[
        ReportType.DATA_EXTRACT,
        ReportType.CREDIT_RISK,
        ReportType.COMPLIANCE,
        ReportType.REGULATORY
    ],
    description="Enriches trades with counterparty information including ratings, jurisdiction, and risk metrics",
    dependencies=["BondRatingEnrichment"]  # Depends on ratings for counterparty analysis
)
class CounterpartyEnrichment(AbstractEnrichment):
    """Enrichment for counterparty data"""
    
    def __init__(self):
        super().__init__("CounterpartyEnrichment")
    
    def load_reference_data(self, keys: List[str]) -> Dict[str, Any]:
        try:
            logger.info(f"Loading counterparty data for {len(keys)} instruments")
            # Implementation would go here
            counterparty_data = {}
            self._store_reference_data(counterparty_data)
            return counterparty_data
        except Exception as e:
            raise ReferenceDataError(f"Failed to load counterparty data: {str(e)}")
    
    def apply_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df is None or df.empty:
                raise EnrichmentDataError("Input dataframe is empty or None")
            
            raw_df = df.copy()
            # Implementation would go here
            enriched_df = df.copy()
            
            self._store_dataframes(raw_df, enriched_df)
            return enriched_df
        except Exception as e:
            if isinstance(e, EnrichmentDataError):
                raise
            raise EnrichmentProcessingError(f"Failed to apply counterparty enrichment: {str(e)}")


@register_enrichment(
    enabled=True,
    category=EnrichmentCategory.LIQUIDITY,
    required_for_reports=[
        ReportType.DATA_EXTRACT,
        ReportType.LIQUIDITY_RISK,
        ReportType.MARKET_RISK,
        ReportType.PORTFOLIO_ANALYTICS
    ],
    description="Provides bond liquidity metrics and regulatory haircuts for risk calculations"
)
class BondLiquidityAndHaircuts(AbstractEnrichment):
    """Enrichment for bond liquidity and haircut data"""
    
    def __init__(self):
        super().__init__("BondLiquidityAndHaircuts")
    
    def load_reference_data(self, keys: List[str]) -> Dict[str, Any]:
        try:
            logger.info(f"Loading liquidity and haircut data for {len(keys)} instruments")
            # Implementation would go here
            liquidity_data = {}
            self._store_reference_data(liquidity_data)
            return liquidity_data
        except Exception as e:
            raise ReferenceDataError(f"Failed to load liquidity and haircut data: {str(e)}")
    
    def apply_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df is None or df.empty:
                raise EnrichmentDataError("Input dataframe is empty or None")
            
            raw_df = df.copy()
            # Implementation would go here
            enriched_df = df.copy()
            
            self._store_dataframes(raw_df, enriched_df)
            return enriched_df
        except Exception as e:
            if isinstance(e, EnrichmentDataError):
                raise
            raise EnrichmentProcessingError(f"Failed to apply liquidity and haircuts enrichment: {str(e)}")


# Example of a disabled enrichment (work in progress)
@register_enrichment(
    enabled=False,  # Not ready for production
    category=EnrichmentCategory.PRICING,
    required_for_reports=[
        ReportType.DATA_EXTRACT,
        ReportType.MARKET_RISK,
        ReportType.TRADING_ANALYTICS
    ],
    description="Enriches trades with real-time pricing and mark-to-market valuations (UNDER DEVELOPMENT)"
)
class RealTimePricingEnrichment(AbstractEnrichment):
    """Enrichment for real-time pricing data (disabled - under development)"""
    
    def __init__(self):
        super().__init__("RealTimePricingEnrichment")
    
    def load_reference_data(self, keys: List[str]) -> Dict[str, Any]:
        # This would be implemented when enabled
        raise NotImplementedError("RealTimePricingEnrichment is under development")
    
    def apply_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        # This would be implemented when enabled
        raise NotImplementedError("RealTimePricingEnrichment is under development")
        return df






class EnrichmentApplier:
    """Main class to apply enrichments to trade data with report-based filtering"""
    
    def __init__(self, 
                 enrichment_names: Optional[List[str]] = None,
                 report_type: Optional[ReportType] = None):
        """
        Initialize with specific enrichments, report type, or all available ones.
        
        Args:
            enrichment_names: List of specific enrichment names to use
            report_type: Type of report to generate enrichments for
            
        Note: If both enrichment_names and report_type are provided, enrichment_names takes precedence
        """
        try:
            if enrichment_names is not None:
                # Use specific enrichments
                self.enrichments = [
                    EnrichmentFactory.create_enrichment(name) 
                    for name in enrichment_names
                ]
                self.report_type = None
                logger.info(f"Initialized with specific enrichments: {enrichment_names}")
                
            elif report_type is not None:
                # Use enrichments for specific report type
                self.enrichments = EnrichmentFactory.create_enrichments_for_report(report_type)
                self.report_type = report_type
                enrichment_names_for_report = [e.name for e in self.enrichments]
                logger.info(f"Initialized for {report_type.value} report with enrichments: {enrichment_names_for_report}")
                
            else:
                # Use all available enrichments
                self.enrichments = EnrichmentFactory.create_all_enrichments()
                self.report_type = None
                logger.info("Initialized with all available enrichments")
            
            self.enrichment_log = {}
            self._raw_dataframe = None
            self._enriched_dataframe = None
            self._enrichment_history = []
            
        except Exception as e:
            raise EnrichmentConfigurationError(f"Failed to initialize EnrichmentApplier: {str(e)}")
    
    @classmethod
    def for_report(cls, report_type: ReportType) -> 'EnrichmentApplier':
        """Create an EnrichmentApplier configured for a specific report type"""
        return cls(report_type=report_type)
    
    @classmethod 
    def for_enrichments(cls, enrichment_names: List[str]) -> 'EnrichmentApplier':
        """Create an EnrichmentApplier with specific enrichments"""
        return cls(enrichment_names=enrichment_names)
    
    @property
    def raw_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the original raw dataframe"""
        return self._raw_dataframe
    
    @property
    def enriched_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the final enriched dataframe"""
        return self._enriched_dataframe
    
    @property
    def enrichment_history(self) -> List[Dict[str, Any]]:
        """Get history of enrichment operations"""
        return self._enrichment_history.copy()
    
    @property
    def enrichment_names(self) -> List[str]:
        """Get list of enrichment names being applied"""
        return [enrichment.name for enrichment in self.enrichments]
    
    @property
    def total_enrichments_applied(self) -> int:
        """Get total count of successful enrichments across all classes"""
        return sum(enrichment.enrichment_count for enrichment in self.enrichments)
    
    def get_enrichment_by_name(self, name: str) -> Optional[AbstractEnrichment]:
        """Get enrichment instance by name"""
        for enrichment in self.enrichments:
            if enrichment.name == name:
                return enrichment
        return None
    
    def get_report_requirements(self) -> Optional[pd.DataFrame]:
        """Get summary of enrichments required for the current report type"""
        if self.report_type is None:
            return None
            
        summary_data = []
        for enrichment in self.enrichments:
            metadata = EnrichmentFactory.get_enrichment_metadata(enrichment.name)
            if metadata:
                summary_data.append({
                    'Enrichment': enrichment.name,
                    'Category': metadata.category.value,
                    'Description': metadata.description,
                    'Dependencies': ', '.join(metadata.dependencies) if metadata.dependencies else 'None'
                })
        
        return pd.DataFrame(summary_data)
    
    def clear_all_data(self):
        """Clear all stored data from applier and enrichments"""
        self._raw_dataframe = None
        self._enriched_dataframe = None
        self.enrichment_log.clear()
        self._enrichment_history.clear()
        
        for enrichment in self.enrichments:
            enrichment.clear_data()
        
        logger.info("Cleared all data from EnrichmentApplier and all enrichments")
        
        return [enrichment.name for enrichment in self.enrichments]
    
    @property
    def total_enrichments_applied(self) -> int:
        """Get total count of successful enrichments across all classes"""
        return sum(enrichment.enrichment_count for enrichment in self.enrichments)
    
    def get_enrichment_by_name(self, name: str) -> Optional[AbstractEnrichment]:
        """Get enrichment instance by name"""
        for enrichment in self.enrichments:
            if enrichment.name == name:
                return enrichment
        return None
    
    def clear_all_data(self):
        """Clear all stored data from applier and enrichments"""
        self._raw_dataframe = None
        self._enriched_dataframe = None
        self.enrichment_log.clear()
        self._enrichment_history.clear()
        
        for enrichment in self.enrichments:
            enrichment.clear_data()
        
        logger.info("Cleared all data from EnrichmentApplier and all enrichments")
    
    def apply_all_enrichments(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Apply all enrichments to the input dataframe.
        
        Args:
            df: Input trade dataframe with Bond ID and ISIN columns
            
        Returns:
            Dictionary with 'raw' and 'enriched' dataframes
            
        Raises:
            EnrichmentDataError: If input dataframe is invalid
            EnrichmentProcessingError: If enrichment processing fails
        """
        try:
            if df is None or df.empty:
                raise EnrichmentDataError("Input dataframe is empty or None")
            
            logger.info("Starting enrichment process")
            
            # Store raw data
            self._raw_dataframe = df.copy()
            current_df = df.copy()
            
            # Start with raw data
            result = {
                'raw': df.copy(),
                'enriched': df.copy()
            }
            
            successful_enrichments = []
            failed_enrichments = []
            
            # Apply each enrichment
            for enrichment in self.enrichments:
                logger.info(f"Applying {enrichment.name}")
                
                try:
                    enrichment_start_time = pd.Timestamp.now()
                    result['enriched'] = enrichment.apply_enrichment(result['enriched'])
                    enrichment_end_time = pd.Timestamp.now()
                    
                    # Record successful enrichment
                    self._enrichment_history.append({
                        'enrichment_name': enrichment.name,
                        'status': 'SUCCESS',
                        'start_time': enrichment_start_time,
                        'end_time': enrichment_end_time,
                        'duration': enrichment_end_time - enrichment_start_time,
                        'rows_processed': len(result['enriched']),
                        'enrichment_count': enrichment.enrichment_count
                    })
                    
                    successful_enrichments.append(enrichment.name)
                    
                    # Consolidate logs
                    for key, logs in enrichment.enrichment_log.items():
                        if key not in self.enrichment_log:
                            self.enrichment_log[key] = []
                        self.enrichment_log[key].extend(logs)
                        
                    logger.info(f"Successfully applied {enrichment.name}")
                    
                except Exception as e:
                    error_msg = f"Failed to apply {enrichment.name}: {str(e)}"
                    logger.error(error_msg)
                    
                    # Record failed enrichment
                    self._enrichment_history.append({
                        'enrichment_name': enrichment.name,
                        'status': 'FAILED',
                        'error_message': str(e),
                        'start_time': pd.Timestamp.now(),
                        'end_time': pd.Timestamp.now()
                    })
                    
                    failed_enrichments.append(enrichment.name)
                    continue
            
            # Store final enriched dataframe
            self._enriched_dataframe = result['enriched'].copy()
            
            logger.info(f"Enrichment process completed. "
                       f"Successful: {len(successful_enrichments)}, "
                       f"Failed: {len(failed_enrichments)}")
            
            if failed_enrichments:
                logger.warning(f"Failed enrichments: {', '.join(failed_enrichments)}")
            
            return result
            
        except Exception as e:
            if isinstance(e, (EnrichmentDataError, EnrichmentProcessingError)):
                raise
            raise EnrichmentProcessingError(f"Failed to apply enrichments: {str(e)}")
    
    def get_enrichment_summary(self) -> pd.DataFrame:
        """Get summary of all enrichments applied per ISIN"""
        summary_data = []
        
        for isin, logs in self.enrichment_log.items():
            for log_entry in logs:
                summary_data.append({
                    'ISIN': isin,
                    'Enrichment': log_entry['enrichment'],
                    'Status': log_entry['status'],
                    'Message': log_entry['message'],
                    'Timestamp': log_entry['timestamp']
                })
        
        return pd.DataFrame(summary_data)
    
    def get_enrichment_performance_summary(self) -> pd.DataFrame:
        """Get performance summary of enrichment operations"""
        return pd.DataFrame(self._enrichment_history)
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall statistics about the enrichment process"""
        if not self._enrichment_history:
            return {}
        
        successful = [h for h in self._enrichment_history if h['status'] == 'SUCCESS']
        failed = [h for h in self._enrichment_history if h['status'] == 'FAILED']
        
        total_duration = sum([h.get('duration', pd.Timedelta(0)) for h in successful], pd.Timedelta(0))
        
        return {
            'total_enrichments_attempted': len(self._enrichment_history),
            'successful_enrichments': len(successful),
            'failed_enrichments': len(failed),
            'success_rate': len(successful) / len(self._enrichment_history) if self._enrichment_history else 0,
            'total_processing_time': total_duration,
            'average_processing_time': total_duration / len(successful) if successful else pd.Timedelta(0),
            'total_records_processed': len(self._enriched_dataframe) if self._enriched_dataframe is not None else 0,
            'total_enrichment_operations': self.total_enrichments_applied
        }

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
def example_usage():
    """Example of how to use the enhanced enrichment framework with reports"""
    
    try:
        # Sample trade data
        trade_data = pd.DataFrame({
            'TradeID': ['T001', 'T002', 'T003'],
            'ISIN': ['US123456789', 'GB987654321', 'DE555666777'],
            'BondID': ['B001', 'B002', 'B003'],
            'Quantity': [1000000, 2000000, 1500000],
            'Price': [99.5, 101.2, 98.8]
        })
        
        print("=== Enhanced Trade Data Enrichment Framework with Reports ===\n")
        
        # Example 1: Using EnrichmentApplier directly
        print("1. Direct EnrichmentApplier Usage:")
        applier = EnrichmentApplier.for_report(ReportType.CREDIT_RISK)
        print(f"Credit Risk enrichments: {applier.enrichment_names}")
        
        applier_liquidity = EnrichmentApplier.for_report(ReportType.LIQUIDITY_RISK)
        print(f"Liquidity Risk enrichments: {applier_liquidity.enrichment_names}")
        
        # Example 2: Using BaseReport integration
        print("\n2. BaseReport Integration:")
        
        # Credit Risk Report
        credit_report = CreditRiskReport()
        print(f"Required enrichments for {credit_report.report_name}: {credit_report.get_required_enrichments()}")
        
        credit_result = credit_report.generate_full_report(trade_data)
        print(f"Credit Risk Report Result: {credit_result}")
        
        # Data Extract Report (gets all enrichments)
        extract_report = DataExtractReport()
        print(f"\nRequired enrichments for {extract_report.report_name}: {extract_report.get_required_enrichments()}")
        
        extract_result = extract_report.generate_full_report(trade_data)
        print(f"Data Extract shape: {extract_result.shape}")
        
        # Liquidity Risk Report (only gets liquidity-related enrichments)
        liquidity_report = LiquidityRiskReport()
        print(f"\nRequired enrichments for {liquidity_report.report_name}: {liquidity_report.get_required_enrichments()}")
        
        liquidity_result = liquidity_report.generate_full_report(trade_data)
        print(f"Liquidity Risk Report Result: {liquidity_result}")
        
        # Example 3: Custom enrichments for a report
        print("\n3. Custom Enrichments Override:")
        custom_enrichments = ['BondRatingEnrichment', 'AssetClassificationEnrichment']
        custom_result = credit_report.generate_full_report(trade_data, custom_enrichments=custom_enrichments)
        print(f"Custom Credit Risk Report Result: {custom_result}")
        
        # Example 4: Enrichment metadata and report mapping
        print("\n4. Enrichment Metadata:")
        print("Report Summary:")
        report_summary = EnrichmentFactory.get_report_summary()
        print(report_summary)
        
        print("\nAll Enrichment Metadata:")
        all_metadata = EnrichmentFactory.get_all_metadata()
        for name, metadata in all_metadata.items():
            print(f"  {name}: {metadata.description} [{'ENABLED' if metadata.enabled else 'DISABLED'}]")
        
        # Example 5: Dependency validation
        print("\n5. Dependency Validation:")
        dependency_issues = EnrichmentFactory.validate_dependencies()
        if dependency_issues:
            print("Dependency Issues Found:")
            for enrichment, issues in dependency_issues.items():
                print(f"  {enrichment}: {issues}")
        else:
            print("All dependencies are valid")
            
    except Exception as e:
        print(f"Example failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# BASE REPORT CLASS INTEGRATION
# =============================================================================

class BaseReport(ABC):
    """
    Base report class that integrates with the enrichment framework.
    Inherit from this class for all your reports.
    """
    
    def __init__(self, report_name: str, report_type: ReportType):
        self.report_name = report_name
        self.report_type = report_type
        self._enrichment_applier = None
        self._raw_data = None
        self._enriched_data = None
        self._generation_timestamp = None
        
    @property
    def enrichment_applier(self) -> Optional[EnrichmentApplier]:
        """Get the enrichment applier instance"""
        return self._enrichment_applier
    
    @property
    def raw_data(self) -> Optional[pd.DataFrame]:
        """Get the raw data used for report generation"""
        return self._raw_data
    
    @property
    def enriched_data(self) -> Optional[pd.DataFrame]:
        """Get the enriched data used for report generation"""
        return self._enriched_data
    
    @property
    def generation_timestamp(self) -> Optional[pd.Timestamp]:
        """Get the timestamp when the report was generated"""
        return self._generation_timestamp
    
    def get_required_enrichments(self) -> List[str]:
        """Get list of enrichments required for this report type"""
        return EnrichmentFactory.get_enrichments_for_report(self.report_type)
    
    def prepare_data(self, trade_data: pd.DataFrame, 
                    custom_enrichments: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for report generation by applying required enrichments.
        
        Args:
            trade_data: Raw trade data DataFrame
            custom_enrichments: Optional list to override default enrichments for this report
            
        Returns:
            Dictionary with 'raw' and 'enriched' DataFrames
            
        Raises:
            EnrichmentDataError: If trade data is invalid
            EnrichmentProcessingError: If enrichment fails
        """
        logger.info(f"Preparing data for {self.report_name} ({self.report_type.value}) report")
        
        # Create enrichment applier based on report type or custom enrichments
        if custom_enrichments:
            self._enrichment_applier = EnrichmentApplier.for_enrichments(custom_enrichments)
        else:
            self._enrichment_applier = EnrichmentApplier.for_report(self.report_type)
        
        # Apply enrichments
        result = self._enrichment_applier.apply_all_enrichments(trade_data)
        
        # Store data
        self._raw_data = result['raw']
        self._enriched_data = result['enriched']
        self._generation_timestamp = pd.Timestamp.now()
        
        logger.info(f"Data preparation completed for {self.report_name}")
        return result
    
    def get_enrichment_summary(self) -> Optional[pd.DataFrame]:
        """Get summary of enrichments applied during data preparation"""
        if self._enrichment_applier is None:
            return None
        return self._enrichment_applier.get_enrichment_summary()
    
    def get_enrichment_performance(self) -> Optional[pd.DataFrame]:
        """Get performance metrics of enrichment operations"""
        if self._enrichment_applier is None:
            return None
        return self._enrichment_applier.get_enrichment_performance_summary()
    
    @abstractmethod
    def generate_report(self) -> Any:
        """
        Generate the actual report content.
        Must be implemented by subclasses.
        
        Returns:
            Report content (could be DataFrame, dict, file path, etc.)
        """
        pass
    
    def generate_full_report(self, trade_data: pd.DataFrame, 
                           custom_enrichments: Optional[List[str]] = None) -> Any:
        """
        Complete report generation pipeline: prepare data + generate report.
        
        Args:
            trade_data: Raw trade data DataFrame
            custom_enrichments: Optional custom enrichments list
            
        Returns:
            Generated report content
        """
        # Prepare data with enrichments
        self.prepare_data(trade_data, custom_enrichments)
        
        # Generate the actual report
        return self.generate_report()

# =============================================================================
# EXAMPLE REPORT IMPLEMENTATIONS
# =============================================================================

class CreditRiskReport(BaseReport):
    """Example credit risk report implementation"""
    
    def __init__(self):
        super().__init__("Credit Risk Analysis", ReportType.CREDIT_RISK)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate credit risk report content"""
        if self._enriched_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Example report generation logic
        report_content = {
            'report_name': self.report_name,
            'generation_time': self._generation_timestamp,
            'total_trades': len(self._enriched_data),
            'enrichments_applied': self._enrichment_applier.enrichment_names if self._enrichment_applier else [],
            'credit_summary': {
                # Example: analyze credit ratings if available
                'avg_rating_available': 'average_rating' in self._enriched_data.columns,
                'total_instruments': self._enriched_data['ISIN'].nunique() if 'ISIN' in self._enriched_data.columns else 0
            }
        }
        
        logger.info(f"Generated {self.report_name} with {report_content['total_trades']} trades")
        return report_content


class DataExtractReport(BaseReport):
    """Example data extract report implementation"""
    
    def __init__(self):
        super().__init__("Complete Data Extract", ReportType.DATA_EXTRACT)
    
    def generate_report(self) -> pd.DataFrame:
        """Generate data extract report (returns enriched DataFrame)"""
        if self._enriched_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        logger.info(f"Generated {self.report_name} with {len(self._enriched_data)} rows")
        return self._enriched_data.copy()


class LiquidityRiskReport(BaseReport):
    """Example liquidity risk report implementation"""
    
    def __init__(self):
        super().__init__("Liquidity Risk Analysis", ReportType.LIQUIDITY_RISK)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate liquidity risk report content"""
        if self._enriched_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # This report would only get liquidity-related enrichments automatically
        report_content = {
            'report_name': self.report_name,
            'generation_time': self._generation_timestamp,
            'total_positions': len(self._enriched_data),
            'enrichments_applied': self._enrichment_applier.enrichment_names if self._enrichment_applier else [],
            'liquidity_summary': {
                'instruments_analyzed': self._enriched_data['ISIN'].nunique() if 'ISIN' in self._enriched_data.columns else 0
            }
        }
        
        logger.info(f"Generated {self.report_name} with {report_content['total_positions']} positions")
        return report_content

# =============================================================================
# USAGE PATTERNS FOR YOUR BASE REPORT CLASS
# =============================================================================

def integration_examples():
    """
    Examples showing how to integrate with your existing base report class.
    
    If you already have a BaseReport class, you can:
    1. Add the enrichment methods to your existing class
    2. Inherit from both your BaseReport and add enrichment functionality
    3. Use composition to include an EnrichmentApplier
    """
    
    print("=== Integration Patterns ===\n")
    
    # Pattern 1: Composition approach (if you can't modify your existing BaseReport)
    class YourExistingReport:
        def __init__(self, name: str):
            self.name = name
            self.enrichment_applier = None
        
        def add_enrichment_support(self, report_type: ReportType):
            """Add enrichment support to existing report"""
            self.enrichment_applier = EnrichmentApplier.for_report(report_type)
        
        def enrich_data(self, trade_data: pd.DataFrame) -> pd.DataFrame:
            """Apply enrichments to trade data"""
            if self.enrichment_applier:
                result = self.enrichment_applier.apply_all_enrichments(trade_data)
                return result['enriched']
            return trade_data
    
    # Usage example
    existing_report = YourExistingReport("My Custom Report")
    existing_report.add_enrichment_support(ReportType.CREDIT_RISK)
    
    trade_data = pd.DataFrame({
        'ISIN': ['US123456789'],
        'Quantity': [1000000]
    })
    
    enriched = existing_report.enrich_data(trade_data)
    print(f"Enriched data shape: {enriched.shape}")
    print(f"Applied enrichments: {existing_report.enrichment_applier.enrichment_names}")


if __name__ == "__main__":
    example_usage()
    print("\n" + "="*50)
    integration_examples()
    
