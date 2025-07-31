import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, FrozenSet, Any
from dataclasses import dataclass


class ProductType(Enum):
    CASH = "Cash"
    REPO = "Repo"
    BOND = "Bond"
    SWAP_HEDGES = "SwapHedges"
    XCCY_SWAP = "XCCYSwap"
    UNKNOWN = "Unknown"


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1


class SystemType(Enum):
    B_E = "B-E"
    MAG = "MAG"


class MarketValueEnrichmentConfig:
    """Configuration class containing all column mappings and product lists"""
    
    # Input columns
    PRD_COLUMN = "PRD"
    CCY_COLUMN = "CCY"
    LEG_PAY_REC_COLUMN = "Leg.PayRec"
    LEG_TYPE_COLUMN = "Leg.Type"
    LEG_START_CASH_COLUMN = "Leg.StartCash"
    LEG_ARG_NOTIONAL_COLUMN = "LegArg.Notional"
    FX_COLUMN = "FX"
    POOL_FACTOR_COLUMN = "PoolFactor"
    MKT_PRICE_COLUMN = "Mkt Price of security"
    SYSTEM_COLUMN = "system"
    BOND_CURRENCY_COLUMN = "Bond.Currency"
    
    # Output columns
    PRODUCT_TYPE_BUCKET_COLUMN = "Product Type Bucket"
    CASH_CCY_COLUMN = "Cash CCY"
    SECURITY_CCY_COLUMN = "Security CCY"
    CASH_AMT_COLUMN = "Cash Amt"
    SECURITY_AMT_COLUMN = "Security Amt"
    EUR_CASH_AMT_COLUMN = "EUR Cash Amt"
    EUR_SECURITY_MV_COLUMN = "EUR Security MV"
    TRADE_DIRECTION_COLUMN = "Trade Direction"
    EUR_DIRECTIONAL_CASH_AMT_COLUMN = "EUR Directional Cash Amt"
    EUR_DIRECTIONAL_SECURITY_MV_COLUMN = "EUR Directional Security MV"
    
    # Product lists as frozen sets
    CASH_PRD_LIST: FrozenSet[str] = frozenset({
        "CASH_DEPOSIT", "CASH_LOAN", "CASH_OVERDRAFT", "MONEY_MARKET"
    })
    
    REPO_PRD_LIST: FrozenSet[str] = frozenset({
        "REPO", "REVERSE_REPO", "SELL_BUY_BACK", "BUY_SELL_BACK"
    })
    
    BOND_PRD_LIST: FrozenSet[str] = frozenset({
        "GOVERNMENT_BOND", "CORPORATE_BOND", "MUNICIPAL_BOND", "TREASURY_BILL"
    })
    
    SWAP_HEDGE_PRD_LIST: FrozenSet[str] = frozenset({
        "INTEREST_RATE_SWAP", "BASIS_SWAP", "OVERNIGHT_INDEX_SWAP"
    })
    
    XCCY_LIST: FrozenSet[str] = frozenset({
        "CROSS_CURRENCY_SWAP", "FX_SWAP", "CURRENCY_SWAP"
    })
    
    # Trade direction mapping
    MAG_DIRECTION_MAP: Dict[str, int] = {
        "PAY": TradeDirection.SHORT.value,
        "RECEIVE": TradeDirection.LONG.value,
        "SELL": TradeDirection.SHORT.value,
        "BUY": TradeDirection.LONG.value
    }


class MarketValueEnrichment:
    """
    Cash and Security Market Value Calculation Enrichment Class
    
    This class extends existing dataframes with market value calculations
    following specific business logic for different product types.
    """
    
    def __init__(self, config: MarketValueEnrichmentConfig = None):
        """
        Initialize the enrichment class with configuration
        
        Args:
            config: Configuration object containing column mappings and product lists
        """
        self.config = config or MarketValueEnrichmentConfig()
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to enrich dataframe with all calculated columns
        
        Args:
            df: Input dataframe to enrich
            
        Returns:
            Enriched dataframe with all new columns
        """
        df_enriched = df.copy()
        
        # Step 1: Create Product Type Bucket
        df_enriched = self._create_product_type_bucket(df_enriched)
        
        # Step 2: Populate Cash CCY and Security CCY
        df_enriched = self._populate_ccy_columns(df_enriched)
        
        # Step 3: Create Cash Amt column
        df_enriched = self._create_cash_amt(df_enriched)
        
        # Step 4: Create Security Amt column
        df_enriched = self._create_security_amt(df_enriched)
        
        # Step 5: Create EUR Cash Amt
        df_enriched = self._create_eur_cash_amt(df_enriched)
        
        # Step 6: Create EUR Security MV
        df_enriched = self._create_eur_security_mv(df_enriched)
        
        # Step 7: Determine Trade Direction
        df_enriched = self._determine_trade_direction(df_enriched)
        
        # Step 8: Create EUR Directional amounts
        df_enriched = self._create_eur_directional_amounts(df_enriched)
        
        return df_enriched
    
    def _create_product_type_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Create Product Type Bucket column based on PRD classification
        """
        conditions = [
            df[self.config.PRD_COLUMN].isin(self.config.CASH_PRD_LIST),
            df[self.config.PRD_COLUMN].isin(self.config.REPO_PRD_LIST),
            df[self.config.PRD_COLUMN].isin(self.config.BOND_PRD_LIST),
            df[self.config.PRD_COLUMN].isin(self.config.SWAP_HEDGE_PRD_LIST),
            df[self.config.PRD_COLUMN].isin(self.config.XCCY_LIST)
        ]
        
        choices = [
            ProductType.CASH.value,
            ProductType.REPO.value,
            ProductType.BOND.value,
            ProductType.SWAP_HEDGES.value,
            ProductType.XCCY_SWAP.value
        ]
        
        df[self.config.PRODUCT_TYPE_BUCKET_COLUMN] = np.select(
            conditions, choices, default=ProductType.UNKNOWN.value
        )
        
        return df
    
    def _populate_ccy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Populate Cash CCY and Security CCY columns
        """
        # Cash CCY logic
        cash_ccy_condition = (
            df[self.config.PRD_COLUMN].isin(self.config.CASH_PRD_LIST) |
            df[self.config.PRD_COLUMN].isin(self.config.REPO_PRD_LIST)
        )
        
        df[self.config.CASH_CCY_COLUMN] = np.where(
            cash_ccy_condition,
            df[self.config.CCY_COLUMN],
            ""
        )
        
        # Security CCY logic
        security_ccy_condition = (
            df[self.config.PRD_COLUMN].isin(self.config.REPO_PRD_LIST) |
            df[self.config.PRD_COLUMN].isin(self.config.BOND_PRD_LIST)
        )
        
        df[self.config.SECURITY_CCY_COLUMN] = np.where(
            security_ccy_condition,
            df[self.config.BOND_CURRENCY_COLUMN],
            ""
        )
        
        return df
    
    def _create_cash_amt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Create Cash Amt column
        """
        cash_conditions = [
            df[self.config.PRD_COLUMN].isin(self.config.CASH_PRD_LIST),
            df[self.config.PRD_COLUMN].isin(self.config.XCCY_LIST),
            df[self.config.PRD_COLUMN].isin(self.config.REPO_PRD_LIST)
        ]
        
        cash_choices = [
            df[self.config.LEG_ARG_NOTIONAL_COLUMN],
            df[self.config.LEG_START_CASH_COLUMN],
            0
        ]
        
        df[self.config.CASH_AMT_COLUMN] = np.select(
            cash_conditions, cash_choices, default=0
        )
        
        return df
    
    def _create_security_amt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Create Security Amt column
        """
        # Condition: PRD not in CashPRDList AND PRD in RepoPRDList, BondPRDList, SwapHedges
        security_condition = (
            ~df[self.config.PRD_COLUMN].isin(self.config.CASH_PRD_LIST) &
            (df[self.config.PRD_COLUMN].isin(self.config.REPO_PRD_LIST) |
             df[self.config.PRD_COLUMN].isin(self.config.BOND_PRD_LIST) |
             df[self.config.PRD_COLUMN].isin(self.config.SWAP_HEDGE_PRD_LIST))
        )
        
        df[self.config.SECURITY_AMT_COLUMN] = np.where(
            security_condition,
            df[self.config.LEG_ARG_NOTIONAL_COLUMN],
            0
        )
        
        return df
    
    def _create_eur_cash_amt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Create EUR Cash Amt column
        """
        df[self.config.EUR_CASH_AMT_COLUMN] = (
            df[self.config.CASH_AMT_COLUMN] * df[self.config.FX_COLUMN]
        )
        
        return df
    
    def _create_eur_security_mv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6: Create EUR Security MV column
        """
        df[self.config.EUR_SECURITY_MV_COLUMN] = (
            df[self.config.SECURITY_AMT_COLUMN] * 
            df[self.config.FX_COLUMN] * 
            df[self.config.POOL_FACTOR_COLUMN] * 
            df[self.config.MKT_PRICE_COLUMN]
        )
        
        return df
    
    def _determine_trade_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 7: Determine Trade Direction
        """
        conditions = [
            df[self.config.SYSTEM_COLUMN] == SystemType.B_E.value,
            (df[self.config.SYSTEM_COLUMN] != SystemType.B_E.value) & 
            (df[self.config.LEG_PAY_REC_COLUMN] == "Pay"),
            df[self.config.SYSTEM_COLUMN] == SystemType.MAG.value
        ]
        
        # For B-E system, direction = 1
        # For non-B-E system with Leg.PayRec = Pay, direction = -1, else 1
        # For MAG system, use Leg.Type to lookup direction in mag_direction_map, else 1
        def get_mag_direction(row):
            if row[self.config.SYSTEM_COLUMN] == SystemType.MAG.value:
                return self.config.MAG_DIRECTION_MAP.get(
                    row[self.config.LEG_TYPE_COLUMN], TradeDirection.LONG.value
                )
            return TradeDirection.LONG.value
        
        choices = [
            TradeDirection.LONG.value,  # B-E system
            TradeDirection.SHORT.value,  # Non-B-E with Pay
            df.apply(get_mag_direction, axis=1)  # MAG system
        ]
        
        df[self.config.TRADE_DIRECTION_COLUMN] = np.select(
            conditions, choices, default=TradeDirection.LONG.value
        )
        
        return df
    
    def _create_eur_directional_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 8: Create EUR Directional Cash Amt and EUR Directional Security MV
        """
        df[self.config.EUR_DIRECTIONAL_CASH_AMT_COLUMN] = (
            df[self.config.EUR_CASH_AMT_COLUMN] * 
            df[self.config.TRADE_DIRECTION_COLUMN]
        )
        
        df[self.config.EUR_DIRECTIONAL_SECURITY_MV_COLUMN] = (
            df[self.config.EUR_SECURITY_MV_COLUMN] * 
            df[self.config.TRADE_DIRECTION_COLUMN]
        )
        
        return df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        "PRD": ["CASH_DEPOSIT", "REPO", "GOVERNMENT_BOND", "INTEREST_RATE_SWAP", "CROSS_CURRENCY_SWAP"],
        "CCY": ["USD", "EUR", "GBP", "JPY", "USD"],
        "Leg.PayRec": ["Receive", "Pay", "Receive", "Pay", "Receive"],
        "Leg.Type": ["RECEIVE", "PAY", "BUY", "SELL", "RECEIVE"],
        "Leg.StartCash": [1000000, 500000, 750000, 0, 2000000],
        "LegArg.Notional": [1000000, 500000, 750000, 1500000, 2000000],
        "FX": [0.85, 1.0, 1.15, 0.007, 0.85],
        "PoolFactor": [1.0, 0.98, 1.02, 1.0, 1.0],
        "Mkt Price of security": [100.5, 99.8, 101.2, 100.0, 100.0],
        "system": ["B-E", "MAG", "B-E", "MAG", "B-E"],
        "Bond.Currency": ["", "EUR", "GBP", "", ""]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize enrichment class and process dataframe
    enrichment = MarketValueEnrichment()
    enriched_df = enrichment.enrich_dataframe(df)
    
    # Display results
    print("Original DataFrame:")
    print(df.to_string(index=False))
    print("\nEnriched DataFrame:")
    print(enriched_df.to_string(index=False))
    
    # Display only new columns
    new_columns = [
        enrichment.config.PRODUCT_TYPE_BUCKET_COLUMN,
        enrichment.config.CASH_CCY_COLUMN,
        enrichment.config.SECURITY_CCY_COLUMN,
        enrichment.config.CASH_AMT_COLUMN,
        enrichment.config.SECURITY_AMT_COLUMN,
        enrichment.config.EUR_CASH_AMT_COLUMN,
        enrichment.config.EUR_SECURITY_MV_COLUMN,
        enrichment.config.TRADE_DIRECTION_COLUMN,
        enrichment.config.EUR_DIRECTIONAL_CASH_AMT_COLUMN,
        enrichment.config.EUR_DIRECTIONAL_SECURITY_MV_COLUMN
    ]
    
    print(f"\nNew Columns Added:")
    print(enriched_df[new_columns].to_string(index=False))
