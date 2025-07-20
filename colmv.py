from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import numpy as np
from collections import defaultdict


class BookingSystem(Enum):
    MAG = "MAG"


class TradeType(Enum):
    BOND_BORROW = "BOND_BORROW"
    BOND_LEND = "BOND_LEND"
    REPO = "REPO"
    REVERSE_REPO = "REVERSE_REPO"
    TRI_PARTY_REPO = "TRI_PARTY_REPO"
    TRI_PARTY_REVERSE_REPO = "TRI_PARTY_REVERSE_REPO"


class SwapType(Enum):
    BOND_BORROW_LEND = "BOND_BORROW_LEND"
    REPO_REVERSE_REPO = "REPO_REVERSE_REPO"
    TRI_PARTY_REPO_REVERSE = "TRI_PARTY_REPO_REVERSE"


class HQLAStatus(Enum):
    LEVEL_1 = "LEVEL_1"
    LEVEL_2A = "LEVEL_2A"
    LEVEL_2B = "LEVEL_2B"
    NON_HQLA = "NON_HQLA"


class BondRating(Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"
    NR = "NR"  # Not Rated


class AssetType(Enum):
    GOVT = "GOVT"
    CORP = "CORP"
    ABS = "ABS"


class SwapClassification(Enum):
    UPGRADE = "UPGRADE"
    DOWNGRADE = "DOWNGRADE"
    NEUTRAL = "NEUTRAL"


class ColumnNames(Enum):
    TRADE_ID = "trade_id"
    BOOKING_SYSTEM = "booking_system"
    TRADE_TYPE = "trade_type"
    COUNTERPARTY = "counterparty"
    COUNTERPARTY_CODE = "counterparty_code"
    NOTIONAL = "notional"
    CURRENCY = "currency"
    MARKET_PRICE = "market_price"
    MATURITY_DATE = "maturity_date"
    COLLATERAL_SWAPS_FLAG = "collateral_swaps_flag"
    IS_SECURED_FLAG = "is_secured_flag"
    HQLA_STATUS = "hqla_status"
    BOND_RATING = "bond_rating"
    ASSET_TYPE = "asset_type"
    # Output columns
    COLLATERAL_SWAP_INDICATOR = "collateral_swap_indicator"
    COLLATERAL_SWAP_ID = "collateral_swap_id"
    SWAP_CLASSIFICATION = "swap_classification"
    MARKET_VALUE = "market_value"


class SwapMatch:
    """Represents a matched collateral swap group"""
    
    def __init__(self, swap_id: str, swap_type: SwapType, trade_ids: List[str], 
                 counterparty: str, maturity_date: str, currency: str, 
                 total_mv_difference: float, classification: SwapClassification = SwapClassification.NEUTRAL):
        self.swap_id = swap_id
        self.swap_type = swap_type
        self.trade_ids = trade_ids
        self.counterparty = counterparty
        self.maturity_date = maturity_date
        self.currency = currency
        self.total_mv_difference = total_mv_difference
        self.classification = classification
    
    def __repr__(self) -> str:
        return (f"SwapMatch(swap_id='{self.swap_id}', swap_type={self.swap_type.name}, "
                f"trades={len(self.trade_ids)}, counterparty='{self.counterparty}', "
                f"classification={self.classification.name})")
    
    def __str__(self) -> str:
        return (f"Collateral Swap {self.swap_id}: {len(self.trade_ids)} trades "
                f"with {self.counterparty} ({self.classification.name})")


class CollateralSwapIdentifier:
    """
    Main class for identifying and classifying collateral swaps in trading data
    """
    
    def __init__(self, use_original_collateral_flag: bool = False,
                 enable_hqla_classification: bool = True,
                 enable_rating_classification: bool = True,
                 enable_asset_type_classification: bool = True,
                 mv_threshold: float = 1000.0):
        
        # Configuration flags
        self.use_original_collateral_flag = use_original_collateral_flag
        self.enable_hqla_classification = enable_hqla_classification
        self.enable_rating_classification = enable_rating_classification
        self.enable_asset_type_classification = enable_asset_type_classification
        
        # Thresholds
        self.mv_threshold = mv_threshold
        
        # Trade type mappings
        self.swap_type_mappings = {
            SwapType.BOND_BORROW_LEND: (
                [TradeType.BOND_BORROW], 
                [TradeType.BOND_LEND]
            ),
            SwapType.REPO_REVERSE_REPO: (
                [TradeType.REPO], 
                [TradeType.REVERSE_REPO]
            ),
            SwapType.TRI_PARTY_REPO_REVERSE: (
                [TradeType.TRI_PARTY_REPO], 
                [TradeType.TRI_PARTY_REVERSE_REPO]
            )
        }
        
        # Classification hierarchies
        self.hqla_hierarchy = {
            HQLAStatus.LEVEL_1: 4,
            HQLAStatus.LEVEL_2A: 3,
            HQLAStatus.LEVEL_2B: 2,
            HQLAStatus.NON_HQLA: 1
        }
        
        self.rating_hierarchy = {
            BondRating.AAA: 10,
            BondRating.AA: 9,
            BondRating.A: 8,
            BondRating.BBB: 7,
            BondRating.BB: 6,
            BondRating.B: 5,
            BondRating.CCC: 4,
            BondRating.CC: 3,
            BondRating.C: 2,
            BondRating.D: 1,
            BondRating.NR: 0
        }
        
        self.asset_type_hierarchy = {
            AssetType.GOVT: 3,
            AssetType.CORP: 2,
            AssetType.ABS: 1
        }
    
    def __repr__(self) -> str:
        return (f"CollateralSwapIdentifier(use_original_flag={self.use_original_collateral_flag}, "
                f"mv_threshold={self.mv_threshold}, hqla={self.enable_hqla_classification}, "
                f"rating={self.enable_rating_classification}, asset_type={self.enable_asset_type_classification})")
    
    def __str__(self) -> str:
        flags = []
        if self.enable_hqla_classification:
            flags.append("HQLA")
        if self.enable_rating_classification:
            flags.append("Rating")
        if self.enable_asset_type_classification:
            flags.append("AssetType")
        
        mode = "Original Flag" if self.use_original_collateral_flag else "Logic-based"
        return (f"Collateral Swap Identifier - Mode: {mode}, "
                f"MV Threshold: ${self.mv_threshold:,.2f}, "
                f"Classifications: {', '.join(flags) if flags else 'None'}")
    
    def identify_collateral_swaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to identify collateral swaps in the dataset
        """
        # Filter for MAG system only
        df_mag = df[df[ColumnNames.BOOKING_SYSTEM.value] == BookingSystem.MAG.value].copy()
        
        # Calculate market value
        df_mag[ColumnNames.MARKET_VALUE.value] = (
            df_mag[ColumnNames.MARKET_PRICE.value] * df_mag[ColumnNames.NOTIONAL.value]
        )
        
        # Initialize output columns
        df_mag[ColumnNames.COLLATERAL_SWAP_INDICATOR.value] = False
        df_mag[ColumnNames.COLLATERAL_SWAP_ID.value] = None
        df_mag[ColumnNames.SWAP_CLASSIFICATION.value] = None
        
        if self.use_original_collateral_flag:
            # Use original collateral swaps flag
            df_mag[ColumnNames.COLLATERAL_SWAP_INDICATOR.value] = (
                df_mag[ColumnNames.COLLATERAL_SWAPS_FLAG.value] == True
            )
        else:
            # Use logic-based identification
            swap_matches = self._identify_swap_matches(df_mag)
            self._apply_swap_matches(df_mag, swap_matches)
        
        return df_mag
    
    def _identify_swap_matches(self, df: pd.DataFrame) -> List[SwapMatch]:
        """
        Identify potential collateral swap matches based on trade types and criteria
        """
        swap_matches = []
        swap_id_counter = 1
        
        for swap_type, (borrow_types, lend_types) in self.swap_type_mappings.items():
            matches = self._find_matches_for_swap_type(df, swap_type, borrow_types, lend_types)
            
            for match in matches:
                match.swap_id = f"CS_{swap_id_counter:06d}"
                swap_id_counter += 1
                swap_matches.append(match)
        
        return swap_matches
    
    def _find_matches_for_swap_type(
        self, 
        df: pd.DataFrame, 
        swap_type: SwapType,
        borrow_types: List[TradeType], 
        lend_types: List[TradeType]
    ) -> List[SwapMatch]:
        """
        Find matches for a specific swap type
        """
        matches = []
        
        # Get borrow and lend trades
        borrow_trades = df[df[ColumnNames.TRADE_TYPE.value].isin([t.value for t in borrow_types])]
        lend_trades = df[df[ColumnNames.TRADE_TYPE.value].isin([t.value for t in lend_types])]
        
        # Group by counterparty, maturity date, and currency
        groupby_cols = [
            ColumnNames.COUNTERPARTY.value,
            ColumnNames.MATURITY_DATE.value,
            ColumnNames.CURRENCY.value
        ]
        
        borrow_groups = borrow_trades.groupby(groupby_cols)
        lend_groups = lend_trades.groupby(groupby_cols)
        
        # Find matching groups
        common_keys = set(borrow_groups.groups.keys()) & set(lend_groups.groups.keys())
        
        for key in common_keys:
            counterparty, maturity_date, currency = key
            
            borrow_group = borrow_groups.get_group(key)
            lend_group = lend_groups.get_group(key)
            
            # Try to match based on market value
            matched_groups = self._match_by_market_value(borrow_group, lend_group)
            
            for borrow_trades_matched, lend_trades_matched in matched_groups:
                all_trade_ids = (
                    borrow_trades_matched[ColumnNames.TRADE_ID.value].tolist() +
                    lend_trades_matched[ColumnNames.TRADE_ID.value].tolist()
                )
                
                borrow_mv = borrow_trades_matched[ColumnNames.MARKET_VALUE.value].sum()
                lend_mv = lend_trades_matched[ColumnNames.MARKET_VALUE.value].sum()
                mv_difference = abs(borrow_mv - lend_mv)
                
                # Classify the swap
                classification = self._classify_swap(borrow_trades_matched, lend_trades_matched)
                
                match = SwapMatch(
                    swap_id="",  # Will be assigned later
                    swap_type=swap_type,
                    trade_ids=all_trade_ids,
                    counterparty=counterparty,
                    maturity_date=maturity_date,
                    currency=currency,
                    total_mv_difference=mv_difference,
                    classification=classification
                )
                
                matches.append(match)
        
        return matches
    
    def _match_by_market_value(
        self, 
        borrow_trades: pd.DataFrame, 
        lend_trades: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Match borrow and lend trades based on market value within threshold
        """
        matches = []
        used_borrow_indices = set()
        used_lend_indices = set()
        
        # Simple greedy matching algorithm
        # In practice, you might want to use more sophisticated algorithms
        for borrow_idx, borrow_row in borrow_trades.iterrows():
            if borrow_idx in used_borrow_indices:
                continue
                
            borrow_mv = borrow_row[ColumnNames.MARKET_VALUE.value]
            
            # Find lend trades that could match
            for lend_idx, lend_row in lend_trades.iterrows():
                if lend_idx in used_lend_indices:
                    continue
                
                lend_mv = lend_row[ColumnNames.MARKET_VALUE.value]
                mv_difference = abs(borrow_mv - lend_mv)
                
                if mv_difference <= self.mv_threshold:
                    # Found a match
                    borrow_match = borrow_trades.loc[[borrow_idx]]
                    lend_match = lend_trades.loc[[lend_idx]]
                    
                    matches.append((borrow_match, lend_match))
                    used_borrow_indices.add(borrow_idx)
                    used_lend_indices.add(lend_idx)
                    break
        
        return matches
    
    def _classify_swap(
        self, 
        borrow_trades: pd.DataFrame, 
        lend_trades: pd.DataFrame
    ) -> SwapClassification:
        """
        Classify swap as upgrade, downgrade, or neutral based on hierarchy
        """
        if not (self.enable_hqla_classification or 
                self.enable_rating_classification or 
                self.enable_asset_type_classification):
            return SwapClassification.NEUTRAL
        
        borrow_score = self._calculate_quality_score(borrow_trades)
        lend_score = self._calculate_quality_score(lend_trades)
        
        if lend_score > borrow_score:
            return SwapClassification.UPGRADE
        elif lend_score < borrow_score:
            return SwapClassification.DOWNGRADE
        else:
            return SwapClassification.NEUTRAL
    
    def _calculate_quality_score(self, trades: pd.DataFrame) -> float:
        """
        Calculate weighted quality score for a group of trades
        """
        total_score = 0.0
        total_mv = 0.0
        
        for _, trade in trades.iterrows():
            mv = trade[ColumnNames.MARKET_VALUE.value]
            score = 0.0
            
            # HQLA score (highest priority)
            if (self.enable_hqla_classification and 
                ColumnNames.HQLA_STATUS.value in trade and 
                pd.notna(trade[ColumnNames.HQLA_STATUS.value])):
                hqla_status = HQLAStatus(trade[ColumnNames.HQLA_STATUS.value])
                score += self.hqla_hierarchy.get(hqla_status, 0) * 1000
            
            # Bond rating score (medium priority)
            if (self.enable_rating_classification and 
                ColumnNames.BOND_RATING.value in trade and 
                pd.notna(trade[ColumnNames.BOND_RATING.value])):
                bond_rating = BondRating(trade[ColumnNames.BOND_RATING.value])
                score += self.rating_hierarchy.get(bond_rating, 0) * 100
            
            # Asset type score (lowest priority)
            if (self.enable_asset_type_classification and 
                ColumnNames.ASSET_TYPE.value in trade and 
                pd.notna(trade[ColumnNames.ASSET_TYPE.value])):
                asset_type = AssetType(trade[ColumnNames.ASSET_TYPE.value])
                score += self.asset_type_hierarchy.get(asset_type, 0) * 10
            
            total_score += score * mv
            total_mv += mv
        
        return total_score / total_mv if total_mv > 0 else 0.0
    
    def _apply_swap_matches(self, df: pd.DataFrame, swap_matches: List[SwapMatch]) -> None:
        """
        Apply swap matches to the dataframe
        """
        for match in swap_matches:
            mask = df[ColumnNames.TRADE_ID.value].isin(match.trade_ids)
            df.loc[mask, ColumnNames.COLLATERAL_SWAP_INDICATOR.value] = True
            df.loc[mask, ColumnNames.COLLATERAL_SWAP_ID.value] = match.swap_id
            df.loc[mask, ColumnNames.SWAP_CLASSIFICATION.value] = match.classification.value
    
    def get_swap_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of identified collateral swaps
        """
        swap_trades = df[df[ColumnNames.COLLATERAL_SWAP_INDICATOR.value] == True]
        
        if swap_trades.empty:
            return pd.DataFrame()
        
        summary = swap_trades.groupby(ColumnNames.COLLATERAL_SWAP_ID.value).agg({
            ColumnNames.TRADE_ID.value: 'count',
            ColumnNames.COUNTERPARTY.value: 'first',
            ColumnNames.MATURITY_DATE.value: 'first',
            ColumnNames.CURRENCY.value: 'first',
            ColumnNames.MARKET_VALUE.value: 'sum',
            ColumnNames.SWAP_CLASSIFICATION.value: 'first'
        }).rename(columns={
            ColumnNames.TRADE_ID.value: 'trade_count',
            ColumnNames.COUNTERPARTY.value: 'counterparty',
            ColumnNames.MATURITY_DATE.value: 'maturity_date',
            ColumnNames.CURRENCY.value: 'currency',
            ColumnNames.MARKET_VALUE.value: 'total_market_value',
            ColumnNames.SWAP_CLASSIFICATION.value: 'classification'
        })
        
        return summary


# Example usage and dummy data generation:
def create_dummy_trading_data() -> pd.DataFrame:
    """
    Create dummy trading data with complex one-to-many and many-to-many relationships
    """
    import datetime
    from datetime import timedelta
    
    np.random.seed(42)  # For reproducible results
    
    data = []
    trade_id_counter = 1
    
    # Scenario 1: 2 Repos + 4 Reverse Repos (many-to-many)
    # All should have similar total market values to form a collateral swap
    counterparty = "COUNTERPARTY_A"
    maturity_date = "2025-12-31"
    currency = "USD"
    
    # 2 Repo trades - total MV ~2,000,000
    for i in range(2):
        data.append({
            ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
            ColumnNames.BOOKING_SYSTEM.value: BookingSystem.MAG.value,
            ColumnNames.TRADE_TYPE.value: TradeType.REPO.value,
            ColumnNames.COUNTERPARTY.value: counterparty,
            ColumnNames.COUNTERPARTY_CODE.value: "CTP_A",
            ColumnNames.NOTIONAL.value: 1000000,  # 1M each
            ColumnNames.CURRENCY.value: currency,
            ColumnNames.MARKET_PRICE.value: 1.0 + i * 0.01,  # 1.00, 1.01
            ColumnNames.MATURITY_DATE.value: maturity_date,
            ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
            ColumnNames.IS_SECURED_FLAG.value: True,
            ColumnNames.HQLA_STATUS.value: HQLAStatus.LEVEL_2A.value,
            ColumnNames.BOND_RATING.value: BondRating.AA.value,
            ColumnNames.ASSET_TYPE.value: AssetType.CORP.value
        })
        trade_id_counter += 1
    
    # 4 Reverse Repo trades - total MV ~2,010,000 (close match within threshold)
    reverse_repo_notionals = [500000, 400000, 600000, 500000]  # Total: 2M
    reverse_repo_prices = [1.002, 1.003, 1.001, 1.004]
    
    for i in range(4):
        data.append({
            ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
            ColumnNames.BOOKING_SYSTEM.value: BookingSystem.MAG.value,
            ColumnNames.TRADE_TYPE.value: TradeType.REVERSE_REPO.value,
            ColumnNames.COUNTERPARTY.value: counterparty,
            ColumnNames.COUNTERPARTY_CODE.value: "CTP_A",
            ColumnNames.NOTIONAL.value: reverse_repo_notionals[i],
            ColumnNames.CURRENCY.value: currency,
            ColumnNames.MARKET_PRICE.value: reverse_repo_prices[i],
            ColumnNames.MATURITY_DATE.value: maturity_date,
            ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
            ColumnNames.IS_SECURED_FLAG.value: True,
            ColumnNames.HQLA_STATUS.value: HQLAStatus.LEVEL_1.value,  # Upgrade scenario
            ColumnNames.BOND_RATING.value: BondRating.AAA.value,
            ColumnNames.ASSET_TYPE.value: AssetType.GOVT.value
        })
        trade_id_counter += 1
    
    # Scenario 2: 1 Bond Borrow + 3 Bond Lends (one-to-many)
    counterparty_b = "COUNTERPARTY_B"
    maturity_date_b = "2025-11-30"
    
    # 1 Bond Borrow - MV = 1,500,000
    data.append({
        ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
        ColumnNames.BOOKING_SYSTEM.value: BookingSystem.MAG.value,
        ColumnNames.TRADE_TYPE.value: TradeType.BOND_BORROW.value,
        ColumnNames.COUNTERPARTY.value: counterparty_b,
        ColumnNames.COUNTERPARTY_CODE.value: "CTP_B",
        ColumnNames.NOTIONAL.value: 1500000,
        ColumnNames.CURRENCY.value: currency,
        ColumnNames.MARKET_PRICE.value: 1.0,
        ColumnNames.MATURITY_DATE.value: maturity_date_b,
        ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
        ColumnNames.IS_SECURED_FLAG.value: True,
        ColumnNames.HQLA_STATUS.value: HQLAStatus.LEVEL_2B.value,
        ColumnNames.BOND_RATING.value: BondRating.BBB.value,
        ColumnNames.ASSET_TYPE.value: AssetType.CORP.value
    })
    trade_id_counter += 1
    
    # 3 Bond Lends - total MV ~1,498,000 (close match)
    lend_notionals = [600000, 500000, 400000]
    lend_prices = [0.999, 0.998, 0.997]
    
    for i in range(3):
        data.append({
            ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
            ColumnNames.BOOKING_SYSTEM.value: BookingSystem.MAG.value,
            ColumnNames.TRADE_TYPE.value: TradeType.BOND_LEND.value,
            ColumnNames.COUNTERPARTY.value: counterparty_b,
            ColumnNames.COUNTERPARTY_CODE.value: "CTP_B",
            ColumnNames.NOTIONAL.value: lend_notionals[i],
            ColumnNames.CURRENCY.value: currency,
            ColumnNames.MARKET_PRICE.value: lend_prices[i],
            ColumnNames.MATURITY_DATE.value: maturity_date_b,
            ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
            ColumnNames.IS_SECURED_FLAG.value: True,
            ColumnNames.HQLA_STATUS.value: HQLAStatus.NON_HQLA.value,  # Downgrade scenario
            ColumnNames.BOND_RATING.value: BondRating.B.value,
            ColumnNames.ASSET_TYPE.value: AssetType.ABS.value
        })
        trade_id_counter += 1
    
    # Scenario 3: Perfect 1-to-1 match (Tri-party repos)
    counterparty_c = "COUNTERPARTY_C"
    maturity_date_c = "2025-10-15"
    
    # 1 Tri-party Repo
    data.append({
        ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
        ColumnNames.BOOKING_SYSTEM.value: BookingSystem.MAG.value,
        ColumnNames.TRADE_TYPE.value: TradeType.TRI_PARTY_REPO.value,
        ColumnNames.COUNTERPARTY.value: counterparty_c,
        ColumnNames.COUNTERPARTY_CODE.value: "CTP_C",
        ColumnNames.NOTIONAL.value: 800000,
        ColumnNames.CURRENCY.value: currency,
        ColumnNames.MARKET_PRICE.value: 1.0,
        ColumnNames.MATURITY_DATE.value: maturity_date_c,
        ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
        ColumnNames.IS_SECURED_FLAG.value: True,
        ColumnNames.HQLA_STATUS.value: HQLAStatus.LEVEL_2A.value,
        ColumnNames.BOND_RATING.value: BondRating.A.value,
        ColumnNames.ASSET_TYPE.value: AssetType.CORP.value
    })
    trade_id_counter += 1
    
    # 1 Tri-party Reverse Repo (exact match)
    data.append({
        ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
        ColumnNames.BOOKING_SYSTEM.value: BookingSystem.MAG.value,
        ColumnNames.TRADE_TYPE.value: TradeType.TRI_PARTY_REVERSE_REPO.value,
        ColumnNames.COUNTERPARTY.value: counterparty_c,
        ColumnNames.COUNTERPARTY_CODE.value: "CTP_C",
        ColumnNames.NOTIONAL.value: 800000,
        ColumnNames.CURRENCY.value: currency,
        ColumnNames.MARKET_PRICE.value: 1.0,
        ColumnNames.MATURITY_DATE.value: maturity_date_c,
        ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
        ColumnNames.IS_SECURED_FLAG.value: True,
        ColumnNames.HQLA_STATUS.value: HQLAStatus.LEVEL_2A.value,  # Neutral scenario
        ColumnNames.BOND_RATING.value: BondRating.A.value,
        ColumnNames.ASSET_TYPE.value: AssetType.CORP.value
    })
    trade_id_counter += 1
    
    # Scenario 4: Non-matching trades (different counterparty/maturity)
    for i in range(3):
        data.append({
            ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
            ColumnNames.BOOKING_SYSTEM.value: BookingSystem.MAG.value,
            ColumnNames.TRADE_TYPE.value: TradeType.REPO.value,
            ColumnNames.COUNTERPARTY.value: f"COUNTERPARTY_D_{i}",
            ColumnNames.COUNTERPARTY_CODE.value: f"CTP_D_{i}",
            ColumnNames.NOTIONAL.value: 500000,
            ColumnNames.CURRENCY.value: currency,
            ColumnNames.MARKET_PRICE.value: 1.0,
            ColumnNames.MATURITY_DATE.value: f"2025-{9+i:02d}-15",  # Different maturities
            ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
            ColumnNames.IS_SECURED_FLAG.value: True,
            ColumnNames.HQLA_STATUS.value: HQLAStatus.LEVEL_1.value,
            ColumnNames.BOND_RATING.value: BondRating.AAA.value,
            ColumnNames.ASSET_TYPE.value: AssetType.GOVT.value
        })
        trade_id_counter += 1
    
    # Scenario 5: Non-MAG system trades (should be filtered out)
    data.append({
        ColumnNames.TRADE_ID.value: f"TRD_{trade_id_counter:06d}",
        ColumnNames.BOOKING_SYSTEM.value: "OTHER_SYSTEM",
        ColumnNames.TRADE_TYPE.value: TradeType.REPO.value,
        ColumnNames.COUNTERPARTY.value: "COUNTERPARTY_OTHER",
        ColumnNames.COUNTERPARTY_CODE.value: "CTP_OTHER",
        ColumnNames.NOTIONAL.value: 1000000,
        ColumnNames.CURRENCY.value: currency,
        ColumnNames.MARKET_PRICE.value: 1.0,
        ColumnNames.MATURITY_DATE.value: "2025-12-31",
        ColumnNames.COLLATERAL_SWAPS_FLAG.value: None,
        ColumnNames.IS_SECURED_FLAG.value: True,
        ColumnNames.HQLA_STATUS.value: HQLAStatus.LEVEL_1.value,
        ColumnNames.BOND_RATING.value: BondRating.AAA.value,
        ColumnNames.ASSET_TYPE.value: AssetType.GOVT.value
    })
    
    return pd.DataFrame(data)


"""
# Example usage:
if __name__ == "__main__":
    # Create dummy data
    dummy_data = create_dummy_trading_data()
    print(f"Created dummy dataset with {len(dummy_data)} trades")
    
    # Initialize the identifier with custom settings
    identifier = CollateralSwapIdentifier(
        use_original_collateral_flag=False,
        enable_hqla_classification=True,
        enable_rating_classification=True,
        enable_asset_type_classification=True,
        mv_threshold=5000.0
    )
    
    print(f"\\n{identifier}")
    print(f"\\n{repr(identifier)}")
    
    # Process the dataset
    result_df = identifier.identify_collateral_swaps(dummy_data)
    
    # Show results
    swap_trades = result_df[result_df[ColumnNames.COLLATERAL_SWAP_INDICATOR.value] == True]
    print(f"\\nIdentified {len(swap_trades)} trades as part of collateral swaps")
    
    # Get summary
    summary = identifier.get_swap_summary(result_df)
    print(f"\\nCollateral Swap Summary:")
    print(summary)
    
    # Show detailed breakdown
    print(f"\\nDetailed Breakdown:")
    for swap_id in summary.index:
        swap_trades_detail = result_df[result_df[ColumnNames.COLLATERAL_SWAP_ID.value] == swap_id]
        print(f"\\n{swap_id}:")
        for _, trade in swap_trades_detail.iterrows():
            print(f"  - {trade[ColumnNames.TRADE_ID.value]}: {trade[ColumnNames.TRADE_TYPE.value]} "
                  f"(MV: ${trade[ColumnNames.MARKET_VALUE.value]:,.2f})")
"""
