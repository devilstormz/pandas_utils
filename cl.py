import pandas as pd
import numpy as np
import logging
from collections import namedtuple
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Enumerations
class TradeType(Enum):
    """Trade type enumerations"""
    BOND_BORROW = 'BB'
    BOND_LEND = 'BL'
    REPO = 'REP'
    REVERSE_REPO = 'REV'
    TRI_PARTY_REPO = 'TP'
    TRI_PARTY_REVERSE = 'TR'
    TRI_PARTY_REPO_ALT = 'TPR'
    TRI_PARTY_REVERSE_ALT = 'TRV'

class SwapType(Enum):
    """Collateral swap type classifications"""
    UPGRADE = 'UPGRADE'
    DOWNGRADE = 'DOWNGRADE'
    NEUTRAL = 'NEUTRAL'

class TransactionType(Enum):
    """Transaction type classifications"""
    COLLATERAL_SWAP = 'COLLATERAL_SWAP'
    FINANCING = 'FINANCING'
    FUNDING = 'FUNDING'
    UNSECURED_BORROW_LEND = 'UNSECURED_BORROW_LEND'

class ApproachType(Enum):
    """Identification approach types"""
    DESK_LOGIC = 'DESK_LOGIC'
    CONTRACT_BASED = 'CONTRACT_BASED'
    FLAG_BASED = 'FLAG_BASED'

class HQLAStatus(Enum):
    """HQLA status classifications"""
    LEVEL_1 = 'LEVEL_1'
    LEVEL_2A = 'LEVEL_2A'
    LEVEL_2B = 'LEVEL_2B'
    NON_HQLA = 'NON_HQLA'

class BondAssetType(Enum):
    """Bond asset type classifications"""
    GOVT = 'GOVT'
    CORP_SENIOR = 'CORP_SENIOR'
    CORP_JUNIOR = 'CORP_JUNIOR'
    ABS_CLO = 'ABS_CLO'

# Configuration structures
ApproachConfig = namedtuple('ApproachConfig', [
    'enabled', 'name', 'description'
])

HierarchyConfig = namedtuple('HierarchyConfig', [
    'hqla_enabled', 'rating_enabled', 'asset_type_enabled',
    'use_worst_rating_field', 'calculate_rating_score'
])

ThresholdConfig = namedtuple('ThresholdConfig', [
    'market_value_threshold', 'rating_threshold'
])

BookConfig = namedtuple('BookConfig', [
    'book_a', 'book_b', 'cpty_a', 'cpty_b'
])

# Main Configuration Class
class CollateralSwapConfig:
    """Configuration class for collateral swap identification"""
    
    def __init__(self):
        self.approaches = {
            ApproachType.DESK_LOGIC: ApproachConfig(
                enabled=True,
                name="Desk Logic Approach",
                description="Waterfall structure based on desk rules"
            ),
            ApproachType.CONTRACT_BASED: ApproachConfig(
                enabled=True,
                name="Contract Based Approach",
                description="Synthetic key using contract ID and trade details"
            ),
            ApproachType.FLAG_BASED: ApproachConfig(
                enabled=True,
                name="Flag Based Approach",
                description="Uses existing collateral swap and secured flags"
            )
        }
        
        self.hierarchy = HierarchyConfig(
            hqla_enabled=True,
            rating_enabled=True,
            asset_type_enabled=True,
            use_worst_rating_field=False,
            calculate_rating_score=True
        )
        
        self.thresholds = ThresholdConfig(
            market_value_threshold=1000.0,
            rating_threshold=0.5
        )
        
        self.books = BookConfig(
            book_a="BOOK_A",
            book_b="BOOK_B",
            cpty_a="CPTY_A",
            cpty_b="CPTY_B"
        )
        
        self.use_original_flags = False
        
    def enable_approach(self, approach: ApproachType, enabled: bool = True):
        """Enable or disable a specific approach"""
        if approach in self.approaches:
            config = self.approaches[approach]
            self.approaches[approach] = ApproachConfig(
                enabled=enabled,
                name=config.name,
                description=config.description
            )

class CollateralSwapEngine:
    """Main collateral swap identification and matching engine"""
    
    def __init__(self, config: CollateralSwapConfig = None):
        self.config = config or CollateralSwapConfig()
        self.logger = self._setup_logging()
        self.audit_data = []
        self.group_counter = 0
        
        # Trade type combinations for valid collateral swaps
        self.valid_combinations = {
            frozenset([TradeType.BOND_BORROW.value, TradeType.BOND_LEND.value]),
            frozenset([TradeType.REPO.value, TradeType.REVERSE_REPO.value]),
            frozenset([TradeType.TRI_PARTY_REPO.value, TradeType.TRI_PARTY_REVERSE.value]),
            frozenset([TradeType.TRI_PARTY_REPO_ALT.value, TradeType.TRI_PARTY_REVERSE_ALT.value])
        }
        
        # Rating scores (higher is worse)
        self.rating_scores = {
            'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
            'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10,
            'BB+': 11, 'BB': 12, 'BB-': 13,
            'B+': 14, 'B': 15, 'B-': 16,
            'CCC+': 17, 'CCC': 18, 'CCC-': 19,
            'CC': 20, 'C': 21, 'D': 22
        }
        
        # Asset type scores (higher is worse)
        self.asset_type_scores = {
            BondAssetType.GOVT.value: 1,
            BondAssetType.CORP_SENIOR.value: 2,
            BondAssetType.CORP_JUNIOR.value: 3,
            BondAssetType.ABS_CLO.value: 4
        }
        
        # HQLA scores (higher is worse)
        self.hqla_scores = {
            HQLAStatus.LEVEL_1.value: 1,
            HQLAStatus.LEVEL_2A.value: 2,
            HQLAStatus.LEVEL_2B.value: 3,
            HQLAStatus.NON_HQLA.value: 4
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('CollateralSwapEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _calculate_market_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market value as notional * price"""
        df = df.copy()
        df['market_value'] = df['notional'] * df['market_price']
        return df
    
    def _calculate_directional_market_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate directional market value based on trade type"""
        df = df.copy()
        
        # Define directional multipliers
        direction_map = {
            TradeType.BOND_BORROW.value: -1,
            TradeType.BOND_LEND.value: 1,
            TradeType.REPO.value: 1,
            TradeType.REVERSE_REPO.value: -1,
            TradeType.TRI_PARTY_REPO.value: 1,
            TradeType.TRI_PARTY_REVERSE.value: -1,
            TradeType.TRI_PARTY_REPO_ALT.value: 1,
            TradeType.TRI_PARTY_REVERSE_ALT.value: -1
        }
        
        df['direction_multiplier'] = df['trade_type'].map(direction_map).fillna(0)
        df['directional_market_value'] = df['market_value'] * df['direction_multiplier']
        
        return df
    
    def _create_synthetic_key(self, df: pd.DataFrame, approach: ApproachType) -> pd.DataFrame:
        """Create synthetic key based on approach"""
        df = df.copy()
        
        if approach == ApproachType.DESK_LOGIC:
            df['synthetic_key'] = df['counterparty_code'].astype(str) + '_' + \
                                df['maturity_date'].astype(str)
        
        elif approach == ApproachType.CONTRACT_BASED:
            df['synthetic_key'] = df['counterparty_code'].astype(str) + '_' + \
                                df['maturity_date'].astype(str) + '_' + \
                                df['contract_id'].astype(str) + '_' + \
                                df['trade_type'].astype(str)
        
        elif approach == ApproachType.FLAG_BASED:
            df['synthetic_key'] = df['counterparty_code'].astype(str) + '_' + \
                                df['maturity_date'].astype(str)
        
        return df
    
    def _identify_collateral_swaps_desk_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Approach 1: Desk logic identification"""
        self.logger.info("Applying Desk Logic approach")
        
        df = df.copy()
        df['collateral_swap_indicator_desk'] = False
        df['trade_group_synthetic_key_desk'] = ''
        df['collateral_swap_id_desk'] = 0
        
        # Filter for valid trade types
        valid_trade_types = [tt.value for tt in TradeType]
        df_valid = df[df['trade_type'].isin(valid_trade_types)].copy()
        
        if df_valid.empty:
            self.logger.warning("No valid trade types found for desk logic approach")
            return df
        
        # Create synthetic key for grouping
        df_valid = self._create_synthetic_key(df_valid, ApproachType.DESK_LOGIC)
        
        # Apply waterfall logic
        for key, group in df_valid.groupby('synthetic_key'):
            trade_types = set(group['trade_type'].unique())
            
            # Check if trade types form valid combination
            valid_combination = any(
                trade_types.issubset(combo) or combo.issubset(trade_types)
                for combo in self.valid_combinations
            )
            
            if not valid_combination:
                continue
            
            # Rule 1: Book A/B + Cpty A
            book_cpty_condition = (
                group['booking_system'].isin([self.config.books.book_a, self.config.books.book_b]) &
                (group['counterparty_code'] == self.config.books.cpty_a)
            )
            
            if book_cpty_condition.any():
                indices = group.index
                df.loc[indices, 'collateral_swap_indicator_desk'] = True
                df.loc[indices, 'trade_group_synthetic_key_desk'] = key
                df.loc[indices, 'collateral_swap_id_desk'] = self._get_next_group_id()
                continue
            
            # Rule 2: Cpty B + Bond Borrow/Lend
            cpty_b_condition = (
                (group['counterparty_code'] == self.config.books.cpty_b) &
                group['trade_type'].isin([TradeType.BOND_BORROW.value, TradeType.BOND_LEND.value])
            )
            
            if cpty_b_condition.any():
                indices = group.index
                df.loc[indices, 'collateral_swap_indicator_desk'] = True
                df.loc[indices, 'trade_group_synthetic_key_desk'] = key
                df.loc[indices, 'collateral_swap_id_desk'] = self._get_next_group_id()
                continue
            
            # Rule 3: Directional market value threshold
            total_directional_value = group['directional_market_value'].sum()
            if abs(total_directional_value) <= self.config.thresholds.market_value_threshold:
                indices = group.index
                df.loc[indices, 'collateral_swap_indicator_desk'] = True
                df.loc[indices, 'trade_group_synthetic_key_desk'] = key
                df.loc[indices, 'collateral_swap_id_desk'] = self._get_next_group_id()
        
        return df
    
    def _identify_collateral_swaps_contract_based(self, df: pd.DataFrame) -> pd.DataFrame:
        """Approach 2: Contract-based identification"""
        self.logger.info("Applying Contract-based approach")
        
        df = df.copy()
        df['collateral_swap_indicator_contract'] = False
        df['trade_group_synthetic_key_contract'] = ''
        df['collateral_swap_id_contract'] = 0
        
        # Create synthetic key
        df = self._create_synthetic_key(df, ApproachType.CONTRACT_BASED)
        
        # Group by synthetic key and check threshold
        for key, group in df.groupby('synthetic_key'):
            total_directional_value = group['directional_market_value'].sum()
            
            if abs(total_directional_value) <= self.config.thresholds.market_value_threshold:
                indices = group.index
                df.loc[indices, 'collateral_swap_indicator_contract'] = True
                df.loc[indices, 'trade_group_synthetic_key_contract'] = key
                df.loc[indices, 'collateral_swap_id_contract'] = self._get_next_group_id()
        
        return df
    
    def _identify_collateral_swaps_flag_based(self, df: pd.DataFrame) -> pd.DataFrame:
        """Approach 3: Flag-based identification"""
        self.logger.info("Applying Flag-based approach")
        
        df = df.copy()
        df['collateral_swap_indicator_flag'] = False
        df['trade_group_synthetic_key_flag'] = ''
        df['collateral_swap_id_flag'] = 0
        
        # Create synthetic key
        df = self._create_synthetic_key(df, ApproachType.FLAG_BASED)
        
        # Apply flag-based logic
        repo_types = [TradeType.REPO.value, TradeType.REVERSE_REPO.value, 
                     TradeType.TRI_PARTY_REPO.value, TradeType.TRI_PARTY_REVERSE.value,
                     TradeType.TRI_PARTY_REPO_ALT.value, TradeType.TRI_PARTY_REVERSE_ALT.value]
        
        bond_types = [TradeType.BOND_BORROW.value, TradeType.BOND_LEND.value]
        
        # Group by synthetic key
        for key, group in df.groupby('synthetic_key'):
            # Rule 1: Repo types with collateral swap flag
            repo_condition = (
                group['trade_type'].isin(repo_types) &
                (group['collateral_swaps_flag'] == True)
            )
            
            # Rule 2: Bond types with secured flag
            bond_condition = (
                group['trade_type'].isin(bond_types) &
                (group['is_secured_flag'] == True)
            )
            
            if repo_condition.any() or bond_condition.any():
                indices = group.index
                df.loc[indices, 'collateral_swap_indicator_flag'] = True
                df.loc[indices, 'trade_group_synthetic_key_flag'] = key
                df.loc[indices, 'collateral_swap_id_flag'] = self._get_next_group_id()
        
        return df
    
    def _get_next_group_id(self) -> int:
        """Get next sequential group ID"""
        self.group_counter += 1
        return self.group_counter
    
    def _calculate_worst_rating_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate worst rating score from S&P, Moody's, Fitch"""
        df = df.copy()
        
        if self.config.hierarchy.use_worst_rating_field and 'worst_bond_rating' in df.columns:
            df['rating_score'] = df['worst_bond_rating'].map(self.rating_scores).fillna(22)
        else:
            # Calculate worst rating from individual ratings
            rating_cols = ['sp_rating', 'moody_rating', 'fitch_rating']
            available_cols = [col for col in rating_cols if col in df.columns]
            
            if available_cols:
                df['rating_score'] = df[available_cols].apply(
                    lambda row: max([self.rating_scores.get(rating, 22) 
                                   for rating in row if pd.notna(rating)] or [22]),
                    axis=1
                )
            else:
                df['rating_score'] = 22  # Default worst score
        
        return df
    
    def _calculate_combined_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined hierarchy score"""
        df = df.copy()
        df['combined_score'] = 0
        
        # HQLA score
        if self.config.hierarchy.hqla_enabled and 'hqla_status' in df.columns:
            df['hqla_score'] = df['hqla_status'].map(self.hqla_scores).fillna(4)
            df['combined_score'] += df['hqla_score']
        
        # Rating score
        if self.config.hierarchy.rating_enabled:
            df = self._calculate_worst_rating_score(df)
            df['combined_score'] += df['rating_score']
        
        # Asset type score
        if self.config.hierarchy.asset_type_enabled and 'bond_asset_type' in df.columns:
            df['asset_type_score'] = df['bond_asset_type'].map(self.asset_type_scores).fillna(4)
            df['combined_score'] += df['asset_type_score']
        
        return df
    
    def _classify_collateral_swap_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify collateral swap type (upgrade/downgrade/neutral)"""
        df = df.copy()
        
        # Calculate combined scores
        df = self._calculate_combined_score(df)
        
        # Determine collateral direction (received or given)
        receive_types = [TradeType.REVERSE_REPO.value, TradeType.BOND_BORROW.value,
                        TradeType.TRI_PARTY_REVERSE.value, TradeType.TRI_PARTY_REVERSE_ALT.value]
        
        df['collateral_received'] = df['trade_type'].isin(receive_types)
        
        # Initialize swap type columns for each approach
        for approach in ['desk', 'contract', 'flag']:
            df[f'collateral_swap_type_{approach}'] = SwapType.NEUTRAL.value
        
        # Group by collateral swap groups and classify
        approach_map = {
            'desk': ApproachType.DESK_LOGIC,
            'contract': ApproachType.CONTRACT_BASED,
            'flag': ApproachType.FLAG_BASED
        }
        
        for approach in ['desk', 'contract', 'flag']:
            if not self.config.approaches[approach_map[approach]].enabled:
                continue
                
            swap_id_col = f'collateral_swap_id_{approach}'
            swap_indicator_col = f'collateral_swap_indicator_{approach}'
            swap_type_col = f'collateral_swap_type_{approach}'
            
            # Only process identified swaps
            swap_df = df[df[swap_indicator_col] == True].copy()
            
            if swap_df.empty:
                continue
            
            for swap_id, group in swap_df.groupby(swap_id_col):
                if swap_id == 0:  # Skip unidentified swaps
                    continue
                
                # Separate received and given collateral
                received = group[group['collateral_received'] == True]
                given = group[group['collateral_received'] == False]
                
                if received.empty or given.empty:
                    continue
                
                # Compare average scores
                avg_received_score = received['combined_score'].mean()
                avg_given_score = given['combined_score'].mean()
                
                # Classify based on score comparison
                if avg_given_score < avg_received_score:  # Better collateral given
                    swap_type = SwapType.DOWNGRADE.value
                elif avg_given_score > avg_received_score:  # Worse collateral given
                    swap_type = SwapType.UPGRADE.value
                else:
                    swap_type = SwapType.NEUTRAL.value
                
                # Update the dataframe
                indices = group.index
                df.loc[indices, swap_type_col] = swap_type
        
        return df
    
    def _classify_transaction_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify transaction type (financing/funding/collateral swap)"""
        df = df.copy()
        
        # Initialize transaction type columns for each approach
        for approach in ['desk', 'contract', 'flag']:
            df[f'transaction_type_{approach}'] = TransactionType.FINANCING.value
        
        # Classify based on each approach
        approach_map = {
            'desk': ApproachType.DESK_LOGIC,
            'contract': ApproachType.CONTRACT_BASED,
            'flag': ApproachType.FLAG_BASED
        }
        
        for approach in ['desk', 'contract', 'flag']:
            if not self.config.approaches[approach_map[approach]].enabled:
                continue
                
            swap_indicator_col = f'collateral_swap_indicator_{approach}'
            transaction_type_col = f'transaction_type_{approach}'
            
            # Collateral swaps
            df.loc[df[swap_indicator_col] == True, transaction_type_col] = TransactionType.COLLATERAL_SWAP.value
            
            # Unsecured bond borrow/lend
            unsecured_condition = (
                (df['is_secured_flag'] != True) &
                df['trade_type'].isin([TradeType.BOND_BORROW.value, TradeType.BOND_LEND.value]) &
                (df[swap_indicator_col] != True)
            )
            df.loc[unsecured_condition, transaction_type_col] = TransactionType.UNSECURED_BORROW_LEND.value
            
            # Repos (funding/financing)
            repo_condition = (
                df['trade_type'].isin([
                    TradeType.REPO.value, TradeType.REVERSE_REPO.value,
                    TradeType.TRI_PARTY_REPO.value, TradeType.TRI_PARTY_REVERSE.value,
                    TradeType.TRI_PARTY_REPO_ALT.value, TradeType.TRI_PARTY_REVERSE_ALT.value
                ]) &
                (df[swap_indicator_col] != True)
            )
            
            # Distinguish between funding and financing based on trade type
            funding_types = [TradeType.REPO.value, TradeType.TRI_PARTY_REPO.value, 
                           TradeType.TRI_PARTY_REPO_ALT.value]
            
            df.loc[repo_condition & df['trade_type'].isin(funding_types), 
                   transaction_type_col] = TransactionType.FUNDING.value
            
            df.loc[repo_condition & ~df['trade_type'].isin(funding_types), 
                   transaction_type_col] = TransactionType.FINANCING.value
        
        return df
    
    def _create_collateral_swap_construction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create collateral swap construction description"""
        df = df.copy()
        
        # Classify based on each approach
        approach_map = {
            'desk': ApproachType.DESK_LOGIC,
            'contract': ApproachType.CONTRACT_BASED,
            'flag': ApproachType.FLAG_BASED
        }
        
        for approach in ['desk', 'contract', 'flag']:
            approach_type = approach_map[approach]
            if not self.config.approaches[approach_type].enabled:
                continue
                
            swap_id_col = f'collateral_swap_id_{approach}'
            swap_indicator_col = f'collateral_swap_indicator_{approach}'
            construction_col = f'collateral_swap_construction_{approach}'
            
            df[construction_col] = ''
            
            # Only process identified swaps
            swap_df = df[df[swap_indicator_col] == True].copy()
            
            if swap_df.empty:
                continue
            
            for swap_id, group in swap_df.groupby(swap_id_col):
                if swap_id == 0:
                    continue
                
                # Count trade types in the group
                trade_type_counts = group['trade_type'].value_counts()
                construction_parts = []
                
                for trade_type, count in trade_type_counts.items():
                    if count == 1:
                        construction_parts.append(trade_type)
                    else:
                        construction_parts.append(f"{count}{trade_type}")
                
                construction = " + ".join(sorted(construction_parts))
                
                # Update the dataframe
                indices = group.index
                df.loc[indices, construction_col] = construction
        
        return df
    
    def process_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to process trades and identify collateral swaps"""
        self.logger.info("Starting collateral swap identification process")
        
        # Validate required columns
        required_cols = [
            'trade_id', 'booking_system', 'trade_type', 'counterparty', 
            'counterparty_code', 'notional', 'currency', 'market_price', 
            'maturity_date', 'contract_id'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Initialize result dataframe
        result_df = df.copy()
        
        # Calculate market values
        result_df = self._calculate_market_value(result_df)
        result_df = self._calculate_directional_market_value(result_df)
        
        # Apply enabled identification approaches
        if self.config.approaches[ApproachType.DESK_LOGIC].enabled:
            result_df = self._identify_collateral_swaps_desk_logic(result_df)
        
        if self.config.approaches[ApproachType.CONTRACT_BASED].enabled:
            result_df = self._identify_collateral_swaps_contract_based(result_df)
        
        if self.config.approaches[ApproachType.FLAG_BASED].enabled:
            result_df = self._identify_collateral_swaps_flag_based(result_df)
        
        # Classify collateral swap types
        result_df = self._classify_collateral_swap_type(result_df)
        
        # Classify transaction types
        result_df = self._classify_transaction_type(result_df)
        
        # Create construction descriptions
        result_df = self._create_collateral_swap_construction(result_df)
        
        # Generate audit information
        self._generate_audit_data(result_df)
        
        self.logger.info("Collateral swap identification process completed")
        
        return result_df
    
    def _generate_audit_data(self, df: pd.DataFrame):
        """Generate audit data for analysis"""
        audit_entry = {
            'timestamp': datetime.now(),
            'total_trades': len(df),
            'approaches_applied': [
                approach.value for approach, config in self.config.approaches.items()
                if config.enabled
            ]
        }
        
        # Count swaps identified by each approach
        for approach in ['desk', 'contract', 'flag']:
            if f'collateral_swap_indicator_{approach}' in df.columns:
                swap_count = df[f'collateral_swap_indicator_{approach}'].sum()
                audit_entry[f'{approach}_swaps_identified'] = swap_count
        
        self.audit_data.append(audit_entry)
    
    def get_audit_summary(self) -> pd.DataFrame:
        """Return audit data as DataFrame"""
        return pd.DataFrame(self.audit_data)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Return configuration summary"""
        return {
            'approaches': {
                approach.value: config._asdict() 
                for approach, config in self.config.approaches.items()
            },
            'hierarchy': self.config.hierarchy._asdict(),
            'thresholds': self.config.thresholds._asdict(),
            'books': self.config.books._asdict(),
            'use_original_flags': self.config.use_original_flags
        }

# Usage Example and Dummy Data Generator
def generate_dummy_data(num_trades: int = 100) -> pd.DataFrame:
    """Generate dummy trade data for testing"""
    np.random.seed(42)  # For reproducible results
    
    # Trade type probabilities (ensure they sum to 1)
    trade_types = ['BB', 'BL', 'REP', 'REV', 'TP', 'TR', 'TPR', 'TRV']
    trade_type_probs = [0.15, 0.15, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05]
    
    # Rating options
    ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 
               'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-']
    rating_probs = [0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12,
                    0.08, 0.08, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0.01]
    
    # Ensure probabilities sum to 1
    rating_probs = np.array(rating_probs)
    rating_probs = rating_probs / rating_probs.sum()
    
    # HQLA statuses
    hqla_statuses = ['LEVEL_1', 'LEVEL_2A', 'LEVEL_2B', 'NON_HQLA']
    hqla_probs = [0.30, 0.25, 0.25, 0.20]
    
    # Bond asset types
    asset_types = ['GOVT', 'CORP_SENIOR', 'CORP_JUNIOR', 'ABS_CLO']
    asset_type_probs = [0.40, 0.35, 0.20, 0.05]
    
    # Generate data
    data = {
        'trade_id': [f'T{str(i+1).zfill(6)}' for i in range(num_trades)],
        'booking_system': np.random.choice(['MAG', 'OTHER'], num_trades, p=[0.8, 0.2]),
        'trade_type': np.random.choice(trade_types, num_trades, p=trade_type_probs),
        'counterparty': np.random.choice([f'CP{i}' for i in range(1, 21)], num_trades),
        'counterparty_code': np.random.choice([f'CP{i}' for i in range(1, 21)], num_trades),
        'notional': np.random.uniform(100000, 10000000, num_trades),
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY'], num_trades, p=[0.6, 0.2, 0.1, 0.1]),
        'market_price': np.random.uniform(0.95, 1.05, num_trades),
        'maturity_date': np.random.choice(pd.date_range('2025-08-01', '2026-12-31', freq='D').strftime('%Y-%m-%d'), num_trades),
        'contract_id': [f'C{str(np.random.randint(1, 1001)).zfill(4)}' for _ in range(num_trades)],
        'collateral_swaps_flag': np.random.choice([True, False], num_trades, p=[0.3, 0.7]),
        'is_secured_flag': np.random.choice([True, False], num_trades, p=[0.6, 0.4]),
        'hqla_status': np.random.choice(hqla_statuses, num_trades, p=hqla_probs),
        'bond_asset_type': np.random.choice(asset_types, num_trades, p=asset_type_probs),
        'sp_rating': np.random.choice(ratings, num_trades, p=rating_probs),
        'moody_rating': np.random.choice(ratings, num_trades, p=rating_probs),
        'fitch_rating': np.random.choice(ratings, num_trades, p=rating_probs)
    }
    
    df = pd.DataFrame(data)
    
    # Create some realistic collateral swap pairs
    # Generate paired trades with same counterparty and maturity
    swap_pairs = []
    for i in range(0, min(40, num_trades//3), 2):  # Create up to 20 pairs
        if i+1 < len(df):
            # Make a BB/BL pair
            base_idx = i
            pair_idx = i + 1
            
            # Same counterparty and maturity
            df.iloc[pair_idx, df.columns.get_loc('counterparty')] = df.iloc[base_idx]['counterparty']
            df.iloc[pair_idx, df.columns.get_loc('counterparty_code')] = df.iloc[base_idx]['counterparty_code']
            df.iloc[pair_idx, df.columns.get_loc('maturity_date')] = df.iloc[base_idx]['maturity_date']
            df.iloc[pair_idx, df.columns.get_loc('contract_id')] = df.iloc[base_idx]['contract_id']
            
            # Make complementary trade types
            if df.iloc[base_idx]['trade_type'] == 'BB':
                df.iloc[pair_idx, df.columns.get_loc('trade_type')] = 'BL'
            elif df.iloc[base_idx]['trade_type'] == 'BL':
                df.iloc[pair_idx, df.columns.get_loc('trade_type')] = 'BB'
            elif df.iloc[base_idx]['trade_type'] == 'REP':
                df.iloc[pair_idx, df.columns.get_loc('trade_type')] = 'REV'
            elif df.iloc[base_idx]['trade_type'] == 'REV':
                df.iloc[pair_idx, df.columns.get_loc('trade_type')] = 'REP'
            
            # Similar notionals for better matching
            df.iloc[pair_idx, df.columns.get_loc('notional')] = df.iloc[base_idx]['notional'] * np.random.uniform(0.95, 1.05)
            
            # Set flags for collateral swaps
            df.iloc[base_idx, df.columns.get_loc('collateral_swaps_flag')] = True
            df.iloc[pair_idx, df.columns.get_loc('collateral_swaps_flag')] = True
            df.iloc[base_idx, df.columns.get_loc('is_secured_flag')] = True
            df.iloc[pair_idx, df.columns.get_loc('is_secured_flag')] = True
    
    return df

if __name__ == "__main__":
    # Generate dummy data
    print("Generating dummy trade data...")
    sample_data = generate_dummy_data(200)
    
    # Create and configure the engine
    config = CollateralSwapConfig()
    
    # Enable all approaches
    config.enable_approach(ApproachType.DESK_LOGIC, True)
    config.enable_approach(ApproachType.CONTRACT_BASED, False)
    config.enable_approach(ApproachType.FLAG_BASED, False)
    
    # Create engine
    engine = CollateralSwapEngine(config)
    
    # Process trades
    print("Processing trades...")
    result_df = engine.process_trades(sample_data)
    
    # Display results
    print(f"\nProcessed {len(result_df)} trades")
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"Result columns: {len(result_df.columns)}")
    
    # Show some statistics
    print("\n=== RESULTS SUMMARY ===")
    
    for approach in ['desk', 'contract', 'flag']:
        if f'collateral_swap_indicator_{approach}' in result_df.columns:
            swap_count = result_df[f'collateral_swap_indicator_{approach}'].sum()
            print(f"{approach.title()} Approach - Collateral Swaps Identified: {swap_count}")
            
            if swap_count > 0:
                swap_types = result_df[result_df[f'collateral_swap_indicator_{approach}'] == True][f'collateral_swap_type_{approach}'].value_counts()
                print(f"  Swap Types: {dict(swap_types)}")
                
                transaction_types = result_df[result_df[f'collateral_swap_indicator_{approach}'] == True][f'transaction_type_{approach}'].value_counts()
                print(f"  Transaction Types: {dict(transaction_types)}")
    
    # Show sample results
    print("\n=== SAMPLE RESULTS ===")
    display_cols = ['trade_id', 'trade_type', 'counterparty_code', 'notional', 'maturity_date']
    
    # Add result columns that exist
    for approach in ['desk', 'contract', 'flag']:
        result_cols = [f'collateral_swap_indicator_{approach}', f'collateral_swap_type_{approach}', 
                      f'transaction_type_{approach}']
        for col in result_cols:
            if col in result_df.columns:
                display_cols.append(col)
    
    # Show first few identified swaps
    for approach in ['desk', 'contract', 'flag']:
        if f'collateral_swap_indicator_{approach}' in result_df.columns:
            swaps = result_df[result_df[f'collateral_swap_indicator_{approach}'] == True]
            if len(swaps) > 0:
                print(f"\n{approach.title()} Approach - Sample Identified Swaps:")
                print(swaps[display_cols].head().to_string(index=False))
                break
    
    # Show audit summary
    print("\n=== AUDIT SUMMARY ===")
    audit_df = engine.get_audit_summary()
    if not audit_df.empty:
        print(audit_df.to_string(index=False))
    
    # Show configuration
    print("\n=== CONFIGURATION ===")
    config_summary = engine.get_config_summary()
    for key, value in config_summary.items():
        print(f"{key}: {value}")
    
    print("\nCollateral Swap Engine test completed successfully!")
