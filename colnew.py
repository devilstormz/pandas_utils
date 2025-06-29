import pandas as pd
from enum import Enum
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import uuid

# =============================================================================
# ENUMS AND GLOBAL VARIABLES
# =============================================================================

class TradeColumns(Enum):
    """Enum for trade dataframe columns"""
    TRADE_ID = 'trade_id'
    BOOKING_SYSTEM = 'booking_system'
    COUNTERPARTY = 'cpty'
    NOTIONAL = 'notional'
    CURRENCY = 'currency'
    START_DATE = 'start_date'
    MATURITY_DATE = 'maturity_date'
    TRADE_TYPE = 'trade_type'
    ISIN = 'isin'
    BOND_ISSUER = 'bond_issuer'
    BOND_RATING = 'bond_rating'
    REPO_RATE = 'repo_rate'

class TradeType(Enum):
    """Enum for trade types"""
    BOND_BORROW = 'Bond Borrow'
    BOND_LEND = 'Bond Lend'
    REPO = 'Repo'
    REVERSE_REPO = 'Reverse Repo'

class CollateralSwapType(Enum):
    """Enum for collateral swap classification"""
    UPGRADE = 'Upgrade'
    DOWNGRADE = 'Downgrade'
    NEUTRAL = 'Neutral'

class BondCategory(Enum):
    """Bond category hierarchy (higher number = higher quality)"""
    ABS = 1
    CORP = 2
    GOVT = 3

class SwapColumns(Enum):
    """Enum for swap result columns"""
    SWAP_ID = 'swap_id'
    SWAP_TYPE = 'collateral_swap_type'
    IS_MATCHED = 'is_matched'
    MATCH_CONFIDENCE = 'match_confidence'

# Global configuration
RATING_HIERARCHY = {
    'AAA': 10, 'AA+': 9, 'AA': 8, 'AA-': 7,
    'A+': 6, 'A': 5, 'A-': 4,
    'BBB+': 3, 'BBB': 2, 'BBB-': 1,
    'BB+': 0, 'BB': -1, 'BB-': -2,
    'B+': -3, 'B': -4, 'B-': -5,
    'CCC+': -6, 'CCC': -7, 'CCC-': -8,
    'CC': -9, 'C': -10, 'D': -11
}

BOND_CATEGORY_MAPPING = {
    'Government': BondCategory.GOVT,
    'Corporate': BondCategory.CORP,
    'Asset-Backed': BondCategory.ABS,
    'Municipal': BondCategory.GOVT,
    'Agency': BondCategory.GOVT
}

# Matching tolerances
MATCHING_CONFIG = {
    'notional_tolerance': 0.01,  # 1% tolerance
    'date_tolerance_days': 1,    # 1 day tolerance
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class MatchingCriteria:
    """Criteria for matching trades"""
    def __init__(self, counterparty, notional, currency, start_date, maturity_date, 
                 notional_tolerance=None, date_tolerance_days=None):
        self.counterparty = counterparty
        self.notional = notional
        self.currency = currency
        self.start_date = start_date
        self.maturity_date = maturity_date
        self.notional_tolerance = notional_tolerance or MATCHING_CONFIG['notional_tolerance']
        self.date_tolerance_days = date_tolerance_days or MATCHING_CONFIG['date_tolerance_days']

class CollateralSwap:
    """Represents a matched collateral swap"""
    def __init__(self, swap_id, trade_1, trade_2, swap_type, construction_method, match_confidence):
        self.swap_id = swap_id
        self.trade_1 = trade_1
        self.trade_2 = trade_2
        self.swap_type = swap_type
        self.construction_method = construction_method
        self.match_confidence = match_confidence

# =============================================================================
# COLLATERAL SWAP MATCHING ENGINE
# =============================================================================

class CollateralSwapMatchingEngine:
    """
    Matching engine to identify and classify collateral swaps
    """
    
    def __init__(self, use_notional_tolerance=True, use_date_tolerance=True, 
                 use_hierarchy_ranking=True, notional_tolerance=None, date_tolerance_days=None):
        """
        Initialize the matching engine with configurable options
        
        Args:
            use_notional_tolerance (bool): Enable/disable notional tolerance matching
            use_date_tolerance (bool): Enable/disable date tolerance matching  
            use_hierarchy_ranking (bool): Enable/disable bond category hierarchy in scoring
            notional_tolerance (float): Custom notional tolerance (overrides default)
            date_tolerance_days (int): Custom date tolerance in days (overrides default)
        """
        self.matched_swaps = []
        self.unmatched_trades = []
        
        # Configuration flags
        self.use_notional_tolerance = use_notional_tolerance
        self.use_date_tolerance = use_date_tolerance
        self.use_hierarchy_ranking = use_hierarchy_ranking
        
        # Tolerance settings
        self.notional_tolerance = notional_tolerance or MATCHING_CONFIG['notional_tolerance']
        self.date_tolerance_days = date_tolerance_days or MATCHING_CONFIG['date_tolerance_days']
        
    def _calculate_bond_score(self, bond_category, rating):
        """Calculate composite score for bond quality"""
        rating_score = RATING_HIERARCHY.get(rating, -11)
        
        if self.use_hierarchy_ranking:
            category_score = BOND_CATEGORY_MAPPING.get(bond_category, BondCategory.ABS).value * 100
            return category_score + rating_score
        else:
            # Use only rating-based scoring when hierarchy is disabled
            return rating_score
    
    def _determine_swap_type(self, trade_1, trade_2):
        """Determine if swap is upgrade, downgrade, or neutral"""
        
        # Calculate bond scores for both sides
        score_1 = self._calculate_bond_score(trade_1['bond_category'], trade_1[TradeColumns.BOND_RATING.value])
        score_2 = self._calculate_bond_score(trade_2['bond_category'], trade_2[TradeColumns.BOND_RATING.value])
        
        # Determine direction based on trade types
        if ((trade_1[TradeColumns.TRADE_TYPE.value] in [TradeType.BOND_BORROW.value, TradeType.REVERSE_REPO.value] and
             trade_2[TradeColumns.TRADE_TYPE.value] in [TradeType.BOND_LEND.value, TradeType.REPO.value]) or
            (trade_1[TradeColumns.TRADE_TYPE.value] in [TradeType.BOND_LEND.value, TradeType.REPO.value] and
             trade_2[TradeColumns.TRADE_TYPE.value] in [TradeType.BOND_BORROW.value, TradeType.REVERSE_REPO.value])):
            
            # If borrowing/receiving higher quality and lending/giving lower quality = upgrade
            borrow_score = score_1 if trade_1[TradeColumns.TRADE_TYPE.value] in [TradeType.BOND_BORROW.value, TradeType.REVERSE_REPO.value] else score_2
            lend_score = score_2 if trade_1[TradeColumns.TRADE_TYPE.value] in [TradeType.BOND_BORROW.value, TradeType.REVERSE_REPO.value] else score_1
            
            if borrow_score > lend_score:
                return CollateralSwapType.UPGRADE
            elif borrow_score < lend_score:
                return CollateralSwapType.DOWNGRADE
            else:
                return CollateralSwapType.NEUTRAL
        
        return CollateralSwapType.NEUTRAL
    
    def _get_construction_method(self, trade_1, trade_2):
        """Determine how the swap was constructed"""
        types = sorted([trade_1[TradeColumns.TRADE_TYPE.value], trade_2[TradeColumns.TRADE_TYPE.value]])
        
        if TradeType.BOND_BORROW.value in types and TradeType.BOND_LEND.value in types:
            return "Bond Borrow + Bond Lend"
        elif TradeType.REPO.value in types and TradeType.REVERSE_REPO.value in types:
            return "Repo + Reverse Repo"
        else:
            return "Mixed: " + " + ".join(types)
    
    def _calculate_match_confidence(self, trade_1, trade_2, criteria):
        """Calculate confidence score for the match"""
        confidence = 1.0
        
        # Notional difference penalty
        notional_diff = abs(trade_1[TradeColumns.NOTIONAL.value] - trade_2[TradeColumns.NOTIONAL.value]) / trade_1[TradeColumns.NOTIONAL.value]
        confidence -= notional_diff * 0.5
        
        # Date difference penalty
        start_diff = abs((trade_1[TradeColumns.START_DATE.value] - trade_2[TradeColumns.START_DATE.value]).days)
        maturity_diff = abs((trade_1[TradeColumns.MATURITY_DATE.value] - trade_2[TradeColumns.MATURITY_DATE.value]).days)
        date_penalty = (start_diff + maturity_diff) * 0.01
        confidence -= date_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _trades_match(self, trade_1, trade_2, criteria):
        """Check if two trades match based on criteria"""
        
        # Same counterparty
        if trade_1[TradeColumns.COUNTERPARTY.value] != trade_2[TradeColumns.COUNTERPARTY.value]:
            return False
        
        # Same currency
        if trade_1[TradeColumns.CURRENCY.value] != trade_2[TradeColumns.CURRENCY.value]:
            return False
        
        # Notional matching with optional tolerance
        if self.use_notional_tolerance:
            notional_diff = abs(trade_1[TradeColumns.NOTIONAL.value] - trade_2[TradeColumns.NOTIONAL.value])
            notional_tolerance = trade_1[TradeColumns.NOTIONAL.value] * self.notional_tolerance
            if notional_diff > notional_tolerance:
                return False
        else:
            # Exact notional match required
            if trade_1[TradeColumns.NOTIONAL.value] != trade_2[TradeColumns.NOTIONAL.value]:
                return False
        
        # Date matching with optional tolerance
        if self.use_date_tolerance:
            start_diff = abs((trade_1[TradeColumns.START_DATE.value] - trade_2[TradeColumns.START_DATE.value]).days)
            maturity_diff = abs((trade_1[TradeColumns.MATURITY_DATE.value] - trade_2[TradeColumns.MATURITY_DATE.value]).days)
            
            if start_diff > self.date_tolerance_days or maturity_diff > self.date_tolerance_days:
                return False
        else:
            # Exact date match required
            if (trade_1[TradeColumns.START_DATE.value] != trade_2[TradeColumns.START_DATE.value] or
                trade_1[TradeColumns.MATURITY_DATE.value] != trade_2[TradeColumns.MATURITY_DATE.value]):
                return False
        
        # Complementary trade types
        type_1 = trade_1[TradeColumns.TRADE_TYPE.value]
        type_2 = trade_2[TradeColumns.TRADE_TYPE.value]
        
        valid_combinations = [
            (TradeType.BOND_BORROW.value, TradeType.BOND_LEND.value),
            (TradeType.BOND_LEND.value, TradeType.BOND_BORROW.value),
            (TradeType.REPO.value, TradeType.REVERSE_REPO.value),
            (TradeType.REVERSE_REPO.value, TradeType.REPO.value),
        ]
        
        return (type_1, type_2) in valid_combinations
    
    def match_trades(self, trades_df):
        """
        Main method to match trades and identify collateral swaps
        """
        trades_list = trades_df.to_dict('records')
        matched_trade_ids = set()
        self.matched_swaps = []
        self.unmatched_trades = []
        
        # Add bond category based on issuer
        for trade in trades_list:
            issuer = trade[TradeColumns.BOND_ISSUER.value]
            if 'Government' in issuer or 'Treasury' in issuer:
                trade['bond_category'] = 'Government'
            elif 'Corporate' in issuer or 'Corp' in issuer:
                trade['bond_category'] = 'Corporate'
            else:
                trade['bond_category'] = 'Asset-Backed'
        
        # Try to match each trade with others
        for i, trade_1 in enumerate(trades_list):
            if trade_1[TradeColumns.TRADE_ID.value] in matched_trade_ids:
                continue
                
            for j, trade_2 in enumerate(trades_list[i+1:], i+1):
                if trade_2[TradeColumns.TRADE_ID.value] in matched_trade_ids:
                    continue
                
                criteria = MatchingCriteria(
                    counterparty=trade_1[TradeColumns.COUNTERPARTY.value],
                    notional=trade_1[TradeColumns.NOTIONAL.value],
                    currency=trade_1[TradeColumns.CURRENCY.value],
                    start_date=trade_1[TradeColumns.START_DATE.value],
                    maturity_date=trade_1[TradeColumns.MATURITY_DATE.value],
                    notional_tolerance=self.notional_tolerance,
                    date_tolerance_days=self.date_tolerance_days
                )
                
                if self._trades_match(trade_1, trade_2, criteria):
                    # Create collateral swap
                    swap_id = str(uuid.uuid4())[:8]
                    swap_type = self._determine_swap_type(trade_1, trade_2)
                    construction_method = self._get_construction_method(trade_1, trade_2)
                    match_confidence = self._calculate_match_confidence(trade_1, trade_2, criteria)
                    
                    swap = CollateralSwap(
                        swap_id=swap_id,
                        trade_1=trade_1,
                        trade_2=trade_2,
                        swap_type=swap_type,
                        construction_method=construction_method,
                        match_confidence=match_confidence
                    )
                    
                    self.matched_swaps.append(swap)
                    matched_trade_ids.add(trade_1[TradeColumns.TRADE_ID.value])
                    matched_trade_ids.add(trade_2[TradeColumns.TRADE_ID.value])
                    break
        
        # Collect unmatched trades
        for trade in trades_list:
            if trade[TradeColumns.TRADE_ID.value] not in matched_trade_ids:
                self.unmatched_trades.append(trade)
        
        # Create result dataframe
        return self._create_result_dataframe(trades_df)
    
    def _create_result_dataframe(self, original_df):
        """Create result dataframe with swap information"""
        result_df = original_df.copy()
        
        # Initialize new columns
        result_df[SwapColumns.SWAP_ID.value] = None
        result_df[SwapColumns.SWAP_TYPE.value] = None
        result_df[SwapColumns.IS_MATCHED.value] = False
        result_df[SwapColumns.MATCH_CONFIDENCE.value] = 0.0
        
        # Fill in swap information
        for swap in self.matched_swaps:
            trade_1_id = swap.trade_1[TradeColumns.TRADE_ID.value]
            trade_2_id = swap.trade_2[TradeColumns.TRADE_ID.value]
            
            # Update both trades in the swap
            for trade_id in [trade_1_id, trade_2_id]:
                mask = result_df[TradeColumns.TRADE_ID.value] == trade_id
                result_df.loc[mask, SwapColumns.SWAP_ID.value] = swap.swap_id
                result_df.loc[mask, SwapColumns.SWAP_TYPE.value] = swap.swap_type.value
                result_df.loc[mask, SwapColumns.IS_MATCHED.value] = True
                result_df.loc[mask, SwapColumns.MATCH_CONFIDENCE.value] = swap.match_confidence
        
        return result_df
    
    def get_unmatched_trades(self):
        """Return dataframe of unmatched trades"""
        if not self.unmatched_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.unmatched_trades)
    
    def create_transaction_summary(self):
        """Create summary dataframe of collateral swaps"""
        if not self.matched_swaps:
            return pd.DataFrame()
        
        summaries = []
        for swap in self.matched_swaps:
            trade_1 = swap.trade_1
            trade_2 = swap.trade_2
            
            # Determine which trade is borrowing/receiving vs lending/giving
            if trade_1[TradeColumns.TRADE_TYPE.value] in [TradeType.BOND_BORROW.value, TradeType.REVERSE_REPO.value]:
                receiving_trade = trade_1
                giving_trade = trade_2
            else:
                receiving_trade = trade_2
                giving_trade = trade_1
            
            summary = {
                'swap_id': swap.swap_id,
                'counterparty': trade_1[TradeColumns.COUNTERPARTY.value],
                'collateral_swap_type': swap.swap_type.value,
                'construction_method': swap.construction_method,
                'notional': trade_1[TradeColumns.NOTIONAL.value],
                'currency': trade_1[TradeColumns.CURRENCY.value],
                'start_date': trade_1[TradeColumns.START_DATE.value],
                'maturity_date': trade_1[TradeColumns.MATURITY_DATE.value],
                'receiving_isin': receiving_trade[TradeColumns.ISIN.value],
                'receiving_issuer': receiving_trade[TradeColumns.BOND_ISSUER.value],
                'receiving_rating': receiving_trade[TradeColumns.BOND_RATING.value],
                'giving_isin': giving_trade[TradeColumns.ISIN.value],
                'giving_issuer': giving_trade[TradeColumns.BOND_ISSUER.value],
                'giving_rating': giving_trade[TradeColumns.BOND_RATING.value],
                'match_confidence': swap.match_confidence
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)

# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def create_sample_data():
    """Create sample trade data for testing"""
    np.random.seed(42)
    
    counterparties = ['Goldman Sachs', 'JP Morgan', 'Deutsche Bank', 'Credit Suisse', 'UBS']
    currencies = ['USD', 'EUR', 'GBP']
    isins = [
        'US912828XG20', 'US912828XH12', 'US912828XI10',  # Government bonds
        'US459200HU26', 'US459200HV00', 'US459200HW82',  # Corporate bonds
        'US912810SD19', 'US912810SE91', 'US912810SF73'   # Asset-backed
    ]
    
    bond_issuers = [
        'US Treasury', 'US Treasury', 'US Treasury',
        'IBM Corporate', 'Microsoft Corporate', 'Apple Corporate',
        'ABS Issuer 1', 'ABS Issuer 2', 'ABS Issuer 3'
    ]
    
    ratings = ['AAA', 'AA+', 'AA', 'A+', 'A', 'BBB+', 'BBB']
    
    trades = []
    trade_id = 1
    
    # Create matched pairs
    for i in range(20):
        base_date = datetime.now() + timedelta(days=np.random.randint(1, 30))
        maturity_date = base_date + timedelta(days=np.random.randint(30, 365))
        notional = np.random.choice([1000000, 5000000, 10000000, 25000000])
        cpty = np.random.choice(counterparties)
        currency = np.random.choice(currencies)
        
        # Create a matching pair
        isin_1 = np.random.choice(isins[:3])  # Government
        isin_2 = np.random.choice(isins[3:6])  # Corporate
        
        issuer_1 = bond_issuers[isins.index(isin_1)]
        issuer_2 = bond_issuers[isins.index(isin_2)]
        
        # Trade 1 - Borrow/Reverse Repo
        trade_type_1 = np.random.choice([TradeType.BOND_BORROW.value, TradeType.REVERSE_REPO.value])
        trades.append({
            TradeColumns.TRADE_ID.value: f'T{trade_id:06d}',
            TradeColumns.BOOKING_SYSTEM.value: 'SYSTEM_A',
            TradeColumns.COUNTERPARTY.value: cpty,
            TradeColumns.NOTIONAL.value: notional,
            TradeColumns.CURRENCY.value: currency,
            TradeColumns.START_DATE.value: base_date,
            TradeColumns.MATURITY_DATE.value: maturity_date,
            TradeColumns.TRADE_TYPE.value: trade_type_1,
            TradeColumns.ISIN.value: isin_1,
            TradeColumns.BOND_ISSUER.value: issuer_1,
            TradeColumns.BOND_RATING.value: np.random.choice(ratings[:4]),  # Higher rating
            TradeColumns.REPO_RATE.value: np.random.uniform(1.0, 3.0)
        })
        trade_id += 1
        
        # Trade 2 - Lend/Repo (matching)
        trade_type_2 = TradeType.BOND_LEND.value if trade_type_1 == TradeType.BOND_BORROW.value else TradeType.REPO.value
        trades.append({
            TradeColumns.TRADE_ID.value: f'T{trade_id:06d}',
            TradeColumns.BOOKING_SYSTEM.value: 'SYSTEM_B',
            TradeColumns.COUNTERPARTY.value: cpty,
            TradeColumns.NOTIONAL.value: notional,
            TradeColumns.CURRENCY.value: currency,
            TradeColumns.START_DATE.value: base_date,
            TradeColumns.MATURITY_DATE.value: maturity_date,
            TradeColumns.TRADE_TYPE.value: trade_type_2,
            TradeColumns.ISIN.value: isin_2,
            TradeColumns.BOND_ISSUER.value: issuer_2,
            TradeColumns.BOND_RATING.value: np.random.choice(ratings[4:]),  # Lower rating
            TradeColumns.REPO_RATE.value: np.random.uniform(1.0, 3.0)
        })
        trade_id += 1
    
    # Add some unmatched trades
    for i in range(5):
        base_date = datetime.now() + timedelta(days=np.random.randint(1, 30))
        maturity_date = base_date + timedelta(days=np.random.randint(30, 365))
        
        trades.append({
            TradeColumns.TRADE_ID.value: f'T{trade_id:06d}',
            TradeColumns.BOOKING_SYSTEM.value: np.random.choice(['SYSTEM_A', 'SYSTEM_B']),
            TradeColumns.COUNTERPARTY.value: np.random.choice(counterparties),
            TradeColumns.NOTIONAL.value: np.random.choice([1000000, 5000000, 10000000]),
            TradeColumns.CURRENCY.value: np.random.choice(currencies),
            TradeColumns.START_DATE.value: base_date,
            TradeColumns.MATURITY_DATE.value: maturity_date,
            TradeColumns.TRADE_TYPE.value: np.random.choice([t.value for t in TradeType]),
            TradeColumns.ISIN.value: np.random.choice(isins),
            TradeColumns.BOND_ISSUER.value: bond_issuers[isins.index(np.random.choice(isins))],
            TradeColumns.BOND_RATING.value: np.random.choice(ratings),
            TradeColumns.REPO_RATE.value: np.random.uniform(1.0, 3.0)
        })
        trade_id += 1
    
    return pd.DataFrame(trades)

# =============================================================================
# MAIN EXECUTION AND EXPORT
# =============================================================================

def main():
    """Main function to demonstrate the collateral swap matching system"""
    
    # Create sample data
    print("Creating sample trade data...")
    trades_df = create_sample_data()
    print("Created {} sample trades".format(len(trades_df)))
    
    # Test different engine configurations
    print("\n" + "="*60)
    print("TESTING DIFFERENT ENGINE CONFIGURATIONS")
    print("="*60)
    
    # Configuration 1: Default (all features enabled)
    print("\n1. Default Configuration (all features enabled):")
    engine1 = CollateralSwapMatchingEngine()
    result1 = engine1.match_trades(trades_df)
    summary1 = engine1.create_transaction_summary()
    print("   - Matched swaps: {}".format(len(engine1.matched_swaps)))
    print("   - Unmatched trades: {}".format(len(engine1.unmatched_trades)))
    
    # Configuration 2: No tolerances (exact matching only)
    print("\n2. Exact Matching (no tolerances):")
    engine2 = CollateralSwapMatchingEngine(
        use_notional_tolerance=False, 
        use_date_tolerance=False
    )
    result2 = engine2.match_trades(trades_df)
    summary2 = engine2.create_transaction_summary()
    print("   - Matched swaps: {}".format(len(engine2.matched_swaps)))
    print("   - Unmatched trades: {}".format(len(engine2.unmatched_trades)))
    
    # Configuration 3: Rating-only classification (no hierarchy)
    print("\n3. Rating-Only Classification (no bond category hierarchy):")
    engine3 = CollateralSwapMatchingEngine(use_hierarchy_ranking=False)
    result3 = engine3.match_trades(trades_df)
    summary3 = engine3.create_transaction_summary()
    print("   - Matched swaps: {}".format(len(engine3.matched_swaps)))
    print("   - Unmatched trades: {}".format(len(engine3.unmatched_trades)))
    if not summary3.empty:
        print("   - Swap type distribution:")
        for swap_type, count in summary3['collateral_swap_type'].value_counts().items():
            print("     * {}: {}".format(swap_type, count))
    
    # Configuration 4: Custom tolerances
    print("\n4. Custom Tolerances (5% notional, 3 days):")
    engine4 = CollateralSwapMatchingEngine(
        notional_tolerance=0.05, 
        date_tolerance_days=3
    )
    result4 = engine4.match_trades(trades_df)
    summary4 = engine4.create_transaction_summary()
    print("   - Matched swaps: {}".format(len(engine4.matched_swaps)))
    print("   - Unmatched trades: {}".format(len(engine4.unmatched_trades)))
    
    # Use the default configuration for final export
    print("\n" + "="*60)
    print("EXPORTING RESULTS FROM DEFAULT CONFIGURATION")
    print("="*60)
    
    engine = engine1
    result_df = result1
    transaction_summary_df = summary1
    unmatched_df = engine.get_unmatched_trades()
    
    # Create summary statistics
    swap_count_by_cpty = result_df[result_df[SwapColumns.IS_MATCHED.value]].groupby(
        TradeColumns.COUNTERPARTY.value
    ).agg({
        SwapColumns.SWAP_ID.value: 'nunique',
        TradeColumns.NOTIONAL.value: 'sum'
    }).rename(columns={
        SwapColumns.SWAP_ID.value: 'swap_count',
        TradeColumns.NOTIONAL.value: 'total_notional'
    })
    
    swap_type_summary = transaction_summary_df.groupby('collateral_swap_type').agg({
        'swap_id': 'count',
        'notional': 'sum'
    }).rename(columns={'swap_id': 'count', 'notional': 'total_notional'})
    
    # Export to Excel
    print("\nExporting results to Excel...")
    with pd.ExcelWriter('collateral_swap_analysis.xlsx', engine='openpyxl') as writer:
        # Main results
        result_df.to_excel(writer, sheet_name='All_Trades_With_Swaps', index=False)
        
        # Transaction summary
        if not transaction_summary_df.empty:
            transaction_summary_df.to_excel(writer, sheet_name='Swap_Summary', index=False)
        
        # Unmatched trades
        if not unmatched_df.empty:
            unmatched_df.to_excel(writer, sheet_name='Unmatched_Trades', index=False)
        
        # Summary pivots
        swap_count_by_cpty.to_excel(writer, sheet_name='Summary_By_Counterparty')
        swap_type_summary.to_excel(writer, sheet_name='Summary_By_Swap_Type')
    
    # Print final summary
    print("\nFinal Results (Default Configuration):")
    print("- Total trades: {}".format(len(trades_df)))
    print("- Matched trades: {}".format(len(result_df[result_df[SwapColumns.IS_MATCHED.value]])))
    print("- Unmatched trades: {}".format(len(unmatched_df)))
    print("- Collateral swaps identified: {}".format(len(engine.matched_swaps)))
    
    if not transaction_summary_df.empty:
        print("\nSwap Type Distribution:")
        for swap_type, count in transaction_summary_df['collateral_swap_type'].value_counts().items():
            print("- {}: {}".format(swap_type, count))
    
    print("\nResults exported to: collateral_swap_analysis.xlsx")
    
    return result_df, transaction_summary_df, unmatched_df, engine

if __name__ == "__main__":
    result_df, transaction_summary_df, unmatched_df, engine = main()
