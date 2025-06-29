import pandas as pd
import numpy as np

def handle_kondor_duplicates(df):
    """
    Handle duplicate Kondor tickets with same trade_id by overriding the amount column
    for the ticket with smaller PV (accrued interest ticket).
    
    Parameters:
    df (DataFrame): Trade dataframe with columns including 'booking_system', 'trade_id', 'amount', 'PV'
    
    Returns:
    DataFrame: Updated dataframe with corrected amount values
    """
    
    # Column name variables
    BOOKING_SYSTEM_COL = 'booking_system'
    TRADE_ID_COL = 'trade_id'
    AMOUNT_COL = 'amount'
    PV_COL = 'PV'
    
    # Value variables
    KONDOR_SYSTEM = 'kondor'
    EXPECTED_DUPLICATES = 2
    
    # Create a copy to avoid modifying original dataframe
    df_updated = df.copy()
    
    # Filter for Kondor bookings only
    kondor_mask = df_updated[BOOKING_SYSTEM_COL] == KONDOR_SYSTEM
    kondor_df = df_updated[kondor_mask].copy()
    
    # Find duplicate trade_ids in Kondor bookings
    duplicate_trade_ids = kondor_df[kondor_df.duplicated(TRADE_ID_COL, keep=False)][TRADE_ID_COL].unique()
    
    print(f"Found {len(duplicate_trade_ids)} Kondor trade IDs with duplicates")
    
    # Process each duplicate trade_id
    for trade_id in duplicate_trade_ids:
        # Get all rows for this trade_id
        trade_rows = kondor_df[kondor_df[TRADE_ID_COL] == trade_id].copy()
        
        if len(trade_rows) == EXPECTED_DUPLICATES:  # Assuming pairs of duplicates
            # Find the row with smaller PV (accrued interest ticket)
            min_pv_idx = trade_rows[PV_COL].idxmin()
            max_pv_idx = trade_rows[PV_COL].idxmax()
            
            # Get the PV value from the smaller PV row (accrued interest)
            accrued_interest_pv = trade_rows.loc[min_pv_idx, PV_COL]
            
            # Override the amount column for the smaller PV row with its PV value
            df_updated.loc[min_pv_idx, AMOUNT_COL] = accrued_interest_pv
            
            print(f"Trade ID {trade_id}: Updated amount for accrued interest ticket (smaller PV: {accrued_interest_pv})")
            
        else:
            print(f"Warning: Trade ID {trade_id} has {len(trade_rows)} duplicates (expected {EXPECTED_DUPLICATES})")
    
    return df_updated

# Alternative approach with more detailed tracking
def handle_kondor_duplicates_detailed(df):
    """
    Enhanced version with detailed tracking and validation
    """
    # Column name variables
    BOOKING_SYSTEM_COL = 'booking_system'
    TRADE_ID_COL = 'trade_id'
    AMOUNT_COL = 'amount'
    PV_COL = 'PV'
    
    # Value variables
    KONDOR_SYSTEM = 'kondor'
    MIN_DUPLICATES = 1
    SMALLEST_PV_INDEX = 0
    LARGEST_PV_INDEX = -1
    
    # Update tracking keys
    TRADE_ID_KEY = 'trade_id'
    ACCRUED_IDX_KEY = 'accrued_idx'
    NOTIONAL_IDX_KEY = 'notional_idx'
    ORIGINAL_AMOUNT_KEY = 'original_amount'
    NEW_AMOUNT_KEY = 'new_amount'
    ACCRUED_PV_KEY = 'accrued_pv'
    NOTIONAL_PV_KEY = 'notional_pv'
    
    df_updated = df.copy()
    
    # Filter for Kondor bookings
    kondor_mask = df_updated[BOOKING_SYSTEM_COL] == KONDOR_SYSTEM
    
    # Group by trade_id for Kondor bookings
    kondor_groups = df_updated[kondor_mask].groupby(TRADE_ID_COL)
    
    updates_made = []
    
    for trade_id, group in kondor_groups:
        if len(group) > MIN_DUPLICATES:  # Has duplicates
            # Sort by PV to identify smaller (accrued interest) and larger (notional) amounts
            group_sorted = group.sort_values(PV_COL)
            
            # Get indices
            accrued_idx = group_sorted.index[SMALLEST_PV_INDEX]  # Smallest PV
            notional_idx = group_sorted.index[LARGEST_PV_INDEX]  # Largest PV
            
            # Store original values for tracking
            original_amount = df_updated.loc[accrued_idx, AMOUNT_COL]
            new_amount = df_updated.loc[accrued_idx, PV_COL]
            
            # Update the amount for the accrued interest ticket
            df_updated.loc[accrued_idx, AMOUNT_COL] = new_amount
            
            # Track the update
            updates_made.append({
                TRADE_ID_KEY: trade_id,
                ACCRUED_IDX_KEY: accrued_idx,
                NOTIONAL_IDX_KEY: notional_idx,
                ORIGINAL_AMOUNT_KEY: original_amount,
                NEW_AMOUNT_KEY: new_amount,
                ACCRUED_PV_KEY: df_updated.loc[accrued_idx, PV_COL],
                NOTIONAL_PV_KEY: df_updated.loc[notional_idx, PV_COL]
            })
    
    # Print summary
    print(f"\nSummary of updates made:")
    print(f"Total duplicate trade IDs processed: {len(updates_made)}")
    
    for update in updates_made:
        print(f"Trade ID {update[TRADE_ID_KEY]}: "
              f"Amount changed from {update[ORIGINAL_AMOUNT_KEY]} to {update[NEW_AMOUNT_KEY]} "
              f"(Accrued PV: {update[ACCRUED_PV_KEY]}, Notional PV: {update[NOTIONAL_PV_KEY]})")
    
    return df_updated, updates_made

# Function to validate the results
def validate_kondor_updates(original_df, updated_df):
    """
    Validate that updates were applied correctly
    """
    # Column name variables
    BOOKING_SYSTEM_COL = 'booking_system'
    TRADE_ID_COL = 'trade_id'
    AMOUNT_COL = 'amount'
    PV_COL = 'PV'
    
    # Value variables
    KONDOR_SYSTEM = 'kondor'
    KEEP_PARAM = False
    
    # Display columns for validation
    VALIDATION_COLS = [TRADE_ID_COL, AMOUNT_COL, PV_COL]
    
    # Output messages
    VALIDATION_HEADER = "\nValidation Results:"
    DUPLICATE_COUNT_MSG = "Kondor bookings with duplicate trade IDs:"
    ORIGINAL_LABEL = "Original:"
    UPDATED_LABEL = "Updated:"
    
    kondor_mask = original_df[BOOKING_SYSTEM_COL] == KONDOR_SYSTEM
    
    # Check for duplicates in original data
    original_kondor = original_df[kondor_mask]
    duplicate_trade_ids = original_kondor[original_kondor.duplicated(TRADE_ID_COL, keep=KEEP_PARAM)][TRADE_ID_COL].unique()
    
    print(f"{VALIDATION_HEADER}")
    print(f"{DUPLICATE_COUNT_MSG} {len(duplicate_trade_ids)}")
    
    for trade_id in duplicate_trade_ids:
        orig_rows = original_df[(original_df[TRADE_ID_COL] == trade_id) & kondor_mask]
        updated_rows = updated_df[(updated_df[TRADE_ID_COL] == trade_id) & kondor_mask]
        
        print(f"\nTrade ID {trade_id}:")
        print(f"{ORIGINAL_LABEL}")
        print(orig_rows[VALIDATION_COLS].to_string(index=KEEP_PARAM))
        print(f"{UPDATED_LABEL}")
        print(updated_rows[VALIDATION_COLS].to_string(index=KEEP_PARAM))

# Example usage:
"""
# Assuming your dataframe has columns: 'booking_system', 'trade_id', 'amount', 'PV'
# Usage example:

# Method 1: Simple approach
df_corrected = handle_kondor_duplicates(df)

# Method 2: Detailed approach with tracking
df_corrected, update_log = handle_kondor_duplicates_detailed(df)

# Validate the changes
validate_kondor_updates(df, df_corrected)
"""
