# Keep rows that are either:
# 1. NOT in removal list, OR
# 2. Are exception counterparty AND meet condition
df_filtered = df[
    (~df['cpty'].isin(removal_list)) |  # Everything not in removal list
    ((df['cpty'] == exception_cpty) & (df['other_column'] == keep_condition_value))  # Exception subset
]

# Step 1: Filter the data (keep everything except removal list + exception subset)
removal_list = ['CPTY_A', 'CPTY_B', 'CPTY_C', 'CPTY_D']
exception_cpty = 'CPTY_D'
keep_condition_value = 'CLIENT_FACING'

df_filtered = df[
    (~df['cpty'].isin(removal_list)) |  # Keep everything NOT in removal list
    ((df['cpty'] == exception_cpty) & (df['trade_type'] == keep_condition_value))  # Keep exception subset
]

# Step 2: Override the cpty values for the retained subset
new_cpty_value = 'CLIENT_DESK_CLEAN'
mask_override = (df_filtered['cpty'] == exception_cpty) & (df_filtered['trade_type'] == keep_condition_value)
df_filtered.loc[mask_override, 'cpty'] = new_cpty_value

df_final = df[
    (~df['cpty'].isin(removal_list)) |
    ((df['cpty'] == exception_cpty) & (df['trade_type'] == keep_condition_value))
].copy()

# Then override
df_final.loc[
    (df_final['cpty'] == exception_cpty) & (df_final['trade_type'] == keep_condition_value), 
    'cpty'
] = 'CLIENT_DESK_CLEAN'

######################################################
edit
####################################################

import pandas as pd
import numpy as np
from datetime import datetime, date
import re

# Create dummy data to demonstrate the issue
np.random.seed(42)

# Sample trade IDs with embedded maturity dates
trade_ids_with_dates = [
    'FX_SWAP_20251215_USD_EUR',
    'IR_SWAP_2024-03-15_5Y',
    'XCCY_20250630_EURUSD',
    'BOND_LEGACY_1900_SYSTEM',  # This one has the 1900 issue
    'FWD_20241225_GBPUSD',
    'SWAP_1900_LEGACY_DATA',    # Another legacy issue
    'OPT_20250315_CALL',
    'FX_20241201_SPOT'
]

# Create sample data
data = {
    'trade_id': trade_ids_with_dates * 3,  # Replicate to get more data
    'maturity_date': [
        '2025-12-15', '2024-03-15', '2025-06-30', '1900-01-01',  # Legacy issue
        '2024-12-25', '1900-01-01',  # Another legacy issue
        '2025-03-15', '2024-12-01'
    ] * 3,
    'trade_type': np.random.choice(['FX_SWAP', 'IR_SWAP', 'XCCY', 'BOND', 'FWD', 'OPT'], 24),
    'notional': np.random.randint(100000, 10000000, 24)
}

df = pd.DataFrame(data)
df['maturity_date'] = pd.to_datetime(df['maturity_date'])

print("=== ORIGINAL DATA WITH LEGACY ISSUES ===")
print(f"Total rows: {len(df)}")
print(f"Rows with 1900-01-01 maturity: {len(df[df['maturity_date'] == '1900-01-01'])}")
print("\nSample data:")
print(df[['trade_id', 'maturity_date']].head(10))

print("\n=== APPROACH 1: EXTRACT DATES FROM TRADE ID ===")

def extract_date_from_trade_id(trade_id):
    """Extract date from trade ID using various patterns"""
    
    # Pattern 1: YYYYMMDD format
    pattern1 = r'(\d{8})'
    match1 = re.search(pattern1, trade_id)
    if match1:
        try:
            date_str = match1.group(1)
            return pd.to_datetime(date_str, format='%Y%m%d')
        except:
            pass
    
    # Pattern 2: YYYY-MM-DD format
    pattern2 = r'(\d{4}-\d{2}-\d{2})'
    match2 = re.search(pattern2, trade_id)
    if match2:
        try:
            return pd.to_datetime(match2.group(1))
        except:
            pass
    
    # Pattern 3: YYYYMM format (assume last day of month)
    pattern3 = r'(\d{6})'
    match3 = re.search(pattern3, trade_id)
    if match3:
        try:
            date_str = match3.group(1)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            # Get last day of month
            if month == 12:
                next_month = pd.Timestamp(year + 1, 1, 1)
            else:
                next_month = pd.Timestamp(year, month + 1, 1)
            last_day = next_month - pd.Timedelta(days=1)
            return last_day
        except:
            pass
    
    return None

# Apply extraction
df['extracted_date'] = df['trade_id'].apply(extract_date_from_trade_id)

print("Extraction results:")
print(df[['trade_id', 'maturity_date', 'extracted_date']].head(10))

print("\n=== APPROACH 2: CONDITIONAL OVERRIDE ===")

# Create a copy for fixing
df_fixed = df.copy()

# Method 1: Override only 1900-01-01 dates with extracted dates
legacy_mask = df_fixed['maturity_date'] == '1900-01-01'
has_extracted = df_fixed['extracted_date'].notna()

# Override condition: legacy date AND we have an extracted date
override_mask = legacy_mask & has_extracted
df_fixed.loc[override_mask, 'maturity_date'] = df_fixed.loc[override_mask, 'extracted_date']

print(f"Legacy dates found: {legacy_mask.sum()}")
print(f"Extracted dates available: {has_extracted.sum()}")
print(f"Successfully overridden: {override_mask.sum()}")

print("\nBefore and after comparison:")
comparison = df[['trade_id', 'maturity_date']].copy()
comparison['fixed_maturity_date'] = df_fixed['maturity_date']
comparison['was_fixed'] = override_mask
print(comparison[legacy_mask])

print("\n=== APPROACH 3: COMPREHENSIVE SOLUTION ===")

def fix_legacy_maturity_dates(df, trade_id_col='trade_id', maturity_col='maturity_date', 
                            legacy_date='1900-01-01', backup_years=5):
    """
    Comprehensive function to fix legacy maturity dates
    """
    df_result = df.copy()
    
    # Step 1: Try to extract from trade ID
    df_result['extracted_date'] = df_result[trade_id_col].apply(extract_date_from_trade_id)
    
    # Step 2: Identify legacy dates
    legacy_mask = df_result[maturity_col] == legacy_date
    
    # Step 3: Override with extracted dates where possible
    can_fix_mask = legacy_mask & df_result['extracted_date'].notna()
    df_result.loc[can_fix_mask, maturity_col] = df_result.loc[can_fix_mask, 'extracted_date']
    
    # Step 4: For remaining legacy dates, set to far future or flag for manual review
    still_legacy_mask = df_result[maturity_col] == legacy_date
    if still_legacy_mask.any():
        print(f"⚠️  {still_legacy_mask.sum()} trades still have legacy dates - setting to {backup_years} years from now")
        future_date = pd.Timestamp.now() + pd.DateOffset(years=backup_years)
        df_result.loc[still_legacy_mask, maturity_col] = future_date
        df_result.loc[still_legacy_mask, 'needs_manual_review'] = True
    
    # Step 5: Add flags for tracking
    df_result['date_was_fixed'] = can_fix_mask
    df_result['date_source'] = 'original'
    df_result.loc[can_fix_mask, 'date_source'] = 'extracted_from_trade_id'
    df_result.loc[still_legacy_mask, 'date_source'] = 'default_future'
    
    # Clean up temporary column
    df_result = df_result.drop('extracted_date', axis=1)
    
    return df_result

# Apply comprehensive fix
df_comprehensive = fix_legacy_maturity_dates(df)

print("\nComprehensive fix results:")
print(f"Total trades: {len(df_comprehensive)}")
print(f"Dates fixed from trade ID: {df_comprehensive['date_was_fixed'].sum()}")
print(f"Dates set to future: {df_comprehensive.get('needs_manual_review', pd.Series([])).sum()}")

print("\nDate source breakdown:")
print(df_comprehensive['date_source'].value_counts())

print("\n=== APPROACH 4: PRODUCTION-READY WORKFLOW ===")

def production_maturity_fix(df, legacy_cutoff_year=1950):
    """
    Production-ready solution for fixing maturity dates
    """
    df_prod = df.copy()
    
    # Identify suspicious dates (anything before cutoff year)
    suspicious_mask = df_prod['maturity_date'].dt.year < legacy_cutoff_year
    
    if suspicious_mask.any():
        print(f"Found {suspicious_mask.sum()} suspicious maturity dates")
        
        # Try to extract from trade ID
        df_prod['temp_extracted'] = df_prod['trade_id'].apply(extract_date_from_trade_id)
        
        # Fix where possible
        can_fix = suspicious_mask & df_prod['temp_extracted'].notna()
        df_prod.loc[can_fix, 'maturity_date'] = df_prod.loc[can_fix, 'temp_extracted']
        
        # Flag remaining issues
        still_suspicious = df_prod['maturity_date'].dt.year < legacy_cutoff_year
        if still_suspicious.any():
            df_prod.loc[still_suspicious, 'data_quality_flag'] = 'SUSPICIOUS_MATURITY_DATE'
        
        df_prod.loc[can_fix, 'data_quality_flag'] = 'MATURITY_DATE_FIXED'
        df_prod = df_prod.drop('temp_extracted', axis=1)
    
    return df_prod

df_production = production_maturity_fix(df)

print("\nProduction fix summary:")
if 'data_quality_flag' in df_production.columns:
    print(df_production['data_quality_flag'].value_counts())

print("\n=== FINAL VERIFICATION ===")
print("Before fix - maturity date distribution:")
print(df['maturity_date'].dt.year.value_counts().sort_index())

print("\nAfter fix - maturity date distribution:")
print(df_comprehensive['maturity_date'].dt.year.value_counts().sort_index())

# Show final clean data
print("\n=== SAMPLE OF CLEANED DATA ===")
cols_to_show = ['trade_id', 'maturity_date', 'date_source']
if 'date_was_fixed' in df_comprehensive.columns:
    cols_to_show.append('date_was_fixed')
print(df_comprehensive[cols_to_show].head(10))




############################################



############################################






import pandas as pd
import numpy as np
from datetime import datetime, date
import re

# Create dummy data to demonstrate the issue
np.random.seed(42)

# Sample trade IDs with embedded maturity dates
trade_ids_with_dates = [
    'FX_SWAP_20251215_USD_EUR',
    'IR_SWAP_2024-03-15_5Y',
    'XCCY_20250630_EURUSD',
    'BOND_LEGACY_1900_SYSTEM',  # This one has the 1900 issue
    'FWD_20241225_GBPUSD',
    'SWAP_1900_LEGACY_DATA',    # Another legacy issue
    'OPT_20250315_CALL',
    'FX_20241201_SPOT'
]

# Create sample data
data = {
    'trade_id': trade_ids_with_dates * 3,  # Replicate to get more data
    'maturity_date': [
        '2025-12-15', '2024-03-15', '2025-06-30', '1900-01-01',  # Legacy issue
        '2024-12-25', '1900-01-01',  # Another legacy issue
        '2025-03-15', '2024-12-01'
    ] * 3,
    'trade_type': np.random.choice(['FX_SWAP', 'IR_SWAP', 'XCCY', 'BOND', 'FWD', 'OPT'], 24),
    'notional': np.random.randint(100000, 10000000, 24)
}

df = pd.DataFrame(data)
df['maturity_date'] = pd.to_datetime(df['maturity_date'])

print("=== ORIGINAL DATA WITH LEGACY ISSUES ===")
print(f"Total rows: {len(df)}")
print(f"Rows with 1900-01-01 maturity: {len(df[df['maturity_date'] == '1900-01-01'])}")
print("\nSample data:")
print(df[['trade_id', 'maturity_date']].head(10))

print("\n=== APPROACH 1: EXTRACT DATES FROM TRADE ID ===")

def extract_date_from_trade_id(trade_id):
    """Extract date from trade ID using various patterns"""
    
    # Pattern 1: YYYYMMDD format
    pattern1 = r'(\d{8})'
    match1 = re.search(pattern1, trade_id)
    if match1:
        try:
            date_str = match1.group(1)
            return pd.to_datetime(date_str, format='%Y%m%d')
        except:
            pass
    
    # Pattern 2: YYYY-MM-DD format
    pattern2 = r'(\d{4}-\d{2}-\d{2})'
    match2 = re.search(pattern2, trade_id)
    if match2:
        try:
            return pd.to_datetime(match2.group(1))
        except:
            pass
    
    # Pattern 3: YYYYMM format (assume last day of month)
    pattern3 = r'(\d{6})'
    match3 = re.search(pattern3, trade_id)
    if match3:
        try:
            date_str = match3.group(1)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            # Get last day of month
            if month == 12:
                next_month = pd.Timestamp(year + 1, 1, 1)
            else:
                next_month = pd.Timestamp(year, month + 1, 1)
            last_day = next_month - pd.Timedelta(days=1)
            return last_day
        except:
            pass
    
    return None

# Apply extraction
df['extracted_date'] = df['trade_id'].apply(extract_date_from_trade_id)

print("Extraction results:")
print(df[['trade_id', 'maturity_date', 'extracted_date']].head(10))

print("\n=== APPROACH 2: CONDITIONAL OVERRIDE ===")

# Create a copy for fixing
df_fixed = df.copy()

# Method 1: Override only 1900-01-01 dates with extracted dates
legacy_mask = df_fixed['maturity_date'] == '1900-01-01'
has_extracted = df_fixed['extracted_date'].notna()

# Override condition: legacy date AND we have an extracted date
override_mask = legacy_mask & has_extracted
df_fixed.loc[override_mask, 'maturity_date'] = df_fixed.loc[override_mask, 'extracted_date']

print(f"Legacy dates found: {legacy_mask.sum()}")
print(f"Extracted dates available: {has_extracted.sum()}")
print(f"Successfully overridden: {override_mask.sum()}")

print("\nBefore and after comparison:")
comparison = df[['trade_id', 'maturity_date']].copy()
comparison['fixed_maturity_date'] = df_fixed['maturity_date']
comparison['was_fixed'] = override_mask
print(comparison[legacy_mask])

print("\n=== APPROACH 3: COMPREHENSIVE SOLUTION ===")

def fix_legacy_maturity_dates(df, trade_id_col='trade_id', maturity_col='maturity_date', 
                            legacy_date='1900-01-01', backup_years=5):
    """
    Comprehensive function to fix legacy maturity dates
    """
    df_result = df.copy()
    
    # Step 1: Try to extract from trade ID
    df_result['extracted_date'] = df_result[trade_id_col].apply(extract_date_from_trade_id)
    
    # Step 2: Identify legacy dates
    legacy_mask = df_result[maturity_col] == legacy_date
    
    # Step 3: Override with extracted dates where possible
    can_fix_mask = legacy_mask & df_result['extracted_date'].notna()
    df_result.loc[can_fix_mask, maturity_col] = df_result.loc[can_fix_mask, 'extracted_date']
    
    # Step 4: For remaining legacy dates, set to far future or flag for manual review
    still_legacy_mask = df_result[maturity_col] == legacy_date
    if still_legacy_mask.any():
        print(f"⚠️  {still_legacy_mask.sum()} trades still have legacy dates - setting to {backup_years} years from now")
        future_date = pd.Timestamp.now() + pd.DateOffset(years=backup_years)
        df_result.loc[still_legacy_mask, maturity_col] = future_date
        df_result.loc[still_legacy_mask, 'needs_manual_review'] = True
    
    # Step 5: Add flags for tracking
    df_result['date_was_fixed'] = can_fix_mask
    df_result['date_source'] = 'original'
    df_result.loc[can_fix_mask, 'date_source'] = 'extracted_from_trade_id'
    df_result.loc[still_legacy_mask, 'date_source'] = 'default_future'
    
    # Clean up temporary column
    df_result = df_result.drop('extracted_date', axis=1)
    
    return df_result

# Apply comprehensive fix
df_comprehensive = fix_legacy_maturity_dates(df)

print("\nComprehensive fix results:")
print(f"Total trades: {len(df_comprehensive)}")
print(f"Dates fixed from trade ID: {df_comprehensive['date_was_fixed'].sum()}")
print(f"Dates set to future: {df_comprehensive.get('needs_manual_review', pd.Series([])).sum()}")

print("\nDate source breakdown:")
print(df_comprehensive['date_source'].value_counts())

print("\n=== APPROACH 4: PRODUCTION-READY WORKFLOW ===")

def production_maturity_fix(df, legacy_cutoff_year=1950):
    """
    Production-ready solution for fixing maturity dates
    """
    df_prod = df.copy()
    
    # Identify suspicious dates (anything before cutoff year)
    suspicious_mask = df_prod['maturity_date'].dt.year < legacy_cutoff_year
    
    if suspicious_mask.any():
        print(f"Found {suspicious_mask.sum()} suspicious maturity dates")
        
        # Try to extract from trade ID
        df_prod['temp_extracted'] = df_prod['trade_id'].apply(extract_date_from_trade_id)
        
        # Fix where possible
        can_fix = suspicious_mask & df_prod['temp_extracted'].notna()
        df_prod.loc[can_fix, 'maturity_date'] = df_prod.loc[can_fix, 'temp_extracted']
        
        # Flag remaining issues
        still_suspicious = df_prod['maturity_date'].dt.year < legacy_cutoff_year
        if still_suspicious.any():
            df_prod.loc[still_suspicious, 'data_quality_flag'] = 'SUSPICIOUS_MATURITY_DATE'
        
        df_prod.loc[can_fix, 'data_quality_flag'] = 'MATURITY_DATE_FIXED'
        df_prod = df_prod.drop('temp_extracted', axis=1)
    
    return df_prod

df_production = production_maturity_fix(df)

print("\nProduction fix summary:")
if 'data_quality_flag' in df_production.columns:
    print(df_production['data_quality_flag'].value_counts())

print("\n=== FINAL VERIFICATION ===")
print("Before fix - maturity date distribution:")
print(df['maturity_date'].dt.year.value_counts().sort_index())

print("\nAfter fix - maturity date distribution:")
print(df_comprehensive['maturity_date'].dt.year.value_counts().sort_index())

# Show final clean data
print("\n=== SAMPLE OF CLEANED DATA ===")
cols_to_show = ['trade_id', 'maturity_date', 'date_source']
if 'date_was_fixed' in df_comprehensive.columns:
    cols_to_show.append('date_was_fixed')
print(df_comprehensive[cols_to_show].head(10))





#############################################################


import pandas as pd
import numpy as np

# Create dummy data
np.random.seed(42)

# Generate sample data
data = {
    'trade_id': range(1, 101),  # 100 trades
    'cpty': np.random.choice(['CPTY_A', 'CPTY_B', 'CPTY_C', 'CPTY_D', 'CPTY_E', 'CPTY_F', 'GOOD_BANK', 'OTHER_BANK'], 100),
    'trade_type': np.random.choice(['WASH', 'CLIENT_FACING', 'INTERDESK', 'EXTERNAL'], 100),
    'amount': np.random.randint(1000, 100000, 100),
    'currency': np.random.choice(['USD', 'EUR', 'GBP'], 100)
}

df = pd.DataFrame(data)

print("=== ORIGINAL DATA ===")
print(f"Total rows: {len(df)}")
print("\nCounterparty distribution:")
print(df['cpty'].value_counts().sort_index())

print("\n=== FILTERING SETUP ===")
removal_list = ['CPTY_A', 'CPTY_B', 'CPTY_C', 'CPTY_D']  # Remove these 4 counterparties
exception_cpty = 'CPTY_D'  # But keep some rows from this one
keep_condition_value = 'CLIENT_FACING'  # Keep rows where trade_type is CLIENT_FACING

print(f"Removal list: {removal_list}")
print(f"Exception counterparty: {exception_cpty}")
print(f"Keep condition: trade_type == '{keep_condition_value}'")

# Show what we have for the exception counterparty before filtering
print(f"\n=== BEFORE FILTERING: {exception_cpty} breakdown ===")
cpty_d_data = df[df['cpty'] == exception_cpty]
print(f"Total {exception_cpty} rows: {len(cpty_d_data)}")
if len(cpty_d_data) > 0:
    print("Trade type breakdown:")
    print(cpty_d_data['trade_type'].value_counts())

print("\n=== APPLYING FILTER ===")

# Method 1: Two-step approach (clearer logic)
print("\nMethod 1: Two-step approach")
df_step1 = df[~df['cpty'].isin(removal_list)]
print(f"After removing {removal_list}: {len(df_step1)} rows")

exception_subset = df[(df['cpty'] == exception_cpty) & (df['trade_type'] == keep_condition_value)]
print(f"Exception subset from {exception_cpty}: {len(exception_subset)} rows")

df_filtered_method1 = pd.concat([df_step1, exception_subset], ignore_index=True)
print(f"Final result Method 1: {len(df_filtered_method1)} rows")

# Method 2: Single boolean expression
print("\nMethod 2: Single boolean expression")
df_filtered_method2 = df[
    (~df['cpty'].isin(removal_list)) |  # Keep everything NOT in removal list
    ((df['cpty'] == exception_cpty) & (df['trade_type'] == keep_condition_value))  # Keep exception subset
]
print(f"Final result Method 2: {len(df_filtered_method2)} rows")

# Verify both methods give same result
print(f"\nBoth methods give same result: {len(df_filtered_method1) == len(df_filtered_method2)}")

print("\n=== FINAL RESULTS ===")
df_final = df_filtered_method2  # Use method 2 result

print(f"Original rows: {len(df)}")
print(f"Final rows: {len(df_final)}")
print(f"Rows removed: {len(df) - len(df_final)}")

print("\nFinal counterparty distribution:")
print(df_final['cpty'].value_counts().sort_index())

# Show what we kept from the exception counterparty
if exception_cpty in df_final['cpty'].values:
    kept_exception = df_final[df_final['cpty'] == exception_cpty]
    print(f"\nKept from {exception_cpty}: {len(kept_exception)} rows")
    print("Trade types kept:")
    print(kept_exception['trade_type'].value_counts())
else:
    print(f"\nNo rows kept from {exception_cpty}")

# Show some sample data
print("\n=== SAMPLE OF FINAL DATA ===")
print(df_final.head(10))

print("\n=== FINAL TWEAK: OVERRIDE CPTY VALUES ===")
# Override the cpty value for the retained subset
new_cpty_value = 'CLIENT_DESK_CLEAN'  # New value for the exception subset

# Before override
if exception_cpty in df_final['cpty'].values:
    before_count = len(df_final[df_final['cpty'] == exception_cpty])
    print(f"Before override: {before_count} rows with cpty = '{exception_cpty}'")

# Apply the override
mask_override = (df_final['cpty'] == exception_cpty) & (df_final['trade_type'] == keep_condition_value)
df_final.loc[mask_override, 'cpty'] = new_cpty_value

# After override
after_count = len(df_final[df_final['cpty'] == new_cpty_value])
print(f"After override: {after_count} rows with cpty = '{new_cpty_value}'")

print("\nFinal counterparty distribution after override:")
print(df_final['cpty'].value_counts().sort_index())

print("\n=== VERIFICATION ===")
# Check that we removed the right counterparties (none from removal list should remain)
removed_cptys_still_present = set(removal_list) & set(df_final['cpty'].unique())
print(f"Counterparties from removal list still present (should be empty): {removed_cptys_still_present}")

# Verify the new counterparty only has the right trade types
if new_cpty_value in df_final['cpty'].values:
    new_cpty_trade_types = df_final[df_final['cpty'] == new_cpty_value]['trade_type'].unique()
    print(f"{new_cpty_value} trade types: {list(new_cpty_trade_types)}")
    unexpected_types = set(new_cpty_trade_types) - {keep_condition_value}
    if unexpected_types:
        print(f"⚠️  Unexpected trade types found: {unexpected_types}")
    else:
        print(f"✅ Only expected trade type '{keep_condition_value}' found for {new_cpty_value}")

print("\n=== COMPLETE WORKFLOW SUMMARY ===")
print(f"1. Started with {len(df)} total rows")
print(f"2. Removed counterparties: {[cpty for cpty in removal_list if cpty != exception_cpty]}")
print(f"3. Kept {after_count} rows from '{exception_cpty}' where trade_type = '{keep_condition_value}'")
print(f"4. Renamed those {after_count} rows to counterparty '{new_cpty_value}'")
print(f"5. Final dataset: {len(df_final)} rows")

# Show the final clean dataset sample
print("\n=== FINAL CLEAN DATASET SAMPLE ===")
print(df_final.head(10))
