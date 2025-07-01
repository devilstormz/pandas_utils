import pandas as pd
import numpy as np
import re
import yaml
from io import StringIO

class TradeDataProcessor:
    """
    A configurable trade data pre-processing class that applies system-specific rules
    using numpy.where for vectorized operations with YAML-based configuration.
    """
    
    def __init__(self, column_config=None, config_yaml=None, config_dict=None):
        """
        Initialize the processor with column and rule configuration.
        
        Args:
            column_config (dict): Dictionary mapping logical column names to actual DataFrame column names
            config_yaml (str): YAML string or file path containing rule configuration
            config_dict (dict): Dictionary containing rule configuration (alternative to YAML)
        """
        # Default column mapping - can be overridden
        self.columns = {
            'risk_system': 'RiskSystem',
            'system': 'System', 
            'book': 'Book',
            'cpty_code': 'CptyCode',
            'cpty_name': 'CptyName',
            'risk_group': 'RiskGroup',
            'full_leg_ext_id': 'FullLegExtId',
            'underlying': 'Underlying'
        }
        
        # Override with provided config
        if column_config:
            self.columns.update(column_config)
        
        # Load configuration
        if config_yaml:
            self.config = self._load_yaml_config(config_yaml)
        elif config_dict:
            self.config = config_dict
        else:
            self.config = self._get_default_config()
    
    def _load_yaml_config(self, config_yaml):
        """
        Load configuration from YAML string or file path.
        
        Args:
            config_yaml (str): YAML string or file path
            
        Returns:
            dict: Parsed configuration
        """
        try:
            # Try to load as file path first
            with open(config_yaml, 'r') as file:
                return yaml.safe_load(file)
        except (FileNotFoundError, OSError):
            # If not a file, treat as YAML string
            return yaml.safe_load(config_yaml)
    
    def _get_default_config(self):
        """
        Get default configuration as nested dictionary (equivalent to YAML structure).
        
        Returns:
            dict: Default configuration
        """
        return {
            'global_settings': {
                'target_risk_system': 'legacy'
            },
            'systems': {
                'A': {
                    'rules': [
                        {
                            'name': 'book_and_system_rule',
                            'description': 'Book in (booka, boookb) and system is B-E',
                            'conditions': [
                                {'field': 'book', 'operator': 'in', 'value': ['booka', 'boookb']},
                                {'field': 'system', 'operator': 'eq', 'value': 'B-E'}
                            ],
                            'actions': [
                                {'field': 'cpty_code', 'operator': 'set', 'value': 'X1'},
                                {'field': 'cpty_name', 'operator': 'set', 'value': 'x2'}
                            ]
                        },
                        {
                            'name': 'book_system_risk_group_rule',
                            'description': 'Book = bookc, system is B-E, and RiskGroup = Y',
                            'conditions': [
                                {'field': 'book', 'operator': 'eq', 'value': 'bookc'},
                                {'field': 'system', 'operator': 'eq', 'value': 'B-E'},
                                {'field': 'risk_group', 'operator': 'eq', 'value': 'Y'}
                            ],
                            'actions': [
                                {'field': 'cpty_code', 'operator': 'set', 'value': 'Y1'},
                                {'field': 'cpty_name', 'operator': 'set', 'value': 'y2'}
                            ]
                        },
                        {
                            'name': 'isin_extraction_rule',
                            'description': 'RiskGroup=Own issuance - extract ISIN',
                            'conditions': [
                                {'field': 'risk_group', 'operator': 'eq', 'value': 'Own issuance'}
                            ],
                            'actions': [
                                {
                                    'field': 'underlying', 
                                    'operator': 'regex_extract_transform',
                                    'source_field': 'full_leg_ext_id',
                                    'pattern': '^([A-Z]{2}[A-Z0-9]{10})_.*_BOND_(?:BUY|SELL)_?',
                                    'group': 1,
                                    'prefix': 'B:'
                                }
                            ]
                        }
                    ]
                },
                'B': {
                    'rules': [
                        {
                            'name': 'abc_search_rule',
                            'description': 'FullLegExtId contains ABC',
                            'conditions': [
                                {'field': 'full_leg_ext_id', 'operator': 'contains', 'value': 'ABC'}
                            ],
                            'actions': [
                                {'field': 'cpty_code', 'operator': 'set', 'value': 'ABC'},
                                {'field': 'cpty_name', 'operator': 'set', 'value': 'ABC_2'}
                            ]
                        }
                    ]
                },
                'C': {
                    'rules': [
                        {
                            'name': 'abc_search_rule',
                            'description': 'FullLegExtId contains ABC',
                            'conditions': [
                                {'field': 'full_leg_ext_id', 'operator': 'contains', 'value': 'ABC'}
                            ],
                            'actions': [
                                {'field': 'cpty_code', 'operator': 'set', 'value': 'ABC'},
                                {'field': 'cpty_name', 'operator': 'set', 'value': 'ABC_2'}
                            ]
                        },
                        {
                            'name': 'aaaa_and_risk_group_rule',
                            'description': 'FullLegExtId contains AAAA and RiskGroup=Z',
                            'conditions': [
                                {'field': 'full_leg_ext_id', 'operator': 'contains', 'value': 'AAAA'},
                                {'field': 'risk_group', 'operator': 'eq', 'value': 'Z'}
                            ],
                            'actions': [
                                {'field': 'cpty_code', 'operator': 'set', 'value': 'AAAA1'},
                                {'field': 'cpty_name', 'operator': 'set', 'value': 'AAAA2'}
                            ]
                        }
                    ]
                }
            }
        }
    
    def _build_condition_mask(self, df, conditions):
        """
        Build a boolean mask from generalized condition specifications.
        
        Args:
            df (pd.DataFrame): Input dataframe
            conditions (list): List of condition dictionaries
            
        Returns:
            np.ndarray: Boolean mask
        """
        mask = np.ones(len(df), dtype=bool)
        
        for condition in conditions:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            
            # Get the actual column name
            column_name = self.columns.get(field, field)
            
            if operator == 'eq':
                mask &= (df[column_name] == value).values
            elif operator == 'in':
                mask &= df[column_name].isin(value).values
            elif operator == 'contains':
                mask &= df[column_name].str.contains(value, na=False).values
            elif operator == 'not_eq':
                mask &= (df[column_name] != value).values
            elif operator == 'not_in':
                mask &= ~df[column_name].isin(value).values
            elif operator == 'regex_match':
                mask &= df[column_name].str.match(value, na=False).values
            elif operator == 'is_null':
                mask &= df[column_name].isnull().values
            elif operator == 'is_not_null':
                mask &= df[column_name].notnull().values
            elif operator == 'gt':
                mask &= (df[column_name] > value).values
            elif operator == 'lt':
                mask &= (df[column_name] < value).values
            elif operator == 'gte':
                mask &= (df[column_name] >= value).values
            elif operator == 'lte':
                mask &= (df[column_name] <= value).values
            else:
                raise ValueError(f"Unknown condition operator: {operator}")
        
        return mask
    
    def _apply_actions(self, df, mask, actions):
        """
        Apply generalized actions to dataframe using numpy.where for vectorized operations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            mask (np.ndarray): Boolean mask indicating which rows to update
            actions (list): List of action dictionaries
        """
        for action in actions:
            field = action['field']
            operator = action['operator']
            
            # Get the actual column name
            column_name = self.columns.get(field, field)
            
            if operator == 'set':
                value = action['value']
                df[column_name] = np.where(mask, value, df[column_name])
                
            elif operator == 'set_if_null':
                value = action['value']
                current_null = df[column_name].isnull().values
                df[column_name] = np.where(mask & current_null, value, df[column_name])
                
            elif operator == 'regex_extract':
                source_field = action.get('source_field', field)
                source_column = self.columns.get(source_field, source_field)
                pattern = action['pattern']
                group = action.get('group', 0)
                
                def extract_regex_vectorized(series, mask_filter):
                    result = series.copy()
                    masked_series = series[mask_filter]
                    
                    # Extract using regex
                    extracted = masked_series.str.extract(f'({pattern})', expand=False)
                    if isinstance(extracted, pd.DataFrame) and group < len(extracted.columns):
                        extracted = extracted.iloc[:, group]
                    
                    # Only update where extraction was successful and not null
                    success_mask = extracted.notnull()
                    if success_mask.any():
                        result[mask_filter] = np.where(
                            success_mask, extracted, masked_series
                        )
                    
                    return result
                
                df[column_name] = extract_regex_vectorized(df[source_column], mask)
                
            elif operator == 'regex_extract_transform':
                source_field = action.get('source_field', field)
                source_column = self.columns.get(source_field, source_field)
                pattern = action['pattern']
                group = action.get('group', 0)
                prefix = action.get('prefix', '')
                suffix = action.get('suffix', '')
                
                def extract_transform_vectorized(series, mask_filter):
                    result = df[column_name].copy()
                    masked_series = series[mask_filter]
                    
                    # Extract using regex
                    matches = masked_series.str.extract(pattern, expand=False)
                    if isinstance(matches, pd.DataFrame):
                        if group < len(matches.columns):
                            extracted = matches.iloc[:, group]
                        else:
                            extracted = matches.iloc[:, 0]
                    else:
                        extracted = matches
                    
                    # Apply transformation where extraction was successful
                    success_mask = extracted.notnull() & (extracted != '')
                    transformed = prefix + extracted.astype(str) + suffix
                    
                    if success_mask.any():
                        result[mask_filter] = np.where(
                            success_mask, transformed, result[mask_filter]
                        )
                    
                    return result
                
                df[column_name] = extract_transform_vectorized(df[source_column], mask)
                
            elif operator == 'concatenate':
                parts = action['parts']
                separator = action.get('separator', '')
                
                concatenated_values = []
                for part in parts:
                    if isinstance(part, dict):
                        if 'field' in part:
                            part_column = self.columns.get(part['field'], part['field'])
                            concatenated_values.append(df[part_column].astype(str))
                        elif 'value' in part:
                            concatenated_values.append(pd.Series([part['value']] * len(df)))
                    else:
                        concatenated_values.append(pd.Series([str(part)] * len(df)))
                
                combined = concatenated_values[0]
                for val in concatenated_values[1:]:
                    combined = combined + separator + val
                
                df[column_name] = np.where(mask, combined, df[column_name])
                
            elif operator == 'copy_from':
                source_field = action['source_field']
                source_column = self.columns.get(source_field, source_field)
                df[column_name] = np.where(mask, df[source_column], df[column_name])
                
            else:
                raise ValueError(f"Unknown action operator: {operator}")
    
    def process_trades(self, df, apply_to_legacy_only=True):
        """
        Process the trade dataframe applying all configured rules using vectorized operations.
        
        Args:
            df (pd.DataFrame): Input dataframe with trade data
            apply_to_legacy_only (bool): If True, only apply rules to legacy risk system trades
            
        Returns:
            tuple: (processed_dataframe, changes_applied_list)
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Track changes for logging
        changes_applied = []
        
        # Create base legacy mask if needed
        if apply_to_legacy_only:
            target_risk_system = self.config['global_settings']['target_risk_system']
            legacy_mask = (processed_df[self.columns['risk_system']] == target_risk_system).values
        else:
            legacy_mask = np.ones(len(processed_df), dtype=bool)
        
        # Process each system configuration
        for system_name, system_config in self.config['systems'].items():
            # Create system mask
            system_mask = (processed_df[self.columns['system']] == system_name).values
            
            # Combine with legacy mask
            base_mask = system_mask & legacy_mask
            
            # Skip if no trades for this system
            if not np.any(base_mask):
                continue
            
            # Apply each rule for this system
            for rule in system_config['rules']:
                try:
                    # Build condition mask
                    condition_mask = self._build_condition_mask(processed_df, rule['conditions'])
                    
                    # Combine all masks
                    final_mask = base_mask & condition_mask
                    
                    # Apply actions if any trades match
                    if np.any(final_mask):
                        trades_affected = np.sum(final_mask)
                        self._apply_actions(processed_df, final_mask, rule['actions'])
                        
                        changes_applied.append({
                            'system': system_name,
                            'rule_name': rule['name'],
                            'rule_description': rule['description'],
                            'trades_affected': trades_affected
                        })
                        
                except Exception as e:
                    print(f"Error applying rule '{rule['name']}' for system {system_name}: {str(e)}")
                    continue
        
        return processed_df, changes_applied
    
    def process_system_trades(self, df, system_name, apply_to_legacy_only=True):
        """
        Process trades for a specific system only using vectorized operations.
        
        Args:
            df (pd.DataFrame): Input dataframe with trade data
            system_name (str): Name of the system to process
            apply_to_legacy_only (bool): If True, only apply rules to legacy risk system trades
            
        Returns:
            tuple: (processed_dataframe, changes_applied_list)
        """
        if system_name not in self.config['systems']:
            raise ValueError(f"No configuration found for system '{system_name}'")
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        changes_applied = []
        
        # Create base masks
        system_mask = (processed_df[self.columns['system']] == system_name).values
        
        if apply_to_legacy_only:
            target_risk_system = self.config['global_settings']['target_risk_system']
            legacy_mask = (processed_df[self.columns['risk_system']] == target_risk_system).values
            base_mask = system_mask & legacy_mask
        else:
            base_mask = system_mask
        
        # Get system configuration
        system_config = self.config['systems'][system_name]
        
        # Apply each rule for this system
        for rule in system_config['rules']:
            try:
                # Build condition mask
                condition_mask = self._build_condition_mask(processed_df, rule['conditions'])
                
                # Combine all masks
                final_mask = base_mask & condition_mask
                
                # Apply actions if any trades match
                if np.any(final_mask):
                    trades_affected = np.sum(final_mask)
                    self._apply_actions(processed_df, final_mask, rule['actions'])
                    
                    changes_applied.append({
                        'system': system_name,
                        'rule_name': rule['name'],
                        'rule_description': rule['description'],
                        'trades_affected': trades_affected
                    })
                    
            except Exception as e:
                print(f"Error applying rule '{rule['name']}' for system {system_name}: {str(e)}")
                continue
        
        return processed_df, changes_applied
    
    def add_custom_rule(self, system_name, rule_name, conditions, actions, description):
        """
        Add a custom rule to an existing system configuration.
        
        Args:
            system_name (str): Name of the system to add the rule to
            rule_name (str): Unique name for the rule
            conditions (list): List of condition dictionaries
            actions (list): List of action dictionaries
            description (str): Description of the rule
        """
        if system_name not in self.config['systems']:
            self.config['systems'][system_name] = {'rules': []}
        
        new_rule = {
            'name': rule_name,
            'description': description,
            'conditions': conditions,
            'actions': actions
        }
        
        self.config['systems'][system_name]['rules'].append(new_rule)
    
    def export_config_yaml(self):
        """
        Export current configuration as YAML string.
        
        Returns:
            str: YAML configuration
        """
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)
    
    def get_processing_summary(self):
        """
        Get a summary of all configured processing rules.
        
        Returns:
            dict: Summary of rules by system
        """
        summary = {}
        for system_name, system_config in self.config['systems'].items():
            summary[system_name] = [
                {
                    'name': rule['name'],
                    'description': rule['description'],
                    'conditions_count': len(rule['conditions']),
                    'actions_count': len(rule['actions'])
                }
                for rule in system_config['rules']
            ]
        return summary


# Example YAML configuration string
EXAMPLE_YAML_CONFIG = """
global_settings:
  target_risk_system: 'legacy'

systems:
  A:
    rules:
      - name: 'book_and_system_rule'
        description: 'Book in (booka, boookb) and system is B-E'
        conditions:
          - field: 'book'
            operator: 'in'
            value: ['booka', 'boookb']
          - field: 'system'
            operator: 'eq'
            value: 'B-E'
        actions:
          - field: 'cpty_code'
            operator: 'set'
            value: 'X1'
          - field: 'cpty_name'
            operator: 'set'
            value: 'x2'
            
      - name: 'isin_extraction_rule'
        description: 'Extract ISIN for Own issuance'
        conditions:
          - field: 'risk_group'
            operator: 'eq'
            value: 'Own issuance'
        actions:
          - field: 'underlying'
            operator: 'regex_extract_transform'
            source_field: 'full_leg_ext_id'
            pattern: '^([A-Z]{2}[A-Z0-9]{10})_.*_BOND_(?:BUY|SELL)_?'
            group: 1
            prefix: 'B:'

  B:
    rules:
      - name: 'abc_search_rule'
        description: 'FullLegExtId contains ABC'
        conditions:
          - field: 'full_leg_ext_id'
            operator: 'contains'
            value: 'ABC'
        actions:
          - field: 'cpty_code'
            operator: 'set'
            value: 'ABC'
          - field: 'cpty_name'
            operator: 'set'
            value: 'ABC_2'
"""

# Example usage:
if __name__ == "__main__":
    # Example DataFrame structure
    sample_data = {
        'RiskSystem': ['legacy', 'legacy', 'legacy', 'new', 'legacy'],
        'System': ['A', 'A', 'B', 'A', 'C'],
        'Book': ['booka', 'bookc', 'book1', 'booka', 'book2'],
        'CptyCode': ['OLD1', 'OLD2', 'OLD3', 'OLD4', 'OLD5'],
        'CptyName': ['Old Name 1', 'Old Name 2', 'Old Name 3', 'Old Name 4', 'Old Name 5'],
        'RiskGroup': ['Group1', 'Y', 'Group2', 'Own issuance', 'Z'],
        'FullLegExtId': ['LEG1', 'LEG2', 'ABC_123', 'US123456789_1000000_BOND_BUY_', 'AAAA_456'],
        'Underlying': ['UND1', 'UND2', 'UND3', 'UND4', 'UND5']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize processor with YAML config
    processor = TradeDataProcessor(config_yaml=EXAMPLE_YAML_CONFIG)
    
    # Or initialize with dictionary config
    # processor = TradeDataProcessor()  # Uses default config
    
    # Process all trades
    processed_df, changes = processor.process_trades(df)
    
    print("Original DataFrame:")
    print(df)
    print("\nProcessed DataFrame:")
    print(processed_df)
    print("\nChanges Applied:")
    for change in changes:
        print(f"System {change['system']}: {change['rule_name']} - {change['trades_affected']} trades affected")
    
    # Export current config as YAML
    print("\nCurrent Configuration (YAML):")
    print(processor.export_config_yaml())
