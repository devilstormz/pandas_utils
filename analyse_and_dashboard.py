import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import the TradeReconciliation class (assuming it's in the same file or imported)
class TradeReconciliation:
    """
    A class for reconciling trade data between two different trading systems.
    Provides detailed analysis with pivot tables and break identification.
    """
    
    def __init__(self, dataset1: pd.DataFrame, dataset2: pd.DataFrame, 
                 dataset1_name: str = "Dataset1", dataset2_name: str = "Dataset2"):
        """
        Initialize the reconciliation class with two datasets.
        """
        self.dataset1 = dataset1.copy()
        self.dataset2 = dataset2.copy()
        self.dataset1_name = dataset1_name
        self.dataset2_name = dataset2_name
        
        # Validate required columns
        required_cols = ['trade_id', 'booking_system', 'currency', 'notional', 'underlying_static']
        self._validate_columns(required_cols)
        
        # Store results
        self.pivot_results = {}
        self.break_results = {}
        
    def _validate_columns(self, required_cols):
        """Validate that both datasets have required columns."""
        for col in required_cols:
            if col not in self.dataset1.columns:
                raise ValueError(f"Column '{col}' missing from {self.dataset1_name}")
            if col not in self.dataset2.columns:
                raise ValueError(f"Column '{col}' missing from {self.dataset2_name}")
    
    def _create_pivot_summary(self, df: pd.DataFrame, index_cols, dataset_name: str):
        """Create pivot summary with sum of notional and count of trades."""
        pivot = df.groupby(index_cols).agg({
            'notional': ['sum', 'count']
        }).round(2)
        
        # Flatten column names
        pivot.columns = [f'{dataset_name}_{col[1]}' if col[1] else f'{dataset_name}_{col[0]}'
                        for col in pivot.columns]
        pivot.columns = [col.replace('sum', 'notional_sum').replace('count', 'trade_count') 
                        for col in pivot.columns]
        
        return pivot.reset_index()
    
    def _combine_and_calculate_differences(self, pivot1: pd.DataFrame, pivot2: pd.DataFrame, index_cols):
        """Combine pivots and calculate differences."""
        # Merge the two pivots
        combined = pd.merge(pivot1, pivot2, on=index_cols, how='outer').fillna(0)
        
        # Calculate differences
        notional_col1 = f'{self.dataset1_name}_notional_sum'
        notional_col2 = f'{self.dataset2_name}_notional_sum'
        count_col1 = f'{self.dataset1_name}_trade_count'
        count_col2 = f'{self.dataset2_name}_trade_count'
        
        combined['notional_difference'] = combined[notional_col1] - combined[notional_col2]
        combined['trade_count_difference'] = combined[count_col1] - combined[count_col2]
        
        # Add break type classification
        combined['break_type'] = combined.apply(self._classify_break_type, axis=1)
        
        # Round for display
        combined = combined.round(2)
        
        return combined
    
    def _classify_break_type(self, row):
        """Classify the type of break."""
        notional_diff = row['notional_difference']
        count_diff = row['trade_count_difference']
        
        if notional_diff == 0 and count_diff == 0:
            return 'No Break'
        elif notional_diff != 0 and count_diff == 0:
            return 'Notional Only'
        elif notional_diff == 0 and count_diff != 0:
            return 'Count Only'
        else:
            return 'Both Notional & Count'
    
    def _identify_breaks(self, df: pd.DataFrame):
        """Identify rows where there are differences (breaks)."""
        breaks = df[
            (df['notional_difference'] != 0) | 
            (df['trade_count_difference'] != 0)
        ].copy()
        return breaks
    
    def generate_currency_pivot(self):
        """Generate pivot by currency with metrics."""
        pivot1 = self._create_pivot_summary(self.dataset1, ['currency'], self.dataset1_name)
        pivot2 = self._create_pivot_summary(self.dataset2, ['currency'], self.dataset2_name)
        
        combined = self._combine_and_calculate_differences(pivot1, pivot2, ['currency'])
        breaks = self._identify_breaks(combined)
        
        self.pivot_results['currency'] = combined
        self.break_results['currency'] = breaks
        
        return combined, breaks
    
    def generate_currency_counterparty_pivot(self):
        """Generate detailed pivot by currency x counterparty."""
        index_cols = ['currency', 'underlying_static']
        
        pivot1 = self._create_pivot_summary(self.dataset1, index_cols, self.dataset1_name)
        pivot2 = self._create_pivot_summary(self.dataset2, index_cols, self.dataset2_name)
        
        combined = self._combine_and_calculate_differences(pivot1, pivot2, index_cols)
        breaks = self._identify_breaks(combined)
        
        self.pivot_results['currency_counterparty'] = combined
        self.break_results['currency_counterparty'] = breaks
        
        return combined, breaks
    
    def generate_currency_system_counterparty_pivot(self):
        """Generate detailed pivot by currency x system x counterparty."""
        index_cols = ['currency', 'booking_system', 'underlying_static']
        
        pivot1 = self._create_pivot_summary(self.dataset1, index_cols, self.dataset1_name)
        pivot2 = self._create_pivot_summary(self.dataset2, index_cols, self.dataset2_name)
        
        combined = self._combine_and_calculate_differences(pivot1, pivot2, index_cols)
        breaks = self._identify_breaks(combined)
        
        self.pivot_results['currency_system_counterparty'] = combined
        self.break_results['currency_system_counterparty'] = breaks
        
        return combined, breaks
    
    def generate_all_pivots(self):
        """Generate all required pivots."""
        self.generate_currency_pivot()
        self.generate_currency_counterparty_pivot() 
        self.generate_currency_system_counterparty_pivot()
        
        return self.pivot_results
    
    def get_summary_statistics(self):
        """Get high-level summary statistics."""
        stats = []
        
        for dataset_name, dataset in [(self.dataset1_name, self.dataset1), 
                                    (self.dataset2_name, self.dataset2)]:
            stat = {
                'Dataset': dataset_name,
                'Total_Trades': len(dataset),
                'Total_Notional': dataset['notional'].sum(),
                'Unique_Currencies': dataset['currency'].nunique(),
                'Unique_Systems': dataset['booking_system'].nunique(),
                'Unique_Counterparties': dataset['underlying_static'].nunique()
            }
            stats.append(stat)
        
        # Add difference row
        diff_stat = {
            'Dataset': 'Difference',
            'Total_Trades': len(self.dataset1) - len(self.dataset2),
            'Total_Notional': self.dataset1['notional'].sum() - self.dataset2['notional'].sum(),
            'Unique_Currencies': self.dataset1['currency'].nunique() - self.dataset2['currency'].nunique(),
            'Unique_Systems': self.dataset1['booking_system'].nunique() - self.dataset2['booking_system'].nunique(),
            'Unique_Counterparties': self.dataset1['underlying_static'].nunique() - self.dataset2['underlying_static'].nunique()
        }
        stats.append(diff_stat)
        
        return pd.DataFrame(stats).round(2)


def create_sample_data():
    """Create sample datasets for testing."""
    np.random.seed(42)
    
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']
    systems = ['SystemA', 'SystemB', 'SystemC']
    counterparties = ['CP001', 'CP002', 'CP003', 'CP004', 'CP005', 'CP006', 'CP007']
    
    # Dataset 1
    n1 = 150
    dataset1 = pd.DataFrame({
        'trade_id': [f'T{i:04d}' for i in range(1, n1+1)],
        'booking_system': np.random.choice(systems, n1),
        'currency': np.random.choice(currencies, n1),
        'notional': np.random.uniform(10000, 2000000, n1).round(2),
        'underlying_static': np.random.choice(counterparties, n1)
    })
    
    # Dataset 2 - similar but with some differences to create breaks
    n2 = 140
    dataset2 = pd.DataFrame({
        'trade_id': [f'T{i:04d}' for i in range(1, n2+1)],
        'booking_system': np.random.choice(systems, n2),
        'currency': np.random.choice(currencies, n2),
        'notional': np.random.uniform(10000, 2000000, n2).round(2),
        'underlying_static': np.random.choice(counterparties, n2)
    })
    
    # Introduce some systematic differences to create meaningful breaks
    mask = dataset2['currency'] == 'USD'
    dataset2.loc[mask, 'notional'] *= 1.1  # 10% difference in USD notionals
    
    return dataset1, dataset2


def create_break_analysis_charts(reconciler):
    """Create charts for break analysis."""
    charts = {}
    
    # Get all break data
    all_breaks = []
    for key, breaks_df in reconciler.break_results.items():
        if not breaks_df.empty:
            breaks_copy = breaks_df.copy()
            breaks_copy['analysis_level'] = key
            all_breaks.append(breaks_copy)
    
    if not all_breaks:
        return charts
    
    combined_breaks = pd.concat(all_breaks, ignore_index=True)
    
    # Chart 1: Break Type Distribution
    break_type_counts = combined_breaks['break_type'].value_counts()
    fig_break_types = px.pie(
        values=break_type_counts.values,
        names=break_type_counts.index,
        title="Distribution of Break Types"
    )
    charts['break_types'] = fig_break_types
    
    # Chart 2: Top 10 Counterparty Breaks (by absolute notional difference)
    if 'underlying_static' in combined_breaks.columns:
        cp_breaks = combined_breaks.groupby('underlying_static').agg({
            'notional_difference': lambda x: x.abs().sum(),
            'trade_count_difference': lambda x: x.abs().sum()
        }).reset_index()
        
        cp_breaks_top10 = cp_breaks.nlargest(10, 'notional_difference')
        
        fig_cp_breaks = px.bar(
            cp_breaks_top10,
            x='underlying_static',
            y='notional_difference',
            title="Top 10 Counterparties by Absolute Notional Differences",
            labels={'underlying_static': 'Counterparty', 'notional_difference': 'Absolute Notional Difference'}
        )
        # fig_cp_breaks.update_xaxis(tickangle=45)
        charts['top_counterparty_breaks'] = fig_cp_breaks
    
    # Chart 3: Currency-wise breaks
    curr_breaks = combined_breaks.groupby('currency').agg({
        'notional_difference': lambda x: x.abs().sum(),
        'trade_count_difference': lambda x: x.abs().sum()
    }).reset_index()
    
    fig_curr_breaks = px.bar(
        curr_breaks,
        x='currency',
        y=['notional_difference', 'trade_count_difference'],
        title="Break Analysis by Currency",
        barmode='group'
    )
    charts['currency_breaks'] = fig_curr_breaks
    
    return charts


def create_summary_charts(reconciler):
    """Create summary charts for the dashboard."""
    charts = {}
    
    # Chart 1: Dataset Comparison - Total Values
    summary_stats = reconciler.get_summary_statistics()
    datasets_only = summary_stats[summary_stats['Dataset'] != 'Difference']
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        name='Total Trades',
        x=datasets_only['Dataset'],
        y=datasets_only['Total_Trades'],
        yaxis='y'
    ))
    fig_comparison.add_trace(go.Bar(
        name='Total Notional (M)',
        x=datasets_only['Dataset'],
        y=datasets_only['Total_Notional'] / 1000000,
        yaxis='y2'
    ))
    
    fig_comparison.update_layout(
        title='Dataset Comparison: Trades vs Notional',
        yaxis=dict(title='Number of Trades', side='left'),
        yaxis2=dict(title='Total Notional (Millions)', side='right', overlaying='y'),
        barmode='group'
    )
    charts['dataset_comparison'] = fig_comparison
    
    # Chart 2: Currency Distribution
    curr_dist1 = reconciler.dataset1.groupby('currency')['notional'].sum() / 1000000
    curr_dist2 = reconciler.dataset2.groupby('currency')['notional'].sum() / 1000000
    
    fig_curr_dist = go.Figure()
    fig_curr_dist.add_trace(go.Bar(
        name=reconciler.dataset1_name,
        x=curr_dist1.index,
        y=curr_dist1.values
    ))
    fig_curr_dist.add_trace(go.Bar(
        name=reconciler.dataset2_name,
        x=curr_dist2.index,
        y=curr_dist2.values
    ))
    
    fig_curr_dist.update_layout(
        title='Notional Distribution by Currency (Millions)',
        xaxis_title='Currency',
        yaxis_title='Notional (Millions)',
        barmode='group'
    )
    charts['currency_distribution'] = fig_curr_dist
    
    return charts


def filter_dataframe(df, filters):
    """Apply filters to dataframe."""
    filtered_df = df.copy()
    
    for column, values in filters.items():
        if values and column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column].isin(values)]
    
    return filtered_df


def main():
    """Main Streamlit dashboard function."""
    st.set_page_config(
        page_title="Trade Reconciliation Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Trade Reconciliation Dashboard")
    st.markdown("---")
    
    # Sidebar for data loading and configuration
    st.sidebar.header("Configuration")
    
    # Option to use sample data or upload files
    data_source = st.sidebar.radio(
        "Data Source",
        ["Use Sample Data", "Upload CSV Files"]
    )
    
    if data_source == "Use Sample Data":
        # Use sample data
        if st.sidebar.button("Generate Sample Data"):
            dataset1, dataset2 = create_sample_data()
            st.session_state['dataset1'] = dataset1
            st.session_state['dataset2'] = dataset2
            st.session_state['dataset1_name'] = "TradingSystem1"
            st.session_state['dataset2_name'] = "TradingSystem2"
            st.sidebar.success("Sample data generated!")
    
    else:
        # File upload
        st.sidebar.subheader("Upload Data Files")
        file1 = st.sidebar.file_uploader("Upload Dataset 1 (CSV)", type=['csv'], key="file1")
        file2 = st.sidebar.file_uploader("Upload Dataset 2 (CSV)", type=['csv'], key="file2")
        
        dataset1_name = st.sidebar.text_input("Dataset 1 Name", value="Dataset1")
        dataset2_name = st.sidebar.text_input("Dataset 2 Name", value="Dataset2")
        
        if file1 and file2:
            try:
                dataset1 = pd.read_csv(file1)
                dataset2 = pd.read_csv(file2)
                st.session_state['dataset1'] = dataset1
                st.session_state['dataset2'] = dataset2
                st.session_state['dataset1_name'] = dataset1_name
                st.session_state['dataset2_name'] = dataset2_name
                st.sidebar.success("Files uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading files: {str(e)}")
    
    # Check if data is loaded
    if 'dataset1' not in st.session_state or 'dataset2' not in st.session_state:
        st.info("Please load data using the sidebar configuration.")
        return
    
    # Initialize reconciler
    try:
        reconciler = TradeReconciliation(
            st.session_state['dataset1'],
            st.session_state['dataset2'],
            st.session_state['dataset1_name'],
            st.session_state['dataset2_name']
        )
        
        # Generate all analysis
        reconciler.generate_all_pivots()
        
    except Exception as e:
        st.error(f"Error initializing reconciliation: {str(e)}")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Summary", "🔍 Detailed Analysis", "⚠️ Breaks Analysis", 
        "📊 Charts", "📋 Raw Data"
    ])
    
    # Tab 1: Summary
    with tab1:
        st.header("Summary Statistics")
        
        # Display summary statistics
        summary_stats = reconciler.get_summary_statistics()
        
        # Create metrics for key statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Trades Difference",
                f"{summary_stats.iloc[2]['Total_Trades']:,.0f}",
                help="Difference in number of trades between datasets"
            )
        
        with col2:
            st.metric(
                "Total Notional Difference",
                f"${summary_stats.iloc[2]['Total_Notional']:,.0f}",
                help="Difference in total notional between datasets"
            )
        
        with col3:
            total_breaks = sum(len(breaks) for breaks in reconciler.break_results.values())
            st.metric("Total Breaks", f"{total_breaks:,}")
        
        with col4:
            break_pct = (total_breaks / max(len(reconciler.dataset1), len(reconciler.dataset2))) * 100
            st.metric("Break Percentage", f"{break_pct:.1f}%")
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Summary charts
        summary_charts = create_summary_charts(reconciler)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'dataset_comparison' in summary_charts:
                st.plotly_chart(summary_charts['dataset_comparison'], use_container_width=True, key="summary_dataset_comp")
        
        with col2:
            if 'currency_distribution' in summary_charts:
                st.plotly_chart(summary_charts['currency_distribution'], use_container_width=True, key="summary_currency_dist")
    
    # Tab 2: Detailed Analysis
    with tab2:
        st.header("Detailed Pivot Analysis")
        
        analysis_level = st.selectbox(
            "Select Analysis Level",
            ["Currency", "Currency × Counterparty", "Currency × System × Counterparty"]
        )
        
        # Map selection to key
        level_mapping = {
            "Currency": "currency",
            "Currency × Counterparty": "currency_counterparty",
            "Currency × System × Counterparty": "currency_system_counterparty"
        }
        
        selected_key = level_mapping[analysis_level]
        
        if selected_key in reconciler.pivot_results:
            df = reconciler.pivot_results[selected_key]
            
            # Add filters
            st.subheader("Filters")
            filters = {}
            
            filter_cols = st.columns(min(4, len([col for col in df.columns if col in ['currency', 'booking_system', 'underlying_static']])))
            
            col_idx = 0
            for col in ['currency', 'booking_system', 'underlying_static']:
                if col in df.columns and col_idx < len(filter_cols):
                    with filter_cols[col_idx]:
                        unique_values = df[col].unique()
                        selected_values = st.multiselect(
                            f"Filter by {col.replace('_', ' ').title()}",
                            options=unique_values,
                            key=f"filter_{col}_{selected_key}"
                        )
                        if selected_values:
                            filters[col] = selected_values
                    col_idx += 1
            
            # Apply filters
            filtered_df = filter_dataframe(df, filters)
            
            # Display filtered data
            st.subheader(f"{analysis_level} Analysis")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"{selected_key}_analysis.csv",
                mime="text/csv"
            )
    
    # Tab 3: Breaks Analysis
    with tab3:
        st.header("Breaks Analysis")
        
        # Get currency x counterparty breaks for detailed analysis
        currency_cp_breaks = reconciler.break_results.get('currency_counterparty', pd.DataFrame())
        system_cp_breaks = reconciler.break_results.get('currency_system_counterparty', pd.DataFrame())
        
        # Get all breaks for analysis
        all_breaks_data = []
        for key, breaks_df in reconciler.break_results.items():
            if not breaks_df.empty:
                temp_df = breaks_df.copy()
                temp_df['analysis_level'] = key.replace('_', ' ').title()
                all_breaks_data.append(temp_df)
        
        if all_breaks_data:
            combined_breaks = pd.concat(all_breaks_data, ignore_index=True)
            
            # Break type analysis
            st.subheader("Break Type Distribution")
            break_type_counts = combined_breaks['break_type'].value_counts()
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(break_type_counts.to_frame().reset_index())
            
            with col2:
                fig_pie = px.pie(
                    values=break_type_counts.values,
                    names=break_type_counts.index,
                    title="Break Types Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="break_type_pie")
            
            # Currency x Counterparty Break Analysis with Filters
            st.subheader("Currency × Counterparty Break Analysis")
            
            if not currency_cp_breaks.empty:
                # Add filters for currency x counterparty analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_currencies = st.multiselect(
                        "Filter by Currency",
                        options=currency_cp_breaks['currency'].unique(),
                        key="breaks_currency_filter"
                    )
                
                with col2:
                    selected_counterparties = st.multiselect(
                        "Filter by Counterparty",
                        options=currency_cp_breaks['underlying_static'].unique(),
                        key="breaks_counterparty_filter"
                    )
                
                # Apply filters
                filtered_cp_breaks = currency_cp_breaks.copy()
                if selected_currencies:
                    filtered_cp_breaks = filtered_cp_breaks[filtered_cp_breaks['currency'].isin(selected_currencies)]
                if selected_counterparties:
                    filtered_cp_breaks = filtered_cp_breaks[filtered_cp_breaks['underlying_static'].isin(selected_counterparties)]
                
                # Display filtered chart
                if not filtered_cp_breaks.empty:
                    fig_cp_bar = px.bar(
                        filtered_cp_breaks,
                        x='currency',
                        y='notional_difference',
                        color='underlying_static',
                        title="Currency × Counterparty Breaks - Notional Differences",
                        labels={'notional_difference': 'Notional Difference', 'underlying_static': 'Counterparty'}
                    )
                    st.plotly_chart(fig_cp_bar, use_container_width=True, key="currency_cp_breaks_chart")
                else:
                    st.info("No data available with current filters.")
            
            # Maximum Detail View: Currency × System × Counterparty
            st.subheader("Maximum Detail: Currency × System × Counterparty Breaks")
            
            if not system_cp_breaks.empty:
                # Additional filters for the most detailed view
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    detail_currencies = st.multiselect(
                        "Filter by Currency (Detailed)",
                        options=system_cp_breaks['currency'].unique(),
                        key="detail_currency_filter"
                    )
                
                with col2:
                    detail_systems = st.multiselect(
                        "Filter by Booking System",
                        options=system_cp_breaks['booking_system'].unique(),
                        key="detail_system_filter"
                    )
                
                with col3:
                    detail_counterparties = st.multiselect(
                        "Filter by Counterparty (Detailed)",
                        options=system_cp_breaks['underlying_static'].unique(),
                        key="detail_counterparty_filter"
                    )
                
                # Apply detailed filters
                detailed_breaks = system_cp_breaks.copy()
                if detail_currencies:
                    detailed_breaks = detailed_breaks[detailed_breaks['currency'].isin(detail_currencies)]
                if detail_systems:
                    detailed_breaks = detailed_breaks[detailed_breaks['booking_system'].isin(detail_systems)]
                if detail_counterparties:
                    detailed_breaks = detailed_breaks[detailed_breaks['underlying_static'].isin(detail_counterparties)]
                
                # Display all columns with maximum detail
                st.dataframe(detailed_breaks, use_container_width=True)
                
                # Summary statistics for detailed view
                if not detailed_breaks.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Detailed Breaks", len(detailed_breaks))
                    
                    with col2:
                        total_notional_diff = detailed_breaks['notional_difference'].abs().sum()
                        st.metric("Total Abs. Notional Diff", f"${total_notional_diff:,.0f}")
                    
                    with col3:
                        total_count_diff = detailed_breaks['trade_count_difference'].abs().sum()
                        st.metric("Total Abs. Count Diff", f"{total_count_diff:,.0f}")
                    
                    with col4:
                        avg_notional_diff = detailed_breaks['notional_difference'].abs().mean()
                        st.metric("Avg. Abs. Notional Diff", f"${avg_notional_diff:,.0f}")
            
            else:
                st.info("No detailed breaks found at Currency × System × Counterparty level.")
            
            # General breaks table (kept as before)
            st.subheader("All Breaks Summary")
            
            # Filters for breaks
            break_level = st.selectbox(
                "Filter by Analysis Level",
                options=['All'] + list(combined_breaks['analysis_level'].unique())
            )
            
            break_type_filter = st.selectbox(
                "Filter by Break Type",
                options=['All'] + list(combined_breaks['break_type'].unique())
            )
            
            filtered_breaks = combined_breaks.copy()
            if break_level != 'All':
                filtered_breaks = filtered_breaks[filtered_breaks['analysis_level'] == break_level]
            if break_type_filter != 'All':
                filtered_breaks = filtered_breaks[filtered_breaks['break_type'] == break_type_filter]
            
            st.dataframe(filtered_breaks, use_container_width=True)
            
        else:
            st.info("No breaks found in the analysis.")
    
    # Tab 4: Charts
    with tab4:
        st.header("Break Statistics Visualization")
        
        # Get currency x counterparty breaks for stacked bar charts
        currency_cp_breaks = reconciler.break_results.get('currency_counterparty', pd.DataFrame())
        
        if not currency_cp_breaks.empty:
            st.subheader("Currency × Counterparty Break Statistics")
            
            # Create stacked bar chart for notional differences
            fig_stacked_notional = px.bar(
                currency_cp_breaks,
                x='currency',
                y='notional_difference',
                color='underlying_static',
                title="Notional Differences by Currency × Counterparty",
                labels={'notional_difference': 'Notional Difference', 'underlying_static': 'Counterparty'},
                text='notional_difference'
            )
            fig_stacked_notional.update_traces(texttemplate='%{text:.2s}', textposition='inside')
            fig_stacked_notional.update_layout(barmode='stack')
            st.plotly_chart(fig_stacked_notional, use_container_width=True, key="stacked_notional_chart")
            
            # Create stacked bar chart for trade count differences
            fig_stacked_count = px.bar(
                currency_cp_breaks,
                x='currency',
                y='trade_count_difference',
                color='underlying_static',
                title="Trade Count Differences by Currency × Counterparty",
                labels={'trade_count_difference': 'Trade Count Difference', 'underlying_static': 'Counterparty'},
                text='trade_count_difference'
            )
            fig_stacked_count.update_traces(texttemplate='%{text}', textposition='inside')
            fig_stacked_count.update_layout(barmode='stack')
            st.plotly_chart(fig_stacked_count, use_container_width=True, key="stacked_count_chart")
            
            # Break type distribution by currency
            break_type_by_currency = currency_cp_breaks.groupby(['currency', 'break_type']).size().reset_index(name='count')
            
            fig_break_type_stack = px.bar(
                break_type_by_currency,
                x='currency',
                y='count',
                color='break_type',
                title="Break Type Distribution by Currency",
                labels={'count': 'Number of Breaks'},
                text='count'
            )
            fig_break_type_stack.update_traces(texttemplate='%{text}', textposition='inside')
            fig_break_type_stack.update_layout(barmode='stack')
            st.plotly_chart(fig_break_type_stack, use_container_width=True, key="break_type_stack_chart")
            
            # Absolute values stacked chart for better visualization
            currency_cp_breaks_abs = currency_cp_breaks.copy()
            currency_cp_breaks_abs['abs_notional_difference'] = currency_cp_breaks_abs['notional_difference'].abs()
            currency_cp_breaks_abs['abs_count_difference'] = currency_cp_breaks_abs['trade_count_difference'].abs()
            
            fig_abs_notional = px.bar(
                currency_cp_breaks_abs,
                x='currency',
                y='abs_notional_difference',
                color='underlying_static',
                title="Absolute Notional Differences by Currency × Counterparty",
                labels={'abs_notional_difference': 'Absolute Notional Difference', 'underlying_static': 'Counterparty'},
                text='abs_notional_difference'
            )
            fig_abs_notional.update_traces(texttemplate='%{text:.2s}', textposition='inside')
            fig_abs_notional.update_layout(barmode='stack')
            st.plotly_chart(fig_abs_notional, use_container_width=True, key="abs_stacked_notional_chart")
            
        else:
            st.info("No Currency × Counterparty breaks available for visualization.")
    
    # Tab 5: Raw Data
    with tab5:
        st.header("Raw Dataset Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{reconciler.dataset1_name} ({len(reconciler.dataset1)} records)")
            st.dataframe(reconciler.dataset1, use_container_width=True)
        
        with col2:
            st.subheader(f"{reconciler.dataset2_name} ({len(reconciler.dataset2)} records)")
            st.dataframe(reconciler.dataset2, use_container_width=True)
    
    # Sidebar export options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Export Options")
    
    if st.sidebar.button("Export Analysis to Excel"):
        try:
            reconciler.export_to_excel('trade_reconciliation_dashboard_export.xlsx')
            st.sidebar.success("Analysis exported to Excel!")
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")


if __name__ == "__main__":
    main()