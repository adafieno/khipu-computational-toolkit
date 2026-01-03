"""
Interactive Khipu Analysis Dashboard

Streamlit web application for exploring khipu data with:
- Real-time filtering and selection
- Multi-level drill-down (cluster → provenance → khipu → cord)
- Interactive visualizations with Plotly
- Data export capabilities

Usage: streamlit run scripts/dashboard_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

# Initialize config
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Khipu Analysis Dashboard",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

PROVENANCE_COORDS = {
    # Major archaeological sites
    'Pachacamac': (-12.2667, -76.9167),
    'Incahuasi': (-13.0333, -75.7667),
    'Ica': (-14.0678, -75.7286),
    'Leymebamba': (-6.6833, -77.7833),
    'Huaquerones': (-14.1167, -75.7333),
    'Nazca': (-14.8333, -74.9333),
    'Armatambo, Huaca San Pedro': (-12.1667, -77.0167),
    'Valle de Ica Hacienda Callango Ocucaje': (-14.4833, -75.6667),
    'Chachapoyas': (-6.2308, -77.8691),
    'Puruchuco': (-12.0167, -76.9833),
    'Cajamarquilla': (-11.9833, -76.9167),
    'Cusco': (-13.5319, -71.9675),
    'Lima': (-12.0464, -77.0428),
    # Additional sites and regions
    'unknown': None,  # Exclude from map
    '': None,  # Exclude from map
    'Chillon Valley': (-11.8833, -77.0500),
    'Chicama': (-7.8447, -79.1469),
    'Ancon': (-11.7667, -77.1833),
    'Chancay': (-11.5667, -77.2667),
    'Chachapoyas region': (-6.2308, -77.8691),
    'Ica Valley': (-14.0678, -75.7286),
    'Cuzco': (-13.5319, -71.9675),  # Alternative spelling
}


def get_coords(provenance):
    """Get coordinates for a provenance, with fuzzy matching."""
    if not provenance or pd.isna(provenance) or str(provenance).strip() == '':
        return None, None

    # Direct match
    if provenance in PROVENANCE_COORDS:
        coords = PROVENANCE_COORDS[provenance]
        return coords if coords else (None, None)

    # Fuzzy matching for common patterns
    prov_lower = str(provenance).lower().strip()

    # Return None for generic/unknown values
    if prov_lower in ['unknown', 'peru', 'coast', '']:
        return None, None

    # Specific pattern matching (order matters - most specific first)
    if 'pachacamac' in prov_lower:
        return (-12.2667, -76.9167)
    elif 'incahuasi' in prov_lower or 'inkawasi' in prov_lower:
        return (-13.0333, -75.7667)
    elif 'ica' in prov_lower or 'callango' in prov_lower or 'ocucaje' in prov_lower or 'pisco' in prov_lower:
        return (-14.0678, -75.7286)
    elif 'nazca' in prov_lower or 'nasca' in prov_lower or 'atarco' in prov_lower:
        return (-14.8333, -74.9333)
    elif 'leymebamba' in prov_lower or 'chachapoyas' in prov_lower:
        return (-6.2308, -77.8691)
    elif 'huaquerones' in prov_lower or 'huaqueros' in prov_lower:
        return (-14.1167, -75.7333)
    elif 'armatambo' in prov_lower or 'san pedro' in prov_lower:
        return (-12.1667, -77.0167)
    elif 'paracas' in prov_lower:
        return (-13.8333, -76.2500)
    elif 'lima' in prov_lower or 'ancon' in prov_lower or 'ancón' in prov_lower:
        return (-12.0464, -77.0428)
    elif 'chancay' in prov_lower or 'huando' in prov_lower or 'huaura' in prov_lower or 'huacho' in prov_lower:
        return (-11.5667, -77.2667)
    elif 'chillon' in prov_lower or 'chillón' in prov_lower:
        return (-11.8833, -77.0500)
    elif 'cusco' in prov_lower or 'cuzco' in prov_lower:
        return (-13.5319, -71.9675)
    elif 'chicama' in prov_lower or 'chiquitoy' in prov_lower:
        return (-7.8447, -79.1469)
    elif 'santa' in prov_lower and 'valley' in prov_lower:
        return (-8.9833, -78.6167)
    elif 'arica' in prov_lower or 'chile' in prov_lower:
        return (-18.4783, -70.3126)
    elif 'marquez' in prov_lower or 'márquez' in prov_lower:
        return (-14.5000, -75.5000)
    elif 'puruchuco' in prov_lower:
        return (-12.0167, -76.9833)
    elif 'cajamarquilla' in prov_lower:
        return (-11.9833, -76.9167)

    return None, None


@st.cache_data
def load_data():
    """Load all processed data files."""
    try:
        import sqlite3

        # Load CSV files (cluster_assignments already has structural features)
        clusters = pd.read_csv(
            config.get_processed_file(
                'cluster_assignments_kmeans.csv',
                phase=4))
        summation = pd.read_csv(
            config.get_processed_file(
                'summation_test_results.csv',
                phase=3))
        pca = pd.read_csv(
            config.get_processed_file(
                'cluster_pca_coordinates.csv',
                phase=4))

        # Merge data
        data = clusters.merge(
            summation, on='khipu_id', how='left'
        ).merge(
            pca, on='khipu_id', how='left'
        )

        # Try to load provenance from database if available
        try:
            import os
            db_path = config.get_database_path()
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                provenance = pd.read_sql_query(
                    "SELECT KHIPU_ID, PROVENANCE FROM khipu_main", conn)
                conn.close()
                data = data.merge(
                    provenance,
                    left_on='khipu_id',
                    right_on='KHIPU_ID',
                    how='left')
        except BaseException:
            pass

        # Add PROVENANCE column if it doesn't exist
        if 'PROVENANCE' not in data.columns:
            data['PROVENANCE'] = 'Unknown'
        else:
            data['PROVENANCE'] = data['PROVENANCE'].fillna('Unknown')

        return data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load data
data = load_data()

if data is None:
    st.stop()

# ==================== HEADER ====================
st.markdown(
    '<p class="main-header">🧶 Khipu Analysis Dashboard</p>',
    unsafe_allow_html=True)
st.markdown(
    "**612 Inka Khipus** from Harvard Database • **7 Archetypes** • **26.3% Summation Detection**")
st.markdown("---")

# ==================== SIDEBAR FILTERS ====================
st.sidebar.header("🔍 Filters")

# Cluster filter
selected_clusters = st.sidebar.multiselect(
    "Select Clusters",
    options=sorted(data['cluster'].unique()),
    default=sorted(data['cluster'].unique())
)

# Provenance filter
provenance_counts = data['PROVENANCE'].value_counts()
major_provenances = provenance_counts[provenance_counts >= 10].index.tolist()
selected_provenances = st.sidebar.multiselect(
    "Select Provenances (n≥10)",
    options=major_provenances,
    default=major_provenances[:5] if len(major_provenances) > 5 else major_provenances
)

# Size filter
size_range = st.sidebar.slider(
    "Khipu Size (nodes)",
    min_value=int(data['num_nodes'].min()),
    max_value=int(data['num_nodes'].max()),
    value=(int(data['num_nodes'].min()), int(data['num_nodes'].max()))
)

# Summation filter
summation_filter = st.sidebar.radio(
    "Summation Pattern",
    options=["All", "With Summation", "Without Summation"]
)

# Apply filters
filtered_data = data[
    data['cluster'].isin(selected_clusters) &
    data['PROVENANCE'].isin(selected_provenances) &
    (data['num_nodes'] >= size_range[0]) &
    (data['num_nodes'] <= size_range[1])
].copy()

if summation_filter == "With Summation":
    filtered_data = filtered_data[filtered_data['has_pendant_summation']]
elif summation_filter == "Without Summation":
    filtered_data = filtered_data[~filtered_data['has_pendant_summation']]

st.sidebar.markdown(
    f"**Filtered:** {len(filtered_data)}/{len(data)} khipus ({len(filtered_data)/len(data)*100:.1f}%)")

# ==================== KEY METRICS ====================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Khipus", len(filtered_data))
with col2:
    sum_rate = filtered_data['has_pendant_summation'].mean() * 100
    st.metric("Summation Rate", f"{sum_rate:.1f}%")
with col3:
    avg_size = filtered_data['num_nodes'].mean()
    st.metric("Avg Size", f"{avg_size:.0f} nodes")
with col4:
    unique_provs = filtered_data['PROVENANCE'].nunique()
    st.metric("Provenances", unique_provs)

st.markdown("---")

# ==================== MAIN VISUALIZATIONS ====================

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🗺️ Geographic", "🔬 Clusters", "📈 Features"])

# TAB 1: Overview
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        # PCA scatter plot
        fig_pca = px.scatter(
            filtered_data,
            x='pc1',
            y='pc2',
            color='cluster',
            hover_data=[
                'khipu_id',
                'PROVENANCE',
                'num_nodes'],
            title='Cluster Distribution (PCA)',
            labels={
                'pc1': 'PC1 (45.7% variance)',
                'pc2': 'PC2 (16.1% variance)',
                'cluster': 'Cluster'},
            color_continuous_scale='viridis')
        fig_pca.update_layout(height=500)
        st.plotly_chart(fig_pca, width="stretch")

    with col2:
        # Size vs Depth scatter
        fig_size = px.scatter(
            filtered_data,
            x='num_nodes',
            y='depth',
            color='cluster',
            size='avg_branching',
            hover_data=[
                'khipu_id',
                'PROVENANCE'],
            title='Size vs Hierarchy Depth',
            labels={
                'num_nodes': 'Size (nodes)',
                'depth': 'Depth (levels)',
                'cluster': 'Cluster'},
            color_continuous_scale='plasma')
        fig_size.update_layout(height=500)
        st.plotly_chart(fig_size, width="stretch")

    # Cluster distribution bar chart
    cluster_dist = filtered_data['cluster'].value_counts().sort_index()
    fig_bar = px.bar(
        x=cluster_dist.index,
        y=cluster_dist.values,
        labels={'x': 'Cluster', 'y': 'Number of Khipus'},
        title='Khipu Count by Cluster',
        color=cluster_dist.values,
        color_continuous_scale='Blues'
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, width="stretch")

# TAB 2: Geographic Analysis
with tab2:
    st.markdown("### Geographic Distribution Map")
    st.caption(
        "📍 Showing all khipus in the database (filters applied to charts below)")

    # Use full dataset for map (not filtered) to show complete geographic
    # distribution
    prov_stats = data.groupby('PROVENANCE').agg({
        'khipu_id': 'count',
        'has_pendant_summation': 'mean',
        'num_nodes': 'mean',
        'depth': 'mean'
    }).reset_index()
    prov_stats.columns = [
        'Provenance',
        'Count',
        'Summation_Rate',
        'Avg_Size',
        'Avg_Depth']

    # Add coordinates using fuzzy matching
    coords_list = prov_stats['Provenance'].apply(get_coords)
    prov_stats['Lat'] = coords_list.apply(lambda x: x[0])
    prov_stats['Lon'] = coords_list.apply(lambda x: x[1])
    prov_stats = prov_stats.dropna(subset=['Lat', 'Lon'])

    st.info(
        f"📍 Mapped {len(prov_stats)} locations with coordinates (Total khipus: {prov_stats['Count'].sum()})")

    # Create geographic scatter map
    if len(prov_stats) > 0:
        fig_map = px.scatter_geo(
            prov_stats,
            lat='Lat',
            lon='Lon',
            size='Count',
            color='Summation_Rate',
            hover_name='Provenance',
            hover_data={
                'Count': True,
                'Summation_Rate': ':.1%',
                'Avg_Size': ':.0f',
                'Avg_Depth': ':.1f',
                'Lat': False,
                'Lon': False
            },
            title='Khipu Distribution Across Peru',
            color_continuous_scale='RdYlGn',
            size_max=40,
            projection='natural earth',
        )
        fig_map.update_geos(
            center=dict(lat=-12, lon=-75),
            projection_scale=5,
            showland=True,
            landcolor='lightgray',
            coastlinecolor='white',
            showcountries=True,
            countrycolor='white'
        )
        fig_map.update_layout(height=500)
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.info("No geographic data available for selected filters")

    col1, col2 = st.columns(2)

    with col1:
        # Summation rate by provenance
        prov_sum = filtered_data.groupby('PROVENANCE').agg({
            'has_pendant_summation': 'mean',
            'khipu_id': 'count'
        }).reset_index()
        prov_sum.columns = ['Provenance', 'Summation Rate', 'Count']
        prov_sum = prov_sum.sort_values('Summation Rate', ascending=False)

        fig_prov_sum = px.bar(
            prov_sum,
            x='Provenance',
            y='Summation Rate',
            title='Summation Rate by Provenance',
            labels={'Summation Rate': 'Summation Rate (%)'},
            text='Count',
            color='Summation Rate',
            color_continuous_scale='RdYlGn'
        )
        fig_prov_sum.update_traces(
            texttemplate='n=%{text}',
            textposition='outside')
        fig_prov_sum.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_prov_sum, width="stretch")

    with col2:
        # Average features by provenance
        prov_features = filtered_data.groupby('PROVENANCE').agg({
            'num_nodes': 'mean',
            'depth': 'mean',
            'avg_branching': 'mean'
        }).reset_index()

        fig_prov_feat = go.Figure()
        fig_prov_feat.add_trace(
            go.Bar(
                name='Size',
                x=prov_features['PROVENANCE'],
                y=prov_features['num_nodes']))
        fig_prov_feat.add_trace(
            go.Bar(
                name='Depth',
                x=prov_features['PROVENANCE'],
                y=prov_features['depth'] * 10))
        fig_prov_feat.add_trace(
            go.Bar(
                name='Branching',
                x=prov_features['PROVENANCE'],
                y=prov_features['avg_branching'] *
                10))

        fig_prov_feat.update_layout(
            title='Structural Features by Provenance (normalized)',
            xaxis_tickangle=-45,
            height=500,
            barmode='group'
        )
        st.plotly_chart(fig_prov_feat, width="stretch")

    # Heatmap: Cluster × Provenance
    contingency = pd.crosstab(
        filtered_data['cluster'],
        filtered_data['PROVENANCE'])
    fig_heatmap = px.imshow(
        contingency,
        labels=dict(x="Provenance", y="Cluster", color="Count"),
        title="Cluster × Provenance Enrichment",
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, width="stretch")

# TAB 3: Cluster Analysis
with tab3:
    selected_cluster = st.selectbox(
        "Select Cluster for Detailed View", sorted(
            filtered_data['cluster'].unique()))

    cluster_data = filtered_data[filtered_data['cluster'] == selected_cluster]

    st.markdown(f"### Cluster {selected_cluster} - {len(cluster_data)} khipus")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Size", f"{cluster_data['num_nodes'].mean():.0f}")
    with col2:
        st.metric("Avg Depth", f"{cluster_data['depth'].mean():.1f}")
    with col3:
        st.metric(
            "Summation Rate",
            f"{cluster_data['has_pendant_summation'].mean()*100:.1f}%")

    # Feature distributions
    col1, col2 = st.columns(2)

    with col1:
        fig_size_dist = px.histogram(
            cluster_data,
            x='num_nodes',
            nbins=30,
            title=f'Size Distribution - Cluster {selected_cluster}',
            labels={'num_nodes': 'Size (nodes)'},
            color_discrete_sequence=['steelblue']
        )
        fig_size_dist.update_layout(height=400)
        st.plotly_chart(fig_size_dist, width="stretch")

    with col2:
        fig_numeric = px.histogram(
            cluster_data,
            x='has_numeric',
            nbins=20,
            title=f'Numeric Coverage - Cluster {selected_cluster}',
            labels={'has_numeric': 'Numeric Coverage (%)'},
            color_discrete_sequence=['coral']
        )
        fig_numeric.update_layout(height=400)
        st.plotly_chart(fig_numeric, width="stretch")

# TAB 4: Feature Relationships
with tab4:
    st.markdown("### Feature Correlation Analysis")

    # Select features for correlation
    numeric_features = ['num_nodes', 'depth', 'avg_branching', 'has_numeric',
                        'pendant_match_rate', 'num_white_boundaries']

    feature_x = st.selectbox("X-axis Feature", numeric_features, index=0)
    feature_y = st.selectbox("Y-axis Feature", numeric_features, index=1)
    
    # Option to show trendline
    show_trendline = st.checkbox("Show OLS Trendline", value=False)

    # Prepare data - convert cluster to string for categorical coloring
    plot_data = filtered_data.copy()
    plot_data['cluster'] = plot_data['cluster'].astype(str)

    # Scatter plot with optional trendline
    try:
        fig_corr = px.scatter(
            plot_data,
            x=feature_x,
            y=feature_y,
            color='cluster',
            trendline='ols' if show_trendline else None,
            hover_data=['khipu_id', 'PROVENANCE'],
            title=f'{feature_x} vs {feature_y}',
            labels={feature_x: feature_x.replace('_', ' ').title(),
                    feature_y: feature_y.replace('_', ' ').title()}
        )
    except Exception as e:
        # Fallback without trendline if statsmodels fails
        st.warning(f"Trendline disabled: {str(e)}")
        fig_corr = px.scatter(
            plot_data,
            x=feature_x,
            y=feature_y,
            color='cluster',
            hover_data=['khipu_id', 'PROVENANCE'],
            title=f'{feature_x} vs {feature_y}',
            labels={feature_x: feature_x.replace('_', ' ').title(),
                    feature_y: feature_y.replace('_', ' ').title()}
        )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, width="stretch")

    # Correlation matrix
    st.markdown("### Feature Correlation Matrix")
    corr_matrix = filtered_data[numeric_features].corr()

    fig_corr_matrix = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        color_continuous_scale='RdBu_r',
        aspect='auto',
        zmin=-1,
        zmax=1
    )
    fig_corr_matrix.update_layout(height=500)
    st.plotly_chart(fig_corr_matrix, width="stretch")

# ==================== DATA EXPORT ====================
st.markdown("---")
st.markdown("### 📥 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export Filtered Data (CSV)"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="filtered_khipus.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Summary Statistics"):
        summary = filtered_data.describe()
        summary_csv = summary.to_csv()
        st.download_button(
            label="Download Summary",
            data=summary_csv,
            file_name="summary_statistics.csv",
            mime="text/csv"
        )

with col3:
    st.markdown(f"**Current Selection:** {len(filtered_data)} khipus")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Khipu Analysis Dashboard • Data: Harvard Open Khipu Repository • 612 Analyzed Khipus</p>
</div>
""", unsafe_allow_html=True)
