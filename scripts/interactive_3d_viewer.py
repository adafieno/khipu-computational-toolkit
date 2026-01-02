"""
Interactive 3D Khipu Viewer (Streamlit)

Web-based interface for viewing khipu 3D structures with dropdown selection.
Allows easy browsing through all khipus with interactive controls.

Usage: streamlit run scripts/interactive_3d_viewer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import sqlite3

st.set_page_config(
    page_title="3D Khipu Viewer",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Interactive 3D Khipu Structure Viewer")

@st.cache_data
def get_khipu_list():
    """Get list of all available khipus with metadata."""
    conn = sqlite3.connect("data/khipu.db")
    # Get khipu metadata
    khipu_df = pd.read_sql_query("""
        SELECT KHIPU_ID, PROVENANCE 
        FROM khipu_main 
        ORDER BY KHIPU_ID
    """, conn)
    conn.close()
    
    # Get cord counts from hierarchy
    hierarchy = pd.read_csv("data/processed/phase2/cord_hierarchy.csv")
    cord_counts = hierarchy.groupby('KHIPU_ID').size().reset_index(name='cord_count')
    
    # Merge
    khipu_df = khipu_df.merge(cord_counts, on='KHIPU_ID', how='left')
    khipu_df['cord_count'] = khipu_df['cord_count'].fillna(0).astype(int)
    khipu_df = khipu_df[khipu_df['cord_count'] > 0]
    
    return khipu_df

@st.cache_data
def load_khipu_data(khipu_id):
    """Load hierarchical structure and values for a khipu."""
    hierarchy = pd.read_csv("data/processed/phase2/cord_hierarchy.csv")
    numeric_values = pd.read_csv("data/processed/phase1/cord_numeric_values.csv")
    
    # Filter for specific khipu
    khipu_cords = hierarchy[hierarchy['KHIPU_ID'] == khipu_id].copy()
    khipu_values = numeric_values[numeric_values['khipu_id'] == khipu_id].copy()
    
    # Merge values
    khipu_data = khipu_cords.merge(
        khipu_values[['cord_id', 'numeric_value']],
        left_on='CORD_ID',
        right_on='cord_id',
        how='left'
    )
    
    return khipu_data

def build_network(khipu_data):
    """Build NetworkX graph from cord hierarchy."""
    G = nx.DiGraph()
    
    for _, row in khipu_data.iterrows():
        cord_id = row['CORD_ID']
        parent_id = row['PENDANT_FROM']
        level = row['CORD_LEVEL'] if pd.notna(row['CORD_LEVEL']) else 0
        numeric_value = row['numeric_value'] if pd.notna(row['numeric_value']) else 0
        
        G.add_node(cord_id, level=level, value=numeric_value)
        
        if pd.notna(parent_id) and parent_id != 0:
            # Add parent node if it doesn't exist (main cord)
            if not G.has_node(parent_id):
                G.add_node(parent_id, level=0, value=0)
            G.add_edge(parent_id, cord_id)
    
    return G

def compute_3d_layout(G):
    """Compute 3D positions for nodes using hierarchical layout."""
    pos = {}
    
    # Get level information
    levels = nx.get_node_attributes(G, 'level')
    
    # Group nodes by level
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)
    
    # Assign positions
    for level, nodes in level_nodes.items():
        n = len(nodes)
        
        # Arrange nodes in circular pattern
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius = 1 + level * 0.5  # Increase radius with level
        
        for i, node in enumerate(nodes):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            z = -level  # Vertical position by level
            pos[node] = (x, y, z)
    
    return pos

def create_3d_plot(khipu_data, color_mode='value', elevation=30, azimuth=45):
    """Create 3D visualization of khipu structure."""
    
    if len(khipu_data) == 0:
        return None
    
    G = build_network(khipu_data)
    pos = compute_3d_layout(G)
    
    # Extract coordinates
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    
    # Color mapping
    if color_mode == 'value':
        values = [G.nodes[node]['value'] for node in G.nodes()]
        norm = Normalize(vmin=min(values), vmax=max(values))
        colors = [cm.viridis(norm(v)) for v in values]
    elif color_mode == 'level':
        levels = [G.nodes[node]['level'] for node in G.nodes()]
        norm = Normalize(vmin=min(levels), vmax=max(levels))
        colors = [cm.plasma(norm(level)) for level in levels]
    else:
        colors = ['steelblue'] * len(G.nodes())
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot edges
    for edge in G.edges():
        x_line = [pos[edge[0]][0], pos[edge[1]][0]]
        y_line = [pos[edge[0]][1], pos[edge[1]][1]]
        z_line = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_line, y_line, z_line, 'gray', alpha=0.4, linewidth=0.5)
    
    # Plot nodes
    _ = ax.scatter(xs, ys, zs, c=colors, s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # Set view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Labels and styling
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Hierarchy Level')
    ax.set_title(f'3D Khipu Structure (Colored by {color_mode})', pad=20)
    
    # Remove grid for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    return fig

# Sidebar controls
st.sidebar.header("Khipu Selection")

# Load khipu list
khipu_list = get_khipu_list()

# Create selection options
khipu_list['display'] = khipu_list.apply(
    lambda row: f"{row['KHIPU_ID']} - {row['PROVENANCE']} ({row['cord_count']} cords)" 
    if pd.notna(row['PROVENANCE']) else f"{row['KHIPU_ID']} - Unknown ({row['cord_count']} cords)", 
    axis=1
)

# Dropdown selection
selected_display = st.sidebar.selectbox(
    "Select Khipu",
    options=khipu_list['display'].tolist(),
    index=0
)

# Get selected khipu ID
selected_khipu_id = khipu_list[khipu_list['display'] == selected_display]['KHIPU_ID'].iloc[0]

st.sidebar.markdown("---")
st.sidebar.header("Visualization Options")

# Color mode
color_mode = st.sidebar.radio(
    "Color Mode",
    options=['value', 'level'],
    format_func=lambda x: 'Numeric Value' if x == 'value' else 'Hierarchy Level'
)

# View angle controls
st.sidebar.subheader("View Angle")
elevation = st.sidebar.slider("Elevation", min_value=0, max_value=90, value=30, step=5)
azimuth = st.sidebar.slider("Azimuth", min_value=0, max_value=360, value=45, step=15)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Khipu {selected_khipu_id}")
    
    # Load and display
    with st.spinner("Loading 3D visualization..."):
        khipu_data = load_khipu_data(selected_khipu_id)
        
        if len(khipu_data) > 0:
            fig = create_3d_plot(khipu_data, color_mode, elevation, azimuth)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.error("Unable to create visualization")
        else:
            st.warning(f"No cord data found for khipu {selected_khipu_id}")

with col2:
    st.subheader("Khipu Statistics")
    
    if len(khipu_data) > 0:
        # Basic stats
        st.metric("Total Cords", len(khipu_data))
        st.metric("Max Hierarchy Level", int(khipu_data['CORD_LEVEL'].max()))
        st.metric("Cords with Values", int(khipu_data['numeric_value'].notna().sum()))
        
        # Summary statistics
        st.markdown("#### Numeric Values")
        if khipu_data['numeric_value'].notna().any():
            st.write(f"Mean: {khipu_data['numeric_value'].mean():.2f}")
            st.write(f"Median: {khipu_data['numeric_value'].median():.2f}")
            st.write(f"Max: {khipu_data['numeric_value'].max():.0f}")
        else:
            st.write("No numeric values recorded")
        
        # Level distribution
        st.markdown("#### Hierarchy Levels")
        level_counts = khipu_data['CORD_LEVEL'].value_counts().sort_index()
        st.bar_chart(level_counts)

# Info footer
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip**: Use the elevation and azimuth sliders to rotate the 3D view. "
    "The structure shows the hierarchical relationship between cords."
)
