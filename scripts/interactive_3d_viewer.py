"""
DEPRECATED: This viewer has been replaced by khipu_3d_viewer.py
This file is kept for historical reference only.
Please use: streamlit run scripts/khipu_3d_viewer.py
"""

"""
Interactive 3D Khipu Viewer (Streamlit)

Web-based interface for viewing khipu 3D structures with dropdown selection.
Allows easy browsing through all khipus with interactive controls.

Usage: streamlit run scripts/interactive_3d_viewer.py
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import networkx as nx  # noqa: E402
import sqlite3  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

# Initialize config
config = get_config()

st.set_page_config(
    page_title="3D Khipu Viewer",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Interactive 3D Khipu Structure Viewer")

@st.cache_data
def get_khipu_list():
    """Get list of all available khipus with metadata."""
    conn = sqlite3.connect(config.get_database_path())
    # Get khipu metadata
    khipu_df = pd.read_sql_query("""
        SELECT KHIPU_ID, PROVENANCE 
        FROM khipu_main 
        ORDER BY KHIPU_ID
    """, conn)
    conn.close()
    
    # Get cord counts from hierarchy
    hierarchy = pd.read_csv(config.get_processed_file("cord_hierarchy.csv", 2))
    cord_counts = hierarchy.groupby('KHIPU_ID').size().reset_index(name='cord_count')
    
    # Merge
    khipu_df = khipu_df.merge(cord_counts, on='KHIPU_ID', how='left')
    khipu_df['cord_count'] = khipu_df['cord_count'].fillna(0).astype(int)
    khipu_df = khipu_df[khipu_df['cord_count'] > 0]
    
    return khipu_df

# Temporarily disabled cache to test cord length loading
# @st.cache_data
def load_khipu_data(khipu_id):
    """Load hierarchical structure, values, and knot information for a khipu."""
    import sqlite3
    hierarchy = pd.read_csv(config.get_processed_file("cord_hierarchy.csv", 2))
    numeric_values = pd.read_csv(config.get_processed_file("cord_numeric_values.csv", 1))
    
    # Filter for specific khipu
    khipu_cords = hierarchy[hierarchy['KHIPU_ID'] == khipu_id].copy()
    khipu_values = numeric_values[numeric_values['khipu_id'] == khipu_id].copy()
    
    # Load cord lengths from database
    conn = sqlite3.connect(config.get_database_path())
    cord_lengths = pd.read_sql_query(
        "SELECT CORD_ID, CORD_LENGTH FROM cord WHERE KHIPU_ID = ?",
        conn, params=(int(khipu_id),))
    conn.close()
    
    # Merge values and lengths
    khipu_data = khipu_cords.merge(
        khipu_values[['cord_id', 'numeric_value']],
        left_on='CORD_ID',
        right_on='cord_id',
        how='left'
    ).merge(
        cord_lengths,
        on='CORD_ID',
        how='left',
        suffixes=('_old', '')  # Keep database CORD_LENGTH without suffix
    )
    
    print(f"DEBUG: Columns in khipu_data: {khipu_data.columns.tolist()}")
    print(f"DEBUG: Sample CORD_LENGTH values: {khipu_data['CORD_LENGTH'].head().tolist() if 'CORD_LENGTH' in khipu_data.columns else 'NOT FOUND'}")
    
    # Load knot data from database
    conn = sqlite3.connect(config.get_database_path())
    knots_query = """
        SELECT k.CORD_ID, k.KNOT_ID, k.KNOT_ORDINAL, k.TYPE_CODE, k.NUM_TURNS, k.knot_value_type
        FROM knot k
        JOIN cord c ON k.CORD_ID = c.CORD_ID
        WHERE c.KHIPU_ID = ?
        ORDER BY k.CORD_ID, k.KNOT_ORDINAL
    """
    knots = pd.read_sql_query(knots_query, conn, params=(int(khipu_id),))
    conn.close()
    
    print(f"DEBUG load_khipu_data: Loaded {len(knots)} knots for khipu {khipu_id}")
    
    return khipu_data, knots

def build_network(khipu_data):
    """Build NetworkX graph from cord hierarchy."""
    G = nx.DiGraph()
    
    for _, row in khipu_data.iterrows():
        cord_id = row['CORD_ID']
        parent_id = row['PENDANT_FROM']
        level = row['CORD_LEVEL'] if pd.notna(row['CORD_LEVEL']) else 0
        numeric_value = row['numeric_value'] if pd.notna(row['numeric_value']) else 0
        cord_length = row['CORD_LENGTH'] if pd.notna(row['CORD_LENGTH']) else 30.0  # default 30cm
        
        G.add_node(cord_id, level=level, value=numeric_value, cord_length=cord_length)
        
        # Only add edges for subsidiary pendants (level 2+)
        # Level 1 pendants hang from the khipu itself (not a cord node)
        if pd.notna(parent_id) and parent_id != 0 and level > 1:
            # Parent must be another cord (not khipu ID)
            if G.has_node(parent_id):
                G.add_edge(parent_id, cord_id)
    
    return G

def compute_3d_layout(G):
    """Compute 3D positions with main cord horizontal and pendants hanging down."""
    pos = {}
    
    # Get level information
    levels = nx.get_node_attributes(G, 'level')
    
    # All level 1 nodes (primary pendants) hang from an imaginary main cord
    level_1_nodes = sorted([node for node, level in levels.items() if level == 1])
    
    # Create main cord as horizontal line with pendant attachment points
    main_cord_length = max(len(level_1_nodes), 1)
    main_y = 0
    main_z = 0
    
    # Position level 1 pendants evenly along the main cord
    # Use actual cord lengths scaled to visualization space (divide by 50 to convert cm to reasonable scale)
    for i, node in enumerate(level_1_nodes):
        x_pos = i * 0.8  # Spacing along main cord
        cord_length = G.nodes[node]['cord_length']
        depth = -cord_length / 50.0  # Scale cord length (cm) to visualization units
        pos[node] = (x_pos, main_y, main_z + depth)
    
    # TEMPORARILY DISABLED: Position subsidiary pendants (level 2+) hanging from their parents
    # pendant_spacing = {}
    # for level in range(2, max(levels.values()) + 1 if levels.values() else 2):
    #     pendant_nodes = [node for node, lvl in levels.items() if lvl == level]
    #     
    #     for node in pendant_nodes:
    #         # Find parent
    #         parents = [n for n in G.predecessors(node)]
    #         if parents:
    #             parent_pos = pos[parents[0]]
    #             
    #             # Track spacing for siblings
    #             parent_key = parents[0]
    #             if parent_key not in pendant_spacing:
    #                 pendant_spacing[parent_key] = 0
    #             
    #             # L-shaped elbow: barely visible horizontal offset then drop straight down
    #             sibling_offset = pendant_spacing[parent_key] - 0.5
    #             # Tiny fixed horizontal elbow
    #             x_offset = 0.01  # Barely visible horizontal extension
    #             y_offset = sibling_offset * 0.005  # Minimal lateral spread for siblings
    #             
    #             # Position hangs straight down from horizontally offset point
    #             pos[node] = (
    #                 parent_pos[0] + x_offset,
    #                 parent_pos[1] + y_offset,
    #                 parent_pos[2] - 1.0  # Drop much further down
    #             )
    #             pendant_spacing[parent_key] += 1
    
    return pos, level_1_nodes

def create_3d_plot_plotly(khipu_data, knots, color_mode='value'):
    """Create 3D visualization of khipu structure."""
    
    if len(khipu_data) == 0:
        return None
    
    G = build_network(khipu_data)
    pos, level_1_nodes = compute_3d_layout(G)
    
    # Extract coordinates - ONLY for nodes that have positions (level 1 for now)
    positioned_nodes = [node for node in G.nodes() if node in pos]
    xs = [pos[node][0] for node in positioned_nodes]
    ys = [pos[node][1] for node in positioned_nodes]
    zs = [pos[node][2] for node in positioned_nodes]
    
    # Color mapping
    if color_mode == 'value':
        values = [G.nodes[node]['value'] for node in positioned_nodes]
        norm = Normalize(vmin=min(values), vmax=max(values))
        colors = [cm.viridis(norm(v)) for v in values]
    elif color_mode == 'level':
        levels = [G.nodes[node]['level'] for node in positioned_nodes]
        norm = Normalize(vmin=min(levels), vmax=max(levels))
        colors = [cm.plasma(norm(level)) for level in levels]
    else:
        colors = ['steelblue'] * len(positioned_nodes)
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw main cord as thick horizontal line
    if level_1_nodes:
        main_x = [pos[node][0] for node in level_1_nodes]
        main_y = [0] * len(level_1_nodes)
        main_z = [0] * len(level_1_nodes)
        ax.plot(main_x, main_y, main_z, color='#6B5345', alpha=0.95, linewidth=10, zorder=5)
    
    # Draw pendant cords
    levels = nx.get_node_attributes(G, 'level')
    for node in G.nodes():
        if node in pos:
            node_pos = pos[node]
            node_level = levels.get(node, 1)
            
            # Only draw from main cord for level 1 pendants
            if node_level == 1:
                ax.plot([node_pos[0], node_pos[0]], [0, node_pos[1]], [0, node_pos[2]], 
                       color='#8B7355', alpha=0.7, linewidth=4, zorder=2)
            
            # TEMPORARILY DISABLED: Draw subsidiary connections with L-shaped elbows (level 2+ to their parent)
            # for child in G.successors(node):
            #     if child in pos:
            #         child_pos = pos[child]
            #         child_level = levels.get(child, 2)
            #         
            #         if child_level >= 2:
            #             # Draw L-shaped elbow: horizontal then vertical
            #             # First segment: horizontal from parent to elbow point
            #             elbow_x = child_pos[0]
            #             elbow_y = child_pos[1]
            #             elbow_z = node_pos[2]  # Stay at parent's z level
            #             
            #             ax.plot(
            #                 [node_pos[0], elbow_x],
            #                 [node_pos[1], elbow_y],
            #                 [node_pos[2], elbow_z],
            #                 color='#8B7355',
            #                 alpha=0.6,
            #                 linewidth=3,
            #                 zorder=2)
            #             
            #             # Second segment: vertical drop from elbow to child
            #             ax.plot(
            #                 [elbow_x, child_pos[0]],
            #                 [elbow_y, child_pos[1]],
            #                 [elbow_z, child_pos[2]],
            #                 color='#8B7355',
            #                 alpha=0.6,
            #                 linewidth=3,
            #                 zorder=2)
            #         else:
            #             # Direct line for other connections
            #             ax.plot([node_pos[0], child_pos[0]], [node_pos[1], child_pos[1]], 
            #                    [node_pos[2], child_pos[2]], color='#8B7355', alpha=0.6, linewidth=3, zorder=2)
    
    # Draw actual knots - ONLY for level 1 pendants for now
    knot_positions = []
    knot_colors_list = []
    knot_sizes = []
    knot_labels = []  # Store label info for each knot
    knot_metadata = []  # Store full metadata for hover tooltips
    
    print(f"DEBUG: Processing {len(knots)} total knots")
    
    for _, knot_row in knots.iterrows():
        cord_id = knot_row['CORD_ID']
        if cord_id in pos:
            cord_pos = pos[cord_id]
            cord_level = levels.get(cord_id, 1)
            
            # ONLY render knots on level 1 pendants for now
            if cord_level == 1:
                knot_type = knot_row['TYPE_CODE']
                num_turns = knot_row['NUM_TURNS'] if pd.notna(knot_row['NUM_TURNS']) else 0
                value_type = knot_row['knot_value_type'] if pd.notna(knot_row['knot_value_type']) else 10
                
                print(f"DEBUG: Cord {cord_id} - Type={knot_type}, turns={num_turns}, value_type={value_type}")
                
                # Position based on value_type (decimal position) for consistent zones
                # Lower value_type (1=units) at bottom, higher value_type (10=hundreds) at top
                if value_type == 1:
                    t = 0.85  # Units near bottom
                elif value_type in [2, 3]:
                    t = 0.70  # Tens 
                elif value_type in [8, 9]:
                    t = 0.50  # Higher tens
                elif value_type == 10:
                    t = 0.25  # Hundreds near top
                else:
                    t = 0.5  # Default to middle
                
                knot_x = cord_pos[0]
                knot_y = 0 + t * (cord_pos[1] - 0)
                knot_z = 0 + t * (cord_pos[2] - 0)
                
                knot_positions.append([knot_x, knot_y, knot_z])
                
                # Create label: Type + Turns (for long knots)
                if knot_type == 'L' and num_turns > 0:
                    label = f"{knot_type}{int(num_turns)}"
                else:
                    label = knot_type
                knot_labels.append(label)
                
                # Store metadata for hover tooltip
                numeric_value = G.nodes[cord_id]['value']
                knot_metadata.append({
                    'cord_id': cord_id,
                    'type': knot_type,
                    'turns': int(num_turns) if num_turns > 0 else 0,
                    'value_type': value_type,
                    'numeric_value': numeric_value,
                    'label': label
                })
                
                # Size based on value_type - larger for higher positions
                if value_type == 10:
                    base_size = 140  # Hundreds largest
                elif value_type in [8, 9]:
                    base_size = 110  # Tens medium
                elif value_type in [2, 3]:
                    base_size = 100
                else:
                    base_size = 90  # Units smallest
                
                # Add extra size for long knots based on turns
                if knot_type == 'L':
                    knot_sizes.append(base_size + num_turns * 5)
                else:
                    knot_sizes.append(base_size)
                
                # Store the actual value for coloring, not the RGB
                if color_mode == 'value':
                    knot_colors_list.append(G.nodes[cord_id]['value'])
                elif color_mode == 'level':
                    knot_colors_list.append(G.nodes[cord_id]['level'])
                else:
                    knot_colors_list.append(1.0)  # Default value
    
    # Plot knots - make them VERY visible with variable sizes
    print(f"DEBUG: Attempting to render {len(knot_positions)} knots")
    if knot_positions:
        knot_arr = np.array(knot_positions)
        print(f"DEBUG: Knot array shape: {knot_arr.shape}")
        print(f"DEBUG: Color mode: {color_mode}, colors list length: {len(knot_colors_list)}")
        print(f"DEBUG: Knot sizes: {knot_sizes}")
        # Always render knots with proper colormapping and variable sizes
        if color_mode == 'value' and knot_colors_list:
            scatter_knots = ax.scatter(
                knot_arr[:, 0], knot_arr[:, 1], knot_arr[:, 2],
                c=knot_colors_list,
                cmap=cm.viridis,
                s=knot_sizes,
                alpha=1.0,
                marker='o',
                zorder=20)
        elif color_mode == 'level' and knot_colors_list:
            scatter_knots = ax.scatter(
                knot_arr[:, 0], knot_arr[:, 1], knot_arr[:, 2],
                c=knot_colors_list,
                cmap=cm.plasma,
                s=knot_sizes,
                alpha=1.0,
                marker='o',
                zorder=20)
        else:
            scatter_knots = ax.scatter(
                knot_arr[:, 0], knot_arr[:, 1], knot_arr[:, 2],
                c='#CD853F',  # Peru/tan color - very visible
                s=knot_sizes,
                alpha=1.0,
                marker='o',
                zorder=20)
        
        # Add text labels to each knot with better visibility
        for i, (pos_knot, label) in enumerate(zip(knot_arr, knot_labels)):
            ax.text(pos_knot[0], pos_knot[1], pos_knot[2], 
                   label, 
                   fontsize=9, 
                   ha='center', 
                   va='bottom',
                   color='white',
                   weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'),
                   zorder=25)
        
        # Add interactive hover tooltips if mplcursors is available
        if HAS_MPLCURSORS:
            cursor = mplcursors.cursor(scatter_knots, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                if idx < len(knot_metadata):
                    meta = knot_metadata[idx]
                    tooltip_text = f"Cord: {meta['cord_id']}\n"
                    tooltip_text += f"Type: {meta['type']}"
                    if meta['turns'] > 0:
                        tooltip_text += f" ({meta['turns']} turns)"
                    tooltip_text += f"\nPosition: {meta['value_type']}\n"
                    tooltip_text += f"Value: {meta['numeric_value']}"
                    sel.annotation.set_text(tooltip_text)
                    sel.annotation.get_bbox_patch().set(fc="yellow", alpha=0.9)
    
    # Set view angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Labels and styling
    ax.set_xlabel('Length along main cord')
    ax.set_ylabel('Y')
    ax.set_zlabel('Pendant depth')
    ax.set_title(f'3D Khipu Structure (Colored by {color_mode})', pad=20)
    
    # Subtle grid and background
    ax.grid(True, alpha=0.2)
    ax.xaxis.pane.set_facecolor('#f5f5dc')
    ax.yaxis.pane.set_facecolor('#f5f5dc')
    ax.zaxis.pane.set_facecolor('#f5f5dc')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
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

# Load data once for both columns
khipu_data, knots = load_khipu_data(selected_khipu_id)

with col1:
    st.subheader(f"Khipu {selected_khipu_id}")
    
    # Load and display
    with st.spinner("Loading 3D visualization..."):
        if len(khipu_data) > 0:
            fig = create_3d_plot(khipu_data, knots, color_mode, elevation, azimuth)
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
        st.metric("Total Knots", len(knots))
        st.metric("Max Hierarchy Level", int(khipu_data['CORD_LEVEL'].max()))
        
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
