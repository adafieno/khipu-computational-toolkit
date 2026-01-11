"""
Interactive 3D Khipu Viewer with Plotly (Streamlit)

Web-based interface for viewing khipu 3D structures with interactive hover tooltips.
Shows structural relationships, colors, and knot types without numeric interpretation.
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import streamlit as st  # noqa: E402
import pandas as pd  # noqa: E402
import networkx as nx  # noqa: E402
import sqlite3  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from typing import Tuple, Dict, List  # noqa: E402

# Initialize config
config = get_config()

st.set_page_config(
    page_title="3D Khipu Viewer",
    page_icon="🧶",
    layout="wide"
)

st.title("🧶 Interactive 3D Khipu Structure Viewer")

@st.cache_data
def get_color_mappings():
    """Load Ascher color code to RGB mappings from database."""
    conn = sqlite3.connect(config.get_database_path())
    colors = pd.read_sql_query("""
        SELECT AS_COLOR_CD, COLOR_DESCR, R_DEC, G_DEC, B_DEC
        FROM ascher_color_dc
    """, conn)
    conn.close()
    
    # Create mapping dict with hex colors
    color_map = {}
    for _, row in colors.iterrows():
        code = row['AS_COLOR_CD']
        r = int(row['R_DEC'] * 255)
        g = int(row['G_DEC'] * 255)
        b = int(row['B_DEC'] * 255)
        color_map[code] = f'#{r:02x}{g:02x}{b:02x}'
    
    return color_map

@st.cache_data
def get_khipu_list():
    """Get list of all available khipus with metadata."""
    conn = sqlite3.connect(config.get_database_path())
    khipu_df = pd.read_sql_query("""
        SELECT KHIPU_ID, PROVENANCE 
        FROM khipu_main 
        ORDER BY KHIPU_ID
    """, conn)
    conn.close()
    
    hierarchy = pd.read_csv(config.get_processed_file("cord_hierarchy.csv", 2))
    cord_counts = hierarchy.groupby('KHIPU_ID').size().reset_index(name='cord_count')
    
    khipu_df = khipu_df.merge(cord_counts, on='KHIPU_ID', how='left')
    khipu_df['cord_count'] = khipu_df['cord_count'].fillna(0).astype(int)
    khipu_df = khipu_df[khipu_df['cord_count'] > 0]
    
    return khipu_df

def load_khipu_data(khipu_id):
    """Load hierarchical structure, colors, and knot information for a khipu."""
    hierarchy = pd.read_csv(config.get_processed_file("cord_hierarchy.csv", 2))
    khipu_cords = hierarchy[hierarchy['KHIPU_ID'] == khipu_id].copy()
    
    conn = sqlite3.connect(config.get_database_path())
    
    # Load cord colors from ascher_cord_color table
    color_query = """
        SELECT CORD_ID, COLOR_CD_1, FULL_COLOR
        FROM ascher_cord_color
        WHERE KHIPU_ID = ?
    """
    cord_colors = pd.read_sql_query(color_query, conn, params=(int(khipu_id),))
    
    # Merge hierarchy with colors
    khipu_data = khipu_cords.merge(cord_colors, on='CORD_ID', how='left')
    
    # Load knots (without value interpretations)
    knots_query = """
        SELECT k.CORD_ID, k.KNOT_ID, k.KNOT_ORDINAL, k.TYPE_CODE, k.NUM_TURNS
        FROM knot k
        JOIN cord c ON k.CORD_ID = c.CORD_ID
        WHERE c.KHIPU_ID = ?
        ORDER BY k.CORD_ID, k.KNOT_ORDINAL
    """
    knots = pd.read_sql_query(knots_query, conn, params=(int(khipu_id),))
    
    conn.close()
    
    return khipu_data, knots

def build_network(khipu_data):
    """Build NetworkX graph from cord hierarchy."""
    G = nx.DiGraph()
    
    for _, row in khipu_data.iterrows():
        cord_id = row['CORD_ID']
        parent_id = row['PENDANT_FROM']
        level = row['CORD_LEVEL'] if pd.notna(row['CORD_LEVEL']) else 0
        cord_length = row['CORD_LENGTH'] if pd.notna(row['CORD_LENGTH']) else 30.0
        color = row['COLOR_CD_1'] if pd.notna(row['COLOR_CD_1']) else 'Unknown'
        
        G.add_node(
            cord_id, 
            level=level, 
            cord_length=cord_length,
            color=color
        )
        
        if pd.notna(parent_id) and parent_id != 0:
            if parent_id in khipu_data['CORD_ID'].values:
                G.add_edge(parent_id, cord_id)
    
    return G

def compute_3d_layout(G) -> Tuple[Dict, List]:
    """
    Compute 3D positions for all cords including subsidiaries.
    Returns positions dict and list of level-1 nodes.
    """
    pos = {}
    
    # Get all nodes by level
    level_1_nodes = sorted([n for n, attr in G.nodes(data=True) if attr.get('level') == 1])
    
    # Main cord parameters
    main_y = 0
    
    # Adjust spacing based on number of cords for better visibility
    num_pendants = len(level_1_nodes)
    if num_pendants > 100:
        spacing = 1.5  # More space for crowded khipus
    elif num_pendants > 50:
        spacing = 1.3
    else:
        spacing = 1.2  # Default spacing
    
    # Position level-1 pendants
    for i, node in enumerate(level_1_nodes):
        x_pos = i * spacing
        cord_length = G.nodes[node].get('cord_length', 30.0)
        
        # Normalize cord length to reasonable depth
        depth = -min(cord_length / 30.0, 3.0) if cord_length > 0 else -1.0
        
        pos[node] = (x_pos, main_y, depth)
    
    # Position subsidiaries (level 2+)
    for node in level_1_nodes:
        # Get all descendants (subsidiaries)
        descendants = list(nx.descendants(G, node))
        
        if descendants:
            parent_pos = pos[node]
            
            # Sort descendants by level
            descendants_by_level = {}
            for desc in descendants:
                level = G.nodes[desc].get('level', 2)
                if level not in descendants_by_level:
                    descendants_by_level[level] = []
                descendants_by_level[level].append(desc)
            
            # Position each level
            for level in sorted(descendants_by_level.keys()):
                level_nodes = sorted(descendants_by_level[level])
                
                for j, sub_node in enumerate(level_nodes):
                    # Get parent position
                    parent = list(G.predecessors(sub_node))[0]
                    parent_pos = pos[parent]
                    
                    # Offset subsidiaries slightly
                    offset_x = 0.15 * (j - len(level_nodes)/2)
                    sub_length = G.nodes[sub_node].get('cord_length', 20.0)
                    sub_depth = -min(sub_length / 40.0, 2.0) if sub_length > 0 else -0.5
                    
                    pos[sub_node] = (
                        parent_pos[0] + offset_x,
                        parent_pos[1],
                        parent_pos[2] + sub_depth
                    )
    
    return pos, level_1_nodes

def color_to_rgb(color_name: str, color_map: Dict[str, str]) -> str:
    """Convert Ascher color code to RGB hex string using database values."""
    return color_map.get(color_name, '#CCCCCC')  # Default gray for unknown

def create_3d_plot_plotly(khipu_data, knots, color_map):
    """Create interactive 3D visualization using Plotly - structure only, no numeric interpretation."""
    
    if len(khipu_data) == 0:
        return None
    
    G = build_network(khipu_data)
    pos, level_1_nodes = compute_3d_layout(G)
    
    # No cord limit - show all cords for complete visualization
    truncated = False
    display_nodes = level_1_nodes
    
    fig = go.Figure()
    
    # Draw main cord
    if display_nodes:
        main_x = [pos[node][0] for node in display_nodes if node in pos]
        main_y = [0] * len(main_x)
        main_z = [0] * len(main_x)
        
        fig.add_trace(go.Scatter3d(
            x=main_x, y=main_y, z=main_z,
            mode='lines',
            line=dict(color='#8B7355', width=12),
            name='Main Cord',
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Draw pendant cords with colors
    for node in display_nodes:
        if node in pos:
            node_pos = pos[node]
            node_color = G.nodes[node].get('color', 'Unknown')
            cord_rgb = color_to_rgb(node_color, color_map)
            cord_length = G.nodes[node].get('cord_length', 30.0)
            
            fig.add_trace(go.Scatter3d(
                x=[node_pos[0], node_pos[0]],
                y=[0, node_pos[1]],
                z=[0, node_pos[2]],
                mode='lines',
                line=dict(color=cord_rgb, width=10),
                hovertext=f"Cord {node}<br>Color: {node_color}<br>Length: {cord_length:.1f}cm",
                hoverinfo='text',
                showlegend=False
            ))
            
            # Draw subsidiaries with elbow offset for visibility
            descendants = [d for d in nx.descendants(G, node) if d in pos]
            for desc in descendants:
                desc_pos = pos[desc]
                desc_color = G.nodes[desc].get('color', 'Unknown')
                desc_rgb = color_to_rgb(desc_color, color_map)
                desc_length = G.nodes[desc].get('cord_length', 20.0)
                
                # Get parent position
                parent = list(G.predecessors(desc))[0]
                parent_pos = pos[parent]
                
                # Create elbow: parent -> elbow point -> subsidiary
                elbow_offset = 0.5  # Dramatic horizontal offset to clearly distinguish subsidiaries
                elbow_x = parent_pos[0] + elbow_offset
                elbow_y = parent_pos[1]
                elbow_z = parent_pos[2]
                
                # Draw first segment: parent to elbow
                fig.add_trace(go.Scatter3d(
                    x=[parent_pos[0], elbow_x],
                    y=[parent_pos[1], elbow_y],
                    z=[parent_pos[2], elbow_z],
                    mode='lines',
                    line=dict(color=desc_rgb, width=7),
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                # Draw second segment: elbow to subsidiary end
                fig.add_trace(go.Scatter3d(
                    x=[elbow_x, desc_pos[0]],
                    y=[elbow_y, desc_pos[1]],
                    z=[elbow_z, desc_pos[2]],
                    mode='lines',
                    line=dict(color=desc_rgb, width=7),
                    hovertext=f"Subsidiary {desc}<br>Color: {desc_color}<br>Length: {desc_length:.1f}cm",
                    hoverinfo='text',
                    showlegend=False
                ))
    
    # Draw knots - organize by type for different shapes
    knot_data = {'S': {'x': [], 'y': [], 'z': [], 'hover': [], 'text': []},
                 'L': {'x': [], 'y': [], 'z': [], 'hover': [], 'text': []},
                 'E': {'x': [], 'y': [], 'z': [], 'hover': [], 'text': []}}
    
    # Track subsidiary cord paths for knot positioning
    subsidiary_paths = {}
    for node in display_nodes:
        if node in pos:
            descendants = [d for d in nx.descendants(G, node) if d in pos]
            for desc in descendants:
                parent = list(G.predecessors(desc))[0]
                parent_pos = pos[parent]
                desc_pos = pos[desc]
                
                # Store the elbow path for this subsidiary
                elbow_offset = 0.5
                elbow_x = parent_pos[0] + elbow_offset
                elbow_y = parent_pos[1]
                elbow_z = parent_pos[2]
                
                subsidiary_paths[desc] = {
                    'parent': parent_pos,
                    'elbow': (elbow_x, elbow_y, elbow_z),
                    'end': desc_pos
                }
    
    for _, knot_row in knots.iterrows():
        cord_id = knot_row['CORD_ID']
        if cord_id in pos:
            knot_type = knot_row['TYPE_CODE']
            num_turns = knot_row['NUM_TURNS'] if pd.notna(knot_row['NUM_TURNS']) else 0
            knot_ordinal = knot_row['KNOT_ORDINAL']
            
            # Skip unknown knot types
            if knot_type not in knot_data:
                continue
            
            # Position along cord based on ordinal
            cord_knot_count = len(knots[knots['CORD_ID'] == cord_id])
            if cord_knot_count > 1:
                t = (knot_ordinal + 1) / (cord_knot_count + 1)
            else:
                t = 0.5
            
            # Clamp t to reasonable range
            t = max(0.2, min(0.9, t))
            
            # Calculate position - handle subsidiaries differently
            if cord_id in subsidiary_paths:
                # Position along elbow path
                path = subsidiary_paths[cord_id]
                parent_pos = path['parent']
                elbow_pos = path['elbow']
                end_pos = path['end']
                
                # Split t between the two segments (horizontal elbow + vertical drop)
                # Horizontal segment is short, so give it less of the path (20%)
                if t < 0.2:
                    # Along horizontal elbow segment
                    segment_t = t / 0.2
                    knot_x = parent_pos[0] + segment_t * (elbow_pos[0] - parent_pos[0])
                    knot_y = parent_pos[1] + segment_t * (elbow_pos[1] - parent_pos[1])
                    knot_z = parent_pos[2] + segment_t * (elbow_pos[2] - parent_pos[2])
                else:
                    # Along vertical drop segment
                    segment_t = (t - 0.2) / 0.8
                    knot_x = elbow_pos[0] + segment_t * (end_pos[0] - elbow_pos[0])
                    knot_y = elbow_pos[1] + segment_t * (end_pos[1] - elbow_pos[1])
                    knot_z = elbow_pos[2] + segment_t * (end_pos[2] - elbow_pos[2])
            else:
                # Regular pendant cord - straight line from main cord
                cord_pos = pos[cord_id]
                knot_x = cord_pos[0]
                knot_y = 0 + t * (cord_pos[1] - 0)
                knot_z = 0 + t * (cord_pos[2] - 0)
            
            knot_data[knot_type]['x'].append(knot_x)
            knot_data[knot_type]['y'].append(knot_y)
            knot_data[knot_type]['z'].append(knot_z)
            
            # Text label for long knots showing turns
            if knot_type == 'L' and num_turns > 0:
                knot_data[knot_type]['text'].append(f"{int(num_turns)}")
            else:
                knot_data[knot_type]['text'].append("")
            
            # Create hover text (structure only, no values)
            hover_text = f"<b>Cord:</b> {cord_id}<br>"
            hover_text += f"<b>Knot Type:</b> {knot_type}<br>"
            if knot_type == 'L' and num_turns > 0:
                hover_text += f"<b>Turns:</b> {int(num_turns)}<br>"
            hover_text += f"<b>Ordinal:</b> {knot_ordinal}"
            
            knot_data[knot_type]['hover'].append(hover_text)
    
    # Add knots as separate traces with different shapes
    # S knots - circles
    if knot_data['S']['x']:
        fig.add_trace(go.Scatter3d(
            x=knot_data['S']['x'], 
            y=knot_data['S']['y'], 
            z=knot_data['S']['z'],
            mode='markers',
            marker=dict(
                size=10,
                symbol='circle',
                color='#8B4513',  # Brown
                line=dict(width=1, color='white')
            ),
            hovertext=knot_data['S']['hover'],
            hoverinfo='text',
            name='S (Single)',
            showlegend=True
        ))
    
    # L knots - diamonds with turn count labels
    if knot_data['L']['x']:
        fig.add_trace(go.Scatter3d(
            x=knot_data['L']['x'], 
            y=knot_data['L']['y'], 
            z=knot_data['L']['z'],
            mode='markers+text',
            marker=dict(
                size=12,
                symbol='diamond',
                color='#DAA520',  # Goldenrod
                line=dict(width=1, color='white')
            ),
            text=knot_data['L']['text'],
            textposition='middle right',
            textfont=dict(size=9, color='#000000', family='Arial Black'),
            hovertext=knot_data['L']['hover'],
            hoverinfo='text',
            name='L (Long)',
            showlegend=True
        ))
    
    # E knots - squares
    if knot_data['E']['x']:
        fig.add_trace(go.Scatter3d(
            x=knot_data['E']['x'], 
            y=knot_data['E']['y'], 
            z=knot_data['E']['z'],
            mode='markers',
            marker=dict(
                size=10,
                symbol='square',
                color='#4169E1',  # Royal blue
                line=dict(width=1, color='white')
            ),
            hovertext=knot_data['E']['hover'],
            hoverinfo='text',
            name='E (Figure-8)',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text='Position along main cord', font=dict(color='#000000')),
                showgrid=True, 
                gridcolor='#a0a0a0', 
                gridwidth=2,
                backgroundcolor='#d0d5dd',
                tickfont=dict(color='#000000')
            ),
            yaxis=dict(
                title='', 
                showgrid=True, 
                gridcolor='#b8b8b8',
                showticklabels=False, 
                backgroundcolor='#d0d5dd'
            ),
            zaxis=dict(
                title=dict(text='Cord depth', font=dict(color='#000000')),
                showgrid=True, 
                gridcolor='#a0a0a0',
                gridwidth=2,
                backgroundcolor='#d0d5dd',
                tickfont=dict(color='#000000')
            ),
            bgcolor='#c0c8d0',
            camera=dict(eye=dict(x=1.3, y=-1.5, z=0.8))
        ),
        title=dict(
            text='3D Khipu Structure Viewer',
            font=dict(color='#000000', size=20)
        ),
        showlegend=True,
        hovermode='closest',
        height=750,
        paper_bgcolor='#e8ecf0',
        plot_bgcolor='#c0c8d0',
        font=dict(color='#000000'),
        legend=dict(
            x=1.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#666',
            borderwidth=2,
            font=dict(color='#000000', size=12)
        )
    )
    
    return fig

# Sidebar controls
st.sidebar.header("Controls")

# Get available khipus
with st.spinner("Loading khipu list..."):
    khipus = get_khipu_list()

# Khipu selector
selected_idx = st.sidebar.selectbox(
    "Select Khipu",
    range(len(khipus)),
    format_func=lambda i: f"Khipu {khipus.iloc[i]['KHIPU_ID']} - {khipus.iloc[i]['PROVENANCE']} ({khipus.iloc[i]['cord_count']} cords)"
)

khipu_id = khipus.iloc[selected_idx]['KHIPU_ID']
provenance = khipus.iloc[selected_idx]['PROVENANCE']

# Create a placeholder for loading indicator in sidebar
loading_container = st.sidebar.empty()

# Show loading indicator
loading_container.markdown(
    f'<div style="padding: 10px; background-color: #2e3440; color: #88c0d0; border-radius: 5px; margin: 10px 0;">' +
    f'📊 Loading Khipu {khipu_id}...</div>',
    unsafe_allow_html=True
)

# Load data and render
khipu_data, knots = load_khipu_data(khipu_id)
color_map = get_color_mappings()
fig = create_3d_plot_plotly(khipu_data, knots, color_map) if len(khipu_data) > 0 else None

# Display info
st.sidebar.markdown("---")
st.sidebar.markdown("### Visualization Info")
st.sidebar.markdown("""**Cord Colors:** True colors from Ascher database  
**Knot Shapes:**
- ● Circle = S (Single knot)
- ◆ Diamond = L (Long knot, # shows turns)
- ■ Square = E (Figure-eight knot)

**Scale & Units:**
- **Plot units**: Arbitrary spacing units for 3D layout
- **X-axis**: 1.2-1.5 units between pendants (auto-scaled)
- **Z-axis**: Normalized depth based on cord length (cm)
- **Subsidiary offset**: 0.5 units horizontal from parent
- Actual cord lengths (in cm) shown in hover tooltips

*Note: Shows structural relationships only. No numeric interpretation applied.*
""")

# Load and visualize
st.subheader(f"Khipu {khipu_id}")
if provenance and provenance != 'Unknown':
    st.caption(f"Provenance: {provenance}")

if len(khipu_data) > 0:
    # Display statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Cords", len(khipu_data))
    with col2:
        st.metric("Knots", len(knots))
    with col3:
        level_1_count = len(khipu_data[khipu_data['CORD_LEVEL'] == 1])
        st.metric("Pendants", level_1_count)
    with col4:
        subsidiary_count = len(khipu_data[khipu_data['CORD_LEVEL'] > 1])
        st.metric("Subsidiaries", subsidiary_count)
    with col5:
        max_level = khipu_data['CORD_LEVEL'].max() if 'CORD_LEVEL' in khipu_data.columns else 1
        st.metric("Levels", int(max_level) if pd.notna(max_level) else 1)
    
    # Show color distribution
    color_counts = khipu_data['COLOR_CD_1'].value_counts()
    if not color_counts.empty:
        with st.expander("🎨 Color Distribution in this Khipu"):
            color_df = color_counts.reset_index()
            color_df.columns = ['Color Code', 'Count']
            st.dataframe(color_df, width=300)
    
    # 3D Visualization
    with st.expander("📈 3D Khipu Visualization", expanded=True):
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            # Hide loading indicator after chart is displayed
            loading_container.empty()
        else:
            st.error("Could not create visualization")
            loading_container.empty()
    
    # Cord data table
    with st.expander("📊 View Cord Data"):
        display_cols = ['CORD_ID', 'CORD_LEVEL', 'CORD_LENGTH', 'COLOR_CD_1', 'CORD_CLASSIFICATION']
        available_cols = [col for col in display_cols if col in khipu_data.columns]
        st.dataframe(khipu_data[available_cols].head(50), use_container_width=True)
else:
    st.warning("No data available for this khipu")
