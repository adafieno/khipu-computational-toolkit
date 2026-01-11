"""
DEPRECATED: This viewer has been replaced by khipu_3d_viewer.py
This file is kept for historical reference only.
Please use: streamlit run scripts/khipu_3d_viewer.py
"""

"""
3D Khipu Structure Visualization

Creates interactive 3D visualizations of khipu hierarchical structures
using matplotlib 3D plotting with:
- Hierarchical layout of cord relationships
- Interactive rotation and zoom
- Color-coded nodes by value/level/color
- Parent-child edge visualization
- Export to image and data

Usage: python scripts/visualize_3d_khipu.py --khipu-id <ID>
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import argparse  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402


def load_khipu_data(khipu_id):
    """Load hierarchical structure, values, and knot information for a khipu."""
    import sqlite3
    config = get_config()
    hierarchy = pd.read_csv(config.get_processed_file("cord_hierarchy.csv", 2))
    numeric_values = pd.read_csv(
        config.get_processed_file("cord_numeric_values.csv", 1))

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
        how='left'
    )
    
    # Load knot data from database
    conn = sqlite3.connect(config.get_database_path())
    knots_query = """
        SELECT k.CORD_ID, k.KNOT_ID, k.KNOT_ORDINAL, k.TYPE_CODE, k.NUM_TURNS
        FROM knot k
        JOIN cord c ON k.CORD_ID = c.CORD_ID
        WHERE c.KHIPU_ID = ?
        ORDER BY k.CORD_ID, k.KNOT_ORDINAL
    """
    knots = pd.read_sql_query(knots_query, conn, params=(khipu_id,))
    conn.close()

    return khipu_data, knots


def build_network(khipu_data):
    """Build NetworkX graph from cord hierarchy."""
    G = nx.DiGraph()

    for _, row in khipu_data.iterrows():
        cord_id = row['CORD_ID']
        parent_id = row['PENDANT_FROM']
        level = row['CORD_LEVEL'] if pd.notna(row['CORD_LEVEL']) else 0
        numeric_value = row['numeric_value'] if pd.notna(
            row['numeric_value']) else 0
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
    """Compute 3D positions for nodes with main cord horizontal and pendants hanging down."""
    pos = {}

    # Get level information
    levels = nx.get_node_attributes(G, 'level')

    # Find main cord (level 0 nodes)
    main_cord_nodes = [node for node, level in levels.items() if level == 0]
    
    # Find level 1 nodes (primary pendants)
    level_1_nodes = sorted([node for node, level in levels.items() if level == 1])
    
    # Position main cord horizontally along x-axis
    for i, node in enumerate(sorted(main_cord_nodes)):
        pos[node] = (i * 0.5, 0, 0)  # Horizontal line
    
    # Position level 1 pendants hanging down from main cord
    # Use actual cord lengths scaled to visualization space
    main_y = 0
    main_z = 0
    for i, node in enumerate(level_1_nodes):
        x_pos = i * 0.8  # Spacing along main cord
        cord_length = G.nodes[node]['cord_length']
        depth = -cord_length / 50.0  # Scale cord length (cm) to visualization units
        pos[node] = (x_pos, main_y, main_z + depth)
    
    # TEMPORARILY DISABLED: Position pendants hanging downward
    # pendant_spacing = {}
    # for level in range(2, max(levels.values()) + 1 if levels.values() else 2):
        pendant_nodes = [node for node, lvl in levels.items() if lvl == level]
        
        for node in pendant_nodes:
            # Find parent
            parents = [n for n in G.predecessors(node)]
            if parents:
                parent_pos = pos[parents[0]]
                
                # Track spacing for siblings
                parent_key = parents[0]
                if parent_key not in pendant_spacing:
                    pendant_spacing[parent_key] = 0
                
                # L-shaped elbow: barely visible horizontal offset then drop straight down
                sibling_offset = pendant_spacing[parent_key] - 0.5
                # Tiny fixed horizontal elbow
                x_offset = 0.01  # Barely visible horizontal extension
                y_offset = sibling_offset * 0.005  # Minimal lateral spread for siblings
                
                # Position hangs straight down from horizontally offset point
                pos[node] = (
                    parent_pos[0] + x_offset,
                    parent_pos[1] + y_offset,
                    parent_pos[2] - 1.0  # Drop much further down
                )
                pendant_spacing[parent_key] += 1

    return pos, level_1_nodes


def visualize_3d_khipu(khipu_id, color_mode='value', output_file=None):
    """
    Create 3D visualization of khipu structure.

    Args:
        khipu_id: Khipu ID to visualize
        color_mode: 'value' (numeric value), 'level' (hierarchy level), or 'color' (cord color)
        output_file: Optional output filename (PNG)
    """
    # Load data
    print(f"Loading data for khipu {khipu_id}...")
    khipu_data, knots = load_khipu_data(khipu_id)

    if len(khipu_data) == 0:
        print(f"No data found for khipu {khipu_id}")
        return

    print(f"Building network with {len(khipu_data)} cords...")
    print(f"Found {len(knots)} knots")
    G = build_network(khipu_data)

    print("Computing 3D layout...")
    pos, level_1_nodes = compute_3d_layout(G)

    # Prepare figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]

    # Color mapping
    if color_mode == 'value':
        values = [G.nodes[node]['value'] for node in G.nodes()]
        colors = values
        cmap = cm.viridis
        norm = Normalize(vmin=min(values) if values else 0, vmax=max(values) if values else 1)
        label = 'Numeric Value'
    elif color_mode == 'level':
        levels = [G.nodes[node]['level'] for node in G.nodes()]
        colors = levels
        cmap = cm.plasma
        norm = Normalize(vmin=min(levels) if levels else 0, vmax=max(levels) if levels else 1)
        label = 'Hierarchy Level'
    else:  # color mode
        colors = ['steelblue'] * len(G.nodes())
        cmap = None
        norm = None
        label = 'Cord'

    # Draw main cord as thick horizontal line above the pendants
    if level_1_nodes:
        main_x = [pos[node][0] for node in level_1_nodes]
        main_y = [0] * len(level_1_nodes)
        main_z = [0] * len(level_1_nodes)
        ax.plot(main_x, main_y, main_z, color='#6B5345', alpha=0.95, linewidth=10, zorder=5, label='Main Cord')
    
    # Draw pendant cords from main cord to pendant endpoints
    levels_dict = nx.get_node_attributes(G, 'level')
    for node in G.nodes():
        if node in pos:
            node_pos = pos[node]
            node_level = levels_dict.get(node, 1)
            
            # Only draw from main cord for level 1 pendants
            if node_level == 1:
                ax.plot([node_pos[0], node_pos[0]], [0, node_pos[1]], [0, node_pos[2]], 
                       color='#8B7355', alpha=0.7, linewidth=4, zorder=2)
            
            # Draw subsidiary connections
            for child in G.successors(node):
                if child in pos:
                    child_pos = pos[child]
                    ax.plot([node_pos[0], child_pos[0]], [node_pos[1], child_pos[1]], 
                           [node_pos[2], child_pos[2]], color='#8B7355', alpha=0.6, linewidth=3, zorder=2)
    
    # Draw actual knots along the cords
    knot_positions = []
    knot_colors_list = []
    knot_sizes = []
    
    for _, knot_row in knots.iterrows():
        cord_id = knot_row['CORD_ID']
        if cord_id in pos:
            cord_pos = pos[cord_id]
            cord_level = levels_dict.get(cord_id, 1)
            
            # Get knot position along the cord
            total_knots = len(knots[knots['CORD_ID'] == cord_id])
            knot_ord = knot_row['KNOT_ORDINAL'] if pd.notna(knot_row['KNOT_ORDINAL']) else 1
            
            if cord_level == 1:
                # For level 1 pendants, interpolate from main cord to pendant endpoint
                t = knot_ord / (total_knots + 1)
                knot_x = cord_pos[0]
                knot_y = 0 + t * (cord_pos[1] - 0)
                knot_z = 0 + t * (cord_pos[2] - 0)
            else:
                # For subsidiary pendants, find parent
                parents = list(G.predecessors(cord_id))
                if parents:
                    parent_pos = pos[parents[0]]
                    t = knot_ord / (total_knots + 1)
                    knot_x = parent_pos[0] + t * (cord_pos[0] - parent_pos[0])
                    knot_y = parent_pos[1] + t * (cord_pos[1] - parent_pos[1])
                    knot_z = parent_pos[2] + t * (cord_pos[2] - parent_pos[2])
                else:
                    continue
            
            knot_positions.append([knot_x, knot_y, knot_z])
            
            # Color by parent cord's value
            if cmap and colors:
                node_idx = list(G.nodes()).index(cord_id)
                knot_colors_list.append(colors[node_idx])
            
            # Much smaller size for better proportions
            knot_sizes.append(80)
    
    # Plot knots as prominent spheres
    print(f"Rendering {len(knot_positions)} knots...")
    if knot_positions:
        knot_arr = np.array(knot_positions)
        if cmap and knot_colors_list:
            scatter = ax.scatter(
                knot_arr[:, 0], knot_arr[:, 1], knot_arr[:, 2],
                c=knot_colors_list,
                cmap=cmap,
                norm=norm,
                s=knot_sizes,
                alpha=1.0,
                marker='o',
                zorder=20)
            plt.colorbar(scatter, ax=ax, label=label, shrink=0.5)
        else:
            ax.scatter(
                knot_arr[:, 0], knot_arr[:, 1], knot_arr[:, 2],
                c='#CD853F',
                s=knot_sizes,
                alpha=1.0,
                marker='o',
                zorder=20)

    # Labels and title
    ax.set_xlabel('Length along main cord', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Pendant depth', fontsize=12)
    ax.set_title(
        f'3D Khipu Structure - ID {khipu_id}\n{len(G.nodes())} knots, {len(G.edges())} cords',
        fontsize=14,
        fontweight='bold')

    # Adjust viewing angle for better perspective
    ax.view_init(elev=15, azim=-70)

    # Minimal grid for cleaner look
    ax.grid(True, alpha=0.2)
    
    # Set background color for depth perception
    ax.xaxis.pane.set_facecolor('#f5f5dc')
    ax.yaxis.pane.set_facecolor('#f5f5dc')
    ax.zaxis.pane.set_facecolor('#f5f5dc')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()

    plt.close()


def create_multiple_views(khipu_id, output_prefix='khipu_3d'):
    """Create multiple viewing angles of the same khipu."""
    # Load data
    khipu_data, knots = load_khipu_data(khipu_id)
    G = build_network(khipu_data)
    pos, level_1_nodes = compute_3d_layout(G)

    angles = [
        (20, 45),   # Default
        (30, 90),   # Side view
        (60, 135),  # Top-side view
        (10, 180)   # Front view
    ]

    fig = plt.figure(figsize=(20, 15))

    for idx, (elev, azim) in enumerate(angles, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')

        # Extract positions
        xs = [pos[node][0] for node in G.nodes()]
        ys = [pos[node][1] for node in G.nodes()]
        zs = [pos[node][2] for node in G.nodes()]
        levels = [G.nodes[node]['level'] for node in G.nodes()]

        # Draw cords as thick tubes
        for edge in G.edges():
            x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
            y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
            z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_edge, y_edge, z_edge, color='#8B7355', alpha=0.7, linewidth=3)

        # Draw knots as spheres
        _ = ax.scatter(xs, ys, zs, c=levels, cmap=cm.plasma,
                       s=100, alpha=0.9, edgecolors='#654321', linewidth=1)

        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Level', fontsize=10)
        ax.set_title(f'View {idx}: elev={elev}°, azim={azim}°', fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Khipu {khipu_id} - Multiple Views\n{len(G.nodes())} cords',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    config = get_config()
    output_dir = config.root_dir / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{output_prefix}_{khipu_id}_multiview.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved multi-view to {output_file}")
    plt.close()


def visualize_summation_flow(khipu_id):
    """Visualize summation relationships with highlighted paths."""
    # Load data
    khipu_data, knots = load_khipu_data(khipu_id)
    G = build_network(khipu_data)
    pos, level_1_nodes = compute_3d_layout(G)

    # Identify summation relationships
    summation_edges = []
    for parent in G.nodes():
        children = list(G.successors(parent))
        if len(children) > 1:  # Potential summation
            parent_val = G.nodes[parent]['value']
            child_sum = sum(G.nodes[child]['value'] for child in children)

            if abs(parent_val - child_sum) <= 1:  # Tolerance ±1
                for child in children:
                    summation_edges.append((parent, child))

    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]

    # Draw regular cords as brown tubes
    for edge in G.edges():
        if edge not in summation_edges:
            x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
            y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
            z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_edge, y_edge, z_edge, color='#8B7355', alpha=0.6, linewidth=4)

    # Draw summation cords (highlighted in red)
    for edge in summation_edges:
        x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
        y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
        z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_edge, y_edge, z_edge, color='red', alpha=0.9, linewidth=5)

    # Draw knots as spheres colored by value
    values = [G.nodes[node]['value'] for node in G.nodes()]
    scatter = ax.scatter(xs, ys, zs, c=values, cmap=cm.viridis,
                         s=150, alpha=0.95, edgecolors='#654321', linewidth=1.5)
    plt.colorbar(scatter, ax=ax, label='Numeric Value', shrink=0.5)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Hierarchy Level', fontsize=12)
    ax.set_title(
        f'Summation Flow - Khipu {khipu_id}\n{len(summation_edges)} summation relationships (red edges)',
        fontsize=14,
        fontweight='bold')
    ax.view_init(elev=25, azim=60)
    ax.grid(True, alpha=0.3)

    config = get_config()
    output_dir = config.root_dir / "outputs" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"khipu_{khipu_id}_summation_flow.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summation flow to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='3D Khipu Structure Visualization')
    parser.add_argument(
        '--khipu-id',
        type=int,
        required=True,
        help='Khipu ID to visualize')
    parser.add_argument(
        '--color-mode',
        choices=[
            'value',
            'level',
            'color'],
        default='value',
        help='Node coloring: value (numeric), level (hierarchy), or color (cord color)')
    parser.add_argument('--output', type=str, help='Output filename (PNG)')
    parser.add_argument(
        '--multi-view',
        action='store_true',
        help='Create multiple viewing angles')
    parser.add_argument(
        '--summation-flow',
        action='store_true',
        help='Highlight summation relationships')

    args = parser.parse_args()

    if args.multi_view:
        create_multiple_views(args.khipu_id)
    elif args.summation_flow:
        visualize_summation_flow(args.khipu_id)
    else:
        visualize_3d_khipu(args.khipu_id, args.color_mode, args.output)


if __name__ == "__main__":
    main()
