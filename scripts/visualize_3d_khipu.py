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


def load_khipu_data(khipu_id):
    """Load hierarchical structure and values for a khipu."""
    config = get_config()
    hierarchy = pd.read_csv(config.get_processed_file("cord_hierarchy.csv", 2))
    numeric_values = pd.read_csv(
        config.get_processed_file("cord_numeric_values.csv", 1))

    # Filter for specific khipu
    khipu_cords = hierarchy[hierarchy['KHIPU_ID'] == khipu_id].copy()
    khipu_values = numeric_values[numeric_values['khipu_id'] == khipu_id].copy(
    )

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
        numeric_value = row['numeric_value'] if pd.notna(
            row['numeric_value']) else 0

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
    khipu_data = load_khipu_data(khipu_id)

    if len(khipu_data) == 0:
        print(f"No data found for khipu {khipu_id}")
        return

    print(f"Building network with {len(khipu_data)} cords...")
    G = build_network(khipu_data)

    print("Computing 3D layout...")
    pos = compute_3d_layout(G)

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
        norm = Normalize(vmin=min(values), vmax=max(values))
        label = 'Numeric Value'
    elif color_mode == 'level':
        levels = [G.nodes[node]['level'] for node in G.nodes()]
        colors = levels
        cmap = cm.plasma
        norm = Normalize(vmin=min(levels), vmax=max(levels))
        label = 'Hierarchy Level'
    else:  # color mode
        colors = ['steelblue'] * len(G.nodes())
        cmap = None
        norm = None
        label = 'Cord'

    # Draw edges
    for edge in G.edges():
        x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
        y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
        z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_edge, y_edge, z_edge, 'gray', alpha=0.3, linewidth=0.5)

    # Draw nodes
    if cmap:
        scatter = ax.scatter(
            xs,
            ys,
            zs,
            c=colors,
            cmap=cmap,
            norm=norm,
            s=50,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label=label, shrink=0.5)
    else:
        ax.scatter(xs, ys, zs, c=colors, s=50, alpha=0.8,
                   edgecolors='black', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Hierarchy Level (depth)', fontsize=12)
    ax.set_title(
        f'3D Khipu Structure - ID {khipu_id}\n{len(G.nodes())} cords, {len(G.edges())} connections',
        fontsize=14,
        fontweight='bold')

    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)

    # Grid
    ax.grid(True, alpha=0.3)

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
    khipu_data = load_khipu_data(khipu_id)
    G = build_network(khipu_data)
    pos = compute_3d_layout(G)

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

        # Draw edges
        for edge in G.edges():
            x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
            y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
            z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_edge, y_edge, z_edge, 'gray', alpha=0.2, linewidth=0.5)

        # Draw nodes
        _ = ax.scatter(xs, ys, zs, c=levels, cmap=cm.plasma,
                       s=40, alpha=0.8, edgecolors='black', linewidth=0.5)

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
    khipu_data = load_khipu_data(khipu_id)
    G = build_network(khipu_data)
    pos = compute_3d_layout(G)

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

    # Draw regular edges
    for edge in G.edges():
        if edge not in summation_edges:
            x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
            y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
            z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_edge, y_edge, z_edge, 'gray', alpha=0.2, linewidth=0.5)

    # Draw summation edges (highlighted)
    for edge in summation_edges:
        x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
        y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
        z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_edge, y_edge, z_edge, 'red', alpha=0.8, linewidth=2)

    # Draw nodes
    values = [G.nodes[node]['value'] for node in G.nodes()]
    scatter = ax.scatter(xs, ys, zs, c=values, cmap=cm.viridis,
                         s=60, alpha=0.9, edgecolors='black', linewidth=0.5)
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
