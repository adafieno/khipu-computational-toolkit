"""
Build NetworkX graph representations of all khipus for pattern discovery.
"""

from pathlib import Path
import sys

# Add src to path for runtime
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from graph.graph_builder import KhipuGraphBuilder  # noqa: E402 # type: ignore
from config import get_config  # noqa: E402 # type: ignore


def main():
    print("=" * 80)
    print("KHIPU GRAPH CONSTRUCTION")
    print("=" * 80)
    print()
    print("Converting khipus to NetworkX directed graphs...")
    print("Each graph represents:")
    print("  - Nodes: Cords with numeric values, colors, and hierarchy")
    print("  - Edges: Pendant relationships (parent -> child)")
    print()

    # Get configuration
    config = get_config()

    # Validate setup
    validation = config.validate_setup()
    if not validation['valid']:
        print("\nConfiguration errors:")
        for error in validation['errors']:
            print(f"  • {error}")
        sys.exit(1)

    print(f"Database: {config.get_database_path()}")
    print()

    # Initialize builder
    db_path = config.get_database_path()
    builder = KhipuGraphBuilder(db_path)

    # Analyze a sample khipu first
    print("Analyzing sample khipu structure...")
    print("-" * 80)

    sample_stats = builder.analyze_graph_structure(1000000)

    print(f"Sample khipu {sample_stats['khipu_id']}:")
    print(f"  Nodes (cords): {sample_stats['num_nodes']}")
    print(f"  Edges (pendant relationships): {sample_stats['num_edges']}")
    depth_info = (f"  Depth range: {sample_stats['min_depth']} to {sample_stats['max_depth']} "
                  f"(span: {sample_stats['depth_range']})")
    print(depth_info)
    print(f"  Avg branching factor: {sample_stats['avg_branching_factor']:.2f}")
    print(f"  Max branching factor: {sample_stats['max_branching_factor']}")
    print(f"  Root nodes: {sample_stats['num_roots']}")
    print(f"  Leaf nodes: {sample_stats['num_leaves']}")
    print(f"  Connected components: {sample_stats['weakly_connected_components']}")
    print(f"  Graph density: {sample_stats['density']:.4f}")
    print()

    # Build all graphs
    print("Building all khipu graphs...")
    print("-" * 80)

    # Ensure directories exist
    config.ensure_directories()

    output_dir = config.graphs_dir

    graphs = builder.build_all_graphs(output_dir)

    print()
    print("=" * 80)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated {len(graphs)} graphs")
    print(f"Total nodes across all graphs: {sum(G.number_of_nodes() for G in graphs):,}")
    print(f"Total edges across all graphs: {sum(G.number_of_edges() for G in graphs):,}")
    print()
    print("Output files:")
    print(f"  {output_dir / 'khipu_graphs.pkl'} (NetworkX graphs)")
    print(f"  {output_dir / 'khipu_graphs_metadata.json'} (statistics)")
    print()
    print("Next steps:")
    print("  1. Compute graph similarity metrics")
    print("  2. Cluster khipus by structural patterns")
    print("  3. Find recurring subgraph motifs")
    print("  4. Correlate graph structure with geographic provenance")


if __name__ == "__main__":
    main()
