"""
Graph Builder - Convert khipu hierarchical structure into NetworkX graphs.

Creates directed graphs where:
- Nodes represent cords with attributes (numeric value, color, level)
- Edges represent pendant relationships (parent -> child)
- Node/edge attributes enable pattern mining and clustering
"""

import sqlite3
import pandas as pd
import networkx as nx  # type: ignore
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import pickle


class KhipuGraphBuilder:
    """Build NetworkX graph representations of khipus."""
    
    def __init__(self, db_path: Path):
        """Initialize builder with database path."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def build_khipu_graph(self, khipu_id: int, include_colors: bool = True, 
                          include_numeric: bool = True) -> nx.DiGraph:
        """
        Build directed graph for a single khipu.
        
        Args:
            khipu_id: Khipu to convert
            include_colors: Whether to add color attributes
            include_numeric: Whether to add numeric value attributes
        
        Returns:
            NetworkX DiGraph with cord hierarchy
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get cord structure
        cord_query = """
        SELECT 
            CORD_ID,
            CORD_CLASSIFICATION,
            CORD_LEVEL,
            PENDANT_FROM,
            ATTACHED_TO,
            CORD_ORDINAL,
            CORD_LENGTH
        FROM cord
        WHERE KHIPU_ID = ?
        ORDER BY CORD_LEVEL, CORD_ORDINAL
        """
        
        cords = pd.read_sql_query(cord_query, conn, params=(khipu_id,))
        
        # Initialize graph
        G = nx.DiGraph()
        G.graph['khipu_id'] = khipu_id
        G.graph['num_cords'] = len(cords)
        
        # Add nodes (cords) with attributes
        for _, cord in cords.iterrows():
            cord_id = cord['CORD_ID']
            
            node_attrs = {
                'classification': cord['CORD_CLASSIFICATION'],
                'level': cord['CORD_LEVEL'],
                'ordinal': cord['CORD_ORDINAL'],
                'length': cord['CORD_LENGTH']
            }
            
            # Add numeric value if requested
            if include_numeric:
                numeric_value = self._get_cord_numeric_value(conn, cord_id)
                node_attrs['numeric_value'] = numeric_value
            
            # Add color if requested
            if include_colors:
                color_info = self._get_cord_color(conn, cord_id)
                node_attrs.update(color_info)
            
            G.add_node(cord_id, **node_attrs)
        
        # Add edges (pendant relationships)
        for _, cord in cords.iterrows():
            if pd.notna(cord['PENDANT_FROM']):
                parent_id = int(cord['PENDANT_FROM'])
                child_id = cord['CORD_ID']
                
                # Add edge with relationship metadata
                G.add_edge(parent_id, child_id, 
                          relationship='pendant',
                          child_ordinal=cord['CORD_ORDINAL'])
        
        conn.close()
        
        return G
    
    def _get_cord_numeric_value(self, conn: sqlite3.Connection, cord_id: int) -> Optional[int]:
        """Get numeric value encoded on cord by summing knots."""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT TYPE_CODE, knot_value_type, NUM_TURNS
            FROM knot
            WHERE CORD_ID = ?
            ORDER BY KNOT_ORDINAL
        """, (cord_id,))
        
        knots = cursor.fetchall()
        
        if not knots:
            return None
        
        total = 0
        has_value = False
        
        for knot_type, value_type, num_turns in knots:
            if value_type is None:
                continue
            
            has_value = True
            
            if knot_type == 'L':  # Long knot
                if num_turns is not None:
                    total += int(value_type * num_turns)
            elif knot_type in ('S', 'E'):  # Single or figure-eight
                total += int(value_type)
        
        return total if has_value else None
    
    def _get_cord_color(self, conn: sqlite3.Connection, cord_id: int) -> Dict:
        """Get color information for cord."""
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COLOR_CD_1, FULL_COLOR
            FROM ascher_cord_color
            WHERE CORD_ID = ?
            LIMIT 1
        """, (cord_id,))
        
        result = cursor.fetchone()
        
        if result:
            return {
                'color_primary': result[0],
                'color_full': result[1]
            }
        else:
            return {
                'color_primary': None,
                'color_full': None
            }
    
    def build_all_graphs(self, output_dir: Path, 
                        limit: Optional[int] = None) -> List[nx.DiGraph]:
        """
        Build graphs for all khipus and save to disk.
        
        Args:
            output_dir: Directory to save graph files
            limit: Optional limit on number of khipus to process
        
        Returns:
            List of generated graphs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT KHIPU_ID FROM khipu_main ORDER BY KHIPU_ID"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        khipu_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        print(f"Building graphs for {len(khipu_ids)} khipus...")
        print()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        graphs = []
        
        for i, khipu_id in enumerate(khipu_ids, 1):
            if i % 50 == 0:
                print(f"  Built {i}/{len(khipu_ids)} graphs...")
            
            G = self.build_khipu_graph(khipu_id, 
                                       include_colors=True, 
                                       include_numeric=True)
            graphs.append(G)
        
        print(f"  ✓ Built {len(khipu_ids)} graphs")
        print()
        
        # Save as pickle for efficient loading
        print("  Saving graph collection...")
        pickle_path = output_dir / "khipu_graphs.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"  ✓ Saved to {pickle_path}")
        
        # Save summary statistics
        stats = self._compute_graph_stats(graphs)
        
        json_path = output_dir / "khipu_graphs_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Metadata saved to {json_path}")
        
        return graphs
    
    def _compute_graph_stats(self, graphs: List[nx.DiGraph]) -> Dict:
        """Compute summary statistics for graph collection."""
        stats = {
            'generated_at': datetime.now().isoformat(),
            'total_graphs': len(graphs),
            'total_nodes': sum(G.number_of_nodes() for G in graphs),
            'total_edges': sum(G.number_of_edges() for G in graphs),
            'avg_nodes_per_graph': sum(G.number_of_nodes() for G in graphs) / len(graphs),
            'avg_edges_per_graph': sum(G.number_of_edges() for G in graphs) / len(graphs),
            'max_nodes': max(G.number_of_nodes() for G in graphs),
            'min_nodes': min(G.number_of_nodes() for G in graphs),
            'graphs_with_numeric_data': sum(
                1 for G in graphs 
                if any(G.nodes[n].get('numeric_value') is not None for n in G.nodes())
            ),
            'graphs_with_color_data': sum(
                1 for G in graphs 
                if any(G.nodes[n].get('color_primary') is not None for n in G.nodes())
            )
        }
        
        return stats
    
    def analyze_graph_structure(self, khipu_id: int) -> Dict:
        """
        Analyze structural properties of a khipu graph.
        
        Returns metrics like:
        - Depth (max level)
        - Branching factor (avg children per node)
        - Degree distribution
        - Connected components
        """
        G = self.build_khipu_graph(khipu_id)
        
        # Basic metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Depth (max cord level)
        levels = [G.nodes[n]['level'] for n in G.nodes() if G.nodes[n].get('level') is not None]
        max_depth = max(levels) if levels else 0
        min_depth = min(levels) if levels else 0
        
        # Branching factor (out-degree distribution)
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        avg_branching = sum(out_degrees) / num_nodes if num_nodes > 0 else 0
        max_branching = max(out_degrees) if out_degrees else 0
        
        # Find root nodes (no incoming edges)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        # Find leaf nodes (no outgoing edges)
        leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
        
        # Connectivity
        weakly_connected = nx.number_weakly_connected_components(G)
        
        return {
            'khipu_id': khipu_id,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'max_depth': int(max_depth),
            'min_depth': int(min_depth),
            'depth_range': int(max_depth - min_depth),
            'avg_branching_factor': float(avg_branching),
            'max_branching_factor': int(max_branching),
            'num_roots': len(roots),
            'num_leaves': len(leaves),
            'weakly_connected_components': weakly_connected,
            'density': nx.density(G)
        }
    
    def export_graph_to_gexf(self, khipu_id: int, output_path: Path):
        """
        Export graph to GEXF format for visualization in Gephi.
        
        Args:
            khipu_id: Khipu to export
            output_path: Path for GEXF file
        """
        G = self.build_khipu_graph(khipu_id, include_colors=True, include_numeric=True)
        
        # Convert None values to strings for GEXF compatibility
        for node in G.nodes():
            for key, value in G.nodes[node].items():
                if value is None:
                    G.nodes[node][key] = 'None'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        nx.write_gexf(G, output_path)
        
        return G
