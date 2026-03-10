"""
Subgraph Motif Mining

This script mines recurring structural motifs (subgraph patterns) in khipu graphs:
1. Extract frequent subgraph patterns within clusters
2. Identify common cord arrangement motifs
3. Analyze motif distribution across provenances
4. Correlate motifs with summation patterns
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pandas as pd  # noqa: E402
import sqlite3  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import networkx as nx  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402
from collections import defaultdict, Counter  # noqa: E402


class MotifMiner:
    """Mine recurring structural motifs in khipu graphs."""
    
    def __init__(self, db_path: str = None):
        config = get_config()
        self.config = config
        self.db_path = db_path if db_path else config.get_database_path()
        self.conn = sqlite3.connect(self.db_path)
        
    def load_data(self) -> Dict:
        """Load necessary data files."""
        print("Loading data files...")
        
        data = {
            'clusters': pd.read_csv(self.config.get_processed_file("cluster_assignments_kmeans.csv", 4)),
            'features': pd.read_csv(self.config.get_processed_file("graph_structural_features.csv", 4))
        }
        
        # Load graphs
        graphs_path = self.config.root_dir / "data" / "graphs" / "khipu_graphs.pkl"
        with open(graphs_path, "rb") as f:
            graphs_list = pickle.load(f)
            data['graphs'] = {g.graph['khipu_id']: g for g in graphs_list}
        
        # Load provenance
        query = "SELECT KHIPU_ID, PROVENANCE FROM khipu_main"
        data['provenance'] = pd.read_sql_query(query, self.conn)
        
        print(f"✓ Loaded {len(data['clusters'])} cluster assignments")
        print(f"✓ Loaded {len(data['graphs'])} graphs")
        
        return data
    
    def extract_degree_sequence_motif(self, graph: nx.DiGraph, node: int) -> Tuple:
        """Extract local degree sequence motif around a node."""
        in_deg = graph.in_degree(node)
        out_deg = graph.out_degree(node)
        
        # Get degrees of neighbors
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        
        pred_out_degs = tuple(sorted([graph.out_degree(p) for p in predecessors]))
        succ_in_degs = tuple(sorted([graph.in_degree(s) for s in successors]))
        
        return (in_deg, out_deg, pred_out_degs, succ_in_degs)
    
    def extract_branching_motifs(self, graph: nx.DiGraph) -> List[Tuple]:
        """Extract branching patterns (parent → children structures)."""
        motifs = []
        
        for node in graph.nodes():
            out_deg = graph.out_degree(node)
            
            if out_deg > 0:  # Has children
                # Get children's properties
                children = list(graph.successors(node))
                child_out_degs = tuple(sorted([graph.out_degree(c) for c in children]))
                
                # Check if node has numeric value
                has_numeric = graph.nodes[node].get('numeric_value') is not None
                children_numeric = sum(1 for c in children 
                                      if graph.nodes[c].get('numeric_value') is not None)
                
                motif = (
                    out_deg,  # Number of children
                    child_out_degs,  # Children's branching
                    has_numeric,  # Parent has value
                    children_numeric  # How many children have values
                )
                motifs.append(motif)
        
        return motifs
    
    def extract_depth_motifs(self, graph: nx.DiGraph) -> List[Tuple]:
        """Extract depth-based patterns (levels in hierarchy)."""
        if not nx.is_directed_acyclic_graph(graph):
            return []
        
        # Find root nodes
        roots = [n for n, d in graph.in_degree() if d == 0]
        
        if not roots:
            return []
        
        # Compute node levels
        node_levels = {}
        queue = [(root, 0) for root in roots]
        visited = set()
        
        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            node_levels[node] = level
            
            for successor in graph.successors(node):
                if successor not in visited:
                    queue.append((successor, level + 1))
        
        # Extract level-to-level patterns
        motifs = []
        max_level = max(node_levels.values()) if node_levels else 0
        
        for level in range(max_level):
            level_nodes = [n for n, lvl in node_levels.items() if lvl == level]
            next_level_nodes = [n for n, lvl in node_levels.items() if lvl == level + 1]
            
            # Pattern: (nodes at level, nodes at next level, connections)
            connections = sum(1 for n in level_nodes 
                            for s in graph.successors(n) if s in next_level_nodes)
            
            motifs.append((len(level_nodes), len(next_level_nodes), connections))
        
        return motifs
    
    def mine_cluster_motifs(self, data: Dict, cluster_id: int, 
                           sample_size: int = 50) -> Dict:
        """Mine motifs within a specific cluster."""
        cluster_khipus = data['clusters'][
            data['clusters']['cluster'] == cluster_id
        ]['khipu_id'].tolist()
        
        # Sample if too many
        if len(cluster_khipus) > sample_size:
            import random
            cluster_khipus = random.sample(cluster_khipus, sample_size)
        
        branching_motifs = []
        depth_motifs = []
        
        for khipu_id in cluster_khipus:
            if khipu_id not in data['graphs']:
                continue
            
            graph = data['graphs'][khipu_id]
            branching_motifs.extend(self.extract_branching_motifs(graph))
            depth_motifs.extend(self.extract_depth_motifs(graph))
        
        # Count motif frequencies
        branching_counts = Counter(branching_motifs)
        depth_counts = Counter(depth_motifs)
        
        return {
            'cluster_id': int(cluster_id),
            'num_khipus_analyzed': len(cluster_khipus),
            'branching_motifs': {
                'total': len(branching_motifs),
                'unique': len(branching_counts),
                'most_common': [(str(m), int(c)) for m, c in branching_counts.most_common(10)]
            },
            'depth_motifs': {
                'total': len(depth_motifs),
                'unique': len(depth_counts),
                'most_common': [(str(m), int(c)) for m, c in depth_counts.most_common(10)]
            }
        }
    
    def analyze_all_clusters(self, data: Dict) -> Dict:
        """Analyze motifs across all clusters."""
        print("\n" + "="*80)
        print("MOTIF MINING BY CLUSTER")
        print("="*80)
        
        cluster_ids = sorted(data['clusters']['cluster'].unique())
        cluster_ids = [c for c in cluster_ids if c != -1]  # Remove noise
        
        results = {}
        
        for cluster_id in cluster_ids:
            print(f"\nAnalyzing Cluster {cluster_id}...")
            motifs = self.mine_cluster_motifs(data, cluster_id)
            results[cluster_id] = motifs
            
            print(f"  Khipus analyzed: {motifs['num_khipus_analyzed']}")
            print(f"  Branching motifs: {motifs['branching_motifs']['total']} total, "
                  f"{motifs['branching_motifs']['unique']} unique")
            print(f"  Depth motifs: {motifs['depth_motifs']['total']} total, "
                  f"{motifs['depth_motifs']['unique']} unique")
            
            if motifs['branching_motifs']['most_common']:
                top_motif = motifs['branching_motifs']['most_common'][0]
                print(f"  Most common branching: {top_motif[0]} (count: {top_motif[1]})")
        
        return results
    
    def find_universal_motifs(self, cluster_results: Dict) -> Dict:
        """Find motifs that appear across multiple clusters."""
        print("\n" + "="*80)
        print("UNIVERSAL MOTIFS (ACROSS CLUSTERS)")
        print("="*80)
        
        # Collect all motifs with their cluster sources
        branching_by_cluster = defaultdict(set)
        depth_by_cluster = defaultdict(set)
        
        for cluster_id, results in cluster_results.items():
            for motif, count in results['branching_motifs']['most_common']:
                branching_by_cluster[motif].add(cluster_id)
            
            for motif, count in results['depth_motifs']['most_common']:
                depth_by_cluster[motif].add(cluster_id)
        
        # Find motifs in multiple clusters
        universal_branching = {
            motif: clusters 
            for motif, clusters in branching_by_cluster.items() 
            if len(clusters) >= 3
        }
        
        universal_depth = {
            motif: clusters 
            for motif, clusters in depth_by_cluster.items() 
            if len(clusters) >= 3
        }
        
        print(f"\nBranching motifs in ≥3 clusters: {len(universal_branching)}")
        if universal_branching:
            print("\nTop universal branching motifs:")
            for motif, clusters in list(universal_branching.items())[:5]:
                print(f"  {motif}: in clusters {sorted(clusters)}")
        
        print(f"\nDepth motifs in ≥3 clusters: {len(universal_depth)}")
        if universal_depth:
            print("\nTop universal depth motifs:")
            for motif, clusters in list(universal_depth.items())[:5]:
                print(f"  {motif}: in clusters {sorted(clusters)}")
        
        return {
            'universal_branching': {str(k): [int(v) for v in vals] for k, vals in universal_branching.items()},
            'universal_depth': {str(k): [int(v) for v in vals] for k, vals in universal_depth.items()}
        }
    
    def analyze_simple_chain_motif(self, data: Dict) -> Dict:
        """Analyze the simple linear chain motif (depth=2, single branch)."""
        print("\n" + "="*80)
        print("SIMPLE CHAIN MOTIF ANALYSIS")
        print("="*80)
        
        # Find khipus with simple chain structure
        simple_chains = []
        
        for khipu_id, features in data['features'].iterrows():
            if (features['depth'] == 2 and 
                features['avg_branching'] < 1.1 and
                features['num_nodes'] > 5):
                
                simple_chains.append({
                    'khipu_id': khipu_id,
                    'num_nodes': features['num_nodes'],
                    'cluster': data['clusters'][
                        data['clusters']['khipu_id'] == khipu_id
                    ]['cluster'].values[0] if len(data['clusters'][
                        data['clusters']['khipu_id'] == khipu_id
                    ]) > 0 else -1
                })
        
        print(f"\nIdentified {len(simple_chains)} simple chain khipus")
        
        if simple_chains:
            print(f"  Size range: {min(k['num_nodes'] for k in simple_chains):.0f} to "
                  f"{max(k['num_nodes'] for k in simple_chains):.0f} nodes")
            
            # Cluster distribution
            cluster_dist = Counter([k['cluster'] for k in simple_chains])
            print("\nCluster distribution:")
            for cluster, count in cluster_dist.most_common():
                print(f"  Cluster {cluster}: {count} khipus")
        else:
            print("  (None found with current criteria)")
            cluster_dist = {}
        
        return {
            'count': len(simple_chains),
            'khipu_ids': [int(k['khipu_id']) for k in simple_chains],
            'cluster_distribution': {int(k): int(v) for k, v in cluster_dist.items()}
        }
    
    def export_results(self, results: Dict, output_dir: str = None):
        """Export motif mining results."""
        if output_dir is None:
            output_dir = self.config.processed_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_json = Path(output_dir) / "motif_mining_results.json"
        
        # Convert cluster_motifs keys to int
        cluster_motifs_serializable = {
            int(k): v for k, v in results['cluster_motifs'].items()
        }
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'cluster_motifs': cluster_motifs_serializable,
            'universal_motifs': results['universal_motifs'],
            'simple_chain_analysis': results['simple_chains']
        }
        
        with open(output_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Exported analysis to {output_json}")
    
    def run_analysis(self):
        """Run complete motif mining analysis."""
        print("="*80)
        print("SUBGRAPH MOTIF MINING")
        print("="*80)
        
        # Load data
        data = self.load_data()
        
        # Mine motifs by cluster
        cluster_motifs = self.analyze_all_clusters(data)
        
        # Find universal motifs
        universal_motifs = self.find_universal_motifs(cluster_motifs)
        
        # Analyze simple chain motif
        simple_chains = self.analyze_simple_chain_motif(data)
        
        # Compile results
        results = {
            'cluster_motifs': cluster_motifs,
            'universal_motifs': universal_motifs,
            'simple_chains': simple_chains
        }
        
        # Export
        self.export_results(results)
        
        print("\n" + "="*80)
        print("MOTIF MINING COMPLETE")
        print("="*80)
        
        return results
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    miner = MotifMiner()
    miner.run_analysis()
