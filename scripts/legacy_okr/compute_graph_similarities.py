"""
Compute Graph Similarity Metrics

This script computes pairwise similarity metrics between khipu graphs to enable
clustering and pattern discovery. Uses multiple graph similarity measures:
1. Node/edge count similarity (simple baseline)
2. Degree distribution similarity
3. Graph edit distance (for small subsets)
4. Structural feature similarity (depth, branching, etc.)

For large-scale comparison, uses efficient feature-based methods rather than
expensive graph isomorphism algorithms.
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import pickle  # noqa: E402
import json  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import networkx as nx  # noqa: E402
from datetime import datetime  # noqa: E402
from scipy.spatial.distance import cosine, euclidean  # noqa: E402
from collections import defaultdict  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402


class GraphSimilarityComputer:
    """Compute pairwise similarity metrics between khipu graphs."""
    
    def __init__(self, graphs_path: str = None):
        config = get_config()
        self.config = config
        if graphs_path is None:
            graphs_path = config.root_dir / "data" / "graphs" / "khipu_graphs.pkl"
        print(f"Loading graphs from {graphs_path}...")
        with open(graphs_path, 'rb') as f:
            graphs_list = pickle.load(f)
        
        # Convert list to dict keyed by khipu_id
        self.graphs = {}
        for graph in graphs_list:
            if 'khipu_id' in graph.graph:
                khipu_id = graph.graph['khipu_id']
                self.graphs[khipu_id] = graph
        
        print(f"✓ Loaded {len(self.graphs)} graphs")
        
    def extract_structural_features(self, graph: nx.DiGraph, khipu_id: int) -> Dict:
        """Extract structural features from a graph for similarity comparison."""
        if len(graph.nodes()) == 0:
            return {
                'khipu_id': khipu_id,
                'num_nodes': 0,
                'num_edges': 0,
                'avg_degree': 0,
                'max_degree': 0,
                'density': 0,
                'depth': 0,
                'width': 0,
                'avg_branching': 0,
                'num_roots': 0,
                'num_leaves': 0,
                'has_numeric': 0,
                'has_color': 0,
                'avg_numeric_value': 0,
                'std_numeric_value': 0
            }
        
        # Basic statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        # Degree statistics
        degrees = [d for n, d in graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        # Graph density
        density = nx.density(graph)
        
        # Hierarchy statistics
        in_degrees = [d for n, d in graph.in_degree()]
        out_degrees = [d for n, d in graph.out_degree()]
        
        num_roots = sum(1 for d in in_degrees if d == 0)
        num_leaves = sum(1 for d in out_degrees if d == 0)
        
        # Compute depth (longest path from root)
        depth = 0
        for node in graph.nodes():
            if graph.in_degree(node) == 0:  # Root node
                try:
                    lengths = nx.single_source_shortest_path_length(graph, node)
                    max_length = max(lengths.values()) if lengths else 0
                    depth = max(depth, max_length)
                except Exception:
                    pass
        
        # Width (max nodes at any level)
        if depth > 0:
            level_counts = defaultdict(int)
            for node in graph.nodes():
                if 'level' in graph.nodes[node]:
                    level = graph.nodes[node]['level']
                    level_counts[level] += 1
            width = max(level_counts.values()) if level_counts else num_nodes
        else:
            width = num_nodes
        
        # Branching factor
        avg_branching = np.mean([d for d in out_degrees if d > 0]) if any(d > 0 for d in out_degrees) else 0
        
        # Numeric data presence
        numeric_values = []
        has_numeric_count = 0
        has_color_count = 0
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if 'numeric_value' in node_data and node_data['numeric_value'] is not None:
                has_numeric_count += 1
                numeric_values.append(node_data['numeric_value'])
            if 'color_full' in node_data and node_data['color_full']:
                has_color_count += 1
        
        avg_numeric = np.mean(numeric_values) if numeric_values else 0
        std_numeric = np.std(numeric_values) if numeric_values else 0
        
        return {
            'khipu_id': khipu_id,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'density': density,
            'depth': depth,
            'width': width,
            'avg_branching': avg_branching,
            'num_roots': num_roots,
            'num_leaves': num_leaves,
            'has_numeric': has_numeric_count / num_nodes if num_nodes > 0 else 0,
            'has_color': has_color_count / num_nodes if num_nodes > 0 else 0,
            'avg_numeric_value': avg_numeric,
            'std_numeric_value': std_numeric
        }
    
    def compute_degree_distribution(self, graph: nx.DiGraph) -> np.ndarray:
        """Compute degree distribution as a feature vector."""
        if len(graph.nodes()) == 0:
            return np.zeros(10)
        
        degrees = [d for n, d in graph.degree()]
        hist, _ = np.histogram(degrees, bins=10, range=(0, max(degrees) + 1) if degrees else (0, 1))
        # Normalize to make it a probability distribution
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        return hist
    
    def compute_feature_similarity(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Compute pairwise similarity matrix based on structural features."""
        print("\nComputing feature-based similarity matrix...")
        
        # Select numeric features for similarity
        feature_cols = [
            'num_nodes', 'num_edges', 'avg_degree', 'max_degree', 'density',
            'depth', 'width', 'avg_branching', 'num_roots', 'num_leaves',
            'has_numeric', 'has_color'
        ]
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_df[feature_cols])
        
        # Compute pairwise cosine similarities
        n = len(features_df)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                # Cosine similarity (1 - cosine distance)
                sim = 1 - cosine(features_normalized[i], features_normalized[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return pd.DataFrame(
            similarity_matrix,
            index=features_df['khipu_id'],
            columns=features_df['khipu_id']
        )
    
    def compute_degree_similarity(self, khipu_ids: List[int]) -> pd.DataFrame:
        """Compute similarity based on degree distributions using KS test."""
        print("\nComputing degree distribution similarity...")
        
        degree_dists = {}
        for khipu_id in khipu_ids:
            graph = self.graphs[khipu_id]
            degree_dists[khipu_id] = self.compute_degree_distribution(graph)
        
        n = len(khipu_ids)
        similarity_matrix = np.zeros((n, n))
        
        for i, id_i in enumerate(khipu_ids):
            for j, id_j in enumerate(khipu_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Use 1 - distance as similarity
                    dist = euclidean(degree_dists[id_i], degree_dists[id_j])
                    # Normalize to 0-1 range (max distance is sqrt(2) for normalized distributions)
                    sim = 1 - min(dist / np.sqrt(2), 1.0)
                    similarity_matrix[i, j] = sim
        
        return pd.DataFrame(
            similarity_matrix,
            index=khipu_ids,
            columns=khipu_ids
        )
    
    def find_most_similar_pairs(self, similarity_df: pd.DataFrame, top_k: int = 20) -> List[Tuple]:
        """Find top-k most similar khipu pairs."""
        pairs = []
        
        for i in range(len(similarity_df)):
            for j in range(i + 1, len(similarity_df)):
                khipu_i = similarity_df.index[i]
                khipu_j = similarity_df.index[j]
                similarity = similarity_df.iloc[i, j]
                pairs.append((khipu_i, khipu_j, similarity))
        
        # Sort by similarity descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_k]
    
    def analyze_similarity_distribution(self, similarity_df: pd.DataFrame) -> Dict:
        """Analyze the distribution of similarity scores."""
        # Get upper triangle (exclude diagonal)
        n = len(similarity_df)
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(similarity_df.iloc[i, j])
        
        upper_triangle = np.array(upper_triangle)
        
        return {
            'mean_similarity': float(np.mean(upper_triangle)),
            'median_similarity': float(np.median(upper_triangle)),
            'std_similarity': float(np.std(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle)),
            'q25_similarity': float(np.percentile(upper_triangle, 25)),
            'q75_similarity': float(np.percentile(upper_triangle, 75))
        }
    
    def export_results(self, features_df: pd.DataFrame, similarity_df: pd.DataFrame,
                      top_pairs: List[Tuple], stats: Dict, output_dir: str = None):
        """Export similarity analysis results."""
        if output_dir is None:
            output_dir = self.config.processed_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export structural features
        features_csv = Path(output_dir) / "graph_structural_features.csv"
        features_df.to_csv(features_csv, index=False)
        print(f"\n✓ Exported structural features to {features_csv}")
        
        # Export similarity matrix (CSV - may be large)
        similarity_csv = Path(output_dir) / "graph_similarity_matrix.csv"
        similarity_df.to_csv(similarity_csv)
        print(f"✓ Exported similarity matrix to {similarity_csv}")
        
        # Export top similar pairs
        top_pairs_df = pd.DataFrame(top_pairs, columns=['khipu_id_1', 'khipu_id_2', 'similarity'])
        pairs_csv = Path(output_dir) / "most_similar_khipu_pairs.csv"
        top_pairs_df.to_csv(pairs_csv, index=False)
        print(f"✓ Exported top similar pairs to {pairs_csv}")
        
        # Export analysis JSON
        analysis_json = Path(output_dir) / "graph_similarity_analysis.json"
        analysis = {
            'generated_at': datetime.now().isoformat(),
            'total_khipus': len(features_df),
            'total_comparisons': len(features_df) * (len(features_df) - 1) // 2,
            'similarity_statistics': stats,
            'top_similar_pairs': [
                {'khipu_1': int(p[0]), 'khipu_2': int(p[1]), 'similarity': float(p[2])}
                for p in top_pairs[:10]
            ]
        }
        
        with open(analysis_json, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Exported analysis to {analysis_json}")
    
    def run_analysis(self):
        """Run complete graph similarity analysis."""
        print("="*80)
        print("GRAPH SIMILARITY ANALYSIS")
        print("="*80)
        
        # Extract structural features for all graphs
        print("\nExtracting structural features from all graphs...")
        features = []
        for khipu_id, graph in self.graphs.items():
            features.append(self.extract_structural_features(graph, khipu_id))
        
        features_df = pd.DataFrame(features)
        print(f"✓ Extracted features for {len(features_df)} khipus")
        
        # Compute feature-based similarity
        similarity_df = self.compute_feature_similarity(features_df)
        
        # Find most similar pairs
        print("\nFinding most similar khipu pairs...")
        top_pairs = self.find_most_similar_pairs(similarity_df, top_k=20)
        
        print("\nTop 10 Most Similar Khipu Pairs:")
        print("-"*80)
        for i, (k1, k2, sim) in enumerate(top_pairs[:10], 1):
            print(f"{i:2d}. Khipu {k1} ↔ Khipu {k2}: similarity = {sim:.4f}")
        
        # Analyze similarity distribution
        print("\n" + "-"*80)
        print("SIMILARITY DISTRIBUTION STATISTICS")
        print("-"*80)
        stats = self.analyze_similarity_distribution(similarity_df)
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Export results
        self.export_results(features_df, similarity_df, top_pairs, stats)
        
        print("\n" + "="*80)
        print("GRAPH SIMILARITY ANALYSIS COMPLETE")
        print("="*80)
        
        return features_df, similarity_df, top_pairs, stats


if __name__ == "__main__":
    # Check if sklearn is available
    try:
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError:
        print("Installing scikit-learn for feature normalization...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-learn'])
    
    computer = GraphSimilarityComputer()
    computer.run_analysis()
