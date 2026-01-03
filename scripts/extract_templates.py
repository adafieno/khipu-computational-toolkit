"""
Template Extraction and Analysis

This script analyzes perfect-match khipus to extract structural templates:
1. Perfect summation khipus (match rate = 1.0)
2. Perfect structural matches (similarity = 1.0)
3. Template pattern extraction and generalization
4. Template validation on similar khipus
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
from collections import defaultdict  # noqa: E402


class TemplateExtractor:
    """Extract and analyze structural templates from exemplar khipus."""
    
    def __init__(self, db_path: str = None):
        config = get_config()
        self.config = config
        self.db_path = db_path if db_path else config.get_database_path()
        self.conn = sqlite3.connect(self.db_path)
        
    def load_data(self) -> Dict:
        """Load all necessary data files."""
        print("Loading data files...")
        
        data = {
            'summation': pd.read_csv(self.config.get_processed_file("summation_test_results.csv", 3)),
            'high_match': pd.read_csv(self.config.get_processed_file("high_match_khipus.csv", 4)),
            'similarity': pd.read_csv(self.config.get_processed_file("most_similar_khipu_pairs.csv", 4)),
            'features': pd.read_csv(self.config.get_processed_file("graph_structural_features.csv", 4))
        }
        
        # Load graphs
        graphs_path = self.config.root_dir / "data" / "graphs" / "khipu_graphs.pkl"
        with open(graphs_path, "rb") as f:
            graphs_list = pickle.load(f)
            data['graphs'] = {g.graph['khipu_id']: g for g in graphs_list}
        
        # Load provenance
        query = "SELECT KHIPU_ID, PROVENANCE, REGION FROM khipu_main"
        data['provenance'] = pd.read_sql_query(query, self.conn)
        
        print(f"✓ Loaded {len(data['summation'])} summation results")
        print(f"✓ Loaded {len(data['similarity'])} similarity pairs")
        print(f"✓ Loaded {len(data['graphs'])} graphs")
        
        return data
    
    def identify_perfect_summation_khipus(self, data: Dict) -> List[int]:
        """Identify khipus with perfect summation (match_rate = 1.0)."""
        print("\n" + "="*80)
        print("PERFECT SUMMATION KHIPUS")
        print("="*80)
        
        perfect = data['summation'][
            (data['summation']['has_pendant_summation']) &
            (data['summation']['pendant_match_rate'] == 1.0)
        ]
        
        khipu_ids = perfect['khipu_id'].tolist()
        
        print(f"\nIdentified {len(khipu_ids)} khipus with perfect summation:")
        
        for _, row in perfect.iterrows():
            prov = data['provenance'][
                data['provenance']['KHIPU_ID'] == row['khipu_id']
            ]['PROVENANCE'].values
            prov_str = prov[0] if len(prov) > 0 else 'Unknown'
            
            print(f"  Khipu {row['khipu_id']}: "
                  f"{row['num_pendant_groups']} groups, "
                  f"provenance: {prov_str}")
        
        return khipu_ids
    
    def identify_perfect_structural_matches(self, data: Dict) -> List[Tuple[int, int]]:
        """Identify khipu pairs with perfect structural similarity."""
        print("\n" + "="*80)
        print("PERFECT STRUCTURAL MATCH PAIRS")
        print("="*80)
        
        perfect = data['similarity'][data['similarity']['similarity'] == 1.0]
        
        pairs = [(row['khipu_id_1'], row['khipu_id_2']) 
                 for _, row in perfect.iterrows()]
        
        print(f"\nIdentified {len(pairs)} khipu pairs with perfect structural match")
        print(f"Total unique khipus involved: {len(set([k for p in pairs for k in p]))}")
        
        # Group by first khipu
        groups = defaultdict(list)
        for k1, k2 in pairs:
            groups[k1].append(k2)
        
        print("\nPerfect match groups:")
        for k1, matches in sorted(groups.items()):
            if len(matches) > 1:
                print(f"  Khipu {k1} matches: {matches}")
        
        return pairs
    
    def extract_graph_template(self, graph: nx.DiGraph) -> Dict:
        """Extract structural template from a graph."""
        template = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'depth': nx.dag_longest_path_length(graph) if nx.is_directed_acyclic_graph(graph) else -1,
        }
        
        # Node degree distribution
        in_degrees = [d for _, d in graph.in_degree()]
        out_degrees = [d for _, d in graph.out_degree()]
        
        template['in_degree_dist'] = {
            'mean': float(sum(in_degrees) / len(in_degrees)) if in_degrees else 0,
            'max': max(in_degrees) if in_degrees else 0,
            'histogram': self._create_histogram(in_degrees, max_val=20)
        }
        
        template['out_degree_dist'] = {
            'mean': float(sum(out_degrees) / len(out_degrees)) if out_degrees else 0,
            'max': max(out_degrees) if out_degrees else 0,
            'histogram': self._create_histogram(out_degrees, max_val=20)
        }
        
        # Level structure
        if nx.is_directed_acyclic_graph(graph):
            template['level_structure'] = self._extract_level_structure(graph)
        
        # Numeric properties
        numeric_nodes = [
            node for node, data in graph.nodes(data=True)
            if data.get('numeric_value') is not None
        ]
        template['pct_numeric'] = len(numeric_nodes) / template['num_nodes'] if template['num_nodes'] > 0 else 0
        
        return template
    
    def _create_histogram(self, values: List[int], max_val: int = 20) -> Dict[int, int]:
        """Create histogram of values."""
        hist = defaultdict(int)
        for v in values:
            key = min(v, max_val)
            hist[key] += 1
        return dict(hist)
    
    def _extract_level_structure(self, graph: nx.DiGraph) -> List[int]:
        """Extract number of nodes at each level."""
        # Find root nodes (in_degree = 0)
        roots = [n for n, d in graph.in_degree() if d == 0]
        
        if not roots:
            return []
        
        # BFS to assign levels
        levels = defaultdict(int)
        queue = [(root, 0) for root in roots]
        visited = set()
        
        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            levels[level] += 1
            
            for successor in graph.successors(node):
                if successor not in visited:
                    queue.append((successor, level + 1))
        
        # Convert to list
        max_level = max(levels.keys()) if levels else 0
        return [levels[i] for i in range(max_level + 1)]
    
    def analyze_perfect_summation_templates(self, data: Dict, khipu_ids: List[int]) -> Dict:
        """Analyze templates from perfect summation khipus."""
        print("\n" + "="*80)
        print("PERFECT SUMMATION TEMPLATE ANALYSIS")
        print("="*80)
        
        templates = {}
        
        for khipu_id in khipu_ids:
            if khipu_id not in data['graphs']:
                print(f"⚠ Warning: Graph not found for khipu {khipu_id}")
                continue
            
            graph = data['graphs'][khipu_id]
            template = self.extract_graph_template(graph)
            
            # Get provenance
            prov = data['provenance'][
                data['provenance']['KHIPU_ID'] == khipu_id
            ]['PROVENANCE'].values
            template['provenance'] = prov[0] if len(prov) > 0 else 'Unknown'
            
            templates[khipu_id] = template
            
            print(f"\nKhipu {khipu_id} ({template['provenance']}):")
            print(f"  Nodes: {template['num_nodes']}, Depth: {template['depth']}")
            print(f"  Numeric coverage: {template['pct_numeric']:.1%}")
            print(f"  Avg branching: {template['out_degree_dist']['mean']:.2f}")
            if 'level_structure' in template:
                print(f"  Level structure: {template['level_structure']}")
        
        # Find common patterns
        print("\n" + "-"*80)
        print("COMMON TEMPLATE PATTERNS")
        print("-"*80)
        
        depths = [t['depth'] for t in templates.values() if t['depth'] > 0]
        branchings = [t['out_degree_dist']['mean'] for t in templates.values()]
        numeric_pcts = [t['pct_numeric'] for t in templates.values()]
        
        print(f"\nDepth range: {min(depths)} to {max(depths)}") if depths else None
        print(f"Branching range: {min(branchings):.1f} to {max(branchings):.1f}")
        print(f"Numeric coverage: {min(numeric_pcts):.1%} to {max(numeric_pcts):.1%}")
        
        return templates
    
    def analyze_structural_match_templates(self, data: Dict, pairs: List[Tuple[int, int]]) -> Dict:
        """Analyze templates from structurally identical khipus."""
        print("\n" + "="*80)
        print("STRUCTURAL MATCH TEMPLATE ANALYSIS")
        print("="*80)
        
        # Get all unique khipus in perfect matches
        unique_khipus = list(set([k for p in pairs for k in p]))
        
        print(f"\nAnalyzing {len(unique_khipus)} unique khipus in perfect match pairs...")
        
        templates = {}
        
        for khipu_id in unique_khipus[:20]:  # Analyze first 20 for brevity
            if khipu_id not in data['graphs']:
                continue
            
            graph = data['graphs'][khipu_id]
            template = self.extract_graph_template(graph)
            
            # Get provenance
            prov = data['provenance'][
                data['provenance']['KHIPU_ID'] == khipu_id
            ]['PROVENANCE'].values
            template['provenance'] = prov[0] if len(prov) > 0 else 'Unknown'
            
            # Find all matches
            matches = [p[1] for p in pairs if p[0] == khipu_id]
            matches.extend([p[0] for p in pairs if p[1] == khipu_id])
            template['num_matches'] = len(set(matches))
            
            templates[khipu_id] = template
        
        # Group by template characteristics
        size_groups = defaultdict(list)
        for khipu_id, template in templates.items():
            size = template['num_nodes']
            if size <= 10:
                size_groups['small (≤10)'].append(khipu_id)
            elif size <= 50:
                size_groups['medium (11-50)'].append(khipu_id)
            elif size <= 100:
                size_groups['large (51-100)'].append(khipu_id)
            else:
                size_groups['very large (>100)'].append(khipu_id)
        
        print("\nPerfect match templates by size:")
        for size_cat, khipus in sorted(size_groups.items()):
            print(f"  {size_cat}: {len(khipus)} khipus")
        
        return templates
    
    def compare_templates(self, templates: Dict) -> Dict:
        """Compare templates to find commonalities."""
        print("\n" + "="*80)
        print("TEMPLATE COMPARISON")
        print("="*80)
        
        if len(templates) < 2:
            print("Not enough templates to compare")
            return {}
        
        # Compare structural features
        sizes = [t['num_nodes'] for t in templates.values()]
        depths = [t['depth'] for t in templates.values() if t['depth'] > 0]
        branchings = [t['out_degree_dist']['mean'] for t in templates.values()]
        
        comparison = {
            'num_templates': len(templates),
            'size_stats': {
                'min': min(sizes),
                'max': max(sizes),
                'mean': sum(sizes) / len(sizes),
                'range': max(sizes) - min(sizes)
            },
            'depth_stats': {
                'min': min(depths) if depths else 0,
                'max': max(depths) if depths else 0,
                'mean': sum(depths) / len(depths) if depths else 0
            } if depths else {},
            'branching_stats': {
                'min': min(branchings),
                'max': max(branchings),
                'mean': sum(branchings) / len(branchings)
            }
        }
        
        print(f"\nAnalyzed {comparison['num_templates']} templates")
        print(f"\nSize: {comparison['size_stats']['min']} to {comparison['size_stats']['max']} "
              f"(mean: {comparison['size_stats']['mean']:.1f})")
        
        if comparison['depth_stats']:
            print(f"Depth: {comparison['depth_stats']['min']} to {comparison['depth_stats']['max']} "
                  f"(mean: {comparison['depth_stats']['mean']:.2f})")
        
        print(f"Branching: {comparison['branching_stats']['min']:.2f} to "
              f"{comparison['branching_stats']['max']:.2f} "
              f"(mean: {comparison['branching_stats']['mean']:.2f})")
        
        return comparison
    
    def find_template_applications(self, data: Dict, template: Dict, 
                                  threshold: float = 0.9) -> List[int]:
        """Find khipus that match a template within threshold."""
        candidates = []
        
        template_size = template['num_nodes']
        template_depth = template['depth']
        template_branching = template['out_degree_dist']['mean']
        
        for khipu_id, features in data['features'].iterrows():
            if khipu_id == template.get('khipu_id'):
                continue
            
            # Check similarity
            size_match = abs(features['num_nodes'] - template_size) / template_size < (1 - threshold)
            depth_match = abs(features['depth'] - template_depth) / max(template_depth, 1) < (1 - threshold)
            branch_match = abs(features['avg_branching'] - template_branching) / max(template_branching, 1) < (1 - threshold)
            
            if size_match and depth_match and branch_match:
                candidates.append(khipu_id)
        
        return candidates
    
    def export_results(self, results: Dict, output_dir: str = None):
        """Export template analysis results."""
        if output_dir is None:
            output_dir = self.config.processed_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_json = Path(output_dir) / "template_analysis.json"
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'perfect_summation_khipus': results['perfect_summation_ids'],
            'perfect_summation_templates': results['summation_templates'],
            'perfect_structural_pairs': len(results['structural_pairs']),
            'structural_match_templates': results['structural_templates'],
            'template_comparison': results['comparison']
        }
        
        with open(output_json, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Exported analysis to {output_json}")
    
    def run_analysis(self):
        """Run complete template extraction analysis."""
        print("="*80)
        print("TEMPLATE EXTRACTION AND ANALYSIS")
        print("="*80)
        
        # Load data
        data = self.load_data()
        
        # Identify exemplars
        perfect_summation_ids = self.identify_perfect_summation_khipus(data)
        structural_pairs = self.identify_perfect_structural_matches(data)
        
        # Extract templates
        summation_templates = self.analyze_perfect_summation_templates(
            data, perfect_summation_ids
        )
        
        structural_templates = self.analyze_structural_match_templates(
            data, structural_pairs
        )
        
        # Compare templates
        all_templates = {**summation_templates, **structural_templates}
        comparison = self.compare_templates(all_templates)
        
        # Compile results
        results = {
            'perfect_summation_ids': perfect_summation_ids,
            'summation_templates': summation_templates,
            'structural_pairs': structural_pairs,
            'structural_templates': structural_templates,
            'comparison': comparison
        }
        
        # Export
        self.export_results(results)
        
        print("\n" + "="*80)
        print("TEMPLATE EXTRACTION COMPLETE")
        print("="*80)
        
        return results
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    extractor = TemplateExtractor()
    extractor.run_analysis()
