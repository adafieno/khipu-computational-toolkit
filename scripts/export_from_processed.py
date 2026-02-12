"""
Export Khipu Data to Blob-Compatible JSON Format (from processed CSV files)

This script exports khipu data from processed CSV files into JSON files
suitable for cloud blob storage. This version works without database access.

Usage:
    python scripts/export_from_processed.py --output data/blob_export

Output:
    - khipu_index.json: Index of all khipus with metadata
    - khipus/AS001.json, khipus/AS002.json, etc.: Individual khipu data files
    - colors.json: Color code to RGB mappings (predefined)
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402


# Predefined color mappings (common Ascher colors)
PREDEFINED_COLORS = {
    'AB': {'description': 'Aberdeen Brown', 'rgb': {'r': 139, 'g': 69, 'b': 19}, 'hex': '#8b4513'},
    'BG': {'description': 'Beige', 'rgb': {'r': 245, 'g': 245, 'b': 220}, 'hex': '#f5f5dc'},
    'BL': {'description': 'Blue', 'rgb': {'r': 0, 'g': 0, 'b': 255}, 'hex': '#0000ff'},
    'BN': {'description': 'Brown', 'rgb': {'r': 165, 'g': 42, 'b': 42}, 'hex': '#a52a2a'},
    'GG': {'description': 'Gray-Green', 'rgb': {'r': 128, 'g': 138, 'b': 135}, 'hex': '#808a87'},
    'GN': {'description': 'Green', 'rgb': {'r': 0, 'g': 128, 'b': 0}, 'hex': '#008000'},
    'GY': {'description': 'Gray', 'rgb': {'r': 128, 'g': 128, 'b': 128}, 'hex': '#808080'},
    'MB': {'description': 'Mottled Brown', 'rgb': {'r': 139, 'g': 90, 'b': 43}, 'hex': '#8b5a2b'},
    'MG': {'description': 'Mottled Green', 'rgb': {'r': 107, 'g': 142, 'b': 35}, 'hex': '#6b8e23'},
    'OR': {'description': 'Orange', 'rgb': {'r': 255, 'g': 165, 'b': 0}, 'hex': '#ffa500'},
    'PU': {'description': 'Purple', 'rgb': {'r': 128, 'g': 0, 'b': 128}, 'hex': '#800080'},
    'RD': {'description': 'Red', 'rgb': {'r': 255, 'g': 0, 'b': 0}, 'hex': '#ff0000'},
    'W': {'description': 'White', 'rgb': {'r': 255, 'g': 255, 'b': 255}, 'hex': '#ffffff'},
    'YL': {'description': 'Yellow', 'rgb': {'r': 255, 'g': 255, 'b': 0}, 'hex': '#ffff00'},
    'Unknown': {'description': 'Unknown', 'rgb': {'r': 204, 'g': 204, 'b': 204}, 'hex': '#cccccc'},
}


def export_colors(output_dir: Path) -> Dict:
    """Export predefined color mappings."""
    with open(output_dir / 'colors.json', 'w', encoding='utf-8') as f:
        json.dump(PREDEFINED_COLORS, f, indent=2)
    
    return PREDEFINED_COLORS


def load_processed_data(config):
    """Load all processed CSV files."""
    print("📂 Loading processed data files...")
    
    # Load hierarchy data
    hierarchy_path = config.get_processed_file("cord_hierarchy.csv", 2)
    if not hierarchy_path.exists():
        print(f"❌ Hierarchy file not found: {hierarchy_path}")
        return None, None, None, None
    
    hierarchy = pd.read_csv(hierarchy_path)
    print(f"  ✓ Loaded {len(hierarchy)} cord hierarchy records")
    
    # Load color data
    color_path = config.get_processed_file("color_data.csv", 2)
    if not color_path.exists():
        print(f"⚠️  Color file not found: {color_path}, continuing without colors")
        colors = pd.DataFrame()
    else:
        colors = pd.read_csv(color_path)
        print(f"  ✓ Loaded {len(colors)} color records")
    
    # Load knot data
    knot_path = config.get_processed_file("knot_data.csv", 2)
    if not knot_path.exists():
        print(f"⚠️  Knot file not found: {knot_path}, continuing without knots")
        knots = pd.DataFrame()
    else:
        knots = pd.read_csv(knot_path)
        print(f"  ✓ Loaded {len(knots)} knot records")
    
    # Load khipu metadata (provenance)
    metadata_path = config.get_processed_file("anomaly_detection_results.csv", 4)
    if not metadata_path.exists():
        print(f"⚠️  Metadata file not found: {metadata_path}, continuing without provenance data")
        metadata = pd.DataFrame()
    else:
        metadata = pd.read_csv(metadata_path)[['khipu_id', 'PROVENANCE']]
        metadata.rename(columns={'khipu_id': 'KHIPU_ID'}, inplace=True)
        print(f"  ✓ Loaded metadata for {len(metadata)} khipus")
    
    return hierarchy, colors, knots, metadata


def build_khipu_index(hierarchy: pd.DataFrame, metadata: pd.DataFrame) -> List[Dict]:
    """Build index of all khipus from hierarchy data."""
    print("\n📊 Building khipu index...")
    
    # Group by KHIPU_ID and get statistics
    khipu_stats = hierarchy.groupby('KHIPU_ID').agg({
        'CORD_ID': 'count'
    }).reset_index()
    
    khipu_stats.columns = ['KHIPU_ID', 'cord_count']
    
    # Merge with metadata if available
    if not metadata.empty:
        khipu_stats = khipu_stats.merge(metadata, on='KHIPU_ID', how='left')
    else:
        khipu_stats['PROVENANCE'] = 'Unknown'
    
    khipus = []
    for _, row in khipu_stats.iterrows():
        # Generate khipu ID string (format: AS### or similar)
        khipu_num = int(row['KHIPU_ID'])
        khipu_id_str = f"AS{khipu_num:03d}" if khipu_num < 1000 else f"K{khipu_num}"
        
        khipus.append({
            'id': khipu_id_str,
            'numeric_id': khipu_num,
            'provenance': row.get('PROVENANCE', 'Unknown') if pd.notna(row.get('PROVENANCE')) else 'Unknown',
            'museum_no': '',
            'creation_date': '',
            'cord_count': int(row['cord_count']),
            'knot_count': 0  # Will be updated if knot data is available
        })
    
    return khipus


def update_knot_counts(khipus: List[Dict], knots: pd.DataFrame):
    """Update knot counts in khipu index."""
    if knots.empty:
        return
    
    # Get knot counts per numeric khipu ID
    knot_counts = knots.groupby('KHIPU_ID')['KNOT_ID'].count().to_dict()
    
    for khipu in khipus:
        khipu['knot_count'] = knot_counts.get(khipu['numeric_id'], 0)


def export_khipu_data(khipu_id: str, numeric_id: int, hierarchy: pd.DataFrame, 
                     colors: pd.DataFrame, knots: pd.DataFrame, metadata: pd.DataFrame,
                     output_dir: Path):
    """Export detailed data for a single khipu."""
    
    # Get cords for this khipu using numeric ID
    khipu_cords = hierarchy[hierarchy['KHIPU_ID'] == numeric_id].copy()
    
    if khipu_cords.empty:
        return
    
    # Merge with color data if available
    if not colors.empty:
        khipu_cords = khipu_cords.merge(
            colors[['cord_id', 'color_cd_1', 'full_color']], 
            left_on='CORD_ID',
            right_on='cord_id',
            how='left'
        )
        # Rename for consistency
        khipu_cords.rename(columns={'color_cd_1': 'COLOR_CD_1', 'full_color': 'FULL_COLOR'}, inplace=True)
    else:
        khipu_cords['COLOR_CD_1'] = 'Unknown'
        khipu_cords['FULL_COLOR'] = 'Unknown'
    
    # Build cords list
    cords = []
    for idx, row in khipu_cords.iterrows():
        cords.append({
            'cord_id': int(row['CORD_ID']),
            'position': int(row.get('CORD_ORDINAL', idx)) if pd.notna(row.get('CORD_ORDINAL')) else idx,
            'cluster_position': 0,
            'color_code': row.get('COLOR_CD_1', 'Unknown') if pd.notna(row.get('COLOR_CD_1')) else 'Unknown',
            'full_color': row.get('FULL_COLOR', 'Unknown') if pd.notna(row.get('FULL_COLOR')) else 'Unknown',
            'level': int(row['CORD_LEVEL']) if pd.notna(row.get('CORD_LEVEL')) else 1,
            'pendant_from': int(row['PENDANT_FROM']) if pd.notna(row.get('PENDANT_FROM')) else None,
            'length_cm': float(row['CORD_LENGTH']) if pd.notna(row.get('CORD_LENGTH')) else 30.0
        })
    
    # Get knots for this khipu
    knot_list = []
    if not knots.empty:
        khipu_knots = knots[knots['KHIPU_ID'] == numeric_id]
        
        for _, row in khipu_knots.iterrows():
            knot_list.append({
                'cord_id': int(row['CORD_ID']),
                'knot_id': int(row['KNOT_ID']),
                'ordinal': int(row['KNOT_ORDINAL']) if pd.notna(row.get('KNOT_ORDINAL')) else 0,
                'type': row.get('TYPE_CODE', 'S') if pd.notna(row.get('TYPE_CODE')) else 'S',
                'turns': int(row['NUM_TURNS']) if pd.notna(row.get('NUM_TURNS')) else 0
            })
    
    # Calculate statistics
    level_1_count = len([c for c in cords if c['level'] == 1])
    subsidiary_count = len([c for c in cords if c['level'] > 1])
    
    # Get provenance
    provenance = 'Unknown'
    if not metadata.empty:
        prov_row = metadata[metadata['KHIPU_ID'] == numeric_id]
        if not prov_row.empty:
            provenance = prov_row['PROVENANCE'].iloc[0]
            if pd.isna(provenance):
                provenance = 'Unknown'
    
    # Build khipu data
    khipu_data = {
        'id': khipu_id,
        'provenance': provenance,
        'museum_no': '',
        'creation_date': '',
        'cords': cords,
        'knots': knot_list,
        'statistics': {
            'total_cords': len(cords),
            'total_knots': len(knot_list),
            'pendant_count': level_1_count,
            'subsidiary_count': subsidiary_count
        }
    }
    
    # Save individual khipu file
    khipus_dir = output_dir / 'khipus'
    khipus_dir.mkdir(exist_ok=True)
    
    with open(khipus_dir / f'{khipu_id}.json', 'w', encoding='utf-8') as f:
        json.dump(khipu_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Export khipu data from processed CSV files to blob-compatible JSON format')
    parser.add_argument('--output', default='data/blob_export',
                        help='Output directory for JSON files (default: data/blob_export)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of khipus to export (for testing)')
    args = parser.parse_args()
    
    config = get_config()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Exporting khipu data to blob-compatible JSON format...")
    print(f"📁 Output directory: {output_dir}")
    
    # Load processed data
    hierarchy, colors, knots, metadata = load_processed_data(config)
    if hierarchy is None:
        print("\n❌ Failed to load required data files")
        return 1
    
    # Export color mappings
    print("\n🎨 Exporting color mappings...")
    color_map = export_colors(output_dir)
    print(f"✅ Exported {len(color_map)} color codes")
    
    # Build khipu index
    khipus = build_khipu_index(hierarchy, metadata)
    
    # Update knot counts
    if not knots.empty:
        update_knot_counts(khipus, knots)
    
    print(f"✅ Found {len(khipus)} khipus")
    
    # Save khipu index
    with open(output_dir / 'khipu_index.json', 'w', encoding='utf-8') as f:
        json.dump(khipus, f, indent=2)
    
    # Export individual khipu data
    print("\n📦 Exporting individual khipu data...")
    limit = args.limit if args.limit else len(khipus)
    
    for i, khipu in enumerate(khipus[:limit], 1):
        khipu_id = khipu['id']
        numeric_id = khipu['numeric_id']
        print(f"  [{i}/{limit}] Exporting {khipu_id}...", end='\r')
        export_khipu_data(khipu_id, numeric_id, hierarchy, colors, knots, metadata, output_dir)
    
    print(f"\n✅ Exported {limit} khipus")
    
    # Calculate overall statistics
    total_cords = sum(k['cord_count'] for k in khipus[:limit])
    total_knots = sum(k['knot_count'] for k in khipus[:limit])
    
    print("\n" + "="*60)
    print("✅ Export completed successfully!")
    print("="*60)
    print(f"\n📊 Statistics:")
    print(f"  - Total khipus: {limit}")
    print(f"  - Total cords: {total_cords:,}")
    print(f"  - Total knots: {total_knots:,}")
    print(f"  - Average cords per khipu: {total_cords/limit:.1f}")
    print(f"\n📁 Output files:")
    print(f"  - {output_dir}/colors.json")
    print(f"  - {output_dir}/khipu_index.json")
    print(f"  - {output_dir}/khipus/*.json ({limit} files)")
    print(f"\n💡 These files can now be uploaded to blob storage.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
