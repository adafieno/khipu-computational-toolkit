"""
Export Khipu Data to Blob-Compatible JSON Format

This script exports khipu data from the SQLite database into JSON files
suitable for cloud blob storage. Each khipu is exported as a separate JSON file
containing all necessary data for visualization (hierarchy, colors, knots).

Usage:
    python scripts/export_to_blob_format.py --output data/blob_export

Output:
    - khipu_index.json: Index of all khipus with metadata
    - khipus/AS001.json, khipus/AS002.json, etc.: Individual khipu data files
    - colors.json: Color code to RGB mappings
"""

import sys
import json
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402


def export_color_mappings(conn: sqlite3.Connection, output_dir: Path) -> Dict:
    """Export color code to RGB mappings."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT AS_COLOR_CD, COLOR_DESCR, R_DEC, G_DEC, B_DEC
        FROM ascher_color_dc
    """)
    
    colors = {}
    for row in cursor.fetchall():
        code, descr, r, g, b = row
        colors[code] = {
            'description': descr,
            'rgb': {
                'r': int(r * 255),
                'g': int(g * 255),
                'b': int(b * 255)
            },
            'hex': f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        }
    
    # Save colors.json
    with open(output_dir / 'colors.json', 'w', encoding='utf-8') as f:
        json.dump(colors, f, indent=2)
    
    return colors


def export_khipu_index(conn: sqlite3.Connection, output_dir: Path) -> List[Dict]:
    """Export index of all khipus with metadata."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT k.KHIPU_ID, k.PROVENANCE, k.MUSEUM_NO, k.CREATION_DATE,
               COUNT(DISTINCT c.CORD_ID) as cord_count,
               COUNT(DISTINCT kn.KNOT_ID) as knot_count
        FROM khipu_main k
        LEFT JOIN cord c ON k.KHIPU_ID = c.KHIPU_ID
        LEFT JOIN knot kn ON c.CORD_ID = kn.CORD_ID
        GROUP BY k.KHIPU_ID, k.PROVENANCE, k.MUSEUM_NO, k.CREATION_DATE
        HAVING cord_count > 0
        ORDER BY k.KHIPU_ID
    """)
    
    khipus = []
    for row in cursor.fetchall():
        khipu_id, provenance, museum_no, creation_date, cord_count, knot_count = row
        khipus.append({
            'id': khipu_id,
            'provenance': provenance if provenance else 'Unknown',
            'museum_no': museum_no if museum_no else '',
            'creation_date': creation_date if creation_date else '',
            'cord_count': cord_count,
            'knot_count': knot_count
        })
    
    # Save khipu_index.json
    with open(output_dir / 'khipu_index.json', 'w', encoding='utf-8') as f:
        json.dump(khipus, f, indent=2)
    
    return khipus


def export_khipu_data(conn: sqlite3.Connection, khipu_id: str, output_dir: Path):
    """Export detailed data for a single khipu."""
    # Load cord hierarchy
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.CORD_ID, c.ASCHER_CORD_POS, c.ASCHER_CLUST_POS,
               COALESCE(cc.COLOR_CD_1, 'Unknown') as COLOR_CD_1,
               COALESCE(cc.FULL_COLOR, 'Unknown') as FULL_COLOR,
               0 as CORD_LEVEL,
               NULL as PENDANT_FROM,
               c.LENGTH_CM as CORD_LENGTH
        FROM cord c
        LEFT JOIN ascher_cord_color cc ON c.CORD_ID = cc.CORD_ID
        WHERE c.KHIPU_ID = ?
        ORDER BY c.ASCHER_CORD_POS
    """, (int(khipu_id),))
    
    cords = []
    for row in cursor.fetchall():
        cord_id, pos, clust_pos, color_cd, full_color, level, pendant_from, length = row
        cords.append({
            'cord_id': cord_id,
            'position': pos,
            'cluster_position': clust_pos,
            'color_code': color_cd,
            'full_color': full_color,
            'level': level,
            'pendant_from': pendant_from,
            'length_cm': length if length else 30.0
        })
    
    # Load knots
    cursor.execute("""
        SELECT k.CORD_ID, k.KNOT_ID, k.KNOT_ORDINAL, k.TYPE_CODE, k.NUM_TURNS
        FROM knot k
        JOIN cord c ON k.CORD_ID = c.CORD_ID
        WHERE c.KHIPU_ID = ?
        ORDER BY k.CORD_ID, k.KNOT_ORDINAL
    """, (int(khipu_id),))
    
    knots = []
    for row in cursor.fetchall():
        cord_id, knot_id, ordinal, type_code, num_turns = row
        knots.append({
            'cord_id': cord_id,
            'knot_id': knot_id,
            'ordinal': ordinal,
            'type': type_code,
            'turns': int(num_turns) if num_turns else 0
        })
    
    # Get khipu metadata
    cursor.execute("""
        SELECT KHIPU_ID, PROVENANCE, MUSEUM_NO, CREATION_DATE
        FROM khipu_main
        WHERE KHIPU_ID = ?
    """, (khipu_id,))
    
    row = cursor.fetchone()
    khipu_data = {
        'id': khipu_id,
        'provenance': row[1] if row[1] else 'Unknown',
        'museum_no': row[2] if row[2] else '',
        'creation_date': row[3] if row[3] else '',
        'cords': cords,
        'knots': knots,
        'statistics': {
            'total_cords': len(cords),
            'total_knots': len(knots),
            'pendant_count': len([c for c in cords if c['level'] == 1]),
            'subsidiary_count': len([c for c in cords if c['level'] > 1])
        }
    }
    
    # Save individual khipu file
    khipus_dir = output_dir / 'khipus'
    khipus_dir.mkdir(exist_ok=True)
    
    with open(khipus_dir / f'{khipu_id}.json', 'w', encoding='utf-8') as f:
        json.dump(khipu_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Export khipu data to blob-compatible JSON format')
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
    
    # Connect to database
    db_path = config.get_database_path()
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        print("Please ensure the Open Khipu Repository database is available.")
        return 1
    
    conn = sqlite3.connect(db_path)
    
    # Export color mappings
    print("\n🎨 Exporting color mappings...")
    colors = export_color_mappings(conn, output_dir)
    print(f"✅ Exported {len(colors)} color codes")
    
    # Export khipu index
    print("\n📊 Exporting khipu index...")
    khipus = export_khipu_index(conn, output_dir)
    print(f"✅ Found {len(khipus)} khipus with data")
    
    # Export individual khipu data
    print("\n📦 Exporting individual khipu data...")
    limit = args.limit if args.limit else len(khipus)
    
    for i, khipu in enumerate(khipus[:limit], 1):
        khipu_id = khipu['id']
        print(f"  [{i}/{limit}] Exporting {khipu_id}...", end='\r')
        export_khipu_data(conn, khipu_id, output_dir)
    
    print(f"\n✅ Exported {limit} khipus")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✅ Export completed successfully!")
    print("="*60)
    print(f"\n📁 Output files:")
    print(f"  - {output_dir}/colors.json")
    print(f"  - {output_dir}/khipu_index.json")
    print(f"  - {output_dir}/khipus/*.json ({limit} files)")
    print(f"\n💡 These files can now be uploaded to blob storage.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
