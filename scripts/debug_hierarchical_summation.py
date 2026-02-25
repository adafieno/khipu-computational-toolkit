"""
Debug script to investigate why hierarchical pattern shows 0% detection.
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from config import get_config
from utils.arithmetic_validator import ArithmeticValidator
import sqlite3

def investigate_hierarchical_pattern():
    """Investigate why hierarchical detection returns 0%."""
    config = get_config()
    db_path = config.get_database_path()
    validator = ArithmeticValidator(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 80)
    print("HIERARCHICAL SUMMATION PATTERN INVESTIGATION")
    print("=" * 80)
    print()
    
    # 1. Check how many khipus have parent-child relationships
    print("1. Checking ATTACHED_TO relationships...")
    cursor.execute("""
        SELECT COUNT(DISTINCT KHIPU_ID) 
        FROM cord 
        WHERE ATTACHED_TO IS NOT NULL
    """)
    khipus_with_relationships = cursor.fetchone()[0]
    print(f"   Khipus with ATTACHED_TO relationships: {khipus_with_relationships}")
    
    # 2. Check a sample khipu with relationships
    print("\n2. Sample khipu analysis...")
    cursor.execute("""
        SELECT KHIPU_ID, COUNT(*) as rel_count
        FROM cord
        WHERE ATTACHED_TO IS NOT NULL
        GROUP BY KHIPU_ID
        ORDER BY rel_count DESC
        LIMIT 5
    """)
    
    sample_khipus = cursor.fetchall()
    print(f"   Top 5 khipus by relationship count:")
    for khipu_id, count in sample_khipus:
        print(f"     Khipu {khipu_id}: {count} child cords")
    
    # 3. Detailed analysis of first sample khipu
    if sample_khipus:
        sample_id = sample_khipus[0][0]
        print(f"\n3. Detailed analysis of Khipu {sample_id}:")
        
        # Check what ATTACHED_TO values look like
        cursor.execute("""
            SELECT CORD_ID, ATTACHED_TO, CORD_LEVEL
            FROM cord
            WHERE KHIPU_ID = ? AND ATTACHED_TO IS NOT NULL
            LIMIT 20
        """, (sample_id,))
        
        sample_cords = cursor.fetchall()
        print(f"   Sample ATTACHED_TO values:")
        for cord_id, attached_to, level in sample_cords[:10]:
            print(f"     Cord {cord_id} (level {level}) ATTACHED_TO={attached_to} (type: {type(attached_to).__name__})")
        
        # Try to find if ATTACHED_TO matches any CORD_ID
        print(f"\n   Checking if ATTACHED_TO values exist as CORD_IDs...")
        attached_to_val = sample_cords[0][1] if sample_cords else None
        if attached_to_val:
            cursor.execute("""
                SELECT CORD_ID, CORD_LEVEL
                FROM cord
                WHERE CORD_ID = ?
            """, (attached_to_val,))
            
            match = cursor.fetchone()
            if match:
                print(f"   ✓ ATTACHED_TO value {attached_to_val} FOUND as CORD_ID {match[0]} (level {match[1]})")
            else:
                print(f"   ✗ ATTACHED_TO value {attached_to_val} NOT FOUND in CORD_ID column")
                
                # Try alternate interpretation - maybe it's an ordinal or position?
                cursor.execute("""
                    SELECT CORD_ID, CORD_ORDINAL, CLUSTER_ORDINAL
                    FROM cord
                    WHERE KHIPU_ID = ? AND (CORD_ORDINAL = ? OR CLUSTER_ORDINAL = ?)
                """, (sample_id, attached_to_val, attached_to_val))
                
                alt_matches = cursor.fetchall()
                if alt_matches:
                    print(f"   → Found as ordinal: {alt_matches}")
                else:
                    print(f"   → Not found as ordinal either")
                    
                    # Check cord table schema
                    cursor.execute("PRAGMA table_info(cord)")
                    schema = cursor.fetchall()
                    print(f"\n   Cord table schema:")
                    for col in schema:
                        print(f"     {col[1]} ({col[2]})")
        
        # Get values for these cords using validator
        print(f"\n   Testing value extraction for sample cords...")
        cord_values = {}
        for cord_id, attached_to, level in sample_cords[:5]:
            child_val = validator.get_cord_numeric_value(cord_id)
            
            if child_val.total_value is not None:
                cord_values[cord_id] = child_val.total_value
                print(f"   Cord {cord_id} (level {level}): value={child_val.total_value}, ATTACHED_TO={attached_to}")
            else:
                print(f"   Cord {cord_id} (level {level}): NO VALUE, ATTACHED_TO={attached_to}")
    
    # 4. Check cord hierarchy levels
    print("\n4. Checking cord hierarchy structure...")
    cursor.execute("""
        SELECT CORD_LEVEL, COUNT(*) 
        FROM cord 
        WHERE KHIPU_ID = ?
        GROUP BY CORD_LEVEL
        ORDER BY CORD_LEVEL
    """, (sample_id,))
    
    levels = cursor.fetchall()
    print(f"   Cord levels in sample khipu {sample_id}:")
    for level, count in levels:
        print(f"     Level {level}: {count} cords")
    
    # 5. Check if ATTACHED_TO refers to subsidiaries or parent pendants
    print("\n5. Checking ATTACHED_TO semantics...")
    cursor.execute("""
        SELECT c1.CORD_LEVEL as child_level, c2.CORD_LEVEL as parent_level, COUNT(*)
        FROM cord c1
        JOIN cord c2 ON c1.ATTACHED_TO = c2.CORD_ID
        WHERE c1.KHIPU_ID = ?
        GROUP BY c1.CORD_LEVEL, c2.CORD_LEVEL
    """, (sample_id,))
    
    level_relationships = cursor.fetchall()
    print(f"   Relationship patterns:")
    for child_level, parent_level, count in level_relationships:
        print(f"     Level {child_level} → Level {parent_level}: {count} relationships")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    investigate_hierarchical_pattern()
