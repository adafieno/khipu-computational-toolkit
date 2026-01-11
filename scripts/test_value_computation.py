"""
Test value computation module with real khipu data
"""

import sys
sys.path.insert(0, 'c:/code/khipu-computational-toolkit/src')

from analysis.value_computation import ValueComputer, ComputationMethod
from config import Config
import sqlite3

# Get database path from config
config = Config()
db_path = config.get_database_path()

print(f"Using database: {db_path}")
print(f"Database exists: {db_path.exists()}")
print()

# Connect to database
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Get a sample of cords with knots
query = """
SELECT DISTINCT c.CORD_ID, c.KHIPU_ID, c.CORD_LEVEL, c.CORD_ORDINAL, COUNT(k.TYPE_CODE) as knot_count
FROM cord c
JOIN knot k ON c.CORD_ID = k.CORD_ID
WHERE c.CORD_LEVEL = 1
GROUP BY c.CORD_ID
HAVING knot_count > 0
LIMIT 10
"""

cursor.execute(query)
sample_cords = cursor.fetchall()

print("Testing Value Computation Module")
print("=" * 80)
print("\nSample cords with knots:")
print(f"{'Cord ID':<20} {'Khipu':<10} {'Level':<8} {'Ordinal':<10} {'Knot Count':<12}")
print("-" * 80)
for cord in sample_cords:
    print(f"{cord[0]:<20} {cord[1]:<10} {cord[2]:<8} {cord[3]:<10} {cord[4]:<12}")

print("\n" + "=" * 80)
print("COMPUTING VALUES")
print("=" * 80)

# Test value computation
computer = ValueComputer()

for cord_id, khipu_id, level, ordinal, knot_count in sample_cords[:5]:  # Test first 5
    print(f"\n{cord_id} (Khipu: {khipu_id}, Pendant {ordinal})")
    print("-" * 80)
    
    # Get knots
    knots = computer.get_cord_knots(cord_id)
    print(f"Knots found: {len(knots)}")
    
    for i, knot in enumerate(knots, 1):
        turns_str = f"({knot.num_turns} turns)" if knot.num_turns else ""
        print(f"  {i}. Type={knot.type_code}, Cluster={knot.cluster_ordinal}, Dir={knot.direction} {turns_str}")
    
    # Compute value
    results = computer.compute_all_methods(cord_id)
    
    print("\nComputed Values:")
    for result in results:
        print(f"  • {result.method.value}: {result.value:.0f}")
        print(f"    Confidence: {result.confidence:.2%}")
        print(f"    Breakdown: {result.knot_breakdown}")
        if result.ambiguities:
            print(f"    Issues: {', '.join(result.ambiguities)}")

conn.close()

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
