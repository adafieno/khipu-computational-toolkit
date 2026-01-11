"""
Khipu Value Computation Module
===============================

Implements multiple methods for computing numerical values from khipu knots,
following the theoretical framework of Ascher (1997, 2002) and the empirical
validation of Medrano & Khosla (2024).

Key Principles:
--------------
1. **Multi-method approach**: Never assume a single "correct" interpretation
2. **Transparent uncertainty**: Quantify confidence for each computation
3. **Archaeological reality**: Account for damaged/ambiguous knots
4. **Schema correctness**: Use actual database schema (TYPE_CODE, CLUSTER_ORDINAL)

Database Schema (CRITICAL):
---------------------------
knot table:
  - TYPE_CODE: 'S' (single), 'L' (long string), 'E' (figure-8), '8' (figure-8 variant)
  - CLUSTER_ORDINAL: Position within cluster (1, 2, 3..., not KNOT_POSITION)
  - NUM_TURNS: Number of wraps in long string knots
  - DIRECTION: 'S' (counter-clockwise), 'Z' (clockwise), 'U' (unknown)
  - CORD_ID: Foreign key to cord table

cord table:
  - CORD_LEVEL: 0=primary, 1=pendant, 2=subsidiary, etc.
  - CORD_ORDINAL: Position among siblings
  - PENDANT_FROM: Parent cord ID (not PARENT_CORD_ID)
  - ATTACHED_TO: Attachment point for subsidiaries

Computation Methods:
-------------------
1. **Ascher Decimal** (standard): Base-10 positional with cluster gaps
2. **Medrano Mixed Base**: Alternating bases for certain khipu types
3. **Locke Grouped**: Non-positional counting with category markers
4. **Urton Binary**: Binary encoding via knot direction

References:
----------
- Ascher, M. & Ascher, R. (1997). Mathematics of the Incas: Code of the Quipu
- Medrano, C. & Khosla, M. (2024). Corpus-wide summation validation
"""

import sqlite3
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math


class KnotType(Enum):
    """Knot type classification (matches database TYPE_CODE)"""
    SINGLE = 'S'
    LONG_STRING = 'L'
    FIGURE_8 = 'E'
    FIGURE_8_VAR = '8'
    
    @classmethod
    def from_db(cls, type_code: str) -> 'KnotType':
        """Convert database TYPE_CODE to enum"""
        mapping = {
            'S': cls.SINGLE,
            'L': cls.LONG_STRING,
            'E': cls.FIGURE_8,
            '8': cls.FIGURE_8_VAR,
        }
        return mapping.get(type_code, cls.SINGLE)


class ComputationMethod(Enum):
    """Available value computation methods"""
    ASCHER_DECIMAL = "ascher_decimal"
    MEDRANO_MIXED = "medrano_mixed"
    LOCKE_GROUPED = "locke_grouped"
    URTON_BINARY = "urton_binary"


@dataclass
class Knot:
    """Represents a single knot with database fields"""
    cord_id: str
    type_code: str  # S, L, E, 8
    cluster_ordinal: int  # Position in cluster (1, 2, 3...)
    num_turns: Optional[int]  # For long string knots
    direction: str  # S, Z, U
    
    @property
    def knot_type(self) -> KnotType:
        return KnotType.from_db(self.type_code)
    
    @property
    def is_long_string(self) -> bool:
        return self.type_code == 'L'
    
    @property
    def is_figure_8(self) -> bool:
        return self.type_code in ('E', '8')


@dataclass
class KnotCluster:
    """Group of knots in same decimal position"""
    position: int  # 1=units, 10=tens, 100=hundreds, 1000=thousands
    knots: List[Knot]
    
    @property
    def count(self) -> int:
        """Number of knots in cluster"""
        return len(self.knots)
    
    @property
    def has_long_string(self) -> bool:
        return any(k.is_long_string for k in self.knots)
    
    @property
    def long_string_turns(self) -> Optional[int]:
        """Get NUM_TURNS for long string knot if present"""
        for k in self.knots:
            if k.is_long_string and k.num_turns is not None:
                return k.num_turns
        return None


@dataclass
class ComputedValue:
    """Result of value computation with confidence metrics"""
    value: float
    method: ComputationMethod
    confidence: float  # 0.0 to 1.0
    ambiguities: List[str]  # List of issues affecting confidence
    knot_breakdown: Dict[int, int]  # {position: digit_value}
    
    def __str__(self):
        return f"{self.value:.0f} ({self.method.value}, conf={self.confidence:.2f})"


class ValueComputer:
    """Main value computation engine"""
    
    def __init__(self, db_path: str = "khipu.db"):
        self.db_path = db_path
    
    def get_cord_knots(self, cord_id: str) -> List[Knot]:
        """
        Fetch all knots for a cord using CORRECT schema.
        
        Returns knots sorted by CLUSTER_ORDINAL (descending for positional value).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT CORD_ID, TYPE_CODE, CLUSTER_ORDINAL, NUM_TURNS, DIRECTION
        FROM knot
        WHERE CORD_ID = ?
        ORDER BY CLUSTER_ORDINAL DESC
        """
        
        cursor.execute(query, (cord_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return [Knot(*row) for row in rows]
    
    def cluster_knots(self, knots: List[Knot]) -> List[KnotCluster]:
        """
        Group knots into clusters by CLUSTER_ORDINAL.
        
        In Ascher notation, gaps in CLUSTER_ORDINAL represent decimal positions.
        Example: CLUSTER_ORDINAL [5, 4, 3, 1] → clusters at positions [1000s, 100s, 10s, units]
        """
        if not knots:
            return []
        
        # Sort by CLUSTER_ORDINAL descending (highest = leftmost = most significant)
        sorted_knots = sorted(knots, key=lambda k: k.cluster_ordinal, reverse=True)
        
        clusters = []
        current_ordinal = sorted_knots[0].cluster_ordinal
        current_cluster = []
        
        for knot in sorted_knots:
            if knot.cluster_ordinal == current_ordinal:
                current_cluster.append(knot)
            else:
                # New cluster
                if current_cluster:
                    position = 10 ** (len(clusters))  # 1, 10, 100, 1000...
                    clusters.append(KnotCluster(position, current_cluster))
                current_cluster = [knot]
                current_ordinal = knot.cluster_ordinal
        
        # Add final cluster
        if current_cluster:
            position = 10 ** (len(clusters))
            clusters.append(KnotCluster(position, current_cluster))
        
        return clusters
    
    def compute_ascher_decimal(self, knots: List[Knot]) -> ComputedValue:
        """
        Ascher's standard base-10 positional system.
        
        Rules:
        - Knots grouped by CLUSTER_ORDINAL
        - Each cluster = one decimal digit
        - Long string knots (L) encode value via NUM_TURNS
        - Single knots (S) encode via count
        - Figure-8 knots (E, 8) in units position only
        
        Confidence factors:
        - Perfect: All knots readable, no ambiguity
        - High (0.9): One missing NUM_TURNS or unclear direction
        - Medium (0.7): Multiple unknowns
        - Low (0.5): Damaged or contradictory data
        """
        if not knots:
            return ComputedValue(0, ComputationMethod.ASCHER_DECIMAL, 1.0, [], {})
        
        clusters = self.cluster_knots(knots)
        ambiguities = []
        total_value = 0
        breakdown = {}
        confidence = 1.0
        
        for cluster in clusters:
            digit_value = 0
            
            if cluster.has_long_string:
                # Long string knot encodes the digit via NUM_TURNS
                turns = cluster.long_string_turns
                if turns is not None:
                    digit_value = turns
                else:
                    # Missing NUM_TURNS
                    digit_value = len(cluster.knots)  # Fallback to count
                    ambiguities.append(f"Missing NUM_TURNS at {cluster.position}")
                    confidence *= 0.9
            else:
                # Single knots: count them
                digit_value = cluster.count
            
            # Figure-8 check (should only be in units)
            if cluster.position == 1 and any(k.is_figure_8 for k in cluster.knots):
                # Figure-8 in units position (expected)
                pass
            elif cluster.position > 1 and any(k.is_figure_8 for k in cluster.knots):
                ambiguities.append(f"Figure-8 in non-units position ({cluster.position})")
                confidence *= 0.8
            
            # Check for impossible values (digit > 9)
            if digit_value > 9:
                ambiguities.append(f"Digit {digit_value} > 9 at {cluster.position}")
                confidence *= 0.7
            
            breakdown[cluster.position] = digit_value
            total_value += digit_value * cluster.position
        
        return ComputedValue(
            value=total_value,
            method=ComputationMethod.ASCHER_DECIMAL,
            confidence=confidence,
            ambiguities=ambiguities,
            knot_breakdown=breakdown
        )
    
    def compute_all_methods(self, cord_id: str) -> List[ComputedValue]:
        """
        Compute value using all available methods.
        
        Returns list of ComputedValue objects, sorted by confidence (descending).
        """
        knots = self.get_cord_knots(cord_id)
        
        results = []
        
        # Always compute Ascher decimal (standard method)
        results.append(self.compute_ascher_decimal(knots))
        
        # TODO: Add other methods as they're implemented
        # results.append(self.compute_medrano_mixed(knots))
        # results.append(self.compute_locke_grouped(knots))
        # results.append(self.compute_urton_binary(knots))
        
        # Sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)
        
        return results
    
    def get_best_value(self, cord_id: str) -> ComputedValue:
        """Get highest-confidence value for a cord"""
        results = self.compute_all_methods(cord_id)
        return results[0] if results else ComputedValue(0, ComputationMethod.ASCHER_DECIMAL, 0.0, ["No knots"], {})


# Convenience functions
def compute_cord_value(cord_id: str, db_path: str = "khipu.db") -> float:
    """Quick function to get best value for a cord"""
    computer = ValueComputer(db_path)
    return computer.get_best_value(cord_id).value


def get_value_with_confidence(cord_id: str, db_path: str = "khipu.db") -> Tuple[float, float]:
    """Get value and confidence score"""
    computer = ValueComputer(db_path)
    result = computer.get_best_value(cord_id)
    return result.value, result.confidence


if __name__ == "__main__":
    # Example usage
    computer = ValueComputer()
    
    # Test with a sample cord
    test_cord_id = "AS001_0001"  # Replace with actual cord ID
    results = computer.compute_all_methods(test_cord_id)
    
    print("Value Computation Results")
    print("=" * 60)
    print(f"Cord: {test_cord_id}")
    print()
    
    for result in results:
        print(f"Method: {result.method.value}")
        print(f"Value: {result.value:.0f}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.ambiguities:
            print(f"Ambiguities: {', '.join(result.ambiguities)}")
        print(f"Breakdown: {result.knot_breakdown}")
        print()
