"""
KFG Data Parsers

Core parsing utilities for Khipu Field Guide (KFG) Excel format.
These parsers extract structured data from KFG's string-based formats.

Based on investigation findings documented in:
    docs/KFG_INVESTIGATION_FINDINGS.md
"""

import re
from typing import Dict, List, Optional, Any
import pandas as pd


def parse_cord_hierarchy(cord_name: str) -> Optional[Dict[str, Any]]:
    """
    Parse KFG cord naming convention to extract hierarchy.
    
    The KFG uses a systematic naming scheme where:
    - p1 = Pendant 1 (level 0, attached to primary cord)
    - p6s1 = Subsidiary 1 of pendant 6 (level 1)
    - p10s1s1 = Sub-subsidiary of pendant 10 (level 2)
    - p10s1s1s1 = Sub-sub-subsidiary (level 3, rare)
    
    Args:
        cord_name: Cord identifier (e.g., "p6s1", "p10s1s1")
        
    Returns:
        Dictionary with hierarchy information or None if invalid format:
        {
            'pendant_num': int - Base pendant number
            'subsidiaries': List[int] - Chain of subsidiary indices
            'level': int - Hierarchy depth (0=pendant, 1=subsidiary, etc.)
            'parent': Optional[str] - Parent cord name
            'full_hierarchy': List[str] - Complete path from root
        }
        
    Examples:
        >>> parse_cord_hierarchy("p1")
        {'pendant_num': 1, 'subsidiaries': [], 'level': 0, 'parent': None, 
         'full_hierarchy': ['p1']}
         
        >>> parse_cord_hierarchy("p6s1")
        {'pendant_num': 6, 'subsidiaries': [1], 'level': 1, 'parent': 'p6',
         'full_hierarchy': ['p6', 'p6s1']}
         
        >>> parse_cord_hierarchy("p10s1s1")
        {'pendant_num': 10, 'subsidiaries': [1, 1], 'level': 2, 'parent': 'p10s1',
         'full_hierarchy': ['p10', 'p10s1', 'p10s1s1']}
    """
    if pd.isna(cord_name):
        return None
    
    cord_name = str(cord_name).strip()
    
    # Pattern: p<number>[s<number>]*
    match = re.match(r'^p(\d+)((?:s\d+)*)$', cord_name)
    if not match:
        return None
    
    pendant_num = int(match.group(1))
    subsidiary_part = match.group(2)
    
    if not subsidiary_part:
        # Top-level pendant - attaches to primary cord
        return {
            'pendant_num': pendant_num,
            'subsidiaries': [],
            'level': 0,
            'parent': None,
            'full_hierarchy': [cord_name]
        }
    else:
        # Subsidiary cord - parse subsidiary chain
        # Extract all subsidiary numbers (s1, s2, etc.)
        subsidiary_matches = re.findall(r's(\d+)', subsidiary_part)
        subsidiaries = [int(s) for s in subsidiary_matches]
        
        # Build parent name (remove last subsidiary)
        if len(subsidiaries) == 1:
            parent = f"p{pendant_num}"
        else:
            parent_subs = subsidiaries[:-1]
            parent = f"p{pendant_num}" + ''.join(f"s{s}" for s in parent_subs)
        
        # Build full hierarchy path
        hierarchy = [f"p{pendant_num}"]
        for i in range(1, len(subsidiaries) + 1):
            subs = subsidiaries[:i]
            hierarchy.append(f"p{pendant_num}" + ''.join(f"s{s}" for s in subs))
        
        return {
            'pendant_num': pendant_num,
            'subsidiaries': subsidiaries,
            'level': len(subsidiaries),
            'parent': parent,
            'full_hierarchy': hierarchy
        }


def parse_kfg_knots(knot_string: str) -> List[Dict[str, Any]]:
    """
    Parse KFG knot format into structured knot clusters.
    
    KFG encodes all knots on a cord in a single string with format:
        <count><type>(<position>,<direction>)[,<value>][axis]; ...
    
    Where:
        - count: Number of knots in cluster (e.g., 5)
        - type: S=Single, L=Long, E=Eight, EE=DoubleEight, LL=DoubleLong, 
                BL=BeltedLong, SP=Spiral, TF=Tufted
        - position: Distance from cord attachment in cm
        - direction: Z=Z-spun, S=S-spun, U=Unknown
        - value: Numeric value of cluster (OPTIONAL - may be in separate Value column)
        - axis: [D] or [U] for axis orientation (OPTIONAL)
    
    Official specification: https://khipufieldguide.com/databook/KFGExcelSpecification.html
    
    Args:
        knot_string: Knot description (e.g., "5S(0.0,U),50;3S(7.0,Z),30" or "10E(13.0,U)")
        
    Returns:
        List of knot cluster dictionaries in sequence order:
        [
            {
                'cluster_ordinal': int - Order in sequence (0-based)
                'knot_type': str - Type code (S, L, E, EE, LL, BL, SP, TF)
                'num_knots': int - Count of knots in cluster
                'position_cm': float - Position from attachment
                'direction': str - Spin direction (Z, S, U)
                'cluster_value': int - Numeric value (if present in string)
                'axis_orientation': str - D or U (if present)
            },
            ...
        ]
        
    Examples:
        >>> parse_kfg_knots("5S(0.0,U),50")
        [{'cluster_ordinal': 0, 'knot_type': 'S', 'num_knots': 5,
          'position_cm': 0.0, 'direction': 'U', 'cluster_value': 50}]
          
        >>> parse_kfg_knots("10E(13.0,U)")
        [{'cluster_ordinal': 0, 'knot_type': 'E', 'num_knots': 10,
          'position_cm': 13.0, 'direction': 'U'}]
          
        >>> parse_kfg_knots("4L(23,S),4[D]")
        [{'cluster_ordinal': 0, 'knot_type': 'L', 'num_knots': 4,
          'position_cm': 23.0, 'direction': 'S', 'cluster_value': 4,
          'axis_orientation': 'D'}]
    """
    if pd.isna(knot_string) or str(knot_string).strip() == '':
        return []
    
    knots = []
    
    # Pattern (UPDATED per official spec): value and axis are OPTIONAL
    # <count><type>(<position>,<direction>)[,<value>][axis]
    # Group 1: count, Group 2: type, Group 3: position, 
    # Group 4: direction, Group 5: value (optional), Group 6: axis (optional)
    pattern = r'(\d+)([A-Z]+)\(([\d.]+),([A-Z]+)\)(?:,(\d+))?(?:\[([DU])\])?'
    
    for i, match in enumerate(re.finditer(pattern, str(knot_string))):
        count, knot_type, position, direction, value, axis = match.groups()
        
        knot_dict = {
            'cluster_ordinal': i,
            'knot_type': knot_type,
            'num_knots': int(count),
            'position_cm': float(position),
            'direction': direction
        }
        
        # Add value only if present in knot string
        if value is not None:
            knot_dict['cluster_value'] = int(value)
        
        # Add axis orientation if present
        if axis is not None:
            knot_dict['axis_orientation'] = axis
        
        knots.append(knot_dict)
    
    return knots


def compute_cord_value(knots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute total numeric value and confidence from parsed knots.
    
    Args:
        knots: List of knot dictionaries from parse_kfg_knots()
        
    Returns:
        Dictionary with:
        {
            'numeric_value': int - Total value across all clusters
            'num_clusters': int - Count of knot clusters
            'confidence': float - Confidence score (0.0-1.0)
            'has_unknown_direction': bool - Any knots marked 'U'
            'position_range': Tuple[float, float] - (min, max) positions
        }
        
    Examples:
        >>> knots = [
        ...     {'cluster_value': 50, 'direction': 'Z', 'position_cm': 0.0},
        ...     {'cluster_value': 30, 'direction': 'Z', 'position_cm': 7.0}
        ... ]
        >>> compute_cord_value(knots)
        {'numeric_value': 80, 'num_clusters': 2, 'confidence': 1.0,
         'has_unknown_direction': False, 'position_range': (0.0, 7.0)}
    """
    if not knots:
        return {
            'numeric_value': 0,
            'num_clusters': 0,
            'confidence': 0.0,
            'has_unknown_direction': False,
            'position_range': (0.0, 0.0)
        }
    
    # Sum cluster_value if present, skip knots without value
    total_value = sum(k.get('cluster_value', 0) for k in knots)
    num_clusters = len(knots)
    has_unknown = any(k['direction'] == 'U' for k in knots)
    has_missing_values = any('cluster_value' not in k for k in knots)
    
    positions = [k['position_cm'] for k in knots]
    position_range = (min(positions), max(positions))
    
    # Confidence heuristic (UPDATED for optional values):
    # - Start at 1.0
    # - Reduce by 0.1 for unknown directions
    # - Reduce by 0.2 if values missing (must use separate Value column)
    confidence = 1.0
    if has_unknown:
        confidence -= 0.1
    if has_missing_values:
        confidence -= 0.2
    
    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        'numeric_value': total_value,
        'num_clusters': num_clusters,
        'confidence': confidence,
        'has_unknown_direction': has_unknown,
        'has_missing_values': has_missing_values,
        'position_range': position_range
    }


def parse_kfg_color(color_string: str) -> List[Dict[str, Any]]:
    """
    Parse KFG color format into color sequence.
    
    KFG uses short letter codes for colors, with colon-separation for
    multi-color cords.
    
    Common codes:
        W=White, MB=Mottled Brown, B=Brown, YB=Yellow-Brown, 
        LB=Light Brown, KB=?, AB=?, BG=?, GG=?, PK=Pink, YY=Yellow, GR=Gray/Green
    
    Args:
        color_string: Color code(s) (e.g., "W", "W:MB", "AB:BG")
        
    Returns:
        List of color dictionaries in sequence:
        [
            {
                'color_code': str - Color abbreviation
                'sequence_ord': int - Order in sequence (0-based)
            },
            ...
        ]
        
    Examples:
        >>> parse_kfg_color("W")
        [{'color_code': 'W', 'sequence_ord': 0}]
        
        >>> parse_kfg_color("W:MB")
        [{'color_code': 'W', 'sequence_ord': 0}, 
         {'color_code': 'MB', 'sequence_ord': 1}]
         
        >>> parse_kfg_color("AB:BG:W")
        [{'color_code': 'AB', 'sequence_ord': 0},
         {'color_code': 'BG', 'sequence_ord': 1},
         {'color_code': 'W', 'sequence_ord': 2}]
    """
    if pd.isna(color_string) or str(color_string).strip() == '':
        return []
    
    color_string = str(color_string).strip()
    
    # Split on colon for multi-color cords
    color_codes = color_string.split(':')
    
    return [
        {
            'color_code': code.strip(),
            'sequence_ord': i
        }
        for i, code in enumerate(color_codes) if code.strip()
    ]


def parse_kfg_metadata(khipu_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Parse KFG Khipu metadata sheet (key-value format).
    
    The Khipu sheet uses a single column with "Key:Value" format:
        KFG_Name:KH0001
        Aliases:LL01, UR176
        Provenance:Chuquitanta
        Region:Coast
        ...
    
    Args:
        khipu_df: DataFrame from reading the 'Khipu' sheet
        
    Returns:
        Dictionary with metadata fields:
        {
            'kfg_name': str
            'aliases': List[str]
            'contributors': str
            'kfg_url': str
            'museum_name': str
            'museum_number': str
            'museum_city_state': Optional[str]
            'museum_country': Optional[str]
            'museum_url': Optional[str]
            'provenance': str
            'region': str
            'creation_date': str
            'excel_write_date': str
            'excel_creator': str
        }
    """
    metadata = {}
    
    # Map of field keys to metadata dictionary keys
    field_mapping = {
        'KFG_Name': 'kfg_name',
        'Name': 'kfg_name',           # KH0483 uses 'Name' instead of 'KFG_Name'
        'Aliases': 'aliases',
        'Contributors': 'contributors',
        'KFG URL': 'kfg_url',
        'Museum Name': 'museum_name',
        'Museum Number': 'museum_number',
        'Museum City/State': 'museum_city_state',
        'Museum Country': 'museum_country',
        'Museum URL': 'museum_url',
        'Provenance': 'provenance',
        'Region': 'region',
        'Creation_Date': 'creation_date',
        'Excel Write Date': 'excel_write_date',
        'Excel File Creator': 'excel_creator'
    }
    
    # Parse key-value rows
    for _, row in khipu_df.iterrows():
        cell = str(row.iloc[0])
        if ':' in cell:
            parts = cell.split(':', 1)
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()
                
                if key in field_mapping:
                    dict_key = field_mapping[key]
                    
                    # Special handling for aliases (comma-separated list)
                    if key == 'Aliases':
                        metadata[dict_key] = [a.strip() for a in value.split(',')]
                    else:
                        metadata[dict_key] = value if value and value != 'nan' else None
    
    return metadata


def parse_primary_cord(primary_cord_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Parse KFG PrimaryCord sheet (key-value format).
    
    Format:
        Structure:P
        Thickness:0.15
        Length:55.0
        Color:W
        Fiber:CN
    
    Args:
        primary_cord_df: DataFrame from reading the 'PrimaryCord' sheet
        
    Returns:
        Dictionary with primary cord properties:
        {
            'structure': str - P=plied, B=braid, W=wrapped
            'thickness': Optional[float] - Diameter in mm
            'length': Optional[float] - Length in cm
            'color': Optional[str] - Color code
            'fiber': Optional[str] - CN=cotton, L=llama, etc.
        }
    """
    primary_cord = {}
    
    field_mapping = {
        'Structure': 'structure',
        'Thickness': 'thickness',
        'Length': 'length',
        'Color': 'color',
        'Fiber': 'fiber',
        'Beginning': 'beginning',
        'Termination': 'termination',
        'Twist': 'twist',
        'Notes': 'notes',
        'Plain_Notes': 'plain_notes',
    }
    
    for _, row in primary_cord_df.iterrows():
        cell = str(row.iloc[0])
        if ':' in cell:
            parts = cell.split(':', 1)
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()
                
                if key in field_mapping:
                    dict_key = field_mapping[key]
                    
                    # Convert numeric fields
                    if key in ['Thickness', 'Length']:
                        try:
                            primary_cord[dict_key] = float(value) if value and value != 'nan' else None
                        except ValueError:
                            primary_cord[dict_key] = None
                    else:
                        primary_cord[dict_key] = value if value and value != 'nan' else None
    
    return primary_cord


# ============================================================================
# Testing and Validation
# ============================================================================

def test_parsers():
    """Test all parsers with example data."""
    print("=" * 80)
    print("TESTING KFG PARSERS")
    print("=" * 80)
    print()
    
    # Test cord hierarchy parser
    print("1. Cord Hierarchy Parser")
    print("-" * 80)
    test_cords = ['p1', 'p6s1', 'p10s1s1', 'p10s1s1s1', 'invalid']
    for cord in test_cords:
        result = parse_cord_hierarchy(cord)
        print(f"  {cord:12} → {result}")
    print()
    
    # Test knot parser
    print("2. Knot Parser")
    print("-" * 80)
    test_knots = [
        ("5S(0.0,U),50", "Single knot with value"),
        ("5S(0.0,U),50;3S(7.0,Z),30;1S(14.0,Z),10;6L(23.5,Z),6", "Multiple clusters with values"),
        ("10E(13.0,U)", "Figure-8 WITHOUT value (optional)"),
        ("4L(23,S),4[D]", "Long knot with value AND axis orientation"),
        ("10E(13.0,U);20L(14.0,U);40L(14.5,U)", "Multiple clusters WITHOUT values")
    ]
    for knot_str, description in test_knots:
        result = parse_kfg_knots(knot_str)
        print(f"  {description}")
        print(f"  Input: {knot_str}")
        print(f"  Parsed: {len(result)} clusters")
        for k in result:
            value_str = f" = {k['cluster_value']}" if 'cluster_value' in k else " (no value)"
            axis_str = f" [{k['axis_orientation']}]" if 'axis_orientation' in k else ""
            print(f"    - Cluster {k['cluster_ordinal']}: {k['num_knots']}{k['knot_type']}{value_str}{axis_str}")
        value_info = compute_cord_value(result)
        missing_flag = " (values missing)" if value_info.get('has_missing_values') else ""
        print(f"  Total value: {value_info['numeric_value']} (confidence: {value_info['confidence']:.2f}){missing_flag}")
        print()
    
    # Test color parser
    print("3. Color Parser")
    print("-" * 80)
    test_colors = ['W', 'W:MB', 'AB:BG', 'W:KB:YB']
    for color in test_colors:
        result = parse_kfg_color(color)
        print(f"  {color:15} → {result}")
    print()
    
    print("=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_parsers()
