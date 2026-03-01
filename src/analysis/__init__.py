"""
Analysis module for khipu pattern discovery and hypothesis testing.
"""

from .value_computation import ValueComputer
from .kfg_summation_detector import KFGSummationDetector, SummationMatch, Cord
from .kfg_relation_loader import (
    KFGRelationLoader,
    apply_exclusivity,
    CORD_EXCLUSIVITY_ORDER,
    GROUP_LEVEL_PATTERNS,
    ALL_PATTERNS,
)

__all__ = [
    'ValueComputer',
    'KFGSummationDetector',
    'SummationMatch',
    'Cord',
    'KFGRelationLoader',
    'apply_exclusivity',
    'CORD_EXCLUSIVITY_ORDER',
    'GROUP_LEVEL_PATTERNS',
    'ALL_PATTERNS',
    # 'SummationPatternDetector',  # Deprecated
]
