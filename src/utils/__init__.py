"""
Validation module for khipu analysis.

Provides tools to test hypotheses, validate arithmetic consistency,
and identify high-quality subsets for analysis.
"""

from .arithmetic_validator import ArithmeticValidator

__all__ = [
    'ArithmeticValidator'
]
