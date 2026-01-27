"""
Ranking Module
AHP-based professor ranking system
"""

from .professor_aggregator import ProfessorAggregator
from .ranker import ProfessorRanker

__all__ = [
    "ProfessorAggregator",
    "ProfessorRanker"
]
