"""
Job-related models.
"""
from enum import Enum


class JobType(str, Enum):
    """Type of job to run."""
    
    BENCHMARK = "benchmark"
    PREDICTION = "prediction"