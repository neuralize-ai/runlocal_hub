"""
Response wrapper models for benchmark and prediction operations.
"""

from typing import List, Union
from pydantic import BaseModel

from .benchmark_result import BenchmarkResult
from .prediction import PredictionResult


class BenchmarkResponse(BaseModel):
    """
    Response from benchmark operation including results and job metadata.
    """
    
    # The actual benchmark results
    results: Union[BenchmarkResult, List[BenchmarkResult]]
    
    # Job tracking metadata
    all_job_ids: List[str] = []
    completed_job_ids: List[str] = []
    incomplete_job_ids: List[str] = []
    
    @property
    def has_incomplete_jobs(self) -> bool:
        """Check if there are incomplete jobs."""
        return len(self.incomplete_job_ids) > 0
    
    @property
    def total_jobs(self) -> int:
        """Total number of jobs submitted."""
        return len(self.all_job_ids)
    
    @property
    def completed_count(self) -> int:
        """Number of completed jobs."""
        return len(self.completed_job_ids)
    
    @property
    def incomplete_count(self) -> int:
        """Number of incomplete jobs."""
        return len(self.incomplete_job_ids)
    
    @property
    def completion_rate(self) -> float:
        """Completion rate as a percentage (0-100)."""
        if self.total_jobs == 0:
            return 100.0
        return (self.completed_count / self.total_jobs) * 100.0


class PredictionResponse(BaseModel):
    """
    Response from prediction operation including results and job metadata.
    """
    
    # The actual prediction results
    results: Union[PredictionResult, List[PredictionResult]]
    
    # Job tracking metadata
    all_job_ids: List[str] = []
    completed_job_ids: List[str] = []
    incomplete_job_ids: List[str] = []
    
    @property
    def has_incomplete_jobs(self) -> bool:
        """Check if there are incomplete jobs."""
        return len(self.incomplete_job_ids) > 0
    
    @property
    def total_jobs(self) -> int:
        """Total number of jobs submitted."""
        return len(self.all_job_ids)
    
    @property
    def completed_count(self) -> int:
        """Number of completed jobs."""
        return len(self.completed_job_ids)
    
    @property
    def incomplete_count(self) -> int:
        """Number of incomplete jobs."""
        return len(self.incomplete_job_ids)
    
    @property
    def completion_rate(self) -> float:
        """Completion rate as a percentage (0-100)."""
        if self.total_jobs == 0:
            return 100.0
        return (self.completed_count / self.total_jobs) * 100.0