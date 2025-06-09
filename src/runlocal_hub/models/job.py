"""
Job-related models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..models import BenchmarkStatus
from .device import Device


class JobType(str, Enum):
    """Type of job to run."""

    BENCHMARK = "benchmark"
    PREDICTION = "prediction"


@dataclass
class JobResult:
    """
    Result of a job polling operation.
    """

    job_id: str
    status: BenchmarkStatus
    device_name: Optional[str] = None
    device: Optional[Device] = None
    data: Optional[Any] = None
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if the job is complete (success or failure)."""
        return self.status in [BenchmarkStatus.Complete, BenchmarkStatus.Failed]

    @property
    def is_successful(self) -> bool:
        """Check if the job completed successfully."""
        return self.status == BenchmarkStatus.Complete

    @property
    def is_failed(self) -> bool:
        """Check if the job failed."""
        return self.status == BenchmarkStatus.Failed
