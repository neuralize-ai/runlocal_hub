"""
Benchmark-related models.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from .device import Device
from .job import BenchmarkStatus
from .settings import Framework


class BenchmarkData(BaseModel):
    """Data from a single benchmark run."""

    Success: Optional[bool] = None
    Status: Optional[BenchmarkStatus] = None
    ComputeUnit: str

    # load
    LoadMsArray: Optional[List[float]] = None
    LoadMsAverage: Optional[float] = None
    LoadMsMedian: Optional[float] = None
    # total inference
    InferenceMsArray: Optional[List[float]] = None
    InferenceMsAverage: Optional[float] = None
    InferenceMsMedian: Optional[float] = None
    # GenAI inference
    PrefillTokens: Optional[int] = None
    GenerationTokens: Optional[int] = None
    PrefillTPS: Optional[float] = None
    GenerateTPS: Optional[float] = None
    # peak ram usage
    PeakLoadRamUsage: Optional[float] = None
    # TODO: inference, named "PeakRamUsage" for legacy support
    PeakRamUsage: Optional[float] = None
    # peak genai ram usage
    PeakPrefillRamUsage: Optional[float] = None
    PeakGenerateRamUsage: Optional[float] = None

    FailureReason: Optional[str] = None
    FailureError: Optional[str] = None
    Stdout: Optional[str] = None
    Stderr: Optional[str] = None

    OutputTensorsId: Optional[str] = None

    Versions: Optional[Dict[str, str]] = None
    Settings: Optional[Dict[str, Any]] = None

    @field_validator('LoadMsArray', 'InferenceMsArray', mode='before')
    @classmethod
    def convert_decimal_arrays(cls, v):
        """Convert Decimal arrays to float arrays."""
        if v is None:
            return None
        return [float(item) for item in v]

    @field_validator(
        'LoadMsAverage', 'LoadMsMedian', 'InferenceMsAverage', 'InferenceMsMedian',
        'PrefillTPS', 'GenerateTPS', 'PeakLoadRamUsage', 'PeakRamUsage',
        'PeakPrefillRamUsage', 'PeakGenerateRamUsage', mode='before'
    )
    @classmethod
    def convert_decimal_values(cls, v):
        """Convert Decimal values to float."""
        if v is None:
            return None
        return float(v)



class BenchmarkDbItem(BaseModel):
    """Benchmark database item."""

    UploadId: str
    DeviceInfo: Optional[Device] = None
    Status: BenchmarkStatus
    BenchmarkData: List[BenchmarkData]
    RuntimeFramework: Optional[Framework] = None

    def to_dict(self):
        """Convert to dictionary."""
        d = self.model_dump()
        return d
