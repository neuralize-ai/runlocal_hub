"""
Benchmark-related models.
"""
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from .device import Device


class BenchmarkStatus(str, Enum):
    """Status of a benchmark job."""
    
    Pending = "Pending"  # not started, still in queue
    Complete = "Complete"
    Failed = "Failed"
    Running = "Running"
    Deleted = "Deleted"


class BenchmarkData(BaseModel):
    """Data from a single benchmark run."""
    
    Success: Optional[bool] = None
    Status: Optional[BenchmarkStatus] = None
    ComputeUnit: str

    # load
    LoadMsArray: Optional[List[Decimal]] = None
    LoadMsAverage: Optional[Decimal] = None
    LoadMsMedian: Optional[Decimal] = None
    # total inference
    InferenceMsArray: Optional[List[Decimal]] = None
    InferenceMsAverage: Optional[Decimal] = None
    InferenceMsMedian: Optional[Decimal] = None
    # GenAI inference
    PrefillTokens: Optional[int] = None
    GenerationTokens: Optional[int] = None
    PrefillTPS: Optional[Decimal] = None
    GenerateTPS: Optional[Decimal] = None
    # peak ram usage
    PeakLoadRamUsage: Optional[Decimal] = None
    # TODO: inference, named "PeakRamUsage" for legacy support
    PeakRamUsage: Optional[Decimal] = None
    # peak genai ram usage
    PeakPrefillRamUsage: Optional[Decimal] = None
    PeakGenerateRamUsage: Optional[Decimal] = None

    FailureReason: Optional[str] = None
    FailureError: Optional[str] = None
    Stdout: Optional[str] = None
    Stderr: Optional[str] = None

    OutputTensorsId: Optional[str] = None

    def to_json_dict(self) -> Dict:
        """
        Convert to JSON-friendly dictionary.
        Needed for post requests, not for dynamodb, dynamo can handle Decimals.
        Decimals need to be strings.
        """
        return {
            "Success": self.Success if self.Success is not None else None,
            "Status": self.Status if self.Status is not None else None,
            "FailureReason": self.FailureReason,
            "FailureError": self.FailureError,
            "Stdout": self.Stdout,
            "Stderr": self.Stderr,
            "ComputeUnit": self.ComputeUnit,
            "LoadMsArray": [str(x) for x in self.LoadMsArray]
            if self.LoadMsArray is not None
            else None,
            "LoadMsAverage": str(self.LoadMsAverage)
            if self.LoadMsAverage is not None
            else None,
            "LoadMsMedian": str(self.LoadMsMedian)
            if self.LoadMsMedian is not None
            else None,
            "InferenceMsArray": [str(x) for x in self.InferenceMsArray]
            if self.InferenceMsArray is not None
            else None,
            "InferenceMsAverage": str(self.InferenceMsAverage)
            if self.InferenceMsAverage is not None
            else None,
            "InferenceMsMedian": str(self.InferenceMsMedian)
            if self.InferenceMsMedian is not None
            else None,
            "PrefillTokens": str(self.PrefillTokens)
            if self.PrefillTokens is not None
            else None,
            "GenerationTokens": str(self.GenerationTokens)
            if self.GenerationTokens is not None
            else None,
            "PrefillTPS": str(self.PrefillTPS) if self.PrefillTPS is not None else None,
            "GenerateTPS": str(self.GenerateTPS)
            if self.GenerateTPS is not None
            else None,
            "PeakLoadRamUsage": str(self.PeakLoadRamUsage)
            if self.PeakLoadRamUsage is not None
            else None,
            "PeakRamUsage": str(self.PeakRamUsage)
            if self.PeakRamUsage is not None
            else None,
            "PeakPrefillRamUsage": str(self.PeakPrefillRamUsage)
            if self.PeakPrefillRamUsage is not None
            else None,
            "PeakGenerateRamUsage": str(self.PeakGenerateRamUsage)
            if self.PeakGenerateRamUsage is not None
            else None,
        }


class BenchmarkDbItem(BaseModel):
    """Benchmark database item."""
    
    UploadId: str
    DeviceInfo: Optional[Device] = None
    Status: BenchmarkStatus
    BenchmarkData: List[BenchmarkData]

    def to_dict(self):
        """Convert to dictionary."""
        d = self.dict()
        return d