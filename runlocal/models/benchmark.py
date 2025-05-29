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
        Needed for post requests where Decimals need to be strings.
        """
        from ..utils.json import decimal_to_str, decimal_list_to_str
        
        result = {
            "Success": self.Success,
            "Status": self.Status,
            "FailureReason": self.FailureReason,
            "FailureError": self.FailureError,
            "Stdout": self.Stdout,
            "Stderr": self.Stderr,
            "ComputeUnit": self.ComputeUnit,
            "LoadMsArray": decimal_list_to_str(self.LoadMsArray),
            "LoadMsAverage": decimal_to_str(self.LoadMsAverage),
            "LoadMsMedian": decimal_to_str(self.LoadMsMedian),
            "InferenceMsArray": decimal_list_to_str(self.InferenceMsArray),
            "InferenceMsAverage": decimal_to_str(self.InferenceMsAverage),
            "InferenceMsMedian": decimal_to_str(self.InferenceMsMedian),
            "PrefillTokens": decimal_to_str(self.PrefillTokens),
            "GenerationTokens": decimal_to_str(self.GenerationTokens),
            "PrefillTPS": decimal_to_str(self.PrefillTPS),
            "GenerateTPS": decimal_to_str(self.GenerateTPS),
            "PeakLoadRamUsage": decimal_to_str(self.PeakLoadRamUsage),
            "PeakRamUsage": decimal_to_str(self.PeakRamUsage),
            "PeakPrefillRamUsage": decimal_to_str(self.PeakPrefillRamUsage),
            "PeakGenerateRamUsage": decimal_to_str(self.PeakGenerateRamUsage),
        }
        
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}


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