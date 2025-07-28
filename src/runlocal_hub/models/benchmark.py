"""
Benchmark-related models.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .device import Device
from .job import BenchmarkStatus
from .settings import Framework


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

    Versions: Optional[Dict[str, str]] = None
    Settings: Optional[Dict[str, Any]] = None

    def to_json_dict(self) -> Dict:
        """
        Convert to JSON-friendly dictionary.
        Needed for post requests where Decimals need to be strings.
        """
        from ..utils.json import decimal_list_to_str, decimal_to_str

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
            "PrefillTokens": self.PrefillTokens,
            "GenerationTokens": self.GenerationTokens,
            "PrefillTPS": decimal_to_str(self.PrefillTPS),
            "GenerateTPS": decimal_to_str(self.GenerateTPS),
            "PeakLoadRamUsage": decimal_to_str(self.PeakLoadRamUsage),
            "PeakRamUsage": decimal_to_str(self.PeakRamUsage),
            "PeakPrefillRamUsage": decimal_to_str(self.PeakPrefillRamUsage),
            "PeakGenerateRamUsage": decimal_to_str(self.PeakGenerateRamUsage),
            "Versions": self.Versions,
            "Settings": self.Settings,
        }

        # Remove None values
        return {k: v for k, v in result.items() if v is not None}

    def to_float_dict(self) -> Dict:
        """
        Convert to dictionary with Decimal values converted to float.
        Useful for easier data manipulation and display.
        """

        def decimal_to_float(value):
            if value is None:
                return None
            if isinstance(value, Decimal):
                return float(value)
            return value

        def decimal_list_to_float(value_list):
            if value_list is None:
                return None
            return [float(v) if isinstance(v, Decimal) else v for v in value_list]

        return {
            "Success": self.Success,
            "Status": self.Status,
            "FailureReason": self.FailureReason,
            "FailureError": self.FailureError,
            "Stdout": self.Stdout,
            "Stderr": self.Stderr,
            "ComputeUnit": self.ComputeUnit,
            "LoadMsArray": decimal_list_to_float(self.LoadMsArray),
            "LoadMsAverage": decimal_to_float(self.LoadMsAverage),
            "LoadMsMedian": decimal_to_float(self.LoadMsMedian),
            "InferenceMsArray": decimal_list_to_float(self.InferenceMsArray),
            "InferenceMsAverage": decimal_to_float(self.InferenceMsAverage),
            "InferenceMsMedian": decimal_to_float(self.InferenceMsMedian),
            "PrefillTokens": self.PrefillTokens,  # int, no conversion needed
            "GenerationTokens": self.GenerationTokens,  # int, no conversion needed
            "PrefillTPS": decimal_to_float(self.PrefillTPS),
            "GenerateTPS": decimal_to_float(self.GenerateTPS),
            "PeakLoadRamUsage": decimal_to_float(self.PeakLoadRamUsage),
            "PeakInferenceRamUsage": decimal_to_float(self.PeakRamUsage),
            "PeakPrefillRamUsage": decimal_to_float(self.PeakPrefillRamUsage),
            "PeakGenerateRamUsage": decimal_to_float(self.PeakGenerateRamUsage),
            "OutputTensorsId": self.OutputTensorsId,
            "Versions": self.Versions,
            "Settings": self.Settings,
        }


class BenchmarkDataFloat(BaseModel):
    """Benchmark data with float values instead of Decimal for easier use."""

    Success: Optional[bool] = None
    Status: Optional[BenchmarkStatus] = None
    ComputeUnit: str

    # load
    LoadMsArray: Optional[List[float]] = None
    LoadMsAverage: Optional[float] = None
    LoadMsMedian: Optional[float] = None
    # traditional inference
    InferenceMsArray: Optional[List[float]] = None
    InferenceMsAverage: Optional[float] = None
    InferenceMsMedian: Optional[float] = None
    # GenAI inference
    PrefillTokens: Optional[int] = None
    GenerationTokens: Optional[int] = None
    PrefillTPS: Optional[float] = None
    GenerateTPS: Optional[float] = None
    # peak ram usage (converted to float)
    PeakLoadRamUsage: Optional[float] = None
    PeakInferenceRamUsage: Optional[float] = None
    PeakPrefillRamUsage: Optional[float] = None
    PeakGenerateRamUsage: Optional[float] = None

    FailureReason: Optional[str] = None
    FailureError: Optional[str] = None
    Stdout: Optional[str] = None
    Stderr: Optional[str] = None

    OutputTensorsId: Optional[str] = None

    Versions: Optional[Dict[str, str]] = None
    Settings: Optional[Dict[str, Any]] = None

    @classmethod
    def from_benchmark_data(cls, bd: BenchmarkData) -> "BenchmarkDataFloat":
        """Create a BenchmarkDataFloat from a BenchmarkData instance."""
        return cls(**bd.to_float_dict())


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
