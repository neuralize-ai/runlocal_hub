"""RunLocal API Client Package"""

from .client import RunLocalClient
from .models.device import Device, DeviceUsage
from .models.job import JobType, JobResult
from .models.benchmark import BenchmarkData, BenchmarkStatus
from .models.tensor import IOType, TensorInfo
from .devices.filters import DeviceFilters
from .exceptions import RunLocalError, ConfigurationError, ValidationError, UploadError
from .utils.display import display_benchmark_results, display_failed_benchmarks

__version__ = "0.1.0"
__all__ = [
    "RunLocalClient",
    "Device",
    "DeviceUsage",
    "DeviceFilters",
    "JobType",
    "JobResult",
    "IOType",
    "TensorInfo",
    "BenchmarkData",
    "BenchmarkStatus",
    "RunLocalError",
    "ConfigurationError",
    "ValidationError",
    "UploadError",
    "display_benchmark_results",
    "display_failed_benchmarks",
]
