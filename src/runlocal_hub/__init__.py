"""RunLocal API Client Package"""

from .__version__ import __version__
from .client import RunLocalClient
from .models.device import Device, DeviceUsage
from .models.job import JobType, JobResult
from .models.benchmark import BenchmarkData, BenchmarkStatus
from .models.tensor import IOType, TensorInfo
from .devices.filters import DeviceFilters
from .exceptions import (
    RunLocalError,
    AuthenticationError,
    APIError,
    ModelNotFoundError,
    DeviceNotAvailableError,
    JobTimeoutError,
    TensorError,
    UploadError,
    ValidationError,
    NetworkError,
    ConfigurationError,
)
from .utils.display import display_benchmark_results, display_failed_benchmarks

__all__ = [
    "__version__",
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
    "AuthenticationError",
    "APIError",
    "ModelNotFoundError",
    "DeviceNotAvailableError",
    "JobTimeoutError",
    "TensorError",
    "UploadError",
    "ValidationError",
    "NetworkError",
    "ConfigurationError",
    "display_benchmark_results",
    "display_failed_benchmarks",
]
