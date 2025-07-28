"""RunLocal API Client Package"""

from .__version__ import __version__
from .client import RunLocalClient
from .devices import DeviceFilters
from .exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    DeviceNotAvailableError,
    JobTimeoutError,
    ModelNotFoundError,
    NetworkError,
    RunLocalError,
    TensorError,
    UploadError,
    ValidationError,
)
from .models import (
    BenchmarkData,
    BenchmarkStatus,
    Framework,
    Device,
    DeviceUsage,
    JobResult,
    JobType,
    IOType,
    TensorInfo,
    RuntimeSettings,
)
from .utils import (
    display_benchmark_results,
    display_failed_benchmarks,
    display_model,
)

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
    "RuntimeSettings",
    "Framework",
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
    "display_model",
]
