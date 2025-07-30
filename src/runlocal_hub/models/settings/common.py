from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..job import JobType


class Framework(str, Enum):
    COREML = "CoreML"
    ONNXRUNTIME = "ONNX Runtime"
    OPENVINO = "OpenVINO"
    TFLITE = "TFLite"
    LLAMACPP = "LlamaCpp"


class RuntimeSettings(BaseModel):
    # Overwrite default framework (eg. use openvino for onnx model)
    framework: Optional[Framework] = None
    framework_settings: Optional[Dict[str, Any]] = None


class DeviceBenchmarkRequest(BaseModel):
    device_id: str
    compute_units: Optional[List[str]] = None


class BenchmarkRequest(BaseModel):
    device_requests: List[DeviceBenchmarkRequest]
    test_name: Optional[str] = None
    user_id: Optional[str] = None
    settings: Optional[RuntimeSettings] = None
    input_tensors_id: Optional[str] = None
    job_type: JobType = (
        JobType.BENCHMARK
    )  # Default to benchmark for backward compatibility
    skip_existing: bool = False  # Skip benchmarks that have already been run


class NumThreads(str, Enum):
    ALL_CORES = "allcores"
    HALF_CORES = "halfcores"
