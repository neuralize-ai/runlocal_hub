from enum import Enum
from typing import Optional

from pydantic import BaseModel

from .common import NumThreads


class InferencePrecision(str, Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    I8 = "i8"


class PerformanceHint(str, Enum):
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CUMULATIVE_THROUGHPUT = "cumulative_throughput"


class SchedulingCoreType(str, Enum):
    ANY_CORE = "any_core"
    PCORE_ONLY = "pcore_only"
    ECORE_ONLY = "ecore_only"


class OVExecutionMode(str, Enum):
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"


class CacheMode(str, Enum):
    OPTIMIZE_SIZE = "optimize_size"
    OPTIMIZE_SPEED = "optimize_speed"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CPUSettings(BaseModel):
    denormals_optimization: Optional[bool] = None
    sparse_weights_decompression_rate: Optional[float] = None


class GPUSettings(BaseModel):
    cache_mode: Optional[CacheMode] = None
    enable_loop_unrolling: Optional[bool] = None
    disable_winograd_convolution: Optional[bool] = None
    host_task_priority: Optional[TaskPriority] = None
    enable_kernels_reuse: Optional[bool] = None


class NPUSettings(BaseModel):
    qdq_optimization: Optional[bool] = None
    turbo: Optional[bool] = None
    tiles: Optional[int] = None
    defer_weights_load: Optional[bool] = None


class OpenVINOSettings(BaseModel):
    inference_precision: Optional[InferencePrecision] = None
    performance_hint: Optional[PerformanceHint] = None
    scheduling_core_type: Optional[SchedulingCoreType] = None
    enable_cpu_pinning: Optional[bool] = None
    enable_hyper_threading: Optional[bool] = None
    execution_mode: Optional[OVExecutionMode] = None
    dynamic_quantization_group_size: Optional[int] = None
    inference_num_threads: Optional[int | NumThreads] = None

    # Device-specific settings
    CPU: Optional[CPUSettings] = None
    GPU: Optional[GPUSettings] = None
    NPU: Optional[NPUSettings] = None

    def format(self):
        return {"OpenVINO": self.model_dump(exclude_unset=True)}
