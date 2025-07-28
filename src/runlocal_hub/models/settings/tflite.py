from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel

from .common import NumThreads


class InferencePreference(str, Enum):
    FAST_SINGLE_ANSWER = "FAST_SINGLE_ANSWER"
    SUSTAINED_SPEED = "SUSTAINED_SPEED"
    BALANCED = "BALANCED"


class InferencePriority(str, Enum):
    AUTO = "AUTO"
    MAX_PRECISION = "MAX_PRECISION"
    MIN_LATENCY = "MIN_LATENCY"
    MIN_MEMORY_USAGE = "MIN_MEMORY_USAGE"


class PerformanceMode(str, Enum):
    DEFAULT = "Default"
    SUSTAINED_HIGH_PERFORMANCE = "SustainedHighPerformance"
    BURST = "Burst"
    HIGH_PERFORMANCE = "HighPerformance"
    POWER_SAVER = "PowerSaver"
    LOW_POWER_SAVER = "LowPowerSaver"
    HIGH_POWER_SAVER = "HighPowerSaver"
    LOW_BALANCED = "LowBalanced"
    BALANCED = "Balanced"
    EXTREME_POWER_SAVER = "ExtremePowerSaver"


class Precision(str, Enum):
    QUANTIZED = "Quantized"
    FP16 = "Fp16"


class PdSession(str, Enum):
    UNSIGNED_PD = "UnsignedPd"
    SIGNED_PD = "SignedPd"


class OptimizationStrategy(str, Enum):
    OPTIMIZE_FOR_INFERENCE = "OptimizeForInference"
    OPTIMIZE_FOR_PREPARE = "OptimizeForPrepare"
    OPTIMIZE_FOR_INFERENCE_O3 = "OptimizeForInferenceO3"


class XNNPACKSettings(BaseModel):
    num_threads: Optional[Union[int, NumThreads]] = None
    QS8: Optional[bool] = None
    QU8: Optional[bool] = None
    FORCE_FP16: Optional[bool] = None
    DYNAMIC_FULLY_CONNECTED: Optional[bool] = None
    VARIABLE_OPERATORS: Optional[bool] = None
    TRANSIENT_INDIRECTION_BUFFER: Optional[bool] = None
    ENABLE_LATEST_OPERATORS: Optional[bool] = None


class GPUSettings(BaseModel):
    is_precision_loss_allowed: Optional[bool] = None
    inference_preference: Optional[InferencePreference] = None
    inference_priority1: Optional[InferencePriority] = None
    inference_priority2: Optional[InferencePriority] = None
    inference_priority3: Optional[InferencePriority] = None
    enable_quant: Optional[bool] = None
    enable_serialization: Optional[bool] = None
    max_delegated_partitions: Optional[int] = None


class QNNSettings(BaseModel):
    performance_mode: Optional[PerformanceMode] = None
    precision: Optional[Precision] = None
    pd_session: Optional[PdSession] = None
    optimization_strategy: Optional[OptimizationStrategy] = None
    useConvHmx: Optional[bool] = None
    useFoldRelu: Optional[bool] = None
    vtcm_size: Optional[int] = None
    num_hvx_threads: Optional[int] = None


class TFLiteSettings(BaseModel):
    XNNPACK: Optional[XNNPACKSettings] = None
    GPU: Optional[GPUSettings] = None
    QNN: Optional[QNNSettings] = None
