from enum import Enum
from typing import Optional
from .common import NumThreads

from pydantic import BaseModel


class GraphOptimizationLevel(str, Enum):
    NONE = "none"
    BASIC = "basic"
    EXTENDED = "extended"
    ALL = "all"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ModelFormat(str, Enum):
    NEURAL_NETWORK = "NeuralNetwork"
    ML_PROGRAM = "MLProgram"


class SpecializationStrategy(str, Enum):
    DEFAULT = "Default"
    FAST_PREDICTION = "FastPrediction"


class HtpPerformanceMode(str, Enum):
    BURST = "burst"
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    HIGH_POWER_SAVER = "high_power_saver"
    LOW_BALANCED = "low_balanced"
    LOW_POWER_SAVER = "low_power_saver"
    SUSTAINED_HIGH_PERFORMANCE = "sustained_high_performance"


class XNNPACKSettings(BaseModel):
    intra_op_num_threads: Optional[int | NumThreads] = None


class OpenVINOSettings(BaseModel):
    num_of_threads: Optional[int | NumThreads] = None
    enable_qdq_optimizer: Optional[bool] = None
    disable_dynamic_shapes: Optional[bool] = None


class CoreMLSettings(BaseModel):
    ModelFormat: Optional["ModelFormat"] = None
    RequireStaticInputShapes: Optional[bool] = None
    EnableOnSubgraphs: Optional[bool] = None
    SpecializationStrategy: Optional["SpecializationStrategy"] = None
    AllowLowPrecisionAccumulationOnGPU: Optional[bool] = None


class QNNSettings(BaseModel):
    htp_performance_mode: Optional[HtpPerformanceMode] = None
    htp_graph_finalization_optimization_mode: Optional[int] = None
    enable_htp_fp16_precision: Optional[bool] = None
    offload_graph_io_quantization: Optional[bool] = None


class OnnxSettings(BaseModel):
    intra_op_num_threads: Optional[int | NumThreads] = None
    inter_op_num_threads: Optional[int | NumThreads] = None
    allow_intra_op_spinning: Optional[bool] = None
    allow_inter_op_spinning: Optional[bool] = None
    disable_mem_pattern: Optional[bool] = None
    graph_optimization_level: Optional[GraphOptimizationLevel] = None
    execution_mode: Optional[ExecutionMode] = None

    # Execution Provider specific settings
    XNNPACK: Optional[XNNPACKSettings] = None
    OpenVINO: Optional[OpenVINOSettings] = None
    CoreML: Optional[CoreMLSettings] = None
    QNN: Optional[QNNSettings] = None

    def format(self):
        return {"Onnx": self.model_dump(exclude_unset=True)}
