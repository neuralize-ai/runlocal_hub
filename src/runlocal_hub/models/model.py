from decimal import Decimal
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel


class UploadedModelType(str, Enum):
    MLPACKAGE = "MLPACKAGE"
    MLMODEL = "MLMODEL"
    MLMODELC = "MLMODELC"
    OPENVINO = "OPENVINO"
    ONNX = "ONNX"
    TF_SAVED_MODEL = "TF_SAVED_MODEL"
    EXECUTORCH = "EXECUTORCH"
    TORCHSCRIPT = "TORCHSCRIPT"

    # Keras can be saved in 3 different file formats
    KERAS_H5 = "KERAS_H5"
    KERAS_SAVED_MODEL = "KERAS_SAVED_MODEL"
    KERAS_KERAS = "KERAS_KERAS"

    TFLITE = "TFLITE"
    TFLITE_ZIPPED = "TFLITE_ZIPPED"

    GGUF = "GGUF"


class LicenseInfo(BaseModel):
    name: str
    url: str


class UploadDbItem(BaseModel):
    UploadId: str
    CreatedUtc: str
    UpdatedUtc: str
    FileName: str
    FileSize: Decimal
    Benchmarks: Optional[List[str]] = None
    ModelType: Optional[UploadedModelType] = None
    License: Optional[LicenseInfo] = None

    Tag: Optional[str] = None

    Source: Optional[str] = None
