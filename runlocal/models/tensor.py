"""
Tensor-related models.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class IOType(str, Enum):
    """Type of IO tensor."""

    INPUT = "input"
    OUTPUT = "output"


class TensorInfo(BaseModel):
    """Information about a single tensor."""

    Shape: List[int]
    Dtype: str
    SizeBytes: int


class IOTensorsMetadata(BaseModel):
    """Metadata about uploaded tensors."""

    UserId: str
    Id: str
    IOType: IOType
    TensorMetadata: Dict[str, TensorInfo]
    SourceBenchmarkIds: Optional[List[str]] = None
    CreatedUtc: str


class IOTensorsPresignedUrlResponse(BaseModel):
    """Response containing presigned URL for tensor download."""

    tensors_id: str
    presigned_url: str
    expires_in_seconds: int

