from typing import Dict, List, Optional
from pydantic import BaseModel
from .device import Device
from .benchmark import BenchmarkDataFloat


class BenchmarkResult(BaseModel):
    """Result from a benchmark job including device info and performance data."""

    device: Device
    benchmark_data: List[BenchmarkDataFloat]
    # Optional output tensor file paths
    outputs: Optional[Dict[str, Dict[str, str]]] = (
        None  # Dict[compute_unit, Dict[tensor_name, file_path]]
    )
