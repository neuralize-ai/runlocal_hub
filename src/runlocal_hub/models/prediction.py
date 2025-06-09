from typing import Dict
from pydantic import BaseModel
from .device import Device


class PredictionResult(BaseModel):
    """Result from a prediction job including device info and output file paths."""

    device: Device
    outputs: Dict[
        str, Dict[str, str]
    ]  # Dict[compute_unit, Dict[tensor_name, file_path]]
    job_id: str
    status: str
    modelid: str
