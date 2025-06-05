from typing import Dict
import numpy as np
from pydantic import BaseModel, ConfigDict
from .device import Device


class PredictionResult(BaseModel):
    """Result from a prediction job including device info and outputs."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    device: Device
    outputs: Dict[str, Dict[str, np.ndarray]]
    job_id: str
    elapsed_time: float