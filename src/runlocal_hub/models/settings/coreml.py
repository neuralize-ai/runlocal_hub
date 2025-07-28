from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SpecializationStrategy(str, Enum):
    default = "default"
    fastPrediction = "fastPrediction"


class CoreMLSettings(BaseModel):
    allowLowPrecisionAccumulationOnGPU: Optional[bool] = None
    specializationStrategy: Optional[SpecializationStrategy] = None
