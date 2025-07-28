from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SpecializationStrategy(str, Enum):
    default = "default"
    fastPrediction = "fastPrediction"


class CoreMLSettings(BaseModel):
    allowLowPrecisionAccumulationOnGPU: Optional[bool] = None
    specializationStrategy: Optional[SpecializationStrategy] = None

    def format(self):
        return {"CoreML": self.model_dump(exclude_unset=True)}
