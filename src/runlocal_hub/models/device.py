"""
Device-related models.
"""

from typing import List, Optional

from pydantic import BaseModel


class Device(BaseModel):
    """Information about a device."""

    Name: str
    Year: int
    Soc: str
    Ram: int
    OS: str
    OSVersion: str
    DiscreteGpu: Optional[str] = None
    VRam: Optional[int] = None

    def to_device_id(self) -> str:
        """Convert device to a unique device ID string."""
        device_id = f"Name={self.Name}|Year={self.Year}|Soc={self.Soc}|Ram={self.Ram}"
        if self.DiscreteGpu:
            device_id += f"|DiscreteGpu={self.DiscreteGpu}"
        if self.VRam:
            device_id += f"|VRam={self.VRam}"

        device_id += f"|OS={self.OS}|OSVersion={self.OSVersion}"
        return device_id


class DeviceUsage(BaseModel):
    """Device with available compute units."""

    device: Device
    compute_units: List[str]
    native_device_id: str
