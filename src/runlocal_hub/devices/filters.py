"""
Device filtering criteria.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DeviceFilters:
    """
    Criteria for filtering devices.

    All filters are optional and will be applied if provided.
    """

    device_name: Optional[str] = None
    ram_min: Optional[int] = None
    ram_max: Optional[int] = None
    soc: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    os: Optional[str] = None
    compute_units: Optional[List[str]] = None

    def __post_init__(self):
        """Validate filter values."""
        if self.ram_min is not None and self.ram_min < 0:
            raise ValueError("ram_min must be non-negative")

        if self.ram_max is not None and self.ram_max < 0:
            raise ValueError("ram_max must be non-negative")

        if self.ram_min is not None and self.ram_max is not None:
            if self.ram_min > self.ram_max:
                raise ValueError("ram_min cannot be greater than ram_max")

        if self.year_min is not None and self.year_max is not None:
            if self.year_min > self.year_max:
                raise ValueError("year_min cannot be greater than year_max")
