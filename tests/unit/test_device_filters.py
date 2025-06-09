"""
Tests for device filtering functionality.
"""

import pytest
from runlocal_hub.devices.filters import DeviceFilters


class TestDeviceFilters:
    """Test cases for DeviceFilters validation logic."""

    def test_empty_filters_creation(self):
        """Test creating DeviceFilters with default None values."""
        filters = DeviceFilters()

        assert filters.device_name is None
        assert filters.ram_min is None
        assert filters.ram_max is None
        assert filters.soc is None
        assert filters.year_min is None
        assert filters.year_max is None
        assert filters.os is None
        assert filters.compute_units is None

    def test_valid_filters_creation(self):
        """Test creating DeviceFilters with all valid values."""
        filters = DeviceFilters(
            device_name="iPhone 15 Pro",
            ram_min=4,
            ram_max=16,
            soc="A17 Pro",
            year_min=2020,
            year_max=2024,
            os="iOS",
            compute_units=["CPU", "GPU", "ANE"],
        )

        assert filters.device_name == "iPhone 15 Pro"
        assert filters.ram_min == 4
        assert filters.ram_max == 16
        assert filters.soc == "A17 Pro"
        assert filters.year_min == 2020
        assert filters.year_max == 2024
        assert filters.os == "iOS"
        assert filters.compute_units == ["CPU", "GPU", "ANE"]

    def test_negative_ram_validation(self):
        """Test that negative RAM values raise errors."""
        with pytest.raises(ValueError, match="ram_min must be non-negative"):
            DeviceFilters(ram_min=-1)

        with pytest.raises(ValueError, match="ram_max must be non-negative"):
            DeviceFilters(ram_max=-5)

    def test_ram_range_validation(self):
        """Test RAM range validation."""
        with pytest.raises(ValueError, match="ram_min cannot be greater than ram_max"):
            DeviceFilters(ram_min=16, ram_max=8)

        # Equal values should be valid
        filters = DeviceFilters(ram_min=8, ram_max=8)
        assert filters.ram_min == 8
        assert filters.ram_max == 8

    def test_year_range_validation(self):
        """Test year range validation."""
        with pytest.raises(
            ValueError, match="year_min cannot be greater than year_max"
        ):
            DeviceFilters(year_min=2025, year_max=2020)

        # Equal values should be valid
        filters = DeviceFilters(year_min=2022, year_max=2022)
        assert filters.year_min == 2022
        assert filters.year_max == 2022
