"""
Tests for device selection functionality.
"""

import pytest
from unittest.mock import Mock, patch
from runlocal_hub.devices.selector import DeviceSelector
from runlocal_hub.devices.filters import DeviceFilters
from runlocal_hub.models.device import Device, DeviceUsage
from runlocal_hub.exceptions import DeviceNotAvailableError, ModelNotFoundError


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    return Mock()


@pytest.fixture
def device_selector(mock_http_client):
    """Create a DeviceSelector instance with mock HTTP client."""
    return DeviceSelector(mock_http_client)


@pytest.fixture
def sample_devices():
    """Create sample devices for testing."""
    return [
        Device(
            Name="iPhone 15 Pro",
            Year=2023,
            Soc="A17 Pro",
            Ram=8,
            OS="iOS",
            OSVersion="17.0",
        ),
        Device(
            Name="MacBook Pro M3",
            Year=2023,
            Soc="Apple M3",
            Ram=16,
            OS="macOS",
            OSVersion="14.0",
        ),
        Device(
            Name="iPad Air M2",
            Year=2024,
            Soc="Apple M2",
            Ram=8,
            OS="iPadOS",
            OSVersion="17.0",
        ),
        Device(
            Name="iPhone 14",
            Year=2022,
            Soc="A16 Bionic",
            Ram=6,
            OS="iOS",
            OSVersion="16.0",
        ),
    ]


@pytest.fixture
def sample_device_usage(sample_devices):
    """Create sample DeviceUsage objects for testing."""
    return [
        DeviceUsage(device=sample_devices[0], compute_units=["CPU", "GPU", "ANE"]),
        DeviceUsage(device=sample_devices[1], compute_units=["CPU", "GPU"]),
        DeviceUsage(device=sample_devices[2], compute_units=["CPU", "GPU", "ANE"]),
        DeviceUsage(device=sample_devices[3], compute_units=["CPU", "GPU"]),
    ]


class TestDeviceSelector:
    """Test cases for DeviceSelector functionality."""

    def test_init(self, mock_http_client):
        """Test DeviceSelector initialization."""
        selector = DeviceSelector(mock_http_client)
        assert selector.http_client == mock_http_client

    def test_list_all_devices_filtering(self, device_selector, mock_http_client):
        """Test device listing filters disabled devices and empty compute units."""
        mock_response = [
            {
                "device": {
                    "Name": "iPhone 15 Pro",
                    "Year": 2023,
                    "Soc": "A17 Pro",
                    "Ram": 8,
                    "OS": "iOS",
                    "OSVersion": "17.0",
                    "Disabled": False,
                },
                "compute_units": ["CPU", "GPU", "ANE"],
            },
            {
                "device": {
                    "Name": "Disabled Device",
                    "Year": 2020,
                    "Soc": "Old Chip",
                    "Ram": 4,
                    "OS": "iOS",
                    "OSVersion": "14.0",
                    "Disabled": True,
                },
                "compute_units": ["CPU"],
            },
            {
                "device": {
                    "Name": "No Compute Units",
                    "Year": 2020,
                    "Soc": "Old Chip",
                    "Ram": 4,
                    "OS": "iOS",
                    "OSVersion": "14.0",
                    "Disabled": False,
                },
                "compute_units": [],
            },
        ]
        mock_http_client.get.return_value = mock_response

        with patch(
            "runlocal_hub.devices.selector.handle_api_errors", lambda func: func
        ):
            # Without model_id - returns all non-disabled devices with empty compute units
            devices = device_selector.list_all_devices()
            assert len(devices) == 2
            assert all(d.compute_units == [] for d in devices)

            # With model_id - returns only devices with compute units
            devices = device_selector.list_all_devices(model_id="test-model")
            assert len(devices) == 1
            assert devices[0].device.Name == "iPhone 15 Pro"
            assert devices[0].compute_units == ["CPU", "GPU", "ANE"]

    def test_select_devices_basic_functionality(
        self, device_selector, sample_device_usage
    ):
        """Test basic device selection functionality."""
        with patch.object(
            device_selector, "list_all_devices", return_value=sample_device_usage
        ):
            # No filters, default count=1
            devices = device_selector.select_devices("test-model")
            assert len(devices) == 1
            assert devices[0] in sample_device_usage

            # No filters, count=None (all devices)
            devices = device_selector.select_devices("test-model", count=None)
            assert len(devices) == 4
            assert devices == sample_device_usage

    def test_select_devices_with_filters(self, device_selector, sample_device_usage):
        """Test device selection with single and multiple filters."""
        with patch.object(
            device_selector, "list_all_devices", return_value=sample_device_usage
        ):
            # Single filter
            filters = DeviceFilters(device_name="iPhone")
            devices = device_selector.select_devices(
                "test-model", filters=filters, count=None
            )
            assert len(devices) == 2
            device_names = [d.device.Name for d in devices]
            assert "iPhone 15 Pro" in device_names
            assert "iPhone 14" in device_names

            # Multiple filters (OR logic)
            filters = [
                DeviceFilters(device_name="iPhone 15"),
                DeviceFilters(soc="Apple M3"),
            ]
            devices = device_selector.select_devices(
                "test-model", filters=filters, count=None
            )
            assert len(devices) == 2
            device_names = [d.device.Name for d in devices]
            assert "iPhone 15 Pro" in device_names
            assert "MacBook Pro M3" in device_names

    def test_select_devices_multiple_filters_comprehensive(
        self, device_selector, sample_device_usage
    ):
        """Test comprehensive multiple filter scenarios with OR logic."""
        with patch.object(
            device_selector, "list_all_devices", return_value=sample_device_usage
        ):
            # Test overlapping filters - should not duplicate devices
            filters = [
                DeviceFilters(
                    device_name="iPhone"
                ),  # Matches iPhone 15 Pro and iPhone 14
                DeviceFilters(
                    year_min=2023
                ),  # Matches iPhone 15 Pro, MacBook Pro M3, iPad Air M2
            ]
            devices = device_selector.select_devices(
                "test-model", filters=filters, count=None
            )
            assert len(devices) == 4  # All devices match at least one filter
            device_names = [d.device.Name for d in devices]
            assert "iPhone 15 Pro" in device_names
            assert "iPhone 14" in device_names
            assert "MacBook Pro M3" in device_names
            assert "iPad Air M2" in device_names

            # Test non-overlapping filters
            filters = [
                DeviceFilters(os="macOS"),  # Only MacBook Pro M3
                DeviceFilters(os="iPadOS"),  # Only iPad Air M2
            ]
            devices = device_selector.select_devices(
                "test-model", filters=filters, count=None
            )
            assert len(devices) == 2
            device_names = [d.device.Name for d in devices]
            assert "MacBook Pro M3" in device_names
            assert "iPad Air M2" in device_names

            # Test complex filters with multiple criteria each
            filters = [
                DeviceFilters(device_name="iPhone", ram_min=8),  # Only iPhone 15 Pro
                DeviceFilters(soc="Apple M", year_min=2024),  # Only iPad Air M2
            ]
            devices = device_selector.select_devices(
                "test-model", filters=filters, count=None
            )
            assert len(devices) == 2
            device_names = [d.device.Name for d in devices]
            assert "iPhone 15 Pro" in device_names
            assert "iPad Air M2" in device_names

            # Test filters with compute units
            filters = [
                DeviceFilters(compute_units=["ANE"]),  # iPhone 15 Pro and iPad Air M2
                DeviceFilters(ram_max=6),  # Only iPhone 14
            ]
            devices = device_selector.select_devices(
                "test-model", filters=filters, count=None
            )
            assert len(devices) == 3
            device_names = [d.device.Name for d in devices]
            assert "iPhone 15 Pro" in device_names
            assert "iPad Air M2" in device_names
            assert "iPhone 14" in device_names

            # Verify compute units are filtered correctly for ANE filter
            ane_devices = [d for d in devices if "ANE" in d.compute_units]
            assert len(ane_devices) == 2  # iPhone 15 Pro and iPad Air M2

            # Test three filters
            filters = [
                DeviceFilters(device_name="iPhone 15"),  # iPhone 15 Pro
                DeviceFilters(device_name="MacBook"),  # MacBook Pro M3
                DeviceFilters(device_name="iPad"),  # iPad Air M2
            ]
            devices = device_selector.select_devices(
                "test-model", filters=filters, count=None
            )
            assert len(devices) == 3
            device_names = [d.device.Name for d in devices]
            assert "iPhone 15 Pro" in device_names
            assert "MacBook Pro M3" in device_names
            assert "iPad Air M2" in device_names
            assert "iPhone 14" not in device_names

    def test_select_devices_error_scenarios(self, device_selector, sample_device_usage):
        """Test device selection error scenarios."""
        with patch.object(
            device_selector, "list_all_devices", return_value=sample_device_usage
        ):
            # No matches with single filter
            filters = DeviceFilters(device_name="Android Device")
            with pytest.raises(DeviceNotAvailableError) as exc_info:
                device_selector.select_devices("test-model", filters=filters)
            assert "No devices match the specified criteria" in str(exc_info.value)
            assert "device_name=Android Device" in str(exc_info.value)

            # No matches with multiple filters
            filters = [
                DeviceFilters(device_name="Android"),
                DeviceFilters(soc="Snapdragon"),
            ]
            with pytest.raises(DeviceNotAvailableError) as exc_info:
                device_selector.select_devices("test-model", filters=filters)
            assert "No devices match any of the specified criteria" in str(
                exc_info.value
            )

        # Invalid model ID
        user_models = ["model1", "model2", "model3"]
        with pytest.raises(ModelNotFoundError) as exc_info:
            device_selector.select_devices("invalid-model", user_models=user_models)
        assert "Model 'invalid-model' not found" in str(exc_info.value)

    def test_apply_filters_string_matching(self, device_selector, sample_device_usage):
        """Test string-based filter application (case-insensitive)."""
        # Device name filter (case-insensitive)
        filters = DeviceFilters(device_name="iphone")
        filtered = device_selector._apply_filters(sample_device_usage, filters)
        assert len(filtered) == 2
        device_names = [d.device.Name for d in filtered]
        assert "iPhone 15 Pro" in device_names
        assert "iPhone 14" in device_names

        # SoC filter (case-insensitive)
        filters = DeviceFilters(soc="apple m")
        filtered = device_selector._apply_filters(sample_device_usage, filters)
        assert len(filtered) == 2
        device_names = [d.device.Name for d in filtered]
        assert "MacBook Pro M3" in device_names
        assert "iPad Air M2" in device_names

        # OS filter (case-insensitive)
        filters = DeviceFilters(os="macos")
        filtered = device_selector._apply_filters(sample_device_usage, filters)
        assert len(filtered) == 1
        assert filtered[0].device.Name == "MacBook Pro M3"

    def test_apply_filters_numeric_ranges(self, device_selector, sample_device_usage):
        """Test numeric range filter application."""
        # RAM range filter
        filters = DeviceFilters(ram_min=8)
        filtered = device_selector._apply_filters(sample_device_usage, filters)
        assert len(filtered) == 3  # Excludes iPhone 14 (6GB)
        device_names = [d.device.Name for d in filtered]
        assert "iPhone 14" not in device_names

        # Year range filter
        filters = DeviceFilters(year_min=2023, year_max=2023)
        filtered = device_selector._apply_filters(sample_device_usage, filters)
        assert len(filtered) == 2  # iPhone 15 Pro and MacBook Pro M3
        device_names = [d.device.Name for d in filtered]
        assert "iPhone 15 Pro" in device_names
        assert "MacBook Pro M3" in device_names

    def test_apply_filters_compute_units(self, device_selector, sample_device_usage):
        """Test compute units filter application."""
        # Filter for specific compute unit
        filters = DeviceFilters(compute_units=["ANE"])
        filtered = device_selector._apply_filters(sample_device_usage, filters)
        assert len(filtered) == 2  # iPhone 15 Pro and iPad Air M2 have ANE
        for device_usage in filtered:
            assert device_usage.compute_units == ["ANE"]  # Only ANE remains

        # Filter for multiple compute units
        filters = DeviceFilters(compute_units=["CPU", "GPU"])
        filtered = device_selector._apply_filters(sample_device_usage, filters)
        assert len(filtered) == 4  # All devices have CPU and GPU
        for device_usage in filtered:
            assert set(device_usage.compute_units).issubset({"CPU", "GPU"})

    def test_apply_filters_combined(self, device_selector, sample_device_usage):
        """Test applying multiple filters together."""
        filters = DeviceFilters(
            device_name="iPhone", ram_min=7, year_min=2023, os="iOS"
        )

        filtered = device_selector._apply_filters(sample_device_usage, filters)

        assert len(filtered) == 1
        assert filtered[0].device.Name == "iPhone 15 Pro"

    def test_utility_methods(self, device_selector):
        """Test utility methods for filter details and display."""
        # Test filter details extraction
        filters = DeviceFilters(
            device_name="iPhone",
            ram_min=8,
            ram_max=16,
            year_min=2020,
            soc="A17",
            os="iOS",
            compute_units=["CPU", "GPU"],
        )
        details = device_selector._get_filter_details(filters)
        expected = {
            "device_name": "iPhone",
            "soc": "A17",
            "ram_min": "8GB",
            "ram_max": "16GB",
            "year_min": 2020,
            "os": "iOS",
            "compute_units": ["CPU", "GPU"],
        }
        assert details == expected

        # Test empty filter details
        empty_filters = DeviceFilters()
        assert device_selector._get_filter_details(empty_filters) == {}

    @patch("runlocal_hub.devices.selector.random.sample")
    def test_random_selection_with_count(
        self, mock_sample, device_selector, sample_device_usage
    ):
        """Test that devices are randomly selected when count < available devices."""
        mock_sample.return_value = sample_device_usage[:2]

        with patch.object(
            device_selector, "list_all_devices", return_value=sample_device_usage
        ):
            devices = device_selector.select_devices("test-model", count=2)

        mock_sample.assert_called_once_with(sample_device_usage, 2)
        assert len(devices) == 2

    @patch("runlocal_hub.devices.selector.Console")
    def test_display_selected_devices(
        self, mock_console_class, device_selector, sample_device_usage
    ):
        """Test displaying selected devices."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        device_selector.display_selected_devices(sample_device_usage[:2])

        mock_console_class.assert_called_once()
        mock_console.print.assert_called_once()
