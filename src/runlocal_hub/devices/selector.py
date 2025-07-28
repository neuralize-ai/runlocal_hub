"""
Device selection and filtering logic.
"""

import random
from typing import List, Optional, Union

from rich.console import Console
from rich.table import Table

from runlocal_hub.models.benchmark import Framework

from ..exceptions import DeviceNotAvailableError
from ..http import HTTPClient
from ..models import DeviceUsage
from ..utils.decorators import handle_api_errors
from .filters import DeviceFilters


class DeviceSelector:
    """
    Handles device selection and filtering operations.
    """

    def __init__(self, http_client: HTTPClient):
        """
        Initialize the device selector.

        Args:
            http_client: HTTP client for API requests
        """
        self.http_client = http_client

    @handle_api_errors
    def list_all_devices(
        self, model_id: Optional[str] = None, framework: Optional[Framework] = None
    ) -> List[DeviceUsage]:
        """
        Get a list of available devices for benchmarking.

        Args:
            model_id: Optional ID of a model to get compatible devices
            framework: Optional manual override of runtime framework

        Returns:
            List of available devices with their compute units

        Raises:
            ModelNotFoundError: If the model ID is not found
        """
        endpoint = "/devices/benchmark"
        if model_id is not None:
            endpoint += f"?upload_id={model_id}"

        if framework is not None:
            endpoint += f"&framework={framework.value}"

        response = self.http_client.get(endpoint)

        # Filter to only include devices with compute units if model_id is provided
        if model_id:
            return [
                DeviceUsage(
                    device=device_usage["device"],
                    compute_units=device_usage["compute_units"],
                    native_device_id=device_usage["native_device_id"],
                )
                for device_usage in response
                if device_usage.get("device")
                and device_usage.get("device").get("Disabled") is False
                and device_usage.get("compute_units")
                and len(device_usage.get("compute_units", [])) > 0
            ]

        # Return all devices if no model_id
        return [
            DeviceUsage(
                device=device_usage["device"],
                compute_units=[],
                native_device_id=device_usage["native_device_id"],
            )
            for device_usage in response
            if device_usage.get("device")
            and device_usage.get("device").get("Disabled") is False
        ]

    def select_devices(
        self,
        model_id: str,
        framework: Optional[Framework] = None,
        filters: Optional[Union[DeviceFilters, List[DeviceFilters]]] = None,
        count: Optional[int] = 1,
        user_models: Optional[List[str]] = None,
    ) -> List[DeviceUsage]:
        """
        Select devices based on filter criteria.

        Args:
            model_id: ID of the model to get compatible devices for
            filters: Device filtering criteria. Can be:
                - None: No filtering (all available devices)
                - DeviceFilters: Single filter set
                - List[DeviceFilters]: Multiple filter sets with OR logic (union)
            count: Number of devices to select (default: 1, None = all matching devices)
            user_models: Optional list of user's models for validation

        Returns:
            List of matching devices

        Raises:
            ValueError: If model_id doesn't belong to user
            DeviceNotAvailableError: If no devices match the criteria
        """
        # Validate model_id if user_models provided
        if user_models is not None and model_id not in user_models:
            from ..exceptions import ModelNotFoundError

            available_models_str = f"Available models: {', '.join(user_models[:5])}"
            if len(user_models) > 5:
                available_models_str += f" ... and {len(user_models) - 5} more"

            raise ModelNotFoundError(
                f"Model '{model_id}' not found in your account. {available_models_str}",
                model_id=model_id,
                available_models=user_models,
            )

        # Get all available devices for this model
        all_devices = self.list_all_devices(model_id=model_id, framework=framework)

        # Handle different filter types
        if filters is None:
            # No filtering - use all devices
            filtered_devices = all_devices
        elif isinstance(filters, DeviceFilters):
            # Single filter set
            filtered_devices = self._apply_filters(all_devices, filters)

            # Check if any devices matched
            if not filtered_devices:
                # Create helpful error message with filter details
                filter_details = self._get_filter_details(filters)
                filter_description = ", ".join(
                    [f"{k}={v}" for k, v in filter_details.items()]
                )

                error_message = (
                    f"No devices match the specified criteria ({filter_description}). "
                )
                error_message += (
                    f"Found {len(all_devices)} total devices for this model. "
                )
                error_message += "Try relaxing your filter conditions."

                raise DeviceNotAvailableError(
                    error_message,
                    filters_used=filter_details,
                    available_count=len(all_devices),
                )
        else:
            # Multiple filter sets with OR logic
            seen_device_ids = set()
            filtered_devices = []

            for filter_set in filters:
                # Apply this filter set
                matches = self._apply_filters(all_devices, filter_set)

                # Add unique devices to result
                for device_usage in matches:
                    if device_usage.native_device_id not in seen_device_ids:
                        seen_device_ids.add(device_usage.native_device_id)
                        filtered_devices.append(device_usage)

            # Check if any devices matched
            if not filtered_devices:
                # Create helpful error message with all filter details
                all_filter_details = []
                for i, filter_set in enumerate(filters):
                    filter_details = self._get_filter_details(filter_set)
                    if filter_details:
                        filter_description = ", ".join(
                            [f"{k}={v}" for k, v in filter_details.items()]
                        )
                        all_filter_details.append(
                            f"Filter {i + 1}: {filter_description}"
                        )

                error_message = (
                    "No devices match any of the specified criteria:\n"
                    + "\n".join(all_filter_details)
                    + "\n"
                )
                error_message += (
                    f"Found {len(all_devices)} total devices for this model. "
                )
                error_message += "Try relaxing your filter conditions."

                raise DeviceNotAvailableError(
                    error_message,
                    filters_used={"filters": all_filter_details},
                    available_count=len(all_devices),
                )

        # Apply count logic: None means all devices, otherwise limit to count
        if count is not None and len(filtered_devices) > count:
            # Randomly select 'count' devices from the filtered list
            filtered_devices = random.sample(filtered_devices, count)

        return filtered_devices

    def _get_filter_details(self, filters: DeviceFilters) -> dict:
        """
        Extract filter details for error messages.

        Args:
            filters: Device filters to extract details from

        Returns:
            Dictionary of filter details
        """
        filter_details = {}
        if filters.device_name:
            filter_details["device_name"] = filters.device_name
        if filters.soc:
            filter_details["soc"] = filters.soc
        if filters.ram_min:
            filter_details["ram_min"] = f"{filters.ram_min}GB"
        if filters.ram_max:
            filter_details["ram_max"] = f"{filters.ram_max}GB"
        if filters.year_min:
            filter_details["year_min"] = filters.year_min
        if filters.year_max:
            filter_details["year_max"] = filters.year_max
        if filters.os:
            filter_details["os"] = filters.os
        if filters.compute_units:
            filter_details["compute_units"] = filters.compute_units
        return filter_details

    def _apply_filters(
        self,
        devices: List[DeviceUsage],
        filters: DeviceFilters,
    ) -> List[DeviceUsage]:
        """
        Apply filter criteria to a list of devices.

        Args:
            devices: List of devices to filter
            filters: Filter criteria

        Returns:
            Filtered list of devices
        """
        filtered = devices

        # Filter by device name (substring match)
        if filters.device_name is not None:
            filtered = [
                d
                for d in filtered
                if filters.device_name.lower() in d.device.Name.lower()
            ]

        # Filter by SoC (substring match)
        if filters.soc is not None:
            filtered = [
                d for d in filtered if filters.soc.lower() in d.device.Soc.lower()
            ]

        # Filter by RAM range
        if filters.ram_min is not None:
            filtered = [d for d in filtered if d.device.Ram >= filters.ram_min]

        if filters.ram_max is not None:
            filtered = [d for d in filtered if d.device.Ram <= filters.ram_max]

        # Filter by year range
        if filters.year_min is not None:
            filtered = [d for d in filtered if d.device.Year >= filters.year_min]

        if filters.year_max is not None:
            filtered = [d for d in filtered if d.device.Year <= filters.year_max]

        # Filter by OS (substring match)
        if filters.os is not None:
            filtered = [
                d for d in filtered if filters.os.lower() in d.device.OS.lower()
            ]

        # Filter by compute units - only keep specified compute units
        if filters.compute_units is not None:
            new_filtered = []
            for device_usage in filtered:
                # Keep only compute units that are in the filter
                matching_compute_units = [
                    cu
                    for cu in device_usage.compute_units
                    if cu in filters.compute_units
                ]

                # Only include device if it has at least one matching compute unit
                if matching_compute_units:
                    new_filtered.append(
                        DeviceUsage(
                            device=device_usage.device,
                            compute_units=matching_compute_units,
                            native_device_id=device_usage.native_device_id,
                        )
                    )

            filtered = new_filtered

        return filtered

    def display_selected_devices(self, devices: List[DeviceUsage]) -> None:
        """
        Display selected devices in a nice format using rich.

        Args:
            devices: List of selected devices to display
        """
        console = Console()

        table = Table(
            title=f"Selected [cyan]{len(devices)}[/cyan] device(s)",
            title_style="bold",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Device")
        table.add_column("Year", justify="center")
        table.add_column("SoC", style="cyan")
        table.add_column("RAM", style="dim", justify="right")
        table.add_column("OS", style="dim")
        table.add_column("Compute Units", style="green")

        for device in devices:
            table.add_row(
                device.device.Name,
                str(device.device.Year),
                device.device.Soc,
                f"{device.device.Ram}GB",
                f"{device.device.OS} {device.device.OSVersion}",
                ", ".join(device.compute_units),
            )

        console.print(table)

        print("")
