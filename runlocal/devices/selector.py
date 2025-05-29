"""
Device selection and filtering logic.
"""
from typing import List, Optional

from ..exceptions import DeviceNotAvailableError, ModelNotFoundError
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
    def list_all_devices(self, model_id: Optional[str] = None) -> List[DeviceUsage]:
        """
        Get a list of available devices for benchmarking.
        
        Args:
            model_id: Optional ID of a model to get compatible devices
            
        Returns:
            List of available devices with their compute units
            
        Raises:
            ModelNotFoundError: If the model ID is not found
        """
        endpoint = "/devices/benchmark"
        if model_id:
            endpoint += f"?upload_id={model_id}"
        
        response = self.http_client.get(endpoint)
        
        # Filter to only include devices with compute units if model_id is provided
        if model_id:
            return [
                DeviceUsage(
                    device=device_usage["device"],
                    compute_units=device_usage["compute_units"],
                )
                for device_usage in response
                if device_usage.get("compute_units") and len(device_usage.get("compute_units", [])) > 0
            ]
        
        # Return all devices if no model_id
        return [
            DeviceUsage(
                device=device_usage["device"],
                compute_units=device_usage.get("compute_units", []),
            )
            for device_usage in response
        ]
    
    def select_devices(
        self,
        model_id: str,
        filters: DeviceFilters,
        count: Optional[int] = 1,
        user_models: Optional[List[str]] = None,
    ) -> List[DeviceUsage]:
        """
        Select devices based on filter criteria.
        
        Args:
            model_id: ID of the model to get compatible devices for
            filters: Device filtering criteria
            count: Number of devices to select (default: 1, 0 = all matching devices)
            user_models: Optional list of user's models for validation
            
        Returns:
            List of matching devices
            
        Raises:
            ValueError: If model_id doesn't belong to user
            DeviceNotAvailableError: If no devices match the criteria
        """
        # Validate model_id if user_models provided
        if user_models is not None and model_id not in user_models:
            raise ValueError(
                f"model_id '{model_id}' does not correspond to your available models"
            )
        
        # Get all available devices for this model
        devices = self.list_all_devices(model_id=model_id)
        
        # Apply filters
        filtered_devices = self._apply_filters(devices, filters)
        
        # Check if any devices matched
        if not filtered_devices:
            raise DeviceNotAvailableError(
                "No devices match the specified criteria. "
                "Try relaxing your filter conditions."
            )
        
        # Apply count logic: 0 means all devices, otherwise limit to count
        if count is not None and count > 0 and len(filtered_devices) > count:
            filtered_devices = filtered_devices[:count]
        
        return filtered_devices
    
    def select_device(
        self,
        model_id: str,
        filters: DeviceFilters,
        user_models: Optional[List[str]] = None,
    ) -> Optional[DeviceUsage]:
        """
        Select a single device based on filter criteria.
        
        Args:
            model_id: ID of the model to get compatible devices for
            filters: Device filtering criteria
            user_models: Optional list of user's models for validation
            
        Returns:
            Single matching device or None if no matches
            
        Raises:
            ValueError: If model_id doesn't belong to user
        """
        devices = self.select_devices(
            model_id=model_id,
            filters=filters,
            count=1,
            user_models=user_models,
        )
        
        return devices[0] if devices else None
    
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
                d for d in filtered
                if filters.device_name.lower() in d.device.Name.lower()
            ]
        
        # Filter by SoC (substring match)
        if filters.soc is not None:
            filtered = [
                d for d in filtered
                if filters.soc.lower() in d.device.Soc.lower()
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
        
        return filtered