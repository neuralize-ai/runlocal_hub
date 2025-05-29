"""
Custom exceptions for the RunLocal API client.
"""


class RunLocalError(Exception):
    """Base exception for RunLocal client."""
    pass


class AuthenticationError(RunLocalError):
    """API key or authentication issues."""
    pass


class ModelNotFoundError(RunLocalError):
    """Model ID not found or not accessible."""
    pass


class DeviceNotAvailableError(RunLocalError):
    """No devices match the specified criteria."""
    pass


class JobTimeoutError(RunLocalError):
    """Job didn't complete within the specified timeout."""
    pass


class TensorError(RunLocalError):
    """Issues with tensor upload/download operations."""
    pass


class UploadError(RunLocalError):
    """Model upload failed."""
    pass


class APIError(RunLocalError):
    """Generic API error with status code."""
    
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code