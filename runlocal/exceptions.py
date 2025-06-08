"""
Custom exceptions for the RunLocal API client.
"""

from typing import Dict, List, Optional


class RunLocalError(Exception):
    """
    Base exception for RunLocal client.

    All RunLocal exceptions inherit from this class, making it easy to catch
    any RunLocal-specific error.
    """

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class AuthenticationError(RunLocalError):
    """
    API key or authentication issues.

    Raised when:
    - API key is missing or invalid
    - Authentication fails
    - Access is denied
    """

    pass


class ModelNotFoundError(RunLocalError):
    """
    Model ID not found or not accessible.

    Raised when:
    - Model ID doesn't exist
    - Model doesn't belong to the user
    - Model was deleted
    """

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.model_id = model_id
        self.available_models = available_models or []


class DeviceNotAvailableError(RunLocalError):
    """
    No devices match the specified criteria.

    Raised when:
    - Device filters are too restrictive
    - No devices are available for the model
    - All matching devices are busy
    """

    def __init__(
        self,
        message: str,
        filters_used: Optional[Dict] = None,
        available_count: int = 0,
    ):
        super().__init__(message)
        self.filters_used = filters_used or {}
        self.available_count = available_count


class JobTimeoutError(RunLocalError):
    """
    Job didn't complete within the specified timeout.

    Raised when:
    - Benchmark or prediction takes too long
    - Network issues cause delays
    - Jobs are stuck in queue
    """

    def __init__(
        self,
        message: str,
        timeout: int = 0,
        completed_jobs: int = 0,
        total_jobs: int = 0,
    ):
        super().__init__(message)
        self.timeout = timeout
        self.completed_jobs = completed_jobs
        self.total_jobs = total_jobs


class TensorError(RunLocalError):
    """
    Issues with tensor upload/download operations.

    Raised when:
    - Tensor serialization/deserialization fails
    - Network errors during upload/download
    - Invalid tensor formats
    """

    def __init__(
        self,
        message: str,
        tensor_id: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message)
        self.tensor_id = tensor_id
        self.operation = operation  # 'upload', 'download', 'serialize', etc.


class UploadError(RunLocalError):
    """
    Model upload failed.

    Raised when:
    - File is too large
    - Invalid model format
    - Network issues during upload
    - Server rejects upload
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
    ):
        super().__init__(message)
        self.file_path = file_path
        self.file_size = file_size


class ValidationError(RunLocalError):
    """
    Input validation failed.

    Raised when:
    - Invalid parameters provided
    - Missing required arguments
    - Conflicting options specified
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        expected: Optional[str] = None,
        got: Optional[str] = None,
    ):
        super().__init__(message)
        self.parameter = parameter
        self.expected = expected
        self.got = got


class APIError(RunLocalError):
    """
    Generic API error with status code and response details.

    Raised when:
    - HTTP errors occur
    - API returns error responses
    - Unexpected server responses
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.endpoint = endpoint


class NetworkError(RunLocalError):
    """
    Network-related errors.

    Raised when:
    - Connection timeouts
    - DNS resolution fails
    - Network is unreachable
    """

    def __init__(
        self, message: str, endpoint: Optional[str] = None, retry_count: int = 0
    ):
        super().__init__(message)
        self.endpoint = endpoint
        self.retry_count = retry_count


class ConfigurationError(RunLocalError):
    """
    Configuration or setup errors.

    Raised when:
    - Environment variables are missing
    - Invalid configuration values
    - Setup requirements not met
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        super().__init__(message)
        self.config_key = config_key
        self.suggestion = suggestion

