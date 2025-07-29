"""
Decorators for error handling and common functionality.
"""

import functools
import logging
import random
import time
from typing import Callable, Optional

import requests

from ..exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    ModelNotFoundError,
    NetworkError,
    RunLocalError,
    TensorError,
    UploadError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator to handle API errors and convert them to specific exceptions.

    Provides comprehensive error mapping and context enrichment for better
    debugging and user experience.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function with enhanced error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (AuthenticationError, ValidationError, ConfigurationError):
            # Re-raise already specific errors as-is
            raise
        except APIError as e:
            # Enhance API errors with more specific exceptions
            raise _enhance_api_error(e, func.__name__)
        except requests.exceptions.RequestException as e:
            # Convert requests exceptions to our custom exceptions
            raise _convert_requests_error(e, func.__name__)
        except (ValueError, TypeError) as e:
            # Convert validation errors
            raise ValidationError(
                f"Invalid input in {func.__name__}: {str(e)}",
                parameter=_extract_parameter_from_error(str(e)),
            )
        except FileNotFoundError as e:
            # Handle file-related errors
            raise UploadError(
                f"File not found: {str(e)}",
                file_path=str(e).split("'")[1] if "'" in str(e) else None,
            )
        except Exception as e:
            # Catch any other exceptions and wrap them with context
            if isinstance(e, RunLocalError):
                raise

            logger.error(
                f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True
            )
            raise RunLocalError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                details={"function": func.__name__, "original_error": type(e).__name__},
            )

    return wrapper


def _enhance_api_error(api_error: APIError, function_name: str) -> RunLocalError:
    """
    Convert APIError to more specific exception based on status code and context.

    Args:
        api_error: Original API error
        function_name: Name of the function that failed

    Returns:
        More specific exception
    """
    status_code = api_error.status_code
    message = str(api_error)
    endpoint = api_error.endpoint

    if status_code is not None:
        # Authentication errors
        if status_code == 401:
            return AuthenticationError(
                "Invalid API key or authentication failed. Please check your RUNLOCAL_API_KEY.",
            )

        # Not found errors
        elif status_code == 404:
            if any(keyword in message.lower() for keyword in ["model", "upload"]):
                return ModelNotFoundError(
                    f"Model not found: {message}",
                    model_id=_extract_model_id_from_error(message),
                )
            elif "tensor" in message.lower():
                return TensorError(
                    f"Tensor not found: {message}",
                    tensor_id=_extract_tensor_id_from_error(message),
                    operation="download",
                )
            else:
                return APIError(
                    f"Resource not found: {message}",
                    status_code=status_code,
                    endpoint=endpoint,
                )

        # Bad request errors
        elif status_code == 400:
            if "tensor" in message.lower():
                return TensorError(
                    f"Invalid tensor request: {message}",
                    operation=_extract_operation_from_context(function_name),
                )
            elif "upload" in message.lower():
                return UploadError(f"Upload failed: {message}")
            else:
                return ValidationError(f"Invalid request: {message}")

        # Server errors
        elif status_code >= 500:
            return APIError(
                f"Server error (please try again later): {message}",
                status_code=status_code,
                endpoint=endpoint,
            )

        # Rate limiting
        elif status_code == 429:
            return APIError(
                f"Rate limit exceeded. Please wait and try again: {message}",
                status_code=status_code,
                endpoint=endpoint,
            )

    # Default to generic API error with enhanced context
    return APIError(
        f"API error in {function_name}: {message}",
        status_code=status_code,
        endpoint=endpoint,
    )


def _convert_requests_error(
    requests_error: requests.exceptions.RequestException, function_name: str
) -> RunLocalError:
    """
    Convert requests exception to appropriate RunLocal exception.

    Args:
        requests_error: Original requests exception
        function_name: Name of the function that failed

    Returns:
        Appropriate RunLocal exception
    """
    if hasattr(requests_error, "response") and requests_error.response is not None:
        response = requests_error.response
        endpoint = response.url

        # Try to get response details
        response_data = {}
        try:
            response_data = response.json()
        except:
            response_data = {"text": response.text}

        return APIError(
            f"HTTP {response.status_code} error in {function_name}: {response.text}",
            status_code=response.status_code,
            response_data=response_data,
            endpoint=endpoint,
        )

    # Network-level errors
    if isinstance(requests_error, requests.exceptions.ConnectionError):
        return NetworkError(
            f"Connection failed in {function_name}: {str(requests_error)}",
            endpoint=getattr(requests_error.request, "url", None)
            if hasattr(requests_error, "request")
            else None,
        )
    elif isinstance(requests_error, requests.exceptions.Timeout):
        return NetworkError(
            f"Request timeout in {function_name}: {str(requests_error)}",
            endpoint=getattr(requests_error.request, "url", None)
            if hasattr(requests_error, "request")
            else None,
        )
    else:
        return NetworkError(f"Network error in {function_name}: {str(requests_error)}")


def _extract_model_id_from_error(message: str) -> Optional[str]:
    """Extract model ID from error message if present."""
    # Look for patterns like "model abc123" or "upload_id=abc123"
    import re

    patterns = [
        r"model[:\s]+([a-f0-9]{32})",
        r"upload_id[=\s]+([a-f0-9]{32})",
        r"([a-f0-9]{32})",
    ]

    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            return match.group(1)
    return None


def _extract_tensor_id_from_error(message: str) -> Optional[str]:
    """Extract tensor ID from error message if present."""
    # Look for tensor ID patterns
    import re

    pattern = r"tensor[:\s]+([a-f0-9]{64})"
    match = re.search(pattern, message.lower())
    return match.group(1) if match else None


def _extract_operation_from_context(function_name: str) -> str:
    """Extract operation type from function name."""
    if "upload" in function_name.lower():
        return "upload"
    elif "download" in function_name.lower():
        return "download"
    elif "serialize" in function_name.lower():
        return "serialize"
    elif "deserialize" in function_name.lower():
        return "deserialize"
    else:
        return "unknown"


def _extract_parameter_from_error(error_message: str) -> Optional[str]:
    """Extract parameter name from validation error message."""
    # Look for patterns like "parameter 'xyz'" or "argument xyz"
    import re

    patterns = [
        r"parameter\s+'([^']+)'",
        r"argument\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"'([a-zA-Z_][a-zA-Z0-9_]*)'.*required",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_message)
        if match:
            return match.group(1)
    return None


def with_error_context(context: str):
    """
    Decorator to add context information to errors.

    Args:
        context: Context description for errors
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RunLocalError as e:
                # Add context to existing RunLocal errors
                if hasattr(e, "details"):
                    e.details["context"] = context
                else:
                    e.details = {"context": context}
                raise
            except Exception as e:
                # Wrap other exceptions with context
                raise RunLocalError(
                    f"Error in {context}: {str(e)}",
                    details={"context": context, "original_error": type(e).__name__},
                )

        return wrapper

    return decorator


def with_retry(
    max_attempts: int = 5,
    base_delay: float = 10.0,
    max_delay: float = 300.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
):
    """
    Decorator to add retry logic for transient API errors.

    Retries network errors, server errors (5xx), and rate limiting (429).
    Does NOT retry authentication errors, validation errors, or not found errors.

    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        backoff_factor: Exponential backoff multiplier (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this is a retryable error
                    if not _is_retryable_error(e):
                        raise

                    # Update retry count for NetworkError
                    if isinstance(e, NetworkError):
                        e.retry_count = attempt + 1

                    # Don't sleep after the last attempt
                    if attempt < max_attempts - 1:
                        delay = _calculate_delay(
                            attempt, base_delay, max_delay, backoff_factor, jitter
                        )
                        logger.debug(
                            f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s delay due to: {str(e)}"
                        )
                        time.sleep(delay)
                    else:
                        # All retries exhausted, raise the last exception
                        logger.error(
                            f"All {max_attempts} retry attempts failed for {func.__name__}: {str(e)}"
                        )
                        raise

        return wrapper

    return decorator


def _is_retryable_error(error: Exception) -> bool:
    """
    Check if an error should be retried.

    Args:
        error: Exception to check

    Returns:
        True if the error should be retried, False otherwise
    """
    # Network errors are always retryable
    if isinstance(error, NetworkError):
        return True

    # API errors - check status code
    if isinstance(error, APIError):
        status_code = error.status_code
        if status_code is not None:
            # Server errors (5xx) are retryable
            if 500 <= status_code < 600:
                return True
            # Rate limiting is retryable
            if status_code == 429:
                return True

    # These errors should NOT be retried
    if isinstance(error, (AuthenticationError, ValidationError, ModelNotFoundError)):
        return False

    # Specific requests exceptions that are retryable
    if isinstance(error, requests.exceptions.RequestException):
        # Connection and timeout errors are retryable
        if isinstance(
            error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ):
            return True
        # HTTP errors - check status code
        if hasattr(error, "response") and error.response is not None:
            status_code = error.response.status_code
            return 500 <= status_code < 600 or status_code == 429

    return False


def _calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    backoff_factor: float,
    jitter: bool,
) -> float:
    """
    Calculate delay for retry attempt with exponential backoff and optional jitter.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Exponential backoff multiplier
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    # Calculate exponential backoff delay
    delay = base_delay * (backoff_factor**attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter to prevent thundering herd
    if jitter:
        # Add ±25% random jitter
        jitter_amount = delay * 0.25
        delay += random.uniform(-jitter_amount, jitter_amount)
        # Ensure delay is not negative
        delay = max(0.1, delay)

    return delay
