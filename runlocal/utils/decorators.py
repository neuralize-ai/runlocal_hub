"""
Decorators for error handling and common functionality.
"""
import functools
from typing import Callable

import requests

from ..exceptions import (
    APIError,
    AuthenticationError,
    ModelNotFoundError,
    RunLocalError,
)


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator to handle API errors and convert them to specific exceptions.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError:
            # Re-raise authentication errors as-is
            raise
        except APIError as e:
            # Handle specific API errors
            if e.status_code == 404:
                # Try to extract more context from the error message
                if "model" in str(e).lower() or "upload" in str(e).lower():
                    raise ModelNotFoundError(str(e))
            raise
        except requests.exceptions.RequestException as e:
            # Handle network errors
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif e.response.status_code == 404:
                    raise ModelNotFoundError(f"Resource not found: {e.response.url}")
                else:
                    raise APIError(
                        f"API request failed: {str(e)}", 
                        status_code=e.response.status_code
                    )
            else:
                raise RunLocalError(f"Network error: {str(e)}")
        except Exception as e:
            # Catch any other exceptions and wrap them
            if not isinstance(e, RunLocalError):
                raise RunLocalError(f"Unexpected error: {str(e)}")
            raise
    
    return wrapper