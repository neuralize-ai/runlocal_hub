import pytest
import os
from unittest.mock import Mock

os.environ["RUNLOCAL_API_KEY"] = "test_api_key"


@pytest.fixture
def mock_http_client():
    """Mock HTTPClient for testing"""
    mock = Mock()
    mock.base_url = "https://neuralize-bench.com"
    mock.headers = {"X-API-KEY": "test_api_key"}
    return mock


@pytest.fixture
def sample_user_info():
    """Sample user info response"""
    return {
        "Username": "test_user",
        "Email": "test@example.com",
        "UploadIds": ["model_123", "model_456"],
        "TotalComputeTime": 3600,
        "TotalRequests": 100,
    }
