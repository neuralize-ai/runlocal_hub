import pytest
from unittest.mock import Mock, patch

from runlocal_hub.client import RunLocalClient
from runlocal_hub.exceptions import ConfigurationError


class TestRunLocalClient:
    """Test cases for RunLocalClient"""

    def test_client_initialization(self):
        """Test client initialization with success and failure cases"""
        # Success case with API key
        with patch.dict("os.environ", {"RUNLOCAL_API_KEY": "test_key"}):
            client = RunLocalClient()
            assert client.http_client.api_key == "test_key"
            assert client.http_client.base_url == "https://neuralize-bench.com"
            assert hasattr(client, "device_selector")
            assert hasattr(client, "tensor_handler")
            assert hasattr(client, "job_poller")

        # Failure case without API key
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                RunLocalClient()
            assert "RUNLOCAL_API_KEY" in str(exc_info.value)

    @patch("runlocal_hub.client.HTTPClient")
    def test_health_and_user_endpoints(self, mock_http_class, sample_user_info):
        """Test basic API endpoint calls"""
        mock_http = Mock()
        mock_http.get.side_effect = [
            {"status": "healthy", "version": "1.0.0"},  # health_check response
            sample_user_info,  # get_user_info response
        ]
        mock_http_class.return_value = mock_http

        with patch.dict("os.environ", {"RUNLOCAL_API_KEY": "test_key"}):
            client = RunLocalClient()

            # Test health check
            health_result = client.health_check()
            assert health_result == {"status": "healthy", "version": "1.0.0"}

            # Test user info
            user_result = client.get_user_info()
            assert user_result == sample_user_info

            # Simple verification of endpoint calls
            mock_http.get.assert_any_call("/users/health")
            mock_http.get.assert_any_call("/users")
            assert mock_http.get.call_count == 2

    def test_component_wiring(self):
        """Test that all components are properly wired together"""
        with patch.dict("os.environ", {"RUNLOCAL_API_KEY": "test_key"}):
            client = RunLocalClient()

        # Test actual component integration, not mocking
        assert hasattr(client, "device_selector")
        assert hasattr(client, "tensor_handler")
        assert hasattr(client, "job_poller")
        assert hasattr(client, "http_client")

        # Verify components have the http_client reference
        assert hasattr(client.device_selector, "http_client")
        assert hasattr(client.tensor_handler, "http_client")
        assert hasattr(client.job_poller, "http_client")

    @patch("runlocal_hub.client.HTTPClient")
    def test_configuration(self, mock_http_class):
        """Test API key and debug mode configuration"""
        mock_http = Mock()
        mock_http_class.return_value = mock_http

        # Test debug mode
        with patch.dict("os.environ", {"RUNLOCAL_API_KEY": "test_key"}):
            client = RunLocalClient(debug=True)
            assert client.debug is True
            mock_http_class.assert_called_once_with(
                base_url="https://neuralize-bench.com", api_key="test_key", debug=True
            )

        # Test configuration error details
        mock_http_class.reset_mock()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                RunLocalClient()

            error = exc_info.value
            assert error.config_key == "RUNLOCAL_API_KEY"
            assert error.suggestion is not None
            assert "export RUNLOCAL_API_KEY" in error.suggestion

    @patch("runlocal_hub.client.HTTPClient")
    def test_error_propagation(self, mock_http_class):
        """Test that errors from HTTP client are properly propagated"""
        mock_http = Mock()
        mock_http.get.side_effect = Exception("HTTP error")
        mock_http_class.return_value = mock_http

        with patch.dict("os.environ", {"RUNLOCAL_API_KEY": "test_key"}):
            client = RunLocalClient()

            # Test that HTTP errors are propagated
            with pytest.raises(Exception) as exc_info:
                client.health_check()
            assert "HTTP error" in str(exc_info.value)

    def test_constants(self):
        """Test that class constants are correctly defined"""
        assert RunLocalClient.BASE_URL == "https://neuralize-bench.com"
        assert RunLocalClient.ENV_VAR_NAME == "RUNLOCAL_API_KEY"

