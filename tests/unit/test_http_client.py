import pytest
import responses
import json

from runlocal_hub.http.client import HTTPClient
from runlocal_hub.exceptions import AuthenticationError, APIError


class TestHTTPClient:
    """Test cases for HTTPClient"""

    def test_http_client_initialization(self):
        """Test HTTP client initialization with normal and debug modes"""
        # Normal mode
        client = HTTPClient(base_url="https://neuralize-bench.com", api_key="test_key")
        assert client.base_url == "https://neuralize-bench.com"
        assert client.headers["X-API-KEY"] == "test_key"
        assert client.api_key == "test_key"
        assert client.debug is False

        # Debug mode
        debug_client = HTTPClient(
            base_url="https://neuralize-bench.com", api_key="test_key", debug=True
        )
        assert debug_client.debug is True

    @responses.activate
    def test_successful_requests(self):
        """Test successful GET and POST requests"""
        # GET request
        responses.add(
            responses.GET,
            "https://neuralize-bench.com/test",
            json={"status": "ok"},
            status=200,
        )

        # POST request
        responses.add(
            responses.POST,
            "https://neuralize-bench.com/test",
            json={"id": "123"},
            status=201,
        )

        client = HTTPClient(base_url="https://neuralize-bench.com", api_key="test_key")

        # Test GET
        get_result = client.get("/test")
        assert get_result == {"status": "ok"}
        assert responses.calls[0].request.headers["X-API-KEY"] == "test_key"

        # Test POST
        post_result = client.post("/test", data={"name": "test"})
        assert post_result == {"id": "123"}
        assert responses.calls[1].request.body is not None
        request_body = json.loads(responses.calls[1].request.body)
        assert request_body == {"name": "test"}

    @responses.activate
    @pytest.mark.parametrize(
        "status_code,error_class,detail",
        [
            (401, AuthenticationError, "Invalid API key"),
            (400, APIError, "Bad request"),
            (404, APIError, "Not found"),
            (500, APIError, "Server error"),
        ],
    )
    def test_error_handling(self, status_code, error_class, detail):
        """Test HTTP error handling for various status codes"""
        responses.add(
            responses.GET,
            "https://neuralize-bench.com/test",
            json={"detail": detail},
            status=status_code,
        )

        client = HTTPClient(base_url="https://neuralize-bench.com", api_key="test_key")
        with pytest.raises(error_class) as exc_info:
            client.get("/test")

        if hasattr(exc_info.value, "status_code"):
            assert exc_info.value.status_code == status_code
        assert detail in str(
            exc_info.value
        ) or "Invalid API key or unauthorized access" in str(exc_info.value)

    @responses.activate
    def test_request_features(self):
        """Test query parameters, binary data, and streaming responses"""
        # Query parameters
        responses.add(
            responses.GET,
            "https://neuralize-bench.com/test",
            json={"result": "data"},
            status=200,
        )

        # Binary data upload
        responses.add(
            responses.POST,
            "https://neuralize-bench.com/upload",
            json={"upload_id": "123"},
            status=200,
        )

        # Streaming response
        responses.add(
            responses.GET,
            "https://neuralize-bench.com/stream",
            body="streaming data",
            status=200,
        )

        client = HTTPClient(base_url="https://neuralize-bench.com", api_key="test_key")

        # Test query parameters
        result = client.get("/test", params={"filter": "active", "limit": 10})
        assert result == {"result": "data"}
        assert responses.calls[0].request.url is not None
        assert "filter=active" in responses.calls[0].request.url
        assert "limit=10" in responses.calls[0].request.url

        # Test binary data
        binary_data = b"test binary data"
        upload_result = client.request("POST", "/upload", data=binary_data)
        assert upload_result == {"upload_id": "123"}
        assert responses.calls[1].request.body == binary_data

        # Test streaming
        stream_response = client.request("GET", "/stream", stream=True)
        assert hasattr(stream_response, "status_code")
        assert stream_response.status_code == 200

    @responses.activate
    def test_non_json_responses(self):
        """Test handling of non-JSON responses and errors without detail"""
        # Non-JSON response
        responses.add(
            responses.GET,
            "https://neuralize-bench.com/text",
            body="plain text response",
            status=200,
        )

        # Error without detail field
        responses.add(
            responses.GET,
            "https://neuralize-bench.com/error",
            body="Simple error message",
            status=400,
        )

        client = HTTPClient(base_url="https://neuralize-bench.com", api_key="test_key")

        # Test non-JSON success
        text_result = client.get("/text")
        assert text_result == {"text": "plain text response"}

        # Test error without detail
        with pytest.raises(APIError) as exc_info:
            client.get("/error")
        assert "Simple error message" in str(exc_info.value)

    @responses.activate
    def test_connection_errors(self):
        """Test handling of connection and timeout errors"""

        def connection_error_callback(request):
            raise ConnectionError("Connection failed")

        def timeout_error_callback(request):
            raise TimeoutError("Request timed out")

        # Connection error
        responses.add_callback(
            responses.GET,
            "https://neuralize-bench.com/connection-test",
            callback=connection_error_callback,
        )

        # Timeout error
        responses.add_callback(
            responses.GET,
            "https://neuralize-bench.com/timeout-test",
            callback=timeout_error_callback,
        )

        client = HTTPClient(base_url="https://neuralize-bench.com", api_key="test_key")

        # Test connection error
        with pytest.raises(Exception) as exc_info:
            client.get("/connection-test")
        assert "Connection failed" in str(exc_info.value)

        # Test timeout error
        with pytest.raises(Exception) as exc_info:
            client.get("/timeout-test")
        assert "Request timed out" in str(exc_info.value)

