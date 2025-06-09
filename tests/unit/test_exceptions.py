from runlocal_hub.exceptions import (
    RunLocalError,
    ConfigurationError,
    AuthenticationError,
    ModelNotFoundError,
    DeviceNotAvailableError,
    JobTimeoutError,
    TensorError,
    UploadError,
    ValidationError,
    APIError,
    NetworkError,
)


class TestExceptions:
    """Test cases for custom exceptions"""

    def test_all_exceptions_basic(self):
        """Test all exception types with basic functionality"""
        exception_classes = [
            (RunLocalError, "Base error"),
            (ConfigurationError, "Missing API key"),
            (AuthenticationError, "Invalid API key"),
            (ModelNotFoundError, "Model not found"),
            (DeviceNotAvailableError, "No devices available"),
            (JobTimeoutError, "Job timed out"),
            (TensorError, "Invalid tensor"),
            (UploadError, "Upload failed"),
            (ValidationError, "Invalid parameter"),
            (APIError, "API error"),
            (NetworkError, "Network error"),
        ]

        for exception_class, message in exception_classes:
            error = exception_class(message)
            assert str(error) == message
            assert isinstance(error, Exception)
            if exception_class != RunLocalError:
                assert isinstance(error, RunLocalError)

    def test_exceptions_inheritance(self):
        """Test that all exceptions inherit from RunLocalError"""
        exceptions = [
            ConfigurationError("test"),
            AuthenticationError("test"),
            ModelNotFoundError("test"),
            DeviceNotAvailableError("test"),
            JobTimeoutError("test"),
            TensorError("test"),
            UploadError("test"),
            ValidationError("test"),
            APIError("test"),
            NetworkError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, RunLocalError)
            assert isinstance(exc, Exception)

    def test_exceptions_with_details(self):
        """Test key exceptions with detail attributes"""
        # Test APIError with details
        api_error = APIError(
            "Request failed",
            status_code=400,
            response_data={"error": "Bad request"},
            endpoint="/test",
        )
        assert api_error.status_code == 400
        assert api_error.response_data == {"error": "Bad request"}
        assert api_error.endpoint == "/test"

        # Test ModelNotFoundError with details
        model_error = ModelNotFoundError(
            "Model not found",
            model_id="test_123",
            available_models=["model_1", "model_2"],
        )
        assert model_error.model_id == "test_123"
        assert model_error.available_models == ["model_1", "model_2"]

        # Test RunLocalError with details dict
        base_error = RunLocalError("Error", details={"code": "TEST"})
        assert base_error.details == {"code": "TEST"}

