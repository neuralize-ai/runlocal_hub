"""
Tests for tensor handling functionality.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest
from runlocal_hub.tensors.handler import TensorHandler
from runlocal_hub.models.tensor import (
    IOType,
    IOTensorsMetadata,
    IOTensorsPresignedUrlResponse,
    TensorInfo,
)
from runlocal_hub.exceptions import TensorError


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    return Mock()


@pytest.fixture
def tensor_handler(mock_http_client):
    """Create a TensorHandler instance with mock HTTP client."""
    return TensorHandler(mock_http_client)


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    return {
        "input_image": np.random.rand(1, 224, 224, 3).astype(np.float32),
        "input_mask": np.random.randint(0, 2, (1, 224, 224), dtype=np.uint8),
        "metadata": np.array([1.0, 2.0, 3.0], dtype=np.float64),
    }


@pytest.fixture
def large_tensors():
    """Create larger tensors for testing performance."""
    return {
        "features": np.random.rand(100, 512, 512, 3).astype(np.float32),
        "labels": np.random.randint(0, 1000, (100,), dtype=np.int32),
        "weights": np.random.rand(512, 1000).astype(np.float32),
    }


@pytest.fixture
def sample_tensor_metadata():
    """Create sample tensor metadata."""
    return IOTensorsMetadata(
        UserId="user-123",
        Id="tensor-abc123",
        IOType=IOType.INPUT,
        TensorMetadata={
            "input_image": TensorInfo(
                Shape=[1, 224, 224, 3],
                Dtype="float32",
                SizeBytes=602112,  # 1*224*224*3*4 bytes
            ),
            "input_mask": TensorInfo(
                Shape=[1, 224, 224],
                Dtype="uint8",
                SizeBytes=50176,  # 1*224*224*1 bytes
            ),
        },
        SourceBenchmarkIds=None,
        CreatedUtc="2024-01-15T10:30:00Z",
    )


@pytest.fixture
def sample_presigned_response():
    """Create sample presigned URL response."""
    return IOTensorsPresignedUrlResponse(
        tensors_id="tensor-abc123",
        presigned_url="https://s3.amazonaws.com/bucket/tensor-abc123.npz?signature=xyz",
        expires_in_seconds=3600,
    )


class TestTensorHandler:
    """Test cases for TensorHandler functionality."""

    def test_init(self, mock_http_client):
        """Test TensorHandler initialization."""
        handler = TensorHandler(mock_http_client)
        assert handler.http_client == mock_http_client

    def test_upload_tensors_success(
        self, tensor_handler, mock_http_client, sample_tensors
    ):
        """Test successful tensor upload."""
        # Mock API response
        mock_http_client.post_file.return_value = {"tensors_id": "tensor-uploaded-123"}

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            result = tensor_handler.upload_tensors(sample_tensors, IOType.INPUT)

        assert result == "tensor-uploaded-123"

        # Verify the API call
        mock_http_client.post_file.assert_called_once()
        call_args = mock_http_client.post_file.call_args

        # Check endpoint
        assert call_args[0][0] == "/io-tensors/upload"

        # Check files parameter contains NPZ data
        files = call_args[1]["files"]
        assert "file" in files
        assert files["file"][0] == "tensors.npz"
        assert files["file"][2] == "application/octet-stream"
        assert isinstance(files["file"][1], bytes)

        # Check parameters
        params = call_args[1]["params"]
        assert params["io_type"] == "input"

    def test_upload_tensors_with_source_benchmark(
        self, tensor_handler, mock_http_client, sample_tensors
    ):
        """Test tensor upload with source benchmark ID."""
        mock_http_client.post_file.return_value = {"tensors_id": "tensor-output-456"}

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            result = tensor_handler.upload_tensors(
                sample_tensors, IOType.OUTPUT, source_benchmark_id="benchmark-789"
            )

        assert result == "tensor-output-456"

        # Check parameters include benchmark ID
        call_args = mock_http_client.post_file.call_args
        params = call_args[1]["params"]
        assert params["io_type"] == "output"
        assert params["source_benchmark_id"] == "benchmark-789"

    def test_upload_tensors_empty_dict(self, tensor_handler):
        """Test upload with empty tensor dictionary."""
        from runlocal_hub.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Tensors dictionary cannot be empty"):
            tensor_handler.upload_tensors({}, IOType.INPUT)

    def test_upload_tensors_invalid_names(self, tensor_handler):
        """Test upload with invalid tensor names."""
        from runlocal_hub.exceptions import ValidationError

        invalid_tensors = {"": np.array([1, 2, 3])}

        with pytest.raises(ValidationError, match="Invalid tensor name"):
            tensor_handler.upload_tensors(invalid_tensors, IOType.INPUT)

        invalid_tensors2 = {None: np.array([1, 2, 3])}
        with pytest.raises(ValidationError, match="Invalid tensor name"):
            tensor_handler.upload_tensors(invalid_tensors2, IOType.INPUT)

    def test_upload_tensors_api_error(
        self, tensor_handler, mock_http_client, sample_tensors
    ):
        """Test tensor upload when API returns error."""
        mock_http_client.post_file.side_effect = Exception("API Error")

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            with pytest.raises(TensorError, match="Failed to upload tensors"):
                tensor_handler.upload_tensors(sample_tensors, IOType.INPUT)

    def test_upload_tensors_no_id_returned(
        self, tensor_handler, mock_http_client, sample_tensors
    ):
        """Test tensor upload when API doesn't return tensors_id."""
        mock_http_client.post_file.return_value = {}

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            with pytest.raises(TensorError, match="No tensors_id returned from API"):
                tensor_handler.upload_tensors(sample_tensors, IOType.INPUT)

    def test_download_tensors_success(
        self,
        tensor_handler,
        mock_http_client,
        sample_tensors,
        sample_presigned_response,
    ):
        """Test successful tensor download."""
        # Create NPZ data from sample tensors
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **sample_tensors)
        npz_data = npz_buffer.getvalue()

        # Mock API responses
        mock_http_client.get.return_value = sample_presigned_response.model_dump()
        mock_http_client.download_from_url.return_value = npz_data

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "runlocal_hub.tensors.handler.handle_api_errors", lambda func: func
            ):
                result = tensor_handler.download_tensors("tensor-abc123", temp_dir)

            # Verify files were created
            assert len(result) == 3
            assert "input_image" in result
            assert "input_mask" in result
            assert "metadata" in result

            # Check file paths
            for name, path in result.items():
                assert Path(path).exists()
                assert Path(path).suffix == ".npy"
                assert str(Path(temp_dir)) in path

    def test_download_tensors_default_dir(
        self,
        tensor_handler,
        mock_http_client,
        sample_tensors,
        sample_presigned_response,
    ):
        """Test tensor download with default output directory."""
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **sample_tensors)
        npz_data = npz_buffer.getvalue()

        mock_http_client.get.return_value = sample_presigned_response.model_dump()
        mock_http_client.download_from_url.return_value = npz_data

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            with patch("pathlib.Path.cwd") as mock_cwd:
                mock_cwd.return_value = Path("/mock/current/dir")
                with patch("pathlib.Path.mkdir"):
                    with patch("numpy.save"):
                        result = tensor_handler.download_tensors("tensor-abc123")

        # Check that default path is used
        for path in result.values():
            assert "/mock/current/dir/outputs/tensor-abc123/" in path

    def test_download_tensors_invalid_id(self, tensor_handler):
        """Test download with invalid tensor ID."""
        from runlocal_hub.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Invalid tensors_id"):
            tensor_handler.download_tensors("")

        with pytest.raises(ValidationError, match="Invalid tensors_id"):
            tensor_handler.download_tensors(None)

    def test_download_tensors_api_error(self, tensor_handler, mock_http_client):
        """Test tensor download when API returns error."""
        mock_http_client.get.side_effect = Exception("API Error")

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            with pytest.raises(TensorError, match="Failed to download tensors"):
                tensor_handler.download_tensors("tensor-abc123")

    def test_get_metadata_success(
        self, tensor_handler, mock_http_client, sample_tensor_metadata
    ):
        """Test successful metadata retrieval."""
        # Mock API response with lowercase field names
        api_response = {
            "user_id": "user-123",
            "id": "tensor-abc123",
            "io_type": "input",
            "tensor_metadata": {
                "input_image": {
                    "Shape": [1, 224, 224, 3],
                    "Dtype": "float32",
                    "SizeBytes": 602112,
                }
            },
            "source_benchmark_ids": None,
            "created_utc": "2024-01-15T10:30:00Z",
        }
        mock_http_client.get.return_value = api_response

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            result = tensor_handler.get_metadata("tensor-abc123")

        assert isinstance(result, IOTensorsMetadata)
        assert result.UserId == "user-123"
        assert result.Id == "tensor-abc123"
        assert result.IOType == IOType.INPUT
        assert "input_image" in result.TensorMetadata

        # Verify API call
        mock_http_client.get.assert_called_once_with("/io-tensors/tensor-abc123")

    def test_get_metadata_invalid_id(self, tensor_handler):
        """Test metadata retrieval with invalid ID."""
        from runlocal_hub.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Invalid tensors_id"):
            tensor_handler.get_metadata("")

        with pytest.raises(ValidationError, match="Invalid tensors_id"):
            tensor_handler.get_metadata(None)

    def test_list_tensors_no_filter(self, tensor_handler, mock_http_client):
        """Test listing all tensors without filter."""
        api_response = [
            {
                "user_id": "user-123",
                "id": "tensor-1",
                "io_type": "input",
                "tensor_metadata": {},
                "source_benchmark_ids": None,
                "created_utc": "2024-01-15T10:30:00Z",
            },
            {
                "user_id": "user-123",
                "id": "tensor-2",
                "io_type": "output",
                "tensor_metadata": {},
                "source_benchmark_ids": ["benchmark-1"],
                "created_utc": "2024-01-15T11:00:00Z",
            },
        ]
        mock_http_client.get.return_value = api_response

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            result = tensor_handler.list_tensors()

        assert len(result) == 2
        assert all(isinstance(item, IOTensorsMetadata) for item in result)
        assert result[0].IOType == IOType.INPUT
        assert result[1].IOType == IOType.OUTPUT

        # Verify API call
        mock_http_client.get.assert_called_once_with("/io-tensors", params=None)

    def test_list_tensors_with_filter(self, tensor_handler, mock_http_client):
        """Test listing tensors with IOType filter."""
        api_response = [
            {
                "user_id": "user-123",
                "id": "tensor-input-1",
                "io_type": "input",
                "tensor_metadata": {},
                "source_benchmark_ids": None,
                "created_utc": "2024-01-15T10:30:00Z",
            }
        ]
        mock_http_client.get.return_value = api_response

        with patch("runlocal_hub.tensors.handler.handle_api_errors", lambda func: func):
            result = tensor_handler.list_tensors(IOType.INPUT)

        assert len(result) == 1
        assert result[0].IOType == IOType.INPUT

        # Verify API call with parameters
        mock_http_client.get.assert_called_once_with(
            "/io-tensors", params={"io_type": "input"}
        )


class TestTensorSerialization:
    """Test cases for tensor serialization/deserialization."""

    def test_serialize_tensors_basic(self, tensor_handler, sample_tensors):
        """Test basic tensor serialization."""
        npz_data = tensor_handler._serialize_tensors(sample_tensors)

        assert isinstance(npz_data, bytes)
        assert len(npz_data) > 0

    def test_serialize_tensors_different_dtypes(self, tensor_handler):
        """Test serialization with different data types."""
        tensors = {
            "float32": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "int64": np.array([1, 2, 3], dtype=np.int64),
            "bool": np.array([True, False, True], dtype=bool),
            "string": np.array(["hello", "world"], dtype="U10"),
        }

        npz_data = tensor_handler._serialize_tensors(tensors)
        assert isinstance(npz_data, bytes)

    def test_serialize_large_tensors(self, tensor_handler, large_tensors):
        """Test serialization of large tensors."""
        npz_data = tensor_handler._serialize_tensors(large_tensors)

        assert isinstance(npz_data, bytes)
        # Compressed data should be significantly smaller than raw data
        raw_size = sum(tensor.nbytes for tensor in large_tensors.values())
        assert len(npz_data) < raw_size

    def test_deserialize_tensors_basic(self, tensor_handler, sample_tensors):
        """Test basic tensor deserialization."""
        # Serialize first
        npz_data = tensor_handler._serialize_tensors(sample_tensors)

        # Then deserialize
        deserialized = tensor_handler._deserialize_tensors(npz_data)

        assert len(deserialized) == len(sample_tensors)
        for name in sample_tensors:
            assert name in deserialized
            np.testing.assert_array_equal(sample_tensors[name], deserialized[name])

    def test_serialize_deserialize_round_trip(self, tensor_handler):
        """Test complete round-trip serialization/deserialization."""
        original_tensors = {
            "image": np.random.rand(3, 256, 256).astype(np.float32),
            "labels": np.random.randint(0, 10, (100,), dtype=np.int32),
            "metadata": np.array([42.0, 3.14, 2.718], dtype=np.float64),
        }

        # Round trip
        npz_data = tensor_handler._serialize_tensors(original_tensors)
        recovered_tensors = tensor_handler._deserialize_tensors(npz_data)

        # Verify everything matches
        assert set(original_tensors.keys()) == set(recovered_tensors.keys())
        for name in original_tensors:
            np.testing.assert_array_equal(
                original_tensors[name], recovered_tensors[name]
            )
            assert original_tensors[name].dtype == recovered_tensors[name].dtype

    def test_serialize_invalid_data(self, tensor_handler):
        """Test serialization with invalid data."""

        # Truly invalid data (object that can't be serialized by numpy)
        class UnsuitableObject:
            pass

        invalid_tensors = {"invalid": UnsuitableObject()}

        with pytest.raises(TensorError, match="Failed to serialize tensors"):
            tensor_handler._serialize_tensors(invalid_tensors)

    def test_deserialize_invalid_data(self, tensor_handler):
        """Test deserialization with invalid NPZ data."""
        invalid_data = b"not npz data"

        with pytest.raises(TensorError, match="Failed to deserialize tensors"):
            tensor_handler._deserialize_tensors(invalid_data)


class TestTensorFileHandling:
    """Test cases for tensor file operations."""

    def test_tensor_name_sanitization(
        self, tensor_handler, mock_http_client, sample_presigned_response
    ):
        """Test that tensor names with invalid characters are sanitized."""
        # Create tensors with names containing path separators
        tensors_with_paths = {
            "output0": np.array([1, 2, 3]),
            "output1": np.array([4, 5, 6]),
        }

        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, allow_pickle=False, **tensors_with_paths)
        npz_data = npz_buffer.getvalue()

        mock_http_client.get.return_value = sample_presigned_response.model_dump()
        mock_http_client.download_from_url.return_value = npz_data

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "runlocal_hub.tensors.handler.handle_api_errors", lambda func: func
            ):
                result = tensor_handler.download_tensors("tensor-abc123", temp_dir)

            # Check that path separators were replaced with underscores
            assert "output0" in result
            assert "output1" in result

            # Check actual file paths are sanitized
            for original_name, file_path in result.items():
                if "/" in original_name or "\\" in original_name:
                    assert "/" not in Path(file_path).name
                    assert "\\" not in Path(file_path).name
                    assert "_" in Path(file_path).name

    def test_output_directory_creation(
        self,
        tensor_handler,
        mock_http_client,
        sample_tensors,
        sample_presigned_response,
    ):
        """Test that output directories are created automatically."""
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **sample_tensors)
        npz_data = npz_buffer.getvalue()

        mock_http_client.get.return_value = sample_presigned_response.model_dump()
        mock_http_client.download_from_url.return_value = npz_data

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nested" / "deep" / "path"

            with patch(
                "runlocal_hub.tensors.handler.handle_api_errors", lambda func: func
            ):
                result = tensor_handler.download_tensors("tensor-abc123", nested_dir)

            # Verify the nested directory structure was created
            expected_dir = nested_dir / "tensor-abc123"
            assert expected_dir.exists()
            assert expected_dir.is_dir()


class TestTensorModels:
    """Test cases for tensor-related models."""

    def test_io_type_enum(self):
        """Test IOType enum values."""
        assert IOType.INPUT.value == "input"
        assert IOType.OUTPUT.value == "output"
        assert len(IOType) == 2

    def test_tensor_info_creation(self):
        """Test TensorInfo model creation."""
        info = TensorInfo(Shape=[1, 224, 224, 3], Dtype="float32", SizeBytes=602112)

        assert info.Shape == [1, 224, 224, 3]
        assert info.Dtype == "float32"
        assert info.SizeBytes == 602112

    def test_io_tensors_metadata_creation(self):
        """Test IOTensorsMetadata model creation."""
        metadata = IOTensorsMetadata(
            UserId="user-123",
            Id="tensor-abc",
            IOType=IOType.INPUT,
            TensorMetadata={
                "image": TensorInfo(
                    Shape=[1, 3, 224, 224], Dtype="float32", SizeBytes=602112
                )
            },
            SourceBenchmarkIds=["bench-1", "bench-2"],
            CreatedUtc="2024-01-15T10:30:00Z",
        )

        assert metadata.UserId == "user-123"
        assert metadata.IOType == IOType.INPUT
        assert "image" in metadata.TensorMetadata
        assert metadata.SourceBenchmarkIds == ["bench-1", "bench-2"]

    def test_io_tensors_metadata_optional_fields(self):
        """Test IOTensorsMetadata with optional fields as None."""
        metadata = IOTensorsMetadata(
            UserId="user-123",
            Id="tensor-abc",
            IOType=IOType.OUTPUT,
            TensorMetadata={},
            CreatedUtc="2024-01-15T10:30:00Z",
        )

        assert metadata.SourceBenchmarkIds is None

    def test_presigned_url_response_creation(self):
        """Test IOTensorsPresignedUrlResponse model creation."""
        response = IOTensorsPresignedUrlResponse(
            tensors_id="tensor-123",
            presigned_url="https://s3.amazonaws.com/bucket/file.npz?signature=xyz",
            expires_in_seconds=3600,
        )

        assert response.tensors_id == "tensor-123"
        assert "s3.amazonaws.com" in response.presigned_url
        assert response.expires_in_seconds == 3600


class TestTensorIntegration:
    """Integration test cases for tensor operations."""

    def test_upload_download_integration(self, tensor_handler, mock_http_client):
        """Test full upload -> download cycle."""
        # Create test tensors
        original_tensors = {
            "input": np.random.rand(2, 3, 4).astype(np.float32),
            "target": np.random.randint(0, 10, (2,), dtype=np.int64),
        }

        # Mock upload
        mock_http_client.post_file.return_value = {"tensors_id": "integration-test-123"}

        # Mock download
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **original_tensors)
        npz_data = npz_buffer.getvalue()

        presigned_response = {
            "tensors_id": "integration-test-123",
            "presigned_url": "https://s3.amazonaws.com/bucket/file.npz",
            "expires_in_seconds": 3600,
        }
        mock_http_client.get.return_value = presigned_response
        mock_http_client.download_from_url.return_value = npz_data

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "runlocal_hub.tensors.handler.handle_api_errors", lambda func: func
            ):
                # Upload
                tensors_id = tensor_handler.upload_tensors(
                    original_tensors, IOType.INPUT
                )
                assert tensors_id == "integration-test-123"

                # Download
                downloaded_paths = tensor_handler.download_tensors(tensors_id, temp_dir)

                # Verify downloaded tensors match originals
                assert len(downloaded_paths) == len(original_tensors)
                for name, path in downloaded_paths.items():
                    downloaded_array = np.load(path)
                    np.testing.assert_array_equal(
                        original_tensors[name], downloaded_array
                    )

