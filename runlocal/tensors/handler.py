"""
Tensor upload and download operations handler.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..exceptions import TensorError
from ..http import HTTPClient
from ..models import IOTensorsMetadata, IOTensorsPresignedUrlResponse, IOType
from ..utils.decorators import handle_api_errors


class TensorHandler:
    """
    Handles tensor upload, download, and metadata operations.
    """

    def __init__(self, http_client: HTTPClient):
        """
        Initialize the tensor handler.

        Args:
            http_client: HTTP client for API requests
        """
        self.http_client = http_client

    @handle_api_errors
    def upload_tensors(
        self,
        tensors: Dict[str, np.ndarray],
        io_type: IOType = IOType.INPUT,
        source_benchmark_id: Optional[str] = None,
    ) -> str:
        """
        Upload numpy arrays as IOTensors.

        Args:
            tensors: Dictionary mapping tensor names to numpy arrays
            io_type: Whether these are input or output tensors
            source_benchmark_id: For output tensors, the benchmark that created them

        Returns:
            The tensors_id (hash) of the uploaded tensors

        Raises:
            TensorError: If upload fails
            ValueError: If tensors dict is empty
        """
        if not tensors:
            raise ValueError("Tensors dictionary cannot be empty")

        # Validate tensor names
        for name in tensors:
            if not name or not isinstance(name, str):
                raise ValueError(f"Invalid tensor name: {name}")

        # Serialize tensors to NPZ format
        npz_data = self._serialize_tensors(tensors)

        # Upload to API
        return self._upload_to_api(npz_data, io_type, source_benchmark_id)

    @handle_api_errors
    def download_tensors(
        self, tensors_id: str, output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, str]:
        """
        Download IOTensors and save to disk.

        Args:
            tensors_id: The ID of the tensors to download
            output_dir: Directory to save tensors (defaults to current dir/outputs/)

        Returns:
            Dictionary mapping tensor names to file paths

        Raises:
            TensorError: If download fails
            ValueError: If tensors_id is invalid
        """
        if not tensors_id or not isinstance(tensors_id, str):
            raise ValueError("Invalid tensors_id")

        # Download from API
        npz_data = self._download_from_api(tensors_id)

        # Deserialize NPZ data
        tensors = self._deserialize_tensors(npz_data)

        # Save tensors to disk
        if output_dir is None:
            output_dir = Path.cwd() / "outputs" / tensors_id
        else:
            output_dir = Path(output_dir) / tensors_id

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each tensor to a separate .npy file and return paths
        tensor_paths = {}
        for name, array in tensors.items():
            # Sanitize tensor name for filesystem
            safe_name = name.replace("/", "_").replace("\\", "_")
            file_path = output_dir / f"{safe_name}.npy"
            np.save(file_path, array)
            tensor_paths[name] = str(file_path)

        return tensor_paths

    @handle_api_errors
    def get_metadata(self, tensors_id: str) -> IOTensorsMetadata:
        """
        Get metadata about IOTensors without downloading the data.

        Args:
            tensors_id: The ID of the tensors to get metadata for

        Returns:
            Metadata about the tensors

        Raises:
            TensorError: If metadata retrieval fails
            ValueError: If tensors_id is invalid
        """
        if not tensors_id or not isinstance(tensors_id, str):
            raise ValueError("Invalid tensors_id")

        response = self.http_client.get(f"/io-tensors/{tensors_id}")

        # Map API response fields to our model fields
        # The API uses lowercase field names
        mapped_response = {
            "UserId": response.get("user_id", ""),
            "Id": response.get("id"),
            "IOType": response.get("io_type"),
            "TensorMetadata": response.get("tensor_metadata"),
            "SourceBenchmarkIds": response.get("source_benchmark_ids"),
            "CreatedUtc": response.get("created_utc"),
        }

        return IOTensorsMetadata(**mapped_response)

    @handle_api_errors
    def list_tensors(self, io_type: Optional[IOType] = None) -> List[IOTensorsMetadata]:
        """
        List all IOTensors for the authenticated user.

        Args:
            io_type: Optional filter by input or output type

        Returns:
            List of tensor metadata
        """
        endpoint = "/io-tensors"
        params = {}

        if io_type:
            params["io_type"] = io_type.value

        response = self.http_client.get(endpoint, params=params if params else None)

        # Map API response fields for each item
        mapped_items = []
        for item in response:
            mapped_item = {
                "UserId": item.get("user_id"),
                "Id": item.get("id"),
                "IOType": item.get("io_type"),
                "TensorMetadata": item.get("tensor_metadata"),
                "SourceBenchmarkIds": item.get("source_benchmark_ids"),
                "CreatedUtc": item.get("created_utc"),
            }
            mapped_items.append(IOTensorsMetadata(**mapped_item))

        return mapped_items

    def _serialize_tensors(self, tensors: Dict[str, np.ndarray]) -> bytes:
        """
        Convert tensors to NPZ format.

        Args:
            tensors: Dictionary of numpy arrays

        Returns:
            NPZ file data as bytes

        Raises:
            TensorError: If serialization fails
        """
        try:
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, allow_pickle=True, **tensors)
            npz_buffer.seek(0)
            return npz_buffer.read()
        except Exception as e:
            raise TensorError(f"Failed to serialize tensors: {str(e)}")

    def _deserialize_tensors(self, npz_data: bytes) -> Dict[str, np.ndarray]:
        """
        Convert NPZ data back to tensors.

        Args:
            npz_data: NPZ file data as bytes

        Returns:
            Dictionary of numpy arrays

        Raises:
            TensorError: If deserialization fails
        """
        try:
            npz_buffer = io.BytesIO(npz_data)
            npz_file = np.load(npz_buffer)

            # Convert to regular dict
            tensors = {name: npz_file[name] for name in npz_file.files}

            # Close the npz file to free resources
            npz_file.close()

            return tensors
        except Exception as e:
            raise TensorError(f"Failed to deserialize tensors: {str(e)}")

    def _upload_to_api(
        self,
        npz_data: bytes,
        io_type: IOType,
        source_benchmark_id: Optional[str],
    ) -> str:
        """
        Upload NPZ data to the API.

        Args:
            npz_data: Serialized tensor data
            io_type: Type of tensors (input/output)
            source_benchmark_id: Optional benchmark ID for output tensors

        Returns:
            Tensor ID from the API

        Raises:
            TensorError: If upload fails
        """
        try:
            # Prepare parameters
            params = {"io_type": io_type.value}
            if source_benchmark_id:
                params["source_benchmark_id"] = source_benchmark_id

            # Upload the NPZ file
            files = {"file": ("tensors.npz", npz_data, "application/octet-stream")}

            result = self.http_client.post_file(
                "/io-tensors/upload",
                files=files,
                params=params,
            )

            tensors_id = result.get("tensors_id")
            if not tensors_id:
                raise TensorError("No tensors_id returned from API")

            return tensors_id

        except Exception as e:
            if isinstance(e, TensorError):
                raise
            raise TensorError(f"Failed to upload tensors: {str(e)}")

    def _download_from_api(self, tensors_id: str) -> bytes:
        """
        Download NPZ data from the API using presigned URL.

        Args:
            tensors_id: ID of tensors to download

        Returns:
            NPZ file data as bytes

        Raises:
            TensorError: If download fails
        """
        try:
            # Get presigned URL
            endpoint = f"/io-tensors/{tensors_id}/presigned-url"
            response = self.http_client.get(endpoint)

            # Parse response into our model
            presigned_response = IOTensorsPresignedUrlResponse(**response)

            # Download from presigned URL
            return self.http_client.download_from_url(presigned_response.presigned_url)
        except Exception as e:
            raise TensorError(f"Failed to download tensors: {str(e)}")
