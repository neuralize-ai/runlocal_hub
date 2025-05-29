"""
Simplified RunLocal API client.
"""
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from .exceptions import RunLocalError, UploadError
from .http import HTTPClient
from .utils.decorators import handle_api_errors


class RunLocalClient:
    """
    Simplified Python client for the RunLocal API.
    """

    # DEFAULT_BASE_URL = "https://neuralize-bench.com"
    DEFAULT_BASE_URL = "http://127.0.0.1:8000"  # Local development
    ENV_VAR_NAME = "RUNLOCAL_API_KEY"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, debug: bool = False):
        """
        Initialize the RunLocal client.

        Args:
            api_key: API key for authentication. If not provided, will look for RUNLOCAL_API_KEY env var
            base_url: Base URL for the API. Defaults to production URL
            debug: Enable debug logging
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get(self.ENV_VAR_NAME)
            if not api_key:
                raise RunLocalError(
                    f"{self.ENV_VAR_NAME} not found. Please provide an API key or set the environment variable."
                )

        # Set base URL
        if base_url is None:
            base_url = self.DEFAULT_BASE_URL

        # Initialize HTTP client
        self.http_client = HTTPClient(base_url=base_url, api_key=api_key, debug=debug)
        self.debug = debug

    @handle_api_errors
    def health_check(self) -> Dict:
        """
        Check if the API is available and the API key is valid.

        Returns:
            Health status information

        Raises:
            AuthenticationError: If the API key is invalid
            RunLocalError: If the API is unavailable
        """
        return self.http_client.get("/users/health")

    @handle_api_errors
    def get_user_info(self) -> Dict:
        """
        Get detailed user information for the authenticated user.

        Returns:
            User information including models, datasets, etc.

        Raises:
            AuthenticationError: If the API key is invalid
        """
        return self.http_client.get("/users")

    @handle_api_errors
    def get_models(self) -> List[str]:
        """
        Get a list of model IDs for the authenticated user.

        Returns:
            List of model IDs

        Raises:
            AuthenticationError: If the API key is invalid
        """
        user_data = self.get_user_info()
        return user_data.get("UploadIds", [])

    def upload_model(
        self,
        model_path: Union[Path, str],
        show_progress: bool = True,
    ) -> str:
        """
        Upload a model file or folder to the RunLocal platform.

        Args:
            model_path: Path to the model file or folder to upload
            show_progress: Whether to show progress bar

        Returns:
            Upload ID of the uploaded model

        Raises:
            FileNotFoundError: If the model path doesn't exist
            UploadError: If upload fails
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        upload_filename = model_path.stem

        # Create temporary directory for zipping
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Zip the model
            if self.debug:
                print(f"Zipping {model_path}...")

            zip_path = self._zip_path(model_path, temp_path)
            zip_size = zip_path.stat().st_size

            if self.debug:
                print(f"Zip file created: {zip_path.name} ({zip_size / 1024 / 1024:.2f} MB)")

            # Prepare upload parameters
            params = {
                "upload_filename": upload_filename,
                "upload_source_type": "USER_UPLOADED",
            }

            # Read the zip file
            with open(zip_path, "rb") as f:
                zip_data = f.read()

            if self.debug:
                print("Uploading to server...")

            # Upload and process response
            return self._process_upload_stream(
                endpoint="/uploads/model/coreml",
                data=zip_data,
                params=params,
                show_progress=show_progress,
            )

    def _zip_path(self, path: Path, temp_dir: Path) -> Path:
        """
        Zip a file or folder.

        Args:
            path: Path to file or folder to zip
            temp_dir: Temporary directory to store the zip file

        Returns:
            Path to the created zip file
        """
        zip_name = path.stem if path.is_file() else path.name
        zip_path = temp_dir / f"{zip_name}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if path.is_file():
                # If it's a single file, add it to the zip
                zipf.write(path, path.name)
            else:
                # If it's a directory, add all files recursively
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        # Calculate the relative path from the base directory
                        arcname = file_path.relative_to(path.parent)
                        zipf.write(file_path, arcname)

        return zip_path

    @handle_api_errors
    def _process_upload_stream(
        self,
        endpoint: str,
        data: bytes,
        params: Dict,
        show_progress: bool,
    ) -> str:
        """
        Process a streaming upload response.

        Args:
            endpoint: API endpoint
            data: Binary data to upload
            params: Query parameters
            show_progress: Whether to show progress bar

        Returns:
            Upload ID

        Raises:
            UploadError: If upload fails
        """
        upload_id = None
        already_exists = False
        progress_bar = None

        if show_progress:
            progress_bar = tqdm(total=100, desc="Upload Progress", unit="%")

        try:
            for message in self.http_client.post_streaming(endpoint, data, params):
                # Check for errors
                if message.get("error"):
                    raise UploadError(
                        f"Server error: {message.get('detail', 'Unknown error')}"
                    )

                # Update progress
                if "progress" in message and progress_bar:
                    progress_bar.update(message["progress"] - progress_bar.n)

                # Extract upload_id
                if "upload_id" in message:
                    upload_id = message["upload_id"]
                    already_exists = message.get("already_exists", False)

                # Print status messages
                if "message" in message and not show_progress:
                    print(f"Server: {message['message']}")

        finally:
            if progress_bar:
                progress_bar.close()

        if not upload_id:
            raise UploadError("No upload ID received from server")

        if already_exists:
            print(f"Model already exists with upload_id: {upload_id}")
        else:
            print(f"Model uploaded successfully with upload_id: {upload_id}")

        return upload_id