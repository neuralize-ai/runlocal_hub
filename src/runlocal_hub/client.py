"""
Simplified RunLocal API client.
"""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from tqdm import tqdm

import numpy as np

from .devices import DeviceFilters, DeviceSelector
from .exceptions import ConfigurationError, RunLocalError, UploadError, ValidationError
from .http import HTTPClient
from .jobs import JobPoller
from .models import (
    DeviceUsage,
    IOType,
    JobType,
    PredictionResult,
    BenchmarkResult,
    Device,
    BenchmarkData,
    BenchmarkDataFloat,
)
from .tensors import TensorHandler
from .utils.decorators import handle_api_errors


class RunLocalClient:
    """
    Simplified Python client for the RunLocal API.
    """

    BASE_URL = "https://neuralize-bench.com"
    # BASE_URL = "http://127.0.0.1:8000"  # Local development
    ENV_VAR_NAME = "RUNLOCAL_API_KEY"

    def __init__(
        self,
        debug: bool = False,
    ):
        """
        Initialize the RunLocal client.

        Args:
            api_key: API key for authentication. If not provided, will look for RUNLOCAL_API_KEY env var
            base_url: Base URL for the API. Defaults to production URL
            debug: Enable debug logging
        """
        api_key = os.environ.get(self.ENV_VAR_NAME)
        if not api_key:
            raise ConfigurationError(
                f"API key not found. Please set the {self.ENV_VAR_NAME} environment variable.",
                config_key=self.ENV_VAR_NAME,
                suggestion=f"export {self.ENV_VAR_NAME}=your-api-key-here",
            )

        # Initialize HTTP client
        self.http_client = HTTPClient(
            base_url=self.BASE_URL, api_key=api_key, debug=debug
        )
        self.debug = debug

        # Initialize components
        self.device_selector = DeviceSelector(self.http_client)
        self.tensor_handler = TensorHandler(self.http_client)
        self.job_poller = JobPoller(self.http_client)

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
        model_pipeline_id: Optional[str] = None,
        torchscript_upload_id: Optional[str] = None,
        keras_upload_id: Optional[str] = None,
        show_progress: bool = True,
    ) -> str:
        """
        Upload a model file or folder to the RunLocal platform.

        Args:
            model_path: Path to the model file or folder to upload
            model_pipeline_id: Optional model pipeline to link the model to
            torchscript_upload_id: Optional torchscript model to link the model to
            keras_upload_id: Optional tensorflow model to link the model to
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

        upload_filename = model_path.name

        # Create temporary directory for zipping
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Zip the model
            if self.debug:
                print(f"Zipping {model_path}...")

            zip_path = self._zip_path(model_path, temp_path)
            zip_size = zip_path.stat().st_size

            if self.debug:
                print(
                    f"Zip file created: {zip_path.name} ({zip_size / 1024 / 1024:.2f} MB)"
                )

            # Prepare upload parameters
            params = {
                "upload_filename": upload_filename,
                "upload_source_type": "USER_UPLOADED",
            }

            if model_pipeline_id is not None:
                params["model_pipeline_id"] = model_pipeline_id

            if torchscript_upload_id is not None:
                params["torch_script_upload_id"] = torchscript_upload_id

            if keras_upload_id is not None:
                params["keras_upload_id"] = keras_upload_id

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
            print(f"Model already exists with upload_id: {upload_id}\n")
        else:
            print(f"Model uploaded successfully with upload_id: {upload_id}\n")

        return upload_id

    def benchmark(
        self,
        model_path: Optional[Union[Path, str]] = None,
        model_id: Optional[str] = None,
        device_filters: Optional[Union[DeviceFilters, List[DeviceFilters]]] = None,
        inputs: Optional[Dict[str, np.ndarray]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        show_progress: bool = True,
        device_count: Optional[int] = 1,
        output_dir: Optional[Union[str, Path]] = None,
        skip_output_download: bool = False,
    ) -> Union[BenchmarkResult, List[BenchmarkResult]]:
        """
        Benchmark a model with clean, user-friendly API.

        Args:
            model_path: Path to the model file or folder (if model_id not provided)
            model_id: ID of already uploaded model (if model_path not provided)
            device_filters: Optional filters for device selection. Can be a single DeviceFilters
                          object or a list of DeviceFilters to apply with OR logic (union)
            inputs: Optional dictionary mapping input names to numpy arrays
            timeout: Maximum time in seconds to wait for completion
            poll_interval: Time in seconds between status checks
            show_progress: Whether to show upload progress bar
            device_count: Number of devices to benchmark on (None = all, 1 = single result, >1 = list)
            output_dir: Directory to save output tensors (defaults to ./outputs/)
            skip_output_download: If True, skip downloading output tensors even if inputs are provided

        Returns:
            BenchmarkResult object(s) containing device info and performance data
            (single BenchmarkResult if device_count=1, list of BenchmarkResult otherwise)
            Output tensors are saved as file paths (unless skip_output_download=True)

        Raises:
            ValueError: If neither model_path nor model_id is provided
            RunLocalError: If benchmark fails
        """
        # Validate input parameters
        if model_path is None and model_id is None:
            raise ValidationError(
                "Either model_path or model_id must be provided",
                parameter="model_path/model_id",
                expected="one of model_path or model_id",
                got="neither provided",
            )
        if model_path is not None and model_id is not None:
            raise ValidationError(
                "Only one of model_path or model_id should be provided",
                parameter="model_path/model_id",
                expected="either model_path or model_id",
                got="both provided",
            )

        # Upload model if path provided
        if model_path is not None:
            if self.debug:
                print(f"Uploading model from {model_path}...")
            model_id = self.upload_model(model_path, show_progress=show_progress)

        if model_id is None:
            raise RunLocalError("Model upload failed")

        # Select devices using our device selector
        if self.debug:
            print("Selecting devices...")

        user_models = self.get_models()
        devices = self.device_selector.select_devices(
            model_id=model_id,
            filters=device_filters,
            count=device_count,
            user_models=user_models,
        )

        # Display selected devices
        self.device_selector.display_selected_devices(devices)

        # Run benchmarks using our components
        return self._run_benchmarks(
            model_id=model_id,
            devices=devices,
            inputs=inputs,
            timeout=timeout,
            poll_interval=poll_interval,
            output_dir=output_dir,
            skip_output_download=skip_output_download,
        )

    def predict(
        self,
        inputs: Dict[str, np.ndarray],
        model_path: Optional[Union[Path, str]] = None,
        model_id: Optional[str] = None,
        device_filters: Optional[Union[DeviceFilters, List[DeviceFilters]]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        show_progress: bool = True,
        device_count: Optional[int] = 1,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Run prediction on a model with clean, user-friendly API.

        Args:
            inputs: Dictionary mapping input names to numpy arrays
            model_path: Path to the model file or folder (if model_id not provided)
            model_id: ID of already uploaded model (if model_path not provided)
            device_filters: Optional filters for device selection. Can be a single DeviceFilters
                          object or a list of DeviceFilters to apply with OR logic (union)
            timeout: Maximum time in seconds to wait for completion
            poll_interval: Time in seconds between status checks
            show_progress: Whether to show upload progress bar
            device_count: Number of devices to run prediction on (None = all, 1 = single result, >1 = list)
            output_dir: Directory to save output tensors (defaults to ./outputs)

        Returns:
            PredictionResult object(s) containing device info and output tensor file paths
            (single PredictionResult if device_count=1, list of PredictionResult otherwise)

        Raises:
            ValueError: If neither model_path nor model_id is provided
            RunLocalError: If prediction fails
        """
        # Validate input parameters
        if model_path is None and model_id is None:
            raise ValidationError(
                "Either model_path or model_id must be provided",
                parameter="model_path/model_id",
                expected="one of model_path or model_id",
                got="neither provided",
            )
        if model_path is not None and model_id is not None:
            raise ValidationError(
                "Only one of model_path or model_id should be provided",
                parameter="model_path/model_id",
                expected="either model_path or model_id",
                got="both provided",
            )

        # Upload model if path provided
        if model_path is not None:
            if self.debug:
                print(f"Uploading model from {model_path}...")
            model_id = self.upload_model(model_path, show_progress=show_progress)

        if model_id is None:
            raise RunLocalError("Model upload failed")

        # Select devices using our device selector
        if self.debug:
            print("Selecting devices...")

        user_models = self.get_models()
        devices = self.device_selector.select_devices(
            model_id=model_id,
            filters=device_filters,
            count=device_count,
            user_models=user_models,
        )

        # Display selected devices
        self.device_selector.display_selected_devices(devices)

        # Run predictions using our components
        return self._run_predictions(
            model_id=model_id,
            devices=devices,
            inputs=inputs,
            timeout=timeout,
            poll_interval=poll_interval,
            output_dir=output_dir,
        )

    def _run_benchmarks(
        self,
        model_id: str,
        devices: List[DeviceUsage],
        inputs: Optional[Dict[str, np.ndarray]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        output_dir: Optional[Union[str, Path]] = None,
        skip_output_download: bool = False,
    ) -> Union[BenchmarkResult, List[BenchmarkResult]]:
        """
        Internal method to run benchmarks using refactored components.
        """
        # Upload input tensors if provided
        input_tensors_id = None
        if inputs is not None:
            if self.debug:
                print("Uploading input tensors for benchmarks...")
                for name, tensor in inputs.items():
                    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

            input_tensors_id = self.tensor_handler.upload_tensors(
                inputs, io_type=IOType.INPUT
            )

        # Create device requests for all devices
        device_requests = []
        for device_usage in devices:
            device_id = device_usage.device.to_device_id()
            device_request = {
                "device_id": device_id,
                "compute_units": device_usage.compute_units,
            }
            device_requests.append(device_request)

        # Prepare the data payload
        data: Dict[str, Any] = {
            "device_requests": device_requests,
        }

        # Add input tensors to the payload if provided
        if input_tensors_id is not None:
            data["input_tensors_id"] = input_tensors_id

        # Submit all benchmarks at once
        response = self.http_client.post(
            f"/coreml/benchmark/enqueue?upload_id={model_id}",
            data=data,
        )

        # Extract benchmark IDs from the response
        benchmark_ids = list(response)

        if self.debug:
            print(f"Benchmarks submitted with IDs: {benchmark_ids}")

        # Configure poller with custom interval
        self.job_poller.poll_interval = poll_interval

        # Poll for benchmark completion using our job poller
        device_names = [device.device.Name for device in devices]
        results = self.job_poller.poll_jobs(
            job_ids=benchmark_ids,
            job_type=JobType.BENCHMARK,
            device_names=device_names,
            timeout=timeout,
        )

        # Process results and download output tensors if needed
        processed_results = []
        for result in results:
            if result.is_successful and result.data:
                # Extract device info and benchmark data
                device_info = result.data.get("DeviceInfo", {})
                device = Device(**device_info)

                # Convert benchmark data to float format
                benchmark_data = []
                for bd in result.data.get("BenchmarkData", []):
                    original_bd = BenchmarkData(**bd)
                    benchmark_data.append(
                        BenchmarkDataFloat.from_benchmark_data(original_bd)
                    )

                # Download output tensors if inputs were provided and skip_output_download is False
                output_tensors = None
                if inputs is not None and not skip_output_download:
                    output_tensors = {}
                    for bd in result.data["BenchmarkData"]:
                        if bd.get("Success") and bd.get("OutputTensorsId"):
                            compute_unit = bd["ComputeUnit"]
                            output_tensors_id = bd["OutputTensorsId"]

                            if self.debug:
                                print(
                                    f"Downloading outputs for compute unit '{compute_unit}' (tensor ID: {output_tensors_id})"
                                )

                            output_tensors[compute_unit] = (
                                self.tensor_handler.download_tensors(
                                    output_tensors_id, output_dir=output_dir
                                )
                            )

                benchmark_result = BenchmarkResult(
                    device=device,
                    benchmark_data=benchmark_data,
                    job_id=result.job_id,
                    status=result.data.get("Status", "Unknown"),
                    modelid=result.data.get("UploadId", ""),
                    outputs=output_tensors,
                )
                processed_results.append(benchmark_result)
            else:
                # Don't raise error for individual failures when we have partial results
                # Just log the error and continue
                if self.debug:
                    print(
                        f"Warning: Benchmark failed for device {result.device_name}: {result.error}"
                    )

        # Check if we have at least some results
        if not processed_results and len(results) < len(benchmark_ids):
            print(
                f"\n⚠️  Warning: No benchmarks completed successfully. {len(benchmark_ids) - len(results)} jobs timed out."
            )

        # Return single result or list based on device count
        if len(devices) == 1:
            # For single device, return the result if available, otherwise raise error
            if processed_results:
                return processed_results[0]
            else:
                raise RunLocalError(
                    "No benchmark results available - all jobs failed or timed out"
                )
        else:
            # For multiple devices, always return a list (could be empty or partial)
            return processed_results

    def _run_predictions(
        self,
        model_id: str,
        devices: List[DeviceUsage],
        inputs: Dict[str, np.ndarray],
        timeout: int = 600,
        poll_interval: int = 10,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Internal method to run predictions using refactored components.
        """
        # Upload input tensors (required for predictions)
        if self.debug:
            print("Uploading input tensors...")
            for name, tensor in inputs.items():
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

        input_tensors_id = self.tensor_handler.upload_tensors(
            inputs, io_type=IOType.INPUT
        )

        # Create device requests for all devices
        device_requests = []
        for device_usage in devices:
            device_id = device_usage.device.to_device_id()
            device_request = {
                "device_id": device_id,
                "compute_units": device_usage.compute_units,
            }
            device_requests.append(device_request)

        # Prepare the data payload for prediction jobs
        data = {
            "device_requests": device_requests,
            "input_tensors_id": input_tensors_id,
            "job_type": JobType.PREDICTION.value,
        }

        # Submit all prediction jobs at once
        response = self.http_client.post(
            f"/coreml/benchmark/enqueue?upload_id={model_id}",
            data=data,
        )

        # Extract the benchmark IDs
        benchmark_ids = list(response)

        if self.debug:
            print(f"Prediction jobs submitted with IDs: {benchmark_ids}")

        # Configure poller with custom interval
        self.job_poller.poll_interval = poll_interval

        # Poll for prediction completion using our job poller
        device_names = [device.device.Name for device in devices]
        results = self.job_poller.poll_jobs(
            job_ids=benchmark_ids,
            job_type=JobType.PREDICTION,
            device_names=device_names,
            timeout=timeout,
        )

        # Process results and download output tensors
        processed_results = []
        for result in results:
            if result.is_successful and result.data is not None:
                # Extract output tensor IDs from all compute units
                compute_unit_outputs = {}
                for benchmark_data in result.data["BenchmarkData"]:
                    if benchmark_data.get("Success") and benchmark_data.get(
                        "OutputTensorsId"
                    ):
                        compute_unit = benchmark_data["ComputeUnit"]
                        output_tensors_id = benchmark_data["OutputTensorsId"]

                        if self.debug:
                            print(
                                f"Downloading outputs for compute unit '{compute_unit}' (tensor ID: {output_tensors_id})"
                            )

                        # Download output tensors for this compute unit
                        compute_unit_outputs[compute_unit] = (
                            self.tensor_handler.download_tensors(
                                output_tensors_id, output_dir=output_dir
                            )
                        )

                if compute_unit_outputs:
                    # Extract device info from the data
                    device_info = result.data.get("DeviceInfo", {})
                    device = Device(**device_info)

                    prediction_result = PredictionResult(
                        device=device,
                        outputs=compute_unit_outputs,
                        job_id=result.job_id,
                        status=result.data.get("Status", "Unknown"),
                        modelid=result.data.get("UploadId", ""),
                    )
                    processed_results.append(prediction_result)
                else:
                    if self.debug:
                        print(
                            f"Warning: Prediction completed but no output tensors found for device {result.device_name}"
                        )
            else:
                # Don't raise error for individual failures when we have partial results
                # Just log the error and continue
                if self.debug:
                    print(
                        f"Warning: Prediction failed for device {result.device_name}: {result.error}"
                    )

        # Check if we have at least some results
        if not processed_results and len(results) < len(benchmark_ids):
            print(
                f"\n⚠️  Warning: No predictions completed successfully. {len(benchmark_ids) - len(results)} jobs timed out."
            )

        # Return single result or list based on device count
        if len(devices) == 1:
            # For single device, return the result if available, otherwise raise error
            if processed_results:
                return processed_results[0]
            else:
                raise RunLocalError(
                    "No prediction results available - all jobs failed or timed out"
                )
        else:
            # For multiple devices, always return a list (could be empty or partial)
            return processed_results
