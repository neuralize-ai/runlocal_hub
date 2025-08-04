"""
Simplified RunLocal API client.
"""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from runlocal_hub.models import (
    BenchmarkDbItem,
    BenchmarkRequest,
    BenchmarkStatus,
    RuntimeSettings,
)
from runlocal_hub.models.job import JobResult
from runlocal_hub.models.model import UploadDbItem
from runlocal_hub.utils.display import display_incomplete_panel
from runlocal_hub.utils.json import convert_to_json_friendly

from .devices import DeviceFilters, DeviceSelector
from .exceptions import ConfigurationError, RunLocalError, UploadError, ValidationError
from .http import HTTPClient
from .jobs import JobPoller
from .models import (
    BenchmarkData,
    BenchmarkDataFloat,
    BenchmarkResponse,
    BenchmarkResult,
    Device,
    DeviceUsage,
    IOType,
    JobType,
    PredictionResponse,
    PredictionResult,
)
from .tensors import TensorHandler
from .utils.decorators import handle_api_errors


class RunLocalClient:
    """
    Simplified Python client for the RunLocal API.
    """

    BASE_URL = "https://neuralize-bench.com"
    ENV_VAR_NAME = "RUNLOCAL_API_KEY"

    def __init__(
        self,
        api_key: Optional[str] = None,
        local_server: bool = False,
        verbosity: int = 2,
    ):
        """
        Initialize the RunLocal client.

        Args:
            api_key: Manually specify the RunLocal API key to use
            debug: Enable debug logging (deprecated, use verbosity=3 instead)
            local_server: Use local server instead of production
            verbosity: Control output verbosity (0=silent, 1=minimal, 2=normal 3=verbose, 4=debug). Uses instance default if not specified
        """
        if api_key is None:
            api_key = os.environ.get(self.ENV_VAR_NAME)

        if not api_key:
            raise ConfigurationError(
                f"API key not found. Please set the {self.ENV_VAR_NAME} environment variable.",
                config_key=self.ENV_VAR_NAME,
                suggestion=f"export {self.ENV_VAR_NAME}=your-api-key-here",
            )

        if local_server:
            self.BASE_URL = "http://127.0.0.1:8000"

        # Store verbosity as instance variable
        self.verbosity = verbosity

        # Initialize HTTP client
        self.http_client = HTTPClient(
            base_url=self.BASE_URL, api_key=api_key, debug=verbosity >= 4
        )

        # Initialize components
        self.device_selector = DeviceSelector(self.http_client)
        self.tensor_handler = TensorHandler(self.http_client)
        self.job_poller = JobPoller(self.http_client, verbosity=self.verbosity)

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
    def get_models_ids(self) -> List[str]:
        """
        Get a list of model IDs for the authenticated user.

        Returns:
            List of model IDs

        Raises:
            AuthenticationError: If the API key is invalid
        """
        user_data = self.get_user_info()
        return user_data.get("UploadIds", [])

    @handle_api_errors
    def get_model(self, model_id: str) -> UploadDbItem:
        """
        Get a model from an ID

        Args:
            model_id: ID of the model to fetch

        Returns:
            Model info object

        Raises:
            AuthenticationError: If the API key is invalid
        """
        response = self.http_client.get(f"/uploads/model/coreml/{model_id}")
        return UploadDbItem(**response)

    @handle_api_errors
    def get_models(self) -> List[UploadDbItem]:
        """
        Get all models user models.

        Returns:
            List of models

        Raises:
            AuthenticationError: If the API key is invalid
        """
        response = self.http_client.get("/uploads/user")
        models: List[UploadDbItem] = []

        for item in response:
            models.append(UploadDbItem(**item))

        return models

    @handle_api_errors
    def get_model_benchmarks(self, model: str | UploadDbItem) -> List[BenchmarkResult]:
        """
        Get a benchmark table data for a model

        Args:
            model: model object or ID of the model to fetch benchmark results for

        Returns:
            List of Benchmark results for the model

        Raises:
            AuthenticationError: If the API key is invalid
        """

        model_id: str
        if isinstance(model, str):
            model_id = model
        else:
            model_id = model.UploadId

        response = self.http_client.get(f"/benchmarks/model/{model_id}")
        results: List[BenchmarkResult] = []
        for item in response:
            benchmark_item = BenchmarkDbItem(**item)
            if benchmark_item.Status != BenchmarkStatus.Complete:
                continue

            result = convert_to_json_friendly(benchmark_item)
            device = result.get("DeviceInfo", {})

            # Convert benchmark data to float format
            benchmark_data = []
            for bd in result.get("BenchmarkData", []):
                original_bd = BenchmarkData(**bd)
                benchmark_data.append(
                    BenchmarkDataFloat.from_benchmark_data(original_bd)
                )
            benchmark_result = BenchmarkResult(
                device=device,
                benchmark_data=benchmark_data,
            )

            results.append(benchmark_result)

        return results

    def upload_model(
        self,
        model_path: Union[Path, str],
        model_pipeline_id: Optional[str] = None,
        torchscript_upload_id: Optional[str] = None,
        keras_upload_id: Optional[str] = None,
    ) -> str:
        """
        Upload a model file or folder to the RunLocal platform.

        Args:
            model_path: Path to the model file or folder to upload
            model_pipeline_id: Optional model pipeline to link the model to
            torchscript_upload_id: Optional torchscript model to link the model to
            keras_upload_id: Optional tensorflow model to link the model to

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
            if self.verbosity >= 4:
                print(f"Zipping {model_path}...")

            zip_path = self._zip_path(model_path, temp_path)
            zip_size = zip_path.stat().st_size

            if self.verbosity >= 4:
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

            if self.verbosity >= 4:
                print("Uploading to server...")

            # Upload and process response
            return self._process_upload_stream(
                endpoint="/uploads/model/coreml",
                data=zip_data,
                params=params,
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
    ) -> str:
        """
        Process a streaming upload response.

        Args:
            endpoint: API endpoint
            data: Binary data to upload
            params: Query parameters

        Returns:
            Upload ID

        Raises:
            UploadError: If upload fails
        """
        upload_id = None
        already_exists = False
        progress_bar = None

        if self.verbosity >= 2:
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
                if "message" in message and self.verbosity >= 3:
                    print(f"Server: {message['message']}")

        finally:
            if progress_bar:
                progress_bar.close()

        if not upload_id:
            raise UploadError("No upload ID received from server")

        if self.verbosity >= 1:
            if already_exists:
                print(f"Model already exists with upload_id: {upload_id}\n")
            else:
                print(f"Model uploaded successfully with upload_id: {upload_id}\n")

        return upload_id

    def benchmark(
        self,
        model_path: Optional[Union[Path, str]] = None,
        model_id: Optional[str] = None,
        settings: Optional[RuntimeSettings] = None,
        device_filters: Optional[Union[DeviceFilters, List[DeviceFilters]]] = None,
        inputs: Optional[Dict[str, np.ndarray]] = None,
        timeout: Optional[int] = 600,
        poll_interval: int = 10,
        device_count: Optional[int] = 1,
        output_dir: Optional[Union[str, Path]] = None,
        skip_output_download: bool = False,
        skip_existing: bool = False,
    ) -> BenchmarkResponse:
        """
        Benchmark a model with clean, user-friendly API.

        Args:
            model_path: Path to the model file or folder (if model_id not provided)
            model_id: ID of already uploaded model (if model_path not provided)
            settings: Optional manual override of runtime framework and runtime framework settings configuration
            device_filters: Optional filters for device selection. Can be a single DeviceFilters
                          object or a list of DeviceFilters to apply with OR logic (union)
            inputs: Optional dictionary mapping input names to numpy arrays
            timeout: Maximum time in seconds to wait for completion
            poll_interval: Time in seconds between status checks
            device_count: Number of devices to benchmark on (None = all, 1 = single result, >1 = list)
            output_dir: Directory to save output tensors (defaults to ./outputs/)
            skip_output_download: If True, skip downloading output tensors even if inputs are provided
            skip_existing: If True, skip benchmarks that have already been run on the same device/model combination

        Returns:
            BenchmarkResponse containing:
            - results: BenchmarkResult object (single device) or List[BenchmarkResult] (multiple devices)
            - all_job_ids: List of all submitted job IDs
            - completed_job_ids: List of job IDs that completed
            - incomplete_job_ids: List of job IDs that didn't complete within timeout

            Use response.incomplete_job_ids to check for timed-out jobs
            Use client.check_multiple_jobs(response.incomplete_job_ids) to check status later
            Use client.get_benchmark_results(job_ids, timeout=None) to wait and process jobs

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
            if self.verbosity >= 4:
                print(f"Uploading model from {model_path}...")
            model_id = self.upload_model(model_path)

        if model_id is None:
            raise RunLocalError("Model upload failed")

        # Select devices using our device selector
        if self.verbosity >= 4:
            print("Selecting devices...")

        user_models = self.get_models_ids()
        devices = self.device_selector.select_devices(
            model_id=model_id,
            framework=settings.framework if settings is not None else None,
            filters=device_filters,
            count=device_count,
            user_models=user_models,
        )

        # Display selected devices
        if self.verbosity >= 2:
            self.device_selector.display_selected_devices(devices)

        # Run benchmarks using our components
        return self._run_benchmarks(
            model_id=model_id,
            devices=devices,
            settings=settings,
            inputs=inputs,
            timeout=timeout,
            poll_interval=poll_interval,
            output_dir=output_dir,
            skip_output_download=skip_output_download,
            skip_existing=skip_existing,
        )

    def predict(
        self,
        inputs: Dict[str, np.ndarray],
        model_path: Optional[Union[Path, str]] = None,
        model_id: Optional[str] = None,
        settings: Optional[RuntimeSettings] = None,
        device_filters: Optional[Union[DeviceFilters, List[DeviceFilters]]] = None,
        timeout: Optional[int] = 600,
        poll_interval: int = 10,
        device_count: Optional[int] = 1,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PredictionResponse:
        """
        Run prediction on a model with clean, user-friendly API.

        Args:
            inputs: Dictionary mapping input names to numpy arrays
            model_path: Path to the model file or folder (if model_id not provided)
            model_id: ID of already uploaded model (if model_path not provided)
            settings: Optional manual override of runtime framework and runtime framework settings configuration
            device_filters: Optional filters for device selection. Can be a single DeviceFilters
                          object or a list of DeviceFilters to apply with OR logic (union)
            timeout: Maximum time in seconds to wait for completion
            poll_interval: Time in seconds between status checks
            device_count: Number of devices to run prediction on (None = all, 1 = single result, >1 = list)
            output_dir: Directory to save output tensors (defaults to ./outputs)

        Returns:
            PredictionResponse containing:
            - results: PredictionResult object (single device) or List[PredictionResult] (multiple devices)
            - all_job_ids: List of all submitted job IDs
            - completed_job_ids: List of job IDs that completed
            - incomplete_job_ids: List of job IDs that didn't complete within timeout

            Use response.incomplete_job_ids to check for timed-out jobs
            Use client.check_multiple_jobs(response.incomplete_job_ids) to check status later
            Use client.get_prediction_results(job_ids, timeout=None) to wait and process jobs

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
            if self.verbosity >= 4:
                print(f"Uploading model from {model_path}...")
            model_id = self.upload_model(model_path)

        if model_id is None:
            raise RunLocalError("Model upload failed")

        # Select devices using our device selector
        if self.verbosity >= 4:
            print("Selecting devices...")

        user_models = self.get_models_ids()
        devices = self.device_selector.select_devices(
            model_id=model_id,
            framework=settings.framework if settings is not None else None,
            filters=device_filters,
            count=device_count,
            user_models=user_models,
        )

        # Display selected devices
        if self.verbosity >= 2:
            self.device_selector.display_selected_devices(devices)

        # Run predictions using our components
        return self._run_predictions(
            model_id=model_id,
            devices=devices,
            settings=settings,
            inputs=inputs,
            timeout=timeout,
            poll_interval=poll_interval,
            output_dir=output_dir,
        )

    def _run_benchmarks(
        self,
        model_id: str,
        devices: List[DeviceUsage],
        settings: Optional[RuntimeSettings] = None,
        inputs: Optional[Dict[str, np.ndarray]] = None,
        timeout: Optional[int] = 600,
        poll_interval: int = 10,
        output_dir: Optional[Union[str, Path]] = None,
        skip_output_download: bool = False,
        skip_existing: bool = False,
    ) -> BenchmarkResponse:
        """
        Internal method to run benchmarks using refactored components.
        """
        # Submit jobs using the new common method
        job_ids = self._submit_jobs(
            model_id=model_id,
            devices=devices,
            settings=settings,
            inputs=inputs,
            job_type=JobType.BENCHMARK,
            skip_existing=skip_existing,
        )

        # Configure poller with custom interval
        self.job_poller.poll_interval = poll_interval

        # Poll for benchmark completion using our job poller
        results = self.job_poller.poll_jobs(
            job_ids=job_ids,
            job_type=JobType.BENCHMARK,
            timeout=timeout,
        )

        # Process results using the new common method
        processed_results = self._process_benchmark_results(
            results=results,
            output_dir=output_dir,
            skip_output_download=inputs is not None and not skip_output_download,
        )

        # Collect job tracking information
        completed_job_ids = [result.job_id for result in results]
        incomplete_job_ids = [
            job_id for job_id in job_ids if job_id not in completed_job_ids
        ]

        if incomplete_job_ids:
            self._print_incomplete_message(incomplete_job_ids)

        # Create and return response wrapper
        return BenchmarkResponse(
            results=processed_results[0]
            if processed_results and len(processed_results) == 1
            else processed_results,
            all_job_ids=job_ids,
            completed_job_ids=completed_job_ids,
            incomplete_job_ids=incomplete_job_ids,
        )

    def _run_predictions(
        self,
        model_id: str,
        devices: List[DeviceUsage],
        inputs: Dict[str, np.ndarray],
        settings: Optional[RuntimeSettings] = None,
        timeout: Optional[int] = 600,
        poll_interval: int = 10,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PredictionResponse:
        """
        Internal method to run predictions using refactored components.
        """
        # Submit jobs using the new common method
        job_ids = self._submit_jobs(
            model_id=model_id,
            devices=devices,
            settings=settings,
            inputs=inputs,
            job_type=JobType.PREDICTION,
        )

        # Configure poller with custom interval
        self.job_poller.poll_interval = poll_interval

        # Poll for prediction completion using our job poller
        results = self.job_poller.poll_jobs(
            job_ids=job_ids,
            job_type=JobType.PREDICTION,
            timeout=timeout,
        )

        # Process results using the new common method
        processed_results = self._process_prediction_results(
            results=results,
            output_dir=output_dir,
        )

        # Collect job tracking information
        completed_job_ids = [result.job_id for result in results]
        incomplete_job_ids = [
            job_id for job_id in job_ids if job_id not in completed_job_ids
        ]

        if incomplete_job_ids:
            self._print_incomplete_message(
                incomplete_job_ids, job_type=JobType.PREDICTION
            )

        # Create and return response wrapper
        return PredictionResponse(
            results=processed_results[0]
            if processed_results and len(processed_results) == 1
            else processed_results,
            all_job_ids=job_ids,
            completed_job_ids=completed_job_ids,
            incomplete_job_ids=incomplete_job_ids,
        )

    def _submit_jobs(
        self,
        model_id: str,
        devices: List[DeviceUsage],
        settings: Optional[RuntimeSettings] = None,
        inputs: Optional[Dict[str, np.ndarray]] = None,
        job_type: JobType = JobType.BENCHMARK,
        skip_existing: bool = False,
    ) -> List[str]:
        """
        Internal method to submit jobs to the API and return job IDs.
        """
        # Upload input tensors if provided
        input_tensors_id = None
        if inputs is not None:
            if self.verbosity >= 4:
                print(f"Uploading input tensors for {job_type.value}...")
                for name, tensor in inputs.items():
                    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

            input_tensors_id = self.tensor_handler.upload_tensors(
                inputs, io_type=IOType.INPUT
            )

        # Create device requests for all devices
        device_requests = []
        for device_usage in devices:
            device_request = {
                "device_id": device_usage.native_device_id,
                "compute_units": device_usage.compute_units,
            }
            device_requests.append(device_request)

        # Create benchmark request (works for both benchmarks and predictions)
        benchmark_request: BenchmarkRequest = BenchmarkRequest(
            device_requests=device_requests,
            settings=settings,
            job_type=job_type,
            skip_existing=skip_existing,
        )

        # Add input tensors to the payload if provided
        if input_tensors_id is not None:
            benchmark_request.input_tensors_id = input_tensors_id

        # Submit all jobs at once
        response = self.http_client.post(
            f"/coreml/benchmark/enqueue?upload_id={model_id}",
            data=benchmark_request.model_dump(),
        )

        # Extract job IDs from the response
        job_ids = list(response)

        if self.verbosity >= 4:
            print(f"{job_type.value.title()}s submitted with IDs: {job_ids}")

        return job_ids

    def _process_benchmark_results(
        self,
        results: List[JobResult],
        output_dir: Optional[Union[str, Path]] = None,
        skip_output_download: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Process job results into benchmark results with tensor downloads.

        Args:
            results: List of job results from polling
            output_dir: Directory to save output tensors
            skip_output_download: Skip downloading output tensors

        Returns:
            List of processed benchmark results
        """
        processed_results = []

        for result in results:
            if result.is_successful and result.data:
                # Extract device info and benchmark data
                if result.device is not None:
                    device = result.device
                else:
                    device_info = result.data.get("DeviceInfo", {})
                    device = Device(**device_info)

                # Convert benchmark data to float format
                benchmark_data = []
                for bd in result.data.get("BenchmarkData", []):
                    original_bd = BenchmarkData(**bd)
                    benchmark_data.append(
                        BenchmarkDataFloat.from_benchmark_data(original_bd)
                    )

                output_tensors = None
                if not skip_output_download:
                    output_tensors = {}
                    for bd in result.data["BenchmarkData"]:
                        if bd.get("Success") and bd.get("OutputTensorsId"):
                            compute_unit = bd["ComputeUnit"]
                            output_tensors_id = bd["OutputTensorsId"]

                            if self.verbosity >= 4:
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
                    outputs=output_tensors,
                )
                processed_results.append(benchmark_result)
            else:
                # Log the error but don't include failed results
                if self.verbosity >= 2:
                    print(
                        f"Warning: Benchmark failed for device {result.device_name}: {result.error}"
                    )

        return processed_results

    def _process_prediction_results(
        self,
        results: List[JobResult],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[PredictionResult]:
        """
        Process job results into prediction results with tensor downloads.

        Args:
            results: List of job results from polling
            output_dir: Directory to save output tensors

        Returns:
            List of processed prediction results
        """
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

                        if self.verbosity >= 4:
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
                    if result.device is not None:
                        device = result.device
                    else:
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
                    if self.verbosity >= 2:
                        print(
                            f"Warning: Prediction completed but no output tensors found for device {result.device_name}"
                        )
            else:
                # Log the error but don't include failed results
                if self.verbosity >= 2:
                    print(
                        f"Warning: Prediction failed for device {result.device_name}: {result.error}"
                    )

        return processed_results

    def _print_incomplete_message(
        self, incomplete_job_ids: List[str], job_type: JobType = JobType.BENCHMARK
    ):
        if self.verbosity >= 2:
            display_incomplete_panel(incomplete_job_ids, job_type.value)
        elif self.verbosity == 1:
            print(
                f"Some jobs did not complete within the timeout: {incomplete_job_ids}.\nThey are accessible with response.incomplete_job_ids"
            )

    @handle_api_errors
    def check_job_status(self, job_id: str) -> JobResult:
        """
        Check the status of a single job.

        Args:
            job_id: Job ID to check

        Returns:
            Current job status and data

        Raises:
            RunLocalError: If API request fails
        """
        return self.job_poller._check_job_status(job_id)

    @handle_api_errors
    def check_multiple_jobs(self, job_ids: List[str]) -> List[JobResult]:
        """
        Check the status of multiple jobs.

        Args:
            job_ids: List of job IDs to check

        Returns:
            List of current job statuses and data

        Raises:
            RunLocalError: If API request fails
        """
        results = []
        for job_id in job_ids:
            try:
                result = self.job_poller._check_job_status(job_id)
                results.append(result)
            except Exception as e:
                # Create a failed result for this job
                result = JobResult(
                    job_id=job_id, status=BenchmarkStatus.Failed, error=str(e)
                )
                results.append(result)
        return results

    def get_benchmark_results(
        self,
        job_ids: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        skip_output_download: bool = True,
        timeout: Optional[int] = None,
        poll_interval: int = 10,
    ) -> BenchmarkResponse:
        """
        Get processed benchmark results from job IDs, optionally waiting for completion.

        Args:
            job_ids: List of job IDs to process
            output_dir: Directory to save output tensors
            skip_output_download: Skip downloading output tensors
            timeout: Maximum time in seconds to wait for incomplete jobs
            poll_interval: Time in seconds between status checks when waiting

        Returns:
            BenchmarkResponse containing:
            - results: List[BenchmarkResult] with completed jobs
            - all_job_ids: All job IDs that were checked
            - completed_job_ids: Job IDs that completed successfully
            - incomplete_job_ids: Job IDs that didn't complete (still running/failed)

        Raises:
            RunLocalError: If no jobs completed and timeout specified
        """
        if self.verbosity >= 4:
            print(f"Waiting up to {timeout}s for jobs to complete...")

        # Configure poller with custom interval
        self.job_poller.poll_interval = poll_interval

        # Poll for job completion
        job_results = self.job_poller.poll_jobs(
            job_ids=job_ids,
            job_type=JobType.BENCHMARK,  # Assume benchmark jobs
            timeout=timeout,
        )

        # Filter to only completed jobs
        completed_results = [r for r in job_results if r.is_successful]

        # Process the completed results
        processed_results = self._process_benchmark_results(
            completed_results,
            output_dir=output_dir,
            skip_output_download=skip_output_download,
        )

        # Collect job tracking information
        completed_job_ids = [r.job_id for r in completed_results]
        incomplete_job_ids = [
            job_id for job_id in job_ids if job_id not in completed_job_ids
        ]

        if incomplete_job_ids:
            self._print_incomplete_message(incomplete_job_ids)

        # Create and return response wrapper
        return BenchmarkResponse(
            results=processed_results,
            all_job_ids=job_ids,
            completed_job_ids=completed_job_ids,
            incomplete_job_ids=incomplete_job_ids,
        )

    def get_prediction_results(
        self,
        job_ids: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        timeout: Optional[int] = None,
        poll_interval: int = 10,
    ) -> PredictionResponse:
        """
        Get processed prediction results from job IDs, optionally waiting for completion.

        Args:
            job_ids: List of job IDs to process
            output_dir: Directory to save output tensors
            timeout: Maximum time in seconds to wait for incomplete jobs
            poll_interval: Time in seconds between status checks when waiting

        Returns:
            PredictionResponse containing:
            - results: List[PredictionResult] with completed jobs
            - all_job_ids: All job IDs that were checked
            - completed_job_ids: Job IDs that completed successfully
            - incomplete_job_ids: Job IDs that didn't complete (still running/failed)

        Raises:
            RunLocalError: If no jobs completed and timeout specified
        """
        if self.verbosity >= 4:
            print(f"Waiting up to {timeout}s for jobs to complete...")

        # Configure poller with custom interval
        self.job_poller.poll_interval = poll_interval

        # Poll for job completion
        job_results = self.job_poller.poll_jobs(
            job_ids=job_ids,
            job_type=JobType.PREDICTION,  # Assume prediction jobs
            timeout=timeout,
        )

        # Filter to only completed jobs
        completed_results = [r for r in job_results if r.is_successful]

        # Process the completed results
        processed_results = self._process_prediction_results(
            completed_results,
            output_dir=output_dir,
        )

        # Collect job tracking information
        completed_job_ids = [r.job_id for r in completed_results]
        incomplete_job_ids = [
            job_id for job_id in job_ids if job_id not in completed_job_ids
        ]

        if incomplete_job_ids:
            self._print_incomplete_message(
                incomplete_job_ids, job_type=JobType.PREDICTION
            )

        # Create and return response wrapper
        return PredictionResponse(
            results=processed_results,
            all_job_ids=job_ids,
            completed_job_ids=completed_job_ids,
            incomplete_job_ids=incomplete_job_ids,
        )
