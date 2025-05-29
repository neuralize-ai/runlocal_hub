import io
import json
import os
import tempfile
import time
import zipfile
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from pydantic import BaseModel
from tqdm import tqdm


class Device(BaseModel):
    Name: str
    Year: int
    Soc: str
    Ram: int
    OS: str
    OSVersion: str
    DiscreteGpu: Optional[str] = None
    VRam: Optional[int] = None

    def to_device_id(self) -> str:
        device_id = f"Name={self.Name}|Year={self.Year}|Soc={self.Soc}|Ram={self.Ram}"
        if self.DiscreteGpu:
            device_id += f"|DiscreteGpu={self.DiscreteGpu}"
        if self.VRam:
            device_id += f"|VRam={self.VRam}"

        device_id += f"|OS={self.OS}|OSVersion={self.OSVersion}"
        return device_id


class DeviceUsage(BaseModel):
    device: Device
    compute_units: List[str]


class BenchmarkStatus(str, Enum):
    Pending = "Pending"  # not started, still in queue
    Complete = "Complete"
    Failed = "Failed"
    Running = "Running"
    Deleted = "Deleted"


class BenchmarkData(BaseModel):
    Success: Optional[bool] = None
    Status: Optional[BenchmarkStatus] = None
    ComputeUnit: str

    # load
    LoadMsArray: Optional[List[Decimal]] = None
    LoadMsAverage: Optional[Decimal] = None
    LoadMsMedian: Optional[Decimal] = None
    # total inference
    InferenceMsArray: Optional[List[Decimal]] = None
    InferenceMsAverage: Optional[Decimal] = None
    InferenceMsMedian: Optional[Decimal] = None
    # GenAI inference
    PrefillTokens: Optional[int] = None
    GenerationTokens: Optional[int] = None
    PrefillTPS: Optional[Decimal] = None
    GenerateTPS: Optional[Decimal] = None
    # peak ram usage
    PeakLoadRamUsage: Optional[Decimal] = None
    # TODO: inference, named "PeakRamUsage" for legacy support
    PeakRamUsage: Optional[Decimal] = None
    # peak genai ram usage
    PeakPrefillRamUsage: Optional[Decimal] = None
    PeakGenerateRamUsage: Optional[Decimal] = None

    FailureReason: Optional[str] = None
    FailureError: Optional[str] = None
    Stdout: Optional[str] = None
    Stderr: Optional[str] = None

    OutputTensorsId: Optional[str] = None

    def to_json_dict(self) -> Dict:
        """
        Needed for post requests, not for dynamodb, dynamo can handle Decimals
        Decimals need to be strings
        """
        return {
            "Success": self.Success if self.Success is not None else None,
            "Status": self.Status if self.Status is not None else None,
            "FailureReason": self.FailureReason,
            "FailureError": self.FailureError,
            "Stdout": self.Stdout,
            "Stderr": self.Stderr,
            "ComputeUnit": self.ComputeUnit,
            "LoadMsArray": [str(x) for x in self.LoadMsArray]
            if self.LoadMsArray is not None
            else None,
            "LoadMsAverage": str(self.LoadMsAverage)
            if self.LoadMsAverage is not None
            else None,
            "LoadMsMedian": str(self.LoadMsMedian)
            if self.LoadMsMedian is not None
            else None,
            "InferenceMsArray": [str(x) for x in self.InferenceMsArray]
            if self.InferenceMsArray is not None
            else None,
            "InferenceMsAverage": str(self.InferenceMsAverage)
            if self.InferenceMsAverage is not None
            else None,
            "InferenceMsMedian": str(self.InferenceMsMedian)
            if self.InferenceMsMedian is not None
            else None,
            "PrefillTokens": str(self.PrefillTokens)
            if self.PrefillTokens is not None
            else None,
            "GenerationTokens": str(self.GenerationTokens)
            if self.GenerationTokens is not None
            else None,
            "PrefillTPS": str(self.PrefillTPS) if self.PrefillTPS is not None else None,
            "GenerateTPS": str(self.GenerateTPS)
            if self.GenerateTPS is not None
            else None,
            "PeakLoadRamUsage": str(self.PeakLoadRamUsage)
            if self.PeakLoadRamUsage is not None
            else None,
            "PeakRamUsage": str(self.PeakRamUsage)
            if self.PeakRamUsage is not None
            else None,
            "PeakPrefillRamUsage": str(self.PeakPrefillRamUsage)
            if self.PeakPrefillRamUsage is not None
            else None,
            "PeakGenerateRamUsage": str(self.PeakGenerateRamUsage)
            if self.PeakGenerateRamUsage is not None
            else None,
        }


class BenchmarkDbItem(BaseModel):
    UploadId: str
    DeviceInfo: Optional[Device] = None
    Status: BenchmarkStatus
    BenchmarkData: List[BenchmarkData]

    def to_dict(self):
        d = self.dict()
        return d


class IOType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class JobType(str, Enum):
    BENCHMARK = "benchmark"
    PREDICTION = "prediction"


class TensorInfo(BaseModel):
    """Information about a single tensor"""

    Shape: List[int]
    Dtype: str
    SizeBytes: int


class IOTensorsMetadata(BaseModel):
    UserId: str
    Id: str
    IOType: IOType
    TensorMetadata: Dict[str, TensorInfo]
    SourceBenchmarkIds: Optional[List[str]] = None
    CreatedUtc: str


class RunLocalClient:
    """
    Python client for the RunLocal API
    """

    base_url = "https://neuralize-bench.com"
    env_var_name = "RUNLOCAL_API_KEY"

    def __init__(self, debug=False):
        # Get API key from parameter or environment variable
        api_key = os.environ.get(self.env_var_name)
        if not api_key:
            raise ValueError(
                f"{self.env_var_name} not found in environment. Please set the environment variable."
            )

        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key}
        self.debug = debug

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Dict:
        """
        Make a request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data

        Returns:
            Dict: API response
        """
        url = f"{self.base_url}{endpoint}"

        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"

        if self.debug:
            print(f"Request: {method} {url}")
            print(f"Headers: {headers}")
            if data:
                print(f"Data: {json.dumps(data, indent=2)}")

        try:
            response = requests.request(
                method=method, url=url, headers=headers, json=data
            )

            if self.debug:
                print(f"Response status: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                try:
                    print(f"Response body: {json.dumps(response.json(), indent=2)}")
                except:
                    print(f"Response body: {response.text}")

            # Check for errors
            if response.status_code >= 400:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        error_msg = f"{error_msg}: {error_data['detail']}"
                except:
                    if response.text:
                        error_msg = f"{error_msg}: {response.text}"
                raise Exception(error_msg)

            # Return JSON response, or text if not JSON
            try:
                return response.json()
            except:
                return {"text": response.text}
        except requests.exceptions.RequestException as e:
            if self.debug:
                print(f"Request exception: {str(e)}")
            raise Exception(f"Request failed: {str(e)}")

    def health_check(self) -> Dict:
        """
        Check if the API is available and the API key is valid

        Returns:
            Dict: Health status
        """
        # Use the health endpoint to check authentication
        return self._make_request("GET", "/users/health")

    def get_user_info(self) -> Dict:
        """
        Get detailed user information for the authenticated user

        Returns:
            Dict: User information including models, datasets, etc.
        """
        return self._make_request("GET", "/users")

    def get_models(self) -> List[str]:
        """
        Get a list of model IDs for the authenticated user

        Returns:
            List[str]: List of model IDs
        """
        user_data = self._make_request("GET", "/users")
        return user_data.get("UploadIds", [])

    def _zip_path(self, path: Path, temp_dir: Path) -> Path:
        """
        Zip a file or folder.

        Args:
            path: Path to file or folder to zip
            temp_dir: Temporary directory to store the zip file

        Returns:
            Path to the created zip file
        """
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Create zip file name based on the input path
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

    def _parse_sse_message(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a Server-Sent Event message line"""
        if line.startswith("data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                return None
        return None

    def upload_model(
        self,
        model_path: Union[Path, str],
        show_progress: bool = True,
    ) -> str:
        """
        Upload a model file or folder to the RunLocal platform.

        Args:
            model_path: Path to the model file or folder to upload
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Upload ID of the uploaded model

        Raises:
            Exception: If upload fails
        """

        model_path = Path(model_path)
        upload_filename = model_path.stem

        # TODO: ensure compatible format

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

            # Prepare the request URL and parameters
            upload_url = f"{self.base_url}/uploads/model/coreml"

            params = {
                "upload_filename": upload_filename,
                "upload_source_type": "USER_UPLOADED",
            }

            # Read the zip file
            with open(zip_path, "rb") as f:
                zip_data = f.read()

            if self.debug:
                print("Uploading to server...")

            # Make the request with streaming response
            try:
                response = requests.post(
                    upload_url,
                    params=params,
                    data=zip_data,
                    headers=self.headers,
                    stream=True,
                )

                # Check initial response
                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("detail", response.text)
                    except:
                        pass
                    raise Exception(
                        f"Upload failed with status {response.status_code}: {error_detail}"
                    )

                # Process Server-Sent Events
                upload_id = None
                already_exists = False
                progress_bar = None

                if show_progress:
                    progress_bar = tqdm(total=100, desc="Upload Progress", unit="%")

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        data = self._parse_sse_message(line_str)

                        if data:
                            # Check for errors
                            if data.get("error"):
                                if progress_bar:
                                    progress_bar.close()
                                raise Exception(
                                    f"Server error: {data.get('detail', 'Unknown error')}"
                                )

                            # Update progress
                            if "progress" in data and progress_bar:
                                progress_bar.update(data["progress"] - progress_bar.n)

                            # Extract upload_id
                            if "upload_id" in data:
                                upload_id = data["upload_id"]
                                already_exists = data.get("already_exists", False)

                            # Print status messages
                            if "message" in data and not show_progress:
                                print(f"Server: {data['message']}")

                if progress_bar:
                    progress_bar.close()

                if not upload_id:
                    raise Exception("No upload ID received from server")

                if already_exists:
                    print(f"Model already exists with upload_id: {upload_id}")
                else:
                    print(f"Model uploaded successfully with upload_id: {upload_id}")

                return upload_id

            except requests.exceptions.RequestException as e:
                raise Exception(f"Network error during upload: {str(e)}")

    def list_all_devices(self, model_id: Optional[str] = None) -> List[DeviceUsage]:
        """
        Get a list of available devices for benchmarking

        Args:
            model_id: Optional ID of a model to get compatible devices

        Returns:
            List[Dict]: List of available devices with their properties and compute units
        """
        endpoint = "/devices/benchmark"
        if model_id:
            endpoint += f"?upload_id={model_id}"

        response = self._make_request("GET", endpoint)

        if self.debug:
            print(f"response: {response[:2]}")

        if model_id:
            return [
                DeviceUsage(
                    device=device_usage["device"],
                    compute_units=device_usage["compute_units"],
                )
                for device_usage in response
                if device_usage.get("compute_units")
                and len(device_usage.get("compute_units", [])) > 0
            ]

        return [
            DeviceUsage(
                device=device_usage["device"],
                compute_units=device_usage["compute_units"]
                if device_usage.get("compute_units")
                else [],
            )
            for device_usage in response
        ]

    def select_devices(
        self,
        model_id: str,
        count: Optional[int] = None,
        device_name: Optional[str] = None,
        ram_min: Optional[int] = None,
        ram_max: Optional[int] = None,
        soc: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> List[DeviceUsage]:
        """
        Select a device based on optional criteria. Returns the first matching device.

        Args:
            model_id: Required ID of a model to get compatible devices
            count: Number of devices to select
            device_name: Optional device name to filter by (e.g. "MacBookPro")
            ram_min: Optional minimum RAM amount (inclusive)
            ram_max: Optional maximum RAM amount (inclusive)
            soc: Optional SoC to filter by (e.g. "Apple M2 Pro")
            year_min: Optional minimum year (inclusive)
            year_max: Optional maximum year (inclusive)

        Returns:
            Optional[DeviceUsage]: The first matching device or None if no match found
        """
        # Verify model_id belongs to the user's models
        user_models = self.get_models()
        if model_id not in user_models:
            raise ValueError(
                f"model_id '{model_id}' does not correspond to your available models"
            )

        # Get all available devices
        devices = self.list_all_devices(model_id=model_id)

        if self.debug:
            print(f"Found {len(devices)} devices")

        # Filter devices based on criteria
        if device_name is not None:
            devices = [d for d in devices if device_name in d.device.Name]

        if soc is not None:
            devices = [d for d in devices if soc in d.device.Soc]

        if ram_min is not None:
            devices = [d for d in devices if d.device.Ram >= ram_min]

        if ram_max is not None:
            devices = [d for d in devices if d.device.Ram <= ram_max]

        if year_min is not None:
            devices = [d for d in devices if d.device.Year >= year_min]

        if year_max is not None:
            devices = [d for d in devices if d.device.Year <= year_max]

        num_devices = len(devices)
        if self.debug:
            print(f"Found {num_devices} matching devices")

        if count is not None and num_devices > count:
            devices = devices[:count]

        return devices

    def select_device(
        self,
        model_id: str,
        device_name: Optional[str] = None,
        ram_min: Optional[int] = None,
        ram_max: Optional[int] = None,
        soc: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> Optional[DeviceUsage]:
        devices = self.select_devices(
            model_id=model_id,
            count=1,
            device_name=device_name,
            ram_min=ram_min,
            ram_max=ram_max,
            soc=soc,
            year_min=year_min,
            year_max=year_max,
        )

        if len(devices) > 0:
            return devices[0]

    def benchmark_model(
        self,
        model_id: str,
        device: Device,
        compute_units: List[str],
        inputs: Optional[Dict[str, np.ndarray]] = None,
        timeout=600,
        poll_interval: int = 10,
    ) -> Dict:
        """
        Benchmark a model on a specific device

        Args:
            model_id: The ID of the uploaded model
            device: The device to benchmark on
            compute_units: List of compute units to use for benchmarking
            inputs: Optional dictionary mapping input names to numpy arrays
            timeout: Maximum time in seconds to wait for benchmark completion (default: 600s)
            poll_interval: Time in seconds between status checks (default: 10s)

        Returns:
            Dict: The complete benchmark results
        """

        device_id = device.to_device_id()

        # Upload input tensors if provided
        input_tensors_id = None
        if inputs is not None:
            if self.debug:
                print("Uploading input tensors for benchmark...")
                for name, tensor in inputs.items():
                    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

            input_tensors_id = self.upload_io_tensors(inputs, io_type=IOType.INPUT)

        # Create the device request
        device_request = {"device_id": device_id, "compute_units": compute_units}

        # Prepare the data payload
        data: Dict[str, Any] = {
            "device_requests": [device_request],  # Wrap in list for API compatibility
        }

        # Add input tensors to the payload if provided
        if input_tensors_id is not None:
            data["input_tensors_id"] = input_tensors_id

        # Use the standard benchmark endpoint to submit the request
        response = self._make_request(
            "POST",
            f"/coreml/benchmark/enqueue?upload_id={model_id}",
            data=data,
        )

        # Extract the benchmark ID from the response
        benchmark_id = response[0]

        if self.debug:
            print(f"Benchmark submitted with ID: {benchmark_id}")

        if not benchmark_id:
            if self.debug:
                print(f"Could not extract benchmark ID from response: {response}")
            raise ValueError("Could not extract benchmark ID from response")

        if self.debug:
            print(f"Waiting for benchmark {benchmark_id} to complete...")

        # Poll for benchmark completion
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Wait before checking
            time.sleep(poll_interval)

            try:
                result = self.get_benchmark_by_id(benchmark_id)

                if result is None:
                    continue

                status = result.Status

                # Check if benchmark is complete
                if status in [BenchmarkStatus.Complete, BenchmarkStatus.Failed]:
                    if self.debug:
                        print("Benchmark completed successfully")

                    # Convert the result to a JSON-friendly dictionary
                    result_dict = self._convert_benchmark_to_json_friendly(result)

                    # If input tensors were provided, download output tensors
                    if inputs is not None and status == BenchmarkStatus.Complete:
                        output_tensors = {}
                        for benchmark_data in result.BenchmarkData:
                            if (
                                benchmark_data.Success
                                and benchmark_data.OutputTensorsId
                            ):
                                compute_unit = benchmark_data.ComputeUnit
                                output_tensors_id = benchmark_data.OutputTensorsId

                                if self.debug:
                                    print(
                                        f"Downloading outputs for compute unit '{compute_unit}' (tensor ID: {output_tensors_id})"
                                    )

                                # Download output tensors for this compute unit
                                output_tensors[compute_unit] = self.download_io_tensors(
                                    output_tensors_id
                                )

                        # Add output tensors to the result
                        if output_tensors:
                            # Convert numpy arrays to lists for JSON serialization
                            output_tensors_json = {}
                            for compute_unit, tensors in output_tensors.items():
                                output_tensors_json[compute_unit] = {}
                                for name, tensor in tensors.items():
                                    output_tensors_json[compute_unit][name] = {
                                        "data": tensor.tolist(),
                                        "shape": list(tensor.shape),
                                        "dtype": str(tensor.dtype)
                                    }
                            result_dict["OutputTensors"] = output_tensors_json

                    return result_dict

                # Check if benchmark failed
                if status in [BenchmarkStatus.Failed, BenchmarkStatus.Deleted]:
                    if self.debug:
                        print(f"Benchmark failed: {result}")
                    raise Exception(f"Benchmark failed with status: {status}")

                # Still in progress
                if self.debug:
                    print(f"Benchmark still in progress (status: {status}), waiting...")
            except Exception as e:
                if "404" in str(e):
                    # Benchmark might not be in the database yet, retry
                    if self.debug:
                        print("Benchmark not found yet, retrying...")
                elif self.debug:
                    print(f"Error checking benchmark status: {e}, retrying...")

        # Timeout reached
        raise TimeoutError(
            f"Benchmark did not complete within {timeout} seconds timeout"
        )

    def upload_io_tensors(
        self,
        tensors: Dict[str, np.ndarray],
        io_type: IOType = IOType.INPUT,
        source_benchmark_id: Optional[str] = None,
    ) -> str:
        """
        Upload numpy arrays as IOTensors

        Args:
            tensors: Dictionary mapping tensor names to numpy arrays
            io_type: Whether these are input or output tensors
            source_benchmark_id: For output tensors, the benchmark that created them

        Returns:
            str: The tensors_id (hash) of the uploaded tensors
        """
        # Create NPZ file in memory
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **tensors)
        npz_buffer.seek(0)

        # Prepare the request
        url = f"{self.base_url}/io-tensors/upload"
        params = {"io_type": io_type.value}
        if source_benchmark_id:
            params["source_benchmark_id"] = source_benchmark_id

        # Upload the NPZ file
        files = {"file": ("tensors.npz", npz_buffer, "application/octet-stream")}

        try:
            response = requests.post(
                url,
                params=params,
                files=files,
                headers=self.headers,
            )

            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", response.text)
                except:
                    pass
                raise Exception(
                    f"IOTensor upload failed with status {response.status_code}: {error_detail}"
                )

            result = response.json()
            tensors_id = result["tensors_id"]

            if self.debug:
                print(f"IOTensors uploaded successfully with ID: {tensors_id}")

            return tensors_id

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error during IOTensor upload: {str(e)}")

    def get_io_tensors_metadata(self, tensors_id: str) -> IOTensorsMetadata:
        """
        Get metadata about IOTensors without downloading the data

        Args:
            tensors_id: The ID of the tensors to get metadata for

        Returns:
            IOTensorsMetadata: Metadata about the tensors
        """
        response = self._make_request("GET", f"/io-tensors/{tensors_id}")
        return IOTensorsMetadata(**response)

    def download_io_tensors(self, tensors_id: str) -> Dict[str, np.ndarray]:
        """
        Download IOTensors as numpy arrays

        Args:
            tensors_id: The ID of the tensors to download

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping tensor names to numpy arrays
        """
        url = f"{self.base_url}/io-tensors/{tensors_id}/download"

        try:
            response = requests.get(url, headers=self.headers)

            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("detail", response.text)
                except:
                    pass
                raise Exception(
                    f"IOTensor download failed with status {response.status_code}: {error_detail}"
                )

            # Load NPZ file from response content
            npz_buffer = io.BytesIO(response.content)
            npz_file = np.load(npz_buffer)

            # Convert to regular dict
            tensors = {name: npz_file[name] for name in npz_file.files}

            if self.debug:
                print(f"Downloaded {len(tensors)} tensors")
                for name, tensor in tensors.items():
                    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

            return tensors

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error during IOTensor download: {str(e)}")

    def list_io_tensors(
        self, io_type: Optional[IOType] = None
    ) -> List[IOTensorsMetadata]:
        """
        List all IOTensors for the authenticated user

        Args:
            io_type: Optional filter by input or output type

        Returns:
            List[IOTensorsMetadata]: List of tensor metadata
        """
        endpoint = "/io-tensors"
        if io_type:
            endpoint += f"?io_type={io_type.value}"

        response = self._make_request("GET", endpoint)
        return [IOTensorsMetadata(**item) for item in response]

    def predict_model(
        self,
        model_id: str,
        device: Device,
        compute_units: List[str],
        input_tensors: Dict[str, np.ndarray],
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run prediction on a model with given input tensors

        Args:
            model_id: The ID of the uploaded model
            device: The device to run prediction on
            compute_units: List of compute units to use
            input_tensors: Dictionary mapping input names to numpy arrays
            timeout: Maximum time in seconds to wait for completion (default: 600s)
            poll_interval: Time in seconds between status checks (default: 10s)

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary mapping compute unit names to output tensors
        """
        # Upload input tensors
        if self.debug:
            print("Uploading input tensors...")
            for name, tensor in input_tensors.items():
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

        input_tensors_id = self.upload_io_tensors(input_tensors, io_type=IOType.INPUT)

        # Create device ID
        device_id = device.to_device_id()

        # Create the device request
        device_request = {"device_id": device_id, "compute_units": compute_units}

        # Prepare the data payload for prediction job
        data = {
            "device_requests": [device_request],
            "input_tensors_id": input_tensors_id,
            "job_type": JobType.PREDICTION.value,
        }

        # Submit the prediction job
        response = self._make_request(
            "POST",
            f"/coreml/benchmark/enqueue?upload_id={model_id}",
            data=data,
        )

        # Extract the benchmark ID
        benchmark_id = response[0]

        if self.debug:
            print(f"Prediction job submitted with ID: {benchmark_id}")

        if not benchmark_id:
            raise ValueError("Could not extract benchmark ID from response")

        if self.debug:
            print(f"Waiting for prediction {benchmark_id} to complete...")

        # Poll for prediction completion
        start_time = time.time()
        compute_unit_outputs = {}

        while time.time() - start_time < timeout:
            # Wait before checking
            time.sleep(poll_interval)

            try:
                result = self.get_benchmark_by_id(benchmark_id)

                if result is None:
                    continue

                status = result.Status

                # Check if prediction is complete
                if status == BenchmarkStatus.Complete:
                    # Extract output tensor IDs from all compute units
                    for benchmark_data in result.BenchmarkData:
                        if benchmark_data.Success and benchmark_data.OutputTensorsId:
                            compute_unit = benchmark_data.ComputeUnit
                            output_tensors_id = benchmark_data.OutputTensorsId

                            if self.debug:
                                print(
                                    f"Downloading outputs for compute unit '{compute_unit}' (tensor ID: {output_tensors_id})"
                                )

                            # Download output tensors for this compute unit
                            compute_unit_outputs[compute_unit] = (
                                self.download_io_tensors(output_tensors_id)
                            )

                    if compute_unit_outputs:
                        if self.debug:
                            print(
                                f"Prediction completed successfully. Retrieved outputs for {len(compute_unit_outputs)} compute unit(s)"
                            )
                        return compute_unit_outputs
                    else:
                        raise Exception(
                            "Prediction completed but no output tensors found"
                        )

                # Check if prediction failed
                elif status in [BenchmarkStatus.Failed, BenchmarkStatus.Deleted]:
                    # Get failure reason
                    failure_reason = "Unknown failure"
                    for benchmark_data in result.BenchmarkData:
                        if benchmark_data.FailureReason:
                            failure_reason = benchmark_data.FailureReason
                            break
                    raise Exception(f"Prediction failed: {failure_reason}")

                # Still in progress
                if self.debug:
                    print(
                        f"Prediction still in progress (status: {status}), waiting..."
                    )

            except Exception as e:
                if "404" in str(e):
                    # Benchmark might not be in the database yet, retry
                    if self.debug:
                        print("Prediction job not found yet, retrying...")
                else:
                    # Re-raise prediction failures
                    raise

        # Timeout reached
        raise TimeoutError(
            f"Prediction did not complete within {timeout} seconds timeout"
        )

    def _convert_benchmark_to_json_friendly(self, benchmark: BenchmarkDbItem) -> Dict:
        """
        Convert a BenchmarkDbItem to a JSON-friendly dictionary by:
        1. Replacing Decimal objects with floats
        2. Omitting null/None fields
        3. Cleaning up empty collections

        Args:
            benchmark: The benchmark object to convert

        Returns:
            Dict: A clean, JSON-friendly dictionary representation of the benchmark
        """
        # First convert to dict
        result_dict = benchmark.to_dict()

        # Clean the main benchmark item - remove null fields
        cleaned_dict = self._clean_dict(result_dict)

        # Process BenchmarkData to convert Decimal values to float and remove null fields
        if "BenchmarkData" in cleaned_dict and cleaned_dict["BenchmarkData"]:
            cleaned_benchmark_data = []

            for data_item in cleaned_dict["BenchmarkData"]:
                # First convert Decimal values
                for key, value in list(data_item.items()):
                    # Convert Decimal to float
                    if isinstance(value, Decimal):
                        data_item[key] = float(value)
                    # Convert lists of Decimals to lists of floats
                    elif (
                        isinstance(value, list)
                        and value
                        and isinstance(value[0], Decimal)
                    ):
                        data_item[key] = [float(item) for item in value]

                # Then clean the dictionary to remove nulls
                cleaned_data_item = self._clean_dict(data_item)
                cleaned_benchmark_data.append(cleaned_data_item)

            cleaned_dict["BenchmarkData"] = cleaned_benchmark_data

        return cleaned_dict

    def _clean_dict(self, d: Dict) -> Dict:
        """
        Helper method to clean a dictionary by:
        1. Removing None/null values
        2. Removing empty lists and dictionaries

        Args:
            d: Dictionary to clean

        Returns:
            Dict: Cleaned dictionary without null values or empty collections
        """
        if not isinstance(d, dict):
            return d

        # Create a new dict with only non-null values
        result = {}
        for k, v in d.items():
            # Skip None values
            if v is None:
                continue

            # Clean nested dictionaries
            if isinstance(v, dict):
                cleaned = self._clean_dict(v)
                if cleaned:  # Only include non-empty dicts
                    result[k] = cleaned

            # Clean lists - remove None entries and process any dict items
            elif isinstance(v, list):
                if not v:  # Skip empty lists
                    continue

                cleaned_list = []
                for item in v:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        cleaned_item = self._clean_dict(item)
                        if cleaned_item:  # Only include non-empty dicts
                            cleaned_list.append(cleaned_item)
                    else:
                        cleaned_list.append(item)

                if cleaned_list:  # Only include non-empty lists
                    result[k] = cleaned_list
            else:
                # Regular value, include it
                result[k] = v

        return result

    def get_benchmark_by_id(self, benchmark_id: str) -> Optional[BenchmarkDbItem]:
        """
        Get a specific benchmark by its ID

        Args:
            benchmark_id: The ID of the benchmark to retrieve

        Returns:
            Dict: The benchmark data
        """
        # Use the new benchmark by ID endpoint
        respone = self._make_request("GET", f"/benchmarks/id/{benchmark_id}")

        return BenchmarkDbItem(**respone)

    def benchmark(
        self,
        model_path: Optional[Union[Path, str]] = None,
        model_id: Optional[str] = None,
        device_name: Optional[str] = None,
        ram_min: Optional[int] = None,
        ram_max: Optional[int] = None,
        soc: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        compute_units: Optional[List[str]] = None,
        inputs: Optional[Dict[str, np.ndarray]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        show_progress: bool = True,
    ) -> Dict:
        """
        Single-call method to benchmark a model from file path or model ID

        Args:
            model_path: Path to the model file or folder (if model_id not provided)
            model_id: ID of already uploaded model (if model_path not provided)
            device_name: Optional device name to filter by (e.g. "MacBook")
            ram_min: Optional minimum RAM amount (inclusive)
            ram_max: Optional maximum RAM amount (inclusive)
            soc: Optional SoC to filter by (e.g. "Apple M2 Pro")
            year_min: Optional minimum year (inclusive)
            year_max: Optional maximum year (inclusive)
            compute_units: Optional list of compute units to use. If None, uses all available
            inputs: Optional dictionary mapping input names to numpy arrays
            timeout: Maximum time in seconds to wait for completion (default: 600s)
            poll_interval: Time in seconds between status checks (default: 10s)
            show_progress: Whether to show upload progress bar (default: True)

        Returns:
            Dict: Benchmark results

        Raises:
            ValueError: If neither model_path nor model_id is provided, or if both are provided
            Exception: If benchmark fails
        """
        # Validate input parameters
        if model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided")
        if model_path is not None and model_id is not None:
            raise ValueError("Only one of model_path or model_id should be provided")

        # Upload model if path provided
        if model_path is not None:
            if self.debug:
                print(f"Uploading model from {model_path}...")
            model_id = self.upload_model(model_path, show_progress=show_progress)

        if model_id is None:
            raise Exception("Model upload failed")

        # Select device
        if self.debug:
            print("Selecting device...")

        device_usage = self.select_device(
            model_id=model_id,
            device_name=device_name,
            ram_min=ram_min,
            ram_max=ram_max,
            soc=soc,
            year_min=year_min,
            year_max=year_max,
        )

        if device_usage is None:
            raise Exception("No matching device found with the specified criteria")

        # Use provided compute units or all available
        selected_compute_units = (
            compute_units if compute_units is not None else device_usage.compute_units
        )

        print(
            f"Using device: {device_usage.device.Name} {device_usage.device.Year} ({device_usage.device.Soc})"
        )
        print(f"\twith compute units: {selected_compute_units}")

        # Run benchmark
        return self.benchmark_model(
            model_id=model_id,
            device=device_usage.device,
            compute_units=selected_compute_units,
            inputs=inputs,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def predict(
        self,
        inputs: Dict[str, np.ndarray],
        model_path: Optional[Union[Path, str]] = None,
        model_id: Optional[str] = None,
        device_name: Optional[str] = None,
        ram_min: Optional[int] = None,
        ram_max: Optional[int] = None,
        soc: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        compute_units: Optional[List[str]] = None,
        timeout: int = 600,
        poll_interval: int = 10,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Single-call method to run prediction on a model from file path or model ID

        Args:
            inputs: Dictionary mapping input names to numpy arrays
            model_path: Path to the model file or folder (if model_id not provided)
            model_id: ID of already uploaded model (if model_path not provided)
            device_name: Optional device name to filter by (e.g. "MacBook")
            ram_min: Optional minimum RAM amount (inclusive)
            ram_max: Optional maximum RAM amount (inclusive)
            soc: Optional SoC to filter by (e.g. "Apple M2 Pro")
            year_min: Optional minimum year (inclusive)
            year_max: Optional maximum year (inclusive)
            compute_units: Optional list of compute units to use. If None, uses all available
            timeout: Maximum time in seconds to wait for completion (default: 600s)
            poll_interval: Time in seconds between status checks (default: 10s)
            show_progress: Whether to show upload progress bar (default: True)

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Dictionary mapping compute unit names to output tensors

        Raises:
            ValueError: If neither model_path nor model_id is provided, or if both are provided
            Exception: If prediction fails
        """
        # Validate input parameters
        if model_path is None and model_id is None:
            raise ValueError("Either model_path or model_id must be provided")
        if model_path is not None and model_id is not None:
            raise ValueError("Only one of model_path or model_id should be provided")

        # Upload model if path provided
        if model_path is not None:
            if self.debug:
                print(f"Uploading model from {model_path}...")
            model_id = self.upload_model(model_path, show_progress=show_progress)

        if model_id is None:
            raise Exception("Model upload failed")

        # Select device
        if self.debug:
            print("Selecting device...")
        device_usage = self.select_device(
            model_id=model_id,
            device_name=device_name,
            ram_min=ram_min,
            ram_max=ram_max,
            soc=soc,
            year_min=year_min,
            year_max=year_max,
        )

        if device_usage is None:
            raise Exception("No matching device found with the specified criteria")

        # Use provided compute units or all available
        selected_compute_units = (
            compute_units if compute_units is not None else device_usage.compute_units
        )

        print(
            f"Using device: {device_usage.device.Name} {device_usage.device.Year} ({device_usage.device.Soc})"
        )
        print(f"\twith compute units: {selected_compute_units}")

        # Run prediction
        return self.predict_model(
            model_id=model_id,
            device=device_usage.device,
            compute_units=selected_compute_units,
            input_tensors=inputs,
            timeout=timeout,
            poll_interval=poll_interval,
        )
