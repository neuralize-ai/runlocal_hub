import os
import requests
from typing import Dict, List, Optional
from enum import Enum
import json
from pydantic import BaseModel
from decimal import Decimal
import time


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


class RunLocalClient:
    """
    Python client for the RunLocal API
    """

    base_url = "https://neuralize-bench.com"

    def __init__(
        self,
        env_var_name: str = "RUNLOCAL_API_KEY",
    ):
        """
        Initialize the RunLocal client

        Args:
            api_key: Your API key for authentication (optional, will use environment variable if not provided)
            base_url: The base URL of the RunLocal API (defaults to localhost)
            env_var_name: Name of environment variable to get API key from (defaults to RUNLOCAL_API_KEY)
        """
        # Get API key from parameter or environment variable
        api_key = os.environ.get(env_var_name)
        if not api_key:
            raise ValueError(
                f"{env_var_name} not found in environment. Please set the environment variable."
            )

        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key}

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        debug: bool = False,
    ) -> Dict:
        """
        Make a request to the API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            debug: If True, print debugging information

        Returns:
            Dict: API response
        """
        url = f"{self.base_url}{endpoint}"

        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"

        if debug:
            print(f"Request: {method} {url}")
            print(f"Headers: {headers}")
            if data:
                print(f"Data: {json.dumps(data, indent=2)}")

        try:
            response = requests.request(
                method=method, url=url, headers=headers, json=data
            )

            if debug:
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
            if debug:
                print(f"Request exception: {str(e)}")
            raise Exception(f"Request failed: {str(e)}")

    def health_check(self, debug: bool = False) -> Dict:
        """
        Check if the API is available and the API key is valid

        Args:
            debug: If True, print debugging information

        Returns:
            Dict: Health status
        """
        # Use the health endpoint to check authentication
        return self._make_request("GET", "/users/health", debug=debug)

    def get_models(self, debug: bool = False) -> List[str]:
        """
        Get a list of model IDs for the authenticated user

        Args:
            debug: If True, print debugging information

        Returns:
            List[str]: List of model IDs
        """
        user_data = self._make_request("GET", "/users", debug=debug)
        return user_data.get("UploadIds", [])

    def get_user_info(self, debug: bool = False) -> Dict:
        """
        Get detailed user information for the authenticated user

        Args:
            debug: If True, print debugging information

        Returns:
            Dict: User information including models, datasets, etc.
        """
        return self._make_request("GET", "/users", debug=debug)

    def get_benchmark_devices(
        self, model_id: Optional[str] = None, debug: bool = False
    ) -> List[DeviceUsage]:
        """
        Get a list of available devices for benchmarking

        Args:
            model_id: Optional ID of a model to get compatible devices
            debug: If True, print debugging information

        Returns:
            List[Dict]: List of available devices with their properties and compute units
        """
        endpoint = "/devices/benchmark"
        if model_id:
            endpoint += f"?upload_id={model_id}"

        response = self._make_request("GET", endpoint, debug=debug)

        if debug:
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

    def select_device(
        self,
        model_id: str,
        device_name: Optional[str] = None,
        ram: Optional[int] = None,
        soc: Optional[str] = None,
        year: Optional[int] = None,
        debug: bool = False,
    ) -> Optional[DeviceUsage]:
        """
        Select a device based on optional criteria. Returns the first matching device.

        Args:
            model_id: Required ID of a model to get compatible devices
            device_name: Optional device name to filter by (e.g. "MacBookPro")
            ram: Optional RAM amount to filter by (e.g. 16)
            soc: Optional SoC to filter by (e.g. "M2Pro")
            year: Optional year to filter by (e.g. 2023)
            debug: If True, print debugging information

        Returns:
            Optional[DeviceUsage]: The first matching device or None if no match found
        """
        # Verify model_id belongs to the user's models
        user_models = self.get_models(debug=debug)
        if model_id not in user_models:
            raise ValueError(
                f"model_id '{model_id}' does not correspond to your available models"
            )

        # Get all available devices
        devices = self.get_benchmark_devices(model_id=model_id, debug=debug)

        if debug:
            print(f"Found {len(devices)} devices")

        # Filter devices based on criteria
        matching_devices = devices

        if device_name is not None:
            matching_devices = [
                d for d in matching_devices if device_name in d.device.Name
            ]

        if soc is not None:
            matching_devices = [d for d in matching_devices if soc in d.device.Soc]

        if ram is not None:
            matching_devices = [d for d in matching_devices if d.device.Ram == ram]

        if year is not None:
            matching_devices = [d for d in matching_devices if d.device.Year == year]

        if debug:
            print(f"Found {len(matching_devices)} matching devices")

        # Return the first matching device or None
        return matching_devices[0] if matching_devices else None

    def benchmark_model(
        self,
        model_id: str,
        device: Device,
        compute_units: List[str],
        timeout=600,
        poll_interval: int = 10,
        debug: bool = False,
    ) -> Dict:
        """
        Benchmark a model on a specific device

        Args:
            model_id: The ID of the uploaded model
            device_id: The device identifier to benchmark on
            compute_units: List of compute units to use for benchmarking
            test_name: Optional name for the benchmark test
            wait_for_results: If True, block until benchmark is complete and return results
            timeout: Maximum time in seconds to wait for benchmark completion (default: 300s)
            poll_interval: Time in seconds between status checks (default: 5s)
            debug: If True, print debugging information

        Returns:
            Dict: Benchmark submission info, or if wait_for_results=True, the complete benchmark results
        """

        device_id = device.to_device_id()

        # Create the device request
        device_request = {"device_id": device_id, "compute_units": compute_units}

        # Prepare the data payload
        data = {
            "device_requests": [device_request],  # Wrap in list for API compatibility
        }

        # Use the standard benchmark endpoint to submit the request
        response = self._make_request(
            "POST",
            f"/coreml/benchmark/enqueue?upload_id={model_id}",
            data=data,
            debug=debug,
        )

        # Extract the benchmark ID from the response
        benchmark_id = response[0]

        if debug:
            print(f"Benchmark submitted with ID: {benchmark_id}")

        if not benchmark_id:
            if debug:
                print(f"Could not extract benchmark ID from response: {response}")
            raise ValueError("Could not extract benchmark ID from response")

        if debug:
            print(f"Waiting for benchmark {benchmark_id} to complete...")

        # Poll for benchmark completion
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Wait before checking
            time.sleep(poll_interval)

            try:
                result = self.get_benchmark_by_id(benchmark_id, debug=debug)

                if result is None:
                    continue

                status = result.Status

                # Check if benchmark is complete
                if status in [BenchmarkStatus.Complete, BenchmarkStatus.Failed]:
                    if debug:
                        print("Benchmark completed successfully")
                    # Convert the result to a JSON-friendly dictionary
                    return self._convert_benchmark_to_json_friendly(result)

                # Check if benchmark failed
                if status in [BenchmarkStatus.Failed, BenchmarkStatus.Deleted]:
                    if debug:
                        print(f"Benchmark failed: {result}")
                    raise Exception(f"Benchmark failed with status: {status}")

                # Still in progress
                if debug:
                    print(f"Benchmark still in progress (status: {status}), waiting...")
            except Exception as e:
                if "404" in str(e):
                    # Benchmark might not be in the database yet, retry
                    if debug:
                        print("Benchmark not found yet, retrying...")
                elif debug:
                    print(f"Error checking benchmark status: {e}, retrying...")

        # Timeout reached
        raise TimeoutError(
            f"Benchmark did not complete within {timeout} seconds timeout"
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

    def get_benchmark_by_id(
        self, benchmark_id: str, debug: bool = False
    ) -> Optional[BenchmarkDbItem]:
        """
        Get a specific benchmark by its ID

        Args:
            benchmark_id: The ID of the benchmark to retrieve
            debug: If True, print debugging information

        Returns:
            Dict: The benchmark data
        """
        # Use the new benchmark by ID endpoint
        respone = self._make_request(
            "GET", f"/benchmarks/id/{benchmark_id}", debug=debug
        )

        return BenchmarkDbItem(**respone)
