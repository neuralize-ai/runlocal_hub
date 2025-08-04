"""
Tests for benchmark models with performance data.
"""

import pytest
from runlocal_hub.models.benchmark import (
    BenchmarkData,
    BenchmarkDbItem,
    BenchmarkStatus,
)
from runlocal_hub.models.device import Device


@pytest.fixture
def sample_device():
    """Create a sample device for testing."""
    return Device(
        Name="iPhone 15 Pro",
        Year=2023,
        Soc="A17 Pro",
        Ram=8,
        OS="iOS",
        OSVersion="17.0",
    )


@pytest.fixture
def performance_benchmark_data():
    """Create benchmark data with realistic performance metrics."""
    return BenchmarkData(
        Success=True,
        Status=BenchmarkStatus.Complete,
        ComputeUnit="CPU",
        # Load performance
        LoadMsArray=[125.5, 130.2, 128.7],
        LoadMsAverage=128.13,
        LoadMsMedian=128.7,
        # Inference performance
        InferenceMsArray=[
            15.2,
            14.8,
            15.1,
            15.0,
        ],
        InferenceMsAverage=15.025,
        InferenceMsMedian=15.05,
        # Memory usage
        PeakLoadRamUsage=1024.5,
        PeakRamUsage=1536.8,
        # Output tensor ID
        OutputTensorsId="tensor-output-123",
    )


@pytest.fixture
def failed_benchmark_data():
    """Create benchmark data for a failed benchmark."""
    return BenchmarkData(
        Success=False,
        Status=BenchmarkStatus.Failed,
        ComputeUnit="GPU",
        FailureReason="Model incompatible with GPU",
        FailureError="CoreML error: Model requires iOS 16.0+",
        Stdout="Loading model...\nChecking compatibility...",
        Stderr="Error: Unsupported operation 'conv3d'",
    )


@pytest.fixture
def gpu_benchmark_data():
    """Create benchmark data for GPU execution."""
    return BenchmarkData(
        Success=True,
        Status=BenchmarkStatus.Complete,
        ComputeUnit="GPU",
        # Faster GPU performance
        LoadMsArray=[85.2, 87.1, 86.3],
        LoadMsAverage=86.2,
        LoadMsMedian=86.3,
        InferenceMsArray=[
            8.5,
            8.2,
            8.7,
            8.3,
        ],
        InferenceMsAverage=8.425,
        InferenceMsMedian=8.4,
        # Higher memory usage on GPU
        PeakLoadRamUsage=2048.0,
        PeakRamUsage=3072.5,
        OutputTensorsId="tensor-gpu-456",
    )


@pytest.fixture
def ane_benchmark_data():
    """Create benchmark data for Apple Neural Engine execution."""
    return BenchmarkData(
        Success=True,
        Status=BenchmarkStatus.Complete,
        ComputeUnit="ANE",
        # Very fast ANE performance
        LoadMsArray=[45.1, 44.8, 45.2],
        LoadMsAverage=45.03,
        LoadMsMedian=45.1,
        InferenceMsArray=[
            2.1,
            2.0,
            2.2,
            2.1,
        ],
        InferenceMsAverage=2.1,
        InferenceMsMedian=2.1,
        # Lower memory usage on ANE
        PeakLoadRamUsage=512.5,
        PeakRamUsage=768.2,
        OutputTensorsId="tensor-ane-789",
    )


class TestBenchmarkData:
    """Test cases for BenchmarkData with performance metrics."""

    def test_benchmark_data_creation_complete(self, performance_benchmark_data):
        """Test creating BenchmarkData with complete performance metrics."""
        data = performance_benchmark_data

        assert data.Success is True
        assert data.Status == BenchmarkStatus.Complete
        assert data.ComputeUnit == "CPU"

        # Test load metrics
        assert len(data.LoadMsArray) == 3
        assert data.LoadMsAverage == 128.13
        assert data.LoadMsMedian == 128.7

        # Test inference metrics
        assert len(data.InferenceMsArray) == 4
        assert data.InferenceMsAverage == 15.025
        assert data.InferenceMsMedian == 15.05

        # Test memory metrics
        assert data.PeakLoadRamUsage == 1024.5
        assert data.PeakRamUsage == 1536.8

    def test_benchmark_data_creation_failed(self, failed_benchmark_data):
        """Test creating BenchmarkData for failed benchmark."""
        data = failed_benchmark_data

        assert data.Success is False
        assert data.Status == BenchmarkStatus.Failed
        assert data.ComputeUnit == "GPU"
        assert data.FailureReason == "Model incompatible with GPU"
        assert data.FailureError == "CoreML error: Model requires iOS 16.0+"
        assert "Loading model..." in data.Stdout
        assert "Unsupported operation" in data.Stderr

    def test_benchmark_data_minimal(self):
        """Test creating BenchmarkData with minimal required fields."""
        data = BenchmarkData(ComputeUnit="CPU")

        assert data.ComputeUnit == "CPU"
        assert data.Success is None
        assert data.Status is None
        assert data.LoadMsArray is None
        assert data.InferenceMsArray is None


    def test_performance_comparison_across_compute_units(
        self, performance_benchmark_data, gpu_benchmark_data, ane_benchmark_data
    ):
        """Test comparing performance metrics across different compute units."""
        cpu_data = performance_benchmark_data
        gpu_data = gpu_benchmark_data
        ane_data = ane_benchmark_data

        # ANE should be fastest for inference
        assert ane_data.InferenceMsAverage < gpu_data.InferenceMsAverage
        assert gpu_data.InferenceMsAverage < cpu_data.InferenceMsAverage

        # GPU should be faster than CPU for loading
        assert gpu_data.LoadMsAverage < cpu_data.LoadMsAverage

        # Memory usage patterns
        assert (
            ane_data.PeakRamUsage
            < cpu_data.PeakRamUsage
            < gpu_data.PeakRamUsage
        )




class TestBenchmarkDbItem:
    """Test cases for BenchmarkDbItem with performance data."""

    def test_benchmark_db_item_single_compute_unit(
        self, sample_device, performance_benchmark_data
    ):
        """Test BenchmarkDbItem with single compute unit results."""
        db_item = BenchmarkDbItem(
            UploadId="model-upload-123",
            DeviceInfo=sample_device,
            Status=BenchmarkStatus.Complete,
            BenchmarkData=[performance_benchmark_data],
        )

        assert db_item.UploadId == "model-upload-123"
        assert db_item.DeviceInfo is not None
        assert db_item.DeviceInfo.Name == "iPhone 15 Pro"
        assert db_item.Status == BenchmarkStatus.Complete
        assert len(db_item.BenchmarkData) == 1
        assert db_item.BenchmarkData[0].ComputeUnit == "CPU"

    def test_benchmark_db_item_multiple_compute_units(
        self,
        sample_device,
        performance_benchmark_data,
        gpu_benchmark_data,
        ane_benchmark_data,
    ):
        """Test BenchmarkDbItem with multiple compute unit results."""
        db_item = BenchmarkDbItem(
            UploadId="model-upload-456",
            DeviceInfo=sample_device,
            Status=BenchmarkStatus.Complete,
            BenchmarkData=[
                performance_benchmark_data,
                gpu_benchmark_data,
                ane_benchmark_data,
            ],
        )

        assert len(db_item.BenchmarkData) == 3

        # Verify all compute units are present
        compute_units = [data.ComputeUnit for data in db_item.BenchmarkData]
        assert "CPU" in compute_units
        assert "GPU" in compute_units
        assert "ANE" in compute_units

        # Verify all are successful
        assert all(data.Success for data in db_item.BenchmarkData)

    def test_benchmark_db_item_mixed_success_failure(
        self, sample_device, performance_benchmark_data, failed_benchmark_data
    ):
        """Test BenchmarkDbItem with mixed success and failure results."""
        db_item = BenchmarkDbItem(
            UploadId="model-upload-789",
            DeviceInfo=sample_device,
            Status=BenchmarkStatus.Complete,  # Overall status can be complete even with some failures
            BenchmarkData=[performance_benchmark_data, failed_benchmark_data],
        )

        assert len(db_item.BenchmarkData) == 2

        # Check individual results
        cpu_result = next(
            data for data in db_item.BenchmarkData if data.ComputeUnit == "CPU"
        )
        gpu_result = next(
            data for data in db_item.BenchmarkData if data.ComputeUnit == "GPU"
        )

        assert cpu_result.Success is True
        assert gpu_result.Success is False
        assert gpu_result.FailureReason == "Model incompatible with GPU"

    def test_benchmark_db_item_to_dict(self, sample_device, performance_benchmark_data):
        """Test converting BenchmarkDbItem to dictionary."""
        db_item = BenchmarkDbItem(
            UploadId="model-upload-dict",
            DeviceInfo=sample_device,
            Status=BenchmarkStatus.Complete,
            BenchmarkData=[performance_benchmark_data],
        )

        dict_result = db_item.to_dict()

        assert dict_result["UploadId"] == "model-upload-dict"
        assert dict_result["Status"] == BenchmarkStatus.Complete
        assert dict_result["DeviceInfo"]["Name"] == "iPhone 15 Pro"
        assert len(dict_result["BenchmarkData"]) == 1
        assert dict_result["BenchmarkData"][0]["ComputeUnit"] == "CPU"

    def test_benchmark_db_item_no_device_info(self, performance_benchmark_data):
        """Test BenchmarkDbItem without device information."""
        db_item = BenchmarkDbItem(
            UploadId="model-upload-no-device",
            DeviceInfo=None,
            Status=BenchmarkStatus.Complete,
            BenchmarkData=[performance_benchmark_data],
        )

        assert db_item.DeviceInfo is None
        assert db_item.UploadId == "model-upload-no-device"
        assert len(db_item.BenchmarkData) == 1

    def test_benchmark_db_item_running_status(self, sample_device):
        """Test BenchmarkDbItem with running status and no results yet."""
        db_item = BenchmarkDbItem(
            UploadId="model-upload-running",
            DeviceInfo=sample_device,
            Status=BenchmarkStatus.Running,
            BenchmarkData=[],
        )

        assert db_item.Status == BenchmarkStatus.Running
        assert len(db_item.BenchmarkData) == 0
