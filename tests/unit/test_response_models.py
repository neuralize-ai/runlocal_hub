"""
Tests for response wrapper models (BenchmarkResponse and PredictionResponse).
"""

import pytest
from runlocal_hub.models import (
    BenchmarkResponse,
    PredictionResponse,
    BenchmarkResult,
    PredictionResult,
    Device,
    BenchmarkData,
)


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
def sample_benchmark_result(sample_device):
    """Create a sample benchmark result."""
    benchmark_data = BenchmarkData(
        ComputeUnit="CPU",
        Success=True,
        InferenceMsAverage=10.5,
        LoadMsAverage=2.3,
        PeakLoadRamUsage=100.0,
        PeakRamUsage=150.0,
    )
    return BenchmarkResult(device=sample_device, benchmark_data=[benchmark_data])


@pytest.fixture
def sample_prediction_result(sample_device):
    """Create a sample prediction result."""
    return PredictionResult(
        device=sample_device,
        outputs={"CPU": {"output": "/fake/output.npy"}},
        job_id="job_123",
        status="Complete",
        modelid="model_123",
    )


class TestBenchmarkResponse:
    """Test cases for BenchmarkResponse wrapper model."""

    def test_single_result_response(self, sample_benchmark_result):
        """Test BenchmarkResponse with single result."""
        response = BenchmarkResponse(
            results=sample_benchmark_result,
            all_job_ids=["job_123"],
            completed_job_ids=["job_123"],
            incomplete_job_ids=[],
        )

        assert isinstance(response.results, BenchmarkResult)
        assert response.all_job_ids == ["job_123"]
        assert response.completed_job_ids == ["job_123"]
        assert response.incomplete_job_ids == []

        # Test properties
        assert response.has_incomplete_jobs is False
        assert response.total_jobs == 1
        assert response.completed_count == 1
        assert response.incomplete_count == 0
        assert response.completion_rate == 100.0

    def test_multiple_results_response(self, sample_benchmark_result, sample_device):
        """Test BenchmarkResponse with multiple results."""
        # Create second result
        benchmark_data2 = BenchmarkData(
            ComputeUnit="GPU",
            Success=True,
            InferenceMsAverage=8.2,
            LoadMsAverage=1.8,
            PeakLoadRamUsage=120.0,
            PeakRamUsage=180.0,
        )
        result2 = BenchmarkResult(
            device=sample_device, benchmark_data=[benchmark_data2]
        )

        response = BenchmarkResponse(
            results=[sample_benchmark_result, result2],
            all_job_ids=["job_123", "job_456"],
            completed_job_ids=["job_123", "job_456"],
            incomplete_job_ids=[],
        )

        assert isinstance(response.results, list)
        assert len(response.results) == 2
        assert response.total_jobs == 2
        assert response.completed_count == 2
        assert response.completion_rate == 100.0

    def test_incomplete_jobs_response(self, sample_benchmark_result):
        """Test BenchmarkResponse with incomplete jobs."""
        response = BenchmarkResponse(
            results=sample_benchmark_result,
            all_job_ids=["job_123", "job_456", "job_789"],
            completed_job_ids=["job_123"],
            incomplete_job_ids=["job_456", "job_789"],
        )

        assert response.has_incomplete_jobs is True
        assert response.total_jobs == 3
        assert response.completed_count == 1
        assert response.incomplete_count == 2
        assert (
            abs(response.completion_rate - 33.333333333333336) < 0.0001
        )  # 1/3 * 100, allow for floating point precision

    def test_empty_response(self):
        """Test BenchmarkResponse with no jobs."""
        response = BenchmarkResponse(
            results=[], all_job_ids=[], completed_job_ids=[], incomplete_job_ids=[]
        )

        assert response.total_jobs == 0
        assert response.completed_count == 0
        assert response.incomplete_count == 0
        assert response.completion_rate == 100.0  # Edge case: no jobs = 100% complete


class TestPredictionResponse:
    """Test cases for PredictionResponse wrapper model."""

    def test_single_result_response(self, sample_prediction_result):
        """Test PredictionResponse with single result."""
        response = PredictionResponse(
            results=sample_prediction_result,
            all_job_ids=["job_123"],
            completed_job_ids=["job_123"],
            incomplete_job_ids=[],
        )

        assert isinstance(response.results, PredictionResult)
        assert response.all_job_ids == ["job_123"]
        assert response.completed_job_ids == ["job_123"]
        assert response.incomplete_job_ids == []

        # Test properties
        assert response.has_incomplete_jobs is False
        assert response.total_jobs == 1
        assert response.completed_count == 1
        assert response.incomplete_count == 0
        assert response.completion_rate == 100.0

    def test_multiple_results_response(self, sample_prediction_result, sample_device):
        """Test PredictionResponse with multiple results."""
        # Create second result
        result2 = PredictionResult(
            device=sample_device,
            outputs={"GPU": {"output": "/fake/output2.npy"}},
            job_id="job_456",
            status="Complete",
            modelid="model_123",
        )

        response = PredictionResponse(
            results=[sample_prediction_result, result2],
            all_job_ids=["job_123", "job_456"],
            completed_job_ids=["job_123", "job_456"],
            incomplete_job_ids=[],
        )

        assert isinstance(response.results, list)
        assert len(response.results) == 2
        assert response.total_jobs == 2
        assert response.completed_count == 2
        assert response.completion_rate == 100.0

    def test_incomplete_jobs_response(self, sample_prediction_result):
        """Test PredictionResponse with incomplete jobs."""
        response = PredictionResponse(
            results=sample_prediction_result,
            all_job_ids=["job_123", "job_456", "job_789", "job_abc"],
            completed_job_ids=["job_123", "job_456"],
            incomplete_job_ids=["job_789", "job_abc"],
        )

        assert response.has_incomplete_jobs is True
        assert response.total_jobs == 4
        assert response.completed_count == 2
        assert response.incomplete_count == 2
        assert response.completion_rate == 50.0  # 2/4 * 100

    def test_partial_completion_response(self, sample_prediction_result):
        """Test PredictionResponse with partial completion."""
        response = PredictionResponse(
            results=[sample_prediction_result],
            all_job_ids=["job_123", "job_456", "job_789"],
            completed_job_ids=["job_123"],
            incomplete_job_ids=["job_456", "job_789"],
        )

        assert (
            abs(response.completion_rate - 33.333333333333336) < 0.0001
        )  # Allow for floating point precision        assert response.has_incomplete_jobs is True

    def test_response_properties_consistency(self, sample_prediction_result):
        """Test that response properties are consistent."""
        response = PredictionResponse(
            results=[sample_prediction_result],
            all_job_ids=["job_1", "job_2", "job_3", "job_4", "job_5"],
            completed_job_ids=["job_1", "job_3"],
            incomplete_job_ids=["job_2", "job_4", "job_5"],
        )

        # Verify consistency
        assert response.total_jobs == len(response.all_job_ids)
        assert response.completed_count == len(response.completed_job_ids)
        assert response.incomplete_count == len(response.incomplete_job_ids)
        assert (
            response.completed_count + response.incomplete_count == response.total_jobs
        )

        # Verify completion rate calculation
        expected_rate = (response.completed_count / response.total_jobs) * 100
        assert response.completion_rate == expected_rate

