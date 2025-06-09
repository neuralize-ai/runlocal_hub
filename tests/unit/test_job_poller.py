"""
Tests for job polling functionality.
"""

import time
from unittest.mock import Mock, patch
import pytest
from runlocal_hub.jobs.poller import JobPoller, ProgressTracker
from runlocal_hub.models.job import JobResult, JobType
from runlocal_hub.models.benchmark import (
    BenchmarkStatus,
    BenchmarkDbItem,
    BenchmarkData,
)
from runlocal_hub.models.device import Device


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client."""
    return Mock()


@pytest.fixture
def job_poller(mock_http_client):
    """Create a JobPoller instance with mock HTTP client."""
    return JobPoller(mock_http_client, poll_interval=1)  # Short interval for tests


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
def sample_benchmark_data():
    """Create sample benchmark data."""
    return BenchmarkData(
        Success=True, Status=BenchmarkStatus.Complete, ComputeUnit="CPU"
    )


@pytest.fixture
def sample_benchmark_item(sample_device, sample_benchmark_data):
    """Create a sample benchmark database item."""
    return BenchmarkDbItem(
        UploadId="test-upload-123",
        DeviceInfo=sample_device,
        Status=BenchmarkStatus.Complete,
        BenchmarkData=[sample_benchmark_data],
    )


class TestJobPoller:
    """Test cases for JobPoller functionality."""

    def test_init(self, mock_http_client):
        """Test JobPoller initialization."""
        poller = JobPoller(mock_http_client, poll_interval=5)
        assert poller.http_client == mock_http_client
        assert poller.poll_interval == 5

    def test_init_default_interval(self, mock_http_client):
        """Test JobPoller initialization with default interval."""
        poller = JobPoller(mock_http_client)
        assert poller.poll_interval == 10

    def test_poll_jobs_empty_list(self, job_poller):
        """Test polling with empty job list."""
        result = job_poller.poll_jobs([], JobType.BENCHMARK)
        assert result == []

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    @patch("runlocal_hub.jobs.poller.handle_api_errors", lambda func: func)
    def test_poll_single_job_success(
        self, mock_display, job_poller, mock_http_client, sample_benchmark_item
    ):
        """Test successful polling of a single job."""
        # Mock display
        mock_display_instance = Mock()
        mock_display.return_value = mock_display_instance

        # Mock API response
        mock_http_client.get.return_value = sample_benchmark_item.model_dump()

        result = job_poller.poll_single_job(
            "job-123", JobType.BENCHMARK, "iPhone 15 Pro"
        )

        assert result is not None
        assert result.job_id == "job-123"
        assert result.status == BenchmarkStatus.Complete
        assert result.device_name == "iPhone 15 Pro"
        assert result.is_complete
        assert result.is_successful

        # Verify display methods were called
        mock_display_instance.start_live_display.assert_called_once()
        mock_display_instance.stop_display.assert_called_once()

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    @patch("runlocal_hub.jobs.poller.handle_api_errors", lambda func: func)
    def test_poll_multiple_jobs_success(
        self, mock_display, job_poller, mock_http_client, sample_benchmark_item
    ):
        """Test successful polling of multiple jobs."""
        # Mock display
        mock_display_instance = Mock()
        mock_display.return_value = mock_display_instance

        # Mock API responses for different jobs
        def mock_get(endpoint):
            if "job-1" in endpoint:
                item = sample_benchmark_item.model_copy()
                item.UploadId = "upload-1"
                return item.model_dump()
            elif "job-2" in endpoint:
                item = sample_benchmark_item.model_copy()
                item.UploadId = "upload-2"
                return item.model_dump()

        mock_http_client.get.side_effect = mock_get

        results = job_poller.poll_jobs(
            ["job-1", "job-2"], JobType.BENCHMARK, device_names=["Device 1", "Device 2"]
        )

        assert len(results) == 2
        assert all(r.is_successful for r in results)
        assert results[0].device_name == "Device 1"
        assert results[1].device_name == "Device 2"

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    @patch("runlocal_hub.jobs.poller.handle_api_errors", lambda func: func)
    def test_poll_jobs_with_failure(
        self, mock_display, job_poller, mock_http_client, sample_benchmark_item
    ):
        """Test polling jobs with one failure."""
        # Mock display
        mock_display_instance = Mock()
        mock_display.return_value = mock_display_instance

        # Create failed benchmark item
        failed_item = sample_benchmark_item.model_copy()
        failed_item.Status = BenchmarkStatus.Failed
        failed_item.BenchmarkData[0].Status = BenchmarkStatus.Failed
        failed_item.BenchmarkData[0].FailureReason = "Test failure"

        mock_http_client.get.return_value = failed_item.model_dump()

        results = job_poller.poll_jobs(["job-failed"], JobType.BENCHMARK)

        assert len(results) == 1
        assert results[0].status == BenchmarkStatus.Failed
        assert results[0].is_failed
        assert not results[0].is_successful
        assert "Test failure" in results[0].error

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    @patch("runlocal_hub.jobs.poller.handle_api_errors", lambda func: func)
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_poll_jobs_timeout(
        self, mock_sleep, mock_display, job_poller, mock_http_client
    ):
        """Test polling with timeout."""
        # Mock display
        mock_display_instance = Mock()
        mock_display.return_value = mock_display_instance

        # Create running job that never completes
        running_item = BenchmarkDbItem(
            UploadId="test-upload", Status=BenchmarkStatus.Running, BenchmarkData=[]
        )
        mock_http_client.get.return_value = running_item.model_dump()

        # Use very short timeout
        results = job_poller.poll_jobs(["job-timeout"], JobType.BENCHMARK, timeout=1)

        # Should return empty list for incomplete jobs
        assert len(results) == 0

        # Verify warning was printed
        mock_display_instance.print_warning.assert_called()
        warning_call = mock_display_instance.print_warning.call_args[0][0]
        assert "Timeout" in warning_call
        assert "0/1" in warning_call

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    @patch("runlocal_hub.jobs.poller.handle_api_errors", lambda func: func)
    def test_check_job_status_complete(
        self, mock_display, job_poller, mock_http_client, sample_benchmark_item
    ):
        """Test checking status of completed job."""
        mock_http_client.get.return_value = sample_benchmark_item.model_dump()

        result = job_poller._check_job_status("job-123", "Test Device")

        assert result.job_id == "job-123"
        assert result.status == BenchmarkStatus.Complete
        assert result.device_name == "Test Device"
        assert result.is_complete
        assert result.data is not None

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    @patch("runlocal_hub.jobs.poller.handle_api_errors", lambda func: func)
    def test_check_job_status_running(self, mock_display, job_poller, mock_http_client):
        """Test checking status of running job."""
        running_item = BenchmarkDbItem(
            UploadId="test-upload", Status=BenchmarkStatus.Running, BenchmarkData=[]
        )
        mock_http_client.get.return_value = running_item.model_dump()

        result = job_poller._check_job_status("job-running")

        assert result.job_id == "job-running"
        assert result.status == BenchmarkStatus.Running
        assert not result.is_complete
        assert result.data is None

    def test_should_continue_logic(self, job_poller):
        """Test the _should_continue logic."""
        start_time = time.time()

        # Should continue when jobs incomplete and no timeout
        assert job_poller._should_continue(start_time, 60, set(), ["job1", "job2"])

        # Should not continue when all jobs complete
        assert not job_poller._should_continue(
            start_time, 60, {"job1", "job2"}, ["job1", "job2"]
        )

        # Should not continue when timeout exceeded
        past_time = start_time - 61
        assert not job_poller._should_continue(past_time, 60, set(), ["job1"])

    def test_extract_error_message(self, job_poller):
        """Test error message extraction from failed benchmarks."""
        # Test with FailureReason
        benchmark_data = BenchmarkData(
            ComputeUnit="CPU", FailureReason="Model loading failed"
        )
        benchmark = BenchmarkDbItem(
            UploadId="test",
            Status=BenchmarkStatus.Failed,
            BenchmarkData=[benchmark_data],
        )

        error = job_poller._extract_error_message(benchmark)
        assert error == "Model loading failed"

        # Test with FailureError
        benchmark_data.FailureReason = None
        benchmark_data.FailureError = "Runtime error"
        error = job_poller._extract_error_message(benchmark)
        assert error == "Runtime error"

        # Test with no error info
        benchmark_data.FailureError = None
        error = job_poller._extract_error_message(benchmark)
        assert error == "Unknown failure"

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    @patch("runlocal_hub.jobs.poller.handle_api_errors", lambda func: func)
    def test_poll_jobs_with_progress_callback(
        self, mock_display, job_poller, mock_http_client, sample_benchmark_item
    ):
        """Test polling with progress callback."""
        # Mock display
        mock_display_instance = Mock()
        mock_display.return_value = mock_display_instance

        mock_http_client.get.return_value = sample_benchmark_item.model_dump()

        callback_results = []

        def progress_callback(result):
            callback_results.append(result)

        results = job_poller.poll_jobs(
            ["job-1"], JobType.BENCHMARK, progress_callback=progress_callback
        )

        assert len(results) == 1
        assert len(callback_results) == 1
        assert callback_results[0].job_id == "job-1"

    @patch("runlocal_hub.jobs.poller.JobStatusDisplay")
    def test_poll_jobs_with_api_error(self, mock_display, job_poller, mock_http_client):
        """Test polling when API returns error."""
        # Mock display
        mock_display_instance = Mock()
        mock_display.return_value = mock_display_instance

        # Mock API to raise exception
        mock_http_client.get.side_effect = Exception("API Error")

        results = job_poller.poll_jobs(["job-error"], JobType.BENCHMARK, timeout=1)

        # Should handle error gracefully and return empty results
        assert len(results) == 0

        # Verify error was printed
        mock_display_instance.print_error.assert_called()


class TestProgressTracker:
    """Test cases for ProgressTracker functionality."""

    def test_init(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker()
        assert tracker.completed_jobs == []
        assert tracker.failed_jobs == []
        assert tracker.successful_jobs == []

    def test_track_successful_job(self):
        """Test tracking a successful job."""
        tracker = ProgressTracker()
        result = JobResult(job_id="job-1", status=BenchmarkStatus.Complete)

        tracker(result)

        assert len(tracker.completed_jobs) == 1
        assert len(tracker.successful_jobs) == 1
        assert len(tracker.failed_jobs) == 0
        assert tracker.success_rate == 100.0

    def test_track_failed_job(self):
        """Test tracking a failed job."""
        tracker = ProgressTracker()
        result = JobResult(
            job_id="job-1", status=BenchmarkStatus.Failed, error="Test error"
        )

        tracker(result)

        assert len(tracker.completed_jobs) == 1
        assert len(tracker.successful_jobs) == 0
        assert len(tracker.failed_jobs) == 1
        assert tracker.success_rate == 0.0

    def test_track_mixed_results(self):
        """Test tracking mixed successful and failed jobs."""
        tracker = ProgressTracker()

        # Add successful job
        successful = JobResult(job_id="job-1", status=BenchmarkStatus.Complete)
        tracker(successful)

        # Add failed job
        failed = JobResult(job_id="job-2", status=BenchmarkStatus.Failed)
        tracker(failed)

        # Add another successful job
        successful2 = JobResult(job_id="job-3", status=BenchmarkStatus.Complete)
        tracker(successful2)

        assert len(tracker.completed_jobs) == 3
        assert len(tracker.successful_jobs) == 2
        assert len(tracker.failed_jobs) == 1
        assert (
            abs(tracker.success_rate - 66.67) < 0.1
        )  # 2/3 * 100, allow for floating point precision

    def test_success_rate_no_jobs(self):
        """Test success rate calculation with no jobs."""
        tracker = ProgressTracker()
        assert tracker.success_rate == 0.0

    def test_summary(self):
        """Test progress summary generation."""
        tracker = ProgressTracker()

        # Add some jobs
        tracker(JobResult(job_id="job-1", status=BenchmarkStatus.Complete))
        tracker(JobResult(job_id="job-2", status=BenchmarkStatus.Failed))

        summary = tracker.summary()
        assert "Completed: 2" in summary
        assert "Successful: 1" in summary
        assert "Failed: 1" in summary
        assert "Success Rate: 50.0%" in summary


class TestJobResult:
    """Test cases for JobResult functionality."""

    def test_job_result_properties_complete(self):
        """Test JobResult properties for completed job."""
        result = JobResult(job_id="job-1", status=BenchmarkStatus.Complete)

        assert result.is_complete
        assert result.is_successful
        assert not result.is_failed

    def test_job_result_properties_failed(self):
        """Test JobResult properties for failed job."""
        result = JobResult(
            job_id="job-1", status=BenchmarkStatus.Failed, error="Test error"
        )

        assert result.is_complete
        assert not result.is_successful
        assert result.is_failed

    def test_job_result_properties_running(self):
        """Test JobResult properties for running job."""
        result = JobResult(job_id="job-1", status=BenchmarkStatus.Running)

        assert not result.is_complete
        assert not result.is_successful
        assert not result.is_failed

    def test_job_result_properties_pending(self):
        """Test JobResult properties for pending job."""
        result = JobResult(job_id="job-1", status=BenchmarkStatus.Pending)

        assert not result.is_complete
        assert not result.is_successful
        assert not result.is_failed

