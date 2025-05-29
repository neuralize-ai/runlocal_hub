"""
Job polling logic for async operations.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set

from ..exceptions import JobTimeoutError
from ..http import HTTPClient
from ..models import BenchmarkDbItem, BenchmarkStatus, JobType
from ..utils.decorators import handle_api_errors
from ..utils.json import convert_to_json_friendly


@dataclass
class JobResult:
    """
    Result of a job polling operation.
    """

    job_id: str
    status: BenchmarkStatus
    device_name: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    elapsed_time: Optional[int] = None

    @property
    def is_complete(self) -> bool:
        """Check if the job is complete (success or failure)."""
        return self.status in [BenchmarkStatus.Complete, BenchmarkStatus.Failed]

    @property
    def is_successful(self) -> bool:
        """Check if the job completed successfully."""
        return self.status == BenchmarkStatus.Complete

    @property
    def is_failed(self) -> bool:
        """Check if the job failed."""
        return self.status == BenchmarkStatus.Failed


class JobPoller:
    """
    Handles polling of async jobs until completion.
    """

    def __init__(self, http_client: HTTPClient, poll_interval: int = 10):
        """
        Initialize the job poller.

        Args:
            http_client: HTTP client for API requests
            poll_interval: Time in seconds between status checks
        """
        self.http_client = http_client
        self.poll_interval = poll_interval

    def poll_jobs(
        self,
        job_ids: List[str],
        job_type: JobType,
        device_names: Optional[List[str]] = None,
        timeout: int = 600,
        progress_callback: Optional[Callable[[JobResult], None]] = None,
    ) -> List[JobResult]:
        """
        Poll multiple jobs until completion.

        Args:
            job_ids: List of job IDs to poll
            job_type: Type of jobs being polled
            device_names: Optional list of device names corresponding to job_ids
            timeout: Maximum time in seconds to wait for completion
            progress_callback: Optional callback function called when each job completes

        Returns:
            List of job results

        Raises:
            JobTimeoutError: If not all jobs complete within timeout
        """
        if not job_ids:
            return []

        print(f"Waiting for {len(job_ids)} {job_type.value}(s) to complete...")

        start_time = time.time()
        results: List[JobResult] = []
        completed_ids: Set[str] = set()

        # Create device name mapping
        device_name_map = {}
        if device_names:
            for i, job_id in enumerate(job_ids):
                if i < len(device_names):
                    device_name_map[job_id] = device_names[i]

        while self._should_continue(start_time, timeout, completed_ids, job_ids):
            # Check each job
            for i, job_id in enumerate(job_ids):
                if job_id in completed_ids:
                    continue

                try:
                    result = self._check_job_status(
                        job_id=job_id,
                        device_name=device_name_map.get(job_id),
                        elapsed_time=int(time.time() - start_time),
                    )

                    if result.is_complete:
                        results.append(result)
                        completed_ids.add(job_id)

                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(result)

                        # Print progress
                        self._print_completion_message(
                            result, len(completed_ids), len(job_ids)
                        )

                except Exception as e:
                    # Log error but continue polling other jobs
                    print(f"Error checking {job_type.value} {job_id} status: {e}")

            # Break if all jobs complete
            if len(completed_ids) == len(job_ids):
                break

            # Wait before checking again
            time.sleep(self.poll_interval)

        # Check for timeout
        if len(completed_ids) < len(job_ids):
            incomplete_count = len(job_ids) - len(completed_ids)
            raise JobTimeoutError(
                f"Timeout: Only {len(completed_ids)}/{len(job_ids)} {job_type.value}s "
                f"completed within {timeout}s. {incomplete_count} still running. "
                f"Try increasing the timeout parameter or check your network connection.",
                timeout=timeout,
                completed_jobs=len(completed_ids),
                total_jobs=len(job_ids)
            )

        return results

    def poll_single_job(
        self,
        job_id: str,
        job_type: JobType,
        device_name: Optional[str] = None,
        timeout: int = 600,
        progress_callback: Optional[Callable[[JobResult], None]] = None,
    ) -> Optional[JobResult]:
        """
        Poll a single job until completion.

        Args:
            job_id: Job ID to poll
            job_type: Type of job being polled
            device_name: Optional device name for logging
            timeout: Maximum time in seconds to wait for completion
            progress_callback: Optional callback function called when job completes

        Returns:
            Job result

        Raises:
            JobTimeoutError: If job doesn't complete within timeout
        """
        results = self.poll_jobs(
            job_ids=[job_id],
            job_type=job_type,
            device_names=[device_name] if device_name else None,
            timeout=timeout,
            progress_callback=progress_callback,
        )

        return results[0] if results else None

    @handle_api_errors
    def _check_job_status(
        self,
        job_id: str,
        device_name: Optional[str] = None,
        elapsed_time: Optional[int] = None,
    ) -> Optional[JobResult]:
        """
        Check the status of a single job.

        Args:
            job_id: Job ID to check
            job_type: Type of job
            device_name: Optional device name
            elapsed_time: Optional elapsed time since start

        Returns:
            JobResult if job is complete, None otherwise
        """
        # Get benchmark data from API
        response = self.http_client.get(f"/benchmarks/id/{job_id}")
        benchmark = BenchmarkDbItem(**response)

        if benchmark.Status in [BenchmarkStatus.Complete, BenchmarkStatus.Failed]:
            # Extract error information for failed jobs
            error = None
            if benchmark.Status == BenchmarkStatus.Failed:
                error = self._extract_error_message(benchmark)

            # Convert benchmark data to JSON-friendly format
            result_data = convert_to_json_friendly(benchmark)

            return JobResult(
                job_id=job_id,
                status=benchmark.Status,
                device_name=device_name,
                data=result_data,
                error=error,
                elapsed_time=elapsed_time,
            )

        # Job not complete yet
        return None

    def _should_continue(
        self,
        start_time: float,
        timeout: int,
        completed_ids: Set[str],
        job_ids: List[str],
    ) -> bool:
        """
        Check if polling should continue.

        Args:
            start_time: When polling started
            timeout: Maximum time to wait
            completed_ids: Set of completed job IDs
            job_ids: List of all job IDs

        Returns:
            True if should continue polling
        """
        # Check timeout
        if time.time() - start_time >= timeout:
            return False

        # Check if all jobs complete
        if len(completed_ids) >= len(job_ids):
            return False

        return True

    def _extract_error_message(self, benchmark: BenchmarkDbItem) -> Optional[str]:
        """
        Extract error message from failed benchmark.

        Args:
            benchmark: Benchmark data

        Returns:
            Error message or None
        """
        # Look for failure reasons in benchmark data
        for data in benchmark.BenchmarkData:
            if data.FailureReason:
                return data.FailureReason
            if data.FailureError:
                return data.FailureError

        return "Unknown failure"

    def _print_completion_message(
        self,
        result: JobResult,
        completed_count: int,
        total_count: int,
    ) -> None:
        """
        Print a completion message for a finished job.

        Args:
            result: Job result
            completed_count: Number of completed jobs
            total_count: Total number of jobs
        """
        elapsed_str = f"[{result.elapsed_time}s]" if result.elapsed_time else ""
        device_str = f" ({result.device_name})" if result.device_name else ""

        if result.is_successful:
            print(
                f"{elapsed_str} Job {completed_count}/{total_count}{device_str} completed successfully"
            )
        else:
            error_msg = f": {result.error}" if result.error else ""
            print(
                f"{elapsed_str} Job {completed_count}/{total_count}{device_str} failed{error_msg}"
            )


class ProgressTracker:
    """
    Helper class for tracking job progress with callbacks.
    """

    def __init__(self):
        self.completed_jobs: List[JobResult] = []
        self.failed_jobs: List[JobResult] = []
        self.successful_jobs: List[JobResult] = []

    def __call__(self, result: JobResult) -> None:
        """
        Callback function to track job completion.

        Args:
            result: Completed job result
        """
        self.completed_jobs.append(result)

        if result.is_successful:
            self.successful_jobs.append(result)
        else:
            self.failed_jobs.append(result)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if not self.completed_jobs:
            return 0.0
        return (len(self.successful_jobs) / len(self.completed_jobs)) * 100

    def summary(self) -> str:
        """Get a summary string of the progress."""
        total = len(self.completed_jobs)
        successful = len(self.successful_jobs)
        failed = len(self.failed_jobs)

        return f"Completed: {total}, Successful: {successful}, Failed: {failed}, Success Rate: {self.success_rate:.1f}%"

