"""
Job polling logic for async operations.
"""

import time
from typing import Callable, List, Optional, Set

from ..http import HTTPClient
from ..models import BenchmarkDbItem, BenchmarkStatus, JobResult, JobType
from ..utils.console import JobStatusDisplay
from ..utils.decorators import handle_api_errors
from ..utils.json import convert_to_json_friendly


class JobPoller:
    """
    Handles polling of async jobs until completion.
    """

    def __init__(
        self, http_client: HTTPClient, poll_interval: int = 10, verbosity: int = 2
    ):
        """
        Initialize the job poller.

        Args:
            http_client: HTTP client for API requests
            poll_interval: Time in seconds between status checks
            verbosity: Control output verbosity (0=silent, 1=minimal, 2=normal, 3=verbose, 4=debug)
        """
        self.http_client = http_client
        self.poll_interval = poll_interval
        self.verbosity = verbosity

    def poll_jobs(
        self,
        job_ids: List[str],
        job_type: JobType,
        timeout: Optional[int] = 600,
        progress_callback: Optional[Callable[[JobResult], None]] = None,
    ) -> List[JobResult]:
        """
        Poll multiple jobs until completion.

        Args:
            job_ids: List of job IDs to poll
            job_type: Type of jobs being polled
            timeout: Maximum time in seconds to wait for completion
            progress_callback: Optional callback function called when each job completes

        Returns:
            List of job results

        Raises:
            JobTimeoutError: If not all jobs complete within timeout
        """
        if not job_ids:
            return []

        # Initialize display based on verbosity
        display = None
        if self.verbosity >= 2:
            display = JobStatusDisplay()

        start_time = time.time()
        results: List[JobResult] = []
        completed_ids: Set[str] = set()

        # Track all job states for display
        all_job_results: List[JobResult] = []

        # Initialize job results for display
        for job_id in job_ids:
            all_job_results.append(
                JobResult(
                    job_id=job_id,
                    status=BenchmarkStatus.Pending,
                )
            )

        # Start appropriate display based on verbosity level
        if display:
            # Full table display for verbosity >= 2
            display.start_live_display(all_job_results, job_type, 0)
        elif self.verbosity == 1:
            # Simple progress display for verbosity == 1
            self._print_simple_progress(len(completed_ids), len(job_ids), 0)

        try:
            while True:
                # Update elapsed time for all jobs
                elapsed = int(time.time() - start_time)

                # Check each job
                for job_id in job_ids:
                    if job_id in completed_ids:
                        continue

                    try:
                        result = self._check_job_status(
                            job_id=job_id,
                        )

                        # Update the job result in our tracking list
                        for j, job_result in enumerate(all_job_results):
                            if job_result.job_id == job_id:
                                if result is not None:
                                    all_job_results[j] = result
                                else:
                                    # Job still running, update status
                                    all_job_results[j].status = BenchmarkStatus.Running
                                break

                        if result is not None and result.is_complete:
                            results.append(result)
                            completed_ids.add(job_id)

                            # Call progress callback if provided
                            if progress_callback:
                                progress_callback(result)

                    except Exception as e:
                        # Update job with error status
                        for j, job_result in enumerate(all_job_results):
                            if job_result.job_id == job_id:
                                all_job_results[j].status = BenchmarkStatus.Failed
                                all_job_results[j].error = str(e)
                                break
                        if display:
                            display.print_error(
                                f"Error checking {job_type.value} {job_id}: {e}"
                            )

                # Update display with current status based on verbosity
                if display:
                    # Full table display
                    display.update_display(all_job_results, job_type, elapsed)
                elif self.verbosity == 1:
                    # Simple progress display (overwrite previous line)
                    self._print_simple_progress(
                        len(completed_ids), len(job_ids), elapsed
                    )

                # Break if all jobs complete
                if len(completed_ids) == len(job_ids):
                    break

                # Check if we should continue before sleeping
                if not self._should_continue(
                    start_time, timeout, completed_ids, job_ids
                ):
                    break

                # Wait before checking again
                time.sleep(self.poll_interval)

        finally:
            # Stop the live display if it was started
            if display:
                display.stop_display()
            elif self.verbosity == 1:
                # Print a final newline for simple progress
                print()

        return results

    def poll_single_job(
        self,
        job_id: str,
        job_type: JobType,
        timeout: int = 600,
        progress_callback: Optional[Callable[[JobResult], None]] = None,
    ) -> Optional[JobResult]:
        """
        Poll a single job until completion.

        Args:
            job_id: Job ID to poll
            job_type: Type of job being polled
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
            timeout=timeout,
            progress_callback=progress_callback,
        )

        return results[0] if results else None

    @handle_api_errors
    def _check_job_status(
        self,
        job_id: str,
    ) -> Optional[JobResult]:
        """
        Check the status of a single job.

        Args:
            job_id: Job ID to check
            device_name: Optional device name

        Returns:
            JobResult with current status
        """
        # Get benchmark data from API
        response = self.http_client.get(f"/benchmarks/id/{job_id}")
        benchmark = BenchmarkDbItem(**response)

        # Extract error information for failed jobs
        error = None
        if benchmark.Status == BenchmarkStatus.Failed:
            error = self._extract_error_message(benchmark)

        # Convert benchmark data to JSON-friendly format if complete
        result_data = None
        if benchmark.Status in [BenchmarkStatus.Complete, BenchmarkStatus.Failed]:
            result_data = convert_to_json_friendly(benchmark)

        return JobResult(
            job_id=job_id,
            status=benchmark.Status,
            device=benchmark.DeviceInfo,
            data=result_data,
            error=error,
        )

    def _should_continue(
        self,
        start_time: float,
        timeout: Optional[int],
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
        # Check if all jobs complete
        if len(completed_ids) >= len(job_ids):
            return False

        # Check timeout
        if timeout is not None and time.time() - start_time >= timeout:
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

    def _print_simple_progress(self, completed: int, total: int, elapsed: int) -> None:
        """
        Print simple progress display for verbosity level 1.

        Args:
            completed: Number of completed jobs
            total: Total number of jobs
            elapsed: Elapsed time in seconds
        """
        fraction = f"{completed}/{total}"
        elapsed_str = f"{elapsed}s"
        # Use carriage return to overwrite the same line
        print(
            f"\rProgress: {fraction} completed, elapsed: {elapsed_str}",
            end="",
            flush=True,
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
