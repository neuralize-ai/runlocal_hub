"""
Rich console output utilities for job status display.
"""

from typing import List

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel

from ..models.benchmark import BenchmarkStatus
from ..models.job import JobResult, JobType


class StatusColors:
    """Color mapping for job statuses."""

    PENDING = "yellow"
    RUNNING = "cyan"
    COMPLETE = "green"
    FAILED = "red"
    DELETED = "dim"

    @classmethod
    def get_color(cls, status: BenchmarkStatus) -> str:
        """Get color for a given status."""
        return getattr(cls, status.value.upper(), "white")


class JobStatusDisplay:
    """Rich console display for job status updates."""

    def __init__(self):
        self.console = Console()
        self._live = None
        self._table = None

    def create_status_table(
        self, job_results: List[JobResult], job_type: JobType, elapsed_time: int = 0
    ) -> Table:
        """Create a rich table showing job statuses."""
        # Format elapsed time display
        elapsed_display = f"  Elapsed: {elapsed_time}s"

        table = Table(
            title=f"{job_type.value.title()} Jobs Status - {elapsed_display}",
            title_style="bold",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )

        # Add columns
        table.add_column("Device")
        table.add_column("SoC", style="cyan")
        table.add_column("RAM", style="dim", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        # Add rows
        for result in job_results:
            status_text = Text(
                result.status.value, style=StatusColors.get_color(result.status)
            )

            # Format details
            details = ""
            if result.status == BenchmarkStatus.Failed and result.error:
                details = Text(
                    result.error[:50] + "..."
                    if len(result.error) > 50
                    else result.error,
                    style="red",
                )
            elif result.status == BenchmarkStatus.Complete:
                details = Text("✓ Successfully completed", style="green")
            elif result.status == BenchmarkStatus.Running:
                details = Text("⚡ Processing...", style="cyan")
            elif result.status == BenchmarkStatus.Pending:
                details = Text("  Waiting in queue", style="yellow")

            # Extract device information
            device_name = result.device_name or "Unknown"
            soc = ""
            ram = ""

            if result.device:
                device_name = result.device.Name
                soc = result.device.Soc
                ram = f"{result.device.Ram}GB"

            table.add_row(
                device_name,
                soc,
                ram,
                status_text,
                details,
            )

        return table

    def start_live_display(
        self, initial_results: List[JobResult], job_type: JobType, elapsed_time: int = 0
    ):
        """Start a live updating display."""
        self._table = self.create_status_table(initial_results, job_type, elapsed_time)
        self._live = Live(self._table, console=self.console, refresh_per_second=2)
        self._live.start()

    def update_display(
        self, job_results: List[JobResult], job_type: JobType, elapsed_time: int = 0
    ):
        """Update the live display with new results."""
        if self._live and self._live.is_started:
            self._table = self.create_status_table(job_results, job_type, elapsed_time)
            self._live.update(self._table)

    def stop_display(self):
        """Stop the live display."""
        if self._live and self._live.is_started:
            self._live.stop()

        print("")

    def print_error(self, message: str):
        """Print an error message."""
        self.console.print(f"[red]✗ Error:[/red] {message}")

    def print_info(self, message: str):
        """Print an info message."""
        self.console.print(f"[blue]ℹ Info:[/blue] {message}")

    def print_success(self, message: str):
        """Print a success message."""
        self.console.print(f"[green]✓ Success:[/green] {message}")

    def print_warning(self, message: str):
        """Print a warning message."""
        self.console.print(f"[yellow]{message}[/yellow]")
