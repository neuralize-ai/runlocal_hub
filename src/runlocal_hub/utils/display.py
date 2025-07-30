"""
Display utilities for formatting benchmark results.
"""

from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models.benchmark_result import BenchmarkResult
from ..models.model import UploadDbItem


def _format_versions(versions: Optional[Dict[str, str]]) -> str:
    """
    Format version dictionary into a readable string with newlines.

    Args:
        versions: Dictionary of version information

    Returns:
        Formatted string representation with each entry on a new line
    """
    if not versions:
        return "N/A"

    # Format as "key: value" pairs, one per line
    pairs = [f"{k}: {v}" for k, v in versions.items()]
    return "\n".join(pairs)


def _format_settings(settings: Optional[Dict[str, Any]], indent: int = 0) -> str:
    """
    Format settings dictionary into a readable string with proper indentation.

    Args:
        settings: Dictionary of settings (can be nested)
        indent: Current indentation level

    Returns:
        Formatted string representation with nested structure
    """
    if not settings:
        return "N/A"

    lines = []
    indent_str = "  " * indent

    for key, value in settings.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.append(_format_settings(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}: [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f"{indent_str}{key}: {value}")

    return "\n".join(lines)


def display_benchmark_results(
    results: Union[BenchmarkResult, List[BenchmarkResult]],
    show_mean: bool = False,
    show_inference_array: bool = False,
    show_load_array: bool = False,
    show_ram_usage: bool = False,
    show_failed_benchmarks: bool = False,
    show_versions: bool = False,
    show_settings: bool = False,
):
    """
    Display benchmark results in a formatted table using rich.

    Args:
        results: List of benchmark results to display
        show_mean: Show average times instead of median
        show_inference_array: Show full inference time arrays
        show_load_array: Show full load time arrays
        show_ram_usage: Show RAM usage metrics
        show_failed_benchmarks: Show details about failed benchmarks
        show_versions: Show version information from benchmarks
        show_settings: Show settings information from benchmarks
    """
    console = Console()

    if isinstance(results, BenchmarkResult):
        results = [results]

    if len(results) == 0:
        console.print("[yellow]No benchmark results to display[/yellow]")
        return

    _display_grouped_results(
        results,
        show_mean,
        show_inference_array,
        show_load_array,
        show_ram_usage,
        show_versions,
        show_settings,
        console,
    )

    if show_failed_benchmarks:
        display_failed_benchmarks(results)


def _display_grouped_results(
    results: List[BenchmarkResult],
    show_mean: bool,
    show_inference_array: bool,
    show_load_array: bool,
    show_ram_usage: bool,
    show_versions: bool,
    show_settings: bool,
    console: Console,
):
    """Display results grouped by device in a single table."""
    # Create single table for all results
    table = Table(
        title="[yellow]⚡[/yellow] Benchmark Results",
        title_style="bold",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        expand=True,
    )

    # Add basic columns
    table.add_column("Device")
    table.add_column("SoC", style="cyan")
    table.add_column("RAM", style="dim", justify="right")
    table.add_column("Compute Unit", style="green")

    if show_versions:
        table.add_column("Versions", style="dim")
    if show_settings:
        table.add_column("Settings", style="dim")

    # Add time columns based on preferences
    if show_mean:
        table.add_column("Mean Inference (ms)", justify="right", style="yellow")
        table.add_column("Mean Load (ms)", justify="right", style="yellow")
    else:
        table.add_column("Median Inference (ms)", justify="right", style="yellow")
        table.add_column("Median Load (ms)", justify="right", style="yellow")

    if show_inference_array:
        table.add_column("Inference Array", style="dim")
    if show_load_array:
        table.add_column("Load Array", style="dim")
    if show_ram_usage:
        table.add_column("Peak Inference RAM (MB)", justify="right", style="blue")
        table.add_column("Peak Load RAM (MB)", justify="right", style="blue")

    # Process all results
    for result in results:
        device = result.device
        successful_benchmarks = [
            b
            for b in result.benchmark_data
            if b.Status != "Failed" and b.Success is not False
        ]

        if not successful_benchmarks:
            continue

        # Add rows for each compute unit
        for i, benchmark_data in enumerate(successful_benchmarks):
            row = []

            # Show device info only on first row for this device
            if i == 0:
                row.extend([device.Name, device.Soc, f"{device.Ram} GB"])
            else:
                row.extend(["", "", ""])

            row.append(benchmark_data.ComputeUnit)

            if show_versions:
                row.append(_format_versions(benchmark_data.Versions))

            if show_settings:
                row.append(_format_settings(benchmark_data.Settings))

            # Time metrics
            if show_mean:
                inference_time = (
                    f"{benchmark_data.InferenceMsAverage:.2f}"
                    if benchmark_data.InferenceMsAverage
                    else "N/A"
                )
                load_time = (
                    f"{benchmark_data.LoadMsAverage:.2f}"
                    if benchmark_data.LoadMsAverage
                    else "N/A"
                )
            else:
                inference_time = (
                    f"{benchmark_data.InferenceMsMedian:.2f}"
                    if benchmark_data.InferenceMsMedian
                    else "N/A"
                )
                load_time = (
                    f"{benchmark_data.LoadMsMedian:.2f}"
                    if benchmark_data.LoadMsMedian
                    else "N/A"
                )

            row.extend([inference_time, load_time])

            if show_inference_array:
                if benchmark_data.InferenceMsArray:
                    array_str = ", ".join(
                        [f"{t:.1f}" for t in benchmark_data.InferenceMsArray[:5]]
                    )
                    if len(benchmark_data.InferenceMsArray) > 5:
                        array_str += "..."
                    row.append(array_str)
                else:
                    row.append("N/A")

            if show_load_array:
                if benchmark_data.LoadMsArray:
                    array_str = ", ".join(
                        [f"{t:.1f}" for t in benchmark_data.LoadMsArray[:5]]
                    )
                    if len(benchmark_data.LoadMsArray) > 5:
                        array_str += "..."
                    row.append(array_str)
                else:
                    row.append("N/A")

            if show_ram_usage:
                inference_ram = (
                    f"{benchmark_data.PeakInferenceRamUsage:.1f}"
                    if benchmark_data.PeakInferenceRamUsage
                    else "N/A"
                )
                load_ram = (
                    f"{benchmark_data.PeakLoadRamUsage:.1f}"
                    if benchmark_data.PeakLoadRamUsage
                    else "N/A"
                )
                row.extend([inference_ram, load_ram])

            table.add_row(*row)

    console.print(table)


def display_failed_benchmarks(
    results: Union[BenchmarkResult, List[BenchmarkResult]],
):
    """
    Display details about failed benchmark runs.

    Args:
        results: List of benchmark results to check for failures
    """
    console = Console()

    if isinstance(results, BenchmarkResult):
        results = [results]

    failed_results = []
    for result in results:
        for benchmark_data in result.benchmark_data:
            if benchmark_data.Status == "Failed" or benchmark_data.Success is False:
                failed_results.append((result, benchmark_data))

    if not failed_results:
        return

    console.print("\n[bold red]❌ Failed Benchmarks[/bold red]")

    for result, benchmark_data in failed_results:
        console.print(
            f"\n[yellow]Device:[/yellow] {result.device.Name} ({result.device.Soc})"
        )
        console.print(f"[yellow]Compute Unit:[/yellow] {benchmark_data.ComputeUnit}")

        if benchmark_data.FailureReason:
            console.print(f"[red]Failure Reason:[/red] {benchmark_data.FailureReason}")

        if benchmark_data.FailureError:
            console.print(f"[red]Error:[/red] {benchmark_data.FailureError}")

        if benchmark_data.Stderr:
            console.print("[red]Stderr:[/red]")
            console.print(benchmark_data.Stderr)


def display_model(model: UploadDbItem):
    """
    Display a single model's information in a formatted panel.

    Args:
        model: UploadDbItem object to display
    """
    console = Console()

    # Format file size
    file_size_mb = float(model.FileSize) / (1024 * 1024)
    if file_size_mb >= 1024:
        file_size_str = f"{file_size_mb / 1024:.2f} GB"
    else:
        file_size_str = f"{file_size_mb:.2f} MB"

    # Build the content
    content = Text()

    # Basic info
    content.append("Model ID: ", style="bold cyan")
    content.append(f"{model.UploadId}\n\n")

    content.append("File Name: ", style="bold")
    content.append(f"{model.FileName}\n")

    if model.ModelType:
        content.append("Model Type: ", style="bold")
        content.append(f"{model.ModelType.value}\n")

    content.append("File Size: ", style="bold")
    content.append(f"{file_size_str}\n\n")

    # Optional fields
    if model.Tag:
        content.append("Tag: ", style="bold")
        content.append(f"{model.Tag}\n")

    if model.Source:
        content.append("Source: ", style="bold")
        content.append(f"{model.Source}\n")

    if model.License:
        content.append("\nLicense: ", style="bold")
        content.append(f"{model.License.name}\n")
        content.append("License URL: ", style="bold")
        content.append(f"{model.License.url}\n", style="blue underline")

    # Create panel
    panel = Panel(
        content,
        title="[bold yellow]📦 Model Information[/bold yellow]",
        border_style="cyan",
        expand=False,
        padding=(1, 2),
    )

    console.print(panel)


def display_incomplete_panel(incomplete_job_ids, job_type: str):
    console = Console()

    content = Text()
    content.append(
        "Some jobs didn't complete within the timeout.\n", style="bold yellow"
    )
    content.append("You can access the results later:\n\n")

    content.append("• Using the web client:\n", style="bold cyan")
    content.append("\tCheck the benchmark table on the model's page.\n\n")

    content.append("• Checking job status:\n", style="bold cyan")
    content.append(f"\tclient.check_multiple_jobs({incomplete_job_ids})\n\n")

    content.append("• Resuming polling:\n", style="bold cyan")
    content.append(
        f"\tclient.get_{job_type}_results({incomplete_job_ids}, timeout=...)\n",
    )
    content.append(
        "\nYou can use response.incomplete_job_ids to automate this behaviour"
    )

    panel = Panel(
        content,
        title="[bold yellow]⚠️  Incomplete Jobs[/bold yellow]",
        border_style="yellow",
        expand=False,
        padding=(1, 2),
    )

    console.print(panel)
