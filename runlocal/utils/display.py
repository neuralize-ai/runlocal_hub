"""
Display utilities for formatting benchmark results.
"""

from typing import List, Optional, Union
from rich.console import Console
from rich.table import Table
from ..models.benchmark_result import BenchmarkResult


def display_benchmark_results(
    results: Union[BenchmarkResult, List[BenchmarkResult]],
    show_average: bool = False,
    show_inference_array: bool = False,
    show_load_array: bool = False,
    show_ram_usage: bool = False,
    title: Optional[str] = "Benchmark Results",
) -> None:
    """
    Display benchmark results in a formatted table using rich.

    Args:
        results: List of benchmark results to display
        show_average: Show average times instead of median
        show_inference_array: Show full inference time arrays
        show_load_array: Show full load time arrays
        show_ram_usage: Show RAM usage metrics
        title: Table title
    """
    console = Console()

    if isinstance(results, BenchmarkResult):
        results = [results]

    if len(results) == 0:
        console.print("[yellow]No benchmark results to display[/yellow]")
        return

    # Create table
    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add basic columns
    table.add_column("Device", style="cyan", no_wrap=True)
    table.add_column("SoC", style="cyan")
    table.add_column("RAM", style="cyan")
    table.add_column("Compute Unit", style="green")

    # Add time columns based on preferences
    if show_average:
        table.add_column("Avg Inference (ms)", justify="right", style="yellow")
        table.add_column("Avg Load (ms)", justify="right", style="yellow")
    else:
        table.add_column("Median Inference (ms)", justify="right", style="yellow")
        table.add_column("Median Load (ms)", justify="right", style="yellow")

    # Optional columns
    if show_inference_array:
        table.add_column("Inference Array", style="dim")
    if show_load_array:
        table.add_column("Load Array", style="dim")
    if show_ram_usage:
        table.add_column("Peak Load RAM (MB)", justify="right", style="blue")
        table.add_column("Peak Inference RAM (MB)", justify="right", style="blue")

    # Add rows
    for result in results:
        device = result.device

        for benchmark_data in result.benchmark_data:
            row = []

            # Basic device info
            row.append(device.Name)
            row.append(device.Soc)
            row.append(f"{device.Ram} GB")
            row.append(benchmark_data.ComputeUnit)

            # Time metrics
            if show_average:
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

            row.append(inference_time)
            row.append(load_time)

            # Optional columns
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
                load_ram = (
                    f"{benchmark_data.PeakLoadRamUsage:.1f}"
                    if benchmark_data.PeakLoadRamUsage
                    else "N/A"
                )
                inference_ram = (
                    f"{benchmark_data.PeakInferenceRamUsage:.1f}"
                    if benchmark_data.PeakInferenceRamUsage
                    else "N/A"
                )
                row.append(load_ram)
                row.append(inference_ram)

            table.add_row(*row)

    console.print(table)


def display_failed_benchmarks(
    results: Union[BenchmarkResult, List[BenchmarkResult]],
) -> None:
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

    console.print("\n[red bold]Failed Benchmarks:[/red bold]")

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

