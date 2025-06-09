"""
Display utilities for formatting benchmark results.
"""

from typing import List, Union

from rich.console import Console
from rich.table import Table

from ..models.benchmark_result import BenchmarkResult


def display_benchmark_results(
    results: Union[BenchmarkResult, List[BenchmarkResult]],
    show_mean: bool = False,
    show_inference_array: bool = False,
    show_load_array: bool = False,
    show_ram_usage: bool = False,
    show_failed_benchmarks: bool = False,
):
    """
    Display benchmark results in a formatted table using rich.

    Args:
        results: List of benchmark results to display
        show_average: Show average times instead of median
        show_inference_array: Show full inference time arrays
        show_load_array: Show full load time arrays
        show_ram_usage: Show RAM usage metrics
        show_failed_benchmarks: Show details about failed benchmarks
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

    # Add time columns based on preferences
    if show_mean:
        table.add_column("Mean Inference (ms)", justify="right", style="yellow")
        table.add_column("Mean Load (ms)", justify="right", style="yellow")
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
                row.extend([load_ram, inference_ram])

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
