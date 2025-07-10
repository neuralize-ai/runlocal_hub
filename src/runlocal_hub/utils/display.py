"""
Display utilities for formatting benchmark results.
"""

from typing import List, Union

from rich.console import Console
from rich.table import Table

from ..models.benchmark_result import BenchmarkResult
from ..models.benchmark import BenchmarkStatus, BenchmarkTableSchema


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


def display_benchmark_table(
    table_data: Union[BenchmarkTableSchema, List[BenchmarkTableSchema]],
):
    """
    Display benchmark table data in a formatted table using rich.

    Args:
        table_data: List of BenchmarkTableSchema objects to display
        show_all_columns: If True, show all available columns. If False, show default columns only.
    """
    console = Console()

    if isinstance(table_data, BenchmarkTableSchema):
        table_data = [table_data]

    if len(table_data) == 0:
        console.print("[yellow]No benchmark table data to display[/yellow]")
        return

    # Create table
    table = Table(
        title="[yellow]📊[/yellow] Benchmark Table Data",
        title_style="bold",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        expand=True,
    )

    # Add columns
    table.add_column("Device")
    table.add_column("SoC", style="cyan")
    table.add_column("Compute Units", style="green")

    # Check if we have GenAI metrics in any row
    has_genai_metrics = any(
        row.PrefillTPS is not None or row.GenerateTPS is not None for row in table_data
    )

    if has_genai_metrics:
        table.add_column("Prefill Tokens", justify="right", style="yellow")
        table.add_column("Generate Tokens", justify="right", style="yellow")
        table.add_column("Prefill TPS", justify="right", style="yellow")
        table.add_column("Generate TPS", justify="right", style="yellow")
    else:
        table.add_column("Median Inference (ms)", justify="right", style="yellow")

    # Always show load time
    table.add_column("Median Load (ms)", justify="right", style="yellow")

    if has_genai_metrics:
        table.add_column("Peak Prefill RAM (MB)", justify="right", style="blue")
        table.add_column("Peak Generate RAM (MB)", justify="right", style="blue")
    else:
        table.add_column("Peak Inference RAM (MB)", justify="right", style="blue")

    table.add_column("Peak Load RAM (MB)", justify="right", style="blue")

    # Add rows
    for row in table_data:
        if row.Status != BenchmarkStatus.Complete:
            continue

        # Extract device info
        device_name = row.DeviceInfo.Name if row.DeviceInfo else "Unknown"
        device_soc = row.DeviceInfo.Soc if row.DeviceInfo else "Unknown"

        # Prepare row data
        row_data = [
            device_name,
            device_soc,
            row.ComputeUnits or "N/A",
        ]

        # Add performance metrics
        if has_genai_metrics:
            prefill_tokens = str(row.PrefillTokens) or "N/A"
            generate_tokens = str(row.GenerationTokens) or "N/A"
            prefill_tps = f"{row.PrefillTPS:.2f}" if row.PrefillTPS else "N/A"
            generate_tps = f"{row.GenerateTPS:.2f}" if row.GenerateTPS else "N/A"
            row_data.extend(
                [prefill_tokens, generate_tokens, prefill_tps, generate_tps]
            )
        else:
            inference_time = (
                f"{row.InferenceMsMedian:.2f}" if row.InferenceMsMedian else "N/A"
            )
            row_data.append(inference_time)

        # Load time
        load_time = f"{row.LoadMsMedian:.2f}" if row.LoadMsMedian else "N/A"
        row_data.append(load_time)

        if has_genai_metrics:
            prefill_ram = (
                str(int(row.PeakPrefillRamUsage)) if row.PeakPrefillRamUsage else "N/A"
            )
            generate_ram = (
                str(int(row.PeakGenerateRamUsage))
                if row.PeakGenerateRamUsage
                else "N/A"
            )
            row_data.extend([prefill_ram, generate_ram])
        else:
            inference_ram = str(int(row.PeakRamUsage)) if row.PeakRamUsage else "N/A"
            row_data.append(inference_ram)

        load_ram = str(int(row.PeakLoadRamUsage)) if row.PeakLoadRamUsage else "N/A"
        row_data.append(load_ram)

        table.add_row(*row_data)

    console.print(table)
