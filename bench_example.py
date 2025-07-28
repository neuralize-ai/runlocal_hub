#!/usr/bin/env python3

from runlocal_hub import DeviceFilters, RunLocalClient, display_benchmark_results


def main():
    client = RunLocalClient()

    model_path = "model.mlpackage"

    device_filters = DeviceFilters(
        device_name="MacBook",  # Filter by device name
    )

    try:
        result = client.benchmark(
            model_path=model_path,
            device_filters=device_filters,
            timeout=None,  # Block until finished
        )

        # Ensure result is a list for display function
        results = result if isinstance(result, list) else [result]

        display_benchmark_results(results, show_versions=True, show_settings=True)

    except Exception as e:
        print(f"Benchmark failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
