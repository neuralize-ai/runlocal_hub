#!/usr/bin/env python3

from runlocal_hub import (
    RuntimeSettings,
    DeviceFilters,
    RunLocalClient,
    display_benchmark_results,
)


def main():
    client = RunLocalClient()

    model_path = "model.mlpackage"

    device_filters = DeviceFilters(
        device_name="MacBook",
        year_min=2023,
    )

    settings = RuntimeSettings()

    try:
        response = client.benchmark(
            model_path=model_path,
            settings=settings,
            device_filters=device_filters,
            timeout=600,
            skip_existing=True,
        )

        # Ensure result is a list for display function
        results = (
            response.results
            if isinstance(response.results, list)
            else [response.results]
        )

        display_benchmark_results(results, show_versions=True, show_settings=True)

    except Exception as e:
        print(f"Benchmark failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
