#!/usr/bin/env python3

import sys
import json
from runlocal import RunLocalClient


def print_json(title, data):
    """Helper function to pretty-print JSON data"""
    print(f"\n=== {title} ===")
    try:
        # Use the custom encoder for JSON serialization
        json_str = json.dumps(data, indent=4)
        print(json_str)
    except Exception as e:
        print(f"Error serializing JSON: {e}")

    print()


def main():
    client = RunLocalClient()

    # Get models
    # model_ids = client.get_models()
    # print_json("Your Models", {"models": model_ids})

    model_id = "76bc0f19c9f38fbe2d76068f5048b1d8"

    # Simply select a device based on criteria
    print("\n=== Selecting a device for benchmarking ===")
    device_usage = client.select_device(
        model_id=model_id,
        device_name="MacBook",  # Optional: Filter by device name
        soc="Apple M4",  # Optional: Filter by SoC
        # ram=16,  # Optional: Filter by RAM amount
    )

    if not device_usage:
        print("No matching device found. Exiting.")
        sys.exit(1)

    print(
        f"Selected device: {device_usage.device.Name} {device_usage.device.Year} ({device_usage.device.Soc})"
    )
    print(f"Available compute units: {device_usage.compute_units}")

    print("\n=== Running Model Benchmark ===")
    try:
        device = device_usage.device
        compute_units = device_usage.compute_units

        benchmark_results = client.benchmark_model(
            model_id=model_id,
            device=device,
            compute_units=compute_units,
        )
        print_json("Benchmark Results", benchmark_results)

    except TimeoutError as e:
        print(f"Benchmark timed out: {e}")
        print("This is expected if benchmarks take longer than the timeout period")
    except Exception as e:
        print(f"Benchmark error: {e}")

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
