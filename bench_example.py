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

    model_path = "./HorizonAngle_exp0.mlpackage"

    model_id = client.upload_model(model_path)

    print(f"model_id: {model_id}")

    # Simply select a device based on criteria
    print("\n=== Selecting a device for benchmarking ===")
    device_usage = client.select_device(
        model_id=model_id,
        device_name="MacBook",  # Optional: Filter by device name
        soc="Apple M3",  # Optional: Filter by SoC
        ram_min=18,  # Optional: Minimum RAM amount (inclusive)
        ram_max=18,  # Optional: Maximum RAM amount (inclusive)
        # year_min=2023,  # Optional: Minimum year (inclusive)
    )

    if not device_usage:
        print("No matching device found.")

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
        return
    except Exception as e:
        print(f"Benchmark error: {e}")
        return

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
