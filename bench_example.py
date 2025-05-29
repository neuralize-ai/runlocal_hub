#!/usr/bin/env python3

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

    try:
        benchmark_results = client.benchmark(
            model_path=model_path,
            device_name="MacBook",
            soc="Apple M3",
            ram_min=18,
            ram_max=18,
        )
        print_json("Benchmark Results", benchmark_results)

    except Exception as e:
        print(f"Benchmark error: {e}")
        return

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
