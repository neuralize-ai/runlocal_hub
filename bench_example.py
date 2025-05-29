#!/usr/bin/env python3

import json

import numpy as np

from runlocal.client import RunLocalClient
from runlocal.devices import DeviceFilters


def print_json(title, data):
    """Helper function to pretty-print JSON data"""
    print(f"\n=== {title} ===")
    try:
        # Convert numpy arrays to JSON-serializable format
        data_copy = convert_numpy_to_json(data)
        json_str = json.dumps(data_copy, indent=4)
        print(json_str)
    except Exception as e:
        print(f"Error serializing JSON: {e}")

    print()


def convert_numpy_to_json(obj):
    """Recursively convert numpy arrays to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return {
            "data": obj.tolist(),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj


def main():
    client = RunLocalClient()

    model_path = "./models/HorizonAngle_exp0.mlpackage"
    image = np.zeros([1, 3, 224, 224]).astype(np.float32)
    inputs = {"image": image}

    device_filters = DeviceFilters(
        device_name="MacBook",  # Filter by device name
        soc="Apple M3",  # Filter by SoC
        ram_min=18,  # Minimum RAM requirement
        ram_max=18,  # Maximum RAM requirement
    )

    try:
        benchmark_results = client.benchmark(
            model_path=model_path,
            device_filters=device_filters,
            inputs=inputs,
            timeout=300,  # 5 minute timeout
        )
        print_json("Benchmark Results", benchmark_results)

    except Exception as e:
        print(f"Benchmark failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
