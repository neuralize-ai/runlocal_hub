#!/usr/bin/env python3

import sys
import json
import numpy as np
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
    print("\n=== Selecting a device for prediction ===")
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

    input_tensors = {"image": np.random.rand(1, 3, 224, 224).astype(np.float32)}

    print("\n=== Running Prediction ===")
    try:
        device = device_usage.device
        compute_units = device_usage.compute_units

        # Run prediction
        compute_unit_outputs = client.predict_model(
            model_id=model_id,
            device=device,
            compute_units=compute_units,
            input_tensors=input_tensors,
        )

        # Access outputs for each compute unit
        for compute_unit, output_tensors in compute_unit_outputs.items():
            print(f"\nOutputs for compute unit '{compute_unit}':")
            for name, tensor in output_tensors.items():
                print(f"{name}: {tensor.shape}")
                print(f"\t{tensor}")

    except TimeoutError as e:
        print(f"Prediction timed out: {e}")
        print("This is expected if predictions take longer than the timeout period")
        return
    except Exception as e:
        print(f"Benchmark error: {e}")
        return

    print("\nPrediction completed successfully!")


if __name__ == "__main__":
    main()
