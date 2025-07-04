#!/usr/bin/env python3

import numpy as np

from runlocal_hub import DeviceFilters, RunLocalClient


def main():
    client = RunLocalClient()

    model_path = "model.mlpackage"
    image = np.zeros([1, 3, 224, 224]).astype(np.float32)
    inputs = {"image": image}

    device_filters = DeviceFilters(
        device_name="MacBook",  # Filter by device name
    )

    try:
        result = client.predict(
            inputs=inputs,
            model_path=model_path,
            device_filters=device_filters,
            timeout=300,  # 5 minute timeout
        )

        if isinstance(result, list):
            result = result[0]

        print("Prediction Results:")
        print(
            f"\nDevice: {result.device.Name} ({result.device.Soc}, {result.device.Ram}GB RAM)"
        )
        print(f"Job ID: {result.job_id}")
        print(f"Status: {result.status}")

        for compute_unit, output_tensors in result.outputs.items():
            print(f"\nOutputs for compute unit '{compute_unit}':")
            for name, path in output_tensors.items():
                print(f"  {name}: {path}")

                # Load and show tensor info
                tensor = np.load(path)
                print(f"    shape={tensor.shape}, dtype={tensor.dtype}")
                # Print first few values for small tensors
                if tensor.size <= 10:
                    print(f"    values: {tensor.flatten()}")
                else:
                    print(f"    values: {tensor.flatten()[:5]}... (showing first 5)")

    except Exception as e:
        print(f"Prediction failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
