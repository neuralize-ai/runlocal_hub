#!/usr/bin/env python3

import numpy as np

from runlocal.client import RunLocalClient
from runlocal.devices import DeviceFilters


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
        compute_unit_outputs = client.predict(
            inputs=inputs,
            model_path=model_path,
            device_filters=device_filters,
            timeout=300,  # 5 minute timeout
        )

        print("Prediction Results:")
        for compute_unit, output_tensors in compute_unit_outputs.items():
            print(f"\nOutputs for compute unit '{compute_unit}':")
            for name, tensor in output_tensors.items():
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
                # Print first few values for small tensors
                if tensor.size <= 10:
                    print(f"    values: {tensor.flatten()}")
                else:
                    print(f"    values: {tensor.flatten()[:5]}... (showing first 5)")

    except Exception as e:
        print(f"Prediction failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
