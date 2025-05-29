#!/usr/bin/env python3

import numpy as np

from runlocal import RunLocalClient


def main():
    client = RunLocalClient()

    model_path = "./HorizonAngle_exp0.mlpackage"

    image = np.zeros([1, 3, 224, 224]).astype(np.float32)
    inputs = {"image": image}

    compute_unit_outputs = client.predict(
        inputs=inputs,
        model_path=model_path,
        device_name="MacBook",
        soc="Apple M3",
        ram_min=18,
        ram_max=18,
    )

    for compute_unit, output_tensors in compute_unit_outputs.items():
        print(f"\nOutputs for compute unit '{compute_unit}':")
        for name, tensor in output_tensors.items():
            print(f"  {name}: {tensor.shape}")
            print(f"    {tensor}")


if __name__ == "__main__":
    main()
