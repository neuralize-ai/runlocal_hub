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
        result = client.benchmark(
            model_path=model_path,
            device_filters=device_filters,
            inputs=inputs,
            timeout=300,  # 5 minute timeout
        )

        if isinstance(result, list):
            result = result[0]

        print("Benchmark Results:")
        print(
            f"\nDevice: {result.device.Name} ({result.device.Soc}, {result.device.Ram}GB RAM)"
        )
        print(f"Job ID: {result.job_id}")
        print(f"Status: {result.status}")
        print(f"Elapsed time: {result.elapsed_time:.2f}s")

        print(f"\nPerformance data for {len(result.benchmark_data)} compute unit(s):")
        for bd in result.benchmark_data:
            if bd.Success:
                print(f"\n{bd.ComputeUnit}:")
                if bd.LoadMsArray:
                    print(f"  Median Load time: {np.median(bd.LoadMsArray):.2f} ms")
                if bd.InferenceMsArray:
                    print(
                        f"  Median Inference time: {np.median(bd.InferenceMsArray):.2f} ms"
                    )
            else:
                print(f"\n{bd.ComputeUnit}: FAILED")

        # Show output tensor file paths if available
        if result.output_tensors:
            print("\nOutput tensor files saved:")
            for compute_unit, tensors in result.output_tensors.items():
                print(f"  {compute_unit}:")
                for name, path in tensors.items():
                    print(f"    {name}: {path}")

                    # Show how to load the tensor back
                    loaded_tensor = np.load(path)
                    print(
                        f"      (shape={loaded_tensor.shape}, dtype={loaded_tensor.dtype})"
                    )

    except Exception as e:
        print(f"Benchmark failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
