#!/usr/bin/env python3

import numpy as np

from runlocal_hub import DeviceFilters, RunLocalClient, display_benchmark_results


def main():
    client = RunLocalClient()

    model_path = "model.mlpackage"
    image = np.zeros([1, 3, 224, 224]).astype(np.float32)
    inputs = {"image": image}

    device_filters = DeviceFilters(
        device_name="MacBook",  # Filter by device name
    )

    try:
        result = client.benchmark(
            model_path=model_path,
            device_filters=device_filters,
            inputs=inputs,
            timeout=300,  # 5 minute timeout
        )

        # Ensure result is a list for display function
        results = result if isinstance(result, list) else [result]

        # Display results using the new helper function
        display_benchmark_results(results)

        # Show output tensor file paths if available
        if results[0].outputs:
            print("\nOutput tensor files saved:")
            for compute_unit, tensors in results[0].outputs.items():
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
