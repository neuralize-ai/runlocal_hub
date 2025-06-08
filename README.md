# RunLocal Python Client

A Python client library for benchmarking and running machine learning models on real devices through the [RunLocal](https://edgemeter.runlocal.ai) API.

## What is RunLocal?

RunLocal provides cloud access to physical devices for testing and benchmarking ML models across platforms. This client library enables you to:

- **Benchmark ML models** on real hardware to measure performance metrics
- **Run inference** on actual devices to validate model outputs and accuracy
- **Test across platforms** - MacBooks, iPhones, iPads, Android devices, and Windows machines
- **Support multiple formats** - CoreML, ONNX, OpenVINO, and TensorFlow Lite models
- **Compare compute units** - test performance on CPU, GPU, and Neural Engine (where available)
- **Avoid hardware limitations** - no need to own every device you want to test on

Perfect for ML engineers who need to understand how their models perform across diverse hardware ecosystems before deployment.

## Dependencies

```bash
pip install -r requirements.txt
```

## Installation

```sh
pip install .
```

## Authentication

You'll need an API key to use the RunLocal API. You can generate one on the RunLocal web interface:

1. Log in to [RunLocal](https://edgemeter.runlocal.ai)
2. Go to your user settings (avatar dropdown)
3. Navigate to the "API Keys" section
4. Click "Create New API Key"
5. Save your API key in a secure location

## Usage

```bash
export RUNLOCAL_API_KEY=<your_api_key>
```

### Basic Examples

See the following example files for different usage patterns:

- `bench_example.py` - Simple benchmark example
- `predict_example.py` - Simple prediction example

### Device Filtering

The API uses `DeviceFilters` for intuitive device selection:

```python
from runlocal import DeviceFilters, RunLocalClient

client = RunLocalClient()

# Filter by device characteristics
device_filters = DeviceFilters(
    device_name="MacBook",     # Device name pattern
    soc="Apple M3",           # SoC type
    ram_min=16,               # Minimum RAM (GB)
    ram_max=32,               # Maximum RAM (GB)
    year_min=2022             # Minimum device year
    compute_units=["CPU_AND_GPU", "CPU_AND_NE"]     # Compute units to run
)

# Run benchmark with filters
result = client.benchmark(
    model_path="model.mlpackage",
    device_filters=device_filters,
    timeout=300
)
```

### Multi-Device Operations

Run operations on multiple devices simultaneously:

```python
# Use 2 devices
results = client.benchmark(
    model_path="model.mlpackage",
    count=2
)

# Use all available devices matching criteria
results = client.benchmark(
    model_path="model.mlpackage",
    device_filters=DeviceFilters(device_name="MacBook"),
    count=None  # None means use all matching devices
)

results = client.benchmark(
    model_path="model.onnx",
    device_filters=[
        # M1 or M2 Macs
        DeviceFilters(os="macOS", soc="M1"),
        DeviceFilters(os="macOS", soc="M2"),

        # High-end iPhones (6GB+ RAM)
        DeviceFilters(device_name="iPhone", ram_min=6),

        # Windows machines with 16GB+ RAM
        DeviceFilters(os="Windows", ram_min=16),
    ],
    count=None  # None means use all matching devices
)

# Default behavior (single device)
result = client.benchmark(model_path="model.mlpackage")  # count=1 by default
```

### Prediction with Inputs

Run model predictions with custom inputs:

```python
import numpy as np

# Prepare inputs
image = np.zeros([1, 3, 224, 224]).astype(np.float32)
inputs = {"image": image}

# Run prediction
outputs = client.predict(
    inputs=inputs,
    model_path="model.mlpackage",
    device_filters=DeviceFilters(device_name="iPhone"),
)

# Process outputs by compute unit
for compute_unit, tensors in outputs.items():
    print(f"Compute unit '{compute_unit}':")
    for name, tensor in tensors.items():
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
```
