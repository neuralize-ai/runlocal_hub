# RunLocal Python Client

A Python client library for interacting with the RunLocal API for model optimization and benchmarking.

## Dependencies

```bash
pip install -r requirements.txt
```

## Installation

```sh
pip install .
```

## Authentication

You'll need an API key to use the RunLocal API. You can generate one in the RunLocal web interface:

1. Log in to RunLocal
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

The new API uses `DeviceFilters` for intuitive device selection:

```python
from runlocal.client import RunLocalClient
from runlocal.devices import DeviceFilters

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
