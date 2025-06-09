<h1 align="center">
    <a href="https://runlocal.ai">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="./assets/logo_dark_mode.svg">
            <source media="(prefers-color-scheme: light)" srcset="./assets/logo_light_mode.svg">
            <img alt="runlocal_hub Logo" src="./assets/logo_dark_mode.svg.svg" height="42" style="max-width: 100%;">
        </picture>
    </a>
</h1>

<p align="center">
    Python client for benchmarking and validating ML models on real devices via RunLocal API.
</p>

<p align="center">
    <a href="https://pypi.org/project/runlocal-hub/"><img src="https://badge.fury.io/py/runlocal-hub.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/runlocal-hub/"><img src="https://img.shields.io/pypi/pyversions/runlocal-hub.svg" alt="Python Versions"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<br/>

<div align="center">
  <img src="./assets/benchmark.gif" alt="RunLocal Benchmark Demo" width="800">
</div>

## 🎯 Key Features

- **⚡ Real Hardware Testing** - No simulators or emulators. Test on real devices maintained in our devices lab
- **🌍 Cross-Platform Coverage** - Access MacBooks, iPhones, iPads, Android, and Windows devices from a single API
- **🔧 Multiple ML Formats** - Support for CoreML, ONNX, OpenVINO, TensorFlow Lite, and GGUF models. More frameworks coming soon.
- **📊 Detailed Metrics** - Measure inference time, memory usage, and per-layer performance data
- **🚦 CI/CD Ready** - Integrate performance and accuracy testing into your deployment pipeline

## 🔍 Evaluate Results

All benchmarks performed through the python client can be evaluated on the web platform by logging into your account.
Check out our [public demo models](https://edgemeter.runlocal.ai/public/pipelines) for comprehensive benchmark evaluation across different devices and model formats.

## 🛠 Installation

```bash
pip install runlocal-hub
```

### Development Installation

For development or to install from source:

```bash
git clone https://github.com/neuralize-ai/runlocal_hub.git
cd runlocal_hub
pip install -e .
```

## 🔑 Authentication

Get your API key from the [RunLocal dashboard](https://edgemeter.runlocal.ai):

1. Log in to [RunLocal](https://edgemeter.runlocal.ai)
2. Click your avatar → User Settings
3. Navigate to "API Keys"
4. Click "Create New API Key"
5. Save your key securely

```bash
export RUNLOCAL_API_KEY=<your_api_key>
```

## 🕹 Usage Guide

### Simple Benchmark

```python
from runlocal_hub import RunLocalClient

client = RunLocalClient()

# Benchmark on any available device
result = client.benchmark("model.mlpackage")
print(f"Inference time: {result.mean_inference_time}ms")
```

### Device Filtering

Target specific devices with intuitive filters:

```python
from runlocal_hub import DeviceFilters, RunLocalClient

client = RunLocalClient()

# High-end MacBooks with M-series chips
mac_filters = DeviceFilters(
    device_name="MacBook",
    soc="Apple M",        # Matches M1, M2, M3, etc.
    ram_min=16,           # At least 16GB RAM
    year_min=2021         # Recent models only
)

# Latest iPhones with Neural Engine
iphone_filters = DeviceFilters(
    device_name="iPhone",
    year_min=2022,
    compute_units=["CPU_AND_NE"]
)

# Run benchmarks
results = client.benchmark(
    "model.mlpackage",
    device_filters=[mac_filters, iphone_filters],
    count=None  # Use all matching devices
)
```

### 🧮 Running Predictions

Test your model with real inputs:

```python
import numpy as np

# Prepare input
image = np.random.rand(1, 3, 224, 224).astype(np.float32)
inputs = {"image": image}

# Run prediction on iPhone
outputs = client.predict(
    inputs=inputs,
    model_path="model.mlpackage",
    device_filters=DeviceFilters(device_name="iPhone 15", compute_units=["CPU_AND_NE"])
)

tensors = outputs["CPU_AND_NE"]
for name, tensor in tensors.items():
    print(f"  {name}: {tensor.shape} ({tensor.dtype})")
    print(f"  First values: {tensor.flatten()[:5]}")
```

## 📚 Examples

Check out the example scripts:

- [`bench_example.py`](./bench_example.py) - Simple benchmarking example
- [`predict_example.py`](./predict_example.py) - Prediction with custom inputs, serialised outputs

## 💠 Supported Formats

| Format          | Extension                   | Platforms       |
| --------------- | --------------------------- | --------------- |
| CoreML          | `.mlpackage`/`.mlmodel`     | macOS, iOS      |
| ONNX            | `.onnx`                     | Windows, MacOS  |
| OpenVINO        | directory (`.xml` + `.bin`) | Windows (Intel) |
| TensorFlow Lite | `.tflite`                   | Android         |
| GGUF            | `.gguf`                     | All platforma   |

More frameworks coming soon.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
