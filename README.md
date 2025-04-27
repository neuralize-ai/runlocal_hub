![Company Logo](assets/logo.png)

# On-Device Model Inference Pipelines

This repository contains a collection of example inference pipelines designed to run various machine learning models directly on-device, targeting various formats like CoreML, ONNX, TFLite and OpenVINO.

## More Information

For detailed information on the pipelines, the specific models used, performance benchmarks, and model compression techniques, please visit:

[https://edgemeter.runlocal.ai/public/pipelines](https://edgemeter.runlocal.ai/public/pipelines)

## Setup

Before running any specific pipeline, ensure you have installed the base dependencies listed in the root `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Pipelines

Each sub-directory within `inference_pipelines/` contains a specific model pipeline, including:

*   A Python demo script (`*_demo.py`).
*   A `README.md` with instructions on downloading models and running the demo.
*   A pipeline-specific `requirements.txt` for necessary Python dependencies (install **after** the root requirements).

Explore the sub-directories to find specific examples like:

*   Depth Estimation (`depth_anything_v2`)
*   Multimodal Language Models (`phi3_5v_instruct`)
*   Text-to-Speech (`kokoro`)

*Note: Model files are typically large and are not included directly in the repository. Download links are provided in the respective README files.*
