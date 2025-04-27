# Depth Anything V2 Demo

This script demonstrates how to run the Depth Anything V2 model for depth estimation using either a Core ML or an ONNX model. The default is Core ML.

[Depth Anything V2 Huggingface](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything)

## Running the Demo

1.  **Download the Default Model (Core ML):**
    *   Download the Core ML model (`DepthAnythingV2SmallANE.mlpackage`) from the following URL:
        [https://edgemeter.runlocal.ai/public/uploads/81e5ee9664a89ed0ea46c8c68a8b3bda/benchmark](https://edgemeter.runlocal.ai/demo/uploads/81e5ee9664a89ed0ea46c8c68a8b3bda/benchmark)
    *   Place the downloaded `DepthAnythingV2SmallANE.mlpackage` file in this directory (`inference_pipelines/depth_anything_v2/`).

2.  **Download the Alternative Model (ONNX - Optional):**
    *   If you prefer to use the ONNX model, download `DepthAnythingV2Small.onnx` from:
        [https://edgemeter.runlocal.ai/demo/uploads/49b7b3c0fba33db1593b4ee48cafa387/benchmark](https://edgemeter.runlocal.ai/public/uploads/49b7b3c0fba33db1593b4ee48cafa387/benchmark)
    *   Place the downloaded `DepthAnythingV2Small.onnx` file in this directory.

3.  **Modify the Script (Optional - for ONNX):**
    *   If you want to use the ONNX model you downloaded, open `depth_anything_v2_demo.py`.
    *   Comment out the lines loading the Core ML model:
        ```python
        # model_path = Path(__file__).parent / "DepthAnythingV2SmallANE.mlpackage"
        # model = CoreMLModel.from_path(model_path, compute_units=ct.ComputeUnit.ALL)
        ```
    *   Uncomment the lines loading the ONNX model:
        ```python
        model_path = Path(__file__).parent / "DepthAnythingV2Small.onnx"
        model = OnnxModel.from_path(model_path)
        ```

4.  **Run the Script:**
    *   Navigate to this directory in your terminal.
    *   Execute the script:
        ```bash
        python depth_anything_v2_demo.py
        ```

The script will load the specified model (Core ML by default), process the `example_image.jpg`, save the resulting depth map as `output_depth_map.png`, and display it.
