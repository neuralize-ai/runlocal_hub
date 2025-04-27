# D-FINE Object Detection Demo

This script demonstrates how to run the D-FINE object detection model using either a CoreML or an ONNX model. The default is Core ML.

[Original D-FINE GitHub Repository](https://github.com/Peterande/D-FINE)

## Pipeline Homepage
* https://edgemeter.runlocal.ai/public/pipelines/d756f798-1487-423d-95cc-ea166637cd2e

## Running the Demo

1.  **Download the Default Model (Core ML):**
    *   Download the Core ML model (`D-FINE.mlpackage`) from the following URL:
        [https://edgemeter.runlocal.ai/public/uploads/434ed59ab442e5199f09f54033dd37f9/benchmark](https://edgemeter.runlocal.ai/public/uploads/434ed59ab442e5199f09f54033dd37f9/benchmark)
    *   Place the downloaded `D-FINE.mlpackage` file in this directory (`inference_pipelines/d_fine/`).

2.  **Download the Alternative Model (ONNX - Optional):**
    *   If you prefer to use the ONNX model, download `D-FINE.onnx` from:
        [https://edgemeter.runlocal.ai/public/uploads/a510b6b0913f4914107bf7a12923ce52/benchmark](https://edgemeter.runlocal.ai/public/uploads/a510b6b0913f4914107bf7a12923ce52/benchmark)
    *   Place the downloaded `D-FINE.onnx` file in this directory.

3.  **Modify the Script (Optional - for ONNX):**
    *   If you want to use the ONNX model you downloaded, open `d_fine_demo.py`.
    *   Comment out the lines loading the Core ML model:
        ```python
        # model_path = Path(__file__).parent / "D-FINE.mlpackage"
        # model = CoreMLModel.from_path(model_path)
        ```
    *   Uncomment the lines loading the ONNX model:
        ```python
        model_path = Path(__file__).parent / "D-FINE.onnx"
        model = OnnxModel.from_path(model_path)
        ```

4.  **Run the Script:**
    *   Navigate to this directory in your terminal.
    *   Execute the script:
        ```bash
        python d_fine_demo.py
        ```

The script will load the specified model (Core ML by default), process the `example_image.jpg`, save the image with detected bounding boxes as `annotated_output.jpg`, and display it.
