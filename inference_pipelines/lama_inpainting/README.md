# LaMa Inpainting Demo

This script demonstrates how to run the LaMa (Large Mask Inpainting) model using either a CoreML or an ONNX model.

[Original LaMa GitHub Repository](https://github.com/advimman/lama)

## Pipeline Homepage
* https://edgemeter.runlocal.ai/public/pipelines/a46781fc-0f69-4618-b61b-fa189d0068a5

## Running the Demo

1.  **Download the Model(s):**
    *   Download the ONNX model (`BigLaMa.onnx`):
        [https://edgemeter.runlocal.ai/public/uploads/e24f713fb223bef84898e68dbb90c659/benchmark]
    *   (Optional) Download the Core ML model (`BigLaMa.mlpackage`):
        [https://edgemeter.runlocal.ai/public/uploads/e88b964538568f480febe0abca1b0ce1/benchmark]
    *   Place the downloaded model file(s) in a location accessible to the script.

2.  **Prepare Input Files:**
    *   The demo includes an example image (`assets/example_image.png`) and mask (`assets/example_mask.png`) located in the `assets` subdirectory relative to the script.
    *   You can use these example files or provide paths to your own image and mask files using the `--image-path` and `--mask-path` arguments.

3.  **Run the Script:**
    *   Navigate to this directory (`inference_pipelines/lama_inpainting/`) in your terminal.
    *   Execute the script using command-line arguments to specify the paths:

        **Using ONNX Model (with example assets):**
        ```bash
        python lama_inpainting_demo.py \
          --model-path BigLaMa.onnx \
          --image-path assets/example_image.png \
          --mask-path assets/example_mask.png
        ```

        **Using CoreML Model (with example assets):**
        ```bash
        python lama_inpainting_demo.py \
          --model-path BigLaMa.mlpackage \
          --image-path assets/example_image.png \
          --mask-path assets/example_mask.png
        ```


The script will automatically detect the model type based on the file extension (`.onnx` or `.mlpackage`), load the model, process the image using the mask, and display the inpainted output image.
