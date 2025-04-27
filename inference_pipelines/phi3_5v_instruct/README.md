# Phi-3.5 Vision Instruct Demo

This script demonstrates how to run the Phi-3.5 Vision Instruct model for multimodal tasks (text and image input) using Core ML models.

[Phi-3.5 Vision Instruct Huggingface](https://huggingface.co/edgemeter/Phi-3.5-Vision-Instruct)

## Running the Demo

1.  **Download the Required Core ML Models:**
    You need to download three separate Core ML models for this demo:

    *   **Text Embedding Model (`Phi3.5VTextEmbedding.mlpackage`):**
        Download from: [https://edgemeter.runlocal.ai/public/uploads/e5796daa955561cf6a63ceca5cef3e6c/benchmark](https://edgemeter.runlocal.ai/public/uploads/e5796daa955561cf6a63ceca5cef3e6c/benchmark)

    *   **Image Embedding Model (`Phi3.5VInstruct_ImageEmbedding.mlpackage`):**
        Download from: [https://edgemeter.runlocal.ai/public/uploads/2c050cda7067cfe28245abfb3b454bb3/benchmark](https://edgemeter.runlocal.ai/public/uploads/2c050cda7067cfe28245abfb3b454bb3/benchmark)

    *   **Text Decoder Model (`Phi3.5VInstruct_StatefulTextDecoder_4096.mlpackage`):**
        Download from: [https://edgemeter.runlocal.ai/public/uploads/cafb08ba2b3a12c530372d83695b9d5b/benchmark](https://edgemeter.runlocal.ai/public/uploads/cafb08ba2b3a12c530372d83695b9d5b/benchmark)

2.  **Place Models in Directory:**
    *   Place all three downloaded `.mlpackage` files into this directory (`inference_pipelines/phi3_5v_instruct/`).

3.  **Install Dependencies:**
    *   Ensure you have Python installed. Then, install the required dependencies from the `requirements.txt` file located in this directory:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Script:**
    *   Navigate to this directory in your terminal.
    *   Execute the script:
        ```bash
        python phi3_5v_instruct_demo.py
        ```

The script will:
*   Load the processor and the three Core ML models.
*   Download example images.
*   Run the inference process using the provided prompt and images.
*   Print the generated text response to the console.
