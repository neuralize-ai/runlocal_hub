# CLIP Cosine Similarity Demo

This script demonstrates how to calculate cosine similarity between text descriptions and images using CLIP (Contrastive Language–Image Pre-training) embeddings. It compares a list of text prompts against a list of images and identifies the best matching image for each text.

[SentenceTransformers Multilingual CLIP Huggingface](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)


## Pipeline Homepage
* See more information about this pipeline here: https://edgemeter.runlocal.ai/public/pipelines/68f8b22d-e428-4cb5-b5c3-78ffb723ddf1


## Models Used

This demo primarily uses multilingual CLIP models optimized for Apple Neural Engine (ANE), but ONNX and OpenVINO models are also available.

*   **Image Embedding Model (Core ML):** `CLIPViTL14ImageEmbedding_NeuralEngineOptimized.mlpackage`
    *   Download: [https://edgemeter.runlocal.ai/public/uploads/680604c25826d7df685ed18bd57b26bd/benchmark](https://edgemeter.runlocal.ai/public/uploads/680604c25826d7df685ed18bd57b26bd/benchmark)
*   **Text Embedding Model (Core ML):** `SentenceTransformersMultilingualCLIPViTB32v1_NeuralEngineOptimized.mlpackage`
    *   Download: [https://edgemeter.runlocal.ai/public/uploads/8d144a28c3287ac7914209ec0fd2f277/benchmark](https://edgemeter.runlocal.ai/public/uploads/8d144a28c3287ac7914209ec0fd2f277/benchmark)
*   **Image Embedding Model (ONNX):** `CLIPViTL14ImageEmbedding_NeuralEngineOptimized.onnx`
    *   Download: [https://edgemeter.runlocal.ai/public/uploads/77c19b7b644ca2d44c7715c7a5ae1ae6/benchmark](https://edgemeter.runlocal.ai/public/uploads/77c19b7b644ca2d44c7715c7a5ae1ae6/benchmark)
*   **Text Embedding Model (ONNX):** `SentenceTransformersMultilingualCLIPViTB32v1_NeuralEngineOptimized.onnx`
    *   Download: [https://edgemeter.runlocal.ai/public/uploads/8d144a28c3287ac7914209ec0fd2f277/benchmark](https://edgemeter.runlocal.ai/public/uploads/8d144a28c3287ac7914209ec0fd2f277/benchmark)

## Running the Demo

1.  **Download the Required Models:**
    *   Download 2 models (the image embedding and text embedding models) linked above.

2.  **Place Models in Directory:**
    *   Place the downloaded files into this directory (`inference_pipelines/clip_cosine_similarity/`).
    *   *(Optional)* If using ONNX models, place them here as well and modify the demo script (`clip_cosine_similarity_demo.py`) to load them using `OnnxModel` instead of `CoreMLModel`.

3.  **(Optional) Prepare Assets:**
    *  The `assets` folder already contains example images, but you can add your own images here.

4.  **Install Dependencies:**
    *   Ensure you have Python installed.
    *  Install the required dependencies from the `requirements.txt` file located in this directory:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Script:**
    *   Navigate to this directory (`inference_pipelines/clip_cosine_similarity/`) in your terminal.
    *   Execute the script:
        ```bash
        python clip_cosine_similarity_demo.py
        ```

The script will:
*   Load the Core ML models (`image_model` and `text_model`).
*   Load images from the `assets` folder.
*   Preprocess images using `CLIPImageProcessor` and texts using `AutoTokenizer`.
*   Run inference through the models to obtain image and text embeddings.
*   Calculate the cosine similarity between each text embedding and all image embeddings.
*   Print the best matching image (filename and similarity score) for each text prompt to the console.
