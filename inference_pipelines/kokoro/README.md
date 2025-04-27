# Kokoro Text-to-Speech Demo

This script demonstrates how to run the Kokoro text-to-speech model using Core ML models to generate audio from text input.

[Kokoro Huggingface](https://huggingface.co/hexgrad/Kokoro-82M)

## Running the Demo

1.  **Download the Required Core ML Models:**
    You need to download two separate Core ML models for this demo:

    *   **Predict Duration Model (`Kokoro_DurationPrediction.mlpackage`):**
        Download from: [https://edgemeter.runlocal.ai/public/uploads/434d119ce918ef22198c3169b920df6c/benchmark](https://edgemeter.runlocal.ai/public/uploads/434d119ce918ef22198c3169b920df6c/benchmark)

    *   **Waveform Generation Model (`Kokoro_GenerateWaveform.mlpackage`):**
        Download from: [https://edgemeter.runlocal.ai/public/uploads/b3d6d9599b74f3c86f780d976ccf9441/benchmark](https://edgemeter.runlocal.ai/public/uploads/b3d6d9599b74f3c86f780d976ccf9441/benchmark)

    *(Note: Please rename the downloaded files to `PredictDuration.mlpackage` and `WaveformGeneration.mlpackage` respectively, or update the script with the actual downloaded names if they differ.)*

2.  **Place Models in Directory:**
    *   Place both downloaded `.mlpackage` files into this directory (`inference_pipelines/kokoro/`).

3.  **Install Dependencies:**
    *   Ensure you have Python installed. Then, install the required dependencies from the `requirements.txt` file located in this directory:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Run the Script:**
    *   Navigate to this directory in your terminal.
    *   Execute the script (assuming it's named `kokoro_demo.py`):
        ```bash
        python kokoro_demo.py
        ```

The script will load the models, process the input text, and generate an audio waveform (saving it to a WAV file). Check the script for exact output details.
