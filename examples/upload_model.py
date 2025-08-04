#!/usr/bin/env python3

import sys
from pathlib import Path

from runlocal_hub import RunLocalClient, display_model


def main():
    if len(sys.argv) != 2:
        print("Usage: python upload_model.py <model_path>")
        print("Example: python upload_model.py model.mlpackage")
        print("Example: python upload_model.py model.onnx")
        return

    model_path = sys.argv[1]

    if not Path(model_path).exists():
        print(f"Error: Model path '{model_path}' does not exist.")
        return

    client = RunLocalClient()

    print(f"Uploading model from: {model_path}")

    try:
        upload_id = client.upload_model(model_path)
        print(f"Upload successful! Model ID: {upload_id}")

        # Fetch and display the uploaded model information
        model_info = client.get_model(upload_id)
        print("\nUploaded model details:")
        display_model(model_info)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    main()

