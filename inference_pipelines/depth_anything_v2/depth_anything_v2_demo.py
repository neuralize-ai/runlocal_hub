from pathlib import Path
from typing import Dict, List


import numpy as np
import PIL.Image
import torch

from PIL import Image
import coremltools as ct

import torch.nn.functional as F
import torch.nn as nn
from runlocal_tools.backend.coreml.coreml_model import CoreMLModel
from runlocal_tools.backend.ml_model import Model
from runlocal_tools.backend.onnx.onnx_model import OnnxModel


def run_depth_anything_v2_pipeline(model: Model, pipeline_input: Dict):
    """
    Runs the Depth Anything V2 pipeline on the given input image using the provided model.
    """
    model_size = 518

    # Preprocessing
    image = pipeline_input["image"]
    image_np = np.array(image)
    image_torch = torch.from_numpy(image_np).float()

    # Ensure image has 3 channels if it's grayscale
    if image_torch.ndim == 2:
        image_torch = image_torch.unsqueeze(-1).repeat(1, 1, 3)
    elif image_torch.shape[2] == 1: # Handle grayscale images with channel dim
        image_torch = image_torch.repeat(1, 1, 3)
    elif image_torch.shape[2] == 4: # Handle RGBA images
        image_torch = image_torch[:, :, :3]

    image_torch = image_torch.unsqueeze(0).permute(0, 3, 1, 2)
    forward_input_tensor = F.interpolate(
        image_torch,
        size=(model_size, model_size),
        mode="bilinear",
        align_corners=False,
    )

    output = model.execute({
        'x_1': forward_input_tensor.numpy().astype(np.float32)
    })
    gray_output = torch.from_numpy(output['gray_output'])
    gray_output = F.interpolate(gray_output, size=image_np.shape[:2], mode='bilinear', align_corners=False)
    gray_output = gray_output.squeeze(0).squeeze(0)

    # Normalize output
    depth_min = torch.min(gray_output)
    depth_max = torch.max(gray_output)

    if depth_max - depth_min > 1e-6:
        normalized_output = (gray_output - depth_min) / (depth_max - depth_min)
    else:
        normalized_output = torch.zeros_like(gray_output)

    gray_output_np = (normalized_output * 255.0).cpu().detach().numpy().astype(np.uint8)
    output_image = PIL.Image.fromarray(gray_output_np)

    output_image.save("output_depth_map.png")  # Save the output image
    output_image.show()


if __name__ == "__main__":
    example_image_path = Path(__file__).parent / "assets" / "example_image.jpg"

    example_image = Image.open(example_image_path)

    print("Loading model")
    model_path = Path(__file__).parent / "DepthAnythingV2SmallANE.mlpackage"
    model = CoreMLModel.from_path(model_path, compute_units=ct.ComputeUnit.ALL)

    # or, use ONNX
    # model_path = Path(__file__).parent / "DepthAnythingV2Small.onnx"
    # model = OnnxModel.from_path(model_path)


    pipeline_input_dict = {"image": example_image}
    run_depth_anything_v2_pipeline(model, pipeline_input_dict)



