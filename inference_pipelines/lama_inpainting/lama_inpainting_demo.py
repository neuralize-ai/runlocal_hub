import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from math import ceil
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import coremltools as ct
from runlocal_tools.backend.ml_model import Model
from runlocal_tools.backend.coreml.coreml_model import CoreMLModel
from runlocal_tools.backend.onnx.onnx_model import OnnxModel
import PIL


def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    crop_h = (h - target_size) // 2
    crop_w = (w - target_size) // 2
    return tensor[:, :, crop_h : crop_h + target_size, crop_w : crop_w + target_size]


def run_lama_inpainting(
    model: Model,
    image: PIL.Image,
    mask: PIL.Image,
):
    model_input_keys = ["image", "mask"]
    model_output_keys = ["inpainted"]

    dims = [512, 512]
    size = 512

    image = np.array(image.convert("RGB"))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32) / 255.0
    image_torch = torch.from_numpy(image)
    image_torch = image_torch.unsqueeze(0)

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask_torch = torch.from_numpy(mask)
    mask_torch = mask_torch.unsqueeze(0).unsqueeze(0)

    _, _, orig_h, orig_w = image_torch.size()
    scale_factor = size / min(orig_h, orig_w)

    new_h = ceil(orig_h * scale_factor)
    new_w = ceil(orig_w * scale_factor)

    image_torch = F.interpolate(
        image_torch,
        size=[new_h, new_w],
        mode="bilinear",
        align_corners=False,
    )

    mask_torch = F.interpolate(
        mask_torch,
        size=[new_h, new_w],
        mode="nearest",
    )

    image_torch = center_crop(image_torch, size)
    mask_torch = center_crop(mask_torch, size)

    mask_torch = (mask_torch > 0) * 1


    output = model.execute({
        "image": image_torch.detach().cpu().numpy().astype(np.float32),
        "mask": mask_torch.detach().cpu().numpy().astype(np.float32),
    })


    inpainted = torch.from_numpy(output["inpainted"])
    output = inpainted.squeeze(0)
    output = output.permute(1, 2, 0)

    output_array = output.detach().cpu().numpy().astype(np.uint8)
    output_image = Image.fromarray(output_array)
    output_image.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LaMa Inpainting Demo")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the ONNX or CoreML model file (.onnx or .mlpackage).")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--mask-path", type=str, required=True, help="Path to the input mask file.")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    mask_path = Path(args.mask_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Auto-detect model type based on extension
    if model_path.suffix == ".onnx":
        print(f"Loading ONNX model from: {model_path}")
        model = OnnxModel.from_path(model_path)
    elif model_path.suffix == ".mlpackage":
        print(f"Loading CoreML model from: {model_path}")
        # Assuming default compute units for CoreML, adjust if needed
        model = CoreMLModel.from_path(model_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    else:
        raise ValueError(f"Unsupported model file extension: {model_path.suffix}. Please use .onnx or .mlpackage")

    print("Running inpainting...")
    run_lama_inpainting(model, image, mask)
    print("Inpainting finished. Output image displayed.")





