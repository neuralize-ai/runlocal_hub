from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

import os

# Assuming runlocal_tools structure based on the reference example
from runlocal_tools.backend.ml_model import Model
from runlocal_tools.backend.coreml.coreml_model import CoreMLModel
from runlocal_tools.backend.onnx.onnx_model import OnnxModel


# Class names moved outside the class
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def run_d_fine_pipeline(model: Model, pipeline_input: Dict) -> Dict:
    """
    Runs the D-FINE detection pipeline on the given input image using the provided model.
    """
    image_pil = pipeline_input["image"].convert("RGB")
    original_image_for_drawing = image_pil.copy()
    w, h = image_pil.size
    orig_size = torch.tensor([[w, h]], dtype=torch.float32)

    # Preprocessing
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(image_pil).unsqueeze(0)  # Shape [1, 3, 640, 640]


    model_output = model.execute({
        'image': im_data.numpy().astype(np.float32) # Convert to numpy if model expects it
    })

    labels = torch.from_numpy(model_output['labels'])
    boxes = torch.from_numpy(model_output['boxes'])
    scores = torch.from_numpy(model_output['scores'])

    # Postprocessing
    # Scale boxes back to original image dimensions
    img_w, img_h = orig_size[0].tolist()
    scale_factor = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    boxes = boxes * scale_factor.unsqueeze(0).unsqueeze(0) # Shape [1, 300, 4]

    # Convert tensors to numpy arrays for filtering and drawing
    labels_np = labels[0].detach().cpu().numpy()  # Shape [300]
    boxes_np = boxes[0].detach().cpu().numpy()  # Shape [300, 4]
    scores_np = scores[0].detach().cpu().numpy()  # Shape [300]

    # Apply a confidence threshold
    conf_threshold = 0.5
    indices = np.where(scores_np >= conf_threshold)[0]
    filtered_labels = labels_np[indices]
    filtered_boxes = boxes_np[indices]
    filtered_scores = scores_np[indices]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(original_image_for_drawing)
    for label, box, score in zip(filtered_labels, filtered_boxes, filtered_scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_index = int(label)
        if 0 <= class_index < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_index]
        else:
            class_name = "Unknown" # Handle out-of-bounds index

        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1), f"{class_name}:{score:.2f}", fill='red')

    # Optionally, reassemble full outputs (boxes6 has [x1, y1, x2, y2, score, class])
    if len(filtered_scores) > 0:
        filtered_scores_np_r = filtered_scores.reshape(-1, 1)
        filtered_labels_np_r = filtered_labels.reshape(-1, 1)
        boxes6 = np.concatenate([filtered_boxes, filtered_scores_np_r, filtered_labels_np_r], axis=1)
    else:
        boxes6 = np.empty((0, 6), dtype=np.float32)

    return {
        "annotated_image": original_image_for_drawing,
        "labels": filtered_labels,
        "boxes": filtered_boxes,
        "scores": filtered_scores,
        "boxes6": boxes6
    }


if __name__ == "__main__":


    image_path = Path(__file__).parent / "assets" / "example_image.png"


    model_path = Path(__file__).parent / "D-FINE.mlpackage"
    model = CoreMLModel.from_path(model_path)

    # or, use ONNX
    # model_path = Path(__file__).parent / "D-FINE.onnx"
    # model = OnnxModel.from_path(model_path)

    # Process as image
    image = Image.open(image_path).convert("RGB")
    pipeline_input = {"image": image}

    print("Running D-FINE detection pipeline...")
    pipeline_output = run_d_fine_pipeline(model, pipeline_input)

    annotated_image = pipeline_output["annotated_image"]

    # Save or show the annotated image
    output_image_path = Path(__file__).parent / "assets" / "annotated_output.png"
    annotated_image.save(output_image_path)
    print(f"Annotated image saved to: {output_image_path}")
    annotated_image.show() # Uncomment to display the image directly

    print("Image processing complete.")
    print(f"Detected {len(pipeline_output['labels'])} objects.")