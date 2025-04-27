from pathlib import Path
from typing import Dict, List, Any
import os

import PIL.Image
import numpy as np
import requests
import torch
from PIL import Image
import coremltools as ct
# Use AutoTokenizer for efficient tokenization without loading full model weights
from transformers import CLIPImageProcessor, AutoTokenizer
from transformers.image_utils import PILImageResampling

from runlocal_tools.backend.coreml.coreml_model import CoreMLModel
from runlocal_tools.backend.ml_model import Model



# maximum sequence length for the text model
# Default based on original runner script, can be overridden in main
MAX_SEQ_LENGTH = 64


def calculate_cosine_similarity(target: np.ndarray, output: np.ndarray) -> float:
    dot_product = np.dot(target, output)
    norm_target = np.linalg.norm(target)
    norm_output = np.linalg.norm(output)
    if norm_target == 0 or norm_output == 0:
        return 0.0  # Avoid division by zero
    similarity = dot_product / (norm_target * norm_output)
    return similarity


def load_image(url_or_path: str) -> PIL.Image.Image:
    """Loads an image from a URL or local path."""
    print(f"Loading image from: {url_or_path}")
    try:
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            response = requests.get(url_or_path, stream=True, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            img = Image.open(response.raw)
        else:
            path = Path(url_or_path)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = Image.open(path)
        # Ensure image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image URL {url_or_path}: {e}")
        raise
    except FileNotFoundError as e:
        print(f"Error opening image file {url_or_path}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred loading image {url_or_path}: {e}")
        raise


def run_clip_cosine_similarity_pipeline(
    image_model: Model,
    text_model: Model,
    image_list: List[PIL.Image],
    text_list: List[str],
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> Dict:
    """
    Runs the CLIP Cosine Similarity pipeline using CoreML models.

    Args:
        image_model: The loaded CoreML image embedding model (CoreMLModel instance).
        text_model: The loaded CoreML text embedding model (CoreMLModel instance).
        pipeline_input: A dictionary containing 'image_list' (List[PIL.Image])
                        and 'text_list' (List[str]).
        max_seq_length: Maximum sequence length for text tokenization.

    Returns:
        A dictionary containing the image embeddings, text embeddings,
        and cosine similarity scores. Returns None if errors occur.
    """
    print("Starting CLIP Cosine Similarity pipeline...")

    # ===== 1. Preprocess Images (using Hugging Face Transformers Processor) =====
    print("Preprocessing images...")

    image_processor = CLIPImageProcessor(
        do_resize=True, size=224, resample=PILImageResampling.BICUBIC,
        do_center_crop=True, crop_size=224,
        do_rescale=True, rescale_factor=1 / 255,
        do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True # Already handled in load_image, but safe to keep
    )
    pixel_values_tensor = image_processor(image_list, return_tensors="pt").pixel_values
    pixel_values_np_list = [img_np for img_np in pixel_values_tensor.cpu().detach().numpy()]

    # ===== 2. Preprocess Texts (using Sentence Transformers Tokenizer) =====
    print("Preprocessing texts...")
    # Using the multilingual CLIP model for tokenizer reference
    model_name_or_path = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    text_model_input_dict = tokenizer(
        text_list,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="np" # Return numpy arrays for CoreML
    )

    # ===== 3. Run Image Inference (CoreML, Batch=1) =====
    print("Running image inference...")
    image_model_outputs = []

    image_input_key = image_model.get_input_keys()[0] # Assume single input
    image_output_key = image_model.get_output_keys()[0] # Assume single output

    for i, image_input_np in enumerate(pixel_values_np_list):
        input_dict = {
            image_input_key: image_input_np[None, :].astype(np.float32)
        }
        # Assuming image_model is CoreMLModel with `model.predict`
        output_dict = image_model.execute(input_dict)
        image_sentence_embedding = output_dict[image_output_key][0] # Remove batch dim
        image_model_outputs.append(torch.from_numpy(image_sentence_embedding))

    # ===== 4. Run Text Inference (CoreML, Batch=1) =====
    print("Running text inference...")
    text_model_outputs = []

    text_input_keys = text_model.get_input_keys() # Should be ['input_ids', 'attention_mask'] or similar
    text_output_key = text_model.get_output_keys()[0] # Assume single output

    # Determine the correct keys for input_ids and attention_mask
    # Common names: 'input_ids', 'attention_mask', 'pixel_values' (less likely for text)
    input_ids_key = next((k for k in text_input_keys if 'ids' in k.lower()), None)
    attention_mask_key = next((k for k in text_input_keys if 'mask' in k.lower()), None)

    if not input_ids_key or not attention_mask_key:
        print(f"Warning: Could not reliably determine input_ids/attention_mask keys from {text_input_keys}. Assuming '{text_input_keys[0]}' and '{text_input_keys[1]}'.")
        # Fallback if specific names aren't found
        if len(text_input_keys) >= 2:
            input_ids_key = text_input_keys[0]
            attention_mask_key = text_input_keys[1]
        else:
             raise ValueError(f"Cannot determine input keys for text model from: {text_input_keys}")


    for i, (input_id, attn_mask) in enumerate(zip(text_model_input_dict["input_ids"], text_model_input_dict["attention_mask"])):
        input_dict = {
            # Add batch dimension, ensure int32
            input_ids_key: input_id[None, :].astype(np.int32),
            attention_mask_key: attn_mask[None, :].astype(np.int32)
        }
        # Assuming text_model is CoreMLModel with `model.predict`
        output_dict = text_model.execute(input_dict) # Changed from model.model.predict
        text_sentence_embedding = output_dict[text_output_key][0] # Remove batch dim
        text_model_outputs.append(torch.from_numpy(text_sentence_embedding))


    # ===== 5. Postprocess (Calculate Cosine Similarity) =====
    print("Calculating cosine similarities...")


    # Convert lists of tensors to stacked tensors
    # image_stacked = torch.stack(image_model_outputs, dim=0)
    # text_stacked = torch.stack(text_model_outputs, dim=0)

    all_scores = []

    for text_feature in text_model_outputs:
        scores = []
        for image_feature in image_model_outputs:
            score = calculate_cosine_similarity(text_feature.cpu().detach().numpy(), image_feature.cpu().detach().numpy())
            scores.append(score)
        all_scores.append(scores)

    all_scores = torch.tensor(all_scores)

    # Print results
    print("\n--- Cosine Similarity Results ---")
    for i, text in enumerate(text_list):
        scores = all_scores[i] # Scores for this text against all images
        max_img_idx = torch.argmax(scores)
        max_score = scores[max_img_idx]
        print(f"Text: \"{text}\"")
        image_filename = Path(image_list[max_img_idx].filename).name
        print(f"  Best matching image index: {max_img_idx} (File: {image_filename} Score: {max_score:.4f})")
        print("-" * 20)


if __name__ == "__main__":
    # --- Configuration ---
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_IMAGE_MODEL_PATH = SCRIPT_DIR / "CLIPViTL14ImageEmbedding_NeuralEngineOptimized.mlpackage"
    DEFAULT_TEXT_MODEL_PATH = SCRIPT_DIR / "SentenceTransformersMultilingualCLIPViTB32v1_NeuralEngineOptimized.mlpackage"

    image_model_path = Path(DEFAULT_IMAGE_MODEL_PATH)
    text_model_path = Path(DEFAULT_TEXT_MODEL_PATH)

    # --- Input Data ---
    images = [
        Image.open(SCRIPT_DIR / "assets" / "beach.jpg"),
        Image.open(SCRIPT_DIR / "assets" / "cat.jpg"),
        Image.open(SCRIPT_DIR / "assets" / "dog.jpg"),
    ]

    texts = [
        "A dog playing in the snow",
        "Eine Katze auf einem Sofa",  # German: A cat on a sofa
        "Una playa con palmeras y sol."  # Spanish: A beach with palm trees and sun.
    ]

    # --- Model Loading ---
    print(f"Loading models")
    # or OnnxModel.from_path or OpenVINOModel.from_path of TFLiteModel.from_path
    image_model = CoreMLModel.from_path(image_model_path, compute_units=ct.ComputeUnit.ALL)
    text_model = CoreMLModel.from_path(text_model_path, compute_units=ct.ComputeUnit.ALL)

    # --- Pipeline Execution ---

    print("\nRunning CLIP Cosine Similarity pipeline...")
    run_clip_cosine_similarity_pipeline(
        image_model=image_model,
        text_model=text_model,
        image_list=images,
        text_list=texts,
        max_seq_length=MAX_SEQ_LENGTH
    )
