import time
import traceback
from pathlib import Path
import tempfile
import shutil
from typing import List, Any, Tuple

import PIL
import numpy as np
import torch
import coremltools as ct
from PIL import Image
import requests
from runlocal_tools.backend.coreml.coreml_model import CoreMLModel
from transformers import AutoProcessor

# Define constants
MAX_INPUT_ID = -100 # Based on original phi3 code inspection
DEVICE = "cpu" # Core ML runs on its own compute units, torch tensors are mainly for pre/post


def create_4d_causal_mask(seq_len: int, total_kv_len: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    causal_mask = torch.full((seq_len, total_kv_len), dtype=dtype, fill_value=torch.finfo(torch.float32).min,
                             device=device)  # Start with large negative
    row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
    col_indices = torch.arange(total_kv_len, device=device).unsqueeze(0)
    past_length = total_kv_len - seq_len
    mask_condition = col_indices > (row_indices + past_length)
    causal_mask = causal_mask.masked_fill(mask_condition, torch.finfo(dtype).min)  # Fill future positions with large negative
    causal_mask = causal_mask.masked_fill(~mask_condition, 0.0)  # Fill allowed positions with 0
    expanded_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return expanded_mask.to(dtype=dtype)  # Ensure final dtype


def torch_to_numpy_dict(tensors: list[torch.Tensor], names: list[str]) -> dict[str, np.ndarray]:
    """Converts a list of torch tensors to a dictionary of numpy arrays."""
    return {name: tensor.cpu().numpy().astype(np.float32) for name, tensor in zip(names, tensors)}

def numpy_dict_to_torch_list(np_dict: dict[str, np.ndarray], names: list[str], device: str) -> list[torch.Tensor]:
    """Converts a dictionary of numpy arrays to a list of torch tensors."""
    return [torch.from_numpy(np_dict[name]).to(device) for name in names]


# --- Main Inference Logic ---
def run_inference(
    prompt_str: str,
    images: list[Image.Image],
    processor: AutoProcessor,
    image_embedding_model: CoreMLModel,
    text_embedding_model: CoreMLModel,
    text_decoder_model: CoreMLModel,
    max_new_tokens: int = 500,
):
    """
    Runs the Phi-3.5 Vision inference using Core ML models.
    """

    # --- 1. Preprocessing ---
    print("Preprocessing inputs...")
    placeholder = "".join([f"<|image_{i+1}|>" for i in range(len(images))])
    messages = [{"role": "user", "content": placeholder + prompt_str}]
    prompt_templated = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt_templated, images, return_tensors="pt")
    input_ids = inputs['input_ids'].to(DEVICE)
    pixel_values = inputs.get('pixel_values') # List of tensors

    # --- 2. Embeddings ---
    all_image_features_proj = []
    if pixel_values is not None:
        print("Running image embedding model...")
        image_emb_input_names = ["pixel_values"]
        image_emb_output_names = ["output"]
        for i, pixel_value_tensor in enumerate(pixel_values):
            coreml_input = torch_to_numpy_dict([pixel_value_tensor.unsqueeze(0)], image_emb_input_names) # Add batch dim
            coreml_output = image_embedding_model.execute(coreml_input)
            image_features_proj_list = numpy_dict_to_torch_list(coreml_output, image_emb_output_names, DEVICE)
            all_image_features_proj.append(image_features_proj_list[0])
        image_features_proj = torch.cat(all_image_features_proj, dim=0) # Cat along batch dim created earlier

    # Text Embedding
    print("Running text embedding model...")
    text_emb_input_names = ["input_ids"]
    text_emb_output_names = ["output_embeddings"]
    coreml_input = torch_to_numpy_dict([input_ids], text_emb_input_names)
    coreml_output = text_embedding_model.execute(coreml_input)
    hidden_states_list = numpy_dict_to_torch_list(coreml_output, text_emb_output_names, DEVICE)
    hidden_states = hidden_states_list[0]

    # Combine Embeddings (replace placeholders)
    if images:
        positions = torch.nonzero((input_ids < 0) & (input_ids > MAX_INPUT_ID), as_tuple=True)
        batch_indices = positions[0]
        sequence_indices = positions[1]
        hidden_states = hidden_states.index_put(
            (batch_indices, sequence_indices), # Use the tuple directly
            image_features_proj,
            accumulate=False
        )
        print("Successfully combined text and image embeddings.")


    prompt_length = hidden_states.shape[1]
    print(f"Combined hidden states shape: {hidden_states.shape}, Prompt length: {prompt_length}")

    # --- 3. Greedy Generation Loop ---
    print("Starting token generation loop...")
    generated_token_ids: List[int] = []
    eos_token_id = processor.tokenizer.eos_token_id
    eos_token_id_list = [eos_token_id] if not isinstance(eos_token_id, list) else eos_token_id
    print(f"EOS token ID(s): {eos_token_id_list}")

    next_hidden_states = hidden_states # The first input to the decoder

    # Get decoder input/output names (MLModel provides these)
    decoder_input_names = ["hidden_states", "causal_mask"]
    decoder_output_names = ["logits"]

    generation_start_time = time.time()
    tokens_generated = 0
    for i in range(max_new_tokens):
        current_total_seq_length = prompt_length + len(generated_token_ids)
        seq_length = next_hidden_states.shape[1] # Should be 1 for subsequent steps

        # Create causal mask for the current step
        causal_mask = create_4d_causal_mask(
            seq_length, current_total_seq_length, dtype=torch.float32
        ).to(DEVICE)

        # Prepare Core ML input
        decoder_inputs_list = [next_hidden_states, causal_mask]
        coreml_decoder_input = torch_to_numpy_dict(decoder_inputs_list, decoder_input_names)

        # Run Decoder Model (using MLModel's predict with state)
        coreml_decoder_output = text_decoder_model.execute(coreml_decoder_input)

        # Process output logits
        next_token_logits_list = numpy_dict_to_torch_list(coreml_decoder_output, decoder_output_names, DEVICE)
        next_token_logits = next_token_logits_list[0] # Shape: (batch_size, seq_length_step, vocab_size)

        # Greedy sampling: get the most likely token ID from the *last* position's logits
        current_step_output_token_id = torch.argmax(next_token_logits[:, -1, :], dim=-1).unsqueeze(-1) # Keep batch dim
        token_id = current_step_output_token_id.item()
        generated_token_ids.append(token_id)
        tokens_generated += 1

        # Decode and print the current token (optional, for debugging)
        token_str = processor.tokenizer.decode([token_id])
        print(token_str, end=" ", flush=True) # Print token without newline, flush output

        # Check for EOS token
        if token_id in eos_token_id_list:
            print(f"EOS token ({token_id}) generated at step {i+1}, stopping.")
            break

        # Prepare for the *next* iteration
        # Get the embedding for the token we just generated
        next_input_ids = current_step_output_token_id.to(DEVICE)
        coreml_text_emb_input = torch_to_numpy_dict([next_input_ids], text_emb_input_names)
        coreml_text_emb_output = text_embedding_model.execute(coreml_text_emb_input)
        next_hidden_states_list = numpy_dict_to_torch_list(coreml_text_emb_output, text_emb_output_names, DEVICE)
        next_hidden_states = next_hidden_states_list[0] # Shape: (batch_size, 1, hidden_dim)

    else: # Loop finished without hitting EOS
        print(f"Stopped generation after reaching max_new_tokens ({max_new_tokens}).")

    generation_elapsed = time.time() - generation_start_time
    tokens_per_second = tokens_generated / generation_elapsed if generation_elapsed > 0 else float('inf')
    print(f"Generation finished: {tokens_generated} tokens in {generation_elapsed:.2f} seconds ({tokens_per_second:.2f} tokens/sec).")

    # --- 4. Postprocessing ---
    print("Decoding generated tokens...")
    response = processor.tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print("Inference complete.")
    return {"response": response}


# --- Example Usage ---
if __name__ == '__main__':



    # --- Configuration ---
    IMAGE_EMBEDDING_MODEL_PATH = Path(__file__).parent / "Phi3.5VInstruct_ImageEmbedding.mlpackage"
    TEXT_EMBEDDING_MODEL_PATH = Path(__file__).parent / "Phi3.5VTextEmbedding.mlpackage"
    TEXT_DECODER_MODEL_PATH = Path(__file__).parent / "Phi3.5VInstruct_StatefulTextDecoder_4096.mlpackage"

    HUGGINGFACE_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
    COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_GPU

    try:
        # --- Load Processor ---
        print(f"Loading processor for {HUGGINGFACE_MODEL_ID}...")
        processor = AutoProcessor.from_pretrained(HUGGINGFACE_MODEL_ID, trust_remote_code=True)

        # --- Load Core ML Models ---
        print(f"Loading and compiling CoreML models")
        image_embedding_model = CoreMLModel.from_path(IMAGE_EMBEDDING_MODEL_PATH, compute_units=COMPUTE_UNITS)
        text_embedding_model = CoreMLModel.from_path(TEXT_EMBEDDING_MODEL_PATH, compute_units=COMPUTE_UNITS)
        # set should_compile=True due to iterative inference
        # set make_state=True to use stateful decoder, state is stored inside CoreMLModel
        text_decoder_model = CoreMLModel.from_path(TEXT_DECODER_MODEL_PATH, compute_units=COMPUTE_UNITS, should_compile=True, make_state=True)

        # --- Prepare Input Data ---

        ASSETS_DIR = Path(__file__).parent / "assets"
        example_images = [
            PIL.Image.open(ASSETS_DIR / "AzureCloud_0.jpg").convert("RGB"),
            PIL.Image.open(ASSETS_DIR / "AzureCloud_1.jpg").convert("RGB"),
        ]

        if not example_images:
             print("Error: No images were successfully loaded. Exiting.")
             exit()

        example_prompt = "Summarize the deck of slides."
        print(f"Using prompt: {example_prompt}")

        # --- Run Inference ---
        print("\nRunning inference...")
        inference_result = run_inference(
            prompt_str=example_prompt,
            images=example_images,
            processor=processor,
            image_embedding_model=image_embedding_model,
            text_embedding_model=text_embedding_model,
            text_decoder_model=text_decoder_model,
            max_new_tokens=200, # Limit tokens for example
        )

        # --- Display Result ---
        print("\n--- Inference Output ---")
        print(inference_result.get("response", "No response generated."))
        print("-----------------------")

    except Exception as e:
        print("\n--- An Error Occurred ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print("-------------------------")

    finally:
        # --- Cleanup ---
        print("Cleaning up temporary directory...")
        text_decoder_model.destroy()
