import json
from pathlib import Path
import coremltools as ct
import torch
from huggingface_hub import hf_hub_download
from typing import Generator, Tuple, Optional, Union, Callable, List, Dict, Any
import re
import os
import numpy as np
from runlocal_tools.backend.coreml.coreml_model import CoreMLModel
from runlocal_tools.backend.ml_model import Model
from misaki import en, espeak
import soundfile as sf
# --- End G2P imports ---

# Define constants or helper functions if needed, replicating from KPipeline if necessary and allowed
# Example: LANG_CODES might be useful for context, but G2P is the main issue
ALIASES = {
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j', # Note: Requires pip install misaki[ja]
    'zh': 'z', # Note: Requires pip install misaki[zh]
}

LANG_CODES = dict(
    a='American English', b='British English', e='es', f='fr-fr',
    h='hi', i='it', p='pt-br', j='Japanese', z='Mandarin Chinese',
)

# --- Constants for Fixed Length Processing ---
TARGET_LEN = 512
HOP_LENGTH = 600
MAX_FRAMES = 1200

class KokoroTTSPipeline:
    """
    Simplified pipeline for Kokoro TTS, exposing separate duration
    prediction and waveform generation steps.
    """
    def __init__(
        self,
        duration_prediction_model: Model,
        waveform_generation_model: Model,
        repo_id: str = 'hexgrad/Kokoro-82M',
        lang_code: Optional[str] = None,
        g2p_trf: bool = False,

    ):
        self.repo_id = repo_id
        self.voices = {} # Cache for full voice tensors [510, 1, 256]
        self.lang_code = lang_code.lower() if lang_code else None
        if self.lang_code:
            self.lang_code = ALIASES.get(self.lang_code, self.lang_code)
            # No warning for unknown lang code anymore

        self.device = 'cpu'
        config = hf_hub_download(repo_id=repo_id, filename='config.json')
        with open(config, 'r', encoding='utf-8') as r:
            config = json.load(r)
        self.vocab = config['vocab']

        self.duration_prediction_model = duration_prediction_model
        self.waveform_generation_model = waveform_generation_model

        # --- Initialize G2P ---
        self.g2p = None
        if self.lang_code in 'ab': # English
             # Simplified init without try/except per previous request
             fallback = espeak.EspeakFallback(british=self.lang_code == 'b')
             self.g2p = en.G2P(trf=g2p_trf, british=self.lang_code == 'b', fallback=fallback, unk='')
        elif self.lang_code in LANG_CODES: # Other languages supported by espeak
            language_name = LANG_CODES[self.lang_code]
            self.g2p = espeak.EspeakG2P(language=language_name)

    def _load_voice(self, voice: str) -> torch.Tensor:
        """Loads the *full* voice tensor [510, 1, 256], downloading if necessary."""
        if voice in self.voices:
            return self.voices[voice]

        f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')
        pack = torch.load(f, map_location='cpu', weights_only=True)
        if not isinstance(pack, torch.Tensor) or pack.shape != torch.Size([510, 1, 256]):
             raise TypeError(f"Loaded voice file '{f}' did not contain a valid tensor of shape [510, 1, 256]. Shape was: {pack.shape if isinstance(pack, torch.Tensor) else type(pack)}")
        self.voices[voice] = pack
        return pack

    @staticmethod
    def _tokens_to_ps(tokens: List[en.MToken]) -> str:
        """Helper to convert Misaki English tokens to a phoneme string."""
        return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens if t.phonemes).strip()

    def _phonemize(self, text: str) -> Optional[str]:
        """
        Convert text to phonemes using the initialized G2P tool.
        Returns phoneme string, or None if G2P not available/fails implicitly.
        (Removed try/except)
        """
        if self.g2p is None:
            return None # G2P tool not initialized or language not supported

        if isinstance(self.g2p, en.G2P):
            _processed_text, tokens = self.g2p(text)
            if not tokens:
                return None # G2P returned no tokens
            phonemes = self._tokens_to_ps(tokens)
            return phonemes
        elif isinstance(self.g2p, espeak.EspeakG2P):
            phonemes, _ = self.g2p(text)
            return phonemes
        return None

    def _prepare_fixed_input(self, phonemes: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Converts phoneme string to padded input_ids, mask, and core phoneme length.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor, int]: input_ids [1, 512], mask [1, 512], core_phoneme_length
            or None if phonemes are empty.
        """
        if not phonemes: return None

        input_ids_list = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        if not input_ids_list: return None

        # Truncate core phonemes if necessary
        max_content_len = TARGET_LEN - 2
        core_phoneme_length = len(input_ids_list) # Length *before* truncation/start/end
        if core_phoneme_length > max_content_len:
            input_ids_list = input_ids_list[:max_content_len]
            core_phoneme_length = max_content_len # Update length *after* truncation

        # Add start and end tokens [0]
        processed_ids = [0] + input_ids_list + [0]
        current_len_with_start_end = len(processed_ids)

        # Pad with [0]
        padding_needed = TARGET_LEN - current_len_with_start_end
        if padding_needed < 0: raise ValueError(f"Internal Error: Length {current_len_with_start_end} > {TARGET_LEN}")

        final_ids = processed_ids + ([0] * padding_needed)

        # Create mask (1 for real tokens including start/end, 0 for padding)
        mask_list = ([1] * current_len_with_start_end) + ([0] * padding_needed)

        # Convert to tensors
        input_ids = torch.LongTensor([final_ids]).to(self.device) # Shape: [1, TARGET_LEN]
        mask = torch.LongTensor([mask_list]).to(self.device)      # Shape: [1, TARGET_LEN]

        # Return tensors and the length of the core phoneme sequence (post-truncation)
        return input_ids, mask, core_phoneme_length

    def predict_duration(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        split_pattern: Optional[str] = None, # r'[\n.!?]+',
    ) -> List[Dict[str, Any]]:
        """
        Predicts durations for text segments using fixed-length processing.

        Returns:
            List[Dict[str, Any]]: List of dictionaries, one per valid segment.
                Keys: 'segment_text', 'phonemes', 'input_ids' [1, 512], 'mask' [1, 512],
                      'selected_ref_s' [1, 256], 'speed', 'pred_dur' [512], 'encoded_dur'.
        """
        results = []
        full_voice_tensor = self._load_voice(voice) # Load full voice tensor once [510, 1, 256]

        if split_pattern and isinstance(text, str):
            segments = re.split(split_pattern, text.strip())
            segments = [s.strip() for s in segments if s and s.strip()]
        elif isinstance(text, str):
            segments = [text.strip()]
        else:
            segments = [str(text).strip()]

        if not segments: return []

        for segment_text in segments:
            if not segment_text: continue

            phonemes = self._phonemize(segment_text)
            if not phonemes: continue

            # Prepare fixed input AND get core phoneme length (post-truncation)
            prepared_input = self._prepare_fixed_input(phonemes)
            if not prepared_input: continue
            input_ids, mask, core_phoneme_length = prepared_input

            # --- Select the specific style vector based on core phoneme length ---
            # Indexing like the original KPipeline.infer: pack[len(ps)-1]
            # Use max(0, ...) to handle potential zero length edge case safely
            style_index = max(0, core_phoneme_length - 1)
            # Ensure index is within bounds (0 to 509)
            style_index = min(style_index, 509)
            selected_ref_s = full_voice_tensor[style_index] # Shape should be [1, 256]

            # Check the shape after selection
            if selected_ref_s.shape != torch.Size([1, 256]):
                raise RuntimeError(f"Internal Error: Selected ref_s shape is {selected_ref_s.shape}, expected [1, 256]. Index was {style_index}.")

            # Predict duration using the *selected* style tensor
            speed_tensor = torch.tensor([speed], device=self.device) # KModel expects tensor

            output = self.duration_prediction_model.execute({
                'input_ids': input_ids.numpy().astype(np.int32),
                'mask': mask.numpy().astype(np.int32),
                'style': selected_ref_s.numpy().astype(np.float32),  # Pass the selected [1, 256] tensor
                'speed': speed_tensor.numpy().astype(np.float32)
            })
            pred_dur = torch.from_numpy(output['pred_dur'])
            encoded_dur = torch.from_numpy(output['encoded_dur'])

            results.append({
                'segment_text': segment_text,
                'phonemes': phonemes,
                'input_ids': input_ids,
                'mask': mask,
                'selected_ref_s': selected_ref_s, # Store the selected style
                'speed': speed,
                'pred_dur': pred_dur,
                'encoded_dur': encoded_dur
            })

        return results

    def generate_waveform_for_segment(
        self,
        segment_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Generates waveform for a single segment using pre-calculated durations.
        Assumes fixed-length KModel logic.
        """
        input_ids = segment_data['input_ids']
        mask = segment_data['mask']
        selected_ref_s = segment_data['selected_ref_s'] # The [1, 256] tensor
        pred_dur = segment_data['pred_dur']
        encoded_dur = segment_data['encoded_dur']

        # 1. Mask durations based on input mask
        mask_long = mask.squeeze(0).long().to(pred_dur.device)
        masked_pred_dur = pred_dur * mask_long

        # 2. Calculate expanded indices for alignment
        indices_arange = torch.arange(TARGET_LEN, device=self.device)
        expanded_indices = torch.repeat_interleave(indices_arange, masked_pred_dur)

        # 3. Calculate valid audio length based on *masked* durations
        num_valid_frames = expanded_indices.shape[0]
        valid_audio_len = num_valid_frames * HOP_LENGTH

        # 4. Clamp and Pad indices for waveform generation input (up to MAX_FRAMES)
        if expanded_indices.shape[0] > MAX_FRAMES:
            expanded_indices = expanded_indices[:MAX_FRAMES]

        padded_indices = torch.zeros(MAX_FRAMES, device=self.device, dtype=expanded_indices.dtype)
        current_indices_len = expanded_indices.shape[0]
        if current_indices_len > 0:
            padded_indices[:current_indices_len] = expanded_indices

        # --- Call KModel waveform generation ---
        output = self.waveform_generation_model.execute({
            'input_ids':input_ids.numpy().astype(np.int32),
            'mask':mask.numpy().astype(np.int32),
            'style':selected_ref_s.numpy().astype(np.float32),
            'pred_dur':masked_pred_dur.numpy().astype(np.int32),
            'encoded_dur':encoded_dur.numpy().astype(np.float32),
            'expanded_indices':padded_indices.numpy().astype(np.int32),
        })
        waveform_raw = torch.from_numpy(output['waveform'])

        # --- Truncate waveform to valid length ---
        if waveform_raw.shape[0] >= valid_audio_len:
             final_waveform = waveform_raw[:valid_audio_len]
        else:
             # Waveform is unexpectedly short
             final_waveform = waveform_raw

        return final_waveform.detach().cpu().numpy()


    def run(self, text: str):

        # 1. Predict durations
        print(f"Predicting durations for voice '{voice}'...")
        duration_results = pipeline.predict_duration(text=text, voice=voice, speed=speed_setting)

        if not duration_results:
            print("Duration prediction failed or produced no results.")
        else:
            print(f"Duration prediction complete for {len(duration_results)} segment(s).")

            # 2. Generate waveform for each segment
            total_saved = 0
            for i, segment_data in enumerate(duration_results):
                waveform_np = pipeline.generate_waveform_for_segment(segment_data)
                if waveform_np is not None and waveform_np.size > 0:
                    output_filename = os.path.join(output_dir, f'segment_{total_saved}.wav')
                    sample_rate = 24000
                    sf.write(output_filename, waveform_np, sample_rate)
                    print(f"  Saved waveform ({waveform_np.shape}, sr={sample_rate}) to {output_filename}")
                    total_saved += 1
                else:
                    print(f"  Failed to generate waveform for segment {i+1}.")

            print(f"Finished. Saved {total_saved} waveform(s) to '{output_dir}/'.")

if __name__ == "__main__":
    duration_prediction_model = CoreMLModel.from_path(Path(__file__).parent / "Kokoro_DurationPrediction.mlpackage", should_compile=True, compute_units=ct.ComputeUnit.ALL)
    waveform_generation_model = CoreMLModel.from_path(Path(__file__).parent / "Kokoro_GenerateWaveform.mlpackage", should_compile=True, compute_units=ct.ComputeUnit.CPU_ONLY)

    # --- Configuration ---
    lang = 'a'
    voice = 'af_heart'
    output_dir = "."
    text = """
RunLocal helps engineering teams discover, optimize, evaluate and deploy the best on-device AI model for their use case.
"""
    os.makedirs(output_dir, exist_ok=True)
    speed_setting = 1.0

    # Initialize pipeline
    pipeline = KokoroTTSPipeline(
        duration_prediction_model=duration_prediction_model,
        waveform_generation_model=waveform_generation_model,
        lang_code=lang
    )

    pipeline.run(text)

    # clear temp dir (if should_compile=True)
    duration_prediction_model.destroy()
    waveform_generation_model.destroy()


