from pathlib import Path
from hashlib import sha256
import numpy as np

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
)

import torch
import soundfile as sf

from cog import BasePredictor
import cog

GPU = torch.cuda.is_available()

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.vocoder = load_vocoder()
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        model_path = "/src/checkpoints/e2/E2TTS_Base/model_1200000.safetensors"
        self.device = torch.device("cuda" if GPU else "cpu")
        self.e2 = load_model(UNetT, model_cfg, model_path, device=self.device)


    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
        text_reference: str = cog.Input(),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(str(speaker_reference))
        ref_audio_waveform, ref_text_processed = preprocess_ref_audio_text(
            str(speaker_reference), text_reference, show_info=print
        )
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio_waveform, ref_text_processed, text, self.e2, show_info=print
        )
        sf.write(f"{output_dir}/generated.wav", final_wave, final_sample_rate)
        return cog.Path(f"{output_dir}/generated.wav")
