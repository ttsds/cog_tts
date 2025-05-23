from pathlib import Path
from hashlib import sha256

import torch
import torchaudio
import numpy as np

from hierspeechpp import HierspeechSynthesizer

from cog import BasePredictor
import cog

GPU = torch.cuda.is_available()


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        device = "cuda" if GPU else "cpu"

        self.hierspeechpp = HierspeechSynthesizer(
            hierspeech_ckpt="/src/checkpoints/hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth",
            config_hierspeech="/src/checkpoints/hierspeechpp_eng_kor/config.json",
            text2w2v_ckpt="/src/checkpoints/ttv_libritts_v1/ttv_lt960_ckpt.pth",
            config_text2w2v="/src/checkpoints/ttv_libritts_v1/config.json",
            device=device,
        )
        print(
            f"Model params: {get_model_params(self.hierspeechpp.net_g)+get_model_params(self.hierspeechpp.text2w2v)}"
        )

    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        out_wav, sr = self.hierspeechpp.infer(
            text=text,
            prompt_audio_path=str(speaker_reference),
            noise_scale_ttv=0.333,
            noise_scale_vc=0.333,
            denoise_ratio=0.0,
            scale_norm="prompt",
        )

        output_path = f"{output_dir}/output.wav"
        out_wav = torch.tensor(out_wav)
        torchaudio.save(output_path, out_wav, sr)

        return cog.Path(output_path)
