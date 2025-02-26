from pathlib import Path
from hashlib import sha256
import numpy as np
from tempfile import NamedTemporaryFile

from cog import BasePredictor
import cog

import torch
import torchaudio

from tortoise import api
from tortoise import utils


GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:        
        self.device = "cuda" if GPU else "cpu"
        
        self.tortoise = api.TextToSpeech(kv_cache=True, models_dir="/src/checkpoints/tortoise/.models")

    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(default=None),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # save speaker reference as temporary wav
        temp_wav = NamedTemporaryFile(delete=False, suffix=".wav")
        audio, sr = torchaudio.load(str(speaker_reference))
        # resample to 22050
        audio = torchaudio.transforms.Resample(sr, 22050)(
            audio
        )
        sr = 22050
        torchaudio.save(temp_wav.name, audio, sr)
        speaker_reference = temp_wav.name

        reference_clips = [utils.audio.load_audio(str(speaker_reference), 22050)]
        pcm_audio = self.tortoise.tts_with_preset(text, voice_samples=reference_clips, preset='fast')

        # Save the output audio to a file
        output_path = f"{output_dir}/output.wav"
        torchaudio.save(output_path, pcm_audio[0], 24000)
        
        return cog.Path(output_path)