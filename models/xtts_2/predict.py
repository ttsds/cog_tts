import os
import random
import shutil
from pathlib import Path
from hashlib import sha256
import sys
import numpy as np
import torch
import torchaudio
import argparse
from cog import BasePredictor
import cog
import torch

from TTS.api import TTS

GPU = torch.cuda.is_available()


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class Predictor(BasePredictor):
    def setup(self) -> None:

        self.tts = TTS(
            model_path="/src/checkpoints/xtts",
            config_path="/src/checkpoints/xtts/config.json",
            gpu=GPU,
        )

        print(f"Model params: {get_model_params(self.tts.synthesizer)}")

    def predict(
        self,
        text: str = cog.Input(description="Text to synthesize"),
        speaker_reference: cog.Path = cog.Input(description="Reference audio file"),
        language: str = cog.Input(
            description="Language of the text",
            default="en",
            #  English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi)
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh",
                "ja",
                "hu",
                "ko",
                "hi",
            ],
        ),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.tts.tts_to_file(
            text=text,
            file_path=output_dir + "/output.wav",
            speaker_wav=speaker_reference,
            language=language,
        )

        output_wav, _ = torchaudio.load(output_dir + "/output.wav")
        torchaudio.save(output_dir + "/output.wav", output_wav, 24_000)

        return cog.Path(output_dir + "/output.wav")
