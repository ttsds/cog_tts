import torch
import nltk
from pathlib import Path
from hashlib import sha256
import numpy as np
import os

from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice import se_extractor

from cog import BasePredictor
import cog

GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        nltk.download("averaged_perceptron_tagger_eng")
        device = "cuda" if GPU else "cpu"

        self.base_speaker_en = BaseSpeakerTTS(
            "/src/checkpoints/openvoice/checkpoints/base_speakers/EN/config.json",
            device=device,
        )
        self.base_speaker_en.load_ckpt(
            "/src/checkpoints/openvoice/checkpoints/base_speakers/EN/checkpoint.pth"
        )
        self.source_se_en = torch.load(
            "/src/checkpoints/openvoice/checkpoints/base_speakers/EN/en_default_se.pth",
            map_location=device,
        )

        self.base_speaker_zh = BaseSpeakerTTS(
            "/src/checkpoints/openvoice/checkpoints/base_speakers/ZH/config.json",
            device=device,
        )
        self.base_speaker_zh.load_ckpt(
            "/src/checkpoints/openvoice/checkpoints/base_speakers/ZH/checkpoint.pth"
        )
        self.source_se_zh = torch.load(
            "/src/checkpoints/openvoice/checkpoints/base_speakers/ZH/zh_default_se.pth",
            map_location=device,
        )

        self.tone_color_converter = ToneColorConverter(
            "/src/checkpoints/openvoice/checkpoints/converter/config.json",
            device=device,
        )
        self.tone_color_converter.load_ckpt(
            "/src/checkpoints/openvoice/checkpoints/converter/checkpoint.pth"
        )

    def predict(
        self,
        language: str = cog.Input(
            choices=[
                "en",
                "zh",
            ]
        ),
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        target_se, _ = se_extractor.get_se(
            str(speaker_reference),
            self.tone_color_converter,
            target_dir=str(output_dir),
            vad=True,
        )
        # Generate base speech
        src_path = os.path.join(output_dir, "src.wav")
        if language == "en":
            self.base_speaker_en.tts(
                text, src_path, speaker="default", language="English", speed=1.0
            )
            source_se = self.source_se_en
        elif language == "zh":
            self.base_speaker_zh.tts(
                text, src_path, speaker="default", language="Chinese", speed=1.0
            )
            source_se = self.source_se_zh
        # Run tone color converter
        output_path = os.path.join(output_dir, "output.wav")
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@MyShell",
        )

        return cog.Path(output_path)
