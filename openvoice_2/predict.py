import torch
import nltk
from pathlib import Path
from hashlib import sha256
import numpy as np
import os

from openvoice.api import ToneColorConverter
from openvoice import se_extractor
from melo.api import TTS

from cog import BasePredictor
import cog

GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        nltk.download("averaged_perceptron_tagger_eng")
        device = "cuda" if GPU else "cpu"

        self.models = {
            "en": TTS(
                language="EN",
                config_path="/src/checkpoints/openvoice_en/config.json",
                ckpt_path="/src/checkpoints/openvoice_en/checkpoint.pth",
            ),
            "zh": TTS(
                language="ZH",
                config_path="/src/checkpoints/openvoice_zh/config.json",
                ckpt_path="/src/checkpoints/openvoice_zh/checkpoint.pth",
            ),
            "es": TTS(
                language="ES",
                config_path="/src/checkpoints/openvoice_es/config.json",
                ckpt_path="/src/checkpoints/openvoice_es/checkpoint.pth",
            ),
            "ja": TTS(
                language="JP",
                config_path="/src/checkpoints/openvoice_ja/config.json",
                ckpt_path="/src/checkpoints/openvoice_ja/checkpoint.pth",
            ),
            "ko": TTS(
                language="KR",
                config_path="/src/checkpoints/openvoice_ko/config.json",
                ckpt_path="/src/checkpoints/openvoice_ko/checkpoint.pth",
            ),
            "fr": TTS(
                language="FR",
                config_path="/src/checkpoints/openvoice_fr/config.json",
                ckpt_path="/src/checkpoints/openvoice_fr/checkpoint.pth",
            ),
        }
        self.source_ses = {
            "en": torch.load(
                "/src/checkpoints/openvoice/base_speakers/ses/en-default.pth",
                map_location=device,
            ),
            "zh": torch.load(
                "/src/checkpoints/openvoice/base_speakers/ses/zh.pth",
                map_location=device,
            ),
            "es": torch.load(
                "/src/checkpoints/openvoice/base_speakers/ses/es.pth",
                map_location=device,
            ),
            "ja": torch.load(
                "/src/checkpoints/openvoice/base_speakers/ses/jp.pth",
                map_location=device,
            ),
            "ko": torch.load(
                "/src/checkpoints/openvoice/base_speakers/ses/kr.pth",
                map_location=device,
            ),
            "fr": torch.load(
                "/src/checkpoints/openvoice/base_speakers/ses/fr.pth",
                map_location=device
            ),
        }

        self.tone_color_converter = ToneColorConverter(
            "/src/checkpoints/openvoice/converter/config.json",
            device=device,
        )
        self.tone_color_converter.load_ckpt(
            "/src/checkpoints/openvoice/converter/checkpoint.pth"
        )

    def predict(
        self,
        language: str = cog.Input(
            choices=[
                "en",
                "zh",
                "es",
                "ja",
                "ko",
                "fr",
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
        model = self.models[language]
        src_path = os.path.join(output_dir, "src.wav")
        speaker_ids = model.hps.data.spk2id
        if language == "en":
            speaker_id = speaker_ids["EN-Default"]
        else:
            speaker_id = speaker_ids[language.upper()]
        model.tts_to_file(
            text,
            speaker_id,
            src_path,
            speed=1.0,
        )

        # Run tone color converter
        output_path = os.path.join(output_dir, "output.wav")
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=self.source_ses[language],
            tgt_se=target_se,
            output_path=output_path,
            message="@MyShell",
        )

        return cog.Path(output_path)
