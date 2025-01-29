# Prediction interface for Cog ⚙️
# https://cog.run/python


from gptsovits import GPTSoVITSPipeline, LANGS

import torch
from hashlib import sha256
import numpy as np
import torchaudio
from pathlib import Path
import nltk

import cog
from cog import BasePredictor

GPU = torch.cuda.is_available()

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        soVITS_ckpt = "/src/checkpoints/gptsovits/gsv-v2final-pretrained/s2G2333k.pth"
        gpt_ckpt    = "/src/checkpoints/gptsovits/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        cnhubert_path = "/src/checkpoints/gptsovits/chinese-hubert-base"
        bert_path   = "/src/checkpoints/gptsovits/chinese-roberta-wwm-ext-large"

        self.device = torch.device("cuda" if GPU else "cpu")

        self.pipeline = GPTSoVITSPipeline(
            soVITS_path=soVITS_ckpt,
            gpt_path=gpt_ckpt,
            cnhubert_base_path=cnhubert_path,
            bert_path=bert_path,
            device=self.device,
            is_half=True
        )

        nltk.download('averaged_perceptron_tagger_eng')

    def predict(
        self,
        text: str = cog.Input(),
        language: str = cog.Input(
            choices=LANGS
        ),
        speaker_reference: cog.Path = cog.Input(),
        text_reference: str = cog.Input(),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # run prediction
        audio_int16, sr = self.pipeline.infer(
            ref_wav_path=str(speaker_reference),
            prompt_text=text_reference,
            prompt_language=language,
            text=text,
            text_language=language,
            how_to_cut="none",
            top_k=15,
            top_p=1.0,
            temperature=1.0,
            ref_free=False,
            speed=1.0,
            references=None
        )
        audio_float = torch.from_numpy(audio_int16).float()
        audio_float = audio_float / 32768.0
        audio_float = audio_float.unsqueeze(0)

        # save audio
        audio_path = f"{output_dir}/audio.wav"
        torchaudio.save(audio_path, audio_float, sr)

        return cog.Path(audio_path)
