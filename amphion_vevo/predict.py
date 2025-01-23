from pathlib import Path
from subprocess import run
import shutil
import argparse
import os
import sys
from hashlib import sha256

import cog
from cog import BasePredictor

from transformers import pipeline
import torch
import torchaudio
import numpy as np
import safetensors
from accelerate import load_checkpoint_and_dispatch

# ----------------
# Amphion
# ----------------
os.chdir("/Amphion")
os.environ.update({"WORK_DIR": "/Amphion"})
sys.path.append(".")

from models.vc.vevo.vevo_utils import *

GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        device = "cuda" if GPU else "cpu"
        device = torch.device(device)
        self.device = device

        # ----------------
        # hubert
        # ----------------
        Path("/root/.cache/torch/hub/checkpoints/").mkdir(parents=True, exist_ok=True)
        shutil.copy(
            "/src/checkpoints/hubert_fairseq_large_ll60k.pth",
            "/root/.cache/torch/hub/checkpoints/hubert_fairseq_large_ll60k.pth",
        )

        # ----------------
        # vevo
        # ----------------
        content_style_tokenizer_ckpt_path = "/src/checkpoints/vevo/tokenizer/vq8192"
        ar_cfg_path = "/Amphion/models/vc/vevo/config/PhoneToVq8192.json"
        ar_ckpt_path = "/src/checkpoints/vevo/contentstyle_modeling/PhoneToVq8192"
        fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
        fmt_ckpt_path = "/src/checkpoints/vevo/acoustic_modeling/Vq8192ToMels"
        vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
        vocoder_ckpt_path = "/src/checkpoints/vevo/acoustic_modeling/Vocoder"
        self.vevo = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
            ar_cfg_path=ar_cfg_path,
            ar_ckpt_path=ar_ckpt_path,
            fmt_cfg_path=fmt_cfg_path,
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=vocoder_cfg_path,
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=device,
        )

        # ----------------
        # whisper
        # ----------------
        if GPU:
            self.whisper = pipeline(
                "automatic-speech-recognition",
                model="/src/checkpoints/_whisper-large-v3-turbo",
                torch_dtype=torch.float16,
                device="cuda",
            )
        else:
            self.whisper = pipeline(
                "automatic-speech-recognition",
                model="/src/checkpoints/_whisper-large-v3-turbo",
            )

    def predict(
        self,
        language: str = cog.Input(
            choices=[
                "en",
                "zh",
                "de",
                "fr",
                "ja",
                "ko",
            ]
        ),
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        # random output dir
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        prompt_text = self.whisper(str(speaker_reference))["text"]

        gen_audio = self.vevo.inference_ar_and_fm(
            src_wav_path=None,
            src_text=text,
            style_ref_wav_path=str(speaker_reference),
            timbre_ref_wav_path=str(speaker_reference),
            style_ref_wav_text=prompt_text,
            src_text_language=language,
            style_ref_wav_text_language=language,
        )

        output_path = output_dir + "/test_pred.wav"
        save_audio(gen_audio, output_path=output_path)

        return cog.Path(output_path)
