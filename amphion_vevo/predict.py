from pathlib import Path
import shutil
import os
import sys
from hashlib import sha256

import cog
from cog import BasePredictor

import torch
import numpy as np

# ----------------
# Amphion
# ----------------
os.chdir("/Amphion")
os.environ.update({"WORK_DIR": "/Amphion"})
sys.path.append(".")

from models.vc.vevo.vevo_utils import *

GPU = torch.cuda.is_available()


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


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
            "/src/checkpoints/hubert/hubert_fairseq_large_ll60k.pth",
            "/root/.cache/torch/hub/checkpoints/hubert_fairseq_large_ll60k.pth",
        )

        # ----------------
        # vevo
        # ----------------
        content_style_tokenizer_ckpt_path = "/src/checkpoints/_vevo/tokenizer/vq8192"
        ar_cfg_path = "/Amphion/models/vc/vevo/config/PhoneToVq8192.json"
        ar_ckpt_path = "/src/checkpoints/_vevo/contentstyle_modeling/PhoneToVq8192"
        fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
        fmt_ckpt_path = "/src/checkpoints/_vevo/acoustic_modeling/Vq8192ToMels"
        vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
        vocoder_ckpt_path = "/src/checkpoints/_vevo/acoustic_modeling/Vocoder"
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

        print(f"AR model params: {get_model_params(self.vevo.ar_model)}")
        print(f"FM model params: {get_model_params(self.vevo.fmt_model)}")
        print(f"Vocoder model params: {get_model_params(self.vevo.vocoder_model)}")
        print(
            f"Total params: {get_model_params(self.vevo.ar_model) + get_model_params(self.vevo.fmt_model) + get_model_params(self.vevo.vocoder_model)}"
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
        text_reference: str = cog.Input(),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        # random output dir
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        gen_audio = self.vevo.inference_ar_and_fm(
            src_wav_path=None,
            src_text=text,
            style_ref_wav_path=str(speaker_reference),
            timbre_ref_wav_path=str(speaker_reference),
            style_ref_wav_text=text_reference,
            src_text_language=language,
            style_ref_wav_text_language=language,
        )

        output_path = output_dir + "/test_pred.wav"
        save_audio(gen_audio, output_path=output_path)

        return cog.Path(output_path)
