from pathlib import Path
from subprocess import run
import shutil
import argparse
import os
import sys
from hashlib import sha256

import cog
from cog import BasePredictor

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

from models.tts.maskgct.maskgct_utils import *
from utils.util import load_config

GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        device = "cuda" if GPU else "cpu"
        device = torch.device(device)
        self.device = device

        # ----------------
        # maskgct
        # ----------------
        cfg = load_config("/Amphion/models/tts/maskgct/config/maskgct.json")
        semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
        # 2. build semantic codec
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
        # 3. build acoustic codec
        codec_encoder, codec_decoder = build_acoustic_codec(
            cfg.model.acoustic_codec, device
        )
        # 4. build t2s model
        t2s_model = build_t2s_model(cfg.model.t2s_model, device)
        # 5. build s2a model
        s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)
        # load semantic codec
        base = "/src/checkpoints/_maskgct/"

        if GPU:
            safetensors.torch.load_model(
                semantic_codec, base + "semantic_codec/model.safetensors"
            )
            safetensors.torch.load_model(
                codec_encoder, base + "acoustic_codec/model.safetensors"
            )
            safetensors.torch.load_model(
                codec_decoder, base + "acoustic_codec/model_1.safetensors"
            )
            safetensors.torch.load_model(
                t2s_model, base + "t2s_model/model.safetensors"
            )
            safetensors.torch.load_model(
                s2a_model_1layer, base + "s2a_model/s2a_model_1layer/model.safetensors"
            )
            safetensors.torch.load_model(
                s2a_model_full, base + "s2a_model/s2a_model_full/model.safetensors"
            )
        else:
            # using load_checkpoint_and_dispatch
            load_checkpoint_and_dispatch(
                semantic_codec,
                base + "semantic_codec/model.safetensors",
                device_map={"": "cpu"},
            )
            load_checkpoint_and_dispatch(
                codec_encoder,
                base + "acoustic_codec/model.safetensors",
                device_map={"": "cpu"},
            )
            load_checkpoint_and_dispatch(
                codec_decoder,
                base + "acoustic_codec/model_1.safetensors",
                device_map={"": "cpu"},
            )
            load_checkpoint_and_dispatch(
                t2s_model, base + "t2s_model/model.safetensors", device_map={"": "cpu"}
            )
            load_checkpoint_and_dispatch(
                s2a_model_1layer,
                base + "s2a_model/s2a_model_1layer/model.safetensors",
                device_map={"": "cpu"},
            )
            load_checkpoint_and_dispatch(
                s2a_model_full,
                base + "s2a_model/s2a_model_full/model.safetensors",
                device_map={"": "cpu"},
            )
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.codec_encoder = codec_encoder
        self.codec_decoder = codec_decoder
        self.t2s_model = t2s_model
        self.s2a_model_1layer = s2a_model_1layer
        self.s2a_model_full = s2a_model_full
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std

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
        text_reference: str = cog.Input(),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        # random output dir
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        prompt_text = text_reference

        maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
            self.semantic_model,
            self.semantic_codec,
            self.codec_encoder,
            self.codec_decoder,
            self.t2s_model,
            self.s2a_model_1layer,
            self.s2a_model_full,
            self.semantic_mean,
            self.semantic_std,
            self.device,
        )

        recovered_audio = maskgct_inference_pipeline.maskgct_inference(
            str(speaker_reference),
            prompt_text,
            text,
            language,
            language,
            target_len=None,
        )

        recovered_audio = torch.tensor(recovered_audio)
        torchaudio.save(output_dir + "/test_pred.wav", recovered_audio.unsqueeze(0), 24000)

        return cog.Path(output_dir + "/test_pred.wav")
