from pathlib import Path
from subprocess import run
import shutil
import argparse
import os
import sys
from hashlib import sha256

import cog
from cog import BasePredictor

from encodec import EncodecModel
import nltk
import torch
import numpy as np

# ----------------
# Amphion
# ----------------
os.chdir("/Amphion")
os.environ.update({"WORK_DIR": "/Amphion"})
sys.path.append(".")

from models.tts.naturalspeech2.ns2_inference import NS2Inference
from utils.util import load_config


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        # generic
        EncodecModel.encodec_model_24khz()
        nltk.download("averaged_perceptron_tagger_eng")
        device = "cuda" if GPU else "cpu"

        # ----------------
        # naturalspeech2
        # ----------------
        args = {
            "config": "/Amphion/egs/tts/NaturalSpeech2/exp_config.json",
            "checkpoint_path": "/src/checkpoints/naturalspeech2/checkpoint/epoch-0089_step-0512912_loss-6.367693",
            "dataset": None,
            "testing_set": "test",
            "test_list_file": None,
            "speaker_name": None,
            "text": "",
            "ref_audio": None,
            "vocoder_dir": None,
            "acoustics_dir": None,
            "mode": "single",
            "log_level": "warning",
            "pitch_control": 1.0,
            "energy_control": 1.0,
            "duration_control": 1.0,
            "inference_step": 200,
            "output_dir": None,
            "device": device,
        }
        ns_args = argparse.Namespace(**args)
        cfg = load_config(ns_args.config)
        self.naturalspeech2 = NS2Inference(ns_args, cfg)

        print(f"NS2 model params: {get_model_params(self.naturalspeech2.model)}")
        print(f"NS2 codec params: {get_model_params(self.naturalspeech2.codec)}")

    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        # random output dir
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # run prediction
        self.naturalspeech2.args.text = text
        self.naturalspeech2.args.ref_audio = speaker_reference
        self.naturalspeech2.args.output_dir = output_dir
        self.naturalspeech2.inference()
        result = next(Path(output_dir).rglob("*.wav"))
        return cog.Path(result)
