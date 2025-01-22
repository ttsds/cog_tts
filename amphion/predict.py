from pathlib import Path
from subprocess import run
import shutil
import argparse
import os
import sys

import cog
from cog import BasePredictor, Input

os.chdir("/Amphion")
os.environ.update({"WORK_DIR": "/Amphion"})

sys.path.append(".")

from models.tts.valle_v2.valle_inference import ValleInference
from models.tts.valle_v2.g2p_processor import G2pProcessor
from models.tts.valle.valle_inference import VALLEInference
from models.tts.naturalspeech2.ns2_inference import NS2Inference
from utils.util import load_config

from encodec import EncodecModel
import nltk


class Predictor(BasePredictor):
    def setup(self) -> None:
        # generic
        EncodecModel.encodec_model_24khz()
        nltk.download('averaged_perceptron_tagger_eng')
        Path("/Amphion/ckpts/tts").mkdir(parents=True, exist_ok=True)
        default_valle_args = {
            "config": None,
            "output_dir": "/results_valle1",
            "vocoder_dir": None,
            "acoustics_dir": None,
            "infer_mode": "single",
            "text": "",
            "text_prompt": "",
            "audio_prompt": "",
            "top_k": -100,
            "temperature": 1.0,
            "continual": False,
            "copysyn": False,
            "mode": "single",
            "log_level": "debug",
        }

        # v1_small
        Path("/Amphion/ckpts/tts/valle_v1_small").mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", "-v", "https://huggingface.co/amphion/valle_libritts", "valle_v1_small"],
            cwd="/Amphion/ckpts/tts",
        )
        args = {
            "mode": "single",
            "config": "/Amphion/ckpts/tts/valle_v1_small/args.json",
            "vocoder_dir": "/Amphion/ckpts/tts/valle_v1_small",
            "acoustics_dir": "/Amphion/ckpts/tts/valle_v1_small",
            "infer_mode": "single",
            "top_k": -100,
            "temperature": 1.0,
            "continual": False,
            "copysyn": False,
        }
        new_args = default_valle_args.copy()
        for k, v in args.items():
            new_args[k] = v
        args = argparse.Namespace(**new_args)
        cfg = load_config(args.config)
        self.infer_valle_v1_small = VALLEInference(args, cfg)

        # v2
        run(
            ["git", "clone", "https://huggingface.co/amphion/valle", "valle_v2"],
            cwd="/Amphion/ckpts/tts",
        )

        # move SpeechTokenizer.pt and config.json to tokenizer subdirectory
        Path("/Amphion/ckpts/tts/valle_v2/tokenizer").mkdir(parents=True, exist_ok=True)
        shutil.move(
            "/Amphion/ckpts/tts/valle_v2/config.json", "/Amphion/ckpts/tts/valle_v2/tokenizer"
        )
        shutil.move(
            "/Amphion/ckpts/tts/valle_v2/SpeechTokenizer.pt",
            "/Amphion/ckpts/tts/valle_v2/tokenizer",
        )

    def predict(
        self,
        model: str = cog.Input(
            choices=[
                "valle_v1_small",
                "valle_v1_medium",
                "valle_v1_large",
                "valle_v2",
                "naturalspeech2",
                "maskgct",
                "vevo",
            ]
        ),
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        if model == "valle_v1_small":
            self.infer_valle_v1_small.args.text = text
            self.infer_valle_v1_small.args.text_prompt = text
            self.infer_valle_v1_small.args.audio_prompt = speaker_reference
            self.infer_valle_v1_small.inference()
            return Path(self.infer_valle_v1_small.output_dir + "/output.wav")