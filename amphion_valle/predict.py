from pathlib import Path
from subprocess import run
import shutil
import argparse
import os
import sys
from hashlib import sha256

import cog
from cog import BasePredictor, Input

from encodec import EncodecModel
import nltk
import torch
import librosa
import torchaudio
import numpy as np

# ----------------
# Amphion
# ----------------
os.chdir("/Amphion")
os.environ.update({"WORK_DIR": "/Amphion"})
sys.path.append(".")

from models.tts.valle.valle_inference import VALLEInference
from models.tts.valle_v2.valle_inference import ValleInference as VALLEInferenceV2
from models.tts.valle_v2.g2p_processor import G2pProcessor
from utils.util import load_config

GPU = torch.cuda.is_available()

class Predictor(BasePredictor):
    def setup(self) -> None:
        # generic
        EncodecModel.encodec_model_24khz()
        nltk.download('averaged_perceptron_tagger_eng')
        device = "cuda" if GPU else "cpu"
        
        # ----------------
        # valle_v1_small
        # ----------------
        args = {
            "mode": "single",
            "config": "/src/checkpoints/valle_v1_small/args.json",
            "vocoder_dir": "/src/checkpoints/valle_v1_small",
            "acoustics_dir": "/src/checkpoints/valle_v1_small",
            "output_dir": "/results",
            "mode": "single",
            "log_level": "debug",
            "mode": "single",
            "text_prompt": "",
            "audio_prompt": "",
            "top_k": -100,
            "temperature": 1.0,
            "continual": False,
            "copysyn": False,
            "device": device,
        }
        Path(args["output_dir"]).mkdir(parents=True, exist_ok=True)
        ns_args = argparse.Namespace(**args)
        cfg = load_config(ns_args.config)
        self.valle_v1_small = VALLEInference(ns_args, cfg)
        if GPU:
            self.valle_v1_small.model = self.valle_v1_small.model.to(device)
        
        # ----------------
        # valle_v1_medium
        # ----------------
        args["config"] = "/src/checkpoints/valle_v1_medium/args.json"
        args["vocoder_dir"] = "/src/checkpoints/valle_v1_medium"
        args["acoustics_dir"] = "/src/checkpoints/valle_v1_medium"
        ns_args = argparse.Namespace(**args)
        cfg = load_config(ns_args.config)
        self.valle_v1_medium = VALLEInference(ns_args, cfg)
        if GPU:
            self.valle_v1_medium.model = self.valle_v1_medium.model.to(device)

        # ----------------
        # valle_v2
        # ----------------
        self.g2p_processor = G2pProcessor()
        self.valle_v2 = VALLEInferenceV2(
            ar_path="/src/checkpoints/valle_v2/valle_ar_mls_196000.bin",
            nar_path="/src/checkpoints/valle_v2/valle_nar_mls_164000.bin",
            speechtokenizer_path="/src/checkpoints/valle_v2/tokenizer",
            device=device,
        )

    def predict(
        self,
        model: str = cog.Input(
            choices=[
                "valle_v1_small",
                "valle_v1_medium",
                "valle_v2",
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

        # run prediction
        if model == "valle_v1_small":
            # see https://github.com/open-mmlab/Amphion/tree/main/egs/tts/VALLE
            self.valle_v1_small.args.text = text
            self.valle_v1_small.args.text_prompt = text_reference
            self.valle_v1_small.args.audio_prompt = str(speaker_reference)
            self.valle_v1_small.args.output_dir = output_dir
            self.valle_v1_small.inference()
            return cog.Path(self.valle_v1_small.args.output_dir + "/single/test_pred.wav")
        elif model == "valle_v1_medium":
            # see https://github.com/open-mmlab/Amphion/tree/main/egs/tts/VALLE
            self.valle_v1_medium.args.text = text
            self.valle_v1_medium.args.text_prompt = text_reference
            self.valle_v1_medium.args.audio_prompt = str(speaker_reference)
            self.valle_v1_medium.args.output_dir = output_dir
            self.valle_v1_medium.inference()
            return cog.Path(self.valle_v1_medium.args.output_dir + "/single/test_pred.wav")
        elif model == "valle_v2":
            # see https://github.com/open-mmlab/Amphion/blob/main/egs/tts/VALLE_V2/demo.ipynb
            wav, _ = librosa.load(str(speaker_reference), sr=16000)
            wav = torch.tensor(wav, dtype=torch.float32)
            text_prompt = self.g2p_processor(text_reference, "en")[1]
            text = self.g2p_processor(text, "en")[1]
            text_prompt = torch.tensor(text_prompt, dtype=torch.long)
            text = torch.tensor(text, dtype=torch.long)
            transcript = torch.cat([text_prompt, text])
            batch = {"speech": wav.unsqueeze(0), "phone_ids": transcript.unsqueeze(0)}
            configs = [dict(
                top_p=0.9,
                top_k=5,
                temperature=0.95,
                repeat_penalty=1.0,
                max_length=2000,
                num_beams=1,
            )]
            output_wav = self.valle_v2(batch, configs)
            if GPU:
                torchaudio.save(output_dir + "/test_pred.wav", output_wav.squeeze(0).cpu(), 16000)
            else:
                torchaudio.save(output_dir + "/test_pred.wav", output_wav.squeeze(0), 16000)
            return cog.Path(output_dir + "/test_pred.wav")