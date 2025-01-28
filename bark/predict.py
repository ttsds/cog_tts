import sys
from pathlib import Path
from hashlib import sha256
import os

import cog
from cog import BasePredictor

import torch
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import numpy as np
import soundfile as sf
from bark import SAMPLE_RATE, generate_audio, preload_models

GPU = torch.cuda.is_available()

sys.path.append("/src/checkpoints/bark_vc_code")

from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        device = "cuda" if GPU else "cpu"
        self.device = torch.device(device)
        self.hubert_model = CustomHubert(
            "/src/checkpoints/bark_vc_code/data/models/hubert/hubert.pt", device=device
        )
        self.tokenizers = {
            "en": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_en.pth",
                map_location=device,
            ),
            "de": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_de.pth",
                map_location=device,
                
            ),
            "es": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_es.pth",
                map_location=device,
            ),
            "it": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_it.pth",
                map_location=device,
            ),
            "ja": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_ja.pth",
                map_location=device,
            ),
            "pl": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_pl.pth",
                map_location=device,
            ),
            "pt": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_pt.pth",
                map_location=device,
            ),
            "tr": CustomTokenizer.load_from_checkpoint(
                "/src/checkpoints/bark_vc_code/data//models/hubert/tokenizer_tr.pth",
                map_location=device,
            ),
        }
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(6.0)
        self.encodec.to(device)
        preload_models(
            text_use_small=False,
            coarse_use_small=False,
            fine_use_small=False,
        )

    def predict(
        self,
        language: str = cog.Input(
            choices=[
                "en",
                "de",
                "es",
                "it",
                "ja",
                "pl",
                "pt",
                "tr",
            ]
        ),
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
    ) -> cog.Path:
        """Run a single prediction on the model"""
        # random output dir
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ----------------
        # create prompt
        # ----------------
        wav, sr = torchaudio.load(speaker_reference)
        wav = wav.to(self.device)
        if wav.shape[0] == 2:  # Stereo to mono if needed
            wav = wav.mean(0, keepdim=True)
        print("Extracting semantics...")
        semantic_vectors = self.hubert_model.forward(wav, input_sample_hz=sr)
        print("Tokenizing semantics...")
        semantic_tokens = self.tokenizers[language].get_token(semantic_vectors)
        print("Creating coarse and fine prompts...")
        wav = wav.cpu()
        wav = convert_audio(wav, sr, self.encodec.sample_rate, 1).unsqueeze(0)
        wav = wav.to(self.device)
        with torch.no_grad():
            encoded_frames = self.encodec.encode(wav)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()
        codes = codes.cpu()
        semantic_tokens = semantic_tokens.cpu()
        npz_file = Path(output_dir) / "prompt.npz"
        np.savez(
            npz_file,
            semantic_prompt=semantic_tokens,
            fine_prompt=codes,
            coarse_prompt=codes[:2, :],
        )

        # ----------------
        # bark tts
        # ----------------
        audio_array = generate_audio(
            text, history_prompt=str(npz_file)
        )
        output_path = Path(output_dir) / "test_pred.wav"
        sf.write(str(output_path), audio_array, SAMPLE_RATE)
        return cog.Path(output_path)