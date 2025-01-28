# Prediction interface for Cog ⚙️
# https://cog.run/python

from pathlib import Path
from hashlib import sha256
import os

from tools.vqgan.inference import load_model as load_vqgan
from tools.llama.generate import load_model as load_llama, generate_long

import torch
import numpy as np
import soundfile as sf
import torchaudio
from transformers import AutoTokenizer

from cog import BasePredictor
import cog

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HYDRA_FULL_ERROR"] = "1"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

GPU = torch.cuda.is_available()

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if GPU else "cpu")
        self.vqgan = load_vqgan(
            "vqgan_pretrain",
            "/src/checkpoints/fishspeech/vq-gan-group-fsq-2x1024.pth",
            device=self.device,
        )
        self.llama, self.decode_one_token = load_llama(
            "dual_ar_2_codebook_medium",
            "/src/checkpoints/fishspeech/text2semantic-sft-medium-v1.1-4k.pth",
            self.device,
            torch.bfloat16,
            2048,
            compile=False
        )
        if GPU:
            torch.cuda.synchronize()
        self.tokenizer = AutoTokenizer.from_pretrained("/src/checkpoints/fishspeech")

    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
        text_reference: str = cog.Input(),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # ----------------
        # vqgan
        # ----------------
        audio, sr = torchaudio.load(str(speaker_reference))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, self.vqgan.sampling_rate)
        audios = audio[None].to(self.vqgan.device)
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=self.vqgan.device, dtype=torch.long
        )
        prompt_tokens = self.vqgan.encode(audios, audio_lengths)[0][0]
        
        # ----------------
        # llama
        # ----------------
        generator = generate_long(
            model=self.llama,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=0,
            top_p=0.7,
            repetition_penalty=1.5,
            temperature=0.7,
            tokenizer=self.tokenizer,
            compile=False,
            speaker=None,
            iterative_prompt=True,
            max_length=2048,
            chunk_length=30,
            prompt_text=text_reference,
            prompt_tokens=prompt_tokens,
        )

        codes = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
            elif response.action == "next":
                if codes:
                    codes = torch.cat(codes, dim=1).to(self.device).long()
                    feature_lengths = torch.tensor([codes.shape[1]], device=self.vqgan.device)
                    fake_audios = self.vqgan.decode(
                        indices=codes[None], feature_lengths=feature_lengths, return_audios=True
                    )
                    fake_audio = fake_audios[0, 0].float().cpu().numpy()
                    output_path = Path(output_dir) / f"generated.wav"
                    sf.write(output_path, fake_audio, self.vqgan.sampling_rate)
                print(f"Next sample")
                codes = []
            else:
                print(f"Error: {response}")

        return cog.Path(output_path)