from pathlib import Path
from hashlib import sha256
import numpy as np

from cog import BasePredictor
import cog

import torch
import torch.nn.functional as F
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import torchaudio

GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if GPU else "cpu"

        # Load the TTS models
        self.processor = SpeechT5Processor.from_pretrained("/src/checkpoints/speecht5")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            "/src/checkpoints/speecht5"
        )
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "/src/checkpoints/speecht5_hifigan"
        )

        # Load SpeechBrain x-vector speaker embedding model
        self.classifier = EncoderClassifier.from_hparams(
            source="/src/checkpoints/xvector",
            run_opts={"device": self.device},
            savedir="/tmp/speechbrain_speaker_embedding",
        )

        # to device
        self.classifier.to(self.device)
        self.tts_model.to(self.device)
        self.vocoder.to(self.device)

    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(default=None),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        audio, sampling_rate = torchaudio.load(speaker_reference)
        audio = audio.squeeze(0).to(self.device)

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio = resampler(audio)

        with torch.no_grad():
            speaker_embedding = self.classifier.encode_batch(audio.unsqueeze(0))
            speaker_embedding = F.normalize(speaker_embedding, dim=2)
            speaker_embedding = speaker_embedding.squeeze(0)

        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            wav = self.tts_model.generate(
                inputs["input_ids"],
                speaker_embeddings=speaker_embedding,
                vocoder=self.vocoder,
            )

        wav = wav.squeeze().detach().cpu().numpy()
        torchaudio.save(
            f"{output_dir}/output.wav", torch.tensor(wav).unsqueeze(0), 16000
        )

        return cog.Path(f"{output_dir}/output.wav")
