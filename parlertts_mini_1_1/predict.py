import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor
import torchaudio
from hashlib import sha256
import numpy as np
from pathlib import Path

from cog import BasePredictor
import cog

GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if GPU else "cpu"
        model_path = "/src/checkpoints/parlertts_mini_1_1"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.sr = self.model.config.sampling_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(default=None),
        text_reference: str = cog.Input(default=None),
        prompt: str = cog.Input(default=""),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if speaker_reference is not None and text_reference is not None:
            init_audio, init_sr = torchaudio.load(speaker_reference)
            init_audio = torchaudio.functional.resample(init_audio, init_sr, 16_000)
            init_audio = torchaudio.functional.resample(init_audio, 16_000, self.sr)
            init_audio = init_audio.mean(0)
            input_values = self.feature_extractor(
                init_audio, sampling_rate=self.sr, return_tensors="pt"
            )
            input_values = torch.tensor(input_values.input_values).to(self.device)
            prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            text_reference = text_reference.strip()
            if text_reference[-1] not in [".", "!", "?"]:
                text_reference += "."
            text_input_ids = self.tokenizer(
                text_reference + " " + text, return_tensors="pt"
            ).input_ids.to(self.device)
            generation = self.model.generate(
                input_ids=prompt_input_ids,
                prompt_input_ids=text_input_ids,
                input_values=input_values,
                generation_config=self.model.generation_config,
            )
            generation = generation[0, input_values.shape[2]:].cpu().unsqueeze(0)
            torchaudio.save(output_dir + "/output.wav", generation, self.sr)
            return cog.Path(output_dir + "/output.wav")
        else:
            prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            text_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            generation = self.model.generate(
                input_ids=prompt_input_ids,
                prompt_input_ids=text_input_ids,
                generation_config=self.model.generation_config,
            )
            generation = generation[0].cpu().unsqueeze(0)
            torchaudio.save(output_dir + "/output.wav", generation, self.sr)
            return cog.Path(output_dir + "/output.wav")