import os
import random
import shutil
from pathlib import Path
from hashlib import sha256
import sys
import numpy as np
import torch
import torchaudio
import argparse
from cog import BasePredictor
import cog
from whisperspeech.pipeline import Pipeline


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Initialize all pipeline versions upfront
        self.pipelines = {}

        # Initialize tiny, base, small versions
        for version in ["tiny", "base"]:
            pipe = Pipeline(
                optimize=False,
                torch_compile=False,
                s2a_ref=f"collabora/whisperspeech:s2a-q4-{version}-en+pl.model",
                t2s_ref=f"collabora/whisperspeech:t2s-{version}-en+pl.model",
            )
            pipe.t2s.optimize(
                max_batch_size=1, dtype=torch.float32, torch_compile=False
            )
            pipe.s2a.optimize(
                max_batch_size=1, dtype=torch.float32, torch_compile=False
            )
            self.pipelines[version] = pipe

        # Initialize small version
        small_pipe = pipe = Pipeline(
            t2s_ref="whisperspeech/whisperspeech:t2s-v1.95-small-8lang.model",
            s2a_ref="whisperspeech/whisperspeech:s2a-v1.95-medium-7lang.model",
        )

        # Initialize medium version
        medium_pipe = Pipeline(
            s2a_ref="collabora/whisperspeech:s2a-v1.95-medium-7lang.model",
            t2s_ref="collabora/whisperspeech:t2s-v1.95-medium-7lang.model",
        )

        self.pipelines["small"] = small_pipe
        self.pipelines["medium"] = medium_pipe

        # print s2a and t2s number of parameters for each version
        print(
            f"Tiny params: {get_model_params(self.pipelines['tiny'].s2a) + get_model_params(self.pipelines['tiny'].t2s)}"
        )
        print(
            f"Base params: {get_model_params(self.pipelines['base'].s2a) + get_model_params(self.pipelines['base'].t2s)}"
        )
        print(
            f"Small params: {get_model_params(self.pipelines['small'].s2a) + get_model_params(self.pipelines['small'].t2s)}"
        )
        print(
            f"Medium params: {get_model_params(self.pipelines['medium'].s2a) + get_model_params(self.pipelines['medium'].t2s)}"
        )

    def predict(
        self,
        text: str = cog.Input(description="Text to synthesize"),
        speaker_reference: cog.Path = cog.Input(description="Reference audio file"),
        version: str = cog.Input(
            description="Version of the model to use",
            default="small",
            choices=["tiny", "base", "small", "medium"],
        ),
        language: str = cog.Input(
            description="Language of the text",
            default="en",
            choices=["en", "pl", "de", "fr", "it", "nl", "es", "pt"],
        ),
    ) -> cog.Path:
        print(f"Received request for {version}")

        # Create output directory
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/output.wav"

        # Get appropriate pipeline
        pipe = self.pipelines[version]

        speaker = pipe.extract_spk_emb(speaker_reference)

        # Generate speech
        print("Generating speech")
        pipe.generate_to_file(
            output_path, text=text, lang=language, speaker=str(speaker_reference)
        )

        return cog.Path(output_path)
