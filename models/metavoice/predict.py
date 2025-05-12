import sys
import shutil
from pathlib import Path
from hashlib import sha256
import numpy as np

sys.path.append('/metavoice')

from fam.llm.fast_inference import TTS

from cog import BasePredictor
import cog


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.metavoice = TTS(model_name="/src/checkpoints/metavoice")

    def predict(
        self,
        text: str = cog.Input(),
        speaker_reference: cog.Path = cog.Input(),
    ) -> cog.Path:
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        top_p = 0.95
        guidance_scale = 3.0
        temperature = 1.0

        wav_file_path = self.metavoice.synthesise(
            text=text,
            spk_ref_path=str(speaker_reference),
            top_p=top_p,
            guidance_scale=guidance_scale,
            temperature=temperature,
        )

        shutil.copy(wav_file_path, output_dir + "/output.wav")

        return cog.Path(output_dir + "/output.wav")
