# import os
# import random
# import shutil
# from pathlib import Path
# from hashlib import sha256
# import sys
# import tempfile
# import numpy as np
# import torch
# import torchaudio
# import argparse

# sys.path.append("/voicecraft")

# from inference_tts_scale import inference_one_sample
# from models import voicecraft
# from data.tokenizer import (
#     AudioTokenizer,
#     TextTokenizer,
# )

# from cog import BasePredictor
# import cog

# GPU = torch.cuda.is_available()


# class Predictor(BasePredictor):
#     def setup(self) -> None:
#         self.device = "cuda" if GPU else "cpu"

#         # Load VoiceCraft model and tokenizers
#         self.model_giga330m = voicecraft.VoiceCraft.from_pretrained(
#             "/src/checkpoints/voicecraft_giga330m"
#         )
#         self.model_giga830m = voicecraft.VoiceCraft.from_pretrained(
#             "/src/checkpoints/voicecraft_giga830m"
#         )
#         self.model_giga_libritts_330m = voicecraft.VoiceCraft.from_pretrained(
#             "/src/checkpoints/voicecraft_giga_libritts_330m"
#         )
#         self.model_giga330m_tts_enhanced = voicecraft.VoiceCraft.from_pretrained(
#             "/src/checkpoints/voicecraft_giga330m_tts_enhanced"
#         )
#         self.model_giga830m_tts_enhanced = voicecraft.VoiceCraft.from_pretrained(
#             "/src/checkpoints/voicecraft_giga830m_tts_enhanced"
#         )

#         self.model_giga330m.to(self.device)
#         self.model_giga830m.to(self.device)
#         self.model_giga_libritts_330m.to(self.device)
#         self.model_giga330m_tts_enhanced.to(self.device)
#         self.model_giga830m_tts_enhanced.to(self.device)

#         # Load audio tokenizer
#         encodec_fn = "/src/checkpoints/encodec/encodec_4cb2048_giga.th"
#         self.audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=self.device)

#         # Load text tokenizer
#         self.text_tokenizer = TextTokenizer(backend="espeak")

#         # Set default parameters
#         self.silence_tokens = [1388, 1898, 131]
#         self.codec_audio_sr = 16000
#         self.codec_sr = 50
#         self.top_k = 0
#         self.top_p = 0.9
#         self.temperature = 1
#         self.kvcache = 1
#         self.stop_repetition = -1
#         self.sample_batch_size = 3
#         self.beam_size = 50
#         self.retry_beam_size = 200
#         self.cut_off_sec = 3.6
#         self.margin = 0.04
#         self.cutoff_tolerance = 1

#     def _find_closest_word_boundary(
#         self, alignments, cut_off_sec, margin, cutoff_tolerance=1
#     ):
#         with open(alignments, "r") as file:
#             next(file)
#             cutoff_time = None
#             cutoff_index = None
#             cutoff_time_best = None
#             cutoff_index_best = None
#             lines = [l for l in file.readlines() if "words" in l]
#             try:
#                 for i, line in enumerate(lines):
#                     end = float(line.strip().split(",")[1])
#                     if end >= cut_off_sec and cutoff_time == None:
#                         cutoff_time = end
#                         cutoff_index = i
#                     if (
#                         end >= cut_off_sec
#                         and end < cut_off_sec + cutoff_tolerance
#                         and float(lines[i + 1].strip().split(",")[0]) - end >= margin
#                     ):
#                         cutoff_time_best = end + margin * 2 / 3
#                         cutoff_index_best = i
#                         break
#                 if cutoff_time_best != None:
#                     cutoff_time = cutoff_time_best
#                     cutoff_index = cutoff_index_best
#             except:
#                 pass
#             if cutoff_time == None:
#                 cutoff_time = float(lines[-1].strip().split(",")[1])
#                 cutoff_index = len(lines) - 1
#             return cutoff_time, cutoff_index

#     def predict(
#         self,
#         text: str = cog.Input(description="Text to synthesize"),
#         speaker_reference: cog.Path = cog.Input(description="Reference audio file"),
#         text_reference: str = cog.Input(description="Transcript of reference audio"),
#         version: str = cog.Input(
#             description="Version of the model to use",
#             default="giga330m",
#             choices=[
#                 "giga330m",
#                 "giga830m",
#                 "giga_libritts_330m",
#                 "giga330m_tts_enhanced",
#                 "giga830m_tts_enhanced",
#             ],
#         ),
#     ) -> cog.Path:
#         # Create output directory
#         output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
#         Path(output_dir).mkdir(parents=True, exist_ok=True)

#         # Create temp directory for MFA
#         temp_dir = Path(tempfile.mkdtemp())
#         temp_audio = temp_dir / "reference.wav"
#         temp_text = temp_dir / "reference.txt"

#         # Copy reference audio and text
#         shutil.copy(speaker_reference, temp_audio)
#         with open(temp_text, "w") as f:
#             f.write(text_reference)

#         # Run MFA alignment
#         align_temp = temp_dir / "mfa_alignments"
#         alignments = align_temp / "reference.csv"
#         os.system(
#             f'/bin/bash -c  "source /cog/miniconda/bin/activate && conda activate myenv && mfa align -v --clean -j 1 --output_format csv {temp_dir} \
#                 english_us_arpa english_us_arpa {align_temp} --beam {self.beam_size} --retry_beam {self.retry_beam_size}"'
#         )

#         # Find cutoff point
#         cut_off_sec, cut_off_word_idx = self._find_closest_word_boundary(
#             alignments, self.cut_off_sec, self.margin, self.cutoff_tolerance
#         )

#         # Process transcript
#         orig_split = text_reference.split(" ")
#         cut_off_word_idx = min(cut_off_word_idx, len(orig_split) - 1)
#         target_transcript = " ".join(orig_split[: cut_off_word_idx + 1]) + " " + text

#         # Get audio duration and cutoff
#         info = torchaudio.info(temp_audio)
#         audio_dur = info.num_frames / info.sample_rate
#         cut_off_sec = min(cut_off_sec, audio_dur)
#         prompt_end_frame = int(cut_off_sec * info.sample_rate)

#         # Inference config
#         decode_config = {
#             "top_k": self.top_k,
#             "top_p": self.top_p,
#             "temperature": self.temperature,
#             "stop_repetition": self.stop_repetition,
#             "kvcache": self.kvcache,
#             "codec_audio_sr": self.codec_audio_sr,
#             "codec_sr": self.codec_sr,
#             "silence_tokens": self.silence_tokens,
#             "sample_batch_size": self.sample_batch_size,
#         }

#         if version == "giga330m":
#             model = self.model_giga330m
#             phn2num = self.model_giga330m.args.phn2num
#             config = vars(self.model_giga330m.args)
#         elif version == "giga830m":
#             model = self.model_giga830m
#             phn2num = self.model_giga830m.args.phn2num
#             config = vars(self.model_giga830m.args)
#         elif version == "giga_libritts_330m":
#             model = self.model_giga_libritts_330m
#             phn2num = self.model_giga_libritts_330m.args.phn2num
#             config = vars(self.model_giga_libritts_330m.args)
#         elif version == "giga330m_tts_enhanced":
#             model = self.model_giga330m_tts_enhanced
#             phn2num = self.model_giga330m_tts_enhanced.args.phn2num
#             config = vars(self.model_giga330m_tts_enhanced.args)
#         elif version == "giga830m_tts_enhanced":
#             model = self.model_giga830m_tts_enhanced
#             phn2num = self.model_giga830m_tts_enhanced.args.phn2num
#             config = vars(self.model_giga830m_tts_enhanced.args)

#         # Run inference
#         _, gen_audio = inference_one_sample(
#             model,
#             argparse.Namespace(**config),
#             phn2num,
#             self.text_tokenizer,
#             self.audio_tokenizer,
#             str(temp_audio),
#             target_transcript,
#             self.device,
#             decode_config,
#             prompt_end_frame,
#         )

#         # Save output
#         output_path = f"{output_dir}/output.wav"
#         torchaudio.save(output_path, gen_audio[0].cpu(), self.codec_audio_sr)

#         # Cleanup
#         shutil.rmtree(temp_dir)

#         return cog.Path(output_path)

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
        small_pipe = Pipeline(
            optimize=False,
            torch_compile=False,
            s2a_ref="/src/checkpoints/whisperspeech/s2a-v1.95-medium-7lang.model",
            t2s_ref="/src/checkpoints/whisperspeech/t2s-v1.95-small-8lang.model",
        )

        # Initialize medium version
        medium_pipe = Pipeline(
            optimize=False,
            torch_compile=False,
            s2a_ref="collabora/whisperspeech:s2a-v1.95-medium-7lang.model",
            t2s_ref="collabora/whisperspeech:t2s-v1.95-medium-7lang.model",
        )

        self.pipelines["small"] = small_pipe
        self.pipelines["medium"] = medium_pipe

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

        # Generate speech
        print("Generating speech")
        pipe.s2a.dtype = torch.float32  # hack to fix faulty inference code
        pipe.generate_to_file(
            output_path, text=text, lang=language, speaker=str(speaker_reference)
        )

        return cog.Path(output_path)
