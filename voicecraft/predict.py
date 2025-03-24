import os
import random
import shutil
from pathlib import Path
from hashlib import sha256
import sys
import tempfile
import numpy as np
import torch
import torchaudio
import argparse
import re
from num2words import num2words
from whisperx import load_model, load_align_model, load_audio, align

sys.path.append("/voicecraft")

from inference_tts_scale import inference_one_sample
from models import voicecraft
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

from cog import BasePredictor
import cog

GPU = torch.cuda.is_available()


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if GPU else "cpu"

        # Load VoiceCraft model and tokenizers
        self.model_giga330m = voicecraft.VoiceCraft.from_pretrained(
            "/src/checkpoints/voicecraft_giga330m"
        )
        self.model_giga830m = voicecraft.VoiceCraft.from_pretrained(
            "/src/checkpoints/voicecraft_giga830m"
        )
        self.model_giga_libritts_330m = voicecraft.VoiceCraft.from_pretrained(
            "/src/checkpoints/voicecraft_giga_libritts_330m"
        )
        self.model_giga330m_tts_enhanced = voicecraft.VoiceCraft.from_pretrained(
            "/src/checkpoints/voicecraft_giga330m_tts_enhanced"
        )
        self.model_giga830m_tts_enhanced = voicecraft.VoiceCraft.from_pretrained(
            "/src/checkpoints/voicecraft_giga830m_tts_enhanced"
        )

        self.model_giga330m.to(self.device)
        self.model_giga830m.to(self.device)
        self.model_giga_libritts_330m.to(self.device)
        self.model_giga330m_tts_enhanced.to(self.device)
        self.model_giga830m_tts_enhanced.to(self.device)

        # Load audio tokenizer
        encodec_fn = "/src/checkpoints/encodec/encodec_4cb2048_giga.th"
        self.audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=self.device)

        # Load text tokenizer
        self.text_tokenizer = TextTokenizer(backend="espeak")

        # Load WhisperX models
        asr_options = {
            "hotwords": None,
            "multilingual": False,
            "max_new_tokens": None, 
            "clip_timestamps": None, 
            "hallucination_silence_threshold": None,
        }
        self.whisper_model = load_model("base.en", self.device, compute_type="float32", asr_options=asr_options)
        self.align_model, self.align_metadata = load_align_model(language_code="en", device=self.device)

        # Set default parameters
        self.silence_tokens = [1388, 1898, 131]
        self.codec_audio_sr = 16000
        self.codec_sr = 50
        self.top_k = 0
        self.top_p = 0.9
        self.temperature = 1
        self.kvcache = 1
        self.stop_repetition = -1
        self.sample_batch_size = 3
        self.beam_size = 50
        self.retry_beam_size = 200
        self.cut_off_sec = 3.6
        self.margin = 0.04
        self.cutoff_tolerance = 1

    def _find_word_boundary_from_whisperx(self, segments, cut_off_sec, margin, cutoff_tolerance=1):
        cutoff_time = None
        cutoff_index = None
        cutoff_time_best = None
        cutoff_index_best = None
        
        # Extract word information from segments
        words_info = []
        for segment in segments:
            for word in segment.get("words", []):
                words_info.append(word)
        
        try:
            for i, word in enumerate(words_info):
                end = word.get("end", 0)
                if end >= cut_off_sec and cutoff_time is None:
                    cutoff_time = end
                    cutoff_index = i
                if (
                    end >= cut_off_sec
                    and end < cut_off_sec + cutoff_tolerance
                    and i+1 < len(words_info)
                    and words_info[i+1].get("start", 0) - end >= margin
                ):
                    cutoff_time_best = end + margin * 2 / 3
                    cutoff_index_best = i
                    break
            if cutoff_time_best is not None:
                cutoff_time = cutoff_time_best
                cutoff_index = cutoff_index_best
        except Exception as e:
            print(f"Error finding word boundary: {e}")
            
        if cutoff_time is None and words_info:
            cutoff_time = words_info[-1].get("end", 0)
            cutoff_index = len(words_info) - 1
        elif cutoff_time is None:
            cutoff_time = cut_off_sec
            cutoff_index = 0
            
        return cutoff_time, cutoff_index
        
    def replace_numbers_with_words(self, text):
        text = re.sub(r'(\d+)', r' \1 ', text)  # add spaces around numbers
        
        def replace_with_words(match):
            num = match.group(0)
            try:
                return num2words(num)
            except:
                return num
                
        return re.sub(r'\b\d+\b', replace_with_words, text)

    def predict(
        self,
        text: str = cog.Input(description="Text to synthesize"),
        speaker_reference: cog.Path = cog.Input(description="Reference audio file"),
        text_reference: str = cog.Input(description="Transcript of reference audio"),
        version: str = cog.Input(
            description="Version of the model to use",
            default="giga330m",
            choices=[
                "giga330m",
                "giga830m",
                "giga_libritts_330m",
                "giga330m_tts_enhanced",
                "giga830m_tts_enhanced",
            ],
        ),
    ) -> cog.Path:
        # Create output directory
        output_dir = "/results/" + sha256(np.random.bytes(32)).hexdigest()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        temp_audio = temp_dir / "reference.wav"
        temp_json = temp_dir / "whisperx_result.json"
        
        # Copy reference audio
        shutil.copy(speaker_reference, temp_audio)
        
        # Preprocess reference text
        text_reference = self.replace_numbers_with_words(text_reference)

        # Get audio info
        info = torchaudio.info(temp_audio)
        audio_dur = info.num_frames / info.sample_rate

        # Perform WhisperX alignment
        audio = load_audio(temp_audio)
        result = self.whisper_model.transcribe(str(temp_audio))
        segments = result["segments"]
        
        # Align with reference text
        segments = align(segments, self.align_model, self.align_metadata, audio, self.device, return_char_alignments=False)["segments"]
        
        # Save alignment results for debugging
        import json
        with open(temp_json, "w") as f:
            json.dump(segments, f, indent=2)
        
        # Find cutoff point using WhisperX segments
        cut_off_sec, cut_off_word_idx = self._find_word_boundary_from_whisperx(
            segments, self.cut_off_sec, self.margin, self.cutoff_tolerance
        )

        # Extract words from segments to build transcript
        words = []
        for segment in segments:
            for word in segment.get("words", []):
                words.append(word.get("word", ""))
        
        # Process transcript
        cut_off_word_idx = min(cut_off_word_idx, len(words) - 1)
        target_transcript = " ".join(words[:cut_off_word_idx + 1]) + " " + text

        # Ensure cutoff is within audio bounds
        cut_off_sec = min(cut_off_sec, audio_dur)
        prompt_end_frame = int(cut_off_sec * info.sample_rate)

        # Inference config
        decode_config = {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "stop_repetition": self.stop_repetition,
            "kvcache": self.kvcache,
            "codec_audio_sr": self.codec_audio_sr,
            "codec_sr": self.codec_sr,
            "silence_tokens": self.silence_tokens,
            "sample_batch_size": self.sample_batch_size,
        }

        if version == "giga330m":
            model = self.model_giga330m
            phn2num = self.model_giga330m.args.phn2num
            config = vars(self.model_giga330m.args)
        elif version == "giga830m":
            model = self.model_giga830m
            phn2num = self.model_giga830m.args.phn2num
            config = vars(self.model_giga830m.args)
        elif version == "giga_libritts_330m":
            model = self.model_giga_libritts_330m
            phn2num = self.model_giga_libritts_330m.args.phn2num
            config = vars(self.model_giga_libritts_330m.args)
        elif version == "giga330m_tts_enhanced":
            model = self.model_giga330m_tts_enhanced
            phn2num = self.model_giga330m_tts_enhanced.args.phn2num
            config = vars(self.model_giga330m_tts_enhanced.args)
        elif version == "giga830m_tts_enhanced":
            model = self.model_giga830m_tts_enhanced
            phn2num = self.model_giga830m_tts_enhanced.args.phn2num
            config = vars(self.model_giga830m_tts_enhanced.args)

        # Run inference
        _, gen_audio = inference_one_sample(
            model,
            argparse.Namespace(**config),
            phn2num,
            self.text_tokenizer,
            self.audio_tokenizer,
            str(temp_audio),
            target_transcript,
            self.device,
            decode_config,
            prompt_end_frame,
        )

        # Save output
        output_path = f"{output_dir}/output.wav"
        torchaudio.save(output_path, gen_audio[0].cpu(), self.codec_audio_sr)

        # Cleanup
        shutil.rmtree(temp_dir)

        return cog.Path(output_path)
