import os
import torch
import numpy as np
import torchaudio
import sys

sys.path.append("/hierspeechpp")

import utils

from Mels_preprocess import MelSpectrogramFixed

from hierspeechpp_speechsynthesizer import SynthesizerTrn as HierSpeechSynth
from ttv_v1.text import text_to_sequence
from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2VSynth
from denoiser.infer import denoise

torch.backends.cuda.cufft_plan_cache[0].max_size = 0

def intersperse(lst, item):
    """Insert `item` between each element of `lst`."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def add_blank_token(text: list):
    """Intersperse blank token (0) among the text tokens."""
    text_norm = intersperse(text, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def load_checkpoint(filepath, device):
    """Simple checkpoint loader without prints/asserts."""
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


class HierspeechSynthesizer:
    """
    A class that encapsulates the Hierspeech++ pipeline:
      1) Text-to-W2V + F0 extraction
      2) HiFi-VC style voice conversion
      3) Optional Speech SR to 24k or 48k
      4) Optional denoising
    """

    def __init__(
        self,
        hierspeech_ckpt: str,
        text2w2v_ckpt: str,
        config_hierspeech: str,
        config_text2w2v: str,
        device: str = "cuda"
    ):
        """
        Args:
            hierspeech_ckpt: Path to HierSpeech++ checkpoint.
            text2w2v_ckpt: Path to Text2W2V checkpoint.
            config_hierspeech: JSON config for HierSpeech++.
            config_text2w2v: JSON config for Text2W2V.
            device: 'cuda' or 'cpu'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Store paths
        self.hierspeech_ckpt = hierspeech_ckpt
        self.text2w2v_ckpt = text2w2v_ckpt

        # Load config files as HParams
        self.hps_hierspeech = utils.get_hparams_from_file(config_hierspeech)
        self.hps_text2w2v = utils.get_hparams_from_file(config_text2w2v)
        
        # Build / Load all models
        self._build_mel_fn()
        self._build_models()

    def _build_mel_fn(self):
        """Build the mel-spectrogram transform used in the pipeline."""
        hps = self.hps_hierspeech
        self.mel_fn = MelSpectrogramFixed(
            sample_rate=hps.data.sampling_rate,
            n_fft=hps.data.filter_length,
            win_length=hps.data.win_length,
            hop_length=hps.data.hop_length,
            f_min=hps.data.mel_fmin,
            f_max=hps.data.mel_fmax,
            n_mels=hps.data.n_mel_channels,
            window_fn=torch.hann_window
        ).to(self.device)

    def _build_models(self):
        """Build and load weights for all models in the pipeline."""
        # HiFi-VC style HierSpeech++ model
        hps = self.hps_hierspeech
        self.net_g = HierSpeechSynth(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model
        ).to(self.device)
        self.net_g.load_state_dict(torch.load(self.hierspeech_ckpt, map_location=self.device))
        self.net_g.eval()

        # Text2W2V + F0 model
        hps_t2w2v = self.hps_text2w2v
        self.text2w2v = Text2W2VSynth(
            hps_t2w2v.data.filter_length // 2 + 1,
            hps_t2w2v.train.segment_size // hps_t2w2v.data.hop_length,
            **hps_t2w2v.model
        ).to(self.device)
        self.text2w2v.load_state_dict(torch.load(self.text2w2v_ckpt, map_location=self.device))
        self.text2w2v.eval()

    def infer(
        self,
        text: str,
        prompt_audio_path: str,
        denoise_ratio: float = 0.0,
        noise_scale_ttv: float = 0.333,
        noise_scale_vc: float = 0.333,
        scale_norm: str = 'max',
        return_int16_wav: bool = True
    ):
        """
        Run inference from text + prompt audio. Returns either float32
        or int16 numpy array depending on `return_int16_wav`.

        Args:
            text: Input text string.
            prompt_audio_path: Path to reference audio prompt.
            denoise_ratio: Denoising ratio (0.0 to 1.0).
            noise_scale_ttv: Noise scale for Text2W2V.
            noise_scale_vc: Noise scale for voice conversion.
            scale_norm: If 'prompt', matches amplitude to the prompt;
                        otherwise normalizes to 0.999 max.
            return_int16_wav: If True, returns int16 samples, else float32.

        Returns:
            A tuple (wav_data, sr):
              - wav_data: Numpy array containing the synthesized speech.
              - sr: Sample rate of `wav_data`.
        """
        # Prepare text tokens
        text_seq = text_to_sequence(str(text), ["english_cleaners2"])
        token = add_blank_token(text_seq).unsqueeze(0).to(self.device)
        token_length = torch.LongTensor([token.size(-1)]).to(self.device)

        # Load prompt audio
        audio, sample_rate = torchaudio.load(prompt_audio_path)
        # Support only single channel
        audio = audio[:1, :]

        # Resample to 16k if needed
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                audio, sample_rate, 16000, resampling_method="kaiser_window"
            )

        # Prompt scale for matching amplitude if user wants it
        if scale_norm == 'prompt':
            prompt_audio_max = torch.max(audio.abs())

        # Pad to a multiple for denoiser
        ori_prompt_len = audio.shape[-1]
        multiple_of = 1600
        needed_pad = (ori_prompt_len // multiple_of + 1) * multiple_of - ori_prompt_len
        audio = torch.nn.functional.pad(audio, (0, needed_pad), mode='constant')

        # Denoise the prompt
        if denoise_ratio == 0.0:
            # If ratio is 0, we just replicate the original audio to keep
            # shape consistency for subsequent steps.
            audio = torch.cat([audio.to(self.device), audio.to(self.device)], dim=0)
        else:
            with torch.no_grad():
                denoised_audio = denoise(
                    audio.squeeze(0).to(self.device),
                    self.denoiser,
                    self.hps_denoiser
                )
            audio = torch.cat(
                [audio.to(self.device), denoised_audio[:, :audio.shape[-1]]],
                dim=0
            )

        # Remove extra padding again
        audio = audio[:, :ori_prompt_len]

        # Convert prompt audio to mel
        src_mel = self.mel_fn(audio)
        src_length = torch.LongTensor([src_mel.size(2)]).to(self.device)
        src_length2 = torch.cat([src_length, src_length], dim=0)

        # Text2W2V inference (w2v_x, pitch)
        with torch.no_grad():
            w2v_x, pitch = self.text2w2v.infer_noise_control(
                token,
                token_length,
                src_mel,
                src_length2,
                noise_scale=noise_scale_ttv,
                denoise_ratio=denoise_ratio
            )

            # Clip pitch below log(55) -> zero out
            pitch[pitch < torch.log(torch.tensor([55]).to(self.device))] = 0

            # Hierarchical Speech Synthesis (16k)
            new_src_len = torch.LongTensor([w2v_x.size(2)]).to(self.device)
            converted_audio = self.net_g.voice_conversion_noise_control(
                w2v_x,
                new_src_len,
                src_mel,
                src_length2,
                pitch,
                noise_scale=noise_scale_vc,
                denoise_ratio=denoise_ratio
            )

        # Normalize
        converted_audio = converted_audio.squeeze(0)
        max_abs_value = torch.max(torch.abs(converted_audio))
        if scale_norm == 'prompt':
            # Scale to match prompt's max amplitude
            scale_factor = 32767.0 * prompt_audio_max / (max_abs_value + 1e-8)
        else:
            # Scale to ~ +/- 32767
            scale_factor = 32767.0 * 0.999 / (max_abs_value + 1e-8)

        converted_audio = converted_audio * scale_factor

        # Output as int16 or float32
        if return_int16_wav:
            out_audio_np = converted_audio.detach().cpu().numpy().astype(np.int16)
        else:
            out_audio_np = converted_audio.detach().cpu().numpy().astype(np.float32)

        return out_audio_np, 16000


################################################################################
# Example usage (commented out):
################################################################################

# if __name__ == "__main__":
#     # You can create a HierspeechSynthesizer instance and run inference.
#     # This code is commented out to avoid side effects / CLI usage.
#
#     # Example:
#     # syn = HierspeechSynthesizer(
#     #     hierspeech_ckpt="path/to/hierspeech_ckpt.pth",
#     #     text2w2v_ckpt="path/to/text2w2v_ckpt.pth",
#     #     sr24k_ckpt="path/to/G_340000_24k.pth",
#     #     sr48k_ckpt="path/to/G_100000_48k.pth",
#     #     denoiser_ckpt="path/to/denoiser_ckpt",
#     #     config_hierspeech="path/to/hierspeech_config.json",
#     #     config_text2w2v="path/to/text2w2v_config.json",
#     #     config_sr24="path/to/sr24_config.json",
#     #     config_sr48="path/to/sr48_config.json",
#     #     config_denoiser="path/to/denoiser_config.json",
#     #     device="cuda"
#     # )
#     #
#     # out_wav, sr = syn.infer(
#     #     text="Hello, I'm Hierspeech.",
#     #     prompt_audio_path="path/to/prompt.wav",
#     #     output_sr=48000,
#     #     denoise_ratio=0.8,
#     #     noise_scale_ttv=0.333,
#     #     noise_scale_vc=0.333,
#     #     scale_norm='prompt'
#     # )
#     #
#     # # Save to file:
#     # write("output.wav", sr, out_wav)
#     pass
