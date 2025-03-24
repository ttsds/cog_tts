import os
import re
import torch
import numpy as np
import librosa
import sys

sys.path.append("/gptsovits")
sys.path.append("/gptsovits/GPT_SoVITS")

# Required if you use the same text-processing logic:
import LangSegment
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from text import chinese

# cnhubert-based feature extraction
from feature_extractor import cnhubert

# Model definitions (SoVITS + AR GPT)
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
from transformers import AutoModelForMaskedLM, AutoTokenizer

LANGS = [
    "en", "zh", "ja"
]

class DictToAttrRecursive(dict):
    """Recursively convert dict into an object with attribute-style access."""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

def clean_text_inf(text, language_mode, version):
    """Convert input text into phoneme IDs, mapping words to phonemes, etc."""
    phones, word2ph, norm_text = clean_text(text, language_mode, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_feature(text, word2ph, tokenizer, bert_model, device="cuda"):
    """Extract BERT features at phone level."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        # We'll take the last hidden state or second-to-last layers. 
        # E.g., below uses just [-3:-2], but you can adapt as you see fit.
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_features = []
    for i in range(len(word2ph)):
        repeated_feature = res[i].repeat(word2ph[i], 1)
        phone_level_features.append(repeated_feature)
    phone_level_features = torch.cat(phone_level_features, dim=0)
    return phone_level_features.T

def get_phones_and_bert(text, language_mode, version, 
                        ssl_model, bert_model, tokenizer, 
                        device="cuda", is_half=True):
    """
    High-level function that:
      1. Splits text by language if needed,
      2. Cleans it,
      3. Gets phone IDs,
      4. Gets BERT embeddings for Chinese segments only.
    """
    dtype = torch.float16 if is_half else torch.float32
    # "all_zh" means: interpret as if it's fully Chinese, etc.
    # "zh"/"en"/"ja"/"yue"/"ko"/"auto" all come from the user dictionary.

    lang = language_mode
    if lang == "en":
        # Just interpret as English
        LangSegment.setLangfilters(["en"])
        formattext = " ".join(segment["text"] for segment in LangSegment.getTexts(text))
    else:
        # If it's declared as "zh", "ja", "ko", or "yue", just use the text directly
        formattext = text

    # If it has some mixing, handle it (for Chinese with English letters, etc.)
    while "  " in formattext:
        formattext = formattext.replace("  ", " ")

    # Now do standard cleaning
    if lang == "zh":
        # If there's any a-z, you can apply Chinese mixing logic:
        if re.search(r'[a-zA-Z]', formattext):
            formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(
                formattext, "zh", version, ssl_model, bert_model, tokenizer, device, is_half
            )
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, lang, version)
            # For Chinese text, we do BERT
            bert = get_bert_feature(norm_text, word2ph, tokenizer, bert_model, device)
    elif lang == "yue" and re.search(r'[A-Za-z]', formattext):
        # Cantonese with English letters
        formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
        formattext = chinese.mix_text_normalize(formattext)
        return get_phones_and_bert(
            formattext, "yue", version, ssl_model, bert_model, tokenizer, device, is_half
        )
    else:
        phones, word2ph, norm_text = clean_text_inf(formattext, lang, version)
        # For non-Chinese text, we use zero BERT
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=dtype,
        ).to(device)
    return phones, bert, norm_text

def get_spepc(hps, filename, device="cuda", is_half=True):
    """Compute spectrogram for reference audio."""
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    # A simple normalization to avoid clipping
    if maxx > 1:
        audio /= min(2, maxx)

    audio = audio.unsqueeze(0)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    if is_half:
        return spec.half().to(device)
    return spec.to(device)

def merge_short_texts(text_array, threshold=5):
    """Merge short lines of text if needed (heuristic)."""
    if len(text_array) < 2:
        return text_array
    result = []
    tmp = ""
    for txt in text_array:
        tmp += txt
        if len(tmp) >= threshold:
            result.append(tmp)
            tmp = ""
    if tmp:
        if not result:
            result = [tmp]
        else:
            result[-1] += tmp
    return result

def split_by_cn_punctuation(text):
    """Split text by Chinese punctuation like 。. Then re-join by line."""
    # This is just one of the "cut" strategies. 
    # You can define more if you want.
    text = text.strip("\n")
    segments = text.strip("。").split("。")
    segments = [seg for seg in segments if seg.strip()]
    return "\n".join(segments)


class GPTSoVITSPipeline:
    """
    Instantiate with paths to your SoVITS model, GPT model,
    cnhubert path, BERT path, etc. Then call `infer(...)` with 
    your text, reference audio, etc. 
    """

    def __init__(
        self, 
        soVITS_path,
        gpt_path,
        cnhubert_base_path,
        bert_path,
        device="cuda",
        is_half=True
    ):
        """
        Args:
            soVITS_path (str): path to .pth checkpoint for SoVITS
            gpt_path (str): path to .ckpt for the AR GPT
            cnhubert_base_path (str): directory with Chinese-HuBERT
            bert_path (str): directory with a BERT/RoBERTa model
            device (str): "cuda" or "cpu"
            is_half (bool): use half-precision if True
        """
        self.device = device
        self.is_half = is_half
        self.dtype = torch.float16 if is_half else torch.float32

        # 1) Load BERT
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        if is_half:
            self.bert_model = self.bert_model.half().to(device)
        else:
            self.bert_model = self.bert_model.to(device)
        self.bert_model.eval()

        # 2) Load cnhubert
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl_model = cnhubert.get_model()
        if is_half:
            self.ssl_model = self.ssl_model.half().to(device)
        else:
            self.ssl_model = self.ssl_model.to(device)
        self.ssl_model.eval()

        # 3) Load SoVITS
        checkpoint_dict = torch.load(soVITS_path, map_location="cpu")
        self.sovits_config = DictToAttrRecursive(checkpoint_dict["config"])
        # Distinguish v1 vs v2 by embedding shape, etc.
        emb = checkpoint_dict['weight']['enc_p.text_embedding.weight']
        if emb.shape[0] == 322:
            self.sovits_config.model.version = "v1"
        else:
            self.sovits_config.model.version = "v2"

        # Build SoVITS model
        self.sovits_config.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.sovits_config.data.filter_length // 2 + 1,
            self.sovits_config.train.segment_size // self.sovits_config.data.hop_length,
            n_speakers=self.sovits_config.data.n_speakers,
            **self.sovits_config.model
        )
        # If it's not pretrained or something, we might need to remove enc_q, etc.
        if "pretrained" not in soVITS_path:
            # Some SoVITS merges do remove enc_q; adapt as needed
            if hasattr(self.vq_model, "enc_q"):
                del self.vq_model.enc_q

        # Load weights
        self.vq_model.load_state_dict(checkpoint_dict["weight"], strict=False)
        if is_half:
            self.vq_model = self.vq_model.half().to(device)
        else:
            self.vq_model = self.vq_model.to(device)
        self.vq_model.eval()

        # 4) Load GPT
        gpt_ckpt = torch.load(gpt_path, map_location="cpu")
        self.gpt_config = gpt_ckpt["config"]
        self.max_sec = self.gpt_config["data"]["max_sec"]
        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "dummy", is_train=False)
        self.t2s_model.load_state_dict(gpt_ckpt["weight"])
        if is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(device)
        self.t2s_model.eval()

        # Just for clarity
        print("Loaded SoVITS version:", self.sovits_config.model.version)
        total_params = sum(p.numel() for p in self.t2s_model.parameters())
        print("GPT model param count = {:.2f} M".format(total_params / 1e6))

    def infer(
        self,
        ref_wav_path,
        prompt_text,
        prompt_language,
        text,
        text_language,
        how_to_cut="none",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        ref_free=False,
        speed=1.0,
        references=None
    ):
        """
        Given a reference audio (for style), a reference text (optional), 
        and a target text, run GPT -> SoVITS to produce TTS audio.

        Args:
            ref_wav_path (str): Path to reference audio (3-10s recommended)
            prompt_text (str): Reference text for the reference audio
            prompt_language (str): e.g. "en", "zh", "auto", ...
            text (str): The text to be spoken
            text_language (str): e.g. "en", "zh", "auto", ...
            how_to_cut (str): e.g. "none", "cut_zh_period", ...
            top_k, top_p, temperature: GPT sampling parameters
            ref_free (bool): if True, ignore prompt_text entirely
            speed (float): speed adjustment
            references (list): optional list of extra reference audio paths 
                               (for multi-speaker averaging)

        Returns:
            (wav, sr): int16 numpy audio array, sample rate
        """
        sr_out = int(self.sovits_config.data.sampling_rate)
        if not ref_free and (not ref_wav_path or not os.path.exists(ref_wav_path)):
            raise ValueError("ref_free=False, but no valid ref_wav_path given.")

        # Possibly cut the text first
        if how_to_cut == "cut_zh_period":
            text = split_by_cn_punctuation(text)
        # etc. (define more if you want)

        # Split by newlines => handle each line separately
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        lines = merge_short_texts(lines, threshold=5)

        # ~0.3s pause appended between lines:
        zero_pause = np.zeros(int(sr_out * 0.3), dtype=np.float32)
        output_audio = []

        # 1) Prepare the "prompt" (a.k.a. reference semantic codes):
        if not ref_free:
            # Extract reference semantic from reference wav
            # a) load ref wav 16k
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k_torch = torch.from_numpy(wav16k).to(self.device, dtype=self.dtype)

            # Add 0.3s pause in the reference, so the HuBERT features are not truncated
            pad = torch.zeros(int(0.3*16000), dtype=self.dtype, device=self.device)
            wav16k_torch = torch.cat([wav16k_torch, pad], dim=0)

            with torch.no_grad():
                ssl_content = self.ssl_model.model(wav16k_torch.unsqueeze(0))[
                    "last_hidden_state"
                ].transpose(1, 2)
                codes = self.vq_model.extract_latent(ssl_content)
                prompt_semantic = codes[0, 0]  # shape [T]
                # We keep it as shape [1, T]
                prompt_semantic = prompt_semantic.unsqueeze(0)
        else:
            prompt_semantic = None

        # Convert prompt_text => phones/bert if ref_free=False
        if (not ref_free) and prompt_text:
            phones_prompt, bert_prompt, _ = get_phones_and_bert(
                prompt_text, 
                prompt_language, 
                version=self.sovits_config.model.version,
                ssl_model=self.ssl_model,
                bert_model=self.bert_model,
                tokenizer=self.tokenizer,
                device=self.device,
                is_half=self.is_half
            )
        else:
            phones_prompt = []
            bert_prompt = torch.zeros((1024, 0), dtype=self.dtype, device=self.device)

        # 2) For each line, run GPT -> decode -> append
        for line_idx, one_line in enumerate(lines):
            # Clean & convert
            phones_line, bert_line, norm_text = get_phones_and_bert(
                one_line, 
                text_language, 
                version=self.sovits_config.model.version,
                ssl_model=self.ssl_model,
                bert_model=self.bert_model,
                tokenizer=self.tokenizer,
                device=self.device,
                is_half=self.is_half
            )

            # Merge prompt part if not ref_free
            if not ref_free:
                # merged phones
                all_phones = phones_prompt + phones_line
                # merged BERT
                all_bert = torch.cat([bert_prompt, bert_line], dim=1)
            else:
                all_phones = phones_line
                all_bert = bert_line

            phone_ids = torch.LongTensor(all_phones).unsqueeze(0).to(self.device)
            bert_input = all_bert.unsqueeze(0).to(self.device)
            phone_len = torch.tensor([phone_ids.shape[-1]]).to(self.device)

            # 3) GPT inference
            with torch.no_grad():
                pred_semantic, used_length = self.t2s_model.model.infer_panel(
                    phone_ids,
                    phone_len,
                    prompt_semantic,  # None if ref_free
                    bert_input,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=50 * self.max_sec  # 50 is a default frame rate
                )
                # Keep the newly generated part only if you merged prompt
                if not ref_free:
                    pred_semantic = pred_semantic[:, -used_length:].unsqueeze(0)
                else:
                    pred_semantic = pred_semantic.unsqueeze(0)

            # 4) SoVITS decode
            if references and len(references) > 0:
                ref_specs = []
                for rpath in references:
                    spec = get_spepc(self.sovits_config, rpath, device=self.device, is_half=self.is_half)
                    ref_specs.append(spec)
            else:
                # fallback to single reference if none provided
                if ref_wav_path and os.path.exists(ref_wav_path):
                    ref_specs = [get_spepc(self.sovits_config, ref_wav_path, self.device, self.is_half)]
                else:
                    ref_specs = []

            if len(ref_specs) == 0:
                raise ValueError("No reference specs found. Provide at least one reference audio.")

            audio_out = self.vq_model.decode(
                pred_semantic, 
                torch.LongTensor(phones_line).unsqueeze(0).to(self.device), 
                ref_specs,
                speed=speed
            )
            audio_np = audio_out.detach().cpu().numpy()[0, 0]  # shape [T]

            # Avoid integer clipping
            max_amp = np.abs(audio_np).max()
            if max_amp > 1.0:
                audio_np /= max_amp

            output_audio.append(audio_np.astype(np.float32))
            # small pause
            output_audio.append(zero_pause)

        # Concatenate all lines
        final_wav = np.concatenate(output_audio, axis=0)

        # Convert to int16
        final_wav_int16 = (final_wav * 32767.0).astype(np.int16)
        return final_wav_int16, sr_out