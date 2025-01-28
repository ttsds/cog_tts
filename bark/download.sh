mkdir -p checkpoints
if [ ! -d "checkpoints/bark/suno/bark_v0" ]; then
    mkdir -p checkpoints/bark/suno
    git clone https://huggingface.co/suno/bark checkpoints/bark/suno/bark_v0
    rm -rf checkpoints/bark/suno/bark_v0/.git
    rm checkpoints/bark/suno/bark_v0/coarse.pt
    rm checkpoints/bark/suno/bark_v0/fine.pt
    rm checkpoints/bark/suno/bark_v0/text.pt
fi
if [ ! -d "checkpoints/bark_vc_en" ]; then
    git clone https://huggingface.co/GitMylo/bark-voice-cloning checkpoints/bark_vc_en
    rm -rf checkpoints/bark_vc_en/.git
fi
if [ ! -d "checkpoints/bark_vc_pl" ]; then
    git clone https://huggingface.co/Hobis/bark-voice-cloning-polish-HuBERT-quantizer checkpoints/bark_vc_pl
    rm -rf checkpoints/bark_vc_pl/.git
fi
if [ ! -d "checkpoints/bark_vc_de" ]; then
    git clone https://huggingface.co/CountFloyd/bark-voice-cloning-german-HuBERT-quantizer checkpoints/bark_vc_de
    rm -rf checkpoints/bark_vc_de/.git
fi
if [ ! -d "checkpoints/bark_vc_es" ]; then
    git clone https://huggingface.co/Lancer1408/bark-es-tokenizer checkpoints/bark_vc_es
    rm -rf checkpoints/bark_vc_es/.git
fi
if [ ! -d "checkpoints/bark_vc_pt" ]; then
    git clone https://huggingface.co/MadVoyager/bark-voice-cloning-portuguese-HuBERT-quantizer checkpoints/bark_vc_pt
    rm -rf checkpoints/bark_vc_pt/.git
fi
if [ ! -d "checkpoints/bark_vc_ja" ]; then
    git clone https://huggingface.co/junwchina/bark-voice-cloning-japanese-HuBERT-quantizer checkpoints/bark_vc_ja
    rm -rf checkpoints/bark_vc_ja/.git
fi
if [ ! -d "checkpoints/bark_vc_tr" ]; then
    git clone https://huggingface.co/egeadam/bark-voice-cloning-turkish-HuBERT-quantizer checkpoints/bark_vc_tr
    rm -rf checkpoints/bark_vc_tr/.git
fi
if [ ! -d "checkpoints/bark_vc_it" ]; then
    git clone https://huggingface.co/gpwr/bark-it-tokenizer checkpoints/bark_vc_it
    rm -rf checkpoints/bark_vc_it/.git
fi
if [ ! -d "checkpoints/hubert_base" ]; then
    mkdir -p checkpoints/hubert_base
    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -P checkpoints/hubert_base
fi
if [ ! -d "checkpoints/bark_vc_code" ]; then
    git clone https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer.git checkpoints/bark_vc_code
    rm -rf checkpoints/bark_vc_code/.git
fi
if [ ! -d "checkpoints/bark_vc_code/data" ]; then
    mkdir -p checkpoints/bark_vc_code/data/models/hubert
    mv checkpoints/hubert_base/hubert_base_ls960.pt checkpoints/bark_vc_code/data/models/hubert/hubert.pt
    mv checkpoints/bark_vc_en/quantifier_V1_hubert_base_ls960_23.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_en.pth
    mv checkpoints/bark_vc_pl/polish-HuBERT-quantizer_8_epoch.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_pl.pth
    mv checkpoints/bark_vc_de/german-HuBERT-quantizer_14_epoch.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_de.pth
    mv checkpoints/bark_vc_es/es_tokenizer.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_es.pth
    mv checkpoints/bark_vc_pt/portuguese-HuBERT-quantizer_24_epoch.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_pt.pth
    mv checkpoints/bark_vc_ja/japanese-HuBERT-quantizer_24_epoch.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_ja.pth
    mv checkpoints/bark_vc_tr/turkish_model_epoch_14.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_tr.pth
    mv checkpoints/bark_vc_it/it_tokenizer.pth checkpoints/bark_vc_code/data/models/hubert/tokenizer_it.pth
fi