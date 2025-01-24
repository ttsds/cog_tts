mkdir -p checkpoints
if [ ! -d "checkpoints/bark" ]; then
    git clone https://huggingface.co/suno/bark checkpoints/bark
    rm -rf checkpoints/bark/.git
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