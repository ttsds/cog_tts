mkdir -p checkpoints
if [ ! -d checkpoints/openvoice ]; then
    git clone https://huggingface.co/myshell-ai/OpenVoiceV2 checkpoints/openvoice
    rm -rf checkpoints/openvoice/.git
fi
if [ ! -d checkpoints/openvoice_en ]; then
    git clone https://huggingface.co/myshell-ai/MeloTTS-English checkpoints/openvoice_en
    rm -rf checkpoints/openvoice_en/.git
fi
if [ ! -d checkpoints/openvoice_fr ]; then
    git clone https://huggingface.co/myshell-ai/MeloTTS-French checkpoints/openvoice_fr
    rm -rf checkpoints/openvoice_fr/.git
fi
if [ ! -d checkpoints/openvoice_es ]; then
    git clone https://huggingface.co/myshell-ai/MeloTTS-Spanish checkpoints/openvoice_es
    rm -rf checkpoints/openvoice_es/.git
fi
if [ ! -d checkpoints/openvoice_ja ]; then
    git clone https://huggingface.co/myshell-ai/MeloTTS-Japanese checkpoints/openvoice_ja
    rm -rf checkpoints/openvoice_ja/.git
fi
if [ ! -d checkpoints/openvoice_ko ]; then
    git clone https://huggingface.co/myshell-ai/MeloTTS-Korean checkpoints/openvoice_ko
    rm -rf checkpoints/openvoice_ko/.git
fi
if [ ! -d checkpoints/openvoice_zh ]; then
    git clone https://huggingface.co/myshell-ai/MeloTTS-Chinese checkpoints/openvoice_zh
    rm -rf checkpoints/openvoice_zh/.git
fi