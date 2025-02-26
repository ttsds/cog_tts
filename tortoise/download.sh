mkdir -p checkpoints
if [ ! -d checkpoints/tortoise ]; then
    git clone https://huggingface.co/jbetker/tortoise-tts-v2 checkpoints/tortoise
    rm -rf checkpoints/tortoise/.git
fi
if [ ! -d checkpoints/wav2vec2-large-robust-ft-libritts-voxpopuli ]; then
    git clone https://huggingface.co/jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli checkpoints/wav2vec2-large-robust-ft-libritts-voxpopuli
    rm -rf checkpoints/wav2vec2-large-robust-ft-libritts-voxpopuli/.git
fi
if [ ! -d checkpoints/wav2vec2-large-960h ]; then
    git clone https://huggingface.co/facebook/wav2vec2-large-960h checkpoints/wav2vec2-large-960h
    rm -rf checkpoints/wav2vec2-large-960h/.git
fi
if [ ! -d checkpoints/tacotron_symbols ]; then
    git clone https://huggingface.co/jbetker/tacotron_symbols checkpoints/tacotron_symbols
    rm -rf checkpoints/tacotron_symbols/.git
fi