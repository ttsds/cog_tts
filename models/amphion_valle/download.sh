mkdir -p checkpoints
if [ ! -d "checkpoints/valle_v1_small" ]; then
    git clone https://huggingface.co/amphion/valle_libritts checkpoints/valle_v1_small
    rm -rf checkpoints/valle_v1_small/.git
fi
if [ ! -d "checkpoints/valle_v1_medium" ]; then
    git clone https://huggingface.co/amphion/valle_librilight_6k checkpoints/valle_v1_medium
    rm -rf checkpoints/valle_v1_medium/.git
fi
if [ ! -d "checkpoints/valle_v2" ]; then
    git clone https://huggingface.co/amphion/valle checkpoints/valle_v2
    rm -rf checkpoints/valle_v2/.git
    mkdir checkpoints/valle_v2/tokenizer
    mv checkpoints/valle_v2/SpeechTokenizer.pt checkpoints/valle_v2/tokenizer/SpeechTokenizer.pt
    mv checkpoints/valle_v2/config.json checkpoints/valle_v2/tokenizer/config.json
fi