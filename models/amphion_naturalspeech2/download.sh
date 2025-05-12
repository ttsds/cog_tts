mkdir -p checkpoints
if [ ! -d "checkpoints/naturalspeech2" ]; then
    git clone https://huggingface.co/amphion/naturalspeech2_libritts checkpoints/naturalspeech2
    rm -rf checkpoints/naturalspeech2/.git
fi