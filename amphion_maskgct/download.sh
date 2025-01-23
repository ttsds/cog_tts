mkdir -p checkpoints
if [ ! -d "checkpoints/maskgct" ]; then
    git clone https://huggingface.co/amphion/MaskGCT checkpoints/maskgct
    rm -rf checkpoints/maskgct/.git
fi
if [ ! -d "checkpoints/whisper-large-v3-turbo" ]; then
    git clone https://huggingface.co/openai/whisper-large-v3-turbo checkpoints/whisper-large-v3-turbo
    rm -rf checkpoints/whisper-large-v3-turbo/.git
fi