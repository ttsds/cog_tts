mkdir -p checkpoints
if [ ! -d "checkpoints/_maskgct" ]; then
    git clone https://huggingface.co/amphion/MaskGCT checkpoints/_maskgct
    rm -rf checkpoints/_maskgct/.git
fi
if [ ! -d "checkpoints/whisper-large-v3-turbo" ]; then
    git clone https://huggingface.co/openai/whisper-large-v3-turbo checkpoints/_whisper-large-v3-turbo
    rm -rf checkpoints/_whisper-large-v3-turbo/.git
fi