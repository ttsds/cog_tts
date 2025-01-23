mkdir -p checkpoints
if [ ! -d "checkpoints/vevo" ]; then
    git clone https://huggingface.co/amphion/Vevo checkpoints/vevo
    rm -rf checkpoints/vevo/.git
fi
if [ ! -d "checkpoints/_whisper-large-v3-turbo" ]; then
    git clone https://huggingface.co/openai/whisper-large-v3-turbo checkpoints/_whisper-large-v3-turbo
    rm -rf checkpoints/_whisper-large-v3-turbo/.git
fi