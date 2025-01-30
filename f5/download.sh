mkdir -p checkpoints
if [ ! -d "checkpoints/f5" ]; then
    git clone https://huggingface.co/SWivid/F5-TTS checkpoints/f5
    rm -rf checkpoints/f5/.git
    rm checkpoints/f5/F5TTS_Base/model_1200000.pt
fi