mkdir -p checkpoints
if [ ! -d "checkpoints/e2" ]; then
    git clone https://huggingface.co/SWivid/E2-TTS checkpoints/e2
    rm -rf checkpoints/e2/.git
    rm checkpoints/e2/E2TTS_Base/model_1200000.pt
fi