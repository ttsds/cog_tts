mkdir -p checkpoints
if [ ! -d checkpoints/speecht5 ]; then
    git clone https://huggingface.co/microsoft/speecht5_tts checkpoints/speecht5
    rm -rf checkpoints/speecht5/.git
fi
if [ ! -d checkpoints/xvector ]; then
    git clone https://huggingface.co/speechbrain/spkrec-xvect-voxceleb checkpoints/xvector
    rm -rf checkpoints/xvector/.git
fi
if [ ! -d checkpoints/speecht5_hifigan ]; then
    git clone https://huggingface.co/microsoft/speecht5_hifigan checkpoints/speecht5_hifigan
    rm -rf checkpoints/speecht5_hifigan/.git
fi