mkdir -p checkpoints
if [ ! -d "checkpoints/_maskgct" ]; then
    git clone https://huggingface.co/amphion/MaskGCT checkpoints/maskgct
    rm -rf checkpoints/_maskgct/.git
fi