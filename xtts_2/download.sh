mkdir -p checkpoints
if [ ! -d checkpoints/xtts ]; then
    git clone https://huggingface.co/coqui/XTTS-v2 checkpoints/xtts
    rm -rf checkpoints/xtts/.git
fi
