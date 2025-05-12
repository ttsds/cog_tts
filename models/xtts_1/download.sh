mkdir -p checkpoints
if [ ! -d checkpoints/xtts ]; then
    git clone https://huggingface.co/coqui/XTTS-v1 checkpoints/xtts
    # checkout v1.1.2 branch
    cd checkpoints/xtts
    git checkout v1.1.2
    rm -rf .git
fi
