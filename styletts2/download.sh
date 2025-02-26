mkdir -p checkpoints
if [ ! -d checkpoints/styletts2 ]; then
    git clone https://github.com/yl4579/StyleTTS2.git checkpoints/styletts2
    rm -rf checkpoints/styletts2/.git
fi
if [ ! -d checkpoints/models ]; then
    git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS checkpoints/models
    rm -rf checkpoints/models/.git
fi