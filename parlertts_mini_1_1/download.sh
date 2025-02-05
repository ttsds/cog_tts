mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_mini_1_1 ]; then
    git clone https://huggingface.co/parler-tts/parler-tts-mini-v1.1 checkpoints/parlertts_mini_1_1
    rm -rf checkpoints/parlertts_mini_1_1/.git
fi
