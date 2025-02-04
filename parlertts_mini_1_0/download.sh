mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_mini_0_1 ]; then
    git clone https://huggingface.co/parler-tts/parler-tts-mini-v1 checkpoints/parlertts_mini_1_0
    rm -rf checkpoints/parlertts_mini_1_0/.git
fi