mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_tiny_1_0 ]; then
    git clone https://huggingface.co/parler-tts/parler-tts-tiny-v1 checkpoints/parlertts_tiny_1_0
    rm -rf checkpoints/parlertts_tiny_1_0/.git
fi