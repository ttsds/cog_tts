mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_large_1_0 ]; then
    git clone https://huggingface.co/parler-tts/parler-tts-large-v1 checkpoints/parlertts_large_1_0
    rm -rf checkpoints/parlertts_large_1_0/.git
fi