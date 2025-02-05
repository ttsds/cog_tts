mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_mini_0_1 ]; then
    git clone https://huggingface.co/parler-tts/parler_tts_mini_v0.1 checkpoints/parlertts_mini_0_1
    rm -rf checkpoints/parlertts_mini_0_1/.git
fi