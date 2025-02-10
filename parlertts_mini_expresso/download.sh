mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_mini_expresso ]; then
    git clone https://huggingface.co/parler-tts/parler-tts-mini-expresso checkpoints/parlertts_mini_expresso
    rm -rf checkpoints/parlertts_mini_expresso/.git
fi