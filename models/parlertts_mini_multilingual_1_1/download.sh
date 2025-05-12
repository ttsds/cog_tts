mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_mini_multilingual ]; then
    git clone https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1 checkpoints/parlertts_mini_multilingual
    rm -rf checkpoints/parlertts_mini_multilingual/.git
fi