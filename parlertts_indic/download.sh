mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_indic ]; then
    git clone https://huggingface.co/ai4bharat/indic-parler-tts checkpoints/parlertts_indic
    rm -rf checkpoints/parlertts_indic/.git
fi