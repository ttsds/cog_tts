mkdir -p checkpoints
if [ ! -d checkpoints/parlertts_indic_pt ]; then
    git clone https://huggingface.co/ai4bharat/indic-parler-tts-pretrained checkpoints/parlertts_indic_pt
    rm -rf checkpoints/parlertts_indic_pt/.git
fi