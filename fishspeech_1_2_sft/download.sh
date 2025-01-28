mkdir -p checkpoints
if [ ! -d "checkpoints/fishspeech" ]; then
    git clone https://huggingface.co/fishaudio/fish-speech-1.2-sft checkpoints/fishspeech
    rm -rf checkpoints/fishspeech/.git
fi