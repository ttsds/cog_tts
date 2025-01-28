mkdir -p checkpoints
if [ ! -d "checkpoints/fishspeech" ]; then
    git clone https://huggingface.co/fishaudio/fish-speech-1 checkpoints/fishspeech
    rm -rf checkpoints/fishspeech/.git
    rm checkpoints/fishspeech/fish-speech-v1.1.zip
    rm checkpoints/fishspeech/text2semantic-sft-large-v1.1-4k.pth
    rm checkpoints/fishspeech/text2semantic-sft-medium-v1.1-4k.pth
    rm checkpoints/fishspeech/vits_decoder_v1.1.ckpt
fi