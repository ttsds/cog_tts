mkdir -p checkpoints
if [ ! -d checkpoints/whisperspeech ]; then
    git clone https://huggingface.co/WhisperSpeech/WhisperSpeech checkpoints/whisperspeech
    rm -rf checkpoints/whisperspeech/.git
fi
