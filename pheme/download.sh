mkdir -p checkpoints
if [ ! -d checkpoints/speechtokenizer ]; then
    git clone https://huggingface.co/fnlp/SpeechTokenizer checkpoints/speechtokenizer
    rm -rf checkpoints/speechtokenizer/.git
fi