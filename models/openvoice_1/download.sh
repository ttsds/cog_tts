mkdir -p checkpoints
if [ ! -d checkpoints/openvoice ]; then
    git clone https://huggingface.co/myshell-ai/OpenVoice checkpoints/openvoice
    rm -rf checkpoints/openvoice/.git
fi