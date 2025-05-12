mkdir -p checkpoints
if [ ! -d "checkpoints/_vevo" ]; then
    git clone https://huggingface.co/amphion/Vevo checkpoints/_vevo
    rm -rf checkpoints/_vevo/.git
fi
if [ ! -d "checkpoints/hubert" ]; then
    mkdir -p checkpoints/hubert
    wget https://download.pytorch.org/torchaudio/models/hubert_fairseq_large_ll60k.pth -O checkpoints/hubert/hubert_fairseq_large_ll60k.pth 
fi