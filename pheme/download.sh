mkdir -p checkpoints
if [ ! -d checkpoints/speechtokenizer ]; then
    git clone https://huggingface.co/fnlp/SpeechTokenizer checkpoints/speechtokenizer
    rm -rf checkpoints/speechtokenizer/.git
fi
if [ ! -d checkpoints/uslm ]; then
    git clone https://huggingface.co/fnlp/USLM checkpoints/uslm
    rm -rf checkpoints/uslm/.git
fi
if [ ! -d checkpoints/pheme ]; then
    git clone https://huggingface.co/PolyAI/pheme checkpoints/pheme
    rm -rf checkpoints/pheme/.git
    mkdir -p checkpoints/pheme/t2s
    cp checkpoints/pheme/t2s.bin checkpoints/pheme/t2s/pytorch_model.bin
    cp checkpoints/pheme/config_t2s.json checkpoints/pheme/t2s/config.json
    mkdir -p checkpoints/pheme/s2a
    cp checkpoints/pheme/s2a.ckpt checkpoints/pheme/s2a/
    cp checkpoints/pheme/config_s2a.json checkpoints/pheme/s2a/config.json
fi
if [ ! -d checkpoints/pyannote_embedding ]; then
    git clone git@hf.co:pyannote/embedding checkpoints/pyannote_embedding
    rm -rf checkpoints/pyannote_embedding/.git
fi