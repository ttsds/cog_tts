mkdir -p checkpoints
if [ ! -d "checkpoints/gptsovits" ]; then
    git clone https://huggingface.co/lj1995/GPT-SoVITS checkpoints/gptsovits
    (cd checkpoints/gptsovits && git checkout 021ac208db367e69e1982d66d793d0a2af53bfb8) # v1 version
    rm -rf checkpoints/gptsovits/.git
fi
if [ ! -d "checkpoints/speech_paraformer" ]; then
    git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch checkpoints/speech_paraformer
    rm -rf checkpoints/speech_paraformer/.git
fi
if [ ! -d "checkpoints/speech_fsmn_vad" ]; then
    git clone https://www.modelscope.cn/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch checkpoints/speech_fsmn_vad
    rm -rf checkpoints/speech_fsmn_vad/.git
fi
if [ ! -d "checkpoints/punc_ct-transformer" ]; then
    git clone https://www.modelscope.cn/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch checkpoints/punc_ct-transformer
    rm -rf checkpoints/punc_ct-transformer/.git
fi
if [ ! -d "checkpoints/uvr5_weights" ]; then
    git clone https://huggingface.co/Delik/uvr5_weights checkpoints/uvr5_weights
    rm -rf checkpoints/uvr5_weights/.git
fi