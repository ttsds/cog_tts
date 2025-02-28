mkdir -p checkpoints
if [ ! -d checkpoints/voicecraft_giga330m ]; then
    git clone https://huggingface.co/pyp1/VoiceCraft_giga330M checkpoints/voicecraft_giga330m
    rm -rf checkpoints/voicecraft_giga330m/.git
fi
if [ ! -d checkpoints/voicecraft_giga830m ]; then
    git clone https://huggingface.co/pyp1/VoiceCraft_giga830M checkpoints/voicecraft_giga830m
    rm -rf checkpoints/voicecraft_giga830m/.git
fi
if [ ! -d checkpoints/voicecraft_giga_libritts_330m ]; then
    git clone https://huggingface.co/pyp1/VoiceCraft_gigaHalfLibri330M_TTSEnhanced_max16s checkpoints/voicecraft_giga_libritts_330m
    rm -rf checkpoints/voicecraft_giga_libritts_330m/.git
fi
if [ ! -d checkpoints/voicecraft_giga330m_tts_enhanced ]; then
    git clone https://huggingface.co/pyp1/VoiceCraft_330M_TTSEnhanced checkpoints/voicecraft_giga330m_tts_enhanced
    rm -rf checkpoints/voicecraft_giga330m_tts_enhanced/.git
fi
if [ ! -d checkpoints/voicecraft_giga830m_tts_enhanced ]; then
    git clone https://huggingface.co/pyp1/VoiceCraft_830M_TTSEnhanced checkpoints/voicecraft_giga830m_tts_enhanced
    rm -rf checkpoints/voicecraft_giga830m_tts_enhanced/.git
fi
if [ ! -d checkpoints/encodec ]; then
    mkdir -p checkpoints/encodec
    wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th -O checkpoints/encodec/encodec_4cb2048_giga.th
fi
