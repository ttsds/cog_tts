mkdir -p checkpoints
if [ ! -d checkpoints/hierspeechpp_eng_kor ]; then
    mkdir -p checkpoints/hierspeechpp_eng_kor
    python -m gdown --fuzzy https://drive.google.com/file/d/1_rYQZ7YEIxJbXEpJ3Vf4NXXRxLbcfys9/view?usp=drive_link -O checkpoints/hierspeechpp_eng_kor/hierspeechpp_v1_ckpt.pth
    python -m gdown --fuzzy https://drive.google.com/file/d/1qp4rmTdecnui_DGJbBkrahqp5JCsQ1KZ/view?usp=drive_link -O checkpoints/hierspeechpp_eng_kor/config.json
fi
if [ ! -d checkpoints/ttv_libritts_v1 ]; then
    mkdir -p checkpoints/ttv_libritts_v1
    python -m gdown --fuzzy https://drive.google.com/file/d/1JTi3OOhIFFElj1X1u5jBeNa3CPbVS_gk/view?usp=drive_link -O checkpoints/ttv_libritts_v1/ttv_lt960_ckpt.pth
    python -m gdown --fuzzy https://drive.google.com/file/d/1JMYEGHtljxaTodek4e6cRASQEQ4KVTE6/view?usp=drive_link -O checkpoints/ttv_libritts_v1/config.json
fi