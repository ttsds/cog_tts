mkdir -p checkpoints
if [ ! -d checkpoints/hierspeechpp_libritts460 ]; then
    mkdir -p checkpoints/hierspeechpp_libritts460
    python -m gdown --fuzzy https://drive.google.com/file/d/1JxjU40OZfkICqjP7gD2Qn40EiVEzubDo/view?usp=sharing -O checkpoints/hierspeechpp_libritts460/hierspeechpp_lt460_ckpt.pth
    python -m gdown --fuzzy https://drive.google.com/file/d/1xcbruEoaOiDLm4fgyVb7CSf3oV-SmNlN/view?usp=drive_link -O checkpoints/hierspeechpp_libritts460/config.json
fi
if [ ! -d checkpoints/ttv_libritts_v1 ]; then
    mkdir -p checkpoints/ttv_libritts_v1
    python -m gdown --fuzzy https://drive.google.com/file/d/1JTi3OOhIFFElj1X1u5jBeNa3CPbVS_gk/view?usp=drive_link -O checkpoints/ttv_libritts_v1/ttv_lt960_ckpt.pth
    python -m gdown --fuzzy https://drive.google.com/file/d/1JMYEGHtljxaTodek4e6cRASQEQ4KVTE6/view?usp=drive_link -O checkpoints/ttv_libritts_v1/config.json
fi