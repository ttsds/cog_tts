mkdir -p checkpoints
if [ ! -d checkpoints/hierspeechpp_libritts960 ]; then
    mkdir -p checkpoints/hierspeechpp_libritts960
    python -m gdown --fuzzy https://drive.google.com/file/d/1pNDRafZ7DU1WALkGIkVyEJFIlcxp4DnE/view?usp=drive_link -O checkpoints/hierspeechpp_libritts960/hierspeechpp_lt960_ckpt.pth
    python -m gdown --fuzzy https://drive.google.com/file/d/1AArVxxMSIr8fbZyq2DoXwv76YKBrs3Hd/view?usp=drive_link -O checkpoints/hierspeechpp_libritts960/config.json
fi
if [ ! -d checkpoints/ttv_libritts_v1 ]; then
    mkdir -p checkpoints/ttv_libritts_v1
    python -m gdown --fuzzy https://drive.google.com/file/d/1JTi3OOhIFFElj1X1u5jBeNa3CPbVS_gk/view?usp=drive_link -O checkpoints/ttv_libritts_v1/ttv_lt960_ckpt.pth
    python -m gdown --fuzzy https://drive.google.com/file/d/1JMYEGHtljxaTodek4e6cRASQEQ4KVTE6/view?usp=drive_link -O checkpoints/ttv_libritts_v1/config.json
fi