mkdir -p checkpoints
if [ ! -d "checkpoints/metavoice" ]; then
    git clone https://huggingface.co/metavoiceio/metavoice-1B-v0.1 checkpoints/metavoice
    rm -rf checkpoints/metavoice/.git
fi