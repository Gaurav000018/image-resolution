#!/usr/bin/env bash
# Download DIV2K from Hugging Face mirrors (the official ETH server is very slow).
#   bash download_data.sh [N_TRAIN]   (default 400; this mirror hosts images 0001-0400)
set -e
N=${1:-400}
cd "$(dirname "$0")"
mkdir -p data/raw data/DIV2K_train_HR

# validation set (100 images, ~450 MB) — official zip mirrored by deepinv
if [ ! -d data/DIV2K_valid_HR ]; then
  curl -sSL --retry 5 -o data/raw/DIV2K_valid_HR.zip \
    https://huggingface.co/datasets/deepinv/DIV2K/resolve/main/DIV2K_valid_HR.zip
  echo "valid zip downloaded"
fi

# training set: individual PNGs, 8 parallel connections, skips files already present
seq -f "%04g" 1 "$N" | xargs -P 8 -I{} sh -c '
  f=data/DIV2K_train_HR/{}.png
  [ -s "$f" ] || curl -fsSL --retry 5 -o "$f" \
    https://huggingface.co/datasets/mAiello00/DIV2K/resolve/main/DIV2K_train_HR/{}.png'
echo "train images: $(ls data/DIV2K_train_HR | wc -l)"
