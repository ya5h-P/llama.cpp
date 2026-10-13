#!/usr/bin/env bash
# Download Gemma 4 26B-A4B (Unsloth QAT UD-Q4_K_XL, 14.3 GB) and its MTP drafter (0.25 GB) into models/
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${MODEL_DIR:-$REPO/models}"
URL=https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/resolve/main

mkdir -p "$DIR"
cd "$DIR"
# resumable: partial downloads are kept as .part, finished files are skipped
get() {
  local name="${1##*/}"
  if [[ -f "$name" ]]; then
    echo "$name: already downloaded"
    return
  fi
  curl -L -C - --fail -o "$name.part" "$URL/$1"
  mv "$name.part" "$name"
}

get gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
get MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf
ls -l "$DIR"/*.gguf
