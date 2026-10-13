#!/usr/bin/env bash
# Download a model into models/:
#   download-models.sh [gemma]   Gemma 4 26B-A4B (Unsloth QAT UD-Q4_K_XL, 14.3 GB) and its MTP drafter (0.25 GB)
#   download-models.sh gpt-oss   gpt-oss-20b (Unsloth Q4_K_M, 11.6 GB)
#   download-models.sh qwen3-coder  Qwen3-Coder-30B-A3B-Instruct (Unsloth UD-Q3_K_XL, 13.8 GB)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${MODEL_DIR:-$REPO/models}"
HF=https://huggingface.co/unsloth

mkdir -p "$DIR"
cd "$DIR"
# resumable: partial downloads are kept as .part, finished files are skipped
get() {
  local name="${1##*/}"
  if [[ -f "$name" ]]; then
    echo "$name: already downloaded"
    return
  fi
  curl -L -C - --fail -o "$name.part" "$HF/$1"
  mv "$name.part" "$name"
}

case "${1:-gemma}" in
  gemma)
    get gemma-4-26B-A4B-it-qat-GGUF/resolve/main/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
    get gemma-4-26B-A4B-it-qat-GGUF/resolve/main/MTP/mtp-gemma-4-26B-A4B-it-Q4_0.gguf ;;
  gpt-oss)
    get gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf ;;
  qwen3-coder)
    get Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf ;;
  *)
    echo "unknown model '$1' (gemma, gpt-oss, qwen3-coder)" >&2
    exit 1 ;;
esac
ls -l "$DIR"/*.gguf
