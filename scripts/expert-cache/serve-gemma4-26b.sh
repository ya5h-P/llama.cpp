#!/usr/bin/env bash
# Serve Gemma 4 26B-A4B with the expert cache and MTP speculative decoding on an 8 GB NVIDIA GPU.
# OpenAI-compatible API on http://127.0.0.1:8080/v1, model name gemma-4-26b-a4b-qat.
#
# Settings (env), defaults tuned on an RTX 4060 Laptop 8 GB + i7-13620H:
#   CTX=65536     context size
#   N_CPU_MOE=26  layers whose experts stay in system RAM (the other 4 are fully on the GPU)
#   EC_SLOTS=28   GPU cache slots per CPU layer (0 = cache off, stock llama.cpp behavior)
#   EC_BUDGET=4   experts copied to the GPU per token
#   THREADS=10    CPU threads
#   MTP=1         MTP speculative decoding (0 = off), DRAFT_N=4 draft tokens
#   MODEL_DIR     where the GGUF files are (default: models/ in this repo)
#   EXTRA         extra llama-server arguments
# Keep VRAM under ~7.5 GB: above that the first request can stall. With less VRAM, lower EC_SLOTS or raise N_CPU_MOE.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DIR="${MODEL_DIR:-$REPO/models}"
BIN="${BIN:-$REPO/build/bin/llama-server}"

SPEC=()
if [[ "${MTP:-1}" == 1 ]]; then
  SPEC=(--model-draft "$DIR/mtp-gemma-4-26B-A4B-it-Q4_0.gguf"
        --spec-type draft-mtp --spec-draft-n-max "${DRAFT_N:-4}")
fi

export LLAMA_EXPERT_CACHE="${EC_SLOTS:-28}"
export LLAMA_EXPERT_CACHE_BUDGET="${EC_BUDGET:-4}"
export LLAMA_EXPERT_CACHE_PROFILE="${EC_PROFILE:-$HERE/expert-profile.txt}"

exec "$BIN" \
  -m "$DIR/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" \
  "${SPEC[@]}" \
  --alias gemma-4-26b-a4b-qat \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  -ngl 99 --n-cpu-moe "${N_CPU_MOE:-26}" \
  -c "${CTX:-65536}" -fa on -ctk q8_0 -ctv q8_0 \
  -t "${THREADS:-10}" -tb "${THREADS:-10}" -np 1 \
  --jinja \
  ${EXTRA:-}
