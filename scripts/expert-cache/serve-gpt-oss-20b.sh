#!/usr/bin/env bash
# Serve gpt-oss-20b (Unsloth Q4_K_M) with the expert cache on an 8 GB NVIDIA GPU.
# OpenAI-compatible API on http://127.0.0.1:8080/v1, model name gpt-oss-20b.
#
# Settings (env), defaults tuned on an RTX 4060 Laptop 8 GB + i7-13620H:
#   CTX=65536     context size
#   N_CPU_MOE=24  layers whose experts stay in system RAM (all 24)
#   EC_SLOTS=14   GPU cache slots per CPU layer, of 32 experts (0 = cache off)
#   EC_BUDGET=2   experts copied to the GPU per token (the experts are 13 MB, copies are expensive)
#   THREADS=10    CPU threads
#   UBATCH=1024   prompt batch size: prompts are PCIe-bound, 1024 is ~1.45x faster than 512 at the same slots
#                 (2048: ~1.7x, but needs a slot less and generation gets ~10% slower)
#   NGRAM=1       n-gram speculative decoding: drafts by copying from the context, +35% on code edits,
#                 ~5% slower on new text (0 = off). Drafts are capped at 7 so the cache stays in use.
#   MODEL_DIR     where the GGUF file is (default: models/ in this repo)
#   EXTRA         extra llama-server arguments, e.g. --chat-template-kwargs '{"reasoning_effort":"low"}'
# Measured at 64k: stock llama.cpp ~45 tokens/s, this ~55-58 (7.48 GB VRAM). Keep VRAM under ~7.5 GB.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DIR="${MODEL_DIR:-$REPO/models}"
BIN="${BIN:-$REPO/build/bin/llama-server}"

SPEC=()
if [[ "${NGRAM:-1}" == 1 ]]; then
  SPEC=(--spec-type ngram-mod --spec-ngram-mod-n-min 1 --spec-ngram-mod-n-max 7
        --spec-ngram-mod-n-match 6 --spec-draft-n-max 7)
fi

export LLAMA_EXPERT_CACHE="${EC_SLOTS:-14}"
export LLAMA_EXPERT_CACHE_BUDGET="${EC_BUDGET:-2}"

exec "$BIN" \
  -m "$DIR/gpt-oss-20b-Q4_K_M.gguf" \
  "${SPEC[@]}" \
  --alias gpt-oss-20b \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  -ngl 99 --n-cpu-moe "${N_CPU_MOE:-24}" \
  -c "${CTX:-65536}" -fa on -ub "${UBATCH:-1024}" \
  -t "${THREADS:-10}" -tb "${THREADS:-10}" -np 1 \
  --jinja \
  ${EXTRA:-}
