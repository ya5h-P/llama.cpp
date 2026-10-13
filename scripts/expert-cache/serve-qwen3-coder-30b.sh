#!/usr/bin/env bash
# Serve Qwen3-Coder-30B-A3B-Instruct (Unsloth UD-Q3_K_XL) with the expert cache on an 8 GB NVIDIA GPU.
# OpenAI-compatible API on http://127.0.0.1:8080/v1, model name qwen3-coder-30b-a3b.
#
# Settings (env), defaults tuned on an RTX 4060 Laptop 8 GB + i7-13620H with 16 GB of RAM:
#   CTX=65536     context size (the KV cache is ~3.4 GB at 64k; a smaller context leaves room for more slots)
#   N_CPU_MOE=46  layers whose experts stay in system RAM (of 48; 48 no longer fits in 16 GB of RAM)
#   EC_SLOTS=26   GPU cache slots per CPU layer, of 128 experts (0 = cache off)
#   EC_BUDGET=8   experts copied to the GPU per token (the experts are 2 MB, copies are cheap)
#   THREADS=10    CPU threads
#   UBATCH=512    prompt batch size. 1024 makes prompts ~1.5x faster but needs 2 slots less (EC_SLOTS=24),
#                 which makes generation ~10% slower: UBATCH=1024 EC_SLOTS=24 for long one-off prompts
#   NGRAM=1       n-gram speculative decoding: drafts by copying from the context, 38 -> 58 tokens/s on
#                 code edits, ~5% slower on new text (0 = off). Drafts are capped at 7 so the cache stays in use.
#   MODEL_DIR     where the GGUF file is (default: models/ in this repo)
#   EXTRA         extra llama-server arguments
# Measured at 64k: stock llama.cpp ~30 tokens/s, this ~48 (7.5 GB VRAM). Keep VRAM under ~7.5 GB.
# The IQ4_XS quant (16.4 GB) does not fit in 16 GB of RAM next to the cache and gets slower, not faster.
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

export LLAMA_EXPERT_CACHE="${EC_SLOTS:-26}"
export LLAMA_EXPERT_CACHE_BUDGET="${EC_BUDGET:-8}"

exec "$BIN" \
  -m "$DIR/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf" \
  "${SPEC[@]}" \
  --alias qwen3-coder-30b-a3b \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  -ngl 99 --n-cpu-moe "${N_CPU_MOE:-46}" \
  -c "${CTX:-65536}" -fa on -ctk q8_0 -ctv q8_0 -ub "${UBATCH:-512}" \
  -t "${THREADS:-10}" -tb "${THREADS:-10}" -np 1 \
  --jinja \
  ${EXTRA:-}
