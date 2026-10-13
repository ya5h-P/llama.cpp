# llama.cpp + MoE expert cache

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (build b11046) that runs large Mixture-of-Experts models faster on small GPUs. It keeps the most used experts in a GPU cache, so that most of the expert math runs on the GPU even though the experts do not fit in VRAM.

It works with any MoE model built through llama.cpp's shared `build_moe_ffn` (see [Limitations](#limitations)), and was tuned on an 8 GB laptop GPU (RTX 4060 Laptop, i7-13620H, 64k context):

| model | stock llama.cpp, best `--n-cpu-moe` | this fork, expert cache | speedup |
|---|---|---|---|
| Gemma 4 26B-A4B (UD-Q4_K_XL, MTP speculative decoding) | ~38-44 tokens/s | **~60-69 tokens/s** | ~1.5x |
| gpt-oss-20b (Q4_K_M, no speculative decoding) | ~45 tokens/s | **~55-58 tokens/s** | ~1.25x |
| Qwen3-Coder-30B-A3B (UD-Q3_K_XL, no speculative decoding) | ~30 tokens/s | **~48 tokens/s** | ~1.6x |

Both sides use the same flags (`-t 10 -tb 10 -np 1`, 64k context) and stay under ~7.5 GB of VRAM; with llama.cpp's default flags, stock Gemma gets ~31 tokens/s. The gain is largest when stock has to leave most layers on the CPU and the experts are small (Qwen3-Coder: 37 of 48 layers on the CPU, 2 MB experts), and smallest with a few large experts (gpt-oss: 32 experts of 13 MB, and 11 of 24 layers already fit on the GPU with stock). gpt-oss and Qwen3-Coder have no MTP drafter, but n-gram speculative decoding (see [Speculative decoding](#speculative-decoding)) brings their code edits to ~81 and ~58 tokens/s.

The output is the same model computation: every expert is still computed exactly once, on the GPU or on the CPU. Greedy outputs only differ from stock llama.cpp at near-ties, the same way stock llama.cpp differs between two `--n-cpu-moe` settings.

This is a personal experiment, not an upstream feature, tested on one machine.

## Quick start

Requirements: Linux, an NVIDIA GPU with 8 GB or more and the CUDA toolkit, ~16 GB of RAM, and 12-15 GB of disk per model.

```sh
git clone -b expert-cache https://github.com/ya5h-P/llama.cpp.git
cd llama.cpp

# build (native = the GPU in this machine)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server

# download a model into models/ and run it, one of:
scripts/expert-cache/download-models.sh gemma         # 14.3 GB + 0.25 GB MTP drafter
scripts/expert-cache/serve-gemma4-26b.sh

scripts/expert-cache/download-models.sh gpt-oss       # 11.6 GB
scripts/expert-cache/serve-gpt-oss-20b.sh

scripts/expert-cache/download-models.sh qwen3-coder   # 13.8 GB
scripts/expert-cache/serve-qwen3-coder-30b.sh
```

The server has an OpenAI-compatible API on `http://127.0.0.1:8080/v1` and a web UI on `http://127.0.0.1:8080`. The model names are `gemma-4-26b-a4b-qat`, `gpt-oss-20b` and `qwen3-coder-30b-a3b`:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"gemma-4-26b-a4b-qat","messages":[{"role":"user","content":"Hello"}]}'
```

All three use port 8080, so run one at a time. Each serves one request at a time (`-np 1`); a second request slot would cost VRAM that the cache uses better.

### Settings

The scripts read these environment variables, e.g. `CTX=32768 EC_SLOTS=24 scripts/expert-cache/serve-gemma4-26b.sh`. The defaults are tuned per model for an 8 GB GPU at 64k context:

| variable | Gemma 4 26B | gpt-oss-20b | Qwen3-Coder 30B | meaning |
|---|---|---|---|---|
| `CTX` | 65536 | 65536 | 65536 | context size |
| `N_CPU_MOE` | 26 of 30 | 24 of 24 | 46 of 48 | layers whose experts stay in system RAM |
| `EC_SLOTS` | 28 of 128 | 14 of 32 | 26 of 128 | GPU cache slots per CPU layer (of the layer's experts), `0` turns the cache off |
| `EC_BUDGET` | 4 | 2 | 8 | experts copied to the GPU per token |
| `THREADS` | 10 | 10 | 10 | CPU threads |
| `MTP` / `DRAFT_N` | 1 / 4 | | | MTP speculative decoding on/off, and draft length |
| `NGRAM` | | 1 | 1 | n-gram speculative decoding on/off |
| `MODEL_DIR` | `models/` | `models/` | `models/` | where the GGUF files are |
| `EXTRA` | | | | extra `llama-server` arguments |

**VRAM is the main knob.** Each slot costs one expert of VRAM per CPU layer: Gemma's 28 slots x 26 layers x 3.35 MB = 2.4 GB, Qwen3-Coder's 26 x 46 x 1.96 MB = 2.3 GB. Keep the total under ~7.5 GB on an 8 GB card: above that, the first request can stall. With a smaller context there is room for more slots (Gemma at 16k context: 32 slots gave +16% over 24). Qwen3-Coder has full attention in all 48 layers, so its context is expensive (~3.4 GB at 64k with the 8-bit KV cache) and a smaller context frees the most there. With less VRAM, lower `EC_SLOTS`.

**RAM matters too.** Every CPU layer has to stay in system RAM. If it does not fit, the experts are read from disk again and again and the speed collapses. That is why Qwen3-Coder uses UD-Q3_K_XL (13.8 GB): on a 16 GB machine, the IQ4_XS file (16.4 GB) with the cache was barely faster than stock llama.cpp (~32 vs ~28 tokens/s), and slower as soon as more layers moved to the CPU.

The cache itself is controlled by environment variables read by `libllama`, so it also works with `llama-cli`, `llama-bench` and others:

| variable | meaning |
|---|---|
| `LLAMA_EXPERT_CACHE=n` | n GPU slots per MoE layer whose experts are in host memory (unset or 0 = off) |
| `LLAMA_EXPERT_CACHE_BUDGET=n` | max experts copied per token (default 2) |
| `LLAMA_EXPERT_CACHE_PROFILE=file` | initial expert usage, lines of `layer expert count` (`scripts/expert-cache/expert-profile.txt`, for Gemma 4) |
| `LLAMA_EXPERT_CACHE_CHUNK=kb` | size of the copy chunks (default 256) |
| `LLAMA_EXPERT_CACHE_PREFAULT=1` | map the pages of the memory-mapped experts at startup |
| `LLAMA_EXPERT_CACHE_PP=0` | prompt processing copies all experts from host memory, also the cached ones |
| `GGML_SCHED_NO_EARLY_COPY=1` | disable the CPU/GPU overlap described below |
| `GGML_SCHED_PRELAUNCH=1` | experimental, queue GPU work behind a host flag (no measured gain) |

With `-lv 4`, the server logs the cache hit rate every 256 tokens.

### Speculative decoding

Speculative decoding guesses a few tokens ahead and lets the model check them all in one pass. The model still picks every token, so the output only changes at near-ties, as above. The cache handles batches of up to 8 tokens, so drafts are capped at 7.

Gemma 4 has a trained MTP drafter, which predicts the next tokens from the model's own hidden state. It is on by default.

gpt-oss and Qwen3-Coder have none, so their scripts use n-gram drafting (`ngram-mod`) instead. It needs no extra model or VRAM: it looks for the last few tokens earlier in the context and proposes what followed them there. That works when the answer repeats the context, which is common for code:

| | new text | code edit (rewrite a pasted file) |
|---|---|---|
| gpt-oss-20b, cache | ~58 tokens/s | ~60 tokens/s |
| gpt-oss-20b, cache + n-gram | ~55 tokens/s | **~81 tokens/s** |
| Qwen3-Coder, cache | ~48 tokens/s | ~38 tokens/s |
| Qwen3-Coder, cache + n-gram | ~46 tokens/s | **~58 tokens/s** |

New text gets a few percent slower, so set `NGRAM=0` for chat-only use.

On Gemma, adding n-gram to MTP made code edits slower (~80 -> ~75 tokens/s): llama.cpp tries n-gram first, and its drafts replace MTP's better ones.

### Other models

Any other MoE GGUF works with `llama-server` directly, for example:

```sh
LLAMA_EXPERT_CACHE=14 LLAMA_EXPERT_CACHE_BUDGET=2 build/bin/llama-server \
  -m models/gpt-oss-20b-Q4_K_M.gguf --jinja -ngl 99 --n-cpu-moe 24 \
  -c 65536 -fa on -t 10 -tb 10 -np 1
```

To tune a new model:

1. Find the lowest `--n-cpu-moe` that fits with stock llama.cpp (cache off).
2. Raise it and add slots while VRAM stays under the limit. Moving more layers to the CPU frees VRAM, and slots usually pay that back (gpt-oss: 13 CPU layers stock, 24 layers x 14 slots with the cache).
3. Pick the budget by expert size: large experts favour a small budget (gpt-oss, 13 MB: 2), small ones a larger one (Qwen3-Coder, 2 MB: 8).
4. Check that the CPU layers fit in RAM: if `free` shows little memory left and the speed drops when adding CPU layers, go back a step or use a smaller quant.

## How it works

### The problem

Gemma 4 26B-A4B has 30 layers. Each layer has 128 experts (small feed-forward networks of ~3.35 MB each at 4-bit), and every token uses only 8 of them per layer. So the model is 14 GB, but a token only reads ~1 GB of it.

On an 8 GB GPU, the usual trick is `--n-cpu-moe N`: attention and the shared weights go to VRAM, and the experts of N layers stay in system RAM, where the CPU computes them. That works, but the CPU layers dominate the time. Profiling stock llama.cpp showed the GPU idle 66% of the time, waiting for the CPU.

### The idea: cache the hot experts on the GPU

Expert usage is far from uniform, and it is sticky: 39% of the experts a token picks at a layer were also picked by the previous token. With 32 GPU slots per layer, a cache that keeps the recently used experts would serve ~70-80% of the picks.

So each CPU layer gets `n` GPU slots holding copies of its hottest experts. For each token:

1. The router picks 8 experts per layer, as usual.
2. Two small lookup tables map each expert to its GPU slot, or to its CPU id. Experts not in the cache get slot `-1` on the GPU side, and cached experts get id `-1` on the CPU side.
3. The GPU computes the cached experts from the slots, and the CPU computes the rest from RAM. `MUL_MAT_ID` was changed to skip `-1` ids (zero output rows), so each expert is computed by exactly one side.
4. The two partial results are added, and the down bias, the expert weights and the scales are applied as usual.

Models with separate gate and up tensors run three `MUL_MAT_ID` per side instead of two. Per-expert gate/up biases (gpt-oss) are added on each side with the real expert ids: the rows of the other side's experts then hold only a bias, and the down `MUL_MAT_ID` turns them back into zero rows. Both sides share the stock activation code, so every activation llama.cpp supports for MoE works.

```
            router -> 8 expert ids
                 |
       +---------+----------+
  slot table            cpu_id table
  (-1 = not cached)     (-1 = cached)
       |                     |
  GPU MUL_MAT_ID        CPU MUL_MAT_ID
  on the cache slots    on the experts in RAM
       |                     |
       +------ add ----------+
                 |
            weighted sum
```

### Keeping the cache fresh

After each step, the cache reads back the expert ids of every layer (one small copy) and updates a decayed usage score per expert (x0.9 per token, so it remembers the last ~10 tokens). Missed experts that score higher than the weakest slot are promoted, up to `EC_BUDGET` copies per token.

Copies run on a background thread while the next tokens are computed. The slot is removed from the lookup table before its copy starts and published only after the copy is done, so a running graph never reads a half-copied expert. `expert-profile.txt` (expert usage counts recorded on sample prompts) fills the cache at startup, so the first tokens do not start cold.

Predicting experts before they are needed was also studied and rejected: copying an expert over PCIe takes longer than the CPU takes to compute it, and the next layer's experts cannot be predicted reliably (38% accuracy). A copy only pays off when the expert is reused ~3 or more times, which is what the decayed score selects for.

### Overlapping the GPU and the CPU

The ggml scheduler runs the graph as a sequence of "splits", one per backend. Normally, before the CPU split runs, the scheduler waits for the whole previous GPU split to finish. Here, that includes the cached experts, so the CPU would sit idle while the GPU computes them.

The scheduler change ("early copy") splits that GPU split at the last node the CPU needs. The CPU inputs are copied as soon as they are ready, the cached experts are launched asynchronously, and the CPU starts right away. The GPU and the CPU now compute their experts at the same time. This alone gave +12% without MTP and +20% with MTP.

### Prompt processing

For large batches (prompts), llama.cpp computes the experts of the CPU layers on the GPU anyway: it uploads the experts the batch uses, which is nearly all of them, once per `-ub` tokens. This is limited by PCIe: for Gemma 4, ~11 GB per 512 tokens at ~10 GB/s.

The experts that are already in the cache slots do not need to cross PCIe again. A scheduler callback copies them from the slots (GPU to GPU) and only the others come from host memory. The output is bit-identical. On a ~11-12k token prompt:

| model | slots / experts | prompt tokens/s before | after |
|---|---|---|---|
| Gemma 4 26B | 28 / 128 | ~430-475 | ~490-535 (+15%) |
| gpt-oss-20b | 14 / 32 | ~540 | ~765 (+41%) |
| Qwen3-Coder 30B | 26 / 128 | ~370-415 | ~440-470 (+15%) |

A larger `-ub` halves the uploads per token but needs a larger compute buffer, i.e. fewer slots. For Gemma, 24 slots with `-ub 1024` (`EC_SLOTS=24 EXTRA="-ub 1024"`) gives ~690-845 prompt tokens/s against ~490-535, and generation gets ~5% slower. That is worth it when the prompts are long (coding agents).

### What was measured to not matter

- Async cache bookkeeping: the whole update takes ~55 us per step.
- Hiding the GPU launch behind a host flag (`GGML_SCHED_PRELAUNCH`): the launch is already cheap. The CPU experts (~200-270 us per layer, against a ~130 us memory bandwidth floor) are the critical path.
- Pinning threads to performance cores: slower than letting the OS schedule them.
- A smaller `-ub`: frees little VRAM and makes prompt processing 40-65% slower (prompts are PCIe-bound, see above).

The remaining lever is the hit rate, i.e. more slots, i.e. more VRAM.

## Code

| file | change |
|---|---|
| `src/llama-expert-cache.{h,cpp}` | the cache: slots, lookup tables, scores, background copies |
| `src/llama-graph.{h,cpp}` | `build_moe_ffn` looks up the layer's cache; `build_moe_exps_split`: the split GPU/CPU expert computation; `build_moe_act`: the activation shared by both paths |
| `src/llama-context.cpp` | passes the per-layer caches to the graph, updates the cache after each computed batch |
| `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cuda/mmvq.cu` | `MUL_MAT_ID` skips `-1` ids |
| `ggml/src/ggml-backend.cpp`, `ggml/include/ggml-backend.h` | early copy (and the experimental prelaunch) in the scheduler; expert copy callback for large batches |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | host flag wait kernel for the prelaunch |
| `scripts/expert-cache/` | serve scripts for Gemma 4, gpt-oss and Qwen3-Coder, download script, Gemma startup profile |

## Limitations

- The model has to build its experts with `build_moe_ffn` (nearly all MoE models in llama.cpp do), with a gated FFN (merged `gate_up` or separate `gate` and `up` tensors). Layers with per-expert gate/up scales, weights applied before the FFN, or LoRA adapters fall back to the normal path. Tested on Gemma 4 26B-A4B, gpt-oss-20b and Qwen3-Coder-30B-A3B.
- The startup profile in `scripts/expert-cache/` is for Gemma 4. Other models start without one (unset `LLAMA_EXPERT_CACHE_PROFILE`) and learn their hot experts as they run.
- The split GPU/CPU computation is only used for batches of up to 8 tokens (generation and the check of up to 7 speculative tokens). Larger batches use the normal op offload path, which only reads the cached experts from the slots.
- CUDA only, and only tested on Linux with one GPU.
- Expert layers stored in repacked CPU formats are not cached.

## Upstream

Everything else is stock llama.cpp. See the [upstream README](https://github.com/ggml-org/llama.cpp/blob/master/README.md) for general usage, and [LICENSE](LICENSE) (MIT).
