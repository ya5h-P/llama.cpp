# llama.cpp + MoE expert cache

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (build b11046) that runs large Mixture-of-Experts models faster on small GPUs. It keeps the most used experts in a GPU cache, so that most of the expert math runs on the GPU even though the experts do not fit in VRAM.

It was built and tuned for **Gemma 4 26B-A4B on an 8 GB laptop GPU**:

| RTX 4060 Laptop 8 GB, i7-13620H, 64k context, MTP speculative decoding | tokens/s | VRAM |
|---|---|---|
| stock llama.cpp, best `--n-cpu-moe` | ~31 | 7.1 GB |
| this fork, expert cache | **~62-69** | 7.5 GB |

The output is the same model computation: every expert is still computed exactly once, on the GPU or on the CPU. Greedy outputs only differ from stock llama.cpp at near-ties, the same way stock llama.cpp differs between two `--n-cpu-moe` settings.

This is a personal experiment, not an upstream feature. Only Gemma 4 is wired up, and it was only tested on one machine.

## Quick start

Requirements: Linux, an NVIDIA GPU with 8 GB or more and the CUDA toolkit, ~16 GB of RAM, ~15 GB of disk.

```sh
git clone -b expert-cache https://github.com/ya5h-P/llama.cpp.git
cd llama.cpp

# build (native = the GPU in this machine)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server

# download the model (14.3 GB) and the MTP drafter (0.25 GB) into models/
scripts/expert-cache/download-models.sh

# run
scripts/expert-cache/serve-gemma4-26b.sh
```

The server has an OpenAI-compatible API on `http://127.0.0.1:8080/v1` with the model name `gemma-4-26b-a4b-qat`, and a web UI on `http://127.0.0.1:8080`:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"gemma-4-26b-a4b-qat","messages":[{"role":"user","content":"Hello"}]}'
```

It serves one request at a time (`-np 1`); the second request slot would cost VRAM that the cache uses better.

### Settings

The script reads these environment variables, e.g. `CTX=32768 EC_SLOTS=24 scripts/expert-cache/serve-gemma4-26b.sh`:

| variable | default | meaning |
|---|---|---|
| `CTX` | 65536 | context size |
| `N_CPU_MOE` | 26 | layers whose experts stay in system RAM (Gemma 4 26B has 30) |
| `EC_SLOTS` | 28 | GPU cache slots per CPU layer, `0` turns the cache off |
| `EC_BUDGET` | 4 | experts copied to the GPU per token |
| `THREADS` | 10 | CPU threads |
| `MTP` / `DRAFT_N` | 1 / 4 | MTP speculative decoding on/off, and draft length |
| `MODEL_DIR` | `models/` | where the GGUF files are |
| `EXTRA` | | extra `llama-server` arguments |

**VRAM is the main knob.** Each slot costs ~3.35 MB per CPU layer, so 28 slots x 26 layers = 2.4 GB. Keep the total under ~7.5 GB on an 8 GB card: above that, the first request can stall. With a smaller context there is room for more slots (at 16k context, 32 slots gave +16% over 24). With less VRAM, lower `EC_SLOTS`.

The cache itself is controlled by environment variables read by `libllama`, so it also works with `llama-cli`, `llama-bench` and others:

| variable | meaning |
|---|---|
| `LLAMA_EXPERT_CACHE=n` | n GPU slots per MoE layer whose experts are in host memory (unset or 0 = off) |
| `LLAMA_EXPERT_CACHE_BUDGET=n` | max experts copied per token (default 2) |
| `LLAMA_EXPERT_CACHE_PROFILE=file` | initial expert usage, lines of `layer expert count` (`scripts/expert-cache/expert-profile.txt`) |
| `LLAMA_EXPERT_CACHE_CHUNK=kb` | size of the copy chunks (default 256) |
| `LLAMA_EXPERT_CACHE_PREFAULT=1` | map the pages of the memory-mapped experts at startup |
| `GGML_SCHED_NO_EARLY_COPY=1` | disable the CPU/GPU overlap described below |
| `GGML_SCHED_PRELAUNCH=1` | experimental, queue GPU work behind a host flag (no measured gain) |

With `-lv 4`, the server logs the cache hit rate every 256 tokens.

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
4. The two partial results are added, and the expert weights and scales are applied as usual.

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

### What was measured to not matter

- Async cache bookkeeping: the whole update takes ~55 us per step.
- Hiding the GPU launch behind a host flag (`GGML_SCHED_PRELAUNCH`): the launch is already cheap. The CPU experts (~200-270 us per layer, against a ~130 us memory bandwidth floor) are the critical path.
- Pinning threads to performance cores: slower than letting the OS schedule them.
- A smaller `-ub`: frees little VRAM and makes prompt processing 40-65% slower.

The remaining lever is the hit rate, i.e. more slots, i.e. more VRAM.

## Code

| file | change |
|---|---|
| `src/llama-expert-cache.{h,cpp}` | the cache: slots, lookup tables, scores, background copies |
| `src/llama-graph.{h,cpp}` | `build_moe_exps_split`: the split GPU/CPU expert computation |
| `src/models/gemma4.cpp` | passes the cache of each layer to `build_moe_ffn` |
| `src/llama-context.cpp` | updates the cache after each computed batch |
| `ggml/src/ggml-cpu/ggml-cpu.c`, `ggml/src/ggml-cuda/mmvq.cu` | `MUL_MAT_ID` skips `-1` ids |
| `ggml/src/ggml-backend.cpp` | early copy (and the experimental prelaunch) in the scheduler |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | host flag wait kernel for the prelaunch |
| `scripts/expert-cache/` | run and download scripts, startup profile |

## Limitations

- Only Gemma 4 is wired up. Another MoE model needs its `build_moe_ffn` call to pass `&model.layers[il].exp_cache` (see `src/models/gemma4.cpp`), and needs merged gate/up expert tensors, no expert biases, and a GELU or SiLU activation.
- The cache is only used for batches of up to 8 tokens (generation and MTP verification). Prompt processing uses the normal path.
- CUDA only, and only tested on Linux with one GPU.
- Expert layers stored in repacked CPU formats are not cached.

## Upstream

Everything else is stock llama.cpp. See the [upstream README](https://github.com/ggml-org/llama.cpp/blob/master/README.md) for general usage, and [LICENSE](LICENSE) (MIT).
