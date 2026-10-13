#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;
struct llama_model;
struct llm_moe_expert_cache;

// keeps the most used experts of host-memory MoE layers in GPU slots
// the graph runs the cached experts on the GPU and the other experts on the CPU (see build_moe_exps_split)
//
// env:
//   LLAMA_EXPERT_CACHE=n          number of GPU slots per MoE layer with experts in host memory (0 = off)
//   LLAMA_EXPERT_CACHE_BUDGET=n   max experts copied to the GPU per decoded token (default 2)
//   LLAMA_EXPERT_CACHE_PROFILE=f  initial expert usage, lines of "layer expert count"
//   LLAMA_EXPERT_CACHE_PREFAULT=1 map the pages of memory-mapped experts at startup
struct llama_expert_cache {
    static std::unique_ptr<llama_expert_cache> init(llama_model & model);

    ~llama_expert_cache();

    // call after the graph is computed and the backends are synchronized
    void update(int n_tokens);

    // [n_layer] cache of each layer, nullptr for layers that are not cached; build_moe_ffn looks the layer up here
    std::vector<const llm_moe_expert_cache *> layer_caches;

private:
    struct layer {
        int il;

        struct tensor {
            ggml_tensor *  src;  // host
            ggml_tensor ** dst;  // field of llm_moe_expert_cache
            const char *   name;
        };
        std::vector<tensor> exps;  // expert weights (gate_up or gate+up, and down): copied per slot
        std::vector<tensor> fixed; // small per-expert tensors (biases, scales): copied once, in full

        llm_moe_expert_cache * cache;

        std::vector<int32_t> slot;      // [n_expert] slot of the expert, -1 if not cached
        std::vector<int32_t> cpu_id;    // [n_expert] -1 if cached, else the expert id
        std::vector<int32_t> expert;    // [n_slot] expert in the slot, -1 if empty or loading
        std::vector<int32_t> loading;   // [n_slot] expert being copied into the slot, -1 if none
        std::vector<int32_t> load_slot; // [n_expert] slot the expert is being copied into, -1 if none
        std::vector<float>   score;     // [n_expert] decayed usage

        bool dirty = false;
    };

    struct copy_job {
        int li;
        int s;
        int e;
    };

    void copy_expert(const layer & l, int s, int e);
    void publish(layer & l, int s, int e);
    void upload_tables();
    void worker_loop();

    std::vector<layer> layers;

    int n_slot   = 0;
    int n_expert = 0;
    int n_used   = 0;
    int budget   = 2;

    float decay = 0.9f;

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer * buf = nullptr;
    ggml_tensor         * ids    = nullptr; // [n_used*LLAMA_EXPERT_CACHE_MAX_TOKENS, n_layer] I32
    ggml_tensor         * tables = nullptr; // [n_expert, 2, n_layer] I32: slot and cpu_id of each layer

    std::vector<int32_t> ids_host;
    std::vector<int32_t> tables_host;

    size_t chunk = 256*1024;

    // copies run on a worker thread while the next tokens are computed
    std::thread             worker;
    std::mutex              mtx;
    std::condition_variable cv;
    std::vector<copy_job>   jobs;
    bool                    busy = false;
    bool                    stop = false;

    // stats
    int64_t n_tok    = 0;
    int64_t n_pick   = 0;
    int64_t n_hit    = 0;
    int64_t n_promo  = 0;
    int64_t n_report = 0;

    int64_t n_update = 0;
    int64_t t_wait   = 0;
    int64_t t_read   = 0;
    int64_t t_upload = 0;
    int64_t t_total  = 0;
};
