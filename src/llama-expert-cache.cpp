#include "llama-expert-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// map the pages of a memory-mapped tensor now, else the first use of each expert pays ~800 page faults
static void prefault(const ggml_tensor * t) {
#if defined(__linux__) && defined(MADV_POPULATE_READ)
    const uintptr_t page  = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t begin = (uintptr_t) t->data & ~(page - 1);
    const uintptr_t end   = (uintptr_t) t->data + ggml_nbytes(t);
    madvise((void *) begin, end - begin, MADV_POPULATE_READ);
#else
    GGML_UNUSED(t);
#endif
}

static bool is_plain_host_buffer(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
        return false;
    }
    // repacked layouts can not be copied to the GPU as-is
    const char * name = ggml_backend_buffer_name(t->buffer);
    return strcmp(name, "CPU") == 0 || strcmp(name, "CPU_Mapped") == 0;
}

std::unique_ptr<llama_expert_cache> llama_expert_cache::init(llama_model & model) {
    const char * env = getenv("LLAMA_EXPERT_CACHE");
    const int n_slot = env ? atoi(env) : 0;
    if (n_slot <= 0) {
        return nullptr;
    }

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev = ggml_backend_dev_get(i);
            break;
        }
    }
    if (dev == nullptr) {
        LLAMA_LOG_WARN("%s: no GPU device, expert cache disabled\n", __func__);
        return nullptr;
    }

    auto ec = std::unique_ptr<llama_expert_cache>(new llama_expert_cache());

    for (int il = 0; il < (int) model.layers.size(); ++il) {
        auto & l = model.layers[il];
        auto & c = l.exp_cache;

        layer cl;
        cl.il    = il;
        cl.cache = &c;
        if (l.ffn_gate_up_exps) {
            cl.exps = { { l.ffn_gate_up_exps, &c.gate_up, "gate_up" } };
        } else if (l.ffn_gate_exps && l.ffn_up_exps) {
            cl.exps = { { l.ffn_gate_exps, &c.gate, "gate" }, { l.ffn_up_exps, &c.up, "up" } };
        } else {
            continue;
        }
        if (l.ffn_down_exps == nullptr) {
            continue;
        }
        cl.exps.push_back({ l.ffn_down_exps, &c.down, "down" });

        bool ok = true;
        for (const auto & t : cl.exps) {
            ok = ok && is_plain_host_buffer(t.src) && t.src->ne[2] == cl.exps[0].src->ne[2];
        }
        if (!ok) {
            continue;
        }

        // the GPU side needs its own copy of these, else they would be copied from host memory for each layer
        const layer::tensor fixed[] = {
            { l.ffn_down_exps_s,    &c.down_s,    "down_s"    },
            { l.ffn_gate_up_exps_b, &c.gate_up_b, "gate_up_b" },
            { l.ffn_gate_exps_b,    &c.gate_b,    "gate_b"    },
            { l.ffn_up_exps_b,      &c.up_b,      "up_b"      },
            { l.ffn_down_exps_b,    &c.down_b,    "down_b"    },
        };
        for (const auto & t : fixed) {
            if (is_plain_host_buffer(t.src)) {
                cl.fixed.push_back(t);
            }
        }
        ec->layers.push_back(cl);
    }

    if (ec->layers.empty()) {
        LLAMA_LOG_WARN("%s: no MoE layer with experts in host memory, expert cache disabled\n", __func__);
        return nullptr;
    }

    ec->n_expert = (int) ec->layers[0].exps[0].src->ne[2];
    ec->n_slot   = std::min(n_slot, ec->n_expert);
    ec->n_used   = (int) model.hparams.n_expert_used(ec->layers[0].il);

    if (const char * b = getenv("LLAMA_EXPERT_CACHE_BUDGET")) {
        ec->budget = std::max(0, atoi(b));
    }
    if (const char * c = getenv("LLAMA_EXPERT_CACHE_CHUNK")) {
        ec->chunk = std::max<size_t>(4096, (size_t) atoll(c)*1024);
    }

    ggml_init_params params = {
        /*.mem_size   =*/ (ec->layers.size()*12 + 2)*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ec->ctx = ggml_init(params);

    const int64_t n_ids = (int64_t) ec->n_used*LLAMA_EXPERT_CACHE_MAX_TOKENS;
    ec->ids = ggml_new_tensor_2d(ec->ctx, GGML_TYPE_I32, n_ids, ec->layers.size());
    ggml_set_name(ec->ids, "ffn_exps_cache_ids");

    // slot and cpu_id tables of all layers, uploaded with one copy
    ec->tables = ggml_new_tensor_3d(ec->ctx, GGML_TYPE_I32, ec->n_expert, 2, ec->layers.size());
    ggml_set_name(ec->tables, "ffn_exps_cache_tables");

    for (size_t li = 0; li < ec->layers.size(); ++li) {
        auto & cl = ec->layers[li];
        auto & c  = *cl.cache;
        for (const auto & t : cl.exps) {
            GGML_ASSERT(t.src->ne[2] == ec->n_expert);
            *t.dst = ggml_new_tensor_3d(ec->ctx, t.src->type, t.src->ne[0], t.src->ne[1], ec->n_slot);
            ggml_format_name(*t.dst, "blk.%d.ffn_%s_exps_cache", cl.il, t.name);
        }
        for (const auto & t : cl.fixed) {
            *t.dst = ggml_dup_tensor(ec->ctx, t.src);
            ggml_format_name(*t.dst, "blk.%d.ffn_%s_exps_cache", cl.il, t.name);
        }
        c.slot   = ggml_view_2d(ec->ctx, ec->tables, 1, ec->n_expert, sizeof(int32_t), li*ec->tables->nb[2]);
        c.cpu_id = ggml_view_2d(ec->ctx, ec->tables, 1, ec->n_expert, sizeof(int32_t), li*ec->tables->nb[2] + ec->tables->nb[1]);
        ggml_format_name(c.slot,   "blk.%d.ffn_exps_cache_slot",   cl.il);
        ggml_format_name(c.cpu_id, "blk.%d.ffn_exps_cache_cpu_id", cl.il);

        c.ids = ggml_view_1d(ec->ctx, ec->ids, n_ids, li*ec->ids->nb[1]);
        ggml_format_name(c.ids, "blk.%d.ffn_exps_cache_ids", cl.il);
    }

    for (size_t li = 0; li < ec->layers.size(); ++li) {
        for (size_t ti = 0; ti < ec->layers[li].exps.size(); ++ti) {
            ec->exps_index[ec->layers[li].exps[ti].src] = { (int) li, (int) ti };
        }
    }
    ggml_init_params params_views = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ec->ctx_views = ggml_init(params_views);

    ec->buf = ggml_backend_alloc_ctx_tensors_from_buft(ec->ctx, ggml_backend_dev_buffer_type(dev));
    if (ec->buf == nullptr) {
        for (auto & cl : ec->layers) {
            *cl.cache = {};
        }
        LLAMA_LOG_ERROR("%s: failed to allocate the expert cache, disabled\n", __func__);
        return nullptr;
    }
    // weights: the scheduler runs the ops that use them on the GPU
    ggml_backend_buffer_set_usage(ec->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // initial usage from a profile, else no prior
    std::vector<std::vector<float>> prior(model.layers.size(), std::vector<float>(ec->n_expert, 0.0f));
    if (const char * path = getenv("LLAMA_EXPERT_CACHE_PROFILE")) {
        std::ifstream f(path);
        int il, e;
        float n;
        while (f >> il >> e >> n) {
            if (il >= 0 && il < (int) prior.size() && e >= 0 && e < ec->n_expert) {
                prior[il][e] += n;
            }
        }
    }

    // opt-in: it did not measurably speed up the first requests
    const char * pf = getenv("LLAMA_EXPERT_CACHE_PREFAULT");
    if (pf != nullptr && atoi(pf) != 0) {
        const int64_t t0 = ggml_time_us();
        for (auto & cl : ec->layers) {
            for (const auto & t : cl.exps) {
                prefault(t.src);
            }
        }
        LLAMA_LOG_INFO("%s: prefaulted the host experts in %.1f s\n", __func__, (ggml_time_us() - t0)/1e6);
    }

    for (auto & cl : ec->layers) {
        for (const auto & t : cl.fixed) {
            ggml_backend_tensor_set(*t.dst, t.src->data, 0, ggml_nbytes(t.src));
        }
        cl.slot     .assign(ec->n_expert, -1);
        cl.cpu_id   .resize(ec->n_expert);
        cl.expert   .assign(ec->n_slot, -1);
        cl.loading  .assign(ec->n_slot, -1);
        cl.load_slot.assign(ec->n_expert, -1);
        cl.score    .assign(ec->n_expert, 0.0f);

        // scale so that the scores look like ~ the average number of picks per token
        const auto & p = prior[cl.il];
        double sum = 0.0;
        for (float x : p) {
            sum += x;
        }
        for (int e = 0; e < ec->n_expert; ++e) {
            cl.cpu_id[e] = e;
            cl.score[e]  = sum > 0.0 ? 8.0f*p[e]/sum : 0.0f;
        }

        std::vector<int> order(ec->n_expert);
        for (int e = 0; e < ec->n_expert; ++e) {
            order[e] = e;
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cl.score[a] > cl.score[b]; });

        for (int s = 0; s < ec->n_slot; ++s) {
            ec->copy_expert(cl, s, order[s]);
            ec->publish(cl, s, order[s]);
        }
    }
    ec->layer_caches.assign(model.layers.size(), nullptr);
    for (const auto & cl : ec->layers) {
        ec->layer_caches[cl.il] = cl.cache;
    }

    ec->tables_host.resize(ggml_nelements(ec->tables));
    ec->upload_tables();

    ec->ids_host.resize(ggml_nelements(ec->ids));
    ec->worker = std::thread(&llama_expert_cache::worker_loop, ec.get());

    LLAMA_LOG_INFO("%s: %zu layers, %d slots/layer, %.1f MiB on %s, budget %d experts/token\n", __func__,
            ec->layers.size(), ec->n_slot, ggml_backend_buffer_get_size(ec->buf)/1024.0/1024.0, ggml_backend_dev_name(dev), ec->budget);

    return ec;
}

llama_expert_cache::~llama_expert_cache() {
    if (worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        worker.join();
    }
    if (n_pick > 0) {
        LLAMA_LOG_INFO("%s: %lld tokens, hit rate %.1f%%, %.2f promotions/token\n", __func__,
                (long long) n_tok, 100.0*n_hit/n_pick, (double) n_promo/std::max<int64_t>(1, n_tok));
    }
    if (n_copy_dev + n_copy_host > 0) {
        LLAMA_LOG_INFO("%s: large batches: %.1f GiB of experts copied from the slots, %.1f GiB from host memory\n", __func__,
                n_copy_dev/1024.0/1024.0/1024.0, n_copy_host/1024.0/1024.0/1024.0);
    }
    if (ctx_views) {
        ggml_free(ctx_views);
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

// small chunks, so that the copy engine is not busy for long when the graph needs it
static void tensor_set_chunked(ggml_tensor * dst, const char * src, size_t offset, size_t size, size_t chunk) {
    for (size_t i = 0; i < size; i += chunk) {
        ggml_backend_tensor_set(dst, src + i, offset + i, std::min(chunk, size - i));
    }
}

void llama_expert_cache::copy_expert(const layer & l, int s, int e) {
    for (const auto & t : l.exps) {
        tensor_set_chunked(*t.dst, (const char *) t.src->data + e*t.src->nb[2], s*(*t.dst)->nb[2], t.src->nb[2], chunk);
    }
}

int64_t llama_expert_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, int64_t e, int64_t last) {
    const auto it = exps_index.find(src);
    if (it == exps_index.end()) {
        return -(last - e + 1);
    }
    const layer & l = layers[it->second.first];
    ggml_tensor * slots = *l.exps[it->second.second].dst;

    const size_t size = src->nb[2];

    if (l.slot[e] < 0) {
        int64_t n = 1;
        while (e + n <= last && l.slot[e + n] < 0) {
            n++;
        }
        n_copy_host += n*size;
        return -n;
    }

    int64_t n = 0;
    for (; e + n <= last && l.slot[e + n] >= 0; ++n) {
        ggml_reset(ctx_views);
        ggml_tensor * src_view = ggml_view_3d(ctx_views, slots, slots->ne[0], slots->ne[1], 1, slots->nb[1], slots->nb[2], l.slot[e + n]*slots->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views, dst,   dst->ne[0],   dst->ne[1],   1, dst->nb[1],   dst->nb[2],   (e + n)*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
    }
    n_copy_dev += n*size;

    // as the scheduler does for the host copies: MMQ reads a bit past the last expert, it must not be NaN
    // only needed at the end of the run of used experts, else the next expert is copied anyway
    const int64_t next = e + n;
    if (next > last && next < n_expert) {
        ggml_backend_tensor_set_async(backend, dst, (const char *) src->data + next*size, next*dst->nb[2], std::min<size_t>(size, 512));
    }
    return n;
}

int64_t llama_expert_cache::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, int64_t e, int64_t last, void * user_data) {
    return ((llama_expert_cache *) user_data)->copy_experts(backend, src, dst, e, last);
}

void llama_expert_cache::upload_tables() {
    bool dirty = false;
    for (size_t li = 0; li < layers.size(); ++li) {
        auto & l = layers[li];
        if (l.dirty) {
            int32_t * dst = tables_host.data() + li*2*n_expert;
            std::copy(l.slot.begin(),   l.slot.end(),   dst);
            std::copy(l.cpu_id.begin(), l.cpu_id.end(), dst + n_expert);
            l.dirty = false;
            dirty   = true;
        }
    }
    if (dirty) {
        ggml_backend_tensor_set(tables, tables_host.data(), 0, ggml_nbytes(tables));
    }
}

void llama_expert_cache::publish(layer & l, int s, int e) {
    l.expert[s] = e;
    l.slot[e]   = s;
    l.cpu_id[e] = -1;
    l.dirty     = true;
}

void llama_expert_cache::worker_loop() {
    std::unique_lock<std::mutex> lock(mtx);
    while (true) {
        cv.wait(lock, [&] { return stop || !jobs.empty(); });
        if (stop) {
            return;
        }
        std::vector<copy_job> todo;
        std::swap(todo, jobs);
        busy = true;
        lock.unlock();

        // the slots are not in the lookup tables, so the running graph does not read them
        for (const auto & j : todo) {
            copy_expert(layers[j.li], j.s, j.e);
        }

        lock.lock();
        busy = false;
        cv.notify_all();
    }
}

void llama_expert_cache::update(int n_tokens) {
    const int64_t t0 = ggml_time_us();
    // finish the copies started by the previous update and make them visible
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !busy && jobs.empty(); });
    }
    const int64_t t1 = ggml_time_us();
    t_wait += t1 - t0;
    n_update++;
    for (auto & l : layers) {
        for (int s = 0; s < n_slot; ++s) {
            const int e = l.loading[s];
            if (e >= 0) {
                l.loading[s]   = -1;
                l.load_slot[e] = -1;
                publish(l, s, e);
            }
        }
    }

    std::vector<copy_job> new_jobs;

    if (n_tokens > 0 && n_tokens <= LLAMA_EXPERT_CACHE_MAX_TOKENS) {
        struct miss {
            float score;
            int   li;
            int   e;
        };
        std::vector<miss> misses;

        const int64_t t2 = ggml_time_us();
        ggml_backend_tensor_get(ids, ids_host.data(), 0, ggml_nbytes(ids));
        t_read += ggml_time_us() - t2;

        const float d = powf(decay, (float) n_tokens);

        for (int li = 0; li < (int) layers.size(); ++li) {
            auto & l = layers[li];
            const int32_t * ids_l = ids_host.data() + (size_t) li*n_used*LLAMA_EXPERT_CACHE_MAX_TOKENS;

            for (auto & x : l.score) {
                x *= d;
            }
            for (int i = 0; i < n_used*n_tokens; ++i) {
                const int32_t e = ids_l[i];
                if (e < 0 || e >= n_expert) {
                    continue;
                }
                l.score[e] += 1.0f;
                n_pick++;
                if (l.slot[e] >= 0) {
                    n_hit++;
                } else if (l.load_slot[e] < 0) {
                    misses.push_back({ 0.0f, li, e });
                }
            }
        }
        n_tok += n_tokens;

        for (auto & m : misses) {
            m.score = layers[m.li].score[m.e];
        }
        std::sort(misses.begin(), misses.end(), [](const miss & a, const miss & b) { return a.score > b.score; });

        const int max_promo = std::min(budget*n_tokens, 64);

        for (const auto & m : misses) {
            if ((int) new_jobs.size() >= max_promo) {
                break;
            }
            auto & l = layers[m.li];
            if (l.slot[m.e] >= 0 || l.load_slot[m.e] >= 0) {
                continue; // duplicate miss
            }

            int victim = -1;
            for (int s = 0; s < n_slot; ++s) {
                if (l.loading[s] >= 0) {
                    continue;
                }
                const int ev = l.expert[s];
                if (victim < 0 || ev < 0 || (l.expert[victim] >= 0 && l.score[ev] < l.score[l.expert[victim]])) {
                    victim = s;
                }
            }
            if (victim < 0 || (l.expert[victim] >= 0 && l.score[l.expert[victim]] >= m.score)) {
                continue;
            }

            // the old expert goes back to the CPU now, the new one is used once its copy is done
            const int old = l.expert[victim];
            if (old >= 0) {
                l.slot[old]   = -1;
                l.cpu_id[old] = old;
                l.dirty       = true;
            }
            l.expert[victim]  = -1;
            l.loading[victim] = m.e;
            l.load_slot[m.e]  = victim;

            new_jobs.push_back({ m.li, victim, m.e });
        }
        n_promo += new_jobs.size();
    }

    const int64_t t3 = ggml_time_us();
    upload_tables();
    t_upload += ggml_time_us() - t3;

    if (!new_jobs.empty()) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            jobs = std::move(new_jobs);
        }
        cv.notify_all();
    }

    if (n_tok - n_report >= 256) {
        n_report = n_tok;
        LLAMA_LOG_INFO("%s: %lld tokens, hit rate %.1f%%, %.2f promotions/token\n", __func__,
                (long long) n_tok, 100.0*n_hit/std::max<int64_t>(1, n_pick), (double) n_promo/n_tok);
        LLAMA_LOG_INFO("%s: per update (%lld): wait %.0f us, read %.0f us, upload %.0f us, total %.0f us\n", __func__,
                (long long) n_update, (double) t_wait/n_update, (double) t_read/n_update, (double) t_upload/n_update, (double) t_total/n_update);
    }
    t_total += ggml_time_us() - t0;
}
