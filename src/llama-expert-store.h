#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_ssd_reader;

// whether a pshard model gets the RAM expert store, and its RAM: only a model whose weights do not fit in RAM. The
// inputs are fixed per machine and model, so a model that fits gets "off" on every run
struct llama_expert_store_decision {
    bool         on     = false;
    size_t       budget = 0;    // bytes of RAM for the store's slots (auto: still capped by the RAM free at build)
    const char * why    = "";
};

// exps_b: the routed-expert weights, other_b: every other weight, reserve_b: RAM the runtime keeps for itself,
// mra_mib: -1 = auto, 0 = off, N = N MiB
llama_expert_store_decision llama_expert_store_decide(size_t exps_b, size_t other_b, size_t ram_total_b, size_t reserve_b, int64_t mra_mib);

// bytes of one slot of an expert tensor whose experts are row_bytes each: the rows plus the sector rounding on both ends
size_t llama_expert_store_stride(size_t row_bytes);
// slots per layer that cap bytes of RAM hold, at slot_all bytes for one slot in every layer
uint32_t llama_expert_store_slots(size_t cap, size_t slot_all, uint32_t n_expert);
// of n_ram slots per layer, those pin_b bytes of pinned host memory hold (one host buffer per slot index)
uint32_t llama_expert_store_pinned_slots(size_t pin_b, size_t slot_all, uint32_t n_ram);
// bytes of the pinned staging buffer the store allocates for experts of stride bytes per tensor
size_t llama_expert_store_stage_bytes(size_t stride);

// the model's files in split order, from the split naming of path_model (just it when it does not follow it)
std::vector<std::string> llama_expert_store_files(const std::string & path_model);
// the largest of them, which holds the most experts the store reads ("" = none is readable), and their total bytes
std::string llama_expert_store_largest_file(const std::string & path_model, uint64_t * total_b = nullptr);
// the store reader's rate in GB/s: unbuffered reads of read_bytes (a sector multiple) at random sector-aligned offsets
// of path (a scratch file: opened with every sharing mode), n_inflight of them waited for together, for about
// `seconds` (0 = the file cannot be read)
double llama_expert_store_read_gbps(const std::string & path, size_t read_bytes, int n_inflight, double seconds);
// whether the store's reads of one wait are in flight together (else they run one after another)
bool llama_expert_store_reads_overlap();

// RAM expert store of a model larger than RAM: each MoE layer keeps n_ram of its routed experts in locked RAM slots,
// read from the GGUF files with unbuffered reads, least recently used out first. The CPU expert matmul reads a
// resident expert from its slot (GGML_TENSOR_FLAG_EXPERT_TABLE), any other expert from the mapping. The scheduler's
// host-experts hook brings a CPU split's routed experts in before the split computes
struct llama_expert_store {
    // one routed-expert tensor of layer il, at byte file_off of file split
    struct tensor_src {
        ggml_tensor * t;
        int32_t       il;
        size_t        split;
        size_t        file_off;
    };

    ~llama_expert_store();

    // at load: the layers and their tensors. paths: the model's files in split order. budget: bytes of RAM for the
    // slots (auto: capped by the RAM free at allocate, less reserve twice). false: the store stays off (logged)
    bool build(const std::vector<std::string> & paths, const std::vector<tensor_src> & srcs, uint32_t n_expert,
               uint32_t n_expert_used, size_t budget, size_t reserve, bool auto_budget, const std::string & wl_path);

    // once, after the first context has allocated its buffers: locks the slots, opens the files, seeds each layer
    // with its most routed experts in <model>.pshard_workload and sets the tensors' tables. false: every expert
    // keeps being read from the mapping (logged). n_pin of each layer's slots (the top indices, which the seed fills
    // with the most routed experts) come from dev's pinned host buffers: device uploads from them are kernel copies
    bool allocate(ggml_backend_dev_t dev = nullptr, uint32_t n_pin = 0);

    // the scheduler's host-experts hook (ggml_backend_sched_host_experts_cb)
    static int host_experts_cb(struct ggml_cgraph * g, int j0, void * user_data);
    int host_experts(struct ggml_cgraph * g, int j0);

    // the scheduler's expert-upload hook (ggml_backend_sched_expert_upload_cb): a resident expert is copied from its
    // slot, one in the running layer group's window from its pieces, any other is read from the files into a pinned
    // staging buffer and copied from there; the slots keep their experts, so a prompt does not push the decode set out.
    // ram_only: the running layer group's experts only, each from its slot or window pieces, else false and no copy
    static bool expert_upload_cb(const ggml_tensor * src, ggml_tensor * dst, const ggml_backend_expert_upload * ups, int n,
                                 bool ram_only, ggml_backend_t backend, void * user_data);
    bool upload(const ggml_tensor * src, ggml_tensor * dst, const ggml_backend_expert_upload * ups, int n, bool ram_only,
                ggml_backend_t backend);

    // an expert pool over the store, one pass of t's layer: its VRAM hits take the newest stamps, then its misses
    // become resident (those not in a slot are read from the files) and their tables point at the slots; reader (may
    // be nullptr) copies from the slots until a later read into the layer's slots waits for it. false: t is not
    // served, or the misses exceed the layer's slots (nothing read: the pool copies from the mapping). pinned (may be
    // nullptr): per miss, whether its slot is a pinned one
    bool acquire(const ggml_tensor * t, const int32_t * hits, size_t n_hits, const int32_t * misses, size_t n_misses,
                 ggml_backend_t reader, uint8_t * pinned = nullptr);

    // before a context frees its backends: waits for every copy the store queued on them
    void detach();

    // layer-major prefill: a sweep runs the layer groups [first, second) in order, each over every ubatch. The first
    // window_begin of a sweep borrows the least recently used slots of every layer for two halves of window rows: their
    // RAM becomes one pinned buffer of dev's (without one, the slots hold the rows, pageable). window_begin(g) gets
    // group g's experts not in a slot into one half (it waits for the reads issued ahead, and reads the rest) and
    // issues the reads of group g + 1's into the other half; the device uploads to dev copy group g's experts from the
    // rows. window_end waits for the copies from the group's rows
    using layer_groups = std::vector<std::pair<int32_t, int32_t>>;
    void window_begin(const layer_groups & groups, size_t g, ggml_backend_dev_t dev);
    void window_end();
    // after the last group, and when the sweep stops early: waits for every read into and copy from the window rows
    // and gives the borrowed slots back empty, their RAM locked again
    void sweep_end();
    // the bytes of the slots a sweep over groups borrows (0: no sweep, why says the reason)
    size_t sweep_bytes(const layer_groups & groups, const char ** why);
    // the experts per layer a sweep over groups reads from the files: not in a slot, the lent slots included (0: no
    // sweep)
    uint32_t sweep_reads(const layer_groups & groups);

    // the slots are allocated: the store serves the experts
    bool active() const { return on; }
    // the store holds the routed experts of layer il
    bool has_layer(int32_t il) const;
    // the layers whose routed experts it holds
    uint32_t n_layers() const { return (uint32_t) layers.size(); }
    // microseconds every pass so far waited for SSD reads
    uint64_t wait_us();

    void log_counters();
    // routes the hook saw since the last fold go into <model>.pshard_workload
    void fold_workload(const std::string & workload_path);

    uint32_t n_ram = 0;
    uint32_t n_rp  = 0;             // the top n_rp slots of every layer are pinned
    bool     count_routes = true;   // false while an expert pool tier runs: the pool counts its own routes

private:
    struct entry {
        ggml_tensor * t;
        size_t        split;
        size_t        file_off;
        size_t        row_bytes;
        size_t        stride;                // bytes per slot: the rows plus the sector rounding on both ends
        char *        base = nullptr;        // this tensor's locked slots
        size_t        pin_off = 0;           // this tensor's slot inside a pinned slot buffer
        std::vector<const void *> tbl;       // [n_expert] the rows in a slot, NULL = read the mapping
        uint32_t      lay_k   = 0;           // index among its layer's tensors
    };
    struct layer {
        int32_t               il = -1;
        std::vector<uint32_t> ents;
        std::vector<int32_t>  slot_of;       // [n_expert] -1 = not resident
        std::vector<int32_t>  expert_of;     // [n_ram]
        std::vector<uint64_t> stamp;         // [n_ram] last use, UINT64_MAX = lent to the window rows or out of service
        std::vector<uint64_t> routes;        // [n_expert] routes seen since the last fold
        std::vector<uint32_t> lent;          // the slots lent to the window rows while a sweep runs
        uint32_t              n_out = 0;     // slots out of service: their RAM did not come back locked after a pinned window
        uint64_t              clock = 0;
        std::vector<ggml_backend_t> gpu_readers;   // copies queued on them may still read the slots (and its window rows)
        int32_t               win = -1;      // its part of the running group's half, -1 = none
    };

    void seed();

    // allocate's sizing from the RAM free now: the RAM for the slots and the slots per layer
    struct sizing {
        size_t   cap = 0;
        uint32_t n   = 0;
    };
    sizing size_for() const;

    // counters: [0] one-row passes (decode), [1] multi-row passes, [2] device uploads, [3] device uploads of experts
    // not in a slot while their layer's group runs in a sweep, [4] an expert pool's misses
    struct counters {
        uint64_t passes = 0, requests = 0, hits = 0, reads = 0, bytes = 0, wait_us = 0, fallbacks = 0, pinned = 0;
    };

    // gives the experts (all misses, n <= the slots not lent) the least recently used slots; their tables point there
    // once publish runs
    void claim(layer & L, const int32_t * experts, size_t n);
    // queues the reads of tensor ei's rows of the experts into their slots
    void read_rows(uint32_t ei, const int32_t * experts, size_t n, counters & c);
    void publish(uint32_t ei, const int32_t * experts, size_t n);
    // reads the experts into the least recently used slots
    void fetch(layer & L, const int32_t * experts, size_t n, counters & c);
    // counts the routing and stamps its hits; returns the misses to read, in want
    size_t serve(layer & L, const ggml_tensor * ids);

    // a layer's misses, every read in flight in the order the graph uses the tensors: a tensor's matmul waits only for
    // its own rows, so the nodes before it compute while the later tensors are read
    struct pending_reads {
        int32_t               layer = -1;   // -1: none
        std::vector<int32_t>  x;            // the experts
        std::vector<uint32_t> order;        // the layer's tensors; order[n_used - 1..] are waited for together
        size_t                n_used = 0;
        size_t                k = 0;        // order[..k] are in place
        counters *            c = nullptr;
    };
    pending_reads pend;
    const ggml_tensor * served_ids   = nullptr;   // the routing served last in the running split
    uint32_t            served_layer = UINT32_MAX;
    void read_ahead(uint32_t li, ggml_cgraph * g, int j, size_t n_miss, counters & c);
    // waits for order[k]'s reads and points its tables at the slots
    void pend_step();
    void pend_drain();
    bool pend_has(uint32_t ei) const;

    std::vector<entry> ents;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, uint32_t> layer_of;   // expert tensor -> index in layers
    std::unordered_map<const ggml_tensor *, uint32_t> ent_of;     // expert tensor -> index in ents
    std::vector<uint8_t> used;           // scratch: [n_expert]
    std::vector<int32_t> want;           // scratch

    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    std::vector<std::string> file_paths;
    std::string workload_path;
    size_t   budget_b      = 0;
    size_t   reserve_b     = 0;
    bool     budget_auto   = false;
    bool     failed        = false;
    size_t   slot_all      = 0;          // one slot in every layer
    size_t   experts_all   = 0;
    bool     on            = false;      // the slots are allocated
    char *   arena         = nullptr;    // the locked slots [0, n_lk) of every layer
    size_t   arena_bytes   = 0;
    bool     arena_locked  = false;
    uint32_t n_lk          = 0;
    std::vector<ggml_backend_buffer_t> pin_bufs;   // the pinned slots [n_lk, n_ram): one buffer per slot index
    std::vector<char *>                pin_slot;   // their sector-aligned bases
    char * slot(const entry & e, uint32_t v) const {
        return v < n_lk ? e.base + (size_t) v * e.stride : pin_slot[v - n_lk] + e.pin_off;
    }
    llama_ssd_reader * reader = nullptr; // owned
    std::mutex mtx;

    counters cnt[5];

    // layer-major sweep: the pinned window, or the slots lent per layer merged where adjacent, are cut into two halves
    // of equal bytes. A half holds one group's experts not in a slot as a byte stream: the rows of an expert in one
    // tensor take one piece of the half's areas, or several cut at sector boundaries, so any layer's rows fit any lent
    // slot
    struct window_area {
        char * p;
        size_t cap;
    };
    struct window_piece {
        char *   p;
        size_t   len;
        size_t   file_off;                   // sector aligned
        size_t   need;                       // the bytes of it the rows use, from its start
        uint32_t split;
    };
    struct window_item {
        uint32_t part;
        uint32_t k;                          // the tensor's index among its layer's tensors
        int32_t  x;
        uint32_t p0, p1;                     // its pieces, in file order
    };
    struct window_part {
        int32_t              layer = -1;     // index in layers
        std::vector<int32_t> at;             // [n tensors of the layer][n_expert] the item of the expert's rows, -1 = none
    };
    struct window_half {
        std::vector<window_area>  areas;     // in address order, each within one registration
        std::vector<window_piece> pieces;
        std::vector<window_part>  parts;
        std::vector<window_item>  items;
        size_t  n_issued = 0;                // pieces [0, n_issued) were issued on the read-ahead reader
        int32_t group    = -1;
        bool    pending  = false;            // the read-ahead reader may still write into its areas
    };
    bool               sweep_on  = false;
    window_half        win[2];
    int                win_cur   = -1;       // the half the running group reads, -1 = none
    ggml_backend_dev_t win_dev   = nullptr;  // the device whose uploads read the rows
    llama_ssd_reader * ahead     = nullptr;  // owned: the reads of the next group's rows only
    bool               ahead_off = false;    // it did not open: every group reads its rows when it starts
    bool               lend_refused  = false; // logged once
    bool               ahead_refused = false; // logged once
    ggml_backend_buffer_t win_buf    = nullptr; // the pinned window while a sweep runs, the lent slots' RAM given back
    bool               pin_refused   = false; // logged once: later sweeps keep the window in the lent slots
    size_t             ws_lent       = 0;     // bytes the working-set minimum gave up for the pinned window
    uint32_t           slots_out     = 0;     // slots out of service: no sweep lends slots any more

    // sweep counters: lent bytes, the reads issued ahead, the time window_begin waited for them (ahead) and for the
    // rows it read itself (now), the sweeps with a pinned window and the time to set it up and to give its RAM back
    struct sweep_counters {
        uint64_t sweeps = 0, lent_bytes = 0;
        uint64_t ahead_reads = 0, ahead_bytes = 0, ahead_wait_us = 0, now_reads = 0, now_bytes = 0, now_wait_us = 0, unplaced = 0;
        uint64_t pinned = 0, pin_start_us = 0, pin_stop_us = 0;
    };
    sweep_counters sw;
    uint32_t       lend_last = 0;

    // the slots per layer a sweep over groups lends when each layer holds n (0: none, why says the reason)
    uint32_t lend_for(const layer_groups & groups, uint32_t n, const char ** why) const;
    bool sweep_start(const layer_groups & groups, ggml_backend_dev_t dev);
    void sweep_stop();
    // the lent slots' RAM given back for one pinned buffer of dev's: the sector-aligned start of its bytes (nullptr:
    // the slots keep their RAM)
    char * window_pin(ggml_backend_dev_t dev, size_t bytes);
    // slot v of L: its pages go back to the system (lock = false) or come back committed and locked (true)
    bool slot_pages(const layer & L, uint32_t v, bool lock) const;
    // the lent slots' pages come back locked; a slot whose pages do not leaves the lent list, out of service
    void lent_relock();
    // assigns half h's areas to group g's experts not in a slot
    void window_plan(int h, const layer_groups & groups, size_t g);
    void window_close();

    // device uploads of experts not in a slot: read into one half of a pinned buffer while the copies from the other
    // half run; a half is reused after its event
    ggml_backend_buffer_t stage_buf = nullptr;
    char *                stage     = nullptr;
    size_t                stage_half = 0;
    ggml_backend_dev_t    stage_dev = nullptr;
    ggml_backend_event_t  stage_ev[2] = { nullptr, nullptr };
    bool                  stage_busy[2] = { false, false };
    int                   stage_next = 0;
    ggml_backend_t        stage_backend = nullptr;
    std::vector<int>      miss;                  // scratch
    uint8_t               zero_rows[4096] = {};  // the padding after an expert's rows
    bool ensure_stage(ggml_backend_dev_t dev, size_t stride);
    void stage_fence(int h);
    uint64_t pf_decode0 = 0;             // process page faults at the first one-row pass
    bool     pf_have    = false;
};
