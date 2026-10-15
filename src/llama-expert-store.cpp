#include "llama-expert-store.h"
#include "llama-pshard-workload.h"
#include "llama-ssd-reader.h"

#include "llama-impl.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>

#ifdef _WIN32
#   include <psapi.h>
#else
#   include <sys/mman.h>
#   include <sys/resource.h>
#endif

llama_expert_store_decision llama_expert_store_decide(size_t exps_b, size_t other_b, size_t ram_total_b, size_t reserve_b, int64_t mra_mib) {
    llama_expert_store_decision d;
    if (exps_b == 0) {
        d.why = "off: no routed experts";
        return d;
    }
    if (mra_mib == 0) {
        d.why = "off: -mra 0";
        return d;
    }
    if (exps_b + other_b + reserve_b <= ram_total_b) {
        d.why = "off: the model fits in RAM";
        return d;
    }
    d.on = true;
    if (mra_mib > 0) {
        d.budget = (size_t) mra_mib * 1024 * 1024;
        d.why    = "on: the model is larger than RAM, -mra budget";
    } else {
        d.budget = ram_total_b > other_b + reserve_b ? ram_total_b - other_b - reserve_b : 0;
        d.why    = "on: the model is larger than RAM, RAM minus the other weights and the reserve";
    }
    return d;
}

size_t llama_expert_store_stride(size_t row_bytes) {
    constexpr size_t sector = llama_ssd_reader::sector;
    // the sector-aligned span of any expert's rows
    return GGML_PAD(row_bytes + sector - 1, sector);
}

uint32_t llama_expert_store_slots(size_t cap, size_t slot_all, uint32_t n_expert) {
    return slot_all > 0 ? (uint32_t) std::min<size_t>(n_expert, cap / slot_all) : 0;
}

std::vector<std::string> llama_expert_store_files(const std::string & path_model) {
    int n_split = 0;
    const size_t p = path_model.rfind("-of-");
    if (p != std::string::npos && sscanf(path_model.c_str() + p, "-of-%d", &n_split) == 1 && n_split > 1) {
        char prefix[1024];
        if (llama_split_prefix(prefix, sizeof(prefix), path_model.c_str(), 0, n_split) > 0) {
            std::vector<std::string> paths;
            for (int i = 0; i < n_split; i++) {
                char path[1024];
                llama_split_path(path, sizeof(path), prefix, i, n_split);
                paths.push_back(path);
            }
            return paths;
        }
    }
    return { path_model };
}

static uint64_t file_bytes(const std::string & path) {
#ifdef _WIN32
    // links followed; attribute access only, which a file held open with other sharing (a scratch file) still allows
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wlen > 0 ? (size_t) wlen : 0, L'\0');
    if (wlen <= 0 || MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen) <= 0) {
        return 0;
    }
    HANDLE h = CreateFileW(wpath.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return 0;
    }
    LARGE_INTEGER n = {};
    const BOOL ok = GetFileSizeEx(h, &n);
    CloseHandle(h);
    return ok && n.QuadPart > 0 ? (uint64_t) n.QuadPart : 0;
#else
    FILE * f = ggml_fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return 0;
    }
    fseeko(f, 0, SEEK_END);
    const int64_t n = ftello(f);
    fclose(f);
    return n > 0 ? (uint64_t) n : 0;
#endif
}

double llama_expert_store_read_gbps(const std::string & path, size_t read_bytes, int n_inflight, double seconds) {
    constexpr size_t sector = llama_ssd_reader::sector;
    const uint64_t size = file_bytes(path);
    llama_ssd_reader rd;
    if (read_bytes == 0 || read_bytes % sector != 0 || size < 2 * (uint64_t) read_bytes || !rd.open({ path }, true)) {
        return 0.0;
    }
    std::vector<char> buf((size_t) n_inflight * read_bytes + sector);
    char * base = (char *) GGML_PAD((uintptr_t) buf.data(), sector);
    // new offsets for every cell and run: offsets read just before may be cached on the device
    std::mt19937_64 rng(std::random_device{}());
    const uint64_t n_pos = (size - read_bytes) / sector;
    uint64_t bytes = 0;
    ggml_time_init();
    const int64_t t0 = ggml_time_us();
    int64_t t1 = t0;
    do {
        for (int i = 0; i < n_inflight; i++) {
            rd.read(0, (rng() % n_pos) * sector, read_bytes, base + (size_t) i * read_bytes, read_bytes);
        }
        if (!rd.wait_all()) {
            return 0.0;
        }
        bytes += (uint64_t) n_inflight * read_bytes;
        t1 = ggml_time_us();
    } while ((double) (t1 - t0) < seconds * 1e6);
    return (double) bytes / ((double) (t1 - t0) / 1e6) / 1e9;
}

bool llama_expert_store_reads_overlap() {
    return llama_ssd_reader::overlap;
}

std::string llama_expert_store_largest_file(const std::string & path_model, uint64_t * total_b) {
    std::string file;
    uint64_t file_b = 0;
    uint64_t total  = 0;
    for (const auto & p : llama_expert_store_files(path_model)) {
        const uint64_t b = file_bytes(p);
        total += b;
        if (b > file_b) {
            file   = p;
            file_b = b;
        }
    }
    if (total_b != nullptr) {
        *total_b = total;
    }
    return file;
}

// this process's committed bytes and working set (0 where the system does not say)
static void process_memory(uint64_t & committed, uint64_t & working_set) {
    committed   = 0;
    working_set = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        committed   = pmc.PrivateUsage;
        working_set = pmc.WorkingSetSize;
    }
#endif
}

static uint64_t process_page_faults() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc = {};
    pmc.cb = sizeof(pmc);
    return GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) ? (uint64_t) pmc.PageFaultCount : 0;
#else
    struct rusage ru = {};
    return getrusage(RUSAGE_SELF, &ru) == 0 ? (uint64_t) (ru.ru_minflt + ru.ru_majflt) : 0;
#endif
}

// RAM free for the store: physical memory not in use (the file cache counts as free)
static size_t ram_available() {
#ifdef _WIN32
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (size_t) ms.ullAvailPhys : 0;
#else
#   ifdef __linux__
    // MemAvailable counts the file cache the kernel can reclaim, as Windows' available memory does
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char line[256];
        unsigned long long kb = 0;
        bool found = false;
        while (!found && fgets(line, sizeof(line), f) != nullptr) {
            found = sscanf(line, "MemAvailable: %llu kB", &kb) == 1;
        }
        fclose(f);
        if (found) {
            return (size_t) kb * 1024;
        }
    }
#   endif
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page  = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (size_t) pages * (size_t) page : 0;
#endif
}

// locked RAM: the store's slots must not be paged out under the file cache's pressure
static char * alloc_locked(size_t bytes, bool & locked) {
    locked = false;
#ifdef _WIN32
    void * p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p == nullptr) {
        return nullptr;
    }
    HANDLE proc = GetCurrentProcess();
    SIZE_T ws_min = 0, ws_max = 0;
    DWORD  ws_flags = 0;
    if (GetProcessWorkingSetSizeEx(proc, &ws_min, &ws_max, &ws_flags)) {
        const SIZE_T extra = (SIZE_T) bytes + (64ULL << 20);
        if (SetProcessWorkingSetSizeEx(proc, ws_min + extra, ws_max + extra,
                QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
            if (VirtualLock(p, bytes)) {
                locked = true;
                return (char *) p;
            }
            SetProcessWorkingSetSizeEx(proc, ws_min, ws_max, ws_flags);
        }
    }
    VirtualFree(p, 0, MEM_RELEASE);
    return nullptr;
#else
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    if (mlock(p, bytes) == 0) {
        locked = true;
        return (char *) p;
    }
    munmap(p, bytes);
    return nullptr;
#endif
}

// waits for the device copies that may still read a layer's slots and window rows
static void wait_readers(std::vector<ggml_backend_t> & readers) {
    for (ggml_backend_t b : readers) {
        ggml_backend_synchronize(b);
    }
    readers.clear();
}

static void free_locked(char * p, size_t bytes) {
#ifdef _WIN32
    VirtualUnlock(p, bytes);
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munlock(p, bytes);
    munmap(p, bytes);
#endif
}

llama_expert_store::~llama_expert_store() {
    // the model frees the store before its tensors
    for (auto & e : ents) {
        e.t->flags &= ~GGML_TENSOR_FLAG_EXPERT_TABLE;
        e.t->extra  = nullptr;
    }
    for (int h = 0; h < 2; h++) {
        if (stage_ev[h] != nullptr) {
            ggml_backend_event_free(stage_ev[h]);
        }
    }
    if (stage_buf != nullptr) {
        ggml_backend_buffer_free(stage_buf);
    }
    // every context's detach ended its sweep; reads still in flight land before their memory goes
    if (win[0].pending || win[1].pending) {
        ahead->wait_all();
    }
    if (win_buf != nullptr) {
        ggml_backend_buffer_free(win_buf);
    }
    delete ahead;
    delete reader;
    if (arena != nullptr) {
        free_locked(arena, arena_bytes);
    }
}

bool llama_expert_store::build(const std::vector<std::string> & paths, const std::vector<tensor_src> & srcs, uint32_t n_exp,
        uint32_t n_exp_used, size_t budget, size_t reserve, bool auto_budget, const std::string & wl_path) {
    n_expert      = n_exp;
    n_expert_used = n_exp_used;
    file_paths    = paths;
    budget_b      = budget;
    reserve_b     = reserve;
    budget_auto   = auto_budget;
    workload_path = wl_path;

    std::map<int32_t, std::vector<size_t>> by_il;
    for (size_t i = 0; i < srcs.size(); i++) {
        by_il[srcs[i].il].push_back(i);
    }
    for (const auto & [il, idx] : by_il) {
        layer L;
        L.il = il;
        for (size_t i : idx) {
            const auto & s = srcs[i];
            if (!ggml_is_contiguous(s.t) || s.split >= paths.size()) {
                LLAMA_LOG_WARN("%s: expert store off: %s is not a contiguous tensor of the model's files\n", __func__, s.t->name);
                return false;
            }
            entry e;
            e.t         = s.t;
            e.split     = s.split;
            e.file_off  = s.file_off;
            e.row_bytes = s.t->nb[2];
            e.stride    = llama_expert_store_stride(e.row_bytes);
            e.lay_k     = (uint32_t) L.ents.size();
            slot_all    += e.stride;
            experts_all += ggml_nbytes(s.t);
            L.ents.push_back((uint32_t) ents.size());
            layer_of[s.t] = (uint32_t) layers.size();
            ent_of[s.t]   = (uint32_t) ents.size();
            ents.push_back(std::move(e));
        }
        layers.push_back(std::move(L));
    }
    return slot_all > 0;
}

llama_expert_store::sizing llama_expert_store::size_for() const {
    sizing sz;
    sz.cap = budget_b;
    if (budget_auto) {
        // what the page-locked weights and every other process leave, minus the runtime's reserve twice: its own
        // pinned allocations, and room for the file cache the uploads from the mapping go through
        const size_t avail = ram_available();
        sz.cap = std::min(sz.cap, avail > 2 * reserve_b ? avail - 2 * reserve_b : 0);
    }
    sz.n = llama_expert_store_slots(sz.cap, slot_all, n_expert);
    return sz;
}

uint32_t llama_expert_store::lend_for(const layer_groups & groups, uint32_t n, const char ** why) const {
    constexpr size_t sector = llama_ssd_reader::sector;
    *why = nullptr;
    if (n >= n_expert) {
        *why = "every expert fits in RAM";
        return 0;
    }
    // k lent slots per layer leave n_expert - (n - k) experts per layer out of the slots. A half holds group G when
    // its bytes, half of k slots in every layer, hold those experts' rows in G's layers: the rows of an expert in one
    // tensor take no more than the tensor's stride, wherever they land
    uint64_t grp_max = 0;   // one slot in each layer of the largest group
    for (const auto & G : groups) {
        uint64_t b = 0;
        for (const auto & L : layers) {
            if (L.il < G.first || L.il >= G.second) {
                continue;
            }
            for (uint32_t ei : L.ents) {
                b += ents[ei].stride;
            }
        }
        grp_max = std::max(grp_max, b);
    }
    for (uint32_t k = 1; k < n && n - k >= n_expert_used; k++) {
        const uint64_t out  = n_expert - (n - k);
        const uint64_t half = ((uint64_t) k * slot_all / 2) & ~(uint64_t) (sector - 1);
        if (half >= out * grp_max) {
            return k;
        }
    }
    *why = "the slots cannot lend two layer groups' experts not in RAM and keep the routed experts";
    return 0;
}

size_t llama_expert_store::sweep_bytes(const layer_groups & groups, const char ** why) {
    std::lock_guard<std::mutex> lk(mtx);
    if (failed) {
        *why = "the expert store is off";
        return 0;
    }
    // before allocate: the slots it would take now
    const uint32_t k = lend_for(groups, arena != nullptr ? n_ram : size_for().n, why);
    return (size_t) k * slot_all;
}

uint32_t llama_expert_store::sweep_reads(const layer_groups & groups) {
    std::lock_guard<std::mutex> lk(mtx);
    const char * why = nullptr;
    // sweep_start lends nothing once a slot is out of service
    const uint32_t k = arena != nullptr && slots_out == 0 ? lend_for(groups, n_ram, &why) : 0;
    return k > 0 ? n_expert - n_ram + k : 0;
}

bool llama_expert_store::allocate() {
    std::lock_guard<std::mutex> lk(mtx);
    if (arena != nullptr || failed) {
        return arena != nullptr;
    }
    failed = true;
    const sizing sz = size_for();
    const size_t cap = sz.cap;
    uint32_t n = sz.n;
    {
        uint64_t committed = 0, ws = 0;
        process_memory(committed, ws);
        LLAMA_LOG_INFO("%s: RAM %.2f GiB available, this process %.2f GiB committed / %.2f GiB working set; budget %.2f GiB, "
            "reserve %.2f GiB, cap %.2f GiB -> %u slots\n", __func__, ram_available() / 1073741824.0, committed / 1073741824.0,
            ws / 1073741824.0, budget_b / 1073741824.0, reserve_b / 1073741824.0, cap / 1073741824.0, n);
    }

    if (n < n_expert_used) {
        LLAMA_LOG_WARN("%s: expert store off: %.1f GiB of RAM hold %u slots per layer, below the %u routed experts\n",
            __func__, cap / 1073741824.0, n, n_expert_used);
        return false;
    }
    const int64_t t0 = ggml_time_us();
    while (arena == nullptr && n >= n_expert_used) {
        arena = alloc_locked((size_t) n * slot_all, arena_locked);
        if (arena == nullptr) {
            n = std::min<uint32_t>(n * 3 / 4, n - 1);
        }
    }
    if (arena == nullptr) {
        LLAMA_LOG_WARN("%s: expert store off: no locked RAM for %u slots per layer\n", __func__, n_expert_used);
        return false;
    }
    arena_bytes = (size_t) n * slot_all;
    n_ram = n;
    reader = new llama_ssd_reader();
    if (!reader->open(file_paths)) {
        free_locked(arena, arena_bytes);
        arena = nullptr;
        return false;
    }
    size_t off = 0;
    for (auto & L : layers) {
        for (uint32_t ei : L.ents) {
            auto & e = ents[ei];
            e.base = arena + off;
            off += (size_t) n_ram * e.stride;
            e.tbl.assign(n_expert, nullptr);
            e.t->extra  = e.tbl.data();
            e.t->flags |= GGML_TENSOR_FLAG_EXPERT_TABLE;
        }
        L.slot_of.assign(n_expert, -1);
        L.expert_of.assign(n_ram, -1);
        L.stamp.assign(n_ram, 0);
        L.routes.assign(n_expert, 0);
    }
    used.assign(n_expert, 0);
    failed = false;
    LLAMA_LOG_WARN("%s: expert store on: %u of %u experts per layer in RAM (%.1f GiB locked in %.1f s%s) for %.1f GiB of routed experts in %zu layers, %zu file(s)\n",
        __func__, n_ram, n_expert, arena_bytes / 1073741824.0, (ggml_time_us() - t0) / 1e6,
        budget_auto ? ", auto budget" : ", -mra budget", experts_all / 1073741824.0, layers.size(), file_paths.size());
    seed();
    return true;
}

void llama_expert_store::fetch(layer & L, const int32_t * experts, size_t n, counters & c) {
    constexpr size_t sector = llama_ssd_reader::sector;
    GGML_ASSERT(n <= n_ram - L.lent.size() - L.n_out);
    // device uploads of this layer may still read the slots about to be overwritten
    wait_readers(L.gpu_readers);
    for (size_t i = 0; i < n; i++) {
        const int32_t x = experts[i];
        // the requested experts carry the newest stamps and the lent slots the largest, so the victim is never one
        // of them
        uint32_t v = 0;
        for (uint32_t s = 1; s < n_ram; s++) {
            if (L.stamp[s] < L.stamp[v]) {
                v = s;
            }
        }
        const int32_t old = L.expert_of[v];
        if (old >= 0) {
            L.slot_of[old] = -1;
            for (uint32_t ei : L.ents) {
                ents[ei].tbl[old] = nullptr;
            }
        }
        L.expert_of[v] = x;
        L.slot_of[x]   = (int32_t) v;
        L.stamp[v]     = ++L.clock;
        for (uint32_t ei : L.ents) {
            const auto & e = ents[ei];
            const size_t off = e.file_off + (size_t) x * e.row_bytes;
            const size_t beg = off & ~(sector - 1);
            const size_t end = GGML_PAD(off + e.row_bytes, sector);
            reader->read(e.split, beg, end - beg, e.base + (size_t) v * e.stride, off + e.row_bytes - beg);
            c.bytes += end - beg;
        }
        c.reads++;
    }
    const int64_t t0 = ggml_time_us();
    if (!reader->wait_all()) {
        GGML_ABORT("expert store: an SSD read failed");
    }
    c.wait_us += ggml_time_us() - t0;
    // the rows keep the mapping's alignment within a sector, so the kernels read them as they read the mapping
    for (size_t i = 0; i < n; i++) {
        const int32_t x = experts[i];
        const size_t  v = (size_t) L.slot_of[x];
        for (uint32_t ei : L.ents) {
            auto & e = ents[ei];
            const size_t off = e.file_off + (size_t) x * e.row_bytes;
            e.tbl[x] = e.base + v * e.stride + (off & (sector - 1));
        }
    }
}

void llama_expert_store::serve(layer & L, const ggml_tensor * ids) {
    const int64_t n_ids  = ids->ne[0];
    const int64_t n_rows = ids->ne[1];
    want.clear();
    for (int64_t r = 0; r < n_rows; r++) {
        for (int64_t k = 0; k < n_ids; k++) {
            const int32_t x = *(const int32_t *) ((const char *) ids->data + r * ids->nb[1] + k * ids->nb[0]);
            if (x < 0 || (uint32_t) x >= n_expert) {
                continue;   // a -1 route computes nothing
            }
            L.routes[x]++;
            if (!used[x]) {
                used[x] = 1;
                want.push_back(x);
            }
        }
    }
    for (int32_t x : want) {
        used[x] = 0;
    }
    auto & c = cnt[n_rows > 1 ? 1 : 0];
    if (n_rows == 1 && !pf_have) {
        pf_decode0 = process_page_faults();
        pf_have    = true;
    }
    c.passes++;
    c.requests += want.size();
    // misses go to the SSD unless the pass routes to more experts than half the slots: those would push the hot
    // set out, so such a pass reads its misses from the mapping
    const size_t live = n_ram - L.lent.size() - L.n_out;
    const bool   fill = want.size() <= std::min(live, std::max<size_t>(live / 2, n_expert_used));
    size_t n_miss = 0;
    for (int32_t x : want) {
        const int32_t s = L.slot_of[x];
        if (s >= 0) {
            L.stamp[s] = ++L.clock;
            c.hits++;
        } else {
            want[n_miss++] = x;
        }
    }
    if (n_miss == 0) {
        return;
    }
    if (!fill) {
        c.fallbacks += n_miss;
        return;
    }
    fetch(L, want.data(), n_miss, c);
}

bool llama_expert_store::ensure_stage(ggml_backend_dev_t dev, size_t stride) {
    constexpr size_t sector = llama_ssd_reader::sector;
    if (stage_buf != nullptr && dev == stage_dev && stride <= stage_half) {
        return true;
    }
    // another device or a larger expert: the old halves drain first
    for (int h = 0; h < 2; h++) {
        stage_fence(h);
        if (stage_ev[h] != nullptr) {
            ggml_backend_event_free(stage_ev[h]);
            stage_ev[h] = nullptr;
        }
    }
    if (stage_buf != nullptr) {
        ggml_backend_buffer_free(stage_buf);
        stage_buf = nullptr;
        stage     = nullptr;
    }
    // a half holds enough experts to keep several reads in flight
    const size_t half = GGML_PAD(std::max<size_t>(16 * stride, 256ull << 20), sector);
    ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (buft != nullptr) {
        stage_buf = ggml_backend_buft_alloc_buffer(buft, 2 * half + sector);
    }
    if (stage_buf == nullptr) {
        LLAMA_LOG_WARN("%s: expert store: no staging buffer - experts not in RAM are uploaded from the mapping\n", __func__);
        return false;
    }
    if (ggml_backend_buffer_get_type(stage_buf) != buft) {
        // a refused page-lock falls back to a pageable buffer: still correct, but the copies take the staging ring
        LLAMA_LOG_WARN("%s: expert store: the staging buffer is pageable - device uploads of experts not in RAM are slower\n", __func__);
    }
    stage      = (char *) GGML_PAD((uintptr_t) ggml_backend_buffer_get_base(stage_buf), sector);
    stage_half = half;
    stage_dev  = dev;
    stage_next = 0;
    for (int h = 0; h < 2; h++) {
        stage_ev[h]   = ggml_backend_event_new(dev);
        stage_busy[h] = false;
    }
    LLAMA_LOG_INFO("%s: expert store: %.0f MiB staging buffer for device uploads\n", __func__, 2.0 * half / (1024.0 * 1024.0));
    return true;
}

void llama_expert_store::stage_fence(int h) {
    if (!stage_busy[h]) {
        return;
    }
    if (stage_ev[h] != nullptr) {
        ggml_backend_event_synchronize(stage_ev[h]);
    } else {
        ggml_backend_synchronize(stage_backend);
    }
    stage_busy[h] = false;
}

bool llama_expert_store::expert_upload_cb(const ggml_tensor * src, ggml_tensor * dst, const ggml_backend_expert_upload * ups, int n,
        bool ram_only, ggml_backend_t backend, void * user_data) {
    return ((llama_expert_store *) user_data)->upload(src, dst, ups, n, ram_only, backend);
}

bool llama_expert_store::upload(const ggml_tensor * src, ggml_tensor * dst, const ggml_backend_expert_upload * ups, int n,
        bool ram_only, ggml_backend_t backend) {
    constexpr size_t sector = llama_ssd_reader::sector;
    std::lock_guard<std::mutex> lk(mtx);
    const auto ie = ent_of.find(src);
    const auto il = layer_of.find(src);
    if (arena == nullptr || ie == ent_of.end() || il == layer_of.end()) {
        return false;
    }
    const entry & e = ents[ie->second];
    layer & L = layers[il->second];

    // layer-major sweep: an expert not in a slot is copied from its window pieces, read before its group started
    const window_half * W = L.win >= 0 && win_cur >= 0 && ggml_backend_get_device(backend) == win_dev ? &win[win_cur] : nullptr;
    const int32_t * row = W != nullptr ? W->parts[L.win].at.data() + (size_t) e.lay_k * n_expert : nullptr;

    // the scheduler's prefetch on a copy stream: no SSD read. Only the running group's layers, so window_close waits
    // for that stream before the scheduler can free it
    if (ram_only && (W == nullptr || std::any_of(ups, ups + n, [&](const ggml_backend_expert_upload & u) {
            return L.slot_of[u.expert] < 0 && row[u.expert] < 0; }))) {
        return false;
    }

    const bool staged = ensure_stage(ggml_backend_get_device(backend), e.stride);
    auto & c = cnt[2];
    c.passes++;
    c.requests += n;
    // the queued copies read the slots after this returns: an eviction from this layer waits for them first
    if (std::find(L.gpu_readers.begin(), L.gpu_readers.end(), backend) == L.gpu_readers.end()) {
        L.gpu_readers.push_back(backend);
    }

    auto & cw = cnt[3];
    if (W != nullptr) {
        cw.passes++;
    }

    miss.clear();
    for (int i = 0; i < n; i++) {
        const int32_t x = ups[i].expert;
        GGML_ASSERT(x >= 0 && (uint32_t) x < n_expert);
        GGML_ASSERT(ups[i].pad <= sizeof(zero_rows));
        if (L.slot_of[x] >= 0) {
            continue;
        }
        if (W != nullptr) {
            cw.requests++;
            if (row[x] >= 0) {
                continue;
            }
            cw.fallbacks++;   // no window row: the staging buffer
        }
        miss.push_back(i);
    }

    // a window of misses is read into one half of the staging buffer while the other half's copies run
    const size_t per_half = staged ? std::max<size_t>(1, stage_half / e.stride) : 0;
    int cur = -1;
    auto issue = [&](size_t w0) {
        const size_t w1 = std::min(miss.size(), w0 + per_half);
        const int h = stage_next;
        stage_next ^= 1;
        stage_fence(h);
        char * base = stage + (size_t) h * stage_half;
        for (size_t k = w0; k < w1; k++) {
            const int32_t x   = ups[miss[k]].expert;
            const size_t  off = e.file_off + (size_t) x * e.row_bytes;
            const size_t  beg = off & ~(sector - 1);
            const size_t  end = GGML_PAD(off + e.row_bytes, sector);
            reader->read(e.split, beg, end - beg, base + (k - w0) * e.stride, off + e.row_bytes - beg);
            c.bytes += end - beg;
            c.reads++;
        }
        return h;
    };
    if (staged && !miss.empty()) {
        cur = issue(0);
    }

    // resident experts from their slots, the group's others from their window pieces (and the padding) while the
    // first window reads; the pieces stay until window_end, which waits for these copies
    for (int i = 0; i < n; i++) {
        const int32_t x = ups[i].expert;
        if (L.slot_of[x] >= 0) {
            ggml_backend_tensor_set_async(backend, dst, ents[ie->second].tbl[x], ups[i].dst_off, e.row_bytes);
            c.hits++;
        } else if (row != nullptr && row[x] >= 0) {
            // each piece holds a sector-aligned part of the file: copy the part of the rows it holds
            const window_item & it = W->items[row[x]];
            const size_t off = e.file_off + (size_t) x * e.row_bytes;
            for (uint32_t j = it.p0; j < it.p1; j++) {
                const window_piece & pc = W->pieces[j];
                const size_t b0 = std::max(pc.file_off, off);
                const size_t b1 = std::min(pc.file_off + pc.len, off + e.row_bytes);
                if (b0 < b1) {
                    ggml_backend_tensor_set_async(backend, dst, pc.p + (b0 - pc.file_off), ups[i].dst_off + (b0 - off), b1 - b0);
                }
            }
            cw.hits++;
        } else if (!staged) {
            ggml_backend_tensor_set_async(backend, dst, (const char *) e.t->data + (size_t) x * e.row_bytes, ups[i].dst_off, e.row_bytes);
            c.fallbacks++;
        }
        if (ups[i].pad > 0) {
            ggml_backend_tensor_set_async(backend, dst, zero_rows, ups[i].dst_off + e.row_bytes, ups[i].pad);
        }
    }

    for (size_t w0 = 0; staged && w0 < miss.size(); ) {
        const size_t w1 = std::min(miss.size(), w0 + per_half);
        const int64_t t0 = ggml_time_us();
        if (!reader->wait_all()) {
            GGML_ABORT("expert store: an SSD read failed");
        }
        c.wait_us += ggml_time_us() - t0;
        const char * base = stage + (size_t) cur * stage_half;
        for (size_t k = w0; k < w1; k++) {
            const int32_t x   = ups[miss[k]].expert;
            const size_t  off = e.file_off + (size_t) x * e.row_bytes;
            ggml_backend_tensor_set_async(backend, dst, base + (k - w0) * e.stride + (off & (sector - 1)), ups[miss[k]].dst_off, e.row_bytes);
        }
        if (stage_ev[cur] != nullptr) {
            ggml_backend_event_record(stage_ev[cur], backend);
        }
        stage_busy[cur] = true;
        stage_backend   = backend;
        w0 = w1;
        if (w0 < miss.size()) {
            cur = issue(w0);
        }
    }
    return true;
}

void llama_expert_store::detach() {
    std::lock_guard<std::mutex> lk(mtx);
    sweep_stop();
    window_close();
    for (auto & L : layers) {
        wait_readers(L.gpu_readers);
    }
    for (int h = 0; h < 2; h++) {
        stage_fence(h);
    }
    stage_backend = nullptr;
}

void llama_expert_store::window_close() {
    if (win_cur < 0) {
        return;
    }
    // the copies queued from the running group's rows finish before its half takes another group's experts
    for (const auto & P : win[win_cur].parts) {
        layer & L = layers[P.layer];
        wait_readers(L.gpu_readers);
        L.win = -1;
    }
    win_cur = -1;
}

bool llama_expert_store::sweep_start(const layer_groups & groups, ggml_backend_dev_t dev) {
    constexpr size_t sector = llama_ssd_reader::sector;
    // a slot out of service stops the lending: lend_for counts every slot
    const char * why = "the RAM of a lent slot did not come back locked";
    const uint32_t k = slots_out == 0 ? lend_for(groups, n_ram, &why) : 0;
    if (k == 0) {
        if (!lend_refused) {
            lend_refused = true;
            LLAMA_LOG_WARN("%s: expert store: no layer-major window: %s\n", __func__, why);
        }
        return false;
    }
    const int64_t t0 = ggml_time_us();

    // the device copies queued from any slot finish before the lent slots take window rows
    for (auto & L : layers) {
        wait_readers(L.gpu_readers);
    }

    // each layer lends its k least recently used slots, emptied
    std::vector<uint32_t> order(n_ram);
    for (auto & L : layers) {
        for (uint32_t s = 0; s < n_ram; s++) {
            order[s] = s;
        }
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](uint32_t a, uint32_t b) { return L.stamp[a] < L.stamp[b]; });
        L.lent.assign(order.begin(), order.begin() + k);
        for (uint32_t v : L.lent) {
            const int32_t old = L.expert_of[v];
            if (old >= 0) {
                L.slot_of[old] = -1;
                for (uint32_t ei : L.ents) {
                    ents[ei].tbl[old] = nullptr;
                }
            }
            L.expert_of[v] = -1;
            L.stamp[v]     = UINT64_MAX;
        }
    }

    // the staging buffer is the runtime's own pinned allocation: it comes before the window
    {
        size_t stride_max = 0;
        for (const auto & e : ents) {
            stride_max = std::max(stride_max, e.stride);
        }
        ensure_stage(dev, stride_max);
    }

    // a CUDA page-lock of memory the unbuffered SSD reads land in fails those reads, so the lent slots' RAM becomes
    // one pinned buffer; without it the lent slots hold the window and its copies take the backend's staging ring
    std::vector<window_area> runs;
    size_t lent_b = 0;
    if (char * pin = window_pin(dev, (size_t) k * slot_all); pin != nullptr) {
        lent_b = (size_t) k * slot_all;
        runs.push_back({ pin, lent_b });
    } else {
        std::vector<window_area> areas;
        for (const auto & L : layers) {
            for (uint32_t v : L.lent) {
                for (uint32_t ei : L.ents) {
                    areas.push_back({ ents[ei].base + (size_t) v * ents[ei].stride, ents[ei].stride });
                }
            }
        }
        // adjacent areas merge
        std::sort(areas.begin(), areas.end(), [](const window_area & a, const window_area & b) { return a.p < b.p; });
        for (const auto & A : areas) {
            if (!runs.empty() && runs.back().p + runs.back().cap == A.p) {
                runs.back().cap += A.cap;
            } else {
                runs.push_back(A);
            }
            lent_b += A.cap;
        }
    }

    // the first half takes the first half of the bytes, cut at a sector boundary, so either half holds any group
    // (lend_for)
    for (auto & H : win) {
        H = window_half();
    }
    const size_t half0 = (lent_b / 2) & ~(sector - 1);
    size_t taken = 0;
    for (const auto & R : runs) {
        const size_t a = std::min(R.cap, half0 - taken);
        if (a > 0) {
            win[0].areas.push_back({ R.p, a });
            taken += a;
        }
        if (a < R.cap) {
            win[1].areas.push_back({ R.p + a, R.cap - a });
        }
    }

    win_dev  = dev;
    sweep_on = true;

    // its own files and completion port: the next group's reads stay in flight while the store waits for others
    if (ahead == nullptr && !ahead_off) {
        ahead = new llama_ssd_reader();
        if (!ahead->open(file_paths)) {
            delete ahead;
            ahead = nullptr;
            ahead_off = true;
            LLAMA_LOG_WARN("%s: expert store: no second reader - each layer group reads its experts not in RAM when it starts\n", __func__);
        }
    }

    const int64_t us = ggml_time_us() - t0;
    sw.sweeps++;
    sw.lent_bytes += lent_b;
    lend_last = k;
    LLAMA_LOG_INFO("%s: expert store: layer-major sweep borrows %u of %u slots per layer (%.1f GiB) in %.2f s\n",
        __func__, k, n_ram, lent_b / 1073741824.0, us / 1e6);
    return true;
}

void llama_expert_store::sweep_stop() {
    if (!sweep_on) {
        return;
    }
    window_close();
    // the reads still in flight land, and every copy from either half finishes, before the slots go back
    if (win[0].pending || win[1].pending) {
        ahead->wait_all();
    }
    for (auto & L : layers) {
        wait_readers(L.gpu_readers);
    }
    // the pinned window's RAM goes back to the lent slots
    if (win_buf != nullptr) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_buffer_free(win_buf);
        win_buf = nullptr;
        lent_relock();
        const int64_t us = ggml_time_us() - t0;
        sw.pin_stop_us += us;
        LLAMA_LOG_INFO("%s: expert store: pinned window freed and the lent slots' RAM locked again in %.2f s\n", __func__, us / 1e6);
    }
    // back empty, with the oldest stamps: decode refills them first
    for (auto & L : layers) {
        for (uint32_t v : L.lent) {
            L.expert_of[v] = -1;
            L.stamp[v]     = 0;
        }
        L.lent.clear();
    }
    for (auto & H : win) {
        H = window_half();
    }
    win_dev  = nullptr;
    sweep_on = false;
}

// moves the process working-set minimum (and maximum) by delta bytes: released locked pages must leave it, or the
// system keeps them counted for the process while the driver locks the same RAM again
static void ws_adjust(int64_t delta) {
#ifdef _WIN32
    HANDLE proc = GetCurrentProcess();
    SIZE_T ws_min = 0, ws_max = 0;
    DWORD  ws_flags = 0;
    if (GetProcessWorkingSetSizeEx(proc, &ws_min, &ws_max, &ws_flags)) {
        SetProcessWorkingSetSizeEx(proc, (SIZE_T) ((int64_t) ws_min + delta), (SIZE_T) ((int64_t) ws_max + delta), ws_flags);
    }
#else
    GGML_UNUSED(delta);
#endif
}

char * llama_expert_store::window_pin(ggml_backend_dev_t dev, size_t bytes) {
    constexpr size_t sector = llama_ssd_reader::sector;
#ifdef _WIN32
    ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
#else
    // elsewhere the lent slots keep their pages
    ggml_backend_buffer_type_t buft = nullptr;
    GGML_UNUSED(dev);
#endif
    if (buft == nullptr || pin_refused) {
        return nullptr;
    }
    GGML_ASSERT(win_buf == nullptr);
    const int64_t t0 = ggml_time_us();
    for (const auto & L : layers) {
        for (uint32_t v : L.lent) {
            slot_pages(L, v, false);
        }
    }
    ws_lent = bytes;
    ws_adjust(-(int64_t) ws_lent);
    win_buf = ggml_backend_buft_alloc_buffer(buft, bytes + sector);
    if (win_buf != nullptr && ggml_backend_buffer_get_type(win_buf) != buft) {
        // a refused page-lock falls back to a pageable buffer
        ggml_backend_buffer_free(win_buf);
        win_buf = nullptr;
    }
    if (win_buf == nullptr) {
        pin_refused = true;
        LLAMA_LOG_WARN("%s: expert store: no pinned memory for a %.1f GiB layer-major window - it stays in the lent slots, pageable\n",
            __func__, bytes / 1073741824.0);
        lent_relock();
        return nullptr;
    }
    const int64_t us = ggml_time_us() - t0;
    sw.pinned++;
    sw.pin_start_us += us;
    LLAMA_LOG_INFO("%s: expert store: the lent slots' RAM is a %.1f GiB pinned window: pages released and the window allocated in %.2f s\n",
        __func__, bytes / 1073741824.0, us / 1e6);
    return (char *) GGML_PAD((uintptr_t) ggml_backend_buffer_get_base(win_buf), sector);
}

bool llama_expert_store::slot_pages(const layer & L, uint32_t v, bool lock) const {
    bool ok = true;
#ifdef _WIN32
    constexpr size_t page = 4096;
    for (uint32_t ei : L.ents) {
        char * p = ents[ei].base + (size_t) v * ents[ei].stride;
        const size_t n = ents[ei].stride;
        // whole pages: the arena comes from VirtualAlloc and every stride is a multiple of the sector
        GGML_ASSERT(((uintptr_t) p | n) % page == 0);
        if (lock) {
            ok = VirtualAlloc(p, n, MEM_COMMIT, PAGE_READWRITE) == p && VirtualLock(p, n) && ok;
        } else {
            VirtualUnlock(p, n);
            ok = VirtualFree(p, n, MEM_DECOMMIT) && ok;
        }
    }
#else
    GGML_UNUSED(L);
    GGML_UNUSED(v);
    GGML_UNUSED(lock);
#endif
    return ok;
}

void llama_expert_store::lent_relock() {
    if (ws_lent > 0) {
        ws_adjust((int64_t) ws_lent);
        ws_lent = 0;
    }
    uint32_t n_out = 0;
    for (auto & L : layers) {
        // a slot out of service keeps the largest stamp, so it is never a victim
        const auto out = std::remove_if(L.lent.begin(), L.lent.end(), [&](uint32_t v) { return !slot_pages(L, v, true); });
        const uint32_t n = (uint32_t) (L.lent.end() - out);
        L.lent.erase(out, L.lent.end());
        L.n_out += n;
        n_out   += n;
    }
    if (n_out > 0) {
        slots_out += n_out;
        LLAMA_LOG_WARN("%s: expert store: the RAM of %u lent slots did not come back locked - they stay out of service and no sweep lends slots any more\n",
            __func__, n_out);
    }
}

void llama_expert_store::window_plan(int h, const layer_groups & groups, size_t g) {
    constexpr size_t sector = llama_ssd_reader::sector;
    window_half & H = win[h];
    if (H.pending) {
        // reads of another group still land in its areas
        ahead->wait_all();
        H.pending = false;
    }
    H.pieces.clear();
    H.parts.clear();
    H.items.clear();
    H.n_issued = 0;
    H.group    = (int32_t) g;

    // the rows fill the areas in turn, a piece per area they reach; rows the areas no longer hold go through the
    // staging buffer
    size_t free_b = 0;
    for (const auto & A : H.areas) {
        free_b += A.cap;
    }
    size_t a = 0, a_off = 0;
    for (size_t li = 0; li < layers.size(); li++) {
        const layer & L = layers[li];
        if (L.il < groups[g].first || L.il >= groups[g].second) {
            continue;
        }
        window_part P;
        P.layer = (int32_t) li;
        P.at.assign(L.ents.size() * n_expert, -1);
        for (uint32_t k = 0; k < (uint32_t) L.ents.size(); k++) {
            const entry & e = ents[L.ents[k]];
            for (int32_t x = 0; x < (int32_t) n_expert; x++) {
                if (L.slot_of[x] >= 0) {
                    continue;
                }
                const size_t off = e.file_off + (size_t) x * e.row_bytes;
                const size_t end = GGML_PAD(off + e.row_bytes, sector);
                size_t       pos = off & ~(sector - 1);
                if (end - pos > free_b) {
                    sw.unplaced++;
                    continue;
                }
                free_b -= end - pos;
                window_item it = { (uint32_t) H.parts.size(), k, x, (uint32_t) H.pieces.size(), 0 };
                while (pos < end) {
                    const size_t len = std::min(end - pos, H.areas[a].cap - a_off);
                    H.pieces.push_back({ H.areas[a].p + a_off, len, pos, std::min(len, off + e.row_bytes - pos), (uint32_t) e.split });
                    pos   += len;
                    a_off += len;
                    if (a_off == H.areas[a].cap) {
                        a++;
                        a_off = 0;
                    }
                }
                it.p1 = (uint32_t) H.pieces.size();
                P.at[(size_t) k * n_expert + x] = (int32_t) H.items.size();
                H.items.push_back(it);
            }
        }
        H.parts.push_back(std::move(P));
    }
}

void llama_expert_store::window_begin(const layer_groups & groups, size_t g, ggml_backend_dev_t dev) {
    std::lock_guard<std::mutex> lk(mtx);
    window_close();
    if (arena == nullptr || g >= groups.size()) {
        return;
    }
    if (g == 0 || !sweep_on) {
        sweep_stop();
        if (!sweep_start(groups, dev)) {
            return;   // the uploads take the staging buffer
        }
    }

    // pieces[i] of half H; a read ahead the system does not take is read when the group starts
    auto read_piece = [&](window_half & H, size_t i, bool early) {
        const window_piece & pc = H.pieces[i];
        if (early) {
            if (!ahead->try_read(pc.split, pc.file_off, pc.len, pc.p, pc.need)) {
                return false;
            }
            sw.ahead_reads++;
            sw.ahead_bytes += pc.len;
        } else {
            reader->read(pc.split, pc.file_off, pc.len, pc.p, pc.need);
            sw.now_reads++;
            sw.now_bytes += pc.len;
        }
        return true;
    };

    // group g's rows: the reads issued ahead, then the rest now, a bounded batch in flight at a time
    const int h = (int) (g % 2);
    window_half & H = win[h];
    if (H.group != (int32_t) g) {
        window_plan(h, groups, g);
    }
    const int64_t t0 = ggml_time_us();
    if (H.pending) {
        H.pending = false;
        if (!ahead->wait_all()) {
            LLAMA_LOG_WARN("%s: expert store: a read ahead for layers [%d, %d) failed - reading them again\n",
                __func__, groups[g].first, groups[g].second);
            H.n_issued = 0;
        }
    }
    const int64_t t1 = ggml_time_us();
    for (size_t i = H.n_issued; i < H.pieces.size(); ) {
        const size_t i1 = std::min(H.pieces.size(), i + 256);
        for (; i < i1; i++) {
            read_piece(H, i, false);
        }
        if (!reader->wait_all()) {
            GGML_ABORT("expert store: an SSD read failed");
        }
    }
    H.n_issued = H.pieces.size();
    const int64_t t2 = ggml_time_us();
    sw.ahead_wait_us += t1 - t0;
    sw.now_wait_us   += t2 - t1;
    cnt[3].wait_us   += t2 - t0;

    win_cur = h;
    for (size_t p = 0; p < H.parts.size(); p++) {
        layers[H.parts[p].layer].win = (int32_t) p;
    }

    // group g + 1's rows go into the other half while group g computes: the copies from that half (group g - 1's)
    // were waited for by window_close
    if (g + 1 < groups.size() && ahead != nullptr) {
        window_half & N = win[h ^ 1];
        window_plan(h ^ 1, groups, g + 1);
        while (N.n_issued < N.pieces.size() && read_piece(N, N.n_issued, true)) {
            N.n_issued++;
        }
        N.pending = N.n_issued > 0;
        if (N.n_issued < N.pieces.size() && !ahead_refused) {
            ahead_refused = true;
            LLAMA_LOG_WARN("%s: expert store: the system took %zu of %zu reads ahead - the rest is read when the group starts\n",
                __func__, N.n_issued, N.pieces.size());
        }
    }
}

void llama_expert_store::window_end() {
    std::lock_guard<std::mutex> lk(mtx);
    window_close();
}

void llama_expert_store::sweep_end() {
    std::lock_guard<std::mutex> lk(mtx);
    sweep_stop();
}

bool llama_expert_store::has_layer(int32_t il) const {
    return std::any_of(layers.begin(), layers.end(), [il](const layer & L) { return L.il == il; });
}

uint64_t llama_expert_store::wait_us() {
    std::lock_guard<std::mutex> lk(mtx);
    uint64_t us = 0;
    for (const auto & c : cnt) {
        us += c.wait_us;
    }
    return us;
}

int llama_expert_store::host_experts_cb(struct ggml_cgraph * g, int j0, void * user_data) {
    return ((llama_expert_store *) user_data)->host_experts(g, j0);
}

int llama_expert_store::host_experts(struct ggml_cgraph * g, int j0) {
    std::lock_guard<std::mutex> lk(mtx);
    const int n = ggml_graph_n_nodes(g);
    if (arena == nullptr) {
        return n;   // not allocated yet: every expert is read from the mapping
    }
    const ggml_tensor * served_ids   = nullptr;
    uint32_t            served_layer = UINT32_MAX;
    for (int j = j0; j < n; j++) {
        const ggml_tensor * t = ggml_graph_node(g, j);
        // a node that does not run (a layer-major sweep's other layers) has no routing yet
        if (t->op != GGML_OP_MUL_MAT_ID || !(t->flags & GGML_TENSOR_FLAG_COMPUTE) || t->src[0] == nullptr ||
                !(t->src[0]->flags & GGML_TENSOR_FLAG_EXPERT_TABLE)) {
            continue;
        }
        const auto it = layer_of.find(t->src[0]);
        if (it == layer_of.end()) {
            continue;
        }
        const ggml_tensor * ids = t->src[2];
        if (j > j0) {
            // routing computed in this run (the router on the CPU): compute up to here first
            const ggml_tensor * base = ids;
            while (base->view_src != nullptr) {
                base = base->view_src;
            }
            for (int k = j0; k < j; k++) {
                const ggml_tensor * nd = ggml_graph_node(g, k);
                if (nd == ids || nd == base) {
                    return j;
                }
            }
        }
        // the layer's expert tensors share one routing: serve it once
        if (ids == served_ids && it->second == served_layer) {
            continue;
        }
        serve(layers[it->second], ids);
        served_ids   = ids;
        served_layer = it->second;
    }
    return n;
}

void llama_expert_store::seed() {
    llama_pshard_workload wl;
    if (workload_path.empty() || !wl.load(workload_path) || wl.n_expert != n_expert) {
        LLAMA_LOG_WARN("%s: expert store: no route histogram for %u experts at %s - the slots fill on demand\n",
            __func__, n_expert, workload_path.c_str());
        return;
    }
    std::unordered_map<uint32_t, const std::vector<uint64_t> *> counts;
    for (const auto & c : wl.counts) {
        counts[c.first] = &c.second;
    }
    const int64_t t0 = ggml_time_us();
    counters c;   // the seed is not a pass: kept out of the counters
    uint32_t n_layers = 0;
    std::vector<int32_t> order(n_expert);
    for (auto & L : layers) {
        const auto it = counts.find((uint32_t) L.il);
        if (it == counts.end() || it->second->size() != n_expert) {
            continue;
        }
        const std::vector<uint64_t> & h = *it->second;
        for (uint32_t x = 0; x < n_expert; x++) {
            order[x] = (int32_t) x;
        }
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return h[a] > h[b]; });
        // the most routed is read last, so it holds the newest stamp
        std::vector<int32_t> top(order.begin(), order.begin() + n_ram);
        std::reverse(top.begin(), top.end());
        fetch(L, top.data(), top.size(), c);
        n_layers++;
    }
    const double s  = (ggml_time_us() - t0) / 1e6;
    const double gb = c.bytes / 1e9;
    LLAMA_LOG_WARN("%s: expert store: seeded %u layers with their %u most routed experts (%s, %llu routes) in %.1f s, %.1f GB at %.2f GB/s\n",
        __func__, n_layers, n_ram, wl.source.c_str(), (unsigned long long) wl.samples, s, gb, gb / std::max(1e-6, s));
}

void llama_expert_store::log_counters() {
    std::lock_guard<std::mutex> lk(mtx);
    static const char * names[3] = { "one-row passes", "multi-row passes", "device uploads" };
    for (int i = 0; i < 3; i++) {
        const auto & c = cnt[i];
        if (c.passes == 0) {
            continue;
        }
        const double wait_s = c.wait_us / 1e6;
        LLAMA_LOG_WARN("%s: expert store %s: %llu passes, %llu experts: %.1f%% in RAM, %llu SSD reads (%.2f GB, %.2f GB/s while waited, %.1f s waited), %llu read from the mapping\n",
            __func__, names[i], (unsigned long long) c.passes, (unsigned long long) c.requests,
            c.requests > 0 ? 100.0 * c.hits / c.requests : 0.0, (unsigned long long) c.reads, c.bytes / 1e9,
            wait_s > 0 ? c.bytes / 1e9 / wait_s : 0.0, wait_s, (unsigned long long) c.fallbacks);
    }
    if (const auto & c = cnt[3]; c.passes > 0) {
        LLAMA_LOG_WARN("%s: expert store layer-major window: %llu uploads, %llu experts not in RAM: %llu (%.1f%%) from the window rows, %llu through the staging buffer\n",
            __func__, (unsigned long long) c.passes, (unsigned long long) c.requests, (unsigned long long) c.hits,
            c.requests > 0 ? 100.0 * c.hits / c.requests : 0.0, (unsigned long long) c.fallbacks);
    }
    if (sw.sweeps > 0) {
        LLAMA_LOG_WARN("%s: expert store layer-major sweeps: %llu, %u slots per layer lent (%.1f GiB per sweep); "
            "%llu reads ahead (%.2f GB), %llu reads at a group start (%.2f GB); window_begin waited %.1f s for the reads ahead and %.1f s for the others; %llu rows had no area; "
            "%llu pinned windows (%.1f s releasing the slots' pages and allocating, %.1f s freeing and locking them again), %u slots out of service\n",
            __func__, (unsigned long long) sw.sweeps, lend_last, sw.lent_bytes / (double) sw.sweeps / 1073741824.0,
            (unsigned long long) sw.ahead_reads, sw.ahead_bytes / 1e9, (unsigned long long) sw.now_reads, sw.now_bytes / 1e9,
            sw.ahead_wait_us / 1e6, sw.now_wait_us / 1e6, (unsigned long long) sw.unplaced,
            (unsigned long long) sw.pinned, sw.pin_start_us / 1e6, sw.pin_stop_us / 1e6, slots_out);
    }
    if (pf_have) {
        LLAMA_LOG_WARN("%s: expert store: %llu page faults since the first one-row pass\n", __func__,
            (unsigned long long) (process_page_faults() - pf_decode0));
    }
}

void llama_expert_store::fold_workload(const std::string & workload_path) {
    std::lock_guard<std::mutex> lk(mtx);
    // the same floor as the expert pool's: trivial runs stay out
    if (workload_path.empty() || cnt[0].passes + cnt[1].passes < 64) {
        return;
    }
    std::vector<std::pair<uint32_t, std::vector<uint64_t>>> mine;
    uint64_t routes = 0;
    for (auto & L : layers) {
        uint64_t r = 0;
        for (uint64_t v : L.routes) {
            r += v;
        }
        if (r > 0) {
            mine.push_back({ (uint32_t) L.il, L.routes });
            routes += r;
        }
        std::fill(L.routes.begin(), L.routes.end(), 0);
    }
    if (mine.empty()) {
        return;
    }
    // a file holding only the plan-time calibration is replaced by the first real run, real runs accumulate
    llama_pshard_workload wl;
    const bool had = wl.load(workload_path);
    const bool ok  = (had && wl.source == "runtime" && wl.n_expert == n_expert)
        ? wl.merge_counts(mine)
        : wl.set_counts(mine, n_expert, "runtime");
    if (ok) {
        const bool saved = wl.save(workload_path);
        LLAMA_LOG_WARN("%s: expert store workload: zipf_alpha=%.3f (%llu routes accumulated, %llu this run, %u layers)%s%s\n",
            __func__, wl.zipf_alpha, (unsigned long long) wl.samples, (unsigned long long) routes, wl.n_layers,
            saved ? " -> " : " (not saved)", saved ? workload_path.c_str() : "");
    }
}
