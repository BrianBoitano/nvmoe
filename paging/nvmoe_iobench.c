/* nvmoe_iobench — random expert-extent fetches from an nvmoe pack, timed.
 *
 * This is the I/O half of the Phase 2 paging library, benchmarked standalone
 * BEFORE any llama.cpp integration (docs/DESIGN.md, component 2). It answers
 * the questions the runtime design hangs on:
 *
 *   1. What does ONE cache miss cost? (QD1 p50/p99 latency = the stall a
 *      synchronous fetch-on-miss adds to a layer's compute)
 *   2. How fast can a prefetcher stream misses? (GB/s at QD 2-8 = the
 *      bandwidth available for hiding fetches behind compute)
 *   3. With --gpu: does that speed survive the full path into VRAM?
 *      (NVMe -> pinned staging -> cudaMemcpyHtoDAsync -> VRAM slab)
 *
 * Method: uniform-random extents from a real experts.pack, read via io_uring
 * + O_DIRECT + registered buffers into 4KB-aligned staging buffers. Uniform
 * random is deliberately pessimistic: real routing is skewed and sticky, but
 * the reads that reach the SSD are exactly the ones the VRAM cache missed,
 * so the cold-ish, scattered case is the one to measure. A plain pread(2)
 * QD1 baseline runs first for an apples-to-apples syscall-path comparison
 * (and because pread is the fallback path on kernels without io_uring).
 *
 * Raw syscalls, no liburing: the ring setup is ~80 lines and the repo
 * promise is "gcc and a kernel is all you need". GPU mode needs no CUDA
 * toolkit either — the CUDA *driver* API (libcuda.so.1, ships with the
 * NVIDIA driver) is loaded with dlopen and ~20 self-declared prototypes, so
 * the same plain-gcc build runs CPU-only anywhere and does the VRAM path
 * wherever a driver exists. Note that Docker's default seccomp profile
 * blocks io_uring_setup (EPERM) — run on the host, or with a seccomp
 * profile that allows io_uring. O_DIRECT length/offset must be
 * logical-block-aligned; pack extents are 4KB-aligned by construction.
 *
 * Build:  make -C paging          (or: make -C paging static — CPU-only)
 * Run:    python3 tools/pack_extents.py models/olmoe-q4_0.nvmoe
 *         ./paging/nvmoe-iobench models/olmoe-q4_0.nvmoe [--gpu]
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/io_uring.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#define __NR_io_uring_enter 426
#define __NR_io_uring_register 427
#endif

#define MAX_QD 64

static int sys_uring_setup(unsigned entries, struct io_uring_params *p) {
    return (int) syscall(__NR_io_uring_setup, entries, p);
}
static int sys_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return (int) syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, NULL, 0);
}
static int sys_uring_register(int fd, unsigned op, void *arg, unsigned nr) {
    return (int) syscall(__NR_io_uring_register, fd, op, arg, nr);
}

static void die(const char *what) {
    fprintf(stderr, "fatal: %s: %s\n", what, strerror(errno));
    exit(1);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* xorshift64* — deterministic across runs, seedable for reproducibility */
static uint64_t rng_state;
static uint64_t rng64(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}
static uint32_t rng_below(uint32_t n) {   /* multiply-shift, avoids modulo bias */
    return (uint32_t) (((rng64() >> 32) * n) >> 32);
}

/* --------------------------------------------------- CUDA driver API shim
 * Minimal hand-declared slice of the driver API. Symbol names must be the
 * versioned ABI entry points (cuMemAlloc_v2 etc.) — that is what cuda.h's
 * macros resolve to; the unversioned names are the pre-CUDA-3.2 32-bit ABI.
 */

typedef int CUresult;                     /* CUDA_SUCCESS == 0 */
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext, *CUstream, *CUevent;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_NOT_READY 600
#define CU_EVENT_DISABLE_TIMING 0x2

struct cu {
    CUresult (*Init)(unsigned);
    CUresult (*DriverGetVersion)(int *);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DeviceGetName)(char *, int, CUdevice);
    CUresult (*DevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemHostAlloc)(void **, size_t, unsigned);
    CUresult (*MemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
    CUresult (*MemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*StreamSynchronize)(CUstream);
    CUresult (*EventCreate)(CUevent *, unsigned);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventQuery)(CUevent);
    CUresult (*GetErrorString)(CUresult, const char **);
};

struct gpu {
    struct cu cu;
    CUdeviceptr slab;
    size_t slab_bytes;
    int n_slots;                  /* slab_bytes / max extent len */
    uint64_t slot_counter;        /* round-robin VRAM destination */
    CUstream stream[MAX_QD];
    CUevent event[MAX_QD];
};

static void cu_die(struct gpu *g, const char *what, CUresult r) {
    const char *s = "?";
    if (g && g->cu.GetErrorString) g->cu.GetErrorString(r, &s);
    fprintf(stderr, "fatal: %s: CUresult %d (%s)\n", what, r, s);
    exit(1);
}
#define CU_CHECK(g, call) do { CUresult r_ = (call); \
    if (r_ != CUDA_SUCCESS) cu_die((g), #call, r_); } while (0)

static void *cu_sym(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "fatal: libcuda.so.1 lacks %s\n", name); exit(1); }
    return p;
}

static struct gpu *gpu_init(size_t slab_want, size_t slot_bytes, int qd) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "fatal: dlopen libcuda.so.1: %s\n"
                "hint: --gpu needs the NVIDIA driver (not the toolkit); "
                "run where nvidia-smi works\n", dlerror());
        exit(1);
    }
    struct gpu *g = calloc(1, sizeof *g);
    struct cu *c = &g->cu;
    c->Init                 = cu_sym(lib, "cuInit");
    c->DriverGetVersion     = cu_sym(lib, "cuDriverGetVersion");
    c->DeviceGet            = cu_sym(lib, "cuDeviceGet");
    c->DeviceGetName        = cu_sym(lib, "cuDeviceGetName");
    c->DevicePrimaryCtxRetain = cu_sym(lib, "cuDevicePrimaryCtxRetain");
    c->CtxSetCurrent        = cu_sym(lib, "cuCtxSetCurrent");
    c->MemGetInfo           = cu_sym(lib, "cuMemGetInfo_v2");
    c->MemAlloc             = cu_sym(lib, "cuMemAlloc_v2");
    c->MemHostAlloc         = cu_sym(lib, "cuMemHostAlloc");
    c->MemcpyHtoDAsync      = cu_sym(lib, "cuMemcpyHtoDAsync_v2");
    c->MemcpyDtoH           = cu_sym(lib, "cuMemcpyDtoH_v2");
    c->StreamCreate         = cu_sym(lib, "cuStreamCreate");
    c->StreamSynchronize    = cu_sym(lib, "cuStreamSynchronize");
    c->EventCreate          = cu_sym(lib, "cuEventCreate");
    c->EventRecord          = cu_sym(lib, "cuEventRecord");
    c->EventQuery           = cu_sym(lib, "cuEventQuery");
    c->GetErrorString       = cu_sym(lib, "cuGetErrorString");

    CU_CHECK(g, c->Init(0));
    int ver = 0;
    c->DriverGetVersion(&ver);
    CUdevice dev;
    CU_CHECK(g, c->DeviceGet(&dev, 0));
    char name[128] = "?";
    c->DeviceGetName(name, sizeof name, dev);
    CUcontext ctx;
    CU_CHECK(g, c->DevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(g, c->CtxSetCurrent(ctx));
    size_t free_b = 0, total_b = 0;
    c->MemGetInfo(&free_b, &total_b);

    /* modest slab, halved until it fits — this measures PCIe behavior, not
     * capacity, and the dev box GPU usually has models resident */
    size_t want = slab_want;
    CUresult r;
    while ((r = c->MemAlloc(&g->slab, want)) != CUDA_SUCCESS && want > (64u << 20))
        want /= 2;
    if (r != CUDA_SUCCESS) cu_die(g, "cuMemAlloc(slab)", r);
    g->slab_bytes = want;
    g->n_slots = (int) (want / slot_bytes);
    if (g->n_slots < qd) {
        fprintf(stderr, "fatal: VRAM slab (%zuMB) holds %d slots < qd %d\n",
                want >> 20, g->n_slots, qd);
        exit(1);
    }
    for (int i = 0; i < MAX_QD; i++) {
        CU_CHECK(g, c->StreamCreate(&g->stream[i], 0));
        CU_CHECK(g, c->EventCreate(&g->event[i], CU_EVENT_DISABLE_TIMING));
    }
    printf("gpu: %s, driver %d.%d, VRAM %zu/%zuMB free, slab %zuMB (%d slots)\n",
           name, ver / 1000, ver % 1000 / 10, free_b >> 20, total_b >> 20,
           g->slab_bytes >> 20, g->n_slots);
    return g;
}

static CUdeviceptr gpu_next_dst(struct gpu *g, size_t slot_bytes) {
    CUdeviceptr d = g->slab + (g->slot_counter % g->n_slots) * slot_bytes;
    g->slot_counter++;
    return d;
}

/* ------------------------------------------------------------ ring setup */

struct ring {
    int fd;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    int use_fixed;                 /* registered buffers, else plain READ */
};

static void ring_init(struct ring *r, unsigned entries) {
    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    r->fd = sys_uring_setup(entries, &p);
    if (r->fd < 0) {
        if (errno == EPERM)
            fprintf(stderr, "hint: io_uring is blocked here (Docker's default seccomp "
                            "profile does this) — run on the host\n");
        die("io_uring_setup");
    }
    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (!(p.features & IORING_FEAT_SINGLE_MMAP)) {
        fprintf(stderr, "fatal: kernel lacks IORING_FEAT_SINGLE_MMAP (pre-5.4?)\n");
        exit(1);
    }
    size_t ring_sz = sq_sz > cq_sz ? sq_sz : cq_sz;
    char *rp = mmap(NULL, ring_sz, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, r->fd, IORING_OFF_SQ_RING);
    if (rp == MAP_FAILED) die("mmap sq ring");
    r->sqes = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe),
                   PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                   r->fd, IORING_OFF_SQES);
    if (r->sqes == MAP_FAILED) die("mmap sqes");
    r->sq_head  = (unsigned *) (rp + p.sq_off.head);
    r->sq_tail  = (unsigned *) (rp + p.sq_off.tail);
    r->sq_mask  = (unsigned *) (rp + p.sq_off.ring_mask);
    r->sq_array = (unsigned *) (rp + p.sq_off.array);
    r->cq_head  = (unsigned *) (rp + p.cq_off.head);
    r->cq_tail  = (unsigned *) (rp + p.cq_off.tail);
    r->cq_mask  = (unsigned *) (rp + p.cq_off.ring_mask);
    r->cqes     = (struct io_uring_cqe *) (rp + p.cq_off.cqes);
}

static void ring_push_read(struct ring *r, int file_fd, int slot,
                           void *buf, unsigned len, uint64_t off) {
    unsigned tail = *r->sq_tail;              /* single producer: plain load ok */
    unsigned idx = tail & *r->sq_mask;
    struct io_uring_sqe *s = &r->sqes[idx];
    memset(s, 0, sizeof *s);
    s->opcode = r->use_fixed ? IORING_OP_READ_FIXED : IORING_OP_READ;
    s->fd = file_fd;
    s->addr = (uint64_t) (uintptr_t) buf;
    s->len = len;
    s->off = off;
    s->buf_index = (uint16_t) slot;
    s->user_data = (uint64_t) slot;
    r->sq_array[idx] = idx;
    __atomic_store_n(r->sq_tail, tail + 1, __ATOMIC_RELEASE);
}

/* ------------------------------------------------------------ benchmark */

struct extent { uint64_t off, len; };

struct result {
    double wall_s, gb_s, fetch_s;
    double p50, p95, p99, max;    /* ms */
    uint64_t bytes;
    int fetches;
};

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return x < y ? -1 : x > y;
}

static void finish_stats(struct result *res, uint64_t *lat, int n,
                         uint64_t bytes, uint64_t wall) {
    qsort(lat, n, sizeof *lat, cmp_u64);
    res->fetches = n;
    res->bytes = bytes;
    res->wall_s = wall / 1e9;
    res->gb_s = bytes / (double) wall;                  /* bytes/ns == GB/s */
    res->fetch_s = n / res->wall_s;
    res->p50 = lat[n / 2] / 1e6;
    res->p95 = lat[(int) (n * 0.95)] / 1e6;
    res->p99 = lat[(int) (n * 0.99)] / 1e6;
    res->max = lat[n - 1] / 1e6;
}

/* pread baseline; with g: read -> sync H2D per fetch (the unpipelined path) */
static struct result run_pread(int fd, struct extent *ext, int n_ext,
                               void *buf, int fetches, struct gpu *g,
                               size_t slot_bytes) {
    struct result res;
    uint64_t *lat = malloc(fetches * sizeof *lat);
    uint64_t bytes = 0, t0 = now_ns();
    for (int i = 0; i < fetches; i++) {
        struct extent *e = &ext[rng_below(n_ext)];
        uint64_t t = now_ns();
        ssize_t got = pread(fd, buf, e->len, e->off);
        if (got != (ssize_t) e->len) die("pread (short read?)");
        if (g) {
            CU_CHECK(g, g->cu.MemcpyHtoDAsync(gpu_next_dst(g, slot_bytes), buf,
                                              e->len, g->stream[0]));
            CU_CHECK(g, g->cu.StreamSynchronize(g->stream[0]));
        }
        lat[i] = now_ns() - t;
        bytes += e->len;
    }
    finish_stats(&res, lat, fetches, bytes, now_ns() - t0);
    free(lat);
    return res;
}

/* io_uring pipeline; with g each read completion chains an async H2D copy,
 * and a fetch counts as done when its copy event fires — latency is
 * "runtime wants this expert" to "expert is in VRAM". */
static struct result run_uring(int fd, struct extent *ext, int n_ext,
                               char **bufs, size_t buf_len, int qd, int fetches,
                               struct gpu *g) {
    struct ring r;
    ring_init(&r, MAX_QD);

    struct iovec iov[MAX_QD];
    for (int i = 0; i < qd; i++) {
        iov[i].iov_base = bufs[i];
        iov[i].iov_len = buf_len;
    }
    /* pinned (driver-allocated) pages usually register fine; if not, fall
     * back to plain READ — same O_DIRECT path, no fixed-buffer fast path */
    r.use_fixed = sys_uring_register(r.fd, IORING_REGISTER_BUFFERS, iov, qd) == 0;
    if (!r.use_fixed) {
        if (!g) die("io_uring_register(BUFFERS)");   /* unexpected for plain memory */
        static int warned;
        if (!warned++)
            printf("note: pinned buffers rejected by IORING_REGISTER_BUFFERS "
                   "(%s) — using plain READ\n", strerror(errno));
    }

    uint64_t *lat = malloc(fetches * sizeof *lat);
    uint64_t submit_ns[MAX_QD];
    unsigned ext_len[MAX_QD];
    enum { S_FREE, S_COPYING } state[MAX_QD] = { S_FREE };
    int free_slots[MAX_QD], n_free = qd;
    for (int i = 0; i < qd; i++) free_slots[i] = i;

    int issued = 0, done = 0, to_submit = 0, reads_inflight = 0, copies = 0;
    uint64_t bytes = 0, t0 = now_ns();
    while (done < fetches) {
        while (n_free > 0 && issued < fetches) {
            int slot = free_slots[--n_free];
            struct extent *e = &ext[rng_below(n_ext)];
            submit_ns[slot] = now_ns();
            ext_len[slot] = (unsigned) e->len;
            ring_push_read(&r, fd, slot, bufs[slot], (unsigned) e->len, e->off);
            issued++;
            to_submit++;
        }
        if (to_submit > 0 || reads_inflight > 0) {
            /* block for read completions only if no copies could free a slot
             * meanwhile; with copies pending, submit-and-poll instead */
            unsigned min_c = (copies > 0) ? 0 : 1;
            int ret = sys_uring_enter(r.fd, to_submit, min_c,
                                      min_c ? IORING_ENTER_GETEVENTS : 0);
            if (ret < 0) {
                if (errno == EINTR) continue;
                die("io_uring_enter");
            }
            reads_inflight += to_submit;
            to_submit = 0;
        }
        unsigned head = *r.cq_head;
        unsigned tail = __atomic_load_n(r.cq_tail, __ATOMIC_ACQUIRE);
        while (head != tail) {
            struct io_uring_cqe *c = &r.cqes[head & *r.cq_mask];
            int slot = (int) c->user_data;
            if (c->res < 0) {
                errno = -c->res;
                die("read completion");
            }
            if ((unsigned) c->res != ext_len[slot]) {
                fprintf(stderr, "fatal: short read (%d of %u)\n", c->res, ext_len[slot]);
                exit(1);
            }
            reads_inflight--;
            if (g) {
                CU_CHECK(g, g->cu.MemcpyHtoDAsync(gpu_next_dst(g, buf_len),
                                                  bufs[slot], ext_len[slot],
                                                  g->stream[slot]));
                CU_CHECK(g, g->cu.EventRecord(g->event[slot], g->stream[slot]));
                state[slot] = S_COPYING;
                copies++;
            } else {
                lat[done++] = now_ns() - submit_ns[slot];
                bytes += ext_len[slot];
                free_slots[n_free++] = slot;
            }
            head++;
        }
        __atomic_store_n(r.cq_head, head, __ATOMIC_RELEASE);
        if (g && copies > 0) {
            for (int slot = 0; slot < qd; slot++) {
                if (state[slot] != S_COPYING) continue;
                CUresult q = g->cu.EventQuery(g->event[slot]);
                if (q == CUDA_ERROR_NOT_READY) continue;
                if (q != CUDA_SUCCESS) cu_die(g, "cuEventQuery", q);
                lat[done++] = now_ns() - submit_ns[slot];
                bytes += ext_len[slot];
                state[slot] = S_FREE;
                free_slots[n_free++] = slot;
                copies--;
            }
        }
    }
    uint64_t wall = now_ns() - t0;

    struct result res;
    finish_stats(&res, lat, fetches, bytes, wall);
    free(lat);
    if (r.use_fixed) sys_uring_register(r.fd, IORING_UNREGISTER_BUFFERS, NULL, 0);
    close(r.fd);
    return res;
}

/* one fetch through the whole path, then read back and memcmp — proves the
 * bytes that land in VRAM are the bytes that live on disk */
static void gpu_verify(int fd, struct extent *e, char *buf, struct gpu *g) {
    if (pread(fd, buf, e->len, e->off) != (ssize_t) e->len) die("verify pread");
    CUdeviceptr dst = g->slab;
    CU_CHECK(g, g->cu.MemcpyHtoDAsync(dst, buf, e->len, g->stream[0]));
    CU_CHECK(g, g->cu.StreamSynchronize(g->stream[0]));
    char *back = malloc(e->len);
    CU_CHECK(g, g->cu.MemcpyDtoH(back, dst, e->len));
    if (memcmp(buf, back, e->len) != 0) {
        fprintf(stderr, "fatal: VRAM round-trip mismatch\n");
        exit(1);
    }
    free(back);
    printf("gpu: NVMe -> pinned -> VRAM -> host round trip byte-verified "
           "(%.1fMB extent)\n", e->len / 1048576.0);
}

static void print_row(const char *mode, int qd, struct result *r) {
    printf("%-9s %4d %9d %8.2f %8.2f %8.0f %9.3f %8.3f %8.3f %8.3f\n",
           mode, qd, r->fetches, r->bytes / 1073741824.0, r->gb_s,
           r->fetch_s, r->p50, r->p95, r->p99, r->max);
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int fetches = 3000, use_gpu = 0;
    uint64_t seed = 42;
    size_t slab_mb = 512;
    const char *qd_list = "1,2,4,8,16,32";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fetches") && i + 1 < argc) fetches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--qd") && i + 1 < argc) qd_list = argv[++i];
        else if (!strcmp(argv[i], "--slab-mb") && i + 1 < argc) slab_mb = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--gpu")) use_gpu = 1;
        else if (argv[i][0] != '-' && !dir) dir = argv[i];
        else {
            fprintf(stderr, "usage: %s <pack_dir> [--gpu] [--fetches N] [--qd 1,4,32] "
                            "[--seed S] [--slab-mb M]\n"
                            "  <pack_dir> needs experts.pack + extents.tsv "
                            "(tools/pack_extents.py)\n", argv[0]);
            return 2;
        }
    }
    if (!dir || fetches < 100) {
        fprintf(stderr, "need a pack dir and --fetches >= 100\n");
        return 2;
    }

    char path[4096];
    snprintf(path, sizeof path, "%s/extents.tsv", dir);
    FILE *tsv = fopen(path, "r");
    if (!tsv) {
        fprintf(stderr, "fatal: %s missing — run: python3 tools/pack_extents.py %s\n",
                path, dir);
        return 1;
    }
    int cap = 1024, n_ext = 0;
    struct extent *ext = malloc(cap * sizeof *ext);
    uint64_t max_len = 0, min_len = UINT64_MAX, total = 0;
    while (fscanf(tsv, "%" SCNu64 "\t%" SCNu64, &ext[n_ext].off, &ext[n_ext].len) == 2) {
        if (ext[n_ext].len > max_len) max_len = ext[n_ext].len;
        if (ext[n_ext].len < min_len) min_len = ext[n_ext].len;
        total += ext[n_ext].len;
        if (++n_ext == cap) ext = realloc(ext, (cap *= 2) * sizeof *ext);
    }
    fclose(tsv);
    if (n_ext == 0) { fprintf(stderr, "fatal: no extents parsed\n"); return 1; }

    snprintf(path, sizeof path, "%s/experts.pack", dir);
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) die("open experts.pack (O_DIRECT)");

    struct gpu *g = use_gpu ? gpu_init(slab_mb << 20, max_len, MAX_QD) : NULL;

    /* staging buffers: pinned via the driver in gpu mode (DMA-able for both
     * O_DIRECT and cudaMemcpyAsync), plain aligned pages otherwise */
    char *bufs[MAX_QD];
    for (int i = 0; i < MAX_QD; i++) {
        if (g)
            CU_CHECK(g, g->cu.MemHostAlloc((void **) &bufs[i], max_len, 0));
        else if (posix_memalign((void **) &bufs[i], 4096, max_len))
            die("posix_memalign");
        if ((uintptr_t) bufs[i] % 4096)
            die("staging buffer not 4KB-aligned (O_DIRECT needs it)");
    }

    printf("nvmoe iobench: %s — %d extents, %.1f-%.1fMB, %.1fGB total, "
           "%d fetches/point, seed %" PRIu64 "%s\n",
           dir, n_ext, (double) min_len / 1048576.0, (double) max_len / 1048576.0,
           total / 1073741824.0, fetches, seed,
           g ? ", pinned staging" : "");
    if (g) gpu_verify(fd, &ext[0], bufs[0], g);
    printf("%-9s %4s %9s %8s %8s %8s %9s %8s %8s %8s\n",
           "mode", "qd", "fetches", "GiB", "GB/s", "fetch/s",
           "p50_ms", "p95_ms", "p99_ms", "max_ms");

    struct result r;
    rng_state = seed;
    r = run_pread(fd, ext, n_ext, bufs[0], fetches, g, max_len);
    print_row(g ? "pread+h2d" : "pread", 1, &r);

    char qds[64];
    snprintf(qds, sizeof qds, "%s", qd_list);
    for (char *tok = strtok(qds, ","); tok; tok = strtok(NULL, ",")) {
        int qd = atoi(tok);
        if (qd < 1 || qd > MAX_QD) {
            fprintf(stderr, "skip qd=%s (1..%d)\n", tok, MAX_QD);
            continue;
        }
        rng_state = seed;   /* same extent sequence for every mode */
        r = run_uring(fd, ext, n_ext, bufs, max_len, qd, fetches, g);
        print_row(g ? "uring+h2d" : "uring", qd, &r);
    }
    return 0;
}
