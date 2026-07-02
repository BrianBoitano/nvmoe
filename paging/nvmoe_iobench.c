/* nvmoe_iobench — random expert-extent fetches from an nvmoe pack, timed.
 *
 * This is the I/O half of the Phase 2 paging library, benchmarked standalone
 * BEFORE any llama.cpp integration (docs/DESIGN.md, component 2). It answers
 * two questions the runtime design hangs on:
 *
 *   1. What does ONE cache miss cost? (QD1 p50/p99 latency = the stall a
 *      synchronous fetch-on-miss adds to a layer's compute)
 *   2. How fast can a prefetcher stream misses? (GB/s at QD 8-32 = the
 *      bandwidth available for hiding fetches behind compute)
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
 * promise is "gcc and a kernel is all you need". Note that Docker's default
 * seccomp profile blocks io_uring_setup (EPERM) — run this on the host, or
 * with a seccomp profile that allows io_uring. O_DIRECT length/offset must
 * be logical-block-aligned; pack extents are 4KB-aligned by construction.
 *
 * Build:  make -C paging          (or: make -C paging static)
 * Run:    python3 tools/pack_extents.py models/olmoe-q4_0.nvmoe
 *         ./paging/nvmoe-iobench models/olmoe-q4_0.nvmoe
 */

#define _GNU_SOURCE
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

/* ------------------------------------------------------------ ring setup */

struct ring {
    int fd;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
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

static void ring_push_read_fixed(struct ring *r, int file_fd, int slot,
                                 void *buf, unsigned len, uint64_t off) {
    unsigned tail = *r->sq_tail;              /* single producer: plain load ok */
    unsigned idx = tail & *r->sq_mask;
    struct io_uring_sqe *s = &r->sqes[idx];
    memset(s, 0, sizeof *s);
    s->opcode = IORING_OP_READ_FIXED;
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

static struct result run_pread(int fd, struct extent *ext, int n_ext,
                               void *buf, int fetches) {
    struct result res;
    uint64_t *lat = malloc(fetches * sizeof *lat);
    uint64_t bytes = 0, t0 = now_ns();
    for (int i = 0; i < fetches; i++) {
        struct extent *e = &ext[rng_below(n_ext)];
        uint64_t t = now_ns();
        ssize_t got = pread(fd, buf, e->len, e->off);
        if (got != (ssize_t) e->len) die("pread (short read?)");
        lat[i] = now_ns() - t;
        bytes += e->len;
    }
    finish_stats(&res, lat, fetches, bytes, now_ns() - t0);
    free(lat);
    return res;
}

static struct result run_uring(int fd, struct extent *ext, int n_ext,
                               char **bufs, size_t buf_len, int qd, int fetches) {
    struct ring r;
    ring_init(&r, MAX_QD);

    struct iovec iov[MAX_QD];
    for (int i = 0; i < qd; i++) {
        iov[i].iov_base = bufs[i];
        iov[i].iov_len = buf_len;
    }
    if (sys_uring_register(r.fd, IORING_REGISTER_BUFFERS, iov, qd) < 0)
        die("io_uring_register(BUFFERS)");

    uint64_t *lat = malloc(fetches * sizeof *lat);
    uint64_t submit_ns[MAX_QD];
    int free_slots[MAX_QD], n_free = qd;
    for (int i = 0; i < qd; i++) free_slots[i] = i;

    int issued = 0, done = 0, to_submit = 0;
    uint64_t bytes = 0, t0 = now_ns();
    while (done < fetches) {
        while (n_free > 0 && issued < fetches) {
            int slot = free_slots[--n_free];
            struct extent *e = &ext[rng_below(n_ext)];
            submit_ns[slot] = now_ns();
            ring_push_read_fixed(&r, fd, slot, bufs[slot], (unsigned) e->len, e->off);
            issued++;
            to_submit++;
        }
        int ret = sys_uring_enter(r.fd, to_submit, 1, IORING_ENTER_GETEVENTS);
        if (ret < 0) {
            if (errno == EINTR) continue;
            die("io_uring_enter");
        }
        to_submit = 0;
        unsigned head = *r.cq_head;
        unsigned tail = __atomic_load_n(r.cq_tail, __ATOMIC_ACQUIRE);
        while (head != tail) {
            struct io_uring_cqe *c = &r.cqes[head & *r.cq_mask];
            int slot = (int) c->user_data;
            if (c->res < 0) {
                errno = -c->res;
                die("read completion");
            }
            lat[done++] = now_ns() - submit_ns[slot];
            bytes += (uint64_t) c->res;
            free_slots[n_free++] = slot;
            head++;
        }
        __atomic_store_n(r.cq_head, head, __ATOMIC_RELEASE);
    }
    uint64_t wall = now_ns() - t0;

    struct result res;
    finish_stats(&res, lat, fetches, bytes, wall);
    free(lat);
    sys_uring_register(r.fd, IORING_UNREGISTER_BUFFERS, NULL, 0);
    close(r.fd);
    return res;
}

static void print_row(const char *mode, int qd, struct result *r) {
    printf("%-6s %4d %9d %8.2f %8.2f %8.0f %9.3f %8.3f %8.3f %8.3f\n",
           mode, qd, r->fetches, r->bytes / 1073741824.0, r->gb_s,
           r->fetch_s, r->p50, r->p95, r->p99, r->max);
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    int fetches = 3000;
    uint64_t seed = 42;
    const char *qd_list = "1,2,4,8,16,32";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fetches") && i + 1 < argc) fetches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--qd") && i + 1 < argc) qd_list = argv[++i];
        else if (argv[i][0] != '-' && !dir) dir = argv[i];
        else {
            fprintf(stderr, "usage: %s <pack_dir> [--fetches N] [--qd 1,4,32] [--seed S]\n"
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

    char *bufs[MAX_QD];
    for (int i = 0; i < MAX_QD; i++)
        if (posix_memalign((void **) &bufs[i], 4096, max_len))
            die("posix_memalign");

    printf("nvmoe iobench: %s — %d extents, %.1f-%.1fMB, %.1fGB total, "
           "%d fetches/point, seed %" PRIu64 "\n",
           dir, n_ext, (double) min_len / 1048576.0, (double) max_len / 1048576.0,
           total / 1073741824.0, fetches, seed);
    printf("%-6s %4s %9s %8s %8s %8s %9s %8s %8s %8s\n",
           "mode", "qd", "fetches", "GiB", "GB/s", "fetch/s",
           "p50_ms", "p95_ms", "p99_ms", "max_ms");

    struct result r;
    rng_state = seed;
    r = run_pread(fd, ext, n_ext, bufs[0], fetches);
    print_row("pread", 1, &r);

    char qds[64];
    snprintf(qds, sizeof qds, "%s", qd_list);
    for (char *tok = strtok(qds, ","); tok; tok = strtok(NULL, ",")) {
        int qd = atoi(tok);
        if (qd < 1 || qd > MAX_QD) {
            fprintf(stderr, "skip qd=%s (1..%d)\n", tok, MAX_QD);
            continue;
        }
        rng_state = seed;   /* same extent sequence for every mode */
        r = run_uring(fd, ext, n_ext, bufs, max_len, qd, fetches);
        print_row("uring", qd, &r);
    }
    return 0;
}
