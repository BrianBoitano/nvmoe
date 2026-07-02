# gcc and a kernel is all you need (io_uring: kernel 5.4+, run outside
# Docker's default seccomp profile — it blocks io_uring_setup).
CC     ?= cc
CFLAGS ?= -O2 -Wall -Wextra

# -ldl: --gpu mode dlopens libcuda.so.1 (the driver API; no CUDA toolkit
# needed at build or run time)
nvmoe-iobench: nvmoe_iobench.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

# fully static build, for running on hosts whose glibc is older than the
# build machine's. CPU-only: dlopen does not work in static binaries.
static: nvmoe_iobench.c
	$(CC) $(CFLAGS) -static -o nvmoe-iobench $<

clean:
	rm -f nvmoe-iobench

.PHONY: static clean
