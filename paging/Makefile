# gcc and a kernel is all you need (io_uring: kernel 5.4+, run outside
# Docker's default seccomp profile — it blocks io_uring_setup).
CC     ?= cc
CFLAGS ?= -O2 -Wall -Wextra

nvmoe-iobench: nvmoe_iobench.c
	$(CC) $(CFLAGS) -o $@ $<

# fully static build, for running on hosts whose glibc is older than the
# build machine's (e.g. built in a container, run on the bare-metal host)
static: nvmoe_iobench.c
	$(CC) $(CFLAGS) -static -o nvmoe-iobench $<

clean:
	rm -f nvmoe-iobench

.PHONY: static clean
