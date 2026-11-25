CC ?= gcc
AR ?= ar
OUT ?= /dev/shm/ysyx_compiler_c4vpu/runtime/linux
CPPFLAGS ?=
CFLAGS ?= -O2
WARNFLAGS := -std=c99 -Wall -Wextra -Werror
LDFLAGS ?=
LDLIBS ?=

.PHONY: all check
all: $(OUT)/ysyx_run

$(OUT):
	mkdir -p "$@"

$(OUT)/%.o: %.c ysyx_rt.h z15_regs.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c "$<" -o "$@"

$(OUT)/libysyx_rt.a: $(OUT)/ysyx_rt.o
	$(AR) rcs "$@" "$<"

$(OUT)/ysyx_run: $(OUT)/ysyx_run.o $(OUT)/hal_linux_uio.o $(OUT)/libysyx_rt.a
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o "$@"

$(OUT)/test-z15-abi: tests/z15_abi.c $(OUT)/libysyx_rt.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -I. $(LDFLAGS) $^ $(LDLIBS) -o "$@"

check: $(OUT)/test-z15-abi
	"$<" ../deploy/z15_board_kit/cases/00-mixed/program.yhb
