CC ?= gcc
AR ?= ar
OUT ?= /dev/shm/ysyx_compiler_c2/runtime/linux
CPPFLAGS ?=
CFLAGS ?= -O2
WARNFLAGS := -std=c99 -Wall -Wextra -Werror
LDFLAGS ?=
LDLIBS ?=

.PHONY: all
all: $(OUT)/ysyx_run

$(OUT):
	mkdir -p "$@"

$(OUT)/%.o: %.c ysyx_rt.h z15_regs.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c "$<" -o "$@"

$(OUT)/libysyx_rt.a: $(OUT)/ysyx_rt.o
	$(AR) rcs "$@" "$<"

$(OUT)/ysyx_run: $(OUT)/ysyx_run.o $(OUT)/hal_linux_uio.o $(OUT)/libysyx_rt.a
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o "$@"
