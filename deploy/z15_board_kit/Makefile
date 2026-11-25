CC ?= gcc
AR ?= ar
OUT ?= build
CPPFLAGS ?=
CFLAGS ?= -O2
LDFLAGS ?=
LDLIBS ?=
WARNFLAGS = -std=c99 -Wall -Wextra -Werror

.PHONY: all
all: $(OUT)/ysyx_run

$(OUT):
	mkdir -p "$@"

$(OUT)/%.o: source/%.c source/ysyx_rt.h source/z15_regs.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c "$<" -o "$@"

$(OUT)/libysyx_rt.a: $(OUT)/ysyx_rt.o
	$(AR) rcs "$@" "$<"

$(OUT)/ysyx_run: $(OUT)/ysyx_run.o $(OUT)/hal_linux_uio.o $(OUT)/libysyx_rt.a
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o "$@"
