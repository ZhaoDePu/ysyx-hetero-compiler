CC ?= gcc
AR ?= ar
OUT ?= build
CPPFLAGS ?=
CFLAGS ?= -O2
LDFLAGS ?=
LDLIBS ?=
DEMO_LINK ?= $(CC)
DEMO_LIBS ?=
WARNFLAGS = -std=c99 -Wall -Wextra -Werror

.PHONY: all demos
all: $(OUT)/ysyx_run demos
demos: $(OUT)/z15_tinylm_demo $(OUT)/z15_yolo26_demo

$(OUT):
	mkdir -p "$@"

$(OUT)/%.o: source/%.c source/ysyx_rt.h source/z15_regs.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c "$<" -o "$@"

$(OUT)/libysyx_rt.a: $(OUT)/ysyx_rt.o
	$(AR) rcs "$@" "$<"

$(OUT)/ysyx_run: $(OUT)/ysyx_run.o $(OUT)/hal_linux_uio.o $(OUT)/libysyx_rt.a
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o "$@"

$(OUT)/demo_support.o $(OUT)/z15_tinylm_demo.o $(OUT)/z15_yolo26_demo.o: source/demo_support.h

$(OUT)/z15_tinylm_demo: $(OUT)/z15_tinylm_demo.o $(OUT)/demo_support.o $(OUT)/hal_linux_uio.o $(OUT)/libysyx_rt.a
	$(DEMO_LINK) $(LDFLAGS) $^ $(DEMO_LIBS) $(LDLIBS) -o "$@"

$(OUT)/z15_yolo26_demo: $(OUT)/z15_yolo26_demo.o $(OUT)/demo_support.o $(OUT)/hal_linux_uio.o $(OUT)/libysyx_rt.a
	$(DEMO_LINK) $(LDFLAGS) $^ $(DEMO_LIBS) $(LDLIBS) -lm -o "$@"
