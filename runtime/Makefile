CC ?= gcc
AR ?= ar
OUT ?= /dev/shm/ysyx_compiler_c5/runtime/linux
CPPFLAGS ?=
CFLAGS ?= -O2
WARNFLAGS := -std=c99 -Wall -Wextra -Werror
LDFLAGS ?=
LDLIBS ?=

.PHONY: all check
all: $(OUT)/ysyx_run $(OUT)/ysyx_yolo26n_cli $(OUT)/libysyx_yolo26n.a

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
	env -u YSYX_C8_GPU_UPLOAD "$<" ../deploy/z15_board_kit/cases/00-mixed/program.yhb ../deploy/z15_board_kit/cases/51-vpu-attention/program.yhb 1752
	YSYX_C8_GPU_UPLOAD=always "$<" ../deploy/z15_board_kit/cases/00-mixed/program.yhb ../deploy/z15_board_kit/cases/51-vpu-attention/program.yhb 3504

$(OUT)/yolo26n_post.o $(OUT)/ysyx_yolo26n_api.o $(OUT)/ysyx_yolo26n_cli.o: yolo26n_post.h ../include/ysyx_yolo26n_api.h
$(OUT)/hal_reference_c6.o $(OUT)/ysyx_yolo26n_api.o: hal_reference_c6.h
$(OUT)/yolo26n_post.o: CFLAGS += -ffp-contract=off

$(OUT)/libysyx_yolo26n.a: $(addprefix $(OUT)/,ysyx_yolo26n_api.o yolo26n_post.o ysyx_rt.o hal_linux_uio.o hal_reference_c6.o)
	$(AR) rcs "$@" $^

$(OUT)/ysyx_yolo26n_cli: $(OUT)/ysyx_yolo26n_cli.o $(OUT)/libysyx_yolo26n.a
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o "$@"

$(OUT)/ysyx_rt.o: c10_session_core.h
$(OUT)/ysyx_yolo26n_api.o: c10_session.h c10_session_core.h

.PHONY: c10-session-check
c10-session-check: $(OUT)/test-c10-session
	"$<" ../deploy/z15_board_kit/cases/00-mixed/program.yhb

$(OUT)/test-c10-session: tests/c10_session.c $(OUT)/libysyx_rt.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -I. $(LDFLAGS) $^ $(LDLIBS) -o "$@"
