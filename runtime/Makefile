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

.PHONY: c10
c10: $(OUT)/z15_yolo_live_c10

$(OUT)/c10_service.o: c10_session.h c10_session_core.h c10_pixels.h c10_v4l2.h
$(OUT)/c10_v4l2.o: c10_v4l2.h c10_pixels.h
$(OUT)/c10_pixels.o: c10_pixels.h c10_jpeg.h
$(OUT)/c10_jpeg.o: c10_jpeg.h c10_jpeg_tables.h

$(OUT)/z15_yolo_live_c10: $(addprefix $(OUT)/,c10_service.o c10_pixels.o c10_v4l2.o c10_jpeg.o libysyx_yolo26n.a)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -lm -o "$@"

c10: $(OUT)/z15_c10_benchmark

$(OUT)/c10_benchmark.o: c10_session.h c10_session_core.h hal_reference_c6.h

$(OUT)/z15_c10_benchmark: $(addprefix $(OUT)/,c10_benchmark.o ysyx_rt.o hal_linux_uio.o hal_reference_c6.o)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -lm -o "$@"

.PHONY: c11-qwen
c11-qwen: $(OUT)/ysyx_qwen_chat

$(OUT)/c11_qwen_ref.o $(OUT)/c11_qwen_chat.o: c11_qwen_ref.h
$(OUT)/c11_qwen_bpe.o $(OUT)/c11_qwen_chat.o: c11_qwen_bpe.h

$(OUT)/ysyx_qwen_chat: $(addprefix $(OUT)/,c11_qwen_chat.o c11_qwen_bpe.o c11_qwen_ref.o)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o "$@"

.PHONY: c11b-qwen
c11b-qwen: $(OUT)/z15_qwen25_chat

$(OUT)/c11b_rope_words.h: ../tools/c11b_emit_rope.py ../hetero_compiler/c11_qwen_rope.py | $(OUT)
	cd .. && python3 -B -m tools.c11b_emit_rope > "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/c11b_qwen_z15.o: c11b_qwen_z15.c c11b_qwen_ops.h $(OUT)/c11b_rope_words.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -I"$(OUT)" -c "$<" -o "$@"

$(OUT)/c11b_qwen_ref.o: c11_qwen_ref.c c11_qwen_ref.h c11b_reference.inc c11b_qwen_ops.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DYSYX_C11B_NATIVE -c "$<" -o "$@"

$(OUT)/c11b_qwen_chat.o: c11_qwen_chat.c c11_qwen_ref.h c11_qwen_bpe.h c11b_qwen_ops.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DYSYX_C11B_NATIVE -c "$<" -o "$@"

$(OUT)/z15_qwen25_chat: $(addprefix $(OUT)/,c11b_qwen_chat.o c11b_qwen_ref.o c11_qwen_bpe.o c11b_qwen_ops.o c11b_qwen_z15.o hal_linux_uio.o)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o "$@"
