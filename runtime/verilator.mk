CC ?= gcc
CXX ?= g++
CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror
CXXFLAGS := -std=c++17 -O2

OBJECTS := $(addprefix $(OUT)/,ysyx_rt.o z15_cmdq.o hal_linux_uio.o ysyx_run.o hal_verilator_z15.o)
RTL_OBJECTS := $(addprefix $(RTL_BUILD)/,Vz15_bd_axi_shell__ALL.a verilated.o verilated_dpi.o verilated_threads.o)

.PHONY: all
all: $(OUT)/ysyx_run $(OUT)/ysyx_yolo26n_cli

$(OUT)/%.o: %.c ysyx_rt.h z15_regs.h
	$(CC) $(CFLAGS) -DYSYX_WITH_VERILATOR -c "$<" -o "$@"

$(OUT)/hal_verilator_z15.o: hal_verilator_z15.cpp ysyx_rt.h $(OUT)/observed_rig.inc
	$(CXX) $(CXXFLAGS) -I"$(OUT)" -I"$(RTL_BUILD)" -I"$(RTL_SRC)/app" \
	    -isystem "$(SDK)/include" -isystem "$(SDK)/include/vltstd" -c "$<" -o "$@"

$(OUT)/ysyx_run: $(OBJECTS) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/yolo26n_post.o $(OUT)/ysyx_yolo26n_api.o $(OUT)/ysyx_yolo26n_cli.o: yolo26n_post.h ../include/ysyx_yolo26n_api.h
$(OUT)/ysyx_yolo26n_api.o $(OUT)/ysyx_yolo26n_cli.o: yolo26n_target.h
$(OUT)/yolo26n_post.o: CFLAGS += -ffp-contract=off

$(OUT)/ysyx_yolo26n_cli: $(addprefix $(OUT)/,ysyx_yolo26n_cli.o ysyx_yolo26n_api.o yolo26n_post.o ysyx_rt.o z15_cmdq.o hal_linux_uio.o hal_verilator_z15.o) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/c10_resident_rtl.o: tests/c10_resident_rtl.c c10_session_core.h ysyx_rt.h
	$(CC) $(CFLAGS) -I. -DYSYX_WITH_VERILATOR -c "$<" -o "$@"

$(OUT)/c10_resident_rtl: $(addprefix $(OUT)/,c10_resident_rtl.o ysyx_rt.o z15_cmdq.o hal_verilator_z15.o) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/ysyx_rt.o: c10_session_core.h
$(OUT)/ysyx_yolo26n_api.o: c10_session.h c10_session_core.h

$(OUT)/c10_benchmark.o: c10_session.h c10_session_core.h hal_reference_c6.h
$(OUT)/hal_reference_c6.o: hal_reference_c6.h

$(OUT)/z15_c10_benchmark: $(addprefix $(OUT)/,c10_benchmark.o ysyx_rt.o z15_cmdq.o hal_linux_uio.o hal_reference_c6.o hal_verilator_z15.o) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -lm -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/c11b_rope_words.h: ../tools/c11b_emit_rope.py ../hetero_compiler/c11_qwen_rope.py
	cd .. && python3 -B -m tools.c11b_emit_rope > "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/c11b_qwen_z15.o: c11b_qwen_z15.c c11b_qwen_z15_cmdq.h z15_cmdq.h c11b_qwen_ops.h c12_qwen_ref.h $(OUT)/c11b_rope_words.h
	$(CC) $(CFLAGS) -I"$(OUT)" -c "$<" -o "$@"

$(OUT)/c11b_qwen_ref.o: c11_qwen_ref.c c11_qwen_ref.h c12_qwen_ref.h c12_qwen_cache.h c11b_reference.inc c11b_qwen_ops.h
	$(CC) $(CFLAGS) -DYSYX_C11B_NATIVE -mavx2 -c "$<" -o "$@"

$(OUT)/c11b_layer0.o: tests/c11b_layer0.c c11_qwen_ref.h c11b_qwen_ops.h
	$(CC) $(CFLAGS) -DYSYX_C11B_NATIVE -DYSYX_WITH_VERILATOR -I. -c "$<" -o "$@"

$(OUT)/c11b_layer0: $(addprefix $(OUT)/,c11b_layer0.o c11b_qwen_ref.o c11b_qwen_ops.o c11b_qwen_z15.o z15_cmdq.o hal_verilator_z15.o) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/c11b_projection.o: tests/c11b_projection.c c11_qwen_ref.h c11b_qwen_ops.h
	$(CC) $(CFLAGS) -DYSYX_C11B_NATIVE -I. -c "$<" -o "$@"

$(OUT)/c11b_full.o: tests/c11b_full.c c11_qwen_ref.h c11b_qwen_ops.h
	$(CC) $(CFLAGS) -DYSYX_C11B_NATIVE -I. -c "$<" -o "$@"

C11B_RTL := $(addprefix $(OUT)/,c11b_qwen_ref.o c11b_qwen_ops.o c11b_qwen_z15.o z15_cmdq.o hal_verilator_z15.o) $(RTL_OBJECTS)
$(OUT)/c11b_projection: $(OUT)/c11b_projection.o $(C11B_RTL)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/c11b_full: $(OUT)/c11b_full.o $(C11B_RTL)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"


$(OUT)/c12_qwen_rtl.o: tests/c12_qwen_rtl.c c11b_qwen_z15_cmdq.h z15_cmdq.h c12_qwen_ref.h c11b_qwen_ops.h
	$(CC) $(CFLAGS) -DYSYX_C11B_NATIVE -I. -c "$<" -o "$@"

$(OUT)/c12_qwen_rtl: $(OUT)/c12_qwen_rtl.o $(C11B_RTL)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/ysyx_rt.o $(OUT)/z15_cmdq.o $(OUT)/hal_reference_c6.o $(OUT)/hal_linux_uio.o: z15_cmdq.h
$(addprefix $(OUT)/,ysyx_run.o ysyx_yolo26n_cli.o c10_benchmark.o): z15_exec_option.h

$(OUT)/c16_prefill_rtl.o: tests/c16_prefill_rtl.c c12_qwen_ref.h c11b_qwen_z15_cmdq.h
	$(CC) $(CFLAGS) -DYSYX_C11B_NATIVE -I. -c "$<" -o "$@"

$(OUT)/c16_prefill_rtl: $(OUT)/c16_prefill_rtl.o $(C11B_RTL)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(addprefix $(OUT)/,c11b_qwen_ref.o ysyx_yolo26n_api.o): c17_simd.h
