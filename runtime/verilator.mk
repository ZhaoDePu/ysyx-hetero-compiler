CC ?= gcc
CXX ?= g++
CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror
CXXFLAGS := -std=c++17 -O2

OBJECTS := $(addprefix $(OUT)/,ysyx_rt.o hal_linux_uio.o ysyx_run.o hal_verilator_z15.o)
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
$(OUT)/yolo26n_post.o: CFLAGS += -ffp-contract=off

$(OUT)/ysyx_yolo26n_cli: $(addprefix $(OUT)/,ysyx_yolo26n_cli.o ysyx_yolo26n_api.o yolo26n_post.o ysyx_rt.o hal_linux_uio.o hal_verilator_z15.o) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/c10_resident_rtl.o: tests/c10_resident_rtl.c c10_session_core.h ysyx_rt.h
	$(CC) $(CFLAGS) -I. -DYSYX_WITH_VERILATOR -c "$<" -o "$@"

$(OUT)/c10_resident_rtl: $(addprefix $(OUT)/,c10_resident_rtl.o ysyx_rt.o hal_verilator_z15.o) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"

$(OUT)/ysyx_rt.o: c10_session_core.h
$(OUT)/ysyx_yolo26n_api.o: c10_session.h c10_session_core.h

$(OUT)/c10_benchmark.o: c10_session.h c10_session_core.h hal_reference_c6.h
$(OUT)/hal_reference_c6.o: hal_reference_c6.h

$(OUT)/z15_c10_benchmark: $(addprefix $(OUT)/,c10_benchmark.o ysyx_rt.o hal_linux_uio.o hal_reference_c6.o hal_verilator_z15.o) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -lm -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"
