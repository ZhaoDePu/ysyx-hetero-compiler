CC ?= gcc
CXX ?= g++
CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror
CXXFLAGS := -std=c++17 -O2

OBJECTS := $(addprefix $(OUT)/,ysyx_rt.o hal_linux_uio.o ysyx_run.o hal_verilator_z15.o)
RTL_OBJECTS := $(addprefix $(RTL_BUILD)/,Vz15_bd_axi_shell__ALL.a verilated.o verilated_dpi.o verilated_threads.o)

.PHONY: all
all: $(OUT)/ysyx_run

$(OUT)/%.o: %.c ysyx_rt.h z15_regs.h
	$(CC) $(CFLAGS) -DYSYX_WITH_VERILATOR -c "$<" -o "$@"

$(OUT)/hal_verilator_z15.o: hal_verilator_z15.cpp ysyx_rt.h $(OUT)/observed_rig.inc
	$(CXX) $(CXXFLAGS) -I"$(OUT)" -I"$(RTL_BUILD)" -I"$(RTL_SRC)/app" \
	    -isystem "$(SDK)/include" -isystem "$(SDK)/include/vltstd" -c "$<" -o "$@"

$(OUT)/ysyx_run: $(OBJECTS) $(RTL_OBJECTS)
	$(CXX) -pthread $^ -o "$@.tmp"
	mv -f -- "$@.tmp" "$@"
