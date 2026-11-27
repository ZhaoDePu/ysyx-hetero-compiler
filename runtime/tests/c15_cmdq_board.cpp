// Include the pinned Phase 18 cmdq_board.cpp fixture with only its main removed.
#include "p18_reference_helpers.inc"

int main()
{
    try {
        Rig rig;
        check_abi(rig);
        const auto transactions = rig.transactions;
        const auto stored = rig.ddr_write_bytes;
        require(rig.write(control, 3) == 2, "simultaneous START/CLEAR accepted");
        finish(rig, 0, 10);
        require(rig.transactions == transactions && rig.ddr_write_bytes == stored &&
                !rig.read(Z15_SCRATCH), "simultaneous START/CLEAR forwarded a transaction");
        require(!rig.write(control, 2) && !rig.read(Z15_ERROR), "CLEAR did not recover");
        puts("C15_BOARD_ERROR_10 PASS no AXI transaction or register write");

        start(rig, 0x200000, {command(write_reg, Z15_CTRL, Z15_START_NPU | Z15_START_DMA),
                             command(end_batch)});
        finish(rig, 0, 9);
        require(rig.transactions == transactions + 1 && rig.ddr_write_bytes == stored &&
                !rig.read(Z15_SCRATCH) && rig.read(Z15_ERROR) == 128,
                "rejected dispatch mutated state or issued engine DMA");
        require(!rig.write(control, 2) && !rig.read(Z15_ERROR), "CLEAR did not recover");
        puts("C15_BOARD_ERROR_9 PASS only command fetch and no engine DMA");

        start(rig, 0x200000, {command(write_reg, Z15_SCRATCH, 0x15), command(wait_idle),
                             command(end_batch)});
        finish(rig, 3);
        require(rig.read(Z15_SCRATCH) == 0x15 && !rig.read(Z15_ERROR), "recovery batch failed");
        puts("C15_BOARD_POST_CLEAR PASS");
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "C15 board negative test failed: %s\n", error.what());
        return 1;
    }
}
