#include "c12_qwen_cache.h"

#include <stdio.h>
#include <string.h>

static int check(unsigned bits)
{
    uint8_t record[68] = {0};
    int32_t values[C12_KV_GROUP] = {0}, restored[C12_KV_GROUP];
    int limit = bits == 4 ? 7 : 127;
    values[0] = -limit * 256;
    values[1] = 3 * 256;
    values[2] = -128;
    values[3] = 127;
    values[63] = limit * 256;
    c12_cache_pack(record, values, bits);
    const unsigned expected = bits == 4 ? 0x39 : 0x81;
    const size_t payload = c12_cache_record_size(bits, 0) - 4;
    if (record[0] != expected || memcmp(record + payload, "\0\1\0\0", 4))
        return 1;
    c12_cache_unpack(record, restored, bits);
    if (restored[0] != values[0] || restored[1] != values[1] ||
        restored[2] != -256 || restored[3] != 0 || restored[63] != values[63])
        return 1;
    memset(values, 0, sizeof(values));
    c12_cache_pack(record, values, bits);
    c12_cache_unpack(record, restored, bits);
    return memcmp(values, restored, sizeof(values)) || record[payload] != 1;
}

int main(void)
{
    if (check(4) || check(8))
        return 1;
    int32_t in[C12_KV_GROUP] = {0}, out[C12_KV_GROUP];
    uint8_t record[56];
    in[1] = -40000;
    in[11] = INT32_MAX;
    in[24] = INT32_MIN;
    in[53] = 1200;
    in[55] = 7;
    c12_cache_pack_outliers(record, in);
    c12_cache_unpack_outliers(record, out);
    if (memcmp(in, out, sizeof(in)) || c12_cache_record_size(4, 1) != sizeof(record))
        return 1;
    puts("C12 KV4, protected KV4 and KV8 scale/packing PASS");
    return 0;
}
