// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t palette[16] = {
    0, UINT64_C(0x8000000000000000), 1, UINT64_C(0x8000000000000001),
    UINT64_C(0x0010000000000000), UINT64_C(0x8010000000000000),
    UINT64_C(0x3ff0000000000000), UINT64_C(0xbff0000000000000),
    UINT64_C(0x4000000000000000), UINT64_C(0xc000000000000000),
    UINT64_C(0x7fefffffffffffff), UINT64_C(0xffefffffffffffff),
    UINT64_C(0x3fb999999999999a), UINT64_C(0xbfb999999999999a),
    UINT64_C(0x4010000000000000), UINT64_C(0xc010000000000000)
};

static int compare_double(const void *a, const void *b) {
    if (*(const double *)a > *(const double *)b) return -1;
    if (*(const double *)a < *(const double *)b) return 1;
    return 0;
}

static uint32_t next(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

int main(void) {
    uint32_t state = 410;
    printf("{\"seed\":410,\"sorted\":[");
    for (int i = 0; i < 10000; ++i) {
        int n = 1 + (next(&state) >> 16) % 16;
        double values[16];
        for (int j = 0; j < n; ++j) {
            uint64_t bits = palette[(next(&state) >> 16) % 16];
            memcpy(&values[j], &bits, sizeof(bits));
        }
        qsort(values, n, sizeof(double), compare_double);
        printf("%s\"", i ? "," : "");
        for (int j = 0; j < n; ++j) {
            uint64_t bits;
            memcpy(&bits, &values[j], sizeof(bits));
            for (int k = 0; k < 16; ++k)
                if (palette[k] == bits) putchar("0123456789abcdef"[k]);
        }
        printf("\"");
    }
    printf("]}\n");
    return fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
}
