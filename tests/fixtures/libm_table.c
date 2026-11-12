// SPDX-License-Identifier: BSD-3-Clause
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static uint32_t state = 1;

static double sample(void) {
    state = state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)state / 4294967296.0;
}

int main(void) {
    const double exp_edges[] = {1e-300, 700, -700, 0, -0.0, 1, -1, 1e-16};
    const double log_edges[] = {1e-300, 1e300, 1, 0.5, 2, 1e-16, 1.0000000000000002, 0.9999999999999999};
    printf("{\"seed\":1,\"values\":[\n");
    for (int i = 0; i < 2000; ++i) {
        double x;
        double y;
        if (i < 8) {
            x = exp_edges[i];
            y = log_edges[i];
        } else {
            const double u = sample();
            switch (i % 4) {
                case 0: x = 700 * u; break;
                case 1: x = -700 * u; break;
                case 2: x = pow(10.0, -300 * u); break;
                default: x = 2 * u - 1; break;
            }
            y = pow(10.0, -300 + 600 * sample());
        }
        printf("%s{\"exp_input\":%.17g,\"exp\":%.17g,\"log_input\":%.17g,\"log\":%.17g}",
               i == 0 ? "" : ",\n", x, exp(x), y, log(y));
    }
    printf("\n]}\n");
    return fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
}
