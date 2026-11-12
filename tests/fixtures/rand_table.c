// SPDX-License-Identifier: BSD-3-Clause
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("{\"seeds\":[");
    for (unsigned int seed = 1; seed <= 2; ++seed) {
        srand(seed);
        printf("%s{\"seed\":%u,\"values\":[", seed == 1 ? "" : ",", seed);
        for (int i = 0; i < 10000; ++i)
            printf("%s%d", i == 0 ? "" : ",", rand());
        printf("]}");
    }
    printf("]}\n");
    return fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
}
