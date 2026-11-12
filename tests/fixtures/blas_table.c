// SPDX-License-Identifier: BSD-3-Clause
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "blas.h"

#ifdef __cplusplus
extern "C" {
#endif
extern double ddot_(int *, double *, int *, double *, int *);
extern double dnrm2_(int *, double *, int *);
#ifdef __cplusplus
}
#endif

static void bits(double value) {
    uint64_t word;
    memcpy(&word, &value, sizeof word);
    printf("\"%016" PRIx64 "\"", word);
}

static void array(const double *values, int n) {
    putchar('[');
    for (int i = 0; i < n; ++i) {
        if (i) putchar(',');
        bits(values[i]);
    }
    putchar(']');
}

static void entry(const char *name, int n, const double *input, const double *other, double a) {
    double x[9], y[9], scaled[9];
    int inc = 1;
    memcpy(x, input, sizeof(double) * n);
    memcpy(y, other, sizeof(double) * n);
    memcpy(scaled, input, sizeof(double) * n);
    printf("{\"name\":\"%s\",\"x\":", name);
    array(x, n);
    printf(",\"y\":"); array(y, n);
    printf(",\"a\":"); bits(a);
    printf(",\"ddot\":"); bits(ddot_(&n, x, &inc, y, &inc));
    printf(",\"dnrm2\":"); bits(dnrm2_(&n, x, &inc));
    daxpy_(&n, &a, x, &inc, y, &inc);
    dscal_(&n, &a, scaled, &inc);
    printf(",\"daxpy\":"); array(y, n);
    printf(",\"dscal\":"); array(scaled, n);
    putchar('}');
}

int main(void) {
    const double x[9] = {1e16, 1, -1e16, 3.25, -0.125, 7.1, -8.9, 0, -0.0};
    const double y[9] = {1, 1, 1, -0.5, 3, 0.1, -2, -0.0, 0};
    const int lengths[] = {0, 1, 4, 5, 6, 9};
    printf("[\n");
    for (int i = 0; i < 6; ++i) {
        char name[20];
        snprintf(name, sizeof name, "length_%d", lengths[i]);
        if (i) printf(",\n");
        entry(name, lengths[i], x, y, -0.75);
    }
    const double large[4] = {1e300, -1e300, 1e-300, -0.0};
    const double tiny[4] = {ldexp(1.0, -1074), -ldexp(1.0, -1074), ldexp(1.0, -1022), -0.0};
    const double zero[4] = {0, -0.0, 0, -0.0};
    const double exceptional[4] = {NAN, INFINITY, -INFINITY, -0.0};
    printf(",\n"); entry("large", 4, large, y, 0.5);
    printf(",\n"); entry("tiny", 4, tiny, y, 0.5);
    printf(",\n"); entry("signed_zero", 4, zero, zero, -1);
    printf(",\n"); entry("zero_scale", 4, exceptional, y, 0);
    printf(",\n"); entry("nonfinite", 4, exceptional, y, 1);
    printf("\n]\n");
    return 0;
}
