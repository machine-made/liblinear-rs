# Reference build

Source: LIBLINEAR 2.50, tag `v250`.
Commit: `491c9f1188b97ba70847c70a68be363d186ddf9d`.

Compiler:
```text
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

Machine: Linux 6.18.54.1-microsoft-standard-WSL2 x86_64.
C library: glibc 2.39.
Python: Python 3.12.3.

Each command runs from the repo root. Output files use a temporary folder.
All source files, including the four BLAS files, use the same flags.
```sh
g++ -std=c++11 -O0 -Ivendor/liblinear reference/driver/ref_driver.cpp vendor/liblinear/linear.cpp vendor/liblinear/newton.cpp vendor/liblinear/blas/daxpy.c vendor/liblinear/blas/ddot.c vendor/liblinear/blas/dnrm2.c vendor/liblinear/blas/dscal.c -o <temporary>/ref_driver_O0
g++ -std=c++11 -O2 -Ivendor/liblinear reference/driver/ref_driver.cpp vendor/liblinear/linear.cpp vendor/liblinear/newton.cpp vendor/liblinear/blas/daxpy.c vendor/liblinear/blas/ddot.c vendor/liblinear/blas/dnrm2.c vendor/liblinear/blas/dscal.c -o <temporary>/ref_driver_O2
```

No other compiler flags are used.
The driver calls `srand(1)` before each `train` call.
The driver sets the C locale and prints each double with `%.17g`.
The runs use `RAYON_NUM_THREADS=2`, `CARGO_BUILD_JOBS=2` and `LC_ALL=C`.
Builds and case runs are serial.

Run `bash scripts/freeze_reference.sh` to build and write both result sets.
Run `python3 scripts/check_reference.py` to check the saved files and rerun all cases twice.
