# Performance against LIBLINEAR 2.50

These stage A measurements compare the faithful port with the unchanged C
library. They cover 20 dataset and solver pairs on 2026-11-13. A ratio below
1 means Rust uses less time or memory. Each ratio divides the Rust median
by the C median. The first run supplies the table; the rerun stays separate.

The CPU was an AMD Ryzen 9 9900X 12-Core Processor, x86_64. The kernel was
Linux 6.18.54.1-microsoft-standard-WSL2. Rust used rustc 1.99.1
(2026-10-13) in release mode. C used g++ 13.3.0 from Ubuntu.

| Dataset | Solver | Train Rust/C | Predict Rust/C | Peak RSS Rust/C |
| --- | ---: | ---: | ---: | ---: |
| a9a | 0 | 1.115 | 0.824 | 1.073 |
| a9a | 1 | 0.854 | 0.852 | 1.071 |
| a9a | 2 | 1.164 | 0.832 | 1.076 |
| a9a | 3 | 0.908 | 0.872 | 1.072 |
| a9a | 5 | 1.022 | 0.850 | 1.008 |
| a9a | 6 | 1.255 | 0.842 | 1.007 |
| a9a | 7 | 1.013 | 0.832 | 1.058 |
| rcv1_train.binary | 0 | 0.969 | 0.956 | 1.616 |
| rcv1_train.binary | 1 | 1.147 | 1.042 | 1.730 |
| rcv1_train.binary | 2 | 1.023 | 0.938 | 1.629 |
| rcv1_train.binary | 3 | 1.033 | 1.076 | 1.730 |
| rcv1_train.binary | 5 | 1.179 | 0.833 | 1.082 |
| rcv1_train.binary | 6 | 1.489 | 0.802 | 1.079 |
| rcv1_train.binary | 7 | 1.183 | 1.021 | 1.713 |
| satimage.scale | 4 | 1.269 | 0.894 | 0.979 |
| satimage.scale | 2 | 1.126 | 0.892 | 0.985 |
| cpusmall_scale | 11 | 1.549 | 0.885 | 0.939 |
| cpusmall_scale | 12 | 0.866 | 0.901 | 0.940 |
| cpusmall_scale | 13 | 1.008 | 0.895 | 0.935 |
| a9a | 21 | 1.408 | 0.784 | 1.072 |

The training ratio range is **0.854 to 1.549**. The prediction
ratio range is **0.784 to 1.076**. The peak resident memory
ratio range is **0.935 to 1.730**.

1. Use all 20 pairs from the passing real-data parity report, with its parameters. Check the pinned download and text hashes. Keep the data outside the repository.
2. Build the unchanged C sources with `g++ -std=c++11 -O2`, as in `reference/BUILD.md`. Reuse the reference case reader. Build the Rust example with `cargo build --locked --release`.
3. Warm up each program once per pair. Run ten C/Rust pairs in that order, one thread per process, at nice 19 under `jobscope`.
4. Use a monotonic clock inside each process around train and predict calls only. Leave file reads, startup and output outside the clocks.
5. Record the median, minimum and maximum of each time. Use GNU time `%M` for each process peak RSS in KiB. Report its median and range, plus the Rust/C ratios.

Rust's public `train` also validates the inputs and scores the training rows.
The C `train` call does not score them. Rust's `predict` allocates its result;
the C driver fills a result vector allocated before the clock starts. Both
predict on the training rows. Peak RSS covers the whole process, including
its case reader and retained model. It does not isolate solver workspace.
The driver checksum checks the prediction count and sum. It does not replace
the bit comparisons in [PARITY.md](PARITY.md).

The same-binary self-test gave **1.00224**, inside 0.97 to
1.03. In the final rerun, all 40 time ratios stayed within 10 percent of the
first run. The largest training change was **6.15%**; the
largest prediction change was **7.85%**.

An earlier check, before a change to commit validation, saw a **12.67%**
prediction-ratio change for `rcv1_train.binary`, solver 0 (0.976777 to 1.10057).
The JSON retains that observation and its samples. WSL2 scheduling and shared
host load can affect short timings. This check did not measure those sources
separately. No results from different runs are averaged.

Five pairs exceed the memory follow-up limit of 1.5 times C. No pair exceeds
2 times C for training or prediction. The next work is to measure these memory
costs before changing the port:

- Measure and reduce peak memory for `rcv1_train.binary`, solver 0: 1.616 times C
- Measure and reduce peak memory for `rcv1_train.binary`, solver 1: 1.730 times C
- Measure and reduce peak memory for `rcv1_train.binary`, solver 2: 1.629 times C
- Measure and reduce peak memory for `rcv1_train.binary`, solver 3: 1.730 times C
- Measure and reduce peak memory for `rcv1_train.binary`, solver 7: 1.713 times C

These numbers do not show performance on other machines, other datasets or
other parameters. They do not test multiple threads. Different case readers
can affect peak RSS. The timing differences do not isolate algorithm cost.
The JSON contains both runs, all samples, compiler versions, the source commit,
and hashes of the data, cases, harness and binaries:
[reference/perf/speed_vs_c.json](../reference/perf/speed_vs_c.json).

Run the following from any directory. Replace both absolute paths. A new
baseline run refuses to overwrite a saved report: first move that report to
the external cache to preserve it. `--rerun` keeps the baseline and requires
the same machine, compilers, inputs, harness and binaries. A later commit is
allowed when those hashes stay the same. The script records each run's commit.

```sh
export RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2
export LIBLINEAR_REAL_DATA_CACHE=/absolute/cache
jobscope python3 -I /absolute/repo/scripts/speed_vs_c.py --self-test
jobscope python3 -I /absolute/repo/scripts/speed_vs_c.py
jobscope python3 -I /absolute/repo/scripts/speed_vs_c.py --rerun
```
