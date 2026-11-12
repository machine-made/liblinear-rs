# Reference files

`corpus/` holds the inputs. `golden/O0/` and `golden/O2/` hold one JSON file for
each input. `MANIFEST.sha256` checks all input and result files, the driver,
`BUILD.md` and `spread.json`. File paths in the manifest start at this folder.
The checker also checks the upstream hashes in `vendor/liblinear/PROVENANCE.md`.

## Input format

Each case is a text file. The header has these fields:

| Field | Value |
| --- | --- |
| solvers | One or more solver numbers from `linear.h` |
| C | Cost parameter |
| eps | Stop limit |
| bias | Negative to omit the bias feature; otherwise its value |
| p | Regression loss width |
| nu | One-class parameter |
| regularize_bias | 1 to apply the penalty to the bias, 0 to omit it |
| weights | Count, then pairs of class label and class weight |
| features | Feature count, before the bias feature |
| rows | Row count |
| data | End of the header |

Rows after `data` use LIBSVM text: a label, then ordered `index:value` pairs.
The driver adds the bias feature and the end marker. An explicit feature count
keeps the size of a feature that is zero in all rows.

The generator uses `random.Random(0xC0FFEE)`. Each solver gets `heart_scale`, a
dense case with 200 rows and 20 features, and a sparse case with 500 rows and
300 features. The sparse case has six nonzero entries per row, or 2 percent.
The `heart_scale` rows are copied as given, also for regression. Solver 21
ignores row labels; the seeded one-class cases use label 1.

Small cases cover C of `1e-6` and `1e6`, eps of `1e-1` and `1e-8`, bias of -1
and 1, a class weight of 10, a zero feature, a copied row and a single row.
Class weights apply to solvers 0 through 7. Solver 21 rejects a bias feature,
so it has no bias-on case. The free-bias cases use only the five solvers that
accept this option: 0, 2, 5, 6 and 11. Solver 4 also has a three-class case.

The analytic case runs solvers 1 and 2 on two rows: `1 1:1` and `-1 1:-1`.
It uses C of 1000, eps of `1e-8` and no bias. Its objective is
`0.5*w*w + C*sum(max(0, 1-y*w*x)^2)`. The minimum has
`w = 4*C/(1+4*C) = 4000/4001`. The checker uses an absolute limit of `1e-4`.
The freeze stops if this check fails.

## Result format

Each JSON file has a `runs` array, with one entry per solver in the input.
Each entry holds the solver, C, eps, bias, feature and class counts, class
labels, model weights, decision values and predictions for all input rows.
The `rho` value is set only for solver 21; it is null for all other solvers.
Class labels are null for regression and one-class models, as in the original.

Weights use the upstream layout: feature first, then class. A bias weight,
if present, is the last feature. A two-class model has one weight per feature,
except solver 4, which has one per class. Other models have one per class.
Each row of `decision_values` has the same number of columns. All model
doubles use 17 significant digits. The driver calls `srand(1)` before each fit.

`spread.json` groups numeric leaves by output field. Array indices share one
field. Each field records the count and the largest absolute and relative
difference. The relative difference is `abs(O2-O0)/max(abs(O0),abs(O2))`, or
zero when both values are zero. Both total and per-case measurements are saved.
The checker makes this report again and requires an exact match.

The self-test changes one byte of a result in a temporary copy and checks for
exit 1. It then removes solver 21, rebuilds the manifest and spread in that
copy, and checks for exit 1 from the solver check. It also checks the spread
function with known unequal values. The saved reference files are kept intact.
