The saved original VGGT `frame0 -> global0` pair shows a distinction between a representation limit and measured computation error. This analysis reads the v49 saved original Torch FP32 references and SDK outputs; it runs no model, ORT, SDK, calibration or board job. Inputs remain the original TUM pairs, requested image size 336 and actual shape 252 x 336. The original pair input is `[2,437,1024]`, and its output is `[1,874,1024]` (894,976 elements).

The unchanged strict component gate is `abs(candidate - reference) <= 1e-5 + 1e-4 * abs(reference)`, using the original saved FP32 reference magnitude. The two heldout outputs give these per-tensor lower bounds:

| Representation of the output | evaluation_00 minimum relative L2 / unavoidable violating elements | evaluation_01 minimum relative L2 / unavoidable violating elements |
| --- | ---: | ---: |
| Any finite FP16 value per element | 0.0002072208568404 / 272,497 | 0.0002071516280069 / 272,573 |
| Observed signed INT8 affine grid, FP32 scale/product | 0.04218913205827 / 893,274 | 0.04268891930040 / 893,391 |

For each finite FP32 reference element `r`, the analyzer chooses the closest member of the ordered finite FP16 grid. It evaluates the rounded center and both adjacent values, then verifies that the selected value is no farther from `r` than either of its own neighbors. Values farther along the ordered grid cannot be closer. Endpoints, signed zero, subnormals and ties are covered. An independent enumeration of all 63,487 distinct finite FP16 numeric values agrees on the minimum distance for every saved reference element. Therefore, whenever this minimum distance exceeds the fixed reference tolerance, every finite FP16 candidate violates that element's gate. Independent element choices also minimize squared error, so the reported relative L2 values are lower bounds for that grid and those references.

The 16 generated mathematical checks cover every finite half value, all 63,486 adjacent midpoints and their FP32 neighbors, subnormal and normal boundaries, finite endpoints and FP32 extremes. A finite FP32 reference and FP16-representable candidate also demonstrate why using candidate magnitude could relax the gate incorrectly. The checks compare each defined INT8 grid against brute-force search over all 256 levels and reject nonfinite references. These checks use generated numbers only.

The actual SDK nonquant outputs for both heldout round-zero calls are exactly FP16-representable. Their measured relative L2 errors are 0.0044525369243 and 0.0049775506393; both remain strictly rejected. Both actual PTQ outputs contain only values on `float32((q - 5) * float32(0.022254416719079018))`, with signed `q` in `[-128,127]`. The analyzer also tests grids with an FP16-rounded scale, an FP16-rounded product, and both; those alternatives do not account for every observed output value. Actual PTQ relative L2 errors are 0.103121542575 and 0.105464552339. An observed error above a grid floor does not identify the cause inside the computation.

The original input has a small number of large channel values: in each of the ten saved pairs, all 864 patch tokens have exactly one element with magnitude above 4, in original channel 555. Whole-input median magnitude is about 0.10 and the 99th percentile about 0.62. The five original camera/register tokens have maximum magnitude 0.180581. Under the measured input INT8 descriptor (scale 0.04107800871133804, zero point 57), the two heldout inputs have conditional nearest-grid relative L2 floors of 0.0395123245614 and 0.0395782970177, compared with nearest FP16 floors of 0.0001854584530 and 0.0001860350276. Only 7 and 1 input elements lie outside that grid's endpoints. These are descriptive saved-input calculations; they do not assert the SDK's rounding rule or downstream error propagation and do not fit new calibration scales.

An experiment retaining FP16 boundaries and quantizing only the four original MLP MatMul operators is a useful controlled diagnostic. Their four weights account for 67,108,864 of 100,788,224 original pair parameter bytes, and 29,326,573,568 of 48,683,114,496 counted pair MatMul FLOPs (60.24%). These static fractions are neither runtime fractions nor proof that the SDK supports the desired partition. MLP intermediate distributions were not captured. The experiment must inspect the actual compiled boundaries and operators, retain original checkpoint/parameter provenance and independent calibration/heldout data, and measure actual board error and complete I/O timing. It has not been run here.

Any FP16 output boundary still has the unavoidable strict failures shown above for these references. The gate remains unchanged and the v49 candidates remain rejected. Full original-model SLAM acceptance using ATE/RPE degradation of at most 10-20% is a separate question requiring trajectory evaluation; a component relative L2 percentage is not an ATE/RPE result. These saved-array findings prove no hardware impossibility, no trajectory quality, and no achievement of 20 Hz fresh poses or P95 at most 100 ms.

Reproduce the generated checks with NumPy and Python:

```sh
python3 check_resolution_math.py --output /path/to/new-generated-checks.json
```

Private saved artifacts are required to reproduce the dataset analysis; the CLI takes explicit paths and refuses to overwrite an existing result:

```sh
python3 analyze_saved_resolution.py --capture-dir /path/to/capture-r1 --normal-dir /path/to/normal-r1 --pair-export /path/to/export.json --output /path/to/new-resolution.json
```

`public-resolution-r1.json` contains only counts, distributions, descriptors, formulas and source/evidence hashes. The original arrays, checkpoints, weight-bearing exports and SDK binaries remain private. Repository source licensing does not grant commercial rights to the original checkpoint.
