# Original VGGT FP16 SDK I/O

The optional `vggt_slam.rknn_half_io_pairs` backend uses the same 24 original
FP16 frame/global graphs and unchanged original Aggregator loop. It changes
only API storage: NEON converts FP32 inputs to FP16, `pass_through=0` remains,
and `want_float=0` returns both outputs. Each output is copied before SDK
release, then widened to an independent FP32 array. Original guards, source and
parameter checks, intermediate concatenations and CPU operations remain.

The helper is compiled and tested on RK3588S. No binary is distributed. The
current backend accepts only source SHA
`77707b735a6291c417a997ed64acbe4e1cd60cc5f9dd150d83fe1836868bd5e3`
and verified library SHA
`d08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9`.
`build_neon_board.py` records the isolated compiler, command and output hash;
its paths describe the measured board installation. A different build requires
new conversion and complete-model validation before changing the pin.

Reproduce the recorded component experiment by copying this directory's four
source files into `experiments/vggt-v56-source-r1` on the board, running
`build_neon_board.py`, then `check_neon_board.py --library LIB --sha256 SHA
--output NEW_CHECK.json`. The generated conversion check covers every finite
FP16 encoding, finite random FP32 values, short tails and unaligned vector
offsets. Then run `probe_half_io_board.py --neon-library LIB
--neon-library-sha256 SHA --output-dir NEW_DIRECTORY`. It retains 12 ABBA
calls, both original outputs, original FP32 errors and all timing stages.

For complete replay, add these options to the existing paired FP16 command:

```sh
--rknn-paired-aggregator --model-threads 1 \
--half-io-library /mnt/vggt-data/vggt-slam2/experiments/vggt-v56-neon-build-r1/fp16_neon.so \
--half-io-library-sha256 d08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9
```

The complete comparison entry also accepts the two library options with
`--paired-aggregator`. Omitting them preserves the preceding SDK I/O path.
The fixed 16-pair comparison improved from 2.1773 to 2.0766 seconds mean
complete service, preserving all 80 canonical arrays and the saved trajectory
bitwise against the preceding FP16 execution. Its strict original FP32 checks
still fail. Neither these timings nor the component equality establish general
model quality, real-time operation or 30-minute stability.

See [measured results](../../../../results/original_calibration_half_io_v56.json).

The same two library options now also select explicit FP16 I/O for all 24
original DINO blocks when a DINO manifest is supplied. The original token
preparation, cache, final norm and FP32 boundaries are retained. The fixed
comparison measured 2.0275 seconds before the guard optimization and 1.9751
seconds afterwards; all 80 arrays and saved poses match the previous FP16
path bitwise. These are sequential runs, not a controlled thermal comparison.
Both complete cache guards remain; the presence lookup now handles exact
`dict` and `OrderedDict` registries, while custom lookup behavior uses the
original path. No new CLI flag is needed.
