# Prepared original VGGT frequency comparison

**Not executed. Explicit authorization to change the system-wide NPU clock is
required by [AGENTS.md](../../../../AGENTS.md).** These files make the proposed
test reviewable; publishing or deploying them does not authorize execution.

The two launchers use the current original-model graphs: first the fixed 16-pair
complete-model comparison, then the real timestamp replay. Run them separately
and sequentially under `npu_frequency_test.py`. Each test requests a supported
1 GHz NPU frequency, runs at most 180 seconds and stops if any known thermal
zone reaches 75 C. The wrapper stops the worker process group and restores the
original actual NPU frequency in `finally`, then verifies the governor and
limits. It changes no CPU clock, governor, driver, model or network setting.

The currently observed actual frequency is 300 MHz, with the userspace request
shown as `undefined`. Restoring an explicit 300 MHz request restores the actual
frequency, but is not a byte-for-byte restoration of that internal request.
Read the source and saved control record rather than assuming every cleanup
succeeded. A worker's success also does not establish numerical or real-time
acceptance: inspect its own result, output arrays, timing and cleanup records.

After authorization, the bounded controller invocation is:

```sh
python npu_frequency_test.py --execute-frequency-test \
  --frequency-hz 1000000000 --temperature-limit-c 75 --timeout-seconds 180 \
  --launcher /absolute/path/run_1ghz_complete.sh --output /new/control-directory
```

Use a separate new control directory for `run_1ghz_replay.sh`; inspect and
confirm frequency restoration after the first worker before starting it.
Existing result directories must not be overwritten.

`test_npu_frequency_test.py` uses fake sysfs, signals and processes only.
The controller is unchanged from its eight passing v59 generated checks;
the saved log is included. It has not been tested by changing the actual clock.
