# Bounded original VGGT frequency comparison

**Executed with explicit user authorization in v63.** Both bounded workers
completed; actual NPU frequency, governor and limits were restored. The matched
complete-model mean decreased from 1.968 to 1.586 s (19.43%), with all 80 arrays
and poses byte-identical to the prior FP16 path. Timestamp replay measured
0.676 fresh poses/s and complete P95 2.028 s; realtime remains unmet.
[Results, sampled clocks, temperatures and evidence hashes](../../../../results/original_1ghz_v63.json).

Explicit authorization for a system-wide NPU clock change is required by
[AGENTS.md](../../../../AGENTS.md). The two authorized tests are complete;
publishing these scripts does not authorize further clock changes.

The two historical launchers use the original-model graphs measured in v63: first the fixed 16-pair
complete-model comparison, then the real timestamp replay. Run them separately
and sequentially under `npu_frequency_test.py`. Each test requests a supported
1 GHz NPU frequency, runs at most 180 seconds and stops if any known thermal
zone reaches 75 C. The wrapper stops the worker process group and restores the
original actual NPU frequency in `finally`, then verifies the governor and
limits. It changes no CPU clock, governor, driver, model or network setting.

Before the first test, actual frequency was 300 MHz and the userspace request
was `undefined`. Both tests restored actual 300 MHz, governor and limits;
the request is now explicit 300 MHz. This is not a byte-for-byte restoration
of the initial internal request.
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
The v63 executions used the controller with eight passing generated checks.
The current controller additionally rejects a frequency change during the run;
all nine generated checks pass, including the new drop-and-restore case.
The current-runtime test and launchers are recorded in
[original_clock_current](../original_clock_current/README.md). The v63 actual runs recorded 97 and 107 samples,
all at 1 GHz, with maximum sampled temperatures 61.923 and 63.769 C. Each
finished within 55 seconds, with no restoration errors. The original launchers
retain their prepared v59 worker-directory names; new v63 control directories
identify this execution. Existing worker directories now contain results and
must not be reused or overwritten.
