# Original checkpoint residency experiment

The original loader already uses verified `torch.load(mmap=True)` and strict
state assignment. This diagnostic applies `MADV_PAGEOUT` once after all original
backend parameter checks and cache installation. It advises reclamation of this
process's clean checkpoint file mappings without changing Tensor objects,
storage addresses, parameter values, model computation or guards.

The helper verifies the existing loader source, checkpoint stat across loading,
and each self mapping's device/inode, file offset, bounds and clean state. It
rejects dirty, anonymous, swapped, locked, huge or device mappings. The actual
advice is only value 21; errors are recorded and stop the experiment. There is
no fallback advice or global cache operation. Metadata checks do not read the
weight values again. The original loader still verifies the complete pinned
checkpoint SHA before deserialization.

Linux describes PAGEOUT as a reclaim request; a successful return does not
guarantee that every page was reclaimed. Consequently the experiment measures
residency and subsequent faults instead of calculating savings from the syscall
return. [Linux syscall documentation](https://man7.org/linux/man-pages/man2/madvise.2.html).

Use the frozen `compare_complete_pageout.py` with the same arguments as the
existing complete-model comparator plus `--checkpoint-pageout`. Keep
`checkpoint_pageout.py` beside it. Run all actual inference on the RK3588S in
the prepared runtime environment. The tested command is reproduced in
`pageout-command.json`; the monitor in `../original_retained_fc2/` holds a
300 MHz observation limit without changing the clock.

The helper receipt records before/immediate-after memory, faults, mapping
identity and Tensor metadata. Each complete inference has a further memory
snapshot taken outside the service timer. The final snapshot is **after backend
cleanup** and is not active inference residency. `ru_maxrss` preserves the
initialization high-water mark; reclaiming later pages does not reduce that
earlier peak. These measurements on a 16 GB board do not establish operation
on physical 8 GB hardware or the total system's physical-memory use.

```sh
# Saved NumPy outputs only; no model loading on the host.
python audit_pageout_complete_saved.py --baseline BASELINE_RUN \
  --candidate PAGEOUT_RUN --output NEW_AUDIT.json

# Generated/fake syscall tests only; no actual advice or model execution.
python test_checkpoint_pageout.py
```

The saved comparison checks matching inputs, all 160 prediction/intermediate
arrays, 32 NPZ files, the saved trajectory and all 12 closed SDK contexts.
It does not replace a timestamp replay, trajectory-quality evaluation or
long-duration stability test. This experiment does not change the production
runtime or its defaults.

[Measured results](../../../../results/original_checkpoint_pageout_v73.json).
