# Original cache-guard cost diagnostic

`profile_cache_compare.py` wraps the unchanged complete-model comparator on
RK3588S. Its first argument is a new JSON profile-output path; remaining
arguments are the comparator's normal arguments. Use the existing deployment
environment and pinned original manifests. Actual model execution is board-only.

It profiles the first five cache fingerprints, including installation, and
passes all later calls through unchanged. The original method is restored in
`finally`. cProfile overhead is included: these timings diagnose call costs
and must not be used as ordinary inference timing or a speedup claim.

The separate unprofiled 16-pair run measured 1.96825 seconds per complete update,
including 0.13809 seconds in its two cache guards. Both runs preserved all 80
saved arrays and the trajectory relative to the preceding baseline. No guard
was removed, no production implementation changed, and no real-time result was
achieved. [Recorded evidence](../../../../results/original_guard_diagnostic_v62.json).
