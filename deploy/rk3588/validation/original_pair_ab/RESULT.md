# Measured original pair comparison

The source freeze and README describe preparation. Root subsequently executed this exact runner on RK3588S: 12 synchronous SDK calls, four AB/BA rounds, three contexts in one process. All four separate/fused final outputs matched bitwise. Both routes still failed the original FP32 reference gate.

The three subsequent SDK IO sums averaged **396.442 ms separate / 334.831 ms fused**, an observed **15.54%** decrease. The instrumented route means were **426.910 / 334.951 ms**; the separate route includes extra bridge finite/hash checks, so this larger difference must not be presented as the inherent fusion gain.

Peak process RSS was 324.008 MiB including three contexts and diagnostic arrays. Temperature and frequency were sampled, not controlled. This short fixed-input test does not prove whole-model quality, fresh pose rate, latency P95, 30-minute operation or 8 GB deployment.

[Full curated evidence](../../../../results/original_pair_v48.json), [engineering notes](../../../../docs/ENGINEERING_NOTES.md).
