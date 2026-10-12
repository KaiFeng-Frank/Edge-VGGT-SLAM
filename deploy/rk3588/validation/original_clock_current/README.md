# Current original VGGT at 1 GHz (v75)

With renewed user authorization, the current v74 original-model runtime completed two sequential bounded tests on RK3588S: a fixed 16-pair full-model comparison and an independent paced TUM fr1/xyz replay. [Results and evidence hashes](../../../../results/original_1ghz_current_v75.json).

| Measurement | v74, 300 MHz | v75, 1 GHz |
| --- | ---: | ---: |
| Full model + SL4, same 16 pairs, mean | 1.589965 s | 1.379500 s |
| Same comparison, excluding first pair | 1.562001 s | 1.346700 s |
| Replay fresh observations | 18 | 22 |
| Replay fresh pose rate, EOF drain included | 0.621433 Hz | 0.775988 Hz |
| Complete input-to-pose P95 | 2.030057 s | 1.570522 s |
| Replay startup | 39.456760 s | 47.102215 s |
| Replay peak process RSS | 3920.395 MiB | 3921.570 MiB |

The complete fixed-input service mean decreased 13.24% in these sequential runs. All 160 prediction/retained/target arrays, 32 NPZ files and the saved trajectory match bytewise. The original FP32 strict comparisons remain 0/80 passing. The replay selects different observations, so it does not inherit fixed-pair quality evidence. Neither one fresh pose per second nor the project's 20 Hz / P95≤100 ms target is met.

Both jobs used the original checkpoint and unchanged production inference sources: retained aggregator, camera split4, grouped DINO, depth, native half I/O, tail cache and staged PAGEOUT. No QKV candidate from the separate component diagnostic was installed. Model threads=1, CPUs4–7, decoder CPU 0, image-size argument 112. The board has 16 GB; process RSS is not proof of physical 8 GB operation. Physical camera and 30-minute endurance remain untested.

143 / 153 frequency samples were all 1 GHz. Maximum sampled temperatures were 63.769 / 66.538°C. Both workers exited 0, their process groups were stopped, and actual NPU frequency 300 MHz plus original governor/min/max were verified after each test. Samples do not prove continuous thermal behavior.

## Reproduction

The included launcher scripts contain the exact frozen v74 arguments with new output directories and pinned private artifact hashes. They require the original licensed checkpoint, original compiled artifacts and SDK; those files are not published. Change output paths to fresh directories when rerunning. Do not overwrite the measured evidence.

Use `../original_clock_test/npu_frequency_test.py --execute-frequency-test --frequency-hz 1000000000 --launcher ABSOLUTE_LAUNCHER_PATH --output NEW_CONTROL_DIRECTORY`, one launcher at a time. Inspect the first control record and restoration before the second. Its 180 s / 75°C bounds remain unchanged. The controller now also fails and restores if any running frequency sample differs from the request. Nine fake-sysfs/process tests pass, including a simulated 600 MHz drop and restoration ordering. Existing project authorization rules for clock changes still apply.

Source reviews, exact child arguments and saved-output audits accompany this record. Full arrays, map captures, checkpoints, weight-bearing graph exports and external binaries remain private. Repository code licensing does not grant additional rights to original model weights.
