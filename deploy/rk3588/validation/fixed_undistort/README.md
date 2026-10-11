# 固定去畸变 map 的 CPU 预处理诊断

这组研究只比较相同原 RGB 观测的 CPU 解码/去畸变/裁边/缩放/FP32
归一化，不做模型前向、SDK、BA、pose 或 GT 评价，未接默认在线链路。
缓存固定 K/D/640×480 图像的原 OpenCV 条带 map；每条带使用原 `cy-y`、
固定点 map、线性插值和零边界。裁边 `[8:-8,16:-16]`、exact-half
`INTER_AREA` 及原 BGR 归一化 `2*(image/255)-.5` 保持不变。

## r2 板端对照

`run-v40-undistort-cache-board.py` 使用冻结 v39 同一 192 帧 manifest/policy。
每帧全 640×480 输出、232×304 半尺寸和 FP32 NCHW 输入均逐字节相同；
24 个预声明等间隔 source index 在同一次解码像素上进行 original/cached/
cached/original（ABBA），各方法 48 次。两条路径在同一进程使用 CPU 4–7、
OpenCV 4.10.0 的实际默认 4 threads；不改变线程数做“缓存加速”。

完整 warm 预处理均值（不含解码）为 **14.679 → 10.943 ms**，其中
undistort/remap 为 **13.866 → 10.167 ms**。地图首次构造 **6.246 ms**
另列，存储 **1,843,200 B**。两条时钟包括输出分配、remap、crop/resize
和归一化；imread、源 hash、像素比较、保存均在外。结果仅证明这次固定
几何的 CPU 对照，不代表神经计算、整条链路、20 Hz、轨迹或地图质量。

`r1` 的未绑定 CPU 的默认 8 threads 假设曾在图像解码/map 构造前被 guard
拒绝，原失败源码与收据保留。`r2` 按原回放启动条件，在绑定 4–7 后读实际
OpenCV 默认 4 threads，并验证退出时线程/affinity 恢复、源 SHA 不变。
cache 的 `map_bytes` 是构造时存储量；关闭后条带引用清除，helper 元数据
的 `actual_board_pixels_validated=False` 是 helper 自身不认证的默认值，
实际 192 帧 gate 由独立 benchmark 顶层记录，不修改冻结 helper 的标签。

板端入口：

```bash
python -B run-v40-undistort-cache-board.py \
  --dataset-dir "$TUM_RGB_SEQUENCE" --input-manifest "$FROZEN_192_INPUT_MANIFEST" \
  --policy "$FROZEN_M16_POLICY" --cache-helper fixed_undistort_cache.py \
  --normalization-helper "$FROZEN_V30_FRAME_ENCODER" \
  --output-dir "$NEW_BOARD_EXPERIMENT_DIR"
```

该脚本在目标平台 guard 之前不导入 OpenCV/NumPy、不解码真实图像或创建
输出。输出目录需为板端 experiments 下全新目录。host 只运行生成像素和
小型合同测试，不处理真实图像。

## 独立候选：一次 full-frame remap

`single_remap_cache.py` 的 `SingleRemapUndistorter` 从冻结原 helper 构造的
80 条带 map 按原 y 顺序拼接，逐字节/hash 验证原 map 值；没有重新调用
full-grid map 构造。每帧只调用一次 remap，retained map 大小仍为
1,843,200 B。构造时两种 map-array 表达同时在场，峰值 3,686,400 B，
不包含所有 Python/hash 暂存，也不等于 RSS 上限；关闭清空当前 maps。

7 个 generated-only 测试通过：随机/梯度/边界/零像素、partial 最后条带、
原 map 字节、owned 输出、parent SHA/budget guard 和确无新 full-grid map。
host 的 OpenCV 4.12 合成结果不能代替板端 4.10 真正 192 帧 gate。

独立 `run-v40-undistort-single-board.py` 的 r3 已完成板端 192 帧 full /
half / normalized 逐字节 gate；另保存的 12 个 sample 数组有独立重读检查。
同帧 ABBA 各 48 次中，完整 prep（不含 decode）均值为
**12.764 → 6.672 ms**（P95 **12.854 → 6.827 ms**），仅 undistort/remap
为 **11.947 → 5.821 ms**。初次 map 构造/拼接/校验 **12.272 ms** 另列，
该实验峰值 RSS **72.43 MiB**，cleanup/source/线程恢复通过。
r2 与 r3 是不同次进程的独立成对实验，不能把它们的 cached 均值差作为
受控调度收益；原 r2 证据和默认 pipeline 均未修改。单 remap 结果没有
集成完整 VO，也没有新的神经计算、pose 精度或 20 Hz 验收结论。

新入口与 r2 使用相同 CLI，改 `--cache-helper single_remap_cache.py`，
原 `fixed_undistort_cache.py` 必须相邻且 byte-pinned，以新输出目录保存。

来源、保留的文件级版权/条款及 OpenCV 仓库许可证见
[OPENCV_NOTICE.md](OPENCV_NOTICE.md)。所有模型/数据/厂商 SDK 均不随此目录
分发，代码与权重许可独立。
