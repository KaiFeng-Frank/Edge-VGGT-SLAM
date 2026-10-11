# 四个横向相关性 dot 的独立累加研究

这是原DPVO权重研究链路的一个CPU相关性候选，默认VGGT不变。它继承冻结v36 owned NHWC帧缓存，仅重排四个有效邻点的dot调度。每个输出保留原FP32通道/lane/尾部累加顺序，边界仍用原dot；准备、池化、双线性采样、offset/level顺序与cache生命周期不变。编译必须保留`-fno-fast-math -ffp-contract=off`。

实际板端build得到32064B库，SHA `bd324375294722faeaf02b71645082e9b11e46f64bbdedee7eb52e185d0bb9cc`。冻结v36与候选使用同一Python `PerFrameCorrelation`，在三份旧全分辨率M96特征上的4个图case各执行ABBA两轮。32次reported调用全部A/B bitwise且原CPU固定`rtol=1e-4, atol=1e-5`门槛通过。实际另存8个首A/B corr输出、4个图和4个原CPU参考，共16NPZ；重复调用保存raw metadata，未另存32份输出。两路各16次成功查询、3个owned帧cache关闭后字节归零。

| case | 边数 | 暖callback A / B |
| --- | ---: | ---: |
| first_two_frames | 96 | 2.423 / 2.208 ms |
| grow_three_frames | 864 | 29.643 / 20.439 ms |
| reorder_slots | 864 | 42.856 / 33.376 ms |
| delete_middle | 384 | 24.622 / 19.238 ms |

每路每case仅3个暖样本，首次排除；未控温控频。内嵌kernel时钟不能与callback相加。此组件尚未接入半分辨率M16或在线递归，未执行CNN、Update、BA、新图像或轨迹；不证明20Hz，也不撤销SDK/半尺寸质量拒绝。[实际收据与独审](../../../../results/dpvo_realtime_budget_v45.json)。

## 复现入口

所有真实编译/相关性执行必须在RK3588S板端，ARM与device-tree守卫在读取真实数组或加载native库之前。源目录内包含原字节的v27 `dpvo_corr.cpp/h`与v36 header/Python loader副本，以维持ABI；这些历史函数名保留，候选build/AB收据使用独立v45 schema。新导出符号`dpvo_corr_v45_math_variant`标识调度版本。`LICENSE`保留Princeton原MIT许可；未分发模型、数据或compiled库，模型权重许可与代码许可分开。

需要历史原始v26 `capture.json`/`motion.json`及其NPZ、原v36库/build收据、隔离AArch64 GNU/OpenMP工具链/bootstrap收据。缺少依赖会明确拒绝，代码不会下载它们。父helper在相邻`dpvo_frame_cache/`目录。

```bash
python -B build-v45-frame-corr-dot4-board.py \
  --source-dir SOURCE --source-freeze-sha256 f1333c8d115ee2a0d1b5cafbfad187a8f9ef1e5842eaad55f467fc33d3a6faac \
  --parent-builder ../dpvo_frame_cache/build-v36-frame-corr-board.py \
  --toolchain-dir TOOLCHAIN --bootstrap-sha256 868c1aaf324f6263db743c19d62e493cc643646a0e2035f70b2efcb08448093e \
  --output-dir NEW_BOARD_BUILD

python -B run-v45-dot4-corr-ab-board.py \
  --parent-ab-helper ../dpvo_frame_cache/run-v36-frame-corr-ab-board.py \
  --frame-helper dpvo_frame_corr.py --capture-dir V26_CAPTURE --motion-dir V26_MOTION \
  --parent-library V36_BUILD/libdpvo_frame_corr.so --parent-build V36_BUILD/build.json \
  --parent-build-sha256 79a3d203ac505349d9154749059750a312217c70dc0bccec9ff42fdc9aac7194 \
  --candidate-library V45_BUILD/libdpvo_frame_corr_dot4.so --candidate-build V45_BUILD/build.json \
  --output-dir NEW_BOARD_AB
```

AB入口绑定本次已测candidate library和build收据SHA。重新编译即使库字节相同，新路径或日志也可能改变build.json的SHA；重建必须建立新的显式来源合同，同时保留本次已测合同，不可绕过硬守卫。build只证明编译，AB另行验证数值。`test_v45_dot4_math.py`仅生成很小的NumPy表达/边界数据，不运行C++；`test_v45_dot4_ab_source.py`仅检查源合同、ABBA与非ARM拒绝。
