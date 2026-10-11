# v49 原模型前缀的保存数据审查

实际捕获由主任务在 RK3588S 串行完成。10 组真实 TUM 输入的捕获收据 SHA 为 `bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc`。捕获总耗时 142.998 秒，包含 5 GB checkpoint 哈希、模型加载和保存；进程峰值 RSS 1725.695 MiB。这里仅运行原 DINO 全部 24 层前缀，以及两组 heldout 的首个原 frame/global block，不能由此推断完整模型内存或输入到位姿性能。

独立只读审查物理读取 24 份新保存数组、2 份旧冻结参考、41 份原模型源码和 3 份当前冻结运行源码。每份数组核对 file/raw SHA、shape、dtype、布局和 finite；原配对 36 份参数元数据逐项对原 export。全部样本的 5 个特殊 token 按 first/other 角色逐 bit 对旧原输入，位置逐 bit 对旧原 positions，并独立对照 18×24 的 y,x+1 网格和前 5 个零位置。8 组 calibration、2 组 evaluation 的 20 张原图 index、原始时间戳和 PNG SHA 互斥，输入 raw SHA 也互斥，旧评估 0/8 排除。所有读取文件的字节最后重新核验不变。

只读审查未运行真实模型、ORT、SDK、校准、编译或图像解码。原 checkpoint 和物理源 PNG 的哈希由冻结板端捕获记录，主机未重新读取这些板端文件。参数值和原 frame/global 连续性由板端冻结脚本核验；本地审查验证其来源、保存数组与身份元数据，不声称又执行了一次原运算。

| 样本 | 原前缀耗时，秒 | A76 温度采样，°C（核 0 / 核 1） |
|---|---:|---:|
| calibration_00 | 7.268 | 86.777 / 86.777 |
| calibration_01 | 8.393 | 87.666 / 88.555 |
| calibration_02 | 10.707 | 85.888 / 85.888 |
| calibration_03 | 11.990 | 86.777 / 86.777 |
| calibration_04 | 12.891 | 85.888 / 85.888 |
| calibration_05 | 14.515 | 85.888 / 85.888 |
| calibration_06 | 15.682 | 85.888 / 86.777 |
| calibration_07 | 15.944 | 86.777 / 86.777 |
| evaluation_00 | 18.222 | 85.888 / 85.888 |
| evaluation_01 | 17.041 | 85.888 / 85.888 |

八组 calibration 的形状和 DINO 层数一致，耗时按捕获顺序从 7.268 秒增加到 15.944 秒。温度只在每组前缀结束后读取一次，既没有连续温度记录，也没有同期 CPU 频率和调度记录；表中温度并不随耗时单调增加。因此只记录耗时变化与高温采样，不作热降频或其他原因的因果结论。evaluation 多执行原 frame0/global0，耗时不宜与 calibration 直接作同范围比较。计时包含 tensor 复制 hook 与验证，不是 pose latency。

完整私有审查记录 `saved-capture-r1.json` 的 SHA 为 `9ee2bae8d0348aa83bca969eac020d0e179ab9c038419d09990f25412e378d30`；可公开的精简记录是 [public-capture-audit-r1.json](public-capture-audit-r1.json)，仅保存身份、指标、调用范围和边界。新捕获 arrays、checkpoint、权重导出与 SDK 二进制不公开。两组 heldout 属于同一短序列，还不构成完整模型精度、SLAM 轨迹质量、20 Hz fresh pose / P95≤100 ms、稠密地图频率、30 分钟稳定性或 8 GB 验收。

后续完成的证据包括[板端 PTQ 编译](public-ptq-compile-audit-r1.json)、[保留输入 ORT 复核](public-heldout-ort-audit-r1.json)、[普通 SDK 对照](public-normal-audit-r1.json)、[profile 数值与普通运行逐 bit 对照](public-profile-numeric-audit-r1.json)及[实际算子放置和耗时](public-profile-operators-audit-r1.json)。[数值审查 CLI](NUMERIC_AUDIT_USAGE.md)只读取保存数据；源码与摘要由[source-freeze-numeric-r1.json](source-freeze-numeric-r1.json)绑定。捕获审查的追加冻结文件位于[capture/source-freeze-audit-r1.json](../capture/source-freeze-audit-r1.json)，其中路径相对于 capture 目录。

编译与算子审查的[发布清单及 CLI](profile-publication-r2.json)绑定两份可移植解析器：[编译日志与 dataset 审查](audit_v49_pair_compile_saved.py)、[运行时算子表审查](audit_v49_pair_profile_saved.py)。编译审查需要 NumPy 读取已保存数组和相邻冻结合同，算子表审查仅用标准库；均不调用模型或 SDK。日志中完整算子表、常量表和实际数组仍留在私有存储。
