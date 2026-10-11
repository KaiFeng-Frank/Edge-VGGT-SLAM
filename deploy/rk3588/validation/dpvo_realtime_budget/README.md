# 保存时钟的端到端预算分析

两个脚本只读固定JSON、使用NumPy统计与保存数值，完全不加载模型、SDK、native库或图像。分析的是DPVO独立研究回放，默认VGGT不变；原SDK严格FP32拒绝、半分辨率质量拒绝和20Hz未验收均保留。

`analyze-v45-steady-performance.py`绑定v41/v43/v44实际run以及v44固定状态closed-cache AB，生成保留的r1收据。`analyze-v45-closed-online.py`复用前者的冻结数学并读取新的v45实际run，输出r2；未覆盖r1源码或收据。v41整体因EOF未知D3而failed，仅其192个已完成观测前缀用于性能描述。

预声明群体是观测35..191的157个steady fresh，排除初始化fresh34与EOF，保留冷context/捕获I/O慢样本。每帧唯一完整Update与两步BA按实际index绑定相关性和callback，校验原观测身份、原E和完整源组、release/finite/domain与时钟包含关系。源码中`_neural`串行先correlate再Update，故成功前缀的调用index可逐项关联；不把EOF或motion probe记作fresh。

v45该群体的平均读入到发布为290.135ms、P95 334.433ms；全158 fresh的P95为338.112ms，分母不同。相关性kernel48.344ms嵌在53.087ms完整相关性内；SDK run56.072ms嵌在SDK forward、fullgraph、callback和108.719ms完整Update内。它们不能再与227.548ms stream相加。其余分解与可复查来源见[结果卡](../../../../results/dpvo_realtime_budget_v45.json)。

仅在“其余阶段保持现状的串行回放”条件下，完全扣除kernel的算术估算仍约241.791ms。它不是实测优化结果，也不是未来异步或流水线的硬下界。不同轮次没有控温控频，时钟差异不用于因果归因。整体频率还包含各发布之间的进度写入/平台采样，不能取单帧均值的倒数替代它。

## 复查保存收据

需要原始字节的run JSON；结果卡中的重新序列化record不能冒充原文件SHA。路径通过CLI指定，脚本不自动下载依赖。首次只读分析产生r1，扩展脚本要求同一冻结r1 SHA。

```bash
python -B analyze-v45-steady-performance.py \
  --v41 V41_RUN --v43 V43_RUN --v44 V44_RUN --closed-ab V44_AB_RUN \
  --output steady-performance-review-r1.json

python -B analyze-v45-closed-online.py \
  --analyzer analyze-v45-steady-performance.py \
  --base-review steady-performance-review-r1.json --v45 V45_RUN \
  --output steady-performance-review-r2.json
```

八个tiny测试只覆盖生成的时钟记录、唯一调用关联、缺失历史、NaN、嵌套包含和百分位口径。真实数组/轨迹的bitwise审核由独立保存数据审核完成，不由这些统计脚本代替。
