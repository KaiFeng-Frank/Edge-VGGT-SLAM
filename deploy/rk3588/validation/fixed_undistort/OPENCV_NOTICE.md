# OpenCV 去畸变缓存的来源与许可

`fixed_undistort_cache.py` 用公开 Python API 实现固定 K/D/图像尺寸的 map
缓存，保留 OpenCV 原 `undistort` 的条带调度、条带 new-K 的 `cy-y`、
`CV_16SC2 + CV_16UC1` 固定点 map 和每条带 `INTER_LINEAR` remap。
`single_remap_cache.py` 是单独实验：拼接这些原 map 的逐字节值，然后一次
full-frame remap。Python 表达与调度改动不是未经验证的原库数值承诺。

来源是官方 [OpenCV 4.10.0 undistort.dispatch.cpp](https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/src/undistort.dispatch.cpp#L264-L305)。
该具体文件仍带 Intel / Willow Garage 等历史版权及 BSD 风格条款，完整保留
于 [OPENCV_UNDISTORT_NOTICE.txt](OPENCV_UNDISTORT_NOTICE.txt)。官方仓库根
[LICENSE](https://github.com/opencv/opencv/blob/4.10.0/LICENSE) 为 Apache-2.0，
原文本另存 [OPENCV_APACHE_LICENSE.txt](OPENCV_APACHE_LICENSE.txt)；两者
分别保留，不以项目根许可证覆盖依赖的文件级 notices。官方 URL、原文件
SHA256 与所保存 notice 的 SHA256 见
[opencv-source-notice.json](opencv-source-notice.json)。

项目侧 Python helper/诊断脚本沿用相邻 [LICENSE](LICENSE)。这里不分发
OpenCV 二进制或 C++ 实现，也不分发模型权重、数据、Rockchip SDK/header；
代码许可不授予这些独立依赖或模型权重的商业使用权。
