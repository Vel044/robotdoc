# 实验 06：摄像头感知层级耗时剖析

> 对应论文章节：第 3.1 节 感知阶段性能剖析（摄像头路径）
> 状态：⬜ 待执行（设计已完成）

## 一、实验目标

量化 LeRobot 摄像头感知链路从硬件帧到达到 tensor 入模之间**每一层**的耗时分布，
回答以下三个问题：

1. **后台线程的端到端时延**（从 `pselect6` 等到帧到 BGR→RGB 完成）由哪些子步骤构成？各占多少？
2. **主线程的感知耗时**为什么只有约 1.9 ms？后台线程异步预取消化了多少时间？
3. 目前 ACT 输入图像分辨率 / 解码格式 / 预处理链是否存在可消除的非必要开销（特别是 libjpeg 解码）？

## 二、被测系统

| 项 | 配置 |
|---|---|
| 硬件 | 树莓派 5（Cortex-A76 ×4，8 GB） + 两个 USB 摄像头（顶部俯视 + 腕部跟拍） |
| 操作系统 | Ubuntu 24.04 (Linux 6.x) |
| 库 | LeRobot v0.3.4 / OpenCV 4.x / Python 3.10 |
| 摄像头参数 | 640×480 @ 30 fps，FOURCC=`MJPG`（与生产配置一致） |
| 后台线程 | `OpenCVCamera._read_loop`（始终启用，主线程通过 `async_read` 取最新缓存帧） |

代码入口：[lerobot/src/lerobot/cameras/opencv/camera_opencv.py](../../lerobot/src/lerobot/cameras/opencv/camera_opencv.py) L539、L667、L754；
观测主路径：[lerobot/src/lerobot/record.py:681](../../lerobot/src/lerobot/record.py)。

## 三、被测层级（9 层）

| # | 层 | 归属 | 计时手段 | 预期量级 |
|---|---|------|---------|---------|
| 1 | `pselect6` 阻塞等待新帧 | 内核 | ftrace `sys_enter_pselect6` / `sys_exit_pselect6` | 23–32 ms |
| 2 | `VIDIOC_DQBUF` ioctl | 内核 | ftrace `sys_enter_ioctl` 配 `cmd` 过滤 | < 0.1 ms |
| 3 | libjpeg MJPEG 解码 | 后台线程 (CPU) | Python `perf_counter` 包 `videocapture.read()` 整体减去 grab 部分 | 10–30 ms |
| 4 | numpy ndarray 构造/拷贝 | 后台线程 | `perf_counter` | < 1 ms |
| 5 | `_postprocess_image` BGR→RGB | 后台线程 | `perf_counter` | 0.1–0.3 ms |
| 6 | `frame_lock` 等 + 取帧 | 主线程 | `perf_counter` 包 `async_read` | ≈ 1.9 ms |
| 7 | `observation_processor` HWC→CHW + dtype | 主线程 | `perf_counter` | 待测 |
| 8 | `normalize_processor` /255 + mean/std | 主线程 | `perf_counter` | 待测 |
| 9 | `device_processor` `to(device)` | 主线程 | `perf_counter` | 待测（CPU only ≈ 0） |

## 四、实验设计

### 4.1 控制变量
- 固定 cs=100、fps=30、分辨率 640×480、FOURCC=MJPG；
- 固定 CPU 频率（使用 `cpufreq-set -g performance`）排除热降频；
- 关闭后台桌面进程，仅运行 `record.py`。

### 4.2 采样
- 3 个 episode × 每 episode 30 帧；
- 每 episode 跳过前 30 帧冷启动；
- 每帧每层一行写入 `layer_timing.csv`。

### 4.3 对照组
- **A 组**：不开 ftrace（仅 Python `perf_counter`）；
- **B 组**：开 ftrace（`sys_enter_pselect6` / `sys_exit_pselect6` / `sys_enter_ioctl` 三组 tracepoint）；
- 对照 A、B 组 Python 测得的层 3–9 数据应一致（误差 < 5%），以验证 ftrace 插桩对用户态测量无显著干扰。

## 五、产出

| 产出 | 说明 |
|------|------|
| `architecture.png` | **图 1 分层架构图**：纵向"主线程 / 后台线程 / OpenCV C++ / 内核 / 主线程预处理"五带，画清 `record.py:681` → USB urb 全栈（draw.io / PlantUML 出 PNG） |
| `layer_timing.png` | **图 2 逐层耗时分布**：水平堆叠条形 + 误差棒，按线程归属配色（主线程绿 / 后台线程蓝 / 内核橙 / 预处理灰），同图标出"端到端时延（1+2+3+4+5）"和"主线程感知耗时（6+7+8+9）"两条参考线 |
| `layer_timing.csv` | 原始数据，列：`episode, frame, layer_id, layer_name, t_ms` |
| `plot_layer_timing.py` | matplotlib 绘图脚本 |
| `实验设计.md` | 测量方法、ftrace 插桩点、Python perf_counter monkey-patch 详细说明 |

## 六、预期结论模板（实测后回填）

1. **后台线程总耗时 ≈ 1/帧率**：受 `pselect6` 物理极限制约，无法低于 33.3 ms @30fps；
2. **主线程感知耗时 ≪ 后台总耗时**：得益于异步线程 + 缓存帧设计，主线程只承担锁等待 + 预处理；
3. **libjpeg 解码是后台线程内最大可优化点**：占后台耗时 30–60%；树莓派 5 GPU（VideoCore VII）不支持 MJPEG 硬解（仅 H.264/H.265），软解是当前唯一路径；
4. **预处理链总和应在 1 ms 内**：若超出则说明 `to(device)` 或 normalize 出现意外拷贝，作为告警阈值；
5. 若实测结果与预期偏离 > 30%，需检查 USB 总线竞争、CPU 热降频、是否启用 `MJPG` FOURCC。

## 七、相关文档

- [robotdoc/摄像头相关/Lerobot相机调用链分析.md](../../摄像头相关/Lerobot相机调用链分析.md)
- [robotdoc/摄像头相关/OpenCV视频流内部实现.md](../../摄像头相关/OpenCV视频流内部实现.md)
- [robotdoc/系统调用/pselect6最终结果.md](../../系统调用/pselect6最终结果.md)
- 已完成：实验 01（chunk_size 扫描）、实验 02（三类任务对比）

## 八、运行步骤

```bash
# 在树莓派 5 上
conda activate lerobot
cd ~/Work/RobotOS/Lerobot

# A 组（无 ftrace）
python lerobot/analysis/06_camera_layer_profiling/profile_camera.py \
    --episodes 3 --frames 30 \
    --out robotdoc/实验/06_摄像头感知层级耗时剖析/layer_timing.csv

# B 组（ftrace 启用）
sudo ./lerobot/analysis/06_camera_layer_profiling/start_ftrace.sh
python lerobot/analysis/06_camera_layer_profiling/profile_camera.py \
    --episodes 3 --frames 30 --use-ftrace \
    --out robotdoc/实验/06_摄像头感知层级耗时剖析/layer_timing_ftrace.csv
sudo ./lerobot/analysis/06_camera_layer_profiling/stop_ftrace.sh

# 出图
python lerobot/analysis/06_camera_layer_profiling/plot_layer_timing.py \
    --csv robotdoc/实验/06_摄像头感知层级耗时剖析/layer_timing.csv \
    --out robotdoc/实验/06_摄像头感知层级耗时剖析/layer_timing.png
```
