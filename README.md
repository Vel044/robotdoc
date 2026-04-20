# LeRobot 文档中心

本目录存放 LeRobot 项目的代码分析文档、实验数据和性能瓶颈分析。
**核心目标**：在树莓派5（ARM CPU，无GPU）上分析 ACT 推理延迟，定位各层瓶颈，探索优化路径。

---

## 目录结构

```
robotdoc/
├── ACT模型/          # ACT 模型原理、架构、推理链路
├── 推理相关/          # 推理流水线、predict_action 时间分析
├── RecordLoop/       # record 录制循环、数据流、全链路、逐段时间
├── 舵机相关/          # Feetech 舵机、ttyACM0、motors_bus、串口读写
├── 摄像头相关/        # OpenCV、V4L2、USB 视频流
├── 系统调用/          # futex、pselect6、ioctl、Linux 内核源码
└── PPT/              # 演示文件（.pptx，内容跨领域）
```

---

## 关键实验结论（速查）

> 以下是跨文档整合的核心数字，Agent 优先读这里。

| 指标 | 测量值 | 说明 |
|------|--------|------|
| **推理延迟** | 2000–4000 ms/帧 | ACT 模型在树莓派5 ARM CPU，单帧推理耗时，是主要瓶颈 |
| **观测耗时** | 5.2 ms/帧 | 舵机 sync_read 1.2 ms + 图像获取 1.9 ms |
| **执行耗时** | 4.8 ms/帧 | sync_write 串口写，无回包等待 |
| **实际 FPS** | ~19 fps（目标 30 fps） | cs=100 配置，stall 周期性出现 |
| **pselect6 调用** | 36191 次 / 40s episode | 摄像头后台线程 V4L2 帧等待，单次阻塞 23–32 ms |
| **futex 调用** | 64569 次 / 40s episode | PyTorch 线程池同步，总等待时间 3252 s（并行多线程累计） |
| **async 满足条件** | N ≥ 2 × T\_infer × F | cs=100 不满足（2s×30fps 需要 N≥120） |
| **strace 开销** | 系统时间 20.12 s → 2.20 s | 去掉 strace 后系统时间骤降，ftrace 更准 |
| **串口波特率** | 1 Mbps | Feetech SCServo 协议 |
| **MJPEG 解码** | 10–30 ms / 帧 | libjpeg 解码是摄像头路径的 CPU 瓶颈 |

**结论**：瓶颈 = ACT 推理（CPU 矩阵乘法），不是 IO（系统调用 overhead 远小于推理）。优化方向：模型量化 / ONNX、增大 chunk_size、gRPC 双进程异步。

---

## 文档索引（含实验摘要）

### ACT模型

| 文档 | 做了什么 | 关键结论 |
|------|----------|----------|
| [ACT模型原理.md](ACT模型/ACT模型原理.md) | 分析 ACT 算法原理：CVAE 编解码器、动作分块（chunk）机制、time ensemble | 动作分块是隐藏推理延迟的核心机制；chunk_size 越大，推理开销被分摊越多 |
| [ACT推理.md](ACT模型/ACT推理.md) | 逐层拆解 ACT 推理流程：encode→latent→decode，追踪张量形状与计算路径 | 解码器的自回归交叉注意力是主要 FLOP 来源；ARM NEON SIMD 做了部分加速 |
| [全部分析.md](ACT模型/全部分析.md) | ACT 模型综合分析汇总 | 综合各子文档结论 |
| [内核数据搜集分析.md](ACT模型/内核数据搜集分析.md) | 用 ftrace 采集 ACT 推理期间的内核态时间数据（同类文档见系统调用/内核数据搜集分析.md） | 内核态时间占比极低，用户态 CPU 计算才是真正瓶颈 |

### 推理相关

| 文档 | 做了什么 | 关键结论 |
|------|----------|----------|
| [async\_inference\_推理执行流水线分析.md](推理相关/async_inference_推理执行流水线分析.md) | 分析 lerobot 官方双进程 gRPC 方案（robot\_client.py + policy\_server.py），推导流水线不 stall 条件 | 不 stall 条件：N ≥ 2 × T\_infer × F；cs=100 + 2s 推理 + 30fps → 需要 N≥120，当前不满足；实际约 20fps；chunk B 前 50 帧因过期被丢弃，time ensemble 失效 |
| [predict\_action时间分析.md](推理相关/predict_action时间分析.md) | 对 predict\_action() 调用打点计时，分析各段耗时分布 | 推理耗时 2000–4000 ms，偶有 2148 ms 峰值；观测+执行合计仅 10 ms，推理占总循环时间 >99% |

---

### RecordLoop

| 文档 | 做了什么 | 关键结论 |
|------|----------|----------|
| [record\_loop数据流分析.md](RecordLoop/record_loop数据流分析.md) | 追踪 SO101 机械臂在 Policy 推理路径下的 record\_loop 主循环，逐帧分析观测→推理→执行数据流 | 40s episode 含 1200 帧@30fps 设计值；实际 5.3 s/100 帧（≈19 fps）；推理 28.5 ms（极小 episode 或量化后才能达到），观测 5.2 ms，执行 4.8 ms |
| [LeRobot观察推理执行循环完整链路分析.md](RecordLoop/LeRobot观察推理执行循环完整链路分析.md) | 从 Python 应用层到 Linux 内核层，建立观测/推理/动作三条完整的软硬件交互链路 | 串口 sync\_read 1.2 ms；V4L2 pselect6 等待最多 33 ms@30fps；PyTorch 用 OpenMP + ARM64 NEON SIMD；sync\_write 1–3 ms 无回包 |
| [LeRobot软件栈全链路分析.md](RecordLoop/LeRobot软件栈全链路分析.md) | 按观测/推理/执行三维度，分层展示 LeRobot 模块→库→标准库→内核的完整软件栈 | 观测依赖 pselect6/ioctl/mmap；推理依赖 futex（PyTorch 线程同步，最重要的系统调用）；执行依赖 TTY→USB-Serial→xHCI |
| [recordloop每部分时间全部分析.md](RecordLoop/recordloop每部分时间全部分析.md) | 用 Python 日志 + ftrace 两种方式，测量 40s episode 内各环节时间占比 | Python 日志：单帧循环 2.2–4.2 s，推理峰值 2148 ms；ftrace：pselect6 36191 次/总睡眠 612 s（多线程并行累计），futex 64569 次/3252 s；去掉 strace 系统时间从 20.12 s→2.20 s，证实 strace 带来大量开销 |

---

### 舵机相关

| 文档 | 做了什么 | 关键结论 |
|------|----------|----------|
| [Feetech\_舵机库执行指南.md](舵机相关/Feetech_舵机库执行指南.md) | 分析 scservo\_sdk 核心架构与使用指南 | GroupSyncRead/Write 是关键抽象；txRxPacket 负责发包+等包 |
| [Feetech舵机串口read\_write\_ioctl系统调用接口分析.md](舵机相关/Feetech舵机串口read_write_ioctl系统调用接口分析.md) | 分析舵机串口 read/write/ioctl 三类系统调用的接口与时序 | 串口 open 用 O\_NONBLOCK；ioctl(TCSETS) 设波特率 1Mbps；写后立即返回，读需等回包 |
| [motors\_bus\_异步读写设计分析.md](舵机相关/motors_bus_异步读写设计分析.md) | 分析为什么上游有 `_async_read` TODO 但无 `_async_write` 规划，评估异步读写可行性 | 异步读可行（隐藏串口往返，代价 1 帧延迟，cs>1 时收益≥代价）；异步写不可行（半双工总线撞车、安全限幅 read-modify-write 链路断裂、最多省 2 ms 不在瓶颈）；真正有效：模型量化 + 增大 chunk\_size |
| [observation读取ACM0数据完整流程.md](舵机相关/observation读取ACM0数据完整流程.md) | 从命令行参数 `--robot.port=/dev/ttyACM0` 追踪参数传递、串口打开、数据读取完整链路 | 参数链：CLI→@parser.wrap→SO101FollowerConfig→FeetechMotorsBus→PortHandler→serial.Serial；Sync Read 指令码 0x82；Present\_Position 寄存器地址 56，sign-magnitude 编码（bit15 为方向位） |
| [record到舵机调用链完整分析.md](舵机相关/record到舵机调用链完整分析.md) | record.py 到舵机执行的完整调用链 | 顶层调用到底层 scservo\_sdk 的完整栈 |
| [ttyACM0串口调用链详细分析.md](舵机相关/ttyACM0串口调用链详细分析.md) | 分析 USB ACM 驱动在内核中的调用链 | ttyACM → USB-Serial → xHCI 完整驱动层次 |
| [Lerobot内舵机读写完整链路.md](舵机相关/Lerobot内舵机读写完整链路.md) | 分层展示 sync\_read（观测）和 sync\_write（执行）的完整技术栈 | Sync Read 包格式：[0xFF 0xFF 0xFE LEN 0x82 ADDR LEN ID1..ID6 CHK]；每电机返回 [0xFF 0xFF ID LEN ERR DATA\_L DATA\_H CHK]；sync\_write 无回包，1–3 ms 完成 |
| [读串口链路.md](舵机相关/读串口链路.md) | USB 读串口链路逐层分析 | 从 Python read() 到内核 URB 的完整路径 |
| [USB串口写路径.md](舵机相关/USB串口写路径.md) | USB 串口写路径分析 | write() → tty\_write → USB bulk transfer |
| [方法论.md](舵机相关/方法论.md) | 串口分析方法论：如何用 strace/ftrace/gdb 定位串口问题 | 方法论文档，非实验结果 |
| [链路拆解.md](舵机相关/链路拆解.md) | 串口链路逐层拆解过程记录 | 分析过程记录 |

---

### 摄像头相关

| 文档 | 做了什么 | 关键结论 |
|------|----------|----------|
| [OpenCV视频流数据流分析.md](摄像头相关/OpenCV视频流数据流分析.md) | 追踪 cv2.VideoCapture.read() 到 V4L2 内核的完整调用链，分析 grabFrame/retrieveFrame 两阶段 | grabFrame：VIDIOC\_DQBUF ioctl 出队，pselect6 等待新帧（最多 33 ms@30fps）；retrieveFrame：libjpeg MJPEG 解码 10–30 ms@1080p（CPU 密集瓶颈）；整体仅 2 个 ioctl（DQBUF + QBUF），其余是 CPU 计算 |
| [视频流读入系统调用分析.md](摄像头相关/视频流读入系统调用分析.md) | V4L2 视频流读入系统调用链分析 | V4L2 mmap 零拷贝 + pselect6 等待 |

---

### 系统调用

| 文档 | 做了什么 | 关键结论 |
|------|----------|----------|
| [futex系统调用.md](系统调用/futex系统调用.md) | 分析 futex 调用来源和操作码分布（op=0/1/9） | op=0(FUTEX\_WAIT)占多数；op=9(FUTEX\_WAIT\_BITSET)用于任务队列；Python multiprocessing SemLock 也走 futex |
| [futex最终结果.md](系统调用/futex最终结果.md) | ftrace 采集 futex 数据，定位调用者线程 | 大量 futex 来自 Trae 远程开发服务器（node/libuv-worker/ckg\_server）；PyTorch 线程池也大量使用 futex 同步 |
| [pselect6调用链路.md](系统调用/pselect6调用链路.md) | 追踪 pselect6 从 Python 层到内核的调用链 | V4L2 等待帧和 GIL 管理都会触发 pselect6 |
| [pselect6最终结果.md](系统调用/pselect6最终结果.md) | 用 ps + ftrace 定位 pselect6 来源线程 | TID 4222/4224（两个摄像头 \_read\_loop 线程）阻塞 23–32 ms（符合 30fps 帧间隔）；主线程 TID 3275 快速返回 200–500 ns（缓冲区有数据时）或 ret=0（数据被消费时） |
| [内核数据搜集分析.md](系统调用/内核数据搜集分析.md) | 用 ftrace 搜集 40s episode 的完整内核时间数据 | pselect6 总睡眠 612 s（多线程并行累计，非单线程串行）；futex 总等待 3252 s（同理）；单帧实际延迟主要是用户态 CPU 计算 |
| [线程分析.md](系统调用/线程分析.md) | 分析 PyTorch 线程模型和 GIL 对推理性能的影响 | PyTorch 用 OpenMP 线程池绕过 GIL；线程同步通过 futex；GIL 主要影响 Python 胶水层 |
| [线程搜索.md](系统调用/线程搜索.md) | 线程搜索过程记录（用 ps/ftrace 找各线程用途） | 过程记录，含线程 TID 与功能对应关系 |
| [ACT->futex系统调用.md](系统调用/ACT->futex系统调用.md) | 追踪 ACT 模型推理触发 futex 的完整路径 | 路径：policy.select\_action→PyTorch GEMM→ATen 线程池→futex wait/wake |
| [futex与pselect6源码分析.md](系统调用/futex与pselect6源码分析.md) | 联合分析 futex + pselect6 的内核实现 | 两者均在内核等待队列上 sleep，被唤醒后恢复执行；耗时取决于等待时长，不是系统调用本身开销 |
| [Linux\_futex\_pselect6\_源码分析.md](系统调用/Linux_futex_pselect6_源码分析.md) | 深度阅读 Linux 内核 futex/pselect6 源码 | futex 核心在 `futex_wait_queue_me()`；pselect6 核心在 `do_pselect()`；两者都用 hrtimer 实现超时 |
| [Linux\_ioctl内核源码完整分析.md](系统调用/Linux_ioctl内核源码完整分析.md) | 阅读 ioctl 系统调用内核源码，追踪到设备驱动 | ioctl 通过 file\_operations→unlocked\_ioctl 分发；V4L2 和 serial 各走不同的 ioctl handler |
| [ioctl\_serial和V4L2调用链分析.md](系统调用/ioctl_serial和V4L2调用链分析.md) | 对比串口 ioctl 与 V4L2 ioctl 的调用链 | serial ioctl 主要是 TCSETS（配置串口参数）；V4L2 ioctl 主要是 VIDIOC\_DQBUF/QBUF（帧缓冲管理） |

---

### PPT

| 文件 | 说明 |
|------|------|
| [全链路.pptx](PPT/全链路.pptx) | LeRobot 全链路演示（观测→推理→执行） |
| [4.13双周会.pptx](PPT/4.13双周会.pptx) | 双周会汇报（2025-04-13） |
| [读写feetech.pptx](PPT/读写feetech.pptx) | Feetech 舵机读写专题 |
| [不同trunk、异步.pptx](PPT/不同trunk、异步.pptx) | trunk 差异与异步推理设计 |

