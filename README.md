# LeRobot 文档中心

**位置**: `robotdoc/README.md`

本目录用于存放与 LeRobot 项目相关的代码分析文档、技术笔记和性能瓶颈分析。

---

## 📁 目录结构

```
robotdoc/
├── ACT模型/                    # ACT 模型相关分析
│   ├── ACT->futex系统调用/
│   ├── ACT模型General/
│   ├── futex/
│   ├── infer分析/
│   ├── pselect6/
│   ├── 分析ACT模型内核态时间/
│   └── 全部分析.md
├── Analysis/                   # 核心代码分析
│   ├── Feetech_舵机库执行指南.md
│   ├── Linux_futex_pselect6_源码分析.md
│   ├── observation读取ACM0数据完整流程.md
│   ├── record到舵机调用链完整分析.md
│   ├── ttyACM0串口调用链详细分析.md
│   ├── predict_action时间分析.md
│   ├── LeRobot软件栈全链路分析.md
│   ├── OpenCV视频流数据流分析.md
│   ├── PyTorch_线程调用链分析.md
│   └── ioctl_serial和V4L2调用链分析.md
├── Hardware/                   # 硬件层分析
│   ├── USB写串口/
│   ├── USB读串口/
│   └── USB读视频/
└── README.md                   # 本文件
```

---

## 📚 核心文档索引

### ACT 模型分析

| 文档 | 说明 |
|------|------|
| [ACT模型原理](ACT模型/ACT模型General/ACT模型原理.md) | ACT 算法核心原理 |
| [infer分析](ACT模型/infer分析/infer分析.md) | ACT 推理过程分析 |
| [futex系统调用](ACT模型/futex/futex系统调用.md) | futex 系统调用详细分析 |
| [pselect6调用链路](ACT模型/pselect6/pselect6调用链路.md) | pselect6 系统调用分析 |
| [内核数据搜集分析](ACT模型/分析ACT模型内核态时间/内核数据搜集分析.md) | 内核态性能分析 |

---

### 核心代码分析

#### 1. Feetech 舵机库执行指南
**位置**: `Analysis/Feetech_舵机库执行指南.md`

**内容**:
- 舵机库核心架构 (PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite)
- 常用常量定义 (COMM_SUCCESS, INST_READ, 等)
- 完整执行流程代码示例

**涉及源码**:
- `scservo_sdk/port_handler.py`
- `scservo_sdk/group_sync_read.py`
- `scservo_sdk/group_sync_write.py`
- `scservo_sdk/protocol_packet_handler.py`
- `scservo_sdk/scservo_def.py`
- `lerobot/src/lerobot/motors/feetech/feetech.py`
- `lerobot/src/lerobot/motors/feetech/tables.py`

---

#### 2. observation 读取 ACM0 数据完整流程
**位置**: `Analysis/observation读取ACM0数据完整流程.md`

**内容**:
- 完整调用链分析 (record.py → 串口通信)
- Present_Position 寄存器详情
- 数据归一化流程
- 时序分析

**涉及源码**:
- `lerobot/src/lerobot/record.py`
- `lerobot/src/lerobot/robots/so101_follower/so101_follower.py`
- `lerobot/src/lerobot/motors/motors_bus.py`
- `lerobot/src/lerobot/motors/feetech/feetech.py`
- `scservo_sdk/group_sync_read.py`

---

#### 3. record 到舵机调用链完整分析
**位置**: `Analysis/record到舵机调用链完整分析.md`

**内容**:
- 录制到执行舵机的完整数据流
- USB 串口写路径分析

---

#### 4. ttyACM0 串口调用链详细分析
**位置**: `Analysis/ttyACM0串口调用链详细分析.md`

**内容**:
- USB ACM 设备驱动调用链
- ttyACM0 设备的内核态实现

---

#### 5. predict_action 时间分析
**位置**: `Analysis/predict_action时间分析.md`

**内容**:
- ACT 模型推理时间分析
- 性能瓶颈识别

---

#### 6. LeRobot 软件栈全链路分析
**位置**: `Analysis/LeRobot软件栈全链路分析.md`

**内容**:
- 从应用层到硬件层的完整调用链
- 各层之间的数据传递

---

#### 7. OpenCV 视频流数据流分析
**位置**: `Analysis/OpenCV视频流数据流分析.md`

**内容**:
- 视频采集到处理的数据流
- 摄像头驱动调用分析

---

#### 8. PyTorch 线程调用链分析
**位置**: `Analysis/PyTorch_*线程调用链分析.md`

**内容**:
- Native ThreadPool 详解
- OpenMP 调用链详解
- GEMM 核心调用链

---

#### 9. Linux futex 和 pselect6 源码分析
**位置**: `Analysis/Linux_futex_pselect6_源码分析.md`

**内容**:
- **futex 系统调用**完整分析 (SYSCALL_DEFINE6 → do_futex → futex_wait/futex_wake)
- **pselect6 系统调用**完整分析 (SYSCALL_DEFINE6 → do_pselect → core_sys_select → do_select)
- **关键数据结构详解**：
  - `struct __kernel_timespec` - 内核时间结构 (Y2038 安全)
  - `__kernel_fd_set` - 文件描述符集合 (位图实现)
  - `sigset_t` - 信号集合
  - `struct sigset_argpack` - pselect6 信号参数包
  - `struct futex_q` - futex 等待队列项
  - `union futex_key` - futex 键
  - `struct poll_wqueues` - poll 等待队列
- **内核编程常见问题**：
  - `__user` 标记详解（定义、作用、为什么有的参数有有的没有）
  - `u32` 类型详解（无符号32位整数）
  - SYSCALL_DEFINE 宏详解（宏参数写法 vs C 函数写法）

**涉及源码**:
- `linux/kernel/futex/syscalls.c` - futex 系统调用入口
- `linux/kernel/futex/futex.h` - futex 内部数据结构
- `linux/kernel/futex/waitwake.c` - futex 等待/唤醒实现
- `linux/fs/select.c` - select/pselect 实现
- `linux/include/uapi/linux/futex.h` - futex UAPI 定义
- `linux/include/uapi/linux/time_types.h` - 时间结构定义
- `linux/include/uapi/linux/posix_types.h` - fd_set 定义
- `linux/include/uapi/asm-generic/signal.h` - sigset_t 定义
- `linux/include/linux/compiler.h` - `__user` 宏定义
- `linux/include/linux/types.h` - `u32` 类型定义
- `linux/include/linux/syscalls.h` - SYSCALL_DEFINE 宏定义

---

#### 10. ioctl serial 和 V4L2 调用链分析
**位置**: `Analysis/ioctl_serial和V4L2调用链分析.md`

**内容**:
- USB 串口 ioctl 调用链
- V4L2 视频设备调用链

---

### 硬件层分析

#### 1. USB 写串口
**位置**: `Hardware/USB写串口/`

**内容**:
- USB 串口写路径分析
- 方法论文档
- 链路拆解

---

#### 2. USB 读串口
**位置**: `Hardware/USB读串口/`

**内容**:
- 读串口链路分析

---

#### 3. USB 读视频
**位置**: `Hardware/USB读视频/`

**内容**:
- 视频流读入系统调用分析

---

## 🎯 环境信息

- **Conda 环境**: `lerobot` (Python 3.10)
- **LeRobot 版本**: 0.3.4
- **Feetech SDK**: 1.0.0 (scservo_sdk)
- **Linux 内核源码**: 项目根目录 `linux/`
- **推理硬件**: 树莓派5 (ARM CPU，无 GPU)
- **开发环境**: MacOS

---

## 📝 文档规范

所有文档都遵循以下规范：
1. **每个代码片段都标注源码位置** (文件路径 + 行号)
2. **使用相对路径** (以 `lerobot/`、`linux/`、`scservo_sdk/` 开头)
3. **包含完整的调用链分析**
4. **标注关键参数和常量**
5. **解释非标准数据结构的定义和用途**
6. **解释内核编程常见问题和设计原因**

---

## 📊 分析重点

**核心瓶颈**: `lerobot/src/lerobot/record.py` — 一个 episode 的完整录制流程

数据流（录制一个 episode）：

```
RecordRunner.start()
  ├─ 初始化 Robot（cameras + motors）
  ├─ 初始化 Teleoperator（遥控臂/手柄）
  ├─ 初始化 Dataset
  └─ 循环 N 个 episode：
       ├─ 采集图像帧 → ImageWriter（异步写盘）
       ├─ 读取关节位置（Feetech 舵机）
       ├─ 获取 action（遥控输入 or 策略推理）
       └─ 执行 action → 物理机器人
```
