# LeRobot 文档中心

本目录存放与 LeRobot 项目相关的代码分析文档、技术笔记和性能瓶颈分析。

## 目录结构

```
robotdoc/
├── ACT模型/          # ACT 模型原理、架构
├── 推理相关/          # 推理流水线、predict_action 时间分析
├── RecordLoop/       # record 录制循环、数据流、全链路
├── 舵机相关/          # Feetech 舵机、ttyACM0、motors_bus、串口读写
├── 摄像头相关/        # OpenCV、V4L2、USB 视频流
├── 系统调用/          # futex、pselect6、ioctl、Linux 内核源码
└── PPT/              # 演示文件（.pptx，内容跨领域）
```

## 文档索引

### ACT模型

| 文档 | 说明 |
|------|------|
| [ACT模型原理.md](ACT模型/ACT模型原理.md) | ACT 算法核心原理、CVAE 编解码器、动作分块 |
| [ACT推理.md](ACT模型/ACT推理.md) | ACT 推理过程分析 |
| [全部分析.md](ACT模型/全部分析.md) | ACT 模型综合分析汇总 |
| [内核数据搜集分析.md](ACT模型/内核数据搜集分析.md) | ACT 推理内核态时间数据搜集与分析 |

附件：`Papers/`（ACT 论文中英文 PDF）、`Picture/`（架构图、流程图）

### 推理相关

| 文档 | 说明 |
|------|------|
| [async_inference_推理执行流水线分析.md](推理相关/async_inference_推理执行流水线分析.md) | 异步推理流水线设计与执行流程 |
| [predict_action时间分析.md](推理相关/predict_action时间分析.md) | predict_action 调用耗时分析、性能瓶颈 |

### RecordLoop

| 文档 | 说明 |
|------|------|
| [record_loop数据流分析.md](RecordLoop/record_loop数据流分析.md) | record 主循环数据流 |
| [LeRobot观察推理执行循环完整链路分析.md](RecordLoop/LeRobot观察推理执行循环完整链路分析.md) | 观察→推理→执行完整循环链路 |
| [LeRobot软件栈全链路分析.md](RecordLoop/LeRobot软件栈全链路分析.md) | 应用层到硬件层完整调用栈 |

### 舵机相关

| 文档 | 说明 |
|------|------|
| [Feetech_舵机库执行指南.md](舵机相关/Feetech_舵机库执行指南.md) | scservo_sdk 核心架构与使用指南 |
| [Feetech舵机串口read_write_ioctl系统调用接口分析.md](舵机相关/Feetech舵机串口read_write_ioctl系统调用接口分析.md) | 舵机串口 read/write/ioctl 系统调用接口 |
| [motors_bus_异步读写设计分析.md](舵机相关/motors_bus_异步读写设计分析.md) | motors_bus 异步读写设计 |
| [observation读取ACM0数据完整流程.md](舵机相关/observation读取ACM0数据完整流程.md) | record.py → ttyACM0 完整读取流程 |
| [record到舵机调用链完整分析.md](舵机相关/record到舵机调用链完整分析.md) | record → 舵机执行完整调用链 |
| [ttyACM0串口调用链详细分析.md](舵机相关/ttyACM0串口调用链详细分析.md) | USB ACM 驱动内核调用链 |
| [舵机读写完整链路.md](舵机相关/舵机读写完整链路.md) | 舵机读写端到端链路 |
| [读串口链路.md](舵机相关/读串口链路.md) | USB 读串口链路分析 |
| [USB串口写路径.md](舵机相关/USB串口写路径.md) | USB 串口写路径分析 |
| [方法论.md](舵机相关/方法论.md) | 串口分析方法论 |
| [链路拆解.md](舵机相关/链路拆解.md) | 串口链路逐层拆解 |

### 摄像头相关

| 文档 | 说明 |
|------|------|
| [OpenCV视频流数据流分析.md](摄像头相关/OpenCV视频流数据流分析.md) | OpenCV 采集到处理的完整数据流 |
| [视频流读入系统调用分析.md](摄像头相关/视频流读入系统调用分析.md) | V4L2 视频流读入系统调用链 |

### 系统调用

| 文档 | 说明 |
|------|------|
| [futex系统调用.md](系统调用/futex系统调用.md) | futex 系统调用分析 |
| [futex最终结果.md](系统调用/futex最终结果.md) | futex 分析结论 |
| [pselect6调用链路.md](系统调用/pselect6调用链路.md) | pselect6 调用链路分析 |
| [pselect6最终结果.md](系统调用/pselect6最终结果.md) | pselect6 分析结论 |
| [线程分析.md](系统调用/线程分析.md) | PyTorch 线程分析 |
| [线程搜索.md](系统调用/线程搜索.md) | 线程搜索过程记录 |
| [ACT->futex系统调用.md](系统调用/ACT->futex系统调用.md) | ACT 模型触发 futex 的完整路径 |
| [futex与pselect6源码分析.md](系统调用/futex与pselect6源码分析.md) | futex + pselect6 联合源码分析 |
| [Linux_futex_pselect6_源码分析.md](系统调用/Linux_futex_pselect6_源码分析.md) | Linux 内核 futex/pselect6 深度源码分析 |
| [Linux_ioctl内核源码完整分析.md](系统调用/Linux_ioctl内核源码完整分析.md) | ioctl 系统调用内核源码完整分析 |
| [ioctl_serial和V4L2调用链分析.md](系统调用/ioctl_serial和V4L2调用链分析.md) | 串口 ioctl 与 V4L2 调用链对比 |

### PPT

| 文件 | 说明 |
|------|------|
| [全链路.pptx](PPT/全链路.pptx) | LeRobot 全链路演示 |
| [4.13双周会.pptx](PPT/4.13双周会.pptx) | 双周会汇报 |
| [读写feetech.pptx](PPT/读写feetech.pptx) | Feetech 舵机读写专题 |
| [不同trunk、异步.pptx](PPT/不同trunk、异步.pptx) | trunk 差异与异步设计 |

## 环境信息

- **Conda 环境**: `lerobot` (Python 3.10)
- **LeRobot 版本**: 0.3.4
- **Feetech SDK**: 1.0.0 (scservo_sdk)
- **Linux 内核源码**: 项目根目录 `linux/`
- **推理硬件**: 树莓派5 (ARM CPU，无 GPU)
- **开发环境**: MacOS

## 核心数据流

**核心瓶颈**: `lerobot/src/lerobot/record.py`

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
