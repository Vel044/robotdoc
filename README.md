# LeRobot 工作区

**位置**: `workspace/README.md`

本目录用于存放与 LeRobot 项目相关的文档、脚本、笔记和输出文件。

---

## 📁 目录结构

```
workspace/
├── docs/                    # 文档和指南
│   ├── Feetech_舵机库执行指南.md
│   ├── SO101_Follower_读取ACM0数据完整流程.md
│   └── Linux_futex_pselect6_源码分析.md
├── scservo_sdk_src/         # 舵机库源码
│   ├── __init__.py
│   ├── port_handler.py
│   ├── packet_handler.py
│   ├── protocol_packet_handler.py
│   ├── group_sync_read.py
│   ├── group_sync_write.py
│   └── scservo_def.py
├── scripts/                 # 自定义脚本
├── notes/                   # 学习笔记和备忘录
├── outputs/                 # 输出文件（模型、日志、数据等）
└── README.md               # 本文件
```

---

## 📚 已有文档

### 1. Feetech 舵机库执行指南
**位置**: `workspace/docs/Feetech_舵机库执行指南.md`

**内容**:
- 舵机库核心架构 (PortHandler, PacketHandler, GroupSyncRead, GroupSyncWrite)
- 常用常量定义 (COMM_SUCCESS, INST_READ, 等)
- 完整执行流程代码示例
- 所有代码片段都标注了源码位置

**涉及源码**:
- `workspace/scservo_sdk_src/port_handler.py`
- `workspace/scservo_sdk_src/group_sync_read.py`
- `workspace/scservo_sdk_src/group_sync_write.py`
- `workspace/scservo_sdk_src/protocol_packet_handler.py`
- `workspace/scservo_sdk_src/scservo_def.py`
- `lerobot/src/lerobot/motors/feetech/feetech.py`
- `lerobot/src/lerobot/motors/feetech/tables.py`

---

### 2. SO101 Follower 读取 ACM0 数据完整流程
**位置**: `workspace/docs/SO101_Follower_读取ACM0数据完整流程.md`

**内容**:
- 完整调用链分析
- 从 record.py 到串口通信的详细流程
- Present_Position 寄存器详情
- 数据归一化流程
- 时序分析
- 所有代码片段都标注了源码位置

**涉及源码**:
- `lerobot/src/lerobot/record.py`
- `lerobot/src/lerobot/robots/so101_follower/so101_follower.py`
- `lerobot/src/lerobot/motors/motors_bus.py`
- `lerobot/src/lerobot/motors/feetech/feetech.py`
- `lerobot/src/lerobot/motors/feetech/tables.py`
- `workspace/scservo_sdk_src/group_sync_read.py`

---

### 3. Linux futex 和 pselect6 源码分析
**位置**: `workspace/docs/Linux_futex_pselect6_源码分析.md`

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
- 所有代码片段都标注了 Linux 内核源码位置

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

## 🔧 舵机库源码

**位置**: `workspace/scservo_sdk_src/`

从 conda 环境复制的 Feetech SDK 源码，方便查看和调试。

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块入口，导出主要类 |
| `port_handler.py` | 串口处理，负责与硬件通信 |
| `packet_handler.py` | 数据包处理（空实现） |
| `protocol_packet_handler.py` | 协议实现，包含所有通信协议 |
| `group_sync_read.py` | 同步读取多个舵机 |
| `group_sync_write.py` | 同步写入多个舵机 |
| `scservo_def.py` | 常量定义和数据打包函数 |

---

## 🎯 环境信息

- **Conda 环境**: `lerobot` (Python 3.10)
- **LeRobot 版本**: 0.3.4
- **Feetech SDK**: 1.0.0 (scservo_sdk)
- **Linux 内核源码**: 工作区 `linux/` 目录
- **工作区路径**: `/Users/vel-virtual/Lerobot/workspace/`

---

## 📝 文档规范

所有文档都遵循以下规范：
1. **每个代码片段都标注源码位置** (文件路径 + 行号)
2. **使用相对路径** (以 `workspace/` 或 `lerobot/` 或 `linux/` 开头)
3. **包含完整的调用链分析**
4. **标注关键参数和常量**
5. **解释非标准数据结构的定义和用途**
6. **解释内核编程常见问题和设计原因**
