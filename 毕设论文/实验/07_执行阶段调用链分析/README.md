# 实验 07：执行阶段调用链分析

> 对应论文章节：第 3.3.1 节 LeRobot 执行路径软件栈、第 3.3.2 节 串口到内核的系统调用路径
> 本实验无新数据，汇总现有文档的链路图规格

---

## 一、实验目标

为论文 3.3 节提供两张结构化调用链图，替代原有的纯文字描述。
图由外部工具（draw.io / Mermaid / Excalidraw 等）绘制，此目录仅存放规格文档。

---

## 二、输出文件

| 文件 | 说明 |
|------|------|
| `execution_callstack_software_spec.md` | 图 1 规格：LeRobot Python 软件栈 |
| `execution_callstack_kernel_spec.md` | 图 2 规格：串口到内核的系统调用路径 |
| `3_性能分析/image/execution_callstack_software.pdf` | 图 1（外部绘制后放入） |
| `3_性能分析/image/execution_callstack_kernel.pdf` | 图 2（外部绘制后放入） |

---

## 三、数据来源

| 原始文档 | 对应图 |
|----------|--------|
| `robotdoc/舵机相关/Lerobot内->scservo_sdk舵机读写完整链路.md` | 图 1 软件栈架构、call stack、寄存器地址 |
| `robotdoc/舵机相关/scservo_sdk写链路.md` | 图 2 写链路 setup→txPacket→ser.write 全流程 |
| `robotdoc/舵机相关/scservo_sdk读链路.md` | 图 2 读链路 setup→txRxPacket→rxPacket→getData 全流程 |
| `robotdoc/舵机相关/Feetech舵机串口read_write_ioctl系统调用接口分析.md` | 图 2 kernel 层 ioctl/termios 部分（补充） |

---

## 四、图 1：LeRobot Python 软件栈

**关键数据点（来自 Lerobot内->scservo_sdk舵机读写完整链路.md）**：

| 路径 | 函数 | 关键数据 |
|------|------|----------|
| 读链路 | `sync_read("Present_Position")` | 寄存器地址=56，长=2字节，bit15 sign-magnitude |
| 写链路 | `sync_write("Goal_Position")` | 寄存器地址=42，长=2字节，无回包 |
| 读帧格式 | `GroupSyncRead` | 14 字节：`[FF FF FE 0A 82 ADDR LEN ID1..ID6 CS]` |
| 写帧格式 | `GroupSyncWrite` | 26 字节：6电机×(1字节ID+2字节数据)+包头 |
| 耗时 | 读 ~1.2 ms/帧，写 ~1~3 ms | 半双工约束 |

---

## 五、图 2：串口到内核系统调用路径

**关键数据点（来自 scservo_sdk写链路.md + scservo_sdk读链路.md）**：

| 层 | 函数 | 关键数据 |
|----|------|----------|
| pyserial | `Serial.write()` / `Serial.read()` | CPython C API 绑定 |
| Glibc | `__libc_write()` / `__libc_read()` | ~38 μs/次 |
| VFS | `vfs_write()` / `vfs_read()` | POSIX 文件抽象层 |
| TTY | `tty_write()` / `n_tty_receive()` | 行规程处理 |
| USB | `usb_bulk_msg()` / `dwc_otg` | 树莓派 5 USB 控制器 |
| 硬件 | STS-3215 × 6 | 1 Mbps，波特率 |

---

## 六、验证清单

- [ ] `execution_callstack_software_spec.md` 存在
- [ ] `execution_callstack_kernel_spec.md` 存在
- [ ] 图 1 PDF 存在于 `3_性能分析/image/execution_callstack_software.pdf`
- [ ] 图 2 PDF 存在于 `3_性能分析/image/execution_callstack_kernel.pdf`
- [ ] `3_3_执行阶段性能剖析.tex` 正确引用两张图
- [ ] `latexmk` 编译无错误