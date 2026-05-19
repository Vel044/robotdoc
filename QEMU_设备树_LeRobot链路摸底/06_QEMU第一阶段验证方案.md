# 06 QEMU 第一阶段验证方案

## 第一阶段目标

QEMU 第一阶段目标不是完整仿真 Raspberry Pi 5 外设，而是先验证：

```text
ARM64 启动 + rootfs + Python/conda 用户态 + LeRobot 主流程 + mock I/O
```

不要一开始就追：

```text
完整 RP1
真实 DWC3/xHCI 时序
真实 UVC 摄像头
真实 CDC ACM 舵机串口
```

这些仍然用树莓派真机验证。

## QEMU 里先测什么

| 真实链路 | QEMU 第一阶段处理 |
| --- | --- |
| ARM64 kernel 启动、cmdline、rootfs | 应该跑通。 |
| Python/conda/LeRobot import | 应该跑通。 |
| ACT `torch==2.7.1+cpu` 推理 | 可以跑 CPU 路径，但性能单独评估。 |
| dataset 写入、视频编码、`save_episode()` | 可以用 mock frame 在 QEMU 内测。 |

## QEMU 里先 mock 什么

| 真实设备 | QEMU 第一阶段替代 |
| --- | --- |
| `/dev/video0` handeye UVC camera | mock 图像、预录帧或 V4L2 loopback。 |
| `/dev/video2` fixed UVC camera | mock 图像、预录帧或 V4L2 loopback。 |
| `/dev/ttyACM0` follower 舵机串口 | 伪串口或 mock `MotorsBus`。 |
| `/dev/ttyACM1` leader 舵机串口 | mock 或跳过 `teleop.connect()`。 |
| RP1 / DWC3 / xhci / UVC / CDC ACM 真实时序 | 树莓派真机验证。 |

## QEMU 和 Pi 5 设备树的关系

如果用 QEMU `virt` 机器：

```text
QEMU 会给一套 virt 机器设备树。
这不是 Pi 5 的 bcm2712-rpi-5-b.dtb。
适合验证 ARM64 kernel、rootfs、用户态、mock I/O。
不适合验证 BCM2712/RP1 真实硬件。
```

如果要逼近 Pi 5 真硬件：

```text
必须处理 BCM2712 / RP1 / DWC3 / xHCI / UVC / CDC ACM。
这已经不是第一阶段。
```

## 第一阶段验收

```text
1. QEMU 能启动新 kernel。
2. 能挂载 rootfs。
3. 能进入 Python/conda 环境。
4. 能 import lerobot.record。
5. 能加载 ACT config 和 safetensors。
6. 能用 mock observation 跑 ACT CPU 推理。
7. 能用 mock action 跑 dataset.add_frame() / save_episode()。
```

