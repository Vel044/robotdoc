# 新内核 / QEMU / 设备树 / LeRobot 链路摸底

这个目录用来集中放“写新 kernel 之前必须搞清楚的东西”。

主线：

```text
新 kernel
  -> 启动链
  -> 设备树
  -> BCM2712 / RP1 / USB
  -> Linux 用户态 ABI
  -> LeRobot record + ACT 负载
  -> QEMU 第一阶段验证
```

## 阅读顺序

| 顺序 | 文件 | 作用 |
| --- | --- | --- |
| 1 | `01_目标与总链路.md` | 先固定目标、验收负载和总链路。 |
| 2 | `02_RaspberryPi5启动链.md` | 搞清楚 Pi 5 怎么加载 kernel、dtb、cmdline、initramfs/rootfs。 |
| 3 | `03_BCM2712与设备树.md` | 搞清楚 BCM2712、`.dts/.dtsi/.dtb` 和新 kernel 必须解析的节点。 |
| 4 | `04_RP1_USB摄像头舵机链路.md` | 搞清楚 RP1、DWC3、xHCI、UVC 摄像头、CDC ACM 舵机串口。 |
| 5 | `05_新Kernel必须提供的Linux接口.md` | 搞清楚 LeRobot 用户态需要哪些 Linux ABI/syscall/设备接口。 |
| 6 | `06_QEMU第一阶段验证方案.md` | 搞清楚 QEMU 第一阶段先测什么、mock 什么、不追什么。 |
| 7 | `07_工具链版本与构建命令.md` | 搞清楚 kernel 构建工具链、版本和命令骨架。 |
| 8 | `08_树莓派真机快照逐文件解析.md` | 逐个解释从树莓派复制过来的 boot、device-tree、hardware、modules 文件。 |

## 证据位置

完整长证据保留在：

```text
../面向QEMU配置的LeRobot工具链摸底_附录.md
```

附录包含完整 `record.py` 调用链、import 表、runtime trace、Python 包版本、树莓派 SSH 核对结果和更细的 kernel 源码文件列表。

## 树莓派真机快照

已复制一份当前树莓派 5 的最小真机文件快照：

```text
树莓派5真机文件快照/2026-05-18/
```

其中包括：

```text
boot/bcm2712-rpi-5-b.dtb
boot/bcm2712-rpi-5-b.from-dtb.dts
boot/config.txt
boot/cmdline.txt
device-tree/proc-device-tree.tar
hardware/lsusb.txt
hardware/v4l2-list-devices.txt
hardware/udev-video0.txt
hardware/udev-video2.txt
hardware/udev-ttyACM0.txt
hardware/udev-ttyACM1.txt
modules/lsmod.txt
```
