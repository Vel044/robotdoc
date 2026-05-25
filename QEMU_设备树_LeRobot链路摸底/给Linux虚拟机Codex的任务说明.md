# 给 Linux 虚拟机 Codex 的 QEMU 任务说明

## 目标

这份文档是给 Linux 虚拟机里的 Codex 用的。

当前项目想搞清楚：

```text
如果要用 QEMU 模拟 Raspberry Pi 5 + LeRobot 的硬件环境，
到底哪些硬件 QEMU 已经能模拟，
哪些可以用通用 virt 机器替代，
哪些必须 mock，
哪些需要以后写 QEMU 设备模型。
```

请不要一开始就假设 QEMU 能完整模拟 Raspberry Pi 5。当前策略是：

```text
先按“完整 Pi 5 硬件”建立清单，
再按 QEMU 现有能力做分级实现。
```

## 已有证据

先阅读这些文档：

```text
robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/01_USB拓扑.md
robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/02_RaspberryPi5启动链.md
robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/03_BCM2712与设备树.md
robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/05_DTS根节点清单.md
```

真机快照证据在：

```text
robotdoc/QEMU_设备树_LeRobot链路摸底/树莓派5真机文件快照/2026-05-18/
```

重点文件：

```text
boot/bcm2712-rpi-5-b.dtb
boot/bcm2712-rpi-5-b.from-dtb.dts
boot/config.txt
boot/cmdline.txt
hardware/lsusb-tree.txt
hardware/lsusb.txt
hardware/udev-video0.txt
hardware/udev-video2.txt
hardware/udev-ttyACM0.txt
hardware/udev-ttyACM1.txt
modules/modules.builtin
modules/modules.dep
proc/uname-a.txt
proc/version.txt
```

## 硬件目标链路

真机 Pi 5 上，LeRobot 外设链路是：

```text
BCM2712
  -> AXI
  -> PCIe pcie@1000120000
  -> RP1
  -> RP1 DWC3 USB host
  -> Linux xhci-hcd root hub
  -> USB hub
  -> UVC camera / CDC ACM serial / HID keyboard
```

注意：

```text
DTS 里描述的是 BCM2712 / AXI / PCIe / RP1 / DWC3 USB host。
UVC 摄像头、CDC ACM 串口板、HID 键盘不是 DTS 里的静态节点，
它们是 USB host 跑起来之后运行时枚举出来的 USB 设备。
```

## 第一步：确认 Linux 虚拟机里的 QEMU 能力

在 Linux 虚拟机里先执行：

```bash
qemu-system-aarch64 --version
qemu-system-aarch64 -machine help | grep -Ei 'virt|raspi|bcm|pi'
qemu-system-aarch64 -device help | grep -Ei 'xhci|usb|uvc|serial|cdc|keyboard|hid|virtio'
```

请把结果写入新文档：

```text
robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/06_QEMU模拟能力核对.md
```

表格建议：

| 目标硬件 | 真机证据 | QEMU 是否现成支持 | 第一阶段处理 | 完整模拟处理 |
| --- | --- | --- | --- | --- |
| ARM64 CPU | `cpus` | 待核对 | 待写 | 待写 |
| GIC / timer | `timer` / interrupt controller | 待核对 | 待写 | 待写 |
| BCM2712 | root compatible | 待核对 | 待写 | 待写 |
| RP1 | DTS `rp1` | 待核对 | 待写 | 待写 |
| DWC3 USB host | `compatible = "snps,dwc3"` | 待核对 | 待写 | 待写 |
| xHCI | `Driver=xhci-hcd` | 待核对 | 待写 | 待写 |
| USB hub | `Driver=hub/4p` | 待核对 | 待写 | 待写 |
| UVC camera | `Driver=uvcvideo` | 待核对 | 待写 | 待写 |
| CDC ACM serial | `Driver=cdc_acm` | 待核对 | 待写 | 待写 |
| HID keyboard | `Driver=usbhid` | 待核对 | 待写 | 待写 |

## 第二步：先用 QEMU virt 启动编译好的 ARM64 kernel

第一阶段不要追求完整 Pi 5。先用：

```text
-machine virt
```

目标是证明：

```text
kernel 能启动
kernel 能解析 QEMU 提供的 dtb
kernel 能输出串口日志
kernel 能挂载 rootfs
kernel 能进入用户态 shell
```

先检查编译产物：

```bash
cd linux
make ARCH=arm64 kernelrelease
file arch/arm64/boot/Image
ls -lh arch/arm64/boot/Image
```

如果使用现有 rootfs 镜像，先确认它是什么格式：

```bash
file rootfs.img
qemu-img info rootfs.img
```

QEMU 启动命令骨架：

```bash
qemu-system-aarch64 \
  -machine virt \
  -cpu max \
  -m 4096 \
  -nographic \
  -kernel linux/arch/arm64/boot/Image \
  -append "console=ttyAMA0 root=/dev/vda rw rootwait" \
  -drive if=none,file=rootfs.img,format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0
```

如果 rootfs 不是 `/dev/vda`，需要按实际块设备改 `root=`。

如果没有 rootfs，先做最小 initramfs 或最小 rootfs，不要先卡在 LeRobot。

## 第三步：确认新 kernel 需要哪些 QEMU virt 驱动

在 `.config` 里至少核对这些方向：

```text
ARM64
GIC
ARM generic timer
PL011 UART
virtio
virtio-blk
ext4
devtmpfs
procfs
sysfs
tmpfs
```

如果 QEMU virt 启动失败，请优先检查：

```text
console=ttyAMA0 是否匹配
root= 是否匹配
virtio-blk 是否 built-in 或 initramfs 能加载
ext4 是否 built-in 或 initramfs 能加载
devtmpfs 是否能自动挂载
```

## 第四步：再模拟通用 USB

在 QEMU virt 跑通后，再加通用 USB：

```bash
-device qemu-xhci
-device usb-kbd
-device usb-hub
-device usb-serial
```

目标是先确认 guest kernel 能看到：

```text
xHCI host
USB root hub
USB keyboard
USB serial-like device
```

注意：

```text
QEMU 的 usb-serial 不一定等价于真机的 CDC ACM / ttyACM。
如果 guest 里出现的是 ttyUSB，而不是真机的 ttyACM，这是预期差异，需要记录。
```

## 第五步：UVC 摄像头和 CDC ACM 先不要硬啃

真机 LeRobot 需要：

```text
/dev/video0
/dev/video2
/dev/ttyACM0
/dev/ttyACM1
```

但第一阶段可以这样处理：

| 真机设备 | 第一阶段 QEMU 处理 |
| --- | --- |
| UVC camera | mock 图像帧、预录视频、或者后续再研究 v4l2loopback |
| CDC ACM serial | mock MotorsBus、伪串口、或者后续再研究 USB 透传 |
| HID keyboard | `usb-kbd` 可以直接模拟 |

请明确记录：

```text
这是用户态验证替代方案，不等于完整 Pi 5 硬件模拟。
```

## 第六步：判断是否需要写 QEMU 设备模型

如果目标升级为“完整模拟 Pi 5”，需要继续判断：

```text
QEMU 是否已有 raspi5 machine
QEMU 是否已有 BCM2712 machine model
QEMU 是否已有 RP1 PCIe endpoint model
QEMU 是否能把 RP1 内部 DWC3 host 暴露成和真机 DTB 一致的设备
QEMU 是否已有 USB UVC camera device model
QEMU 是否已有 CDC ACM device model
```

如果没有，就要写 QEMU 设备模型。请先只输出分析文档，不要直接动 QEMU 源码。

建议输出：

```text
robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/07_完整Pi5模拟缺口.md
```

内容包括：

```text
缺哪个设备模型
真机证据是什么
Linux guest 期待看到什么 compatible / PCI ID / USB class
QEMU 现有哪个模型可以参考
第一阶段是否可以替代
```

## 最终交付

请在 Linux 虚拟机里产出这些文件：

```text
06_QEMU模拟能力核对.md
07_完整Pi5模拟缺口.md
```

如果成功启动 QEMU virt，还要记录：

```text
QEMU 命令
kernel release
完整启动日志里的关键几行
guest 里 uname -a
guest 里 /proc/cmdline
guest 里 lsblk
guest 里 lsusb -t
```

如果启动失败，也要记录：

```text
失败的 QEMU 命令
最后 80 行串口日志
判断卡在哪一步
下一步要改 kernel config、rootfs、cmdline 还是 QEMU 参数
```

## 一句话原则

```text
先用 QEMU virt 验证 kernel 和用户态能跑，
再用真机 DTB / USB 树反推完整 Pi 5 模拟缺口，
不要把“QEMU 能启动 ARM64 Linux”和“QEMU 已完整模拟 Pi 5”混为一谈。
```
