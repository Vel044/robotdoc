# Mac 原生 QEMU 交接

## 目标

这份文档给 Mac 上的 Codex 使用。

当前 Linux VM 里已经完成了 QEMU + Raspberry Pi rootfs + LeRobot USB 外设链路摸底。结论见：

```text
模拟结果.md
```

Mac 端下一步不是复刻 VMware -> Ubuntu VM -> QEMU 的真实 USB 透传实验，而是用原生 QEMU 做更干净的 kernel / DTB / Pi5-lite / RP1 USB 子集开发。

建议分工：

```text
Mac 原生 QEMU：
  用于 kernel 启动、DTB、Pi5-ish machine、RP1-ish USB 子集开发。

Ubuntu VM QEMU：
  保留用于真实 UVC / CDC ACM USB 透传验证。
```

## 已归档的启动配置

当前 Linux VM 使用过的 QEMU 配置和启动脚本已经归档到：

```text
QEMU运行配置/QEMU-Pi5-rootfs.cfg
QEMU运行配置/QEMU启动Pi5-rootfs.sh
```

这两个文件是 Linux VM 版本，里面有 `/home/vel/...` 绝对路径，也包含 Linux host 才能稳定使用的 `usb-host` 真实 USB 透传配置。

Mac 端不要直接照抄运行，要先按 Mac 路径和 Mac 能力改。

## 需要从 Linux VM 搬到 Mac 的文件

只搬 SD/rootfs 镜像还不够。

当前启动命令使用 `-kernel` 从镜像外部加载 kernel：

```text
/home/vel/linux-rpi-6.12/arch/arm64/boot/Image
```

所以 Mac 端至少需要：

```text
1. rootfs / SD raw 镜像
   /home/vel/rpi-images/rpi_backup_20260416.img

2. 当前能启动这个 rootfs 的 ARM64 kernel Image
   /home/vel/linux-rpi-6.12/arch/arm64/boot/Image

3. 归档配置和脚本
   robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU运行配置/QEMU-Pi5-rootfs.cfg
   robotdoc/QEMU_设备树_LeRobot链路摸底/QEMU运行配置/QEMU启动Pi5-rootfs.sh

4. 本文档和实验记录
   robotdoc/QEMU_设备树_LeRobot链路摸底/
```

如果 Mac 端后续改成从镜像 boot 分区加载 kernel，那可以不再依赖外部 `Image`；但当前 Linux VM 验证过的启动方式依赖外部 `-kernel Image`。

## Mac 端路径建议

建议在 Mac 上放成类似结构：

```text
~/qemu-pi5/
  rpi-images/
    rpi_backup_20260416.img
  linux-rpi-6.12/
    arch/arm64/boot/Image
  robotdoc/
    QEMU_设备树_LeRobot链路摸底/
```

然后把归档配置复制一份出来改，不要直接改原始归档：

```text
~/qemu-pi5/QEMU-Pi5-rootfs.mac.cfg
~/qemu-pi5/QEMU启动Pi5-rootfs.mac.sh
```

需要替换的路径：

```text
/home/vel/rpi-images/rpi_backup_20260416.img
  -> /Users/<mac-user>/qemu-pi5/rpi-images/rpi_backup_20260416.img

/home/vel/linux-rpi-6.12/arch/arm64/boot/Image
  -> /Users/<mac-user>/qemu-pi5/linux-rpi-6.12/arch/arm64/boot/Image
```

## Mac 原生 QEMU 启动建议

Apple Silicon Mac 上可以先尝试 HVF：

```bash
qemu-system-aarch64 \
  -accel hvf \
  -readconfig /Users/<mac-user>/qemu-pi5/QEMU-Pi5-rootfs.mac.cfg \
  -cpu host \
  -nographic \
  -no-reboot \
  -kernel /Users/<mac-user>/qemu-pi5/linux-rpi-6.12/arch/arm64/boot/Image \
  -append "console=ttyAMA0 root=/dev/vda2 rw rootwait systemd.unit=multi-user.target"
```

如果 `-cpu host` 不可用，先退回：

```bash
-cpu cortex-a76
```

如果 `-accel hvf` 有问题，先退回纯 TCG：

```bash
-accel tcg
```

## Mac 上先不要保留真实 USB 透传

Linux VM 版本的配置里有这些设备：

```ini
[device "uvc-camera-0"]
  driver = "usb-host"
  hostbus = "1"
  hostport = "4.1"

[device "uvc-camera-1"]
  driver = "usb-host"
  hostbus = "1"
  hostport = "4.2"

[device "cdc-acm-0"]
  driver = "usb-host"
  hostbus = "1"
  hostport = "3.1"

[device "cdc-acm-1"]
  driver = "usb-host"
  hostbus = "1"
  hostport = "3.2"
```

这些是 Linux host 上的真实 USB passthrough 配置。Mac 原生 QEMU 阶段先不要依赖它们。

Mac 端第一版建议保留：

```text
machine virt
memory / smp
virtio-blk root disk
virtio-net
qemu-xhci
usb-kbd
```

先删除或注释：

```text
uvc-camera-0
uvc-camera-1
cdc-acm-0
cdc-acm-1
```

这样 Mac 端先验证：

```text
kernel 能启动
rootfs 能挂载
串口 console 能登录
virtio block/net 正常
QEMU 自动生成的 virt DTB 正常
```

真实 UVC / CDC ACM 透传仍然回到 Ubuntu VM 里验证。

## Mac 端第一阶段验收

启动后，在 guest 里检查：

```bash
uname -a
cat /proc/cmdline
lsblk
mount | head
ip addr
systemctl --failed --no-pager
```

如果保留了 QEMU xHCI / usb-kbd，也可以检查：

```bash
lsusb -t
dmesg | grep -Ei 'xhci|usb|virtio|vda|ttyAMA'
```

第一阶段不要求看到：

```text
/dev/video0
/dev/video2
/dev/ttyACM0
/dev/ttyACM1
```

因为 Mac 原生 QEMU 第一阶段不做真实 USB passthrough。

## 和路线 B 的关系

Mac 原生 QEMU 的价值是后续路线 B：

```text
Pi5 + RP1 USB 子集
```

当前 Linux VM 实验已经证明：

```text
UVC / CDC ACM / LeRobot 用户态这一层能跑。
```

Mac 端后续要解决的是：

```text
怎么让新 kernel 从更接近真 Pi 5 的设备树和板级链路，
初始化到同样的 USB / UVC / CDC ACM 结果。
```

建议顺序：

```text
B1. Pi5-lite machine
    CPU / RAM / UART / timer / interrupt / rootfs 启动。

B2. BCM2712 PCIe 骨架
    设备树里出现 /axi/pcie@1000120000。

B3. RP1-ish endpoint
    PCIe 后面出现 RP1，先只提供必要 MMIO window / IRQ / 子设备空间。

B4. RP1 USB host 子集
    在 RP1 下暴露 usb@200000 / usb@300000。
    后端先复用 QEMU xHCI 或类似机制。
```

不要一开始模拟完整 BCM2712 / RP1。先围绕 LeRobot 必需的 USB 子集推进。

## 给 Mac 端 Codex 的一句话任务

```text
先把 Linux VM 中已验证过的 rootfs + Image 在 Mac 原生 QEMU virt 上启动起来；
不要先处理真实 USB 透传；
启动成功后，再基于 Pi5 + RP1 USB 子集路线，逐步把 virt DTB / QEMU machine 改得更接近 Raspberry Pi 5。
```
