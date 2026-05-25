# 02 Raspberry Pi 5 启动链

## 当前真机启动文件

```text
/boot/firmware/kernel_2712.img                    # Pi 5 实际启动的 Linux kernel 镜像；新 kernel 最终要替换/生成这个角色的文件。
/boot/firmware/bcm2712-rpi-5-b.dtb                # Pi 5 的设备树二进制；告诉 kernel 板子上有哪些硬件、地址、中断、总线和外设。
/boot/firmware/initramfs_2712                     # 早期临时根文件系统；kernel 早期可用它加载脚本/模块，再切到真正 rootfs。
/boot/firmware/config.txt                         # Raspberry Pi firmware 配置；决定加载哪个 kernel、dtb、overlay、initramfs。
/boot/firmware/cmdline.txt                        # 传给 kernel 的启动参数；比如 rootfs、console、rootwait、日志参数。
/lib/modules/6.12.75-v8-16k-TEST-PSELECT6+       # 当前 kernel 对应模块目录；驱动模块要和 uname -r 匹配。
```

## 启动顺序

```text
Raspberry Pi bootloader / firmware 固件
  -> 读取 /boot/firmware/config.txt 指定启动kernel的镜像
  -> 选择 /boot/firmware/kernel_2712.img
  -> 按 Pi 5 板型选择 /boot/firmware/bcm2712-rpi-5-b.dtb  设备树
  -> 读取 /boot/firmware/cmdline.txt  给 kernel 的启动参数文件。
  -> 本机加载 /boot/firmware/initramfs_2712 加载临时 rootfs / early userspace
  -> 把 /boot/firmware/kernel_2712.img、bcm2712-rpi-5-b.dtb、initramfs_2712 放进内存
  -> firmware 设置 CPU 的 PC 跳到 kernel Image 入口，同时在 x0 寄存器里传入 dtb 的物理地址
  -> kernel 从寄存器拿到 dtb 地址，按顺序解析 dtb / cmdline
  -> kernel 根据 root= 挂载 rootfs
  -> 进入用户态 init / udev
  -> 用户态按需从 /lib/modules/6.12.75-v8-16k-TEST-PSELECT6+ 加载 .ko 驱动模块
  -> 模块加载后，出现 /dev/video*、/dev/ttyACM* 等设备节点
```

`initramfs_2712` 是 early userspace 镜像，也就是 kernel 挂载真正 rootfs 前先用的一小段临时根文件系统。当前真机里它已经被加载，证据是运行时设备树 `/proc/device-tree/chosen/` 里有 `linux,initrd-start` 和 `linux,initrd-end`。

`modules` 不在 `/boot/firmware/` 里，而是在真正 rootfs 的 `/lib/modules/<kernel-release>/` 里。当前真机对应：

```text
/lib/modules/6.12.75-v8-16k-TEST-PSELECT6+
```

所以启动链可以理解成两段：

```text
/boot/firmware/                  # firmware 阶段读取：kernel image、dtb、cmdline、initramfs
/lib/modules/<kernel-release>/   # kernel 挂载 rootfs 后，用户态/modprobe/udev 按需加载驱动模块
```

注意：不是所有驱动都从 `/lib/modules` 加载。有些驱动已经 built-in 到 `kernel_2712.img` 里，比如当前真机的 `xhci-hcd`；而 `uvcvideo`、`cdc_acm` 这类如果是 `.ko.xz` 模块，就必须和当前 `uname -r` 对应的 modules 目录匹配。

## `cmdline.txt` 逐项解释

当前真机快照里的 `cmdline.txt` 是一整行 kernel 启动参数：

```text
console=serial0,115200 console=tty1 root=PARTUUID=2428fd84-02 rootfstype=ext4 fsck.repair=yes rootwait quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=US cfg80211.ieee80211_regdom=GB cfg80211.ieee80211_regdom=GB
```

它不是 shell 脚本，也不是配置块；kernel 启动时会把这一整行按空格拆成多个参数。

| 参数                              | 含义                                                 | 对启动链的作用                                                 |
| --------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------- |
| `console=serial0,115200`          | 把串口 `serial0` 作为一个控制台，波特率 `115200`。   | kernel 启动日志可以从串口输出，方便早期调试。                  |
| `console=tty1`                    | 把本地图形/显示终端 `tty1` 也作为控制台。            | 接显示器和键盘时，可以在屏幕上看到登录/日志。                  |
| `root=PARTUUID=2428fd84-02`       | 指定真正 rootfs 所在分区。                           | kernel 挂载这个分区作为 `/` 根文件系统。                       |
| `rootfstype=ext4`                 | rootfs 的文件系统类型是 `ext4`。                     | kernel 用 ext4 驱动去挂载 rootfs。                             |
| `fsck.repair=yes`                 | 启动时如果检查文件系统发现可修复问题，允许自动修复。 | 偏用户态/发行版启动策略，不是硬件初始化核心。                  |
| `rootwait`                        | 等 root 设备出现。                                   | SD 卡/USB/NVMe 等块设备枚举可能较慢，没有它可能找不到 rootfs。 |
| `quiet`                           | 减少 kernel 启动日志输出。                           | 让启动界面更安静；调试新 kernel 时通常可以去掉。               |
| `splash`                          | 显示启动画面。                                       | 桌面发行版体验相关；新 kernel 调试时不关键。                   |
| `plymouth.ignore-serial-consoles` | Plymouth 启动画面不要占用串口控制台。                | 避免图形启动程序干扰串口调试。                                 |
| `cfg80211.ieee80211_regdom=...`   | 设置 Wi-Fi 国家/地区法规域。                         | 影响无线频段/功率；和 LeRobot USB 摄像头/串口链路无直接关系。  |

对新 kernel 来说，最核心的是这几个：

```text
console=...
root=...
rootfstype=...
rootwait
```

因为它们分别决定：日志输出到哪、真正的 rootfs 在哪、用什么文件系统挂载、是否等待存储设备就绪。

## 对新 kernel 的要求

| 项           | 新 kernel 要做什么                                           |
| ------------ | ------------------------------------------------------------ |
| kernel image | 产出 Pi firmware/QEMU 能加载的 ARM64 kernel image。          |
| dtb          | 接收并解析 `bcm2712-rpi-5-b.dtb` 或 QEMU 提供的 dtb。        |
| cmdline      | 读取 `console=`、`root=`、`rootwait` 等启动参数。            |
| rootfs       | 能挂载真实 rootfs，运行用户态程序。                          |
| modules      | 如果驱动做成模块，`/lib/modules/<kernel-release>` 必须匹配。 |

## 当前真机核对

| 对比项         | 真机结果                                                                 | 结论                               |
| -------------- | ------------------------------------------------------------------------ | ---------------------------------- |
| 板卡型号       | `Raspberry Pi 5 Model B Rev 1.1`                                         | 与 Pi 5/BCM2712 目标一致。         |
| boot dtb       | `/boot/firmware/bcm2712-rpi-5-b.dtb` 存在                                | 与 Pi 5 启动链一致。               |
| chosen console | `stdout-path = serial10:115200n8`，实际 cmdline 展开为 `ttyAMA10,115200` | firmware 会补全启动参数。          |
| 当前 kernel    | `6.12.75-v8-16k-TEST-PSELECT6+`                                          | modules 目录必须匹配这个 release。 |
