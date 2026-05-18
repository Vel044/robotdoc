# 从新内核出发的 LeRobot 工具链摸底

本文目的：不是先列 Python 包，而是从“我要写一个新的 kernel”出发，反推这个 kernel 如果要承载当前 SO101 + ACT LeRobot 机器人项目，必须满足哪些构建、启动、Linux ABI、驱动、设备节点和用户态运行条件。

核心问题是：新的 kernel 最终不只是要启动，还要能让 `python -m lerobot.record` 这条真实机器人负载跑起来。因此本文从新 kernel 往上看：

- kernel 怎么构建：`make/gmake`、GCC/binutils、LLVM/clang、Rust、bindgen、dtc、modules 工具。
- kernel 怎么启动：Raspberry Pi 5 firmware / QEMU 加载 kernel image、dtb、initramfs、cmdline。
- kernel 要提供什么接口：ARM64 ELF、glibc/Python 依赖的 syscall、VFS、`/dev`、`/proc`、`/sys`、`mmap`、`futex`、`ioctl`、`poll/select`、TTY、V4L2、USB。
- kernel 要驱动什么硬件：UVC 摄像头、USB ACM 舵机串口、USB xHCI/RP1、设备树描述的 Pi 5 板级硬件。
- LeRobot 这条命令实际压到哪些用户态库：Python、conda、torch、OpenCV、scservo/pyserial、datasets/PyAV 等。

口径说明：

- `python -m lerobot.record ...` 是验证新 kernel 的真实用户态负载，不是本文唯一目标。
- `policy=ACT`，因为命令带 `--policy.path=/home/vel/so101-bottle/last/pretrained_model`。
- 主循环动作来源是 `predict_action() -> ACTPolicy.select_action()`；`teleop` 仍然会构造和 `connect()`，但在本命令的主采集循环里不会走 `teleop.get_action()` 生成动作。
- “所有 import”限定为本命令链路相关文件：入口、config/parser、ACT policy、processor、SO101 robot/teleop、OpenCV camera、Feetech motor、dataset、utils、`scservo_sdk`、`serial`，并补上 `record.py` 直接触发的 package re-export 文件。
- runtime trace 只说明这次非硬件 trace 实际加载了什么；没有出现在 trace 里不等于永远没用到。

## 0. 从新内核反推工具链地图

如果新 kernel 要跑当前机器人项目，它不只要“能启动”，还要能承载一整套 Linux 用户态和硬件 I/O。当前工具链可以按下面这张表理解：

| 从新 kernel 往上看 | 当前项目对应的东西 | 新 kernel 必须满足什么 |
| --- | --- | --- |
| 启动链 | `kernel_2712.img`、`bcm2712-rpi-5-b.dtb`、`initramfs_2712`、`config.txt`、`cmdline.txt` | 能被 Pi firmware 或 QEMU 加载，能接收 dtb/cmdline，能初始化 arm64 基础环境。 |
| 构建链 | `make/gmake`、`gcc/binutils` 或 `clang/LLVM`、Rust 的 `rustc/bindgen/rust-src`、`dtc` | 能从源码产出 kernel image、dtb、modules；Rust kernel 还要通过 `rustavailable`。 |
| Linux 用户态 ABI | `glibc`、Python、conda、PyTorch/OpenCV native `.so` | 至少兼容这些程序依赖的 Linux syscall、ELF、mmap、futex、thread、file、socket、ioctl、select/poll 等接口。 |
| 设备文件模型 | `/dev/video*`、`/dev/ttyACM*`、`/proc`、`/sys`、`/lib/modules` | 用户态靠这些路径发现和打开设备；新 kernel 要么实现，要么在 QEMU 第一阶段 mock。 |
| 摄像头驱动面 | OpenCV -> V4L2 -> UVC -> USB/xHCI/RP1 | 要支持 V4L2 ioctl、buffer queue、UVC timing；否则 LeRobot camera 只能 mock。 |
| 舵机串口驱动面 | scservo_sdk -> pyserial -> TTY -> `cdc_acm` -> USB | 要支持 `open/read/write/ioctl/select/termios` 和 ACM tty；否则 motors bus 只能 mock。 |
| LeRobot 验证负载 | ACT `record.py` 命令 | 用来验证新 kernel 的用户态 ABI、设备节点、driver 行为和实时性。 |

### 0.1 构建目标和输出物

| 目标 | 输出物 | 说明 |
| --- | --- | --- |
| ARM64 kernel image | `linux/arch/arm64/boot/Image`，树莓派启动分区里对应 `/boot/firmware/kernel_2712.img` | Pi 5 用 BCM2712，最终启动的是 firmware 加载的 kernel image。 |
| Device Tree | `linux/arch/arm64/boot/dts/broadcom/bcm2712-rpi-5-b.dtb` | 描述 Pi 5/RP1/USB/摄像头等硬件拓扑，启动时和 kernel 一起交给内核。 |
| Kernel modules | `/lib/modules/<kernel-release>/` | `uvcvideo`、`cdc_acm`、`videobuf2` 等驱动如果编成模块，就从这里加载。 |
| initramfs | `/boot/firmware/initramfs_2712` | 早期用户态，当前真机启动链里存在。 |
| boot config | `/boot/firmware/config.txt`、`/boot/firmware/cmdline.txt` | Raspberry Pi firmware 读取这些文件决定加载哪个 kernel、dtb、initramfs 和 kernel cmdline。 |

当前真机确认的启动文件：

```text
/boot/firmware/kernel_2712.img                    # Pi 5 实际启动的 Linux kernel 镜像；重写 kernel 最终要替换/生成这个角色的文件。
/boot/firmware/bcm2712-rpi-5-b.dtb                # Pi 5 的设备树；告诉 kernel 板子上有哪些硬件、地址、中断、总线和外设。
/boot/firmware/initramfs_2712                     # 早期临时根文件系统；kernel 刚启动时先用它加载脚本/模块，再切到真正 rootfs。
/boot/firmware/config.txt                         # Raspberry Pi firmware 配置；决定加载哪个 kernel、哪个 dtb、哪些 overlay、是否加载 initramfs。
/boot/firmware/cmdline.txt                        # 传给 kernel 的启动参数；比如 rootfs 位置、console 串口、rootwait、日志等级。
/lib/modules/6.12.75-v8-16k-TEST-PSELECT6+       # 当前 kernel 对应的驱动模块目录；uvcvideo、cdc_acm 等 .ko 模块要和 kernel 版本匹配。
```

### 0.2 BCM2712 和设备树链路

`BCM2712` 是 Raspberry Pi 5 的主 SoC。对新 kernel 来说，设备树不是附属文件，而是启动后认识硬件的入口：firmware/QEMU 把 dtb 传给 kernel，kernel 解析 dtb 后才知道内存、CPU、中断、timer、PCIe、RP1、USB host 等硬件在哪里。

Pi 5 的设备树入口和 include 关系：

```text
linux/arch/arm64/boot/dts/broadcom/bcm2712-rpi-5-b.dts   # Raspberry Pi 5 B 板级入口；root compatible、memory、RP1 使能、USB 使能在这里。
  -> linux/arch/arm64/boot/dts/broadcom/bcm2712-ds.dtsi   # Pi 5 共享补充；PMU、thermal、USB phy、系统 timer 等。
     -> linux/arch/arm64/boot/dts/broadcom/bcm2712.dtsi   # BCM2712 SoC 本体；CPU、PSCI、GIC、timer、soc/axi、PCIe 控制器。
  -> linux/arch/arm64/boot/dts/broadcom/rp1.dtsi          # RP1 I/O 芯片；GPIO、UART、I2C、SPI、DMA、CSI、USB host 等。
  -> linux/arch/arm64/boot/dts/broadcom/bcm2712-rpi.dtsi  # Raspberry Pi 公共板级配置；chosen、aliases、firmware、RP1 细节。
```

对新 kernel 最关键的设备树节点：

| 节点 / 属性 | 来自文件 | 中文含义 | 新 kernel 要用它干什么 |
| --- | --- | --- | --- |
| `/ compatible = "raspberrypi,5-model-b", "brcm,bcm2712"` | `bcm2712-rpi-5-b.dts` | 说明这是 Raspberry Pi 5 B，SoC 是 BCM2712。 | 选择 Pi 5/BCM2712 平台初始化路径。 |
| `memory@0` | `bcm2712-rpi-5-b.dts` | 物理内存描述，实际会被 bootloader 填充。 | 建立物理内存管理，避开不可用区域。 |
| `cpus` / `cpu@0..3` | `bcm2712.dtsi` | 4 个 `arm,cortex-a76` CPU，`enable-method = "psci"`。 | 初始化主核，后续通过 PSCI 启动其他 CPU。 |
| `psci` | `bcm2712.dtsi` | ARM 固件调用接口，`method = "smc"`。 | 用 SMC 调用 firmware/EL3，做 CPU on/off、reset 等操作。 |
| `reserved-memory` | `bcm2712.dtsi` / `bcm2712-ds.dtsi` | 保留内存，比如 ATF、CMA、bootloader 配置。 | 内存分配器不能覆盖这些区域。 |
| `chosen` | `bcm2712-rpi.dtsi` | kernel 启动参数和默认 console，`stdout-path = "serial10:115200n8"`。 | 读取 bootargs、console、initramfs 相关信息。 |
| `interrupt-parent = <&gicv2>` / `gicv2` | `bcm2712.dtsi` | 主中断控制器是 ARM GIC-400。 | 建立 IRQ 路由；USB、PCIe、timer、UART 都依赖它。 |
| `timer` | `bcm2712.dtsi` | ARMv8 generic timer。 | 提供调度 tick、高精度时间、sleep、timeout。 |
| `soc@107c000000` / `axi` | `bcm2712.dtsi` | BCM2712 片上总线和地址映射。 | 把外设 `reg` 物理地址映射到内核虚拟地址。 |
| `pcie2` | `bcm2712.dtsi` / `bcm2712-rpi-5-b.dts` | BCM2712 PCIe 控制器，Pi 5 板级里作为 `rp1_target` 使能。 | RP1 通过 PCIe 挂到 BCM2712；没有这层就没有 RP1 外设。 |
| `rp1` | `rp1.dtsi` / `bcm2712-rpi-5-b.dts` | Raspberry Pi 5 的 I/O 芯片。 | GPIO、UART、I2C、SPI、DMA、USB host 等大量外设都在 RP1 后面。 |
| `rp1_usb0` / `rp1_usb1` | `rp1.dtsi` / `bcm2712-rpi-5-b.dts` | RP1 上的 DWC3 USB host 控制器，板级文件里设为 `okay`。 | LeRobot 的 USB 摄像头和 USB ACM 舵机串口最终都从这里枚举。 |
| `aliases usb0/usb1` | `bcm2712-rpi.dtsi` | 给 RP1 USB host 起稳定别名。 | 帮助 Linux/用户态用稳定名字理解设备顺序。 |

重要边界：设备树描述的是板级硬件和总线，不直接描述这次插上的 USB 摄像头和舵机控制板。

```text
设备树静态描述：
BCM2712 -> PCIe pcie2 -> RP1 -> DWC3 USB host

USB 运行时枚举：
DWC3 USB host -> USB core -> UVC camera -> /dev/video0, /dev/video2
DWC3 USB host -> USB core -> CDC ACM servo board -> /dev/ttyACM0, /dev/ttyACM1
```

所以新 kernel 分阶段看：

| 阶段 | 必须搞清楚的设备树部分 | 目的 |
| --- | --- | --- |
| 最小启动 | `memory@0`、`cpus`、`psci`、`chosen`、`timer`、`gicv2`、console UART | kernel 能进 C/Rust 主体、打印日志、管理内存、处理中断和时间。 |
| Pi 5 板级启动 | `reserved-memory`、`soc`、`axi`、firmware、mailbox、`pcie2`、`rp1` | kernel 能按 Pi 5 的真实硬件布局初始化，不踩 firmware/ATF 保留区。 |
| LeRobot 真硬件 | `rp1_usb0`、`rp1_usb1`、RP1 interrupt、DMA、clock、USB power/phy | USB host 能工作，摄像头和舵机板才能被枚举成 `/dev/video*` 和 `/dev/ttyACM*`。 |

SSH 到当前树莓派真机核对结果：

| 对比项 | 文档判断 | 树莓派真机结果 | 结论 |
| --- | --- | --- | --- |
| 板卡型号 | Raspberry Pi 5 / BCM2712 | `/proc/device-tree/model = Raspberry Pi 5 Model B Rev 1.1` | 一致。 |
| root compatible | `"raspberrypi,5-model-b", "brcm,bcm2712"` | `/proc/device-tree/compatible` 正是这两个字符串 | 一致。 |
| boot dtb | `/boot/firmware/bcm2712-rpi-5-b.dtb` | 文件存在，真机正在使用 Pi 5 对应 dtb | 一致。 |
| console / chosen | `stdout-path = serial10:115200n8` | `/proc/device-tree/chosen/stdout-path = serial10:115200n8`；实际 cmdline 里 console 被展开为 `ttyAMA10,115200` | 一致，firmware 会把启动参数补全。 |
| USB aliases | `usb0/usb1 -> RP1 USB host` | `usb0 -> /axi/pcie@1000120000/rp1/usb@200000`，`usb1 -> /axi/pcie@1000120000/rp1/usb@300000` | 一致。 |
| PCIe 到 RP1 | `pcie2` 作为 RP1 目标 | `/proc/device-tree/axi/pcie@1000120000`：`compatible = brcm,bcm2712-pcie`，`status = okay` | 一致。 |
| RP1 | RP1 是 Pi 5 I/O 芯片 | `/proc/device-tree/axi/pcie@1000120000/rp1`：`compatible = simple-bus` | 一致。 |
| RP1 USB host | `rp1_usb0/rp1_usb1` 是 DWC3 USB host | `usb@200000`、`usb@300000`：`compatible = snps,dwc3`，`status = okay` | 一致。 |
| UVC 摄像头 | USB 枚举后出现 `/dev/video0`、`/dev/video2` | 两个 `1bcf:2281` USB 2.0 Camera，driver `uvcvideo`；capture 节点是 `/dev/video0`、`/dev/video2` | 一致；`/dev/video1`、`/dev/video3` 是同两只相机的附加 video 节点。 |
| 舵机串口 | USB CDC ACM 枚举后出现 `/dev/ttyACM0`、`/dev/ttyACM1` | 两个 `1a86:55d3` QinHeng USB Single Serial，driver `cdc_acm`；udev 链接为 `/dev/so101_follower_left`、`/dev/so101_leader_left` | 一致。 |
| USB host 运行态 | 文档原写 `xhci_hcd on Raspberry Pi 5` | `lsusb -t` 显示 root hub driver 为 `xhci-hcd`，但设备树源头是 RP1 后面的 `snps,dwc3` | 需要精确表述为 `RP1 DWC3 host -> xhci-hcd root hub`。 |

### 0.3 Kernel 构建工具链分层

| 层 | 必要工具 | 干什么 |
| --- | --- | --- |
| Kbuild 基础工具 | GNU `make >= 4.0`、shell、`gcc/g++`、`binutils`、`bc`、`bison`、`flex`、`perl`、`python3` | Linux 内核自己的构建系统。注意 macOS 自带 `/usr/bin/make 3.81` 会被当前 `linux/Makefile` 拒绝，需要 GNU make 4.x。 |
| ARM64 目标编译器 | 原生 ARM64 `gcc/binutils`，或交叉 `aarch64-linux-gnu-gcc` / `aarch64-linux-gnu-ld` / `objcopy` / `ar` | 把 C/汇编编译、汇编、链接成 ARM64 kernel。交叉编译时用 `ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-`。 |
| LLVM/Rust 推荐链 | `clang`、`ld.lld`、`llvm-ar`、`llvm-nm`、`llvm-objcopy`、`llvm-objdump`、`llvm-readelf` | Rust-for-Linux 文档里推荐 `make LLVM=1`。Rust kernel 场景下 LLVM/libclang 也被 `bindgen` 使用。 |
| Rust kernel 工具 | `rustc`、`rust-src`、`bindgen`、`libclang`、`rustfmt`、`clippy-driver` | `rustc` 编译 Rust kernel 代码；`rust-src` 提供 `core` 源码；`bindgen` 读取 C 头文件生成 Rust bindings；`rustfmt/clippy` 用于格式化和检查。 |
| Device Tree | `dtc` | 把 `.dts/.dtsi` 编译成 `.dtb/.dtbo`，Pi 5 的 `bcm2712-rpi-5-b.dtb` 属于这一层。 |
| 模块/调试信息 | `kmod`、`depmod`、`modprobe`、`pahole/dwarves`、`libelf` | modules install、模块依赖生成、BTF 调试信息、objtool/modpost 等会碰到这些。 |
| 证书/压缩/打包 | `openssl`、`cpio`、`tar`、`xz`、`zstd`、`rsync` | 模块签名/证书、initramfs、压缩内核或模块、拷贝到 rootfs/boot 分区。 |
| QEMU 验证 | `qemu-system-aarch64`、`qemu-img` | 不负责构建 kernel，只负责在虚拟 ARM64 机器里启动 kernel/rootfs，验证启动链和 mock I/O。 |

### 0.4 当前源码给出的最低版本线

来自 `linux/scripts/min-tool-version.sh`：

| 工具 | 当前源码要求的最低版本 |
| --- | --- |
| `binutils` | `2.25.0` |
| `gcc` | `5.1.0`，`parisc64` 例外是 `12.0.0` |
| `llvm` / `libclang` | `13.0.1`，部分架构例外 |
| `rustc` | `1.78.0` |
| `bindgen` | `0.65.1` |

Rust 支持的检查入口是：

```bash
make LLVM=1 rustavailable
```

它会检查 `rustc`、`bindgen`、`libclang`、`rust-src` 等是否满足内核 Rust 构建条件。

### 0.5 当前实践环境里已经确认的状态

树莓派实践环境：

| 工具 | 当前状态 |
| --- | --- |
| kernel | `6.12.75-v8-16k-TEST-PSELECT6+`，`aarch64` |
| 当前运行 kernel 的构建记录 | `/proc/version` 显示由 `gcc 13.3.0` 和 `GNU ld 2.42` 构建 |
| `gcc/g++` | `12.2.0` |
| `binutils` | `2.40`，包括 `ld/as/ar/objcopy/objdump/nm/readelf/strip` |
| `make/gmake` | GNU Make `4.3` |
| `bc` | `1.07.1` |
| `openssl` | OpenSSL `3.0.16` |
| `perl` | `5.36.0` |
| `python3` | `3.11.2` |
| `dtc` | `1.6.1` |
| `git` | `2.39.5` |
| `rsync/tar/xz/zstd/cpio/pkg-config` | 已存在 |
| `kmod/depmod/modprobe` | `kmod 30`，`depmod/modprobe` 在 `/usr/sbin` 或 `/sbin` |
| `bison/flex` | 当前 PATH 未找到 |
| `pahole/dwarves` | 当前 PATH 未找到 |
| `clang/lld/llvm-*` | 当前 PATH 未找到 |
| `rustc/cargo/rustup/rustfmt/clippy-driver/bindgen` | 当前 PATH 未找到 |
| `qemu-system-aarch64/qemu-img` | 树莓派当前 PATH 未找到 |
| `/lib/modules/.../build` | 链接指向 `/home/vel/linux-rpi-6.12`，但当前 SSH 检查时目标目录不存在 |

本地 Mac 环境：

| 工具 | 当前状态 |
| --- | --- |
| `linux/` 源码 | 本地存在，包含 `rust/` 子树和 `CONFIG_RUST` 支持 |
| GNU make | `gmake 4.4.1` 可用；macOS `/usr/bin/make 3.81` 不够 |
| QEMU | `qemu-system-aarch64 10.2.1`、`qemu-img 10.2.1` |
| Device Tree Compiler | `dtc 1.7.2` |
| ARM64 交叉 GCC | `aarch64-linux-gnu-gcc 13.3.0` |
| Rust | `rustc 1.90.0`、`cargo 1.90.0`、`rustfmt 1.8.0`、`clippy 0.1.90` |
| `bindgen` | 当前 PATH 未找到；`gmake rustavailable` 失败点就是缺 `bindgen` |
| `llvm-*` / `ld.lld` | 当前 PATH 未找到 |
| `pahole` | 当前 PATH 未找到 |

### 0.6 构建命令骨架

Pi 5 对应的 defconfig 在本地源码中存在：

```text
linux/arch/arm64/configs/bcm2712_defconfig
```

GCC 交叉编译骨架：

```bash
cd linux
gmake ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- bcm2712_defconfig
gmake ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- Image modules dtbs -j<N>
```

树莓派原生编译骨架：

```bash
cd linux
make ARCH=arm64 bcm2712_defconfig
make ARCH=arm64 Image modules dtbs -j$(nproc)
```

Rust kernel 推荐先走 LLVM 检查和 LLVM 构建口径。树莓派原生构建用 `make`；本地 Mac 构建同一命令时把 `make` 换成 `gmake`：

```bash
cd linux
make ARCH=arm64 LLVM=1 rustavailable
make ARCH=arm64 LLVM=1 bcm2712_defconfig
make ARCH=arm64 LLVM=1 Image modules dtbs -j<N>
```

如果要启用 Rust kernel 代码，还需要在 `.config` 里有：

```text
CONFIG_RUST=y
```

并确保 `rustavailable` 通过。

### 0.7 新 kernel 要跑 LeRobot 的硬性接口清单

从新 kernel 角度看，下面这些不是“Python 包清单”，而是用户态会压到 kernel 的接口面：

| 接口面 | 当前 LeRobot 触发来源 | 新 kernel 需要提供什么 |
| --- | --- | --- |
| 进程和 ELF 加载 | `python3`、conda 环境里的 native `.so`、`torch`、`cv2`、`av` | 能加载 ARM64 ELF，可运行动态链接程序，支持 `execve`、`mmap`、`mprotect`、`brk`、TLS、signals。 |
| 文件系统和路径 | conda 包、模型目录、dataset 目录、`/dev`、`/proc`、`/sys` | VFS、目录遍历、权限、`stat/open/read/write/close`、挂载 rootfs、暴露 proc/sysfs/devtmpfs 或等价机制。 |
| 线程和同步 | Python runtime、PyTorch、OpenCV、数据写入线程 | `clone`/线程、`futex`、信号、定时器、调度、CPU-only 计算路径。 |
| 时间和等待 | `record_loop()`、`busy_wait()`、摄像头帧率、串口读写超时 | `clock_gettime`、`nanosleep`、高精度时间源、poll/select 超时语义。 |
| 摄像头 I/O | `OpenCVCamera -> cv2.VideoCapture -> /dev/video0,/dev/video2` | V4L2 设备节点、`ioctl`、buffer queue、`mmap`/read buffer、UVC/USB 数据流。 |
| 舵机串口 I/O | `FeetechMotorsBus -> scservo_sdk -> pyserial -> /dev/ttyACM0,/dev/ttyACM1` | TTY/termios、`cdc_acm` 或等价串口设备、`read/write/ioctl/select`。 |
| USB 主机链路 | 摄像头和舵机都挂 USB | xHCI/USB host、枚举、interrupt/bulk/isochronous transfer、udev 可见设备信息。 |
| 模块和驱动加载 | `uvcvideo`、`cdc_acm`、`videobuf2_*` 可能是模块 | kernel release 与 `/lib/modules/<uname -r>` 匹配，`modprobe`/udev 能找到依赖。 |

## 1. 真实运行命令

```bash
python -m lerobot.record \
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=R12254705 \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=R07254705 \
    --robot.disable_torque_on_disconnect=true \
    --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 360, 'fps': 30}}" \
    --dataset.single_task="Put the bottle into the black basket." \
    --policy.path=/home/vel/so101-bottle/last/pretrained_model \
    --dataset.repo_id=${HF_USER}/eval_so101_bottle --dataset.push_to_hub=false \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=1 \
    --resume=true
```

`${HF_USER}` 在 shell 阶段展开；进入 Python 后，`dataset.repo_id` 已经是实际 repo id。

## 2. Policy 目录和 ACT 配置

树莓派上只读确认的 policy 目录：`/home/vel/so101-bottle/last/pretrained_model`。

```text
/home/vel/so101-bottle/last/pretrained_model
  config.json
  model.safetensors
  policy_preprocessor.json
  policy_preprocessor_step_3_normalizer_processor.safetensors
  policy_postprocessor.json
  policy_postprocessor_step_0_unnormalizer_processor.safetensors
  train_config.json
```

`config.json` 关键字段：

| 字段                            | 值                                                                                                 |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| policy type                     | `act`                                                                                              |
| input state                     | `observation.state`, shape `[6]`                                                                   |
| input images                    | `observation.images.handeye`, `observation.images.fixed`, shape `[3, 360, 640]`                    |
| output                          | `action`, shape `[6]`                                                                              |
| config device                   | `cuda`                                                                                             |
| runtime device                  | `cpu`，因为树莓派 `torch.cuda.is_available() = False`，`PreTrainedConfig.__post_init__()` 自动切换 |
| `chunk_size` / `n_action_steps` | `100` / `100`                                                                                      |
| backbone                        | `resnet18`                                                                                         |
| pretrained backbone weights     | `ResNet18_Weights.IMAGENET1K_V1`                                                                   |
| transformer                     | `dim_model=512`, `n_heads=8`, `n_encoder_layers=4`, `n_decoder_layers=1`, `dim_feedforward=3200`   |
| VAE                             | `use_vae=true`, `latent_dim=32`, `n_vae_encoder_layers=4`, `kl_weight=10.0`                        |
| AMP                             | `use_amp=false`                                                                                    |

`policy_preprocessor.json` 里原始步骤是：

```text
rename_observations_processor
  -> to_batch_processor
  -> device_processor(device="cuda")
  -> normalizer_processor(stats=safetensors)
```

但是 `record.py` 调用 `make_pre_post_processors(..., preprocessor_overrides={"device_processor": {"device": cfg.policy.device}})`，所以运行时会覆盖成 `cpu`。

`policy_postprocessor.json` 是：

```text
unnormalizer_processor(stats=safetensors)
  -> device_processor(device="cpu")
```

## 3. CLI 参数到 Config 对象

### 3.1 入口解析

```text
python -m lerobot.record
  -> lerobot/src/lerobot/record.py::main()
  -> lerobot/src/lerobot/record.py::record()
  -> @parser.wrap()
  -> lerobot/src/lerobot/configs/parser.py::wrap()
  -> draccus.parse(config_class=RecordConfig, args=sys.argv[1:])
  -> RecordConfig.__post_init__()
     -> parser.get_path_arg("policy")
     -> PreTrainedConfig.from_pretrained("/home/vel/so101-bottle/last/pretrained_model")
     -> cfg.policy.pretrained_path = "/home/vel/so101-bottle/last/pretrained_model"
```

`parser.wrap()` 会把 `--policy.path=...` 从普通 draccus 参数中过滤掉，再由 `RecordConfig.__post_init__()` 专门加载预训练 policy 配置。

### 3.2 命令参数映射

| 命令参数                                         | Config 类 / 字段                                   | 源码文件                                                                |
| ------------------------------------------------ | -------------------------------------------------- | ----------------------------------------------------------------------- |
| `--robot.type=so101_follower`                    | `SO101FollowerConfig`                              | `lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py`    |
| `--robot.port=/dev/ttyACM0`                      | `SO101FollowerConfig.port`                         | `lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py`    |
| `--robot.id=R12254705`                           | `RobotConfig.id`                                   | `lerobot/src/lerobot/robots/config.py`                                  |
| `--robot.disable_torque_on_disconnect=true`      | `SO101FollowerConfig.disable_torque_on_disconnect` | `lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py`    |
| `--robot.cameras=...`                            | `dict[str, OpenCVCameraConfig]`                    | `lerobot/src/lerobot/cameras/opencv/configuration_opencv.py`            |
| `handeye index_or_path=0`                        | OpenCV index `0` -> `/dev/video0`                  | `lerobot/src/lerobot/cameras/opencv/camera_opencv.py`                   |
| `fixed index_or_path=2`                          | OpenCV index `2` -> `/dev/video2`                  | `lerobot/src/lerobot/cameras/opencv/camera_opencv.py`                   |
| `--teleop.type=so101_leader`                     | `SO101LeaderConfig`                                | `lerobot/src/lerobot/teleoperators/so101_leader/config_so101_leader.py` |
| `--teleop.port=/dev/ttyACM1`                     | `SO101LeaderConfig.port`                           | `lerobot/src/lerobot/teleoperators/so101_leader/config_so101_leader.py` |
| `--teleop.id=R07254705`                          | `TeleoperatorConfig.id`                            | `lerobot/src/lerobot/teleoperators/config.py`                           |
| `--policy.path=.../pretrained_model`             | `ACTConfig.pretrained_path`                        | `lerobot/src/lerobot/configs/policies.py`                               |
| `--dataset.repo_id=${HF_USER}/eval_so101_bottle` | `DatasetRecordConfig.repo_id`                      | `lerobot/src/lerobot/record.py`                                         |
| `--dataset.push_to_hub=false`                    | `DatasetRecordConfig.push_to_hub`                  | `lerobot/src/lerobot/record.py`                                         |
| `--dataset.num_episodes=1`                       | `DatasetRecordConfig.num_episodes`                 | `lerobot/src/lerobot/record.py`                                         |
| `--dataset.episode_time_s=60`                    | `DatasetRecordConfig.episode_time_s`               | `lerobot/src/lerobot/record.py`                                         |
| `--dataset.reset_time_s=1`                       | `DatasetRecordConfig.reset_time_s`                 | `lerobot/src/lerobot/record.py`                                         |
| `--dataset.single_task=...`                      | `DatasetRecordConfig.single_task`                  | `lerobot/src/lerobot/record.py`                                         |
| `--resume=true`                                  | `RecordConfig.resume`                              | `lerobot/src/lerobot/record.py`                                         |

## 4. ACT 主调用链

```text
python -m lerobot.record
  -> record.py::main()
  -> parser.wrap() / draccus.parse()
  -> RecordConfig(policy.path=...)
  -> PreTrainedConfig.from_pretrained(...)
  -> record(cfg)
     -> make_robot_from_config(SO101FollowerConfig)
        -> SO101Follower(...)
        -> FeetechMotorsBus(port="/dev/ttyACM0")
        -> make_cameras_from_configs(...)
           -> OpenCVCamera(index_or_path=0)
           -> OpenCVCamera(index_or_path=2)
     -> make_teleoperator_from_config(SO101LeaderConfig)
        -> SO101Leader(...)
        -> FeetechMotorsBus(port="/dev/ttyACM1")
     -> make_default_processors()
     -> LeRobotDataset(repo_id=${HF_USER}/eval_so101_bottle, resume=true)
     -> make_policy(ACTConfig)
        -> get_policy_class("act")
        -> ACTPolicy.from_pretrained(...)
        -> safetensors load model.safetensors
        -> policy.to("cpu")
     -> make_pre_post_processors(ACTConfig)
        -> policy_preprocessor.json / policy_postprocessor.json
        -> ACT processor override device_processor to cpu
     -> robot.connect()
        -> follower bus connect / configure
        -> camera connect / configure 640x360@30fps
     -> teleop.connect()
        -> leader bus connect / configure
     -> VideoEncodingManager(dataset)
     -> record_loop()
        -> robot.get_observation()
        -> robot_observation_processor(obs)
        -> build_dataset_frame(..., prefix="observation")
        -> predict_action()
           -> numpy image/state -> torch.Tensor
           -> preprocessor(...)
           -> ACTPolicy.select_action(...)
              -> ACTPolicy.predict_action_chunk(...)
              -> ACT.forward(...)
              -> ResNet18 backbone + Transformer encoder/decoder
              -> action chunk queue
           -> postprocessor(...)
        -> robot_action_processor(...)
        -> robot.send_action(...)
        -> build_dataset_frame(..., prefix="action")
        -> dataset.add_frame(...)
        -> busy_wait(1 / fps - dt_s)
     -> dataset.save_episode()
     -> dataset.push_to_hub(...) is skipped because push_to_hub=false
```

主路径判断点在 `record_loop()`：

```text
if policy is not None and preprocessor is not None and postprocessor is not None:
    action_values = predict_action(...)
elif policy is None and isinstance(teleop, Teleoperator):
    act = teleop.get_action()
```

本命令 `policy is not None`，所以主采集循环不走 `teleop.get_action()`。

## 5. 硬件到 kernel driver 链路

### 5.1 OpenCV / UVC camera

```text
SO101Follower.get_observation()
  -> for each camera: cam.async_read(timeout_ms=3000)
  -> OpenCVCamera.async_read()
  -> background thread / OpenCVCamera.read()
  -> cv2.VideoCapture.read()
  -> /dev/video0, /dev/video2
  -> uvcvideo
  -> V4L2 / videobuf2 / media controller
  -> USB host: RP1 DWC3 host -> xhci-hcd root hub on Raspberry Pi 5
```

实际设备：

| LeRobot camera | OpenCV index | 设备节点      | udev / driver            | 配置            |
| -------------- | ------------ | ------------- | ------------------------ | --------------- |
| `handeye`      | `0`          | `/dev/video0` | `ID_USB_DRIVER=uvcvideo`，USB id `1bcf:2281` | `640x360@30fps` |
| `fixed`        | `2`          | `/dev/video2` | `ID_USB_DRIVER=uvcvideo`，USB id `1bcf:2281` | `640x360@30fps` |

相关 kernel 模块/驱动：`uvcvideo`、`uvc`、`videobuf2_vmalloc`、`videobuf2_v4l2`、`videobuf2_common`、`videodev`、`mc`；USB host 侧真机表现为 RP1 `snps,dwc3` 节点驱动出 `xhci-hcd` root hub。

相关 kernel 源码文件：

```text
linux/drivers/usb/dwc3/core.c
linux/drivers/usb/dwc3/host.c
linux/drivers/usb/host/xhci.c
linux/drivers/usb/host/xhci-plat.c
linux/drivers/usb/host/xhci-ring.c
linux/drivers/usb/host/xhci-mem.c
linux/drivers/media/usb/uvc/uvc_driver.c
linux/drivers/media/usb/uvc/uvc_v4l2.c
linux/drivers/media/usb/uvc/uvc_video.c
linux/drivers/media/usb/uvc/uvc_queue.c
linux/drivers/media/usb/uvc/uvcvideo.h
linux/drivers/media/v4l2-core/v4l2-dev.c
linux/drivers/media/v4l2-core/v4l2-ioctl.c
linux/drivers/media/v4l2-core/v4l2-device.c
linux/drivers/media/common/videobuf2/videobuf2-core.c
linux/drivers/media/common/videobuf2/videobuf2-v4l2.c
linux/drivers/media/common/videobuf2/videobuf2-vmalloc.c
```

### 5.2 Feetech / SCServo / CDC ACM serial

```text
SO101Follower / SO101Leader
  -> FeetechMotorsBus
  -> MotorsBus.connect()
     -> scservo_sdk.PortHandler.openPort()
     -> serial.Serial(port="/dev/ttyACM0" or "/dev/ttyACM1", baudrate=1000000, timeout=0)
     -> serialposix.Serial.open()
        -> os.open(...)
        -> termios.tcgetattr / termios.tcsetattr
        -> fcntl / ioctl
  -> MotorsBus.sync_read("Present_Position")
     -> scservo_sdk.GroupSyncRead.txRxPacket()
     -> protocol_packet_handler.txPacket/rxPacket
     -> port_handler.writePort/readPort
     -> serialposix.write/read
     -> os.write / select / os.read
  -> MotorsBus.sync_write("Goal_Position")
     -> scservo_sdk.GroupSyncWrite.txPacket()
     -> protocol_packet_handler.syncWriteTxOnly
     -> port_handler.writePort
     -> serialposix.write
     -> os.write
  -> /dev/ttyACM0, /dev/ttyACM1
  -> cdc_acm
```

实际设备：

| 角色     | 命令 id     | LeRobot port   | udev symlink                          | driver    | USB id                                |
| -------- | ----------- | -------------- | ------------------------------------- | --------- | ------------------------------------- |
| follower | `R12254705` | `/dev/ttyACM0` | `/dev/so101_follower_left -> ttyACM0` | `cdc_acm` | `1a86:55d3` QinHeng USB Single Serial |
| leader   | `R07254705` | `/dev/ttyACM1` | `/dev/so101_leader_left -> ttyACM1`   | `cdc_acm` | `1a86:55d3` QinHeng USB Single Serial |

相关 kernel 模块/驱动：`cdc_acm`、`xhci_hcd`；树莓派 5/RP1 侧还加载了 `rp1_pio`、`rp1_fw`、`rp1_mailbox`、`pisp_be`。真机设备树显示 USB host 源头是 RP1 的 `snps,dwc3` 节点，运行态 root hub driver 是 `xhci-hcd`。

相关 kernel 源码文件：

```text
linux/drivers/usb/class/cdc-acm.c
linux/drivers/usb/dwc3/core.c
linux/drivers/usb/dwc3/host.c
linux/drivers/usb/host/xhci.c
linux/drivers/usb/host/xhci-plat.c
linux/drivers/usb/host/xhci-ring.c
linux/drivers/usb/host/xhci-mem.c
linux/drivers/mfd/rp1.c
linux/drivers/firmware/rp1-fw.c
linux/drivers/mailbox/rp1-mailbox.c
linux/drivers/misc/rp1-pio.c
linux/drivers/media/platform/raspberrypi/pisp_be/pisp_be.c
linux/arch/arm64/boot/dts/broadcom/bcm2712-rpi-5-b.dts
linux/arch/arm64/boot/dts/broadcom/bcm2712-rpi.dtsi
linux/arch/arm64/boot/dts/broadcom/bcm2712.dtsi
linux/arch/arm64/boot/dts/broadcom/rp1.dtsi
```

## 6. QEMU 里需要模拟或替代的东西

| 真实链路                                                                 | QEMU 第一阶段处理                                                                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| ARM64 Linux 启动、Python 用户态、LeRobot import、ACT config/weights 加载 | 可以在 QEMU 内跑                                                                                        |
| `torch==2.7.1+cpu` ACT 推理                                              | 可以跑 CPU 路径，但性能要单独量                                                                         |
| `/dev/video0`、`/dev/video2` UVC 摄像头                                  | 用 mock 图像、预录帧或 V4L2 loopback 等替代                                                             |
| `/dev/ttyACM0` follower 舵机串口                                         | 用伪串口/mock `MotorsBus` 替代                                                                          |
| `/dev/ttyACM1` leader 舵机串口                                           | 本命令主动作不靠 `teleop.get_action()`，但 `teleop.connect()` 仍会碰串口；QEMU 中要 mock 或跳过 connect |
| `uvcvideo` / `cdc_acm` / RP1 / xHCI 真实时序                             | 只能在树莓派真机验证                                                                                    |
| `dataset.add_frame()`、视频编码、`save_episode()`                        | 可以在 QEMU 内用 mock frame 测                                                                          |

## 7. 树莓派实际版本汇总

### 7.1 Python / package 版本

| 项                     | 版本 / 事实                               | 中文说明                                                                                                              |
| ---------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Python                 | `3.10.12`                                 | Python 解释器本体，`python -m lerobot.record` 就是由它启动；QEMU 里至少要能跑同一大版本的 Python 用户态。             |
| conda                  | `25.9.1`                                  | 环境管理器，负责激活 `lerobot` 环境并提供这一组 Python 包、原生 `.so` 动态库和路径。                                  |
| pip                    | `26.1.1`                                  | Python 包安装/元数据工具，用来查询 `site-packages` 中 pip 包版本；不直接参与控制循环。                                |
| LeRobot                | metadata/import `0.3.4`                   | 机器人主项目，`record.py`、SO101、camera、motor、dataset、policy 都在这个包里。                                       |
| torch                  | metadata `2.7.1`, import `2.7.1+cpu`      | PyTorch 深度学习运行时；ACT policy 的张量、模型加载、CPU 推理都靠它。树莓派这里是 CPU-only。                          |
| torchvision            | `0.22.1`                                  | PyTorch 视觉模型/图像工具库；ACT 的 ResNet18 backbone 和部分图像处理依赖它。                                          |
| opencv-python          | `4.12.0.88`                               | OpenCV 的 Python wheel，提供 `cv2`；本命令用它打开 USB 摄像头并读取图像。                                             |
| opencv-python-headless | `4.12.0.88`                               | 无 GUI 版 OpenCV wheel；环境里同时存在，主要也是提供 OpenCV 运行库/依赖。                                             |
| cv2                    | `4.12.0`                                  | Python 里实际 import 的 OpenCV 模块名；`OpenCVCamera` 最终调用 `cv2.VideoCapture.read()`。                            |
| numpy                  | `2.2.6`                                   | 数组库；相机图像、关节状态、dataset frame 和 PyTorch tensor 转换前后的数据都经常用它表示。                            |
| pyserial               | `3.5`                                     | Python 串口库；`scservo_sdk` 最终通过它打开 `/dev/ttyACM0`、`/dev/ttyACM1` 并走 `os.read/os.write`。                  |
| feetech-servo-sdk      | metadata `1.0.0`, import `scservo_sdk` OK | Feetech/SCServo 舵机协议 SDK；负责封包、同步读写舵机寄存器，下面接 pyserial。                                         |
| datasets               | `4.1.1`                                   | HuggingFace datasets 库；LeRobotDataset 的元数据、episode、parquet/arrow 数据组织依赖它。                             |
| av                     | `15.1.0`                                  | PyAV，FFmpeg 的 Python 绑定；LeRobot 视频读写/编码相关链路会用到。                                                    |
| draccus                | metadata `0.10.0`, import `0.8.0`         | dataclass 配置/CLI 解析库；把命令行参数解析成 `RecordConfig`、`SO101FollowerConfig`、`ACTConfig`。                    |
| diffusers              | `0.35.2`                                  | 扩散模型策略相关库；ACT 主路径不用它推理，但 LeRobot policy 包会 import/注册其他策略配置。                            |
| huggingface_hub        | `0.35.3`                                  | HuggingFace Hub 客户端；本地/远程加载 policy config、模型文件、dataset meta 时会用到。                                |
| transformers           | `4.51.3`                                  | Transformer/Tokenizer 生态库；ACT 本身主要用 PyTorch Transformer，但 LeRobot 的语言/多模态策略和 processor 会依赖它。 |
| accelerate             | `1.11.0`                                  | HuggingFace 训练/设备分发辅助库；本命令推理主路径不是核心依赖，但环境和部分策略模块会加载它。                         |
| safetensors            | `0.6.2`                                   | 安全张量权重格式库；ACT 的 `model.safetensors`、normalizer/unnormalizer 统计文件都靠它加载。                          |
| tokenizers             | `0.21.4`                                  | HuggingFace 的高性能 tokenizer 库；语言条件策略/processor 相关，ACT 这条命令只带 task 文本但不靠它生成动作。          |
| pandas                 | `2.3.3`                                   | 表格数据处理库；dataset metadata、统计、parquet/episode 索引处理会用到。                                              |
| pyarrow                | `22.0.0`                                  | Arrow/Parquet 原生库；LeRobotDataset 的数据表、parquet 文件读写依赖它。                                               |
| pillow                 | `12.0.0`                                  | PIL 图像库；图片保存、读取、格式转换和部分 dataset/image writer 路径会用到。                                          |
| rerun-sdk              | `0.22.1`                                  | Rerun 可视化 SDK；只有 `display_data`/可视化路径打开时才关键，本命令默认不显示。                                      |
| wandb                  | metadata `0.21.4`                         | Weights & Biases 实验记录工具；训练日志相关，本命令 record 推理不依赖它，而且当前 import 事实是报错。                 |
| cmake                  | `4.1.2`                                   | C/C++ 构建工具；常用于编译原生扩展或依赖，不是 `record_loop()` 运行时核心库。                                         |

### 7.2 runtime trace 实际加载的 Python distribution

这次 trace 做了四步：`import lerobot.record`、加载 ACT config、`ACTPolicy.from_pretrained()`、构造 robot/teleop 但不 `connect()` 硬件。

```text
accelerate==1.11.0
aiohttp==3.13.2
aiosignal==1.4.0
async-timeout==5.0.1
av==15.1.0
certifi==2025.11.12
charset-normalizer==3.4.4
datasets==4.1.1
dill==0.3.8
draccus==0.10.0
Farama-Notifications==0.0.4
feetech-servo-sdk==1.0.0
frozenlist==1.8.0
gymnasium==0.29.1
huggingface_hub==0.35.3
ImageIO==2.37.2
lerobot==0.3.4
MarkupSafe==3.0.3
mergedeep==1.3.4
mpmath==1.3.0
multidict==6.7.0
multiprocess==0.70.16
networkx==3.4.2
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pillow==12.0.0
polars==1.35.2
propcache==0.4.1
pyarrow==22.0.0
pyserial==3.5
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
requests==2.32.5
six==1.17.0
sympy==1.14.0
torch==2.7.1
torchvision==0.22.1
tqdm==4.67.1
transformers==4.51.3
typing-inspect==0.9.0
xxhash==3.6.0
yarl==1.22.0
```

### 7.3 runtime trace 实际映射的原生动态库桶

```text
conda-lib: libGL, libGLX, libGLdispatch, libX11, libXau, libXdmcp, libbz2, libcrypto.so.3,
           libffi, libfribidi, libgcc_s, libglib-2.0, libgthread-2.0, libiconv,
           liblzma, libpcre2-8, libssl.so.3, libstdc++.so.6.0.34, libuuid,
           libxcb, libz.so.1.3.1
python-lib-dynload: _asyncio, _bz2, _ctypes, _csv, _decimal, _hashlib, _lzma, _socket,
                    _ssl, fcntl, resource, select, termios, zlib 等 43 个扩展模块
torch: torch/_C, libc10, libshm, libtorch, libtorch_cpu, libtorch_global_deps, libtorch_python
torch.libs: libarm_compute, libarm_compute_graph, libgfortran, libgomp, libopenblasp
torchvision: torchvision/_C.so, torchvision/image.so, jpeg/png/webp/z 相关库
numpy: _multiarray_umath, _umath_linalg, random 扩展, numpy OpenBLAS/gfortran
opencv: cv2.abi3.so, Qt5Core/Gui/Test/Widgets, FFmpeg, OpenBLAS, libpng, libssl/libcrypto
av: PyAV 扩展 + av.libs 内 FFmpeg/libx264/libx265/libvpx/libwebp/libopus 等
pyarrow: libarrow.so.2200, libparquet.so.2200, dataset/parquet/compute/json/csv/fs 扩展
pandas: pandas/_libs 下 42 个 C 扩展
PIL/pillow.libs: _imaging, _imagingft, freetype, harfbuzz, jpeg, png, tiff, webp 相关库
safetensors: _safetensors_rust.abi3.so
tokenizers: tokenizers.abi3.so
```

### 7.4 kernel / 系统工具版本

| 项                          | 版本 / 事实                                   |
| --------------------------- | --------------------------------------------- |
| kernel                      | `6.12.75-v8-16k-TEST-PSELECT6+`               |
| architecture                | `aarch64`                                     |
| `/proc/version` GCC         | `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0` |
| `/proc/version` ld/binutils | `GNU ld (GNU Binutils for Ubuntu) 2.42`       |
| v4l2-ctl                    | `1.22.1`                                      |
| udevadm                     | `252`                                         |

## 8. 所有 import 总表

说明：本表由 AST 从本命令链路文件抽取；`顶层 import` 在模块 import 时执行，`函数内延迟 import` 只在对应分支/函数执行时触发。缺失的 `configs/__init__.py`、`policies/act/__init__.py`、`datasets/__init__.py` 在当前源码树不存在。
#### `lerobot/src/lerobot/__init__.py`
| 时机        | 类别     | import 语句                                   |
| ----------- | -------- | --------------------------------------------- |
| 顶层 import | 标准库   | `import itertools`                            |
| 顶层 import | 本地源码 | `from lerobot.__version__ import __version__` |

#### `lerobot/src/lerobot/record.py`
| 时机        | 类别     | import 语句                                                                                                                                                                                                                                                                                                    |
| ----------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import csv`                                                                                                                                                                                                                                                                                                   |
| 顶层 import | 标准库   | `import logging`                                                                                                                                                                                                                                                                                               |
| 顶层 import | 标准库   | `import os`                                                                                                                                                                                                                                                                                                    |
| 顶层 import | 标准库   | `import resource`                                                                                                                                                                                                                                                                                              |
| 顶层 import | 标准库   | `import time`                                                                                                                                                                                                                                                                                                  |
| 顶层 import | 标准库   | `from dataclasses import asdict, dataclass, field`                                                                                                                                                                                                                                                             |
| 顶层 import | 标准库   | `from pathlib import Path`                                                                                                                                                                                                                                                                                     |
| 顶层 import | 标准库   | `from pprint import pformat`                                                                                                                                                                                                                                                                                   |
| 顶层 import | 标准库   | `from typing import Any`                                                                                                                                                                                                                                                                                       |
| 顶层 import | 本地源码 | `from lerobot.cameras import (  # noqa: F401 CameraConfig,  # noqa: F401 )`                                                                                                                                                                                                                                    |
| 顶层 import | 本地源码 | `from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig`                                                                                                                                                                                                                                   |
| 顶层 import | 本地源码 | `from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig`                                                                                                                                                                                                                          |
| 顶层 import | 本地源码 | `from lerobot.configs import parser`                                                                                                                                                                                                                                                                           |
| 顶层 import | 本地源码 | `from lerobot.configs.policies import PreTrainedConfig`                                                                                                                                                                                                                                                        |
| 顶层 import | 本地源码 | `from lerobot.datasets.image_writer import safe_stop_image_writer`                                                                                                                                                                                                                                             |
| 顶层 import | 本地源码 | `from lerobot.datasets.lerobot_dataset import LeRobotDataset`                                                                                                                                                                                                                                                  |
| 顶层 import | 本地源码 | `from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features`                                                                                                                                                                                                  |
| 顶层 import | 本地源码 | `from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts`                                                                                                                                                                                                                                |
| 顶层 import | 本地源码 | `from lerobot.datasets.video_utils import VideoEncodingManager`                                                                                                                                                                                                                                                |
| 顶层 import | 本地源码 | `from lerobot.policies.factory import make_policy, make_pre_post_processors`                                                                                                                                                                                                                                   |
| 顶层 import | 本地源码 | `from lerobot.policies.pretrained import PreTrainedPolicy`                                                                                                                                                                                                                                                     |
| 顶层 import | 本地源码 | `from lerobot.processor import ( PolicyAction,  # 策略输出的动作类型 PolicyProcessorPipeline,  # 策略预/后处理流水线类型 RobotAction,  # 机器人动作类型 RobotObservation,  # 机器人观测类型 RobotProcessorPipeline,  # 机器人处理器流水线类型 make_default_processors,  # 创建默认处理器流水线 )`              |
| 顶层 import | 本地源码 | `from lerobot.processor.rename_processor import rename_stats`                                                                                                                                                                                                                                                  |
| 顶层 import | 本地源码 | `from lerobot.robots import (  # noqa: F401 Robot,  # 机器人基类 RobotConfig,  # 机器人配置基类 # 具体机器人类型（用于类型检查） bi_so100_follower, bi_so101_follower, hope_jr, koch_follower, make_robot_from_config,  # 从配置创建机器人实例 so100_follower, so101_follower, xlerobot, )`                    |
| 顶层 import | 本地源码 | `from lerobot.teleoperators import (  # noqa: F401 Teleoperator,  # 遥操作设备基类 TeleoperatorConfig,  # 遥操作配置基类 # 具体遥操作类型 bi_so100_leader, bi_so101_leader, homunculus, koch_leader, make_teleoperator_from_config,  # 从配置创建遥操作实例 so100_leader, so101_leader, xlebi_so101_leader, )` |
| 顶层 import | 本地源码 | `from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop`                                                                                                                                                                                                                                    |
| 顶层 import | 本地源码 | `from lerobot.utils.control_utils import ( init_keyboard_listener,  # 初始化键盘监听（用于中断控制） is_headless,  # 检查是否无头模式 predict_action,  # 策略推理 sanity_check_dataset_name,  # 检查数据集名称合法性 sanity_check_dataset_robot_compatibility,  # 检查数据集与机器人兼容性 )`                  |
| 顶层 import | 本地源码 | `from lerobot.utils.robot_utils import busy_wait`                                                                                                                                                                                                                                                              |
| 顶层 import | 本地源码 | `from lerobot.utils.utils import ( get_safe_torch_device,  # 获取安全的torch设备 init_logging,  # 初始化日志 log_say,  # 语音播报（可选） )`                                                                                                                                                                   |
| 顶层 import | 本地源码 | `from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data`                                                                                                                                                                                                                                    |

#### `lerobot/src/lerobot/configs/parser.py`
| 时机        | 类别     | import 语句                                  |
| ----------- | -------- | -------------------------------------------- |
| 顶层 import | 标准库   | `import importlib`                           |
| 顶层 import | 标准库   | `import inspect`                             |
| 顶层 import | 标准库   | `import pkgutil`                             |
| 顶层 import | 标准库   | `import sys`                                 |
| 顶层 import | 标准库   | `from argparse import ArgumentError`         |
| 顶层 import | 标准库   | `from collections.abc import Sequence`       |
| 顶层 import | 标准库   | `from functools import wraps`                |
| 顶层 import | 标准库   | `from pathlib import Path`                   |
| 顶层 import | 第三方库 | `import draccus`                             |
| 顶层 import | 本地源码 | `from lerobot.utils.utils import has_method` |

#### `lerobot/src/lerobot/configs/policies.py`
| 时机        | 类别     | import 语句                                                                                             |
| ----------- | -------- | ------------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                                                                            |
| 顶层 import | 标准库   | `import builtins`                                                                                       |
| 顶层 import | 标准库   | `import json`                                                                                           |
| 顶层 import | 标准库   | `import logging`                                                                                        |
| 顶层 import | 标准库   | `import os`                                                                                             |
| 顶层 import | 标准库   | `import tempfile`                                                                                       |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                                                              |
| 顶层 import | 标准库   | `from pathlib import Path`                                                                              |
| 顶层 import | 标准库   | `from typing import TypeVar`                                                                            |
| 顶层 import | 第三方库 | `import draccus`                                                                                        |
| 顶层 import | 第三方库 | `from huggingface_hub import hf_hub_download`                                                           |
| 顶层 import | 第三方库 | `from huggingface_hub.constants import CONFIG_NAME`                                                     |
| 顶层 import | 第三方库 | `from huggingface_hub.errors import HfHubHTTPError`                                                     |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, PolicyFeature`                                          |
| 顶层 import | 本地源码 | `from lerobot.constants import ACTION, OBS_STATE`                                                       |
| 顶层 import | 本地源码 | `from lerobot.optim.optimizers import OptimizerConfig`                                                  |
| 顶层 import | 本地源码 | `from lerobot.optim.schedulers import LRSchedulerConfig`                                                |
| 顶层 import | 本地源码 | `from lerobot.utils.hub import HubMixin`                                                                |
| 顶层 import | 本地源码 | `from lerobot.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available` |

#### `lerobot/src/lerobot/configs/types.py`
| 时机        | 类别   | import 语句                         |
| ----------- | ------ | ----------------------------------- |
| 顶层 import | 标准库 | `from dataclasses import dataclass` |
| 顶层 import | 标准库 | `from enum import Enum`             |
| 顶层 import | 标准库 | `from typing import Any, Protocol`  |

#### `lerobot/src/lerobot/configs/default.py`
| 时机        | 类别     | import 语句                                                       |
| ----------- | -------- | ----------------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                        |
| 顶层 import | 本地源码 | `from lerobot import ( policies,  # noqa: F401 )`                 |
| 顶层 import | 本地源码 | `from lerobot.datasets.transforms import ImageTransformsConfig`   |
| 顶层 import | 本地源码 | `from lerobot.datasets.video_utils import get_safe_default_codec` |

#### `lerobot/src/lerobot/configs/train.py`
| 时机        | 类别     | import 语句                                                                  |
| ----------- | -------- | ---------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import builtins`                                                            |
| 顶层 import | 标准库   | `import datetime as dt`                                                      |
| 顶层 import | 标准库   | `import os`                                                                  |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                                   |
| 顶层 import | 标准库   | `from pathlib import Path`                                                   |
| 顶层 import | 第三方库 | `import draccus`                                                             |
| 顶层 import | 第三方库 | `from huggingface_hub import hf_hub_download`                                |
| 顶层 import | 第三方库 | `from huggingface_hub.errors import HfHubHTTPError`                          |
| 顶层 import | 本地源码 | `from lerobot import envs`                                                   |
| 顶层 import | 本地源码 | `from lerobot.configs import parser`                                         |
| 顶层 import | 本地源码 | `from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig` |
| 顶层 import | 本地源码 | `from lerobot.configs.policies import PreTrainedConfig`                      |
| 顶层 import | 本地源码 | `from lerobot.optim import OptimizerConfig`                                  |
| 顶层 import | 本地源码 | `from lerobot.optim.schedulers import LRSchedulerConfig`                     |
| 顶层 import | 本地源码 | `from lerobot.utils.hub import HubMixin`                                     |

#### `lerobot/src/lerobot/configs/eval.py`
| 时机        | 类别     | import 语句                                             |
| ----------- | -------- | ------------------------------------------------------- |
| 顶层 import | 标准库   | `import datetime as dt`                                 |
| 顶层 import | 标准库   | `import logging`                                        |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`              |
| 顶层 import | 标准库   | `from pathlib import Path`                              |
| 顶层 import | 本地源码 | `from lerobot import envs, policies`                    |
| 顶层 import | 本地源码 | `from lerobot.configs import parser`                    |
| 顶层 import | 本地源码 | `from lerobot.configs.default import EvalConfig`        |
| 顶层 import | 本地源码 | `from lerobot.configs.policies import PreTrainedConfig` |

#### `lerobot/src/lerobot/envs/__init__.py`
| 时机        | 类别     | import 语句                                                   |
| ----------- | -------- | ------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .configs import AlohaEnv, EnvConfig, PushtEnv, XarmEnv` |

#### `lerobot/src/lerobot/envs/configs.py`
| 时机        | 类别     | import 语句                                                                             |
| ----------- | -------- | --------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                                                            |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                                              |
| 顶层 import | 标准库   | `from typing import Any`                                                                |
| 顶层 import | 第三方库 | `import draccus`                                                                        |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, PolicyFeature`                          |
| 顶层 import | 本地源码 | `from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE` |
| 顶层 import | 本地源码 | `from lerobot.robots import RobotConfig`                                                |
| 顶层 import | 本地源码 | `from lerobot.teleoperators.config import TeleoperatorConfig`                           |

#### `lerobot/src/lerobot/envs/utils.py`
| 时机        | 类别     | import 语句                                                     |
| ----------- | -------- | --------------------------------------------------------------- |
| 顶层 import | 标准库   | `import warnings`                                               |
| 顶层 import | 标准库   | `from collections.abc import Mapping, Sequence`                 |
| 顶层 import | 标准库   | `from functools import singledispatch`                          |
| 顶层 import | 标准库   | `from typing import Any`                                        |
| 顶层 import | 第三方库 | `import einops`                                                 |
| 顶层 import | 第三方库 | `import gymnasium as gym`                                       |
| 顶层 import | 第三方库 | `import numpy as np`                                            |
| 顶层 import | 第三方库 | `import torch`                                                  |
| 顶层 import | 第三方库 | `from torch import Tensor`                                      |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, PolicyFeature`  |
| 顶层 import | 本地源码 | `from lerobot.envs.configs import EnvConfig`                    |
| 顶层 import | 本地源码 | `from lerobot.utils.utils import get_channel_first_image_shape` |

#### `lerobot/src/lerobot/policies/__init__.py`
| 时机        | 类别     | import 语句                                                                         |
| ----------- | -------- | ----------------------------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .act.configuration_act import ACTConfig as ACTConfig`                         |
| 顶层 import | 本地源码 | `from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig` |
| 顶层 import | 本地源码 | `from .pi0.configuration_pi0 import PI0Config as PI0Config`                         |
| 顶层 import | 本地源码 | `from .pi0.processor_pi0 import Pi0NewLineProcessor`                                |
| 顶层 import | 本地源码 | `from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig`         |
| 顶层 import | 本地源码 | `from .smolvla.processor_smolvla import SmolVLANewLineProcessor`                    |
| 顶层 import | 本地源码 | `from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig`                 |
| 顶层 import | 本地源码 | `from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig`                 |

#### `lerobot/src/lerobot/policies/factory.py`
| 时机              | 类别     | import 语句                                                                                                                                        |
| ----------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import       | 标准库   | `from __future__ import annotations`                                                                                                               |
| 顶层 import       | 标准库   | `import logging`                                                                                                                                   |
| 顶层 import       | 标准库   | `from typing import Any, TypedDict`                                                                                                                |
| 顶层 import       | 第三方库 | `import torch`                                                                                                                                     |
| 顶层 import       | 第三方库 | `from typing_extensions import Unpack`                                                                                                             |
| 顶层 import       | 本地源码 | `from lerobot.configs.policies import PreTrainedConfig`                                                                                            |
| 顶层 import       | 本地源码 | `from lerobot.configs.types import FeatureType`                                                                                                    |
| 顶层 import       | 本地源码 | `from lerobot.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME`                                                |
| 顶层 import       | 本地源码 | `from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata`                                                                              |
| 顶层 import       | 本地源码 | `from lerobot.datasets.utils import dataset_to_policy_features`                                                                                    |
| 顶层 import       | 本地源码 | `from lerobot.envs.configs import EnvConfig`                                                                                                       |
| 顶层 import       | 本地源码 | `from lerobot.envs.utils import env_to_policy_features`                                                                                            |
| 顶层 import       | 本地源码 | `from lerobot.policies.act.configuration_act import ACTConfig`                                                                                     |
| 顶层 import       | 本地源码 | `from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig`                                                                   |
| 顶层 import       | 本地源码 | `from lerobot.policies.pi0.configuration_pi0 import PI0Config`                                                                                     |
| 顶层 import       | 本地源码 | `from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig`                                                                         |
| 顶层 import       | 本地源码 | `from lerobot.policies.pretrained import PreTrainedPolicy`                                                                                         |
| 顶层 import       | 本地源码 | `from lerobot.policies.sac.configuration_sac import SACConfig`                                                                                     |
| 顶层 import       | 本地源码 | `from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig`                                                    |
| 顶层 import       | 本地源码 | `from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig`                                                                         |
| 顶层 import       | 本地源码 | `from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig`                                                                               |
| 顶层 import       | 本地源码 | `from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig`                                                                               |
| 顶层 import       | 本地源码 | `from lerobot.processor import PolicyAction, PolicyProcessorPipeline`                                                                              |
| 顶层 import       | 本地源码 | `from lerobot.processor.converters import ( batch_to_transition, policy_action_to_transition, transition_to_batch, transition_to_policy_action, )` |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy`                                                                                    |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy`                                                                        |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.act.modeling_act import ACTPolicy`                                                                                          |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy`                                                                                    |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.pi0.modeling_pi0 import PI0Policy`                                                                                          |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy`                                                                              |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.sac.modeling_sac import SACPolicy`                                                                                          |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.sac.reward_model.modeling_classifier import Classifier`                                                                     |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy`                                                                              |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_pre_post_processors`                                                                |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors`                                                    |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.act.processor_act import make_act_pre_post_processors`                                                                      |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.vqbet.processor_vqbet import make_vqbet_pre_post_processors`                                                                |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors`                                                                      |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.pi0fast.processor_pi0fast import make_pi0fast_pre_post_processors`                                                          |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors`                                                                      |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.sac.reward_model.processor_classifier import make_classifier_processor`                                                     |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors`                                                          |

#### `lerobot/src/lerobot/policies/pretrained.py`
| 时机        | 类别     | import 语句                                                                                                    |
| ----------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                                                                                   |
| 顶层 import | 标准库   | `import builtins`                                                                                              |
| 顶层 import | 标准库   | `import logging`                                                                                               |
| 顶层 import | 标准库   | `import os`                                                                                                    |
| 顶层 import | 标准库   | `from importlib.resources import files`                                                                        |
| 顶层 import | 标准库   | `from pathlib import Path`                                                                                     |
| 顶层 import | 标准库   | `from tempfile import TemporaryDirectory`                                                                      |
| 顶层 import | 标准库   | `from typing import TypeVar`                                                                                   |
| 顶层 import | 第三方库 | `import packaging`                                                                                             |
| 顶层 import | 第三方库 | `import safetensors`                                                                                           |
| 顶层 import | 第三方库 | `from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download`                                 |
| 顶层 import | 第三方库 | `from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE`                                                |
| 顶层 import | 第三方库 | `from huggingface_hub.errors import HfHubHTTPError`                                                            |
| 顶层 import | 第三方库 | `from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor` |
| 顶层 import | 第三方库 | `from torch import Tensor, nn`                                                                                 |
| 顶层 import | 本地源码 | `from lerobot.configs.policies import PreTrainedConfig`                                                        |
| 顶层 import | 本地源码 | `from lerobot.configs.train import TrainPipelineConfig`                                                        |
| 顶层 import | 本地源码 | `from lerobot.policies.utils import log_model_loading_keys`                                                    |
| 顶层 import | 本地源码 | `from lerobot.utils.hub import HubMixin`                                                                       |

#### `lerobot/src/lerobot/policies/utils.py`
| 时机        | 类别     | import 语句                     |
| ----------- | -------- | ------------------------------- |
| 顶层 import | 标准库   | `import logging`                |
| 顶层 import | 标准库   | `from collections import deque` |
| 顶层 import | 第三方库 | `import torch`                  |
| 顶层 import | 第三方库 | `from torch import nn`          |

#### `lerobot/src/lerobot/policies/act/configuration_act.py`
| 时机        | 类别     | import 语句                                             |
| ----------- | -------- | ------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`              |
| 顶层 import | 本地源码 | `from lerobot.configs.policies import PreTrainedConfig` |
| 顶层 import | 本地源码 | `from lerobot.configs.types import NormalizationMode`   |
| 顶层 import | 本地源码 | `from lerobot.optim.optimizers import AdamWConfig`      |

#### `lerobot/src/lerobot/policies/act/modeling_act.py`
| 时机        | 类别     | import 语句                                                     |
| ----------- | -------- | --------------------------------------------------------------- |
| 顶层 import | 标准库   | `import math`                                                   |
| 顶层 import | 标准库   | `from collections import deque`                                 |
| 顶层 import | 标准库   | `from collections.abc import Callable`                          |
| 顶层 import | 标准库   | `from itertools import chain`                                   |
| 顶层 import | 第三方库 | `import einops`                                                 |
| 顶层 import | 第三方库 | `import numpy as np`                                            |
| 顶层 import | 第三方库 | `import torch`                                                  |
| 顶层 import | 第三方库 | `import torch.nn.functional as F`                               |
| 顶层 import | 第三方库 | `import torchvision`                                            |
| 顶层 import | 第三方库 | `from torch import Tensor, nn`                                  |
| 顶层 import | 第三方库 | `from torchvision.models._utils import IntermediateLayerGetter` |
| 顶层 import | 第三方库 | `from torchvision.ops.misc import FrozenBatchNorm2d`            |
| 顶层 import | 本地源码 | `from lerobot.constants import ACTION, OBS_IMAGES`              |
| 顶层 import | 本地源码 | `from lerobot.policies.act.configuration_act import ACTConfig`  |
| 顶层 import | 本地源码 | `from lerobot.policies.pretrained import PreTrainedPolicy`      |
| 顶层 import | 标准库   | `import csv, os, time as _time`                                 |

#### `lerobot/src/lerobot/policies/act/processor_act.py`
| 时机        | 类别     | import 语句                                                                                                                                                                                                          |
| ----------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from typing import Any`                                                                                                                                                                                             |
| 顶层 import | 第三方库 | `import torch`                                                                                                                                                                                                       |
| 顶层 import | 本地源码 | `from lerobot.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME`                                                                                                                  |
| 顶层 import | 本地源码 | `from lerobot.policies.act.configuration_act import ACTConfig`                                                                                                                                                       |
| 顶层 import | 本地源码 | `from lerobot.processor import ( AddBatchDimensionProcessorStep, DeviceProcessorStep, NormalizerProcessorStep, PolicyAction, PolicyProcessorPipeline, RenameObservationsProcessorStep, UnnormalizerProcessorStep, )` |
| 顶层 import | 本地源码 | `from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action`                                                                                                                  |

#### `lerobot/src/lerobot/processor/__init__.py`
| 时机        | 类别     | import 语句                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .batch_processor import AddBatchDimensionProcessorStep`                                                                                                                                                                                                                                                                                                                                            |
| 顶层 import | 本地源码 | `from .converters import ( batch_to_transition, create_transition, transition_to_batch, )`                                                                                                                                                                                                                                                                                                               |
| 顶层 import | 本地源码 | `from .core import ( EnvAction, EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey, )`                                                                                                                                                                                                                                                                                            |
| 顶层 import | 本地源码 | `from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep`                                                                                                                                                                                                                                                                                                    |
| 顶层 import | 本地源码 | `from .device_processor import DeviceProcessorStep`                                                                                                                                                                                                                                                                                                                                                      |
| 顶层 import | 本地源码 | `from .factory import ( make_default_processors, make_default_robot_action_processor, make_default_robot_observation_processor, make_default_teleop_action_processor, )`                                                                                                                                                                                                                                 |
| 顶层 import | 本地源码 | `from .gym_action_processor import ( Numpy2TorchActionProcessorStep, Torch2NumpyActionProcessorStep, )`                                                                                                                                                                                                                                                                                                  |
| 顶层 import | 本地源码 | `from .hil_processor import ( AddTeleopActionAsComplimentaryDataStep, AddTeleopEventsAsInfoStep, GripperPenaltyProcessorStep, ImageCropResizeProcessorStep, InterventionActionProcessorStep, RewardClassifierProcessorStep, TimeLimitProcessorStep, )`                                                                                                                                                   |
| 顶层 import | 本地源码 | `from .joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep`                                                                                                                                                                                                                                                                                                        |
| 顶层 import | 本地源码 | `from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep, hotswap_stats`                                                                                                                                                                                                                                                                                                     |
| 顶层 import | 本地源码 | `from .observation_processor import VanillaObservationProcessorStep`                                                                                                                                                                                                                                                                                                                                     |
| 顶层 import | 本地源码 | `from .pipeline import ( ActionProcessorStep, ComplementaryDataProcessorStep, DataProcessorPipeline, DoneProcessorStep, IdentityProcessorStep, InfoProcessorStep, ObservationProcessorStep, PolicyActionProcessorStep, PolicyProcessorPipeline, ProcessorKwargs, ProcessorStep, ProcessorStepRegistry, RewardProcessorStep, RobotActionProcessorStep, RobotProcessorPipeline, TruncatedProcessorStep, )` |
| 顶层 import | 本地源码 | `from .policy_robot_bridge import ( PolicyActionToRobotActionProcessorStep, RobotActionToPolicyActionProcessorStep, )`                                                                                                                                                                                                                                                                                   |
| 顶层 import | 本地源码 | `from .rename_processor import RenameObservationsProcessorStep`                                                                                                                                                                                                                                                                                                                                          |
| 顶层 import | 本地源码 | `from .tokenizer_processor import TokenizerProcessorStep`                                                                                                                                                                                                                                                                                                                                                |

#### `lerobot/src/lerobot/processor/batch_processor.py`
| 时机        | 类别     | import 语句                                                                                                                                                           |
| ----------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                                                                                                                            |
| 顶层 import | 第三方库 | `from torch import Tensor`                                                                                                                                            |
| 顶层 import | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature`                                                                                                |
| 顶层 import | 本地源码 | `from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE`                                                                                       |
| 顶层 import | 本地源码 | `from .core import EnvTransition, PolicyAction`                                                                                                                       |
| 顶层 import | 本地源码 | `from .pipeline import ( ComplementaryDataProcessorStep, ObservationProcessorStep, PolicyActionProcessorStep, ProcessorStep, ProcessorStepRegistry, TransitionKey, )` |

#### `lerobot/src/lerobot/processor/converters.py`
| 时机        | 类别     | import 语句                                                                                   |
| ----------- | -------- | --------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from __future__ import annotations`                                                          |
| 顶层 import | 标准库   | `from collections.abc import Sequence`                                                        |
| 顶层 import | 标准库   | `from functools import singledispatch`                                                        |
| 顶层 import | 标准库   | `from typing import Any`                                                                      |
| 顶层 import | 第三方库 | `import numpy as np`                                                                          |
| 顶层 import | 第三方库 | `import torch`                                                                                |
| 顶层 import | 本地源码 | `from .core import EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey` |

#### `lerobot/src/lerobot/processor/core.py`
| 时机        | 类别     | import 语句                                    |
| ----------- | -------- | ---------------------------------------------- |
| 顶层 import | 标准库   | `from __future__ import annotations`           |
| 顶层 import | 标准库   | `from enum import Enum`                        |
| 顶层 import | 标准库   | `from typing import Any, TypeAlias, TypedDict` |
| 顶层 import | 第三方库 | `import numpy as np`                           |
| 顶层 import | 第三方库 | `import torch`                                 |

#### `lerobot/src/lerobot/processor/delta_action_processor.py`
| 时机        | 类别     | import 语句                                                                                  |
| ----------- | -------- | -------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass`                                                          |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature`          |
| 顶层 import | 本地源码 | `from .core import PolicyAction, RobotAction`                                                |
| 顶层 import | 本地源码 | `from .pipeline import ActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep` |

#### `lerobot/src/lerobot/processor/device_processor.py`
| 时机        | 类别     | import 语句                                                            |
| ----------- | -------- | ---------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass`                                    |
| 顶层 import | 标准库   | `from typing import Any`                                               |
| 顶层 import | 第三方库 | `import torch`                                                         |
| 顶层 import | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature` |
| 顶层 import | 本地源码 | `from lerobot.utils.utils import get_safe_torch_device`                |
| 顶层 import | 本地源码 | `from .core import EnvTransition, PolicyAction, TransitionKey`         |
| 顶层 import | 本地源码 | `from .pipeline import ProcessorStep, ProcessorStepRegistry`           |

#### `lerobot/src/lerobot/processor/factory.py`
| 时机        | 类别     | import 语句                                                                                                                                             |
| ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .converters import ( observation_to_transition, robot_action_observation_to_transition, transition_to_observation, transition_to_robot_action, )` |
| 顶层 import | 本地源码 | `from .core import RobotAction, RobotObservation`                                                                                                       |
| 顶层 import | 本地源码 | `from .pipeline import IdentityProcessorStep, RobotProcessorPipeline`                                                                                   |

#### `lerobot/src/lerobot/processor/gym_action_processor.py`
| 时机              | 类别     | import 语句                                                                       |
| ----------------- | -------- | --------------------------------------------------------------------------------- |
| 顶层 import       | 标准库   | `from dataclasses import dataclass`                                               |
| 顶层 import       | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature`            |
| 顶层 import       | 本地源码 | `from .converters import to_tensor`                                               |
| 顶层 import       | 本地源码 | `from .core import EnvAction, EnvTransition, PolicyAction`                        |
| 顶层 import       | 本地源码 | `from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry` |
| 函数内延迟 import | 本地源码 | `from .core import TransitionKey`                                                 |

#### `lerobot/src/lerobot/processor/hil_processor.py`
| 时机              | 类别     | import 语句                                                                                                                                                            |
| ----------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import       | 标准库   | `import math`                                                                                                                                                          |
| 顶层 import       | 标准库   | `import time`                                                                                                                                                          |
| 顶层 import       | 标准库   | `from dataclasses import dataclass`                                                                                                                                    |
| 顶层 import       | 标准库   | `from typing import Any, Protocol, TypeVar, runtime_checkable`                                                                                                         |
| 顶层 import       | 第三方库 | `import numpy as np`                                                                                                                                                   |
| 顶层 import       | 第三方库 | `import torch`                                                                                                                                                         |
| 顶层 import       | 第三方库 | `import torchvision.transforms.functional as F`                                                                                                                        |
| 顶层 import       | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature`                                                                                                 |
| 顶层 import       | 本地源码 | `from lerobot.teleoperators.teleoperator import Teleoperator`                                                                                                          |
| 顶层 import       | 本地源码 | `from lerobot.teleoperators.utils import TeleopEvents`                                                                                                                 |
| 顶层 import       | 本地源码 | `from .core import EnvTransition, PolicyAction, TransitionKey`                                                                                                         |
| 顶层 import       | 本地源码 | `from .pipeline import ( ComplementaryDataProcessorStep, InfoProcessorStep, ObservationProcessorStep, ProcessorStep, ProcessorStepRegistry, TruncatedProcessorStep, )` |
| 函数内延迟 import | 本地源码 | `from lerobot.policies.sac.reward_model.modeling_classifier import Classifier`                                                                                         |

#### `lerobot/src/lerobot/processor/joint_observations_processor.py`
| 时机        | 类别     | import 语句                                                                                   |
| ----------- | -------- | --------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass`                                                           |
| 顶层 import | 标准库   | `from typing import Any`                                                                      |
| 顶层 import | 第三方库 | `import torch`                                                                                |
| 顶层 import | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature`                        |
| 顶层 import | 本地源码 | `from lerobot.constants import OBS_STATE`                                                     |
| 顶层 import | 本地源码 | `from lerobot.processor.pipeline import ( ObservationProcessorStep, ProcessorStepRegistry, )` |
| 顶层 import | 本地源码 | `from lerobot.robots import Robot`                                                            |

#### `lerobot/src/lerobot/processor/normalize_processor.py`
| 时机        | 类别     | import 语句                                                                                            |
| ----------- | -------- | ------------------------------------------------------------------------------------------------------ |
| 顶层 import | 标准库   | `from __future__ import annotations`                                                                   |
| 顶层 import | 标准库   | `from copy import deepcopy`                                                                            |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                                                             |
| 顶层 import | 标准库   | `from typing import Any`                                                                               |
| 顶层 import | 第三方库 | `import torch`                                                                                         |
| 顶层 import | 第三方库 | `from torch import Tensor`                                                                             |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature` |
| 顶层 import | 本地源码 | `from lerobot.datasets.lerobot_dataset import LeRobotDataset`                                          |
| 顶层 import | 本地源码 | `from .converters import from_tensor_to_numpy, to_tensor`                                              |
| 顶层 import | 本地源码 | `from .core import EnvTransition, PolicyAction, TransitionKey`                                         |
| 顶层 import | 本地源码 | `from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry`                  |

#### `lerobot/src/lerobot/processor/observation_processor.py`
| 时机        | 类别     | import 语句                                                                     |
| ----------- | -------- | ------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass`                                             |
| 顶层 import | 第三方库 | `import einops`                                                                 |
| 顶层 import | 第三方库 | `import numpy as np`                                                            |
| 顶层 import | 第三方库 | `import torch`                                                                  |
| 顶层 import | 第三方库 | `from torch import Tensor`                                                      |
| 顶层 import | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature`          |
| 顶层 import | 本地源码 | `from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE` |
| 顶层 import | 本地源码 | `from .pipeline import ObservationProcessorStep, ProcessorStepRegistry`         |

#### `lerobot/src/lerobot/processor/pipeline.py`
| 时机              | 类别     | import 语句                                                                            |
| ----------------- | -------- | -------------------------------------------------------------------------------------- |
| 顶层 import       | 标准库   | `from __future__ import annotations`                                                   |
| 顶层 import       | 标准库   | `import importlib`                                                                     |
| 顶层 import       | 标准库   | `import json`                                                                          |
| 顶层 import       | 标准库   | `import os`                                                                            |
| 顶层 import       | 标准库   | `import re`                                                                            |
| 顶层 import       | 标准库   | `from abc import ABC, abstractmethod`                                                  |
| 顶层 import       | 标准库   | `from collections.abc import Callable, Iterable, Sequence`                             |
| 顶层 import       | 标准库   | `from copy import deepcopy`                                                            |
| 顶层 import       | 标准库   | `from dataclasses import dataclass, field`                                             |
| 顶层 import       | 标准库   | `from pathlib import Path`                                                             |
| 顶层 import       | 标准库   | `from typing import Any, Generic, TypeAlias, TypedDict, TypeVar, cast`                 |
| 顶层 import       | 第三方库 | `import torch`                                                                         |
| 顶层 import       | 第三方库 | `from huggingface_hub import hf_hub_download`                                          |
| 顶层 import       | 第三方库 | `from safetensors.torch import load_file, save_file`                                   |
| 顶层 import       | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature`                 |
| 顶层 import       | 本地源码 | `from lerobot.utils.hub import HubMixin`                                               |
| 顶层 import       | 本地源码 | `from .converters import batch_to_transition, create_transition, transition_to_batch`  |
| 顶层 import       | 本地源码 | `from .core import EnvAction, EnvTransition, PolicyAction, RobotAction, TransitionKey` |
| 函数内延迟 import | 本地源码 | `from lerobot.constants import HF_LEROBOT_HOME`                                        |

#### `lerobot/src/lerobot/processor/policy_robot_bridge.py`
| 时机        | 类别     | import 语句                                                                                           |
| ----------- | -------- | ----------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import asdict, dataclass`                                                           |
| 顶层 import | 标准库   | `from typing import Any`                                                                              |
| 顶层 import | 第三方库 | `import torch`                                                                                        |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature`                   |
| 顶层 import | 本地源码 | `from lerobot.processor import ActionProcessorStep, PolicyAction, ProcessorStepRegistry, RobotAction` |

#### `lerobot/src/lerobot/processor/rename_processor.py`
| 时机        | 类别     | import 语句                                                             |
| ----------- | -------- | ----------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from copy import deepcopy`                                             |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                              |
| 顶层 import | 标准库   | `from typing import Any`                                                |
| 顶层 import | 本地源码 | `from lerobot.configs.types import PipelineFeatureType, PolicyFeature`  |
| 顶层 import | 本地源码 | `from .pipeline import ObservationProcessorStep, ProcessorStepRegistry` |

#### `lerobot/src/lerobot/processor/tokenizer_processor.py`
| 时机        | 类别     | import 语句                                                                         |
| ----------- | -------- | ----------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from __future__ import annotations`                                                |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                                          |
| 顶层 import | 标准库   | `from typing import TYPE_CHECKING, Any`                                             |
| 顶层 import | 第三方库 | `import torch`                                                                      |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature` |
| 顶层 import | 本地源码 | `from lerobot.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS`    |
| 顶层 import | 本地源码 | `from lerobot.utils.import_utils import _transformers_available`                    |
| 顶层 import | 本地源码 | `from .core import EnvTransition, TransitionKey`                                    |
| 顶层 import | 本地源码 | `from .pipeline import ObservationProcessorStep, ProcessorStepRegistry`             |
| 顶层 import | 第三方库 | `from transformers import AutoTokenizer`                                            |

#### `lerobot/src/lerobot/robots/__init__.py`
| 时机        | 类别     | import 语句                                 |
| ----------- | -------- | ------------------------------------------- |
| 顶层 import | 本地源码 | `from .config import RobotConfig`           |
| 顶层 import | 本地源码 | `from .robot import Robot`                  |
| 顶层 import | 本地源码 | `from .utils import make_robot_from_config` |

#### `lerobot/src/lerobot/robots/config.py`
| 时机        | 类别     | import 语句                         |
| ----------- | -------- | ----------------------------------- |
| 顶层 import | 标准库   | `import abc`                        |
| 顶层 import | 标准库   | `from dataclasses import dataclass` |
| 顶层 import | 标准库   | `from pathlib import Path`          |
| 顶层 import | 第三方库 | `import draccus`                    |

#### `lerobot/src/lerobot/robots/robot.py`
| 时机        | 类别     | import 语句                                                    |
| ----------- | -------- | -------------------------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                                   |
| 顶层 import | 标准库   | `import builtins`                                              |
| 顶层 import | 标准库   | `from pathlib import Path`                                     |
| 顶层 import | 标准库   | `from typing import Any`                                       |
| 顶层 import | 第三方库 | `import draccus`                                               |
| 顶层 import | 本地源码 | `from lerobot.constants import HF_LEROBOT_CALIBRATION, ROBOTS` |
| 顶层 import | 本地源码 | `from lerobot.motors import MotorCalibration`                  |
| 顶层 import | 本地源码 | `from .config import RobotConfig`                              |

#### `lerobot/src/lerobot/robots/utils.py`
| 时机              | 类别     | import 语句                                      |
| ----------------- | -------- | ------------------------------------------------ |
| 顶层 import       | 标准库   | `import logging`                                 |
| 顶层 import       | 标准库   | `from pprint import pformat`                     |
| 顶层 import       | 本地源码 | `from lerobot.robots import RobotConfig`         |
| 顶层 import       | 本地源码 | `from .robot import Robot`                       |
| 函数内延迟 import | 本地源码 | `from .koch_follower import KochFollower`        |
| 函数内延迟 import | 本地源码 | `from .so100_follower import SO100Follower`      |
| 函数内延迟 import | 本地源码 | `from .so101_follower import SO101Follower`      |
| 函数内延迟 import | 本地源码 | `from .lekiwi import LeKiwi`                     |
| 函数内延迟 import | 本地源码 | `from .stretch3 import Stretch3Robot`            |
| 函数内延迟 import | 本地源码 | `from .viperx import ViperX`                     |
| 函数内延迟 import | 本地源码 | `from .hope_jr import HopeJrHand`                |
| 函数内延迟 import | 本地源码 | `from .hope_jr import HopeJrArm`                 |
| 函数内延迟 import | 本地源码 | `from .bi_so100_follower import BiSO100Follower` |
| 函数内延迟 import | 本地源码 | `from .bi_so101_follower import BiSO101Follower` |
| 函数内延迟 import | 本地源码 | `from .xlerobot import XLerobot`                 |
| 函数内延迟 import | 本地源码 | `from .reachy2 import Reachy2Robot`              |
| 函数内延迟 import | 第三方库 | `from tests.mocks.mock_robot import MockRobot`   |

#### `lerobot/src/lerobot/robots/so101_follower/__init__.py`
| 时机        | 类别     | import 语句                                              |
| ----------- | -------- | -------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .config_so101_follower import SO101FollowerConfig` |
| 顶层 import | 本地源码 | `from .so101_follower import SO101Follower`              |

#### `lerobot/src/lerobot/robots/so101_follower/config_so101_follower.py`
| 时机        | 类别     | import 语句                                |
| ----------- | -------- | ------------------------------------------ |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field` |
| 顶层 import | 本地源码 | `from lerobot.cameras import CameraConfig` |
| 顶层 import | 本地源码 | `from ..config import RobotConfig`         |

#### `lerobot/src/lerobot/robots/so101_follower/so101_follower.py`
| 时机        | 类别     | import 语句                                                                       |
| ----------- | -------- | --------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import logging`                                                                  |
| 顶层 import | 标准库   | `import time`                                                                     |
| 顶层 import | 标准库   | `from functools import cached_property`                                           |
| 顶层 import | 标准库   | `from typing import Any`                                                          |
| 顶层 import | 本地源码 | `from lerobot.cameras.utils import make_cameras_from_configs`                     |
| 顶层 import | 本地源码 | `from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError` |
| 顶层 import | 本地源码 | `from lerobot.motors import Motor, MotorCalibration, MotorNormMode`               |
| 顶层 import | 本地源码 | `from lerobot.motors.feetech import ( FeetechMotorsBus, OperatingMode, )`         |
| 顶层 import | 本地源码 | `from ..robot import Robot`                                                       |
| 顶层 import | 本地源码 | `from ..utils import ensure_safe_goal_position`                                   |
| 顶层 import | 本地源码 | `from .config_so101_follower import SO101FollowerConfig`                          |

#### `lerobot/src/lerobot/teleoperators/__init__.py`
| 时机        | 类别     | import 语句                                                      |
| ----------- | -------- | ---------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .config import TeleoperatorConfig`                         |
| 顶层 import | 本地源码 | `from .teleoperator import Teleoperator`                         |
| 顶层 import | 本地源码 | `from .utils import TeleopEvents, make_teleoperator_from_config` |

#### `lerobot/src/lerobot/teleoperators/config.py`
| 时机        | 类别     | import 语句                         |
| ----------- | -------- | ----------------------------------- |
| 顶层 import | 标准库   | `import abc`                        |
| 顶层 import | 标准库   | `from dataclasses import dataclass` |
| 顶层 import | 标准库   | `from pathlib import Path`          |
| 顶层 import | 第三方库 | `import draccus`                    |

#### `lerobot/src/lerobot/teleoperators/teleoperator.py`
| 时机        | 类别     | import 语句                                                           |
| ----------- | -------- | --------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                                          |
| 顶层 import | 标准库   | `import builtins`                                                     |
| 顶层 import | 标准库   | `from pathlib import Path`                                            |
| 顶层 import | 标准库   | `from typing import Any`                                              |
| 顶层 import | 第三方库 | `import draccus`                                                      |
| 顶层 import | 本地源码 | `from lerobot.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS` |
| 顶层 import | 本地源码 | `from lerobot.motors.motors_bus import MotorCalibration`              |
| 顶层 import | 本地源码 | `from .config import TeleoperatorConfig`                              |

#### `lerobot/src/lerobot/teleoperators/utils.py`
| 时机              | 类别     | import 语句                                                       |
| ----------------- | -------- | ----------------------------------------------------------------- |
| 顶层 import       | 标准库   | `from enum import Enum`                                           |
| 顶层 import       | 本地源码 | `from .config import TeleoperatorConfig`                          |
| 顶层 import       | 本地源码 | `from .teleoperator import Teleoperator`                          |
| 函数内延迟 import | 本地源码 | `from .keyboard import KeyboardTeleop`                            |
| 函数内延迟 import | 本地源码 | `from .koch_leader import KochLeader`                             |
| 函数内延迟 import | 本地源码 | `from .so100_leader import SO100Leader`                           |
| 函数内延迟 import | 本地源码 | `from .so101_leader import SO101Leader`                           |
| 函数内延迟 import | 本地源码 | `from .stretch3_gamepad import Stretch3GamePad`                   |
| 函数内延迟 import | 本地源码 | `from .widowx import WidowX`                                      |
| 函数内延迟 import | 第三方库 | `from tests.mocks.mock_teleop import MockTeleop`                  |
| 函数内延迟 import | 本地源码 | `from .gamepad.teleop_gamepad import GamepadTeleop`               |
| 函数内延迟 import | 本地源码 | `from .keyboard.teleop_keyboard import KeyboardEndEffectorTeleop` |
| 函数内延迟 import | 本地源码 | `from .homunculus import HomunculusGlove`                         |
| 函数内延迟 import | 本地源码 | `from .homunculus import HomunculusArm`                           |
| 函数内延迟 import | 本地源码 | `from .bi_so100_leader import BiSO100Leader`                      |
| 函数内延迟 import | 本地源码 | `from .bi_so101_leader import BiSO101Leader`                      |
| 函数内延迟 import | 本地源码 | `from .xlebi_so101_leader import XleBiSO101Leader`                |
| 函数内延迟 import | 本地源码 | `from .reachy2_teleoperator import Reachy2Teleoperator`           |

#### `lerobot/src/lerobot/teleoperators/keyboard/__init__.py`
| 时机        | 类别     | import 语句                                                                                 |
| ----------- | -------- | ------------------------------------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig` |
| 顶层 import | 本地源码 | `from .teleop_keyboard import KeyboardEndEffectorTeleop, KeyboardTeleop`                    |

#### `lerobot/src/lerobot/teleoperators/keyboard/configuration_keyboard.py`
| 时机        | 类别     | import 语句                               |
| ----------- | -------- | ----------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass`       |
| 顶层 import | 本地源码 | `from ..config import TeleoperatorConfig` |

#### `lerobot/src/lerobot/teleoperators/keyboard/teleop_keyboard.py`
| 时机        | 类别     | import 语句                                                                                 |
| ----------- | -------- | ------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import logging`                                                                            |
| 顶层 import | 标准库   | `import os`                                                                                 |
| 顶层 import | 标准库   | `import sys`                                                                                |
| 顶层 import | 标准库   | `import time`                                                                               |
| 顶层 import | 标准库   | `from queue import Queue`                                                                   |
| 顶层 import | 标准库   | `from typing import Any`                                                                    |
| 顶层 import | 本地源码 | `from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError`           |
| 顶层 import | 本地源码 | `from ..teleoperator import Teleoperator`                                                   |
| 顶层 import | 本地源码 | `from ..utils import TeleopEvents`                                                          |
| 顶层 import | 本地源码 | `from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig` |
| 顶层 import | 第三方库 | `from pynput import keyboard`                                                               |

#### `lerobot/src/lerobot/teleoperators/so101_leader/__init__.py`
| 时机        | 类别     | import 语句                                          |
| ----------- | -------- | ---------------------------------------------------- |
| 顶层 import | 本地源码 | `from .config_so101_leader import SO101LeaderConfig` |
| 顶层 import | 本地源码 | `from .so101_leader import SO101Leader`              |

#### `lerobot/src/lerobot/teleoperators/so101_leader/config_so101_leader.py`
| 时机        | 类别     | import 语句                               |
| ----------- | -------- | ----------------------------------------- |
| 顶层 import | 标准库   | `from dataclasses import dataclass`       |
| 顶层 import | 本地源码 | `from ..config import TeleoperatorConfig` |

#### `lerobot/src/lerobot/teleoperators/so101_leader/so101_leader.py`
| 时机        | 类别     | import 语句                                                                       |
| ----------- | -------- | --------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import logging`                                                                  |
| 顶层 import | 标准库   | `import time`                                                                     |
| 顶层 import | 本地源码 | `from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError` |
| 顶层 import | 本地源码 | `from lerobot.motors import Motor, MotorCalibration, MotorNormMode`               |
| 顶层 import | 本地源码 | `from lerobot.motors.feetech import ( FeetechMotorsBus, OperatingMode, )`         |
| 顶层 import | 本地源码 | `from ..teleoperator import Teleoperator`                                         |
| 顶层 import | 本地源码 | `from .config_so101_leader import SO101LeaderConfig`                              |

#### `lerobot/src/lerobot/cameras/__init__.py`
| 时机        | 类别     | import 语句                                                 |
| ----------- | -------- | ----------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .camera import Camera`                                |
| 顶层 import | 本地源码 | `from .configs import CameraConfig, ColorMode, Cv2Rotation` |
| 顶层 import | 本地源码 | `from .utils import make_cameras_from_configs`              |

#### `lerobot/src/lerobot/cameras/camera.py`
| 时机        | 类别     | import 语句                                    |
| ----------- | -------- | ---------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                   |
| 顶层 import | 标准库   | `from typing import Any`                       |
| 顶层 import | 第三方库 | `import numpy as np`                           |
| 顶层 import | 本地源码 | `from .configs import CameraConfig, ColorMode` |

#### `lerobot/src/lerobot/cameras/configs.py`
| 时机        | 类别     | import 语句                         |
| ----------- | -------- | ----------------------------------- |
| 顶层 import | 标准库   | `import abc`                        |
| 顶层 import | 标准库   | `from dataclasses import dataclass` |
| 顶层 import | 标准库   | `from enum import Enum`             |
| 顶层 import | 第三方库 | `import draccus`                    |

#### `lerobot/src/lerobot/cameras/utils.py`
| 时机              | 类别     | import 语句                                                |
| ----------------- | -------- | ---------------------------------------------------------- |
| 顶层 import       | 标准库   | `import platform`                                          |
| 顶层 import       | 标准库   | `from pathlib import Path`                                 |
| 顶层 import       | 标准库   | `from typing import TypeAlias`                             |
| 顶层 import       | 本地源码 | `from .camera import Camera`                               |
| 顶层 import       | 本地源码 | `from .configs import CameraConfig, Cv2Rotation`           |
| 函数内延迟 import | 本地源码 | `from .opencv import OpenCVCamera`                         |
| 函数内延迟 import | 本地源码 | `from .realsense.camera_realsense import RealSenseCamera`  |
| 函数内延迟 import | 本地源码 | `from .reachy2_camera.reachy2_camera import Reachy2Camera` |
| 函数内延迟 import | 第三方库 | `import cv2`                                               |
| 函数内延迟 import | 第三方库 | `import cv2`                                               |

#### `lerobot/src/lerobot/cameras/opencv/__init__.py`
| 时机        | 类别     | import 语句                                            |
| ----------- | -------- | ------------------------------------------------------ |
| 顶层 import | 本地源码 | `from .camera_opencv import OpenCVCamera`              |
| 顶层 import | 本地源码 | `from .configuration_opencv import OpenCVCameraConfig` |

#### `lerobot/src/lerobot/cameras/opencv/configuration_opencv.py`
| 时机        | 类别     | import 语句                                                  |
| ----------- | -------- | ------------------------------------------------------------ |
| 顶层 import | 标准库   | `from dataclasses import dataclass`                          |
| 顶层 import | 标准库   | `from pathlib import Path`                                   |
| 顶层 import | 本地源码 | `from ..configs import CameraConfig, ColorMode, Cv2Rotation` |

#### `lerobot/src/lerobot/cameras/opencv/camera_opencv.py`
| 时机        | 类别     | import 语句                                                                       |
| ----------- | -------- | --------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import logging`                                                                  |
| 顶层 import | 标准库   | `import math`                                                                     |
| 顶层 import | 标准库   | `import os`                                                                       |
| 顶层 import | 标准库   | `import platform`                                                                 |
| 顶层 import | 标准库   | `import time`                                                                     |
| 顶层 import | 标准库   | `from pathlib import Path`                                                        |
| 顶层 import | 标准库   | `from threading import Event, Lock, Thread`                                       |
| 顶层 import | 标准库   | `from typing import Any`                                                          |
| 顶层 import | 第三方库 | `import cv2`                                                                      |
| 顶层 import | 第三方库 | `import numpy as np`                                                              |
| 顶层 import | 本地源码 | `from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError` |
| 顶层 import | 本地源码 | `from ..camera import Camera`                                                     |
| 顶层 import | 本地源码 | `from ..utils import get_cv2_backend, get_cv2_rotation`                           |
| 顶层 import | 本地源码 | `from .configuration_opencv import ColorMode, OpenCVCameraConfig`                 |

#### `lerobot/src/lerobot/cameras/realsense/__init__.py`
| 时机        | 类别     | import 语句                                                  |
| ----------- | -------- | ------------------------------------------------------------ |
| 顶层 import | 本地源码 | `from .camera_realsense import RealSenseCamera`              |
| 顶层 import | 本地源码 | `from .configuration_realsense import RealSenseCameraConfig` |

#### `lerobot/src/lerobot/cameras/realsense/configuration_realsense.py`
| 时机        | 类别     | import 语句                                                  |
| ----------- | -------- | ------------------------------------------------------------ |
| 顶层 import | 标准库   | `from dataclasses import dataclass`                          |
| 顶层 import | 本地源码 | `from ..configs import CameraConfig, ColorMode, Cv2Rotation` |

#### `lerobot/src/lerobot/cameras/realsense/camera_realsense.py`
| 时机        | 类别     | import 语句                                                                       |
| ----------- | -------- | --------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import logging`                                                                  |
| 顶层 import | 标准库   | `import time`                                                                     |
| 顶层 import | 标准库   | `from threading import Event, Lock, Thread`                                       |
| 顶层 import | 标准库   | `from typing import Any`                                                          |
| 顶层 import | 第三方库 | `import cv2`                                                                      |
| 顶层 import | 第三方库 | `import numpy as np`                                                              |
| 顶层 import | 第三方库 | `import pyrealsense2 as rs`                                                       |
| 顶层 import | 本地源码 | `from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError` |
| 顶层 import | 本地源码 | `from ..camera import Camera`                                                     |
| 顶层 import | 本地源码 | `from ..configs import ColorMode`                                                 |
| 顶层 import | 本地源码 | `from ..utils import get_cv2_rotation`                                            |
| 顶层 import | 本地源码 | `from .configuration_realsense import RealSenseCameraConfig`                      |

#### `lerobot/src/lerobot/motors/__init__.py`
| 时机        | 类别     | import 语句                                                                 |
| ----------- | -------- | --------------------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .motors_bus import Motor, MotorCalibration, MotorNormMode, MotorsBus` |

#### `lerobot/src/lerobot/motors/motors_bus.py`
| 时机        | 类别     | import 语句                                                                       |
| ----------- | -------- | --------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                                                      |
| 顶层 import | 标准库   | `import logging`                                                                  |
| 顶层 import | 标准库   | `import time`                                                                     |
| 顶层 import | 标准库   | `from contextlib import contextmanager`                                           |
| 顶层 import | 标准库   | `from dataclasses import dataclass`                                               |
| 顶层 import | 标准库   | `from enum import Enum`                                                           |
| 顶层 import | 标准库   | `from functools import cached_property`                                           |
| 顶层 import | 标准库   | `from pprint import pformat`                                                      |
| 顶层 import | 标准库   | `from typing import Protocol, TypeAlias`                                          |
| 顶层 import | 第三方库 | `import serial`                                                                   |
| 顶层 import | 第三方库 | `from deepdiff import DeepDiff`                                                   |
| 顶层 import | 第三方库 | `from tqdm import tqdm`                                                           |
| 顶层 import | 本地源码 | `from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError` |
| 顶层 import | 本地源码 | `from lerobot.utils.utils import enter_pressed, move_cursor_up`                   |

#### `lerobot/src/lerobot/motors/feetech/__init__.py`
| 时机        | 类别     | import 语句                                                                   |
| ----------- | -------- | ----------------------------------------------------------------------------- |
| 顶层 import | 本地源码 | `from .feetech import DriveMode, FeetechMotorsBus, OperatingMode, TorqueMode` |
| 顶层 import | 本地源码 | `from .tables import *`                                                       |

#### `lerobot/src/lerobot/motors/feetech/feetech.py`
| 时机              | 类别     | import 语句                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import       | 标准库   | `import logging`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 顶层 import       | 标准库   | `from copy import deepcopy`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 顶层 import       | 标准库   | `from enum import Enum`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 顶层 import       | 标准库   | `from pprint import pformat`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 顶层 import       | 本地源码 | `from lerobot.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude`                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 顶层 import       | 本地源码 | `from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address`                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 顶层 import       | 本地源码 | `from .tables import ( FIRMWARE_MAJOR_VERSION,   # 固件主版本号寄存器 (0, 1) FIRMWARE_MINOR_VERSION,   # 固件次版本号寄存器 (1, 1) MODEL_BAUDRATE_TABLE,     # 电机型号→波特率表映射 MODEL_CONTROL_TABLE,      # 电机型号→控制表映射 MODEL_ENCODING_TABLE,     # 电机型号→编码表映射 MODEL_NUMBER,             # 型号寄存器地址 (3, 2) MODEL_NUMBER_TABLE,       # 型号编号映射 (如 sts3215→777) MODEL_PROTOCOL,           # 电机型号→协议版本映射 MODEL_RESOLUTION,         # 电机分辨率 (一圈脉冲数) SCAN_BAUDRATES,           # 电机扫描时尝试的波特率列表 )` |
| 函数内延迟 import | 本地源码 | `import scservo_sdk as scs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 函数内延迟 import | 本地源码 | `import scservo_sdk as scs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 函数内延迟 import | 本地源码 | `import scservo_sdk as scs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 函数内延迟 import | 本地源码 | `import scservo_sdk as scs`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

#### `lerobot/src/lerobot/motors/feetech/tables.py`
无 import。

#### `lerobot/src/lerobot/datasets/lerobot_dataset.py`
| 时机        | 类别     | import 语句                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import contextlib`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 顶层 import | 标准库   | `import gc`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 顶层 import | 标准库   | `import logging`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 顶层 import | 标准库   | `import shutil`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 顶层 import | 标准库   | `import tempfile`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 顶层 import | 标准库   | `from collections.abc import Callable`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 顶层 import | 标准库   | `from pathlib import Path`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 顶层 import | 第三方库 | `import datasets`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 顶层 import | 第三方库 | `import numpy as np`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 顶层 import | 第三方库 | `import packaging.version`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 顶层 import | 第三方库 | `import pandas as pd`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 顶层 import | 第三方库 | `import PIL.Image`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 顶层 import | 第三方库 | `import torch`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 顶层 import | 第三方库 | `import torch.utils`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 顶层 import | 第三方库 | `from huggingface_hub import HfApi, snapshot_download`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 顶层 import | 第三方库 | `from huggingface_hub.errors import RevisionNotFoundError`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 顶层 import | 本地源码 | `from lerobot.constants import HF_LEROBOT_HOME`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 顶层 import | 本地源码 | `from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 顶层 import | 本地源码 | `from lerobot.datasets.image_writer import AsyncImageWriter, write_image`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 顶层 import | 本地源码 | `from lerobot.datasets.utils import ( DEFAULT_EPISODES_PATH, DEFAULT_FEATURES, DEFAULT_IMAGE_PATH, INFO_PATH, _validate_feature_names, check_delta_timestamps, check_version_compatibility, create_empty_dataset_info, create_lerobot_dataset_card, embed_images, flatten_dict, get_delta_indices, get_hf_dataset_cache_dir, get_hf_dataset_size_in_mb, get_hf_features_from_features, get_parquet_file_size_in_mb, get_parquet_num_frames, get_safe_version, get_video_size_in_mb, hf_transform_to_torch, is_valid_version, load_episodes, load_info, load_nested_dataset, load_stats, load_tasks, to_parquet_with_hf_images, update_chunk_file_indices, validate_episode_buffer, validate_frame, write_info, write_json, write_stats, write_tasks, )` |
| 顶层 import | 本地源码 | `from lerobot.datasets.video_utils import ( VideoFrame, concatenate_video_files, decode_video_frames, encode_video_frames, get_safe_default_codec, get_video_duration_in_s, get_video_info, )`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |

#### `lerobot/src/lerobot/datasets/utils.py`
| 时机        | 类别     | import 语句                                                                                                                      |
| ----------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import contextlib`                                                                                                              |
| 顶层 import | 标准库   | `import importlib.resources`                                                                                                     |
| 顶层 import | 标准库   | `import json`                                                                                                                    |
| 顶层 import | 标准库   | `import logging`                                                                                                                 |
| 顶层 import | 标准库   | `from collections import deque`                                                                                                  |
| 顶层 import | 标准库   | `from collections.abc import Iterable, Iterator`                                                                                 |
| 顶层 import | 标准库   | `from pathlib import Path`                                                                                                       |
| 顶层 import | 标准库   | `from pprint import pformat`                                                                                                     |
| 顶层 import | 标准库   | `from typing import Any, Deque, Generic, TypeVar`                                                                                |
| 顶层 import | 第三方库 | `import datasets`                                                                                                                |
| 顶层 import | 第三方库 | `import numpy as np`                                                                                                             |
| 顶层 import | 第三方库 | `import packaging.version`                                                                                                       |
| 顶层 import | 第三方库 | `import pandas`                                                                                                                  |
| 顶层 import | 第三方库 | `import pandas as pd`                                                                                                            |
| 顶层 import | 第三方库 | `import pyarrow.parquet as pq`                                                                                                   |
| 顶层 import | 第三方库 | `import torch`                                                                                                                   |
| 顶层 import | 第三方库 | `from datasets import Dataset, concatenate_datasets`                                                                             |
| 顶层 import | 第三方库 | `from datasets.table import embed_table_storage`                                                                                 |
| 顶层 import | 第三方库 | `from huggingface_hub import DatasetCard, DatasetCardData, HfApi`                                                                |
| 顶层 import | 第三方库 | `from huggingface_hub.errors import RevisionNotFoundError`                                                                       |
| 顶层 import | 第三方库 | `from PIL import Image as PILImage`                                                                                              |
| 顶层 import | 第三方库 | `from torchvision import transforms`                                                                                             |
| 顶层 import | 本地源码 | `from lerobot.configs.types import FeatureType, PolicyFeature`                                                                   |
| 顶层 import | 本地源码 | `from lerobot.datasets.backward_compatibility import ( FUTURE_MESSAGE, BackwardCompatibilityError, ForwardCompatibilityError, )` |
| 顶层 import | 本地源码 | `from lerobot.utils.utils import is_valid_numpy_dtype_string`                                                                    |

#### `lerobot/src/lerobot/datasets/video_utils.py`
| 时机              | 类别     | import 语句                                               |
| ----------------- | -------- | --------------------------------------------------------- |
| 顶层 import       | 标准库   | `import glob`                                             |
| 顶层 import       | 标准库   | `import importlib`                                        |
| 顶层 import       | 标准库   | `import logging`                                          |
| 顶层 import       | 标准库   | `import shutil`                                           |
| 顶层 import       | 标准库   | `import tempfile`                                         |
| 顶层 import       | 标准库   | `import warnings`                                         |
| 顶层 import       | 标准库   | `from dataclasses import dataclass, field`                |
| 顶层 import       | 标准库   | `from pathlib import Path`                                |
| 顶层 import       | 标准库   | `from threading import Lock`                              |
| 顶层 import       | 标准库   | `from typing import Any, ClassVar`                        |
| 顶层 import       | 第三方库 | `import av`                                               |
| 顶层 import       | 第三方库 | `import fsspec`                                           |
| 顶层 import       | 第三方库 | `import pyarrow as pa`                                    |
| 顶层 import       | 第三方库 | `import torch`                                            |
| 顶层 import       | 第三方库 | `import torchvision`                                      |
| 顶层 import       | 第三方库 | `from datasets.features.features import register_feature` |
| 顶层 import       | 第三方库 | `from PIL import Image`                                   |
| 函数内延迟 import | 第三方库 | `from torchcodec.decoders import VideoDecoder`            |

#### `lerobot/src/lerobot/datasets/image_writer.py`
| 时机        | 类别     | import 语句                |
| ----------- | -------- | -------------------------- |
| 顶层 import | 标准库   | `import multiprocessing`   |
| 顶层 import | 标准库   | `import queue`             |
| 顶层 import | 标准库   | `import threading`         |
| 顶层 import | 标准库   | `from pathlib import Path` |
| 顶层 import | 第三方库 | `import numpy as np`       |
| 顶层 import | 第三方库 | `import PIL.Image`         |
| 顶层 import | 第三方库 | `import torch`             |

#### `lerobot/src/lerobot/datasets/pipeline_features.py`
| 时机        | 类别     | import 语句                                                   |
| ----------- | -------- | ------------------------------------------------------------- |
| 顶层 import | 标准库   | `import re`                                                   |
| 顶层 import | 标准库   | `from collections.abc import Sequence`                        |
| 顶层 import | 标准库   | `from typing import Any`                                      |
| 顶层 import | 本地源码 | `from lerobot.configs.types import PipelineFeatureType`       |
| 顶层 import | 本地源码 | `from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE` |
| 顶层 import | 本地源码 | `from lerobot.datasets.utils import hw_to_dataset_features`   |
| 顶层 import | 本地源码 | `from lerobot.processor import DataProcessorPipeline`         |

#### `lerobot/src/lerobot/datasets/compute_stats.py`
| 时机        | 类别     | import 语句                                              |
| ----------- | -------- | -------------------------------------------------------- |
| 顶层 import | 第三方库 | `import numpy as np`                                     |
| 顶层 import | 本地源码 | `from lerobot.datasets.utils import load_image_as_numpy` |

#### `lerobot/src/lerobot/datasets/backward_compatibility.py`
| 时机        | 类别     | import 语句                |
| ----------- | -------- | -------------------------- |
| 顶层 import | 第三方库 | `import packaging.version` |

#### `lerobot/src/lerobot/datasets/transforms.py`
| 时机        | 类别     | import 语句                                                                           |
| ----------- | -------- | ------------------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import collections`                                                                  |
| 顶层 import | 标准库   | `from collections.abc import Callable, Sequence`                                      |
| 顶层 import | 标准库   | `from dataclasses import dataclass, field`                                            |
| 顶层 import | 标准库   | `from typing import Any`                                                              |
| 顶层 import | 第三方库 | `import torch`                                                                        |
| 顶层 import | 第三方库 | `from torchvision.transforms import v2`                                               |
| 顶层 import | 第三方库 | `from torchvision.transforms.v2 import ( Transform, functional as F,  # noqa: N812 )` |

#### `lerobot/src/lerobot/utils/control_utils.py`
| 时机              | 类别     | import 语句                                                           |
| ----------------- | -------- | --------------------------------------------------------------------- |
| 顶层 import       | 标准库   | `import logging`                                                      |
| 顶层 import       | 标准库   | `import time`                                                         |
| 顶层 import       | 标准库   | `import traceback`                                                    |
| 顶层 import       | 标准库   | `from contextlib import nullcontext`                                  |
| 顶层 import       | 标准库   | `from copy import copy`                                               |
| 顶层 import       | 标准库   | `from functools import cache`                                         |
| 顶层 import       | 标准库   | `from typing import Any`                                              |
| 顶层 import       | 第三方库 | `import numpy as np`                                                  |
| 顶层 import       | 第三方库 | `import torch`                                                        |
| 顶层 import       | 第三方库 | `from deepdiff import DeepDiff`                                       |
| 顶层 import       | 第三方库 | `from termcolor import colored`                                       |
| 顶层 import       | 本地源码 | `from lerobot.datasets.lerobot_dataset import LeRobotDataset`         |
| 顶层 import       | 本地源码 | `from lerobot.datasets.utils import DEFAULT_FEATURES`                 |
| 顶层 import       | 本地源码 | `from lerobot.policies.pretrained import PreTrainedPolicy`            |
| 顶层 import       | 本地源码 | `from lerobot.processor import PolicyAction, PolicyProcessorPipeline` |
| 顶层 import       | 本地源码 | `from lerobot.robots import Robot`                                    |
| 函数内延迟 import | 第三方库 | `import pynput`                                                       |
| 函数内延迟 import | 本地源码 | `from lerobot.constants import ACTION`                                |
| 函数内延迟 import | 第三方库 | `from pynput import keyboard`                                         |

#### `lerobot/src/lerobot/utils/robot_utils.py`
| 时机        | 类别   | import 语句       |
| ----------- | ------ | ----------------- |
| 顶层 import | 标准库 | `import platform` |
| 顶层 import | 标准库 | `import time`     |

#### `lerobot/src/lerobot/utils/utils.py`
| 时机              | 类别     | import 语句                               |
| ----------------- | -------- | ----------------------------------------- |
| 顶层 import       | 标准库   | `import logging`                          |
| 顶层 import       | 标准库   | `import os`                               |
| 顶层 import       | 标准库   | `import os.path as osp`                   |
| 顶层 import       | 标准库   | `import platform`                         |
| 顶层 import       | 标准库   | `import select`                           |
| 顶层 import       | 标准库   | `import subprocess`                       |
| 顶层 import       | 标准库   | `import sys`                              |
| 顶层 import       | 标准库   | `import time`                             |
| 顶层 import       | 标准库   | `from copy import copy, deepcopy`         |
| 顶层 import       | 标准库   | `from datetime import datetime, timezone` |
| 顶层 import       | 标准库   | `from pathlib import Path`                |
| 顶层 import       | 标准库   | `from statistics import mean`             |
| 顶层 import       | 第三方库 | `import numpy as np`                      |
| 顶层 import       | 第三方库 | `import torch`                            |
| 函数内延迟 import | 标准库   | `import gc`                               |
| 函数内延迟 import | 标准库   | `import msvcrt`                           |

#### `lerobot/src/lerobot/utils/visualization_utils.py`
| 时机        | 类别     | import 语句              |
| ----------- | -------- | ------------------------ |
| 顶层 import | 标准库   | `import numbers`         |
| 顶层 import | 标准库   | `import os`              |
| 顶层 import | 标准库   | `from typing import Any` |
| 顶层 import | 第三方库 | `import numpy as np`     |
| 顶层 import | 第三方库 | `import rerun as rr`     |

#### `lerobot/src/lerobot/utils/hub.py`
| 时机        | 类别     | import 语句                                              |
| ----------- | -------- | -------------------------------------------------------- |
| 顶层 import | 标准库   | `import builtins`                                        |
| 顶层 import | 标准库   | `from pathlib import Path`                               |
| 顶层 import | 标准库   | `from tempfile import TemporaryDirectory`                |
| 顶层 import | 标准库   | `from typing import Any, TypeVar`                        |
| 顶层 import | 第三方库 | `from huggingface_hub import HfApi`                      |
| 顶层 import | 第三方库 | `from huggingface_hub.utils import validate_hf_hub_args` |

#### `lerobot/src/lerobot/utils/encoding_utils.py`
无 import。

#### `lerobot/src/lerobot/utils/import_utils.py`
| 时机        | 类别   | import 语句        |
| ----------- | ------ | ------------------ |
| 顶层 import | 标准库 | `import importlib` |
| 顶层 import | 标准库 | `import logging`   |

#### `lerobot/src/lerobot/utils/io_utils.py`
| 时机        | 类别     | import 语句                  |
| ----------- | -------- | ---------------------------- |
| 顶层 import | 标准库   | `import json`                |
| 顶层 import | 标准库   | `import warnings`            |
| 顶层 import | 标准库   | `from pathlib import Path`   |
| 顶层 import | 标准库   | `from typing import TypeVar` |
| 顶层 import | 第三方库 | `import imageio`             |

#### `lerobot/src/lerobot/constants.py`
| 时机        | 类别     | import 语句                                     |
| ----------- | -------- | ----------------------------------------------- |
| 顶层 import | 标准库   | `import os`                                     |
| 顶层 import | 标准库   | `from pathlib import Path`                      |
| 顶层 import | 第三方库 | `from huggingface_hub.constants import HF_HOME` |

#### `lerobot/src/lerobot/errors.py`
无 import。

#### `lerobot/src/lerobot/optim/__init__.py`
| 时机        | 类别     | import 语句                                                  |
| ----------- | -------- | ------------------------------------------------------------ |
| 顶层 import | 本地源码 | `from .optimizers import OptimizerConfig as OptimizerConfig` |

#### `lerobot/src/lerobot/optim/optimizers.py`
| 时机        | 类别     | import 语句                                                                   |
| ----------- | -------- | ----------------------------------------------------------------------------- |
| 顶层 import | 标准库   | `import abc`                                                                  |
| 顶层 import | 标准库   | `from dataclasses import asdict, dataclass, field`                            |
| 顶层 import | 标准库   | `from pathlib import Path`                                                    |
| 顶层 import | 标准库   | `from typing import Any`                                                      |
| 顶层 import | 第三方库 | `import draccus`                                                              |
| 顶层 import | 第三方库 | `import torch`                                                                |
| 顶层 import | 第三方库 | `from safetensors.torch import load_file, save_file`                          |
| 顶层 import | 本地源码 | `from lerobot.constants import ( OPTIMIZER_PARAM_GROUPS, OPTIMIZER_STATE, )`  |
| 顶层 import | 本地源码 | `from lerobot.datasets.utils import flatten_dict, unflatten_dict, write_json` |
| 顶层 import | 本地源码 | `from lerobot.utils.io_utils import deserialize_json_into_object`             |

#### `lerobot/src/lerobot/optim/schedulers.py`
| 时机              | 类别     | import 语句                                                       |
| ----------------- | -------- | ----------------------------------------------------------------- |
| 顶层 import       | 标准库   | `import abc`                                                      |
| 顶层 import       | 标准库   | `import math`                                                     |
| 顶层 import       | 标准库   | `from dataclasses import asdict, dataclass`                       |
| 顶层 import       | 标准库   | `from pathlib import Path`                                        |
| 顶层 import       | 第三方库 | `import draccus`                                                  |
| 顶层 import       | 第三方库 | `from torch.optim import Optimizer`                               |
| 顶层 import       | 第三方库 | `from torch.optim.lr_scheduler import LambdaLR, LRScheduler`      |
| 顶层 import       | 本地源码 | `from lerobot.constants import SCHEDULER_STATE`                   |
| 顶层 import       | 本地源码 | `from lerobot.datasets.utils import write_json`                   |
| 顶层 import       | 本地源码 | `from lerobot.utils.io_utils import deserialize_json_into_object` |
| 函数内延迟 import | 第三方库 | `from diffusers.optimization import get_scheduler`                |

#### `scservo_sdk/__init__.py`
| 时机        | 类别     | import 语句                       |
| ----------- | -------- | --------------------------------- |
| 顶层 import | 本地源码 | `from .port_handler import *`     |
| 顶层 import | 本地源码 | `from .packet_handler import *`   |
| 顶层 import | 本地源码 | `from .group_sync_read import *`  |
| 顶层 import | 本地源码 | `from .group_sync_write import *` |

#### `scservo_sdk/port_handler.py`
| 时机        | 类别     | import 语句       |
| ----------- | -------- | ----------------- |
| 顶层 import | 标准库   | `import time`     |
| 顶层 import | 第三方库 | `import serial`   |
| 顶层 import | 标准库   | `import sys`      |
| 顶层 import | 标准库   | `import platform` |

#### `scservo_sdk/scservo_def.py`
无 import。

#### `scservo_sdk/packet_handler.py`
| 时机        | 类别     | import 语句                              |
| ----------- | -------- | ---------------------------------------- |
| 顶层 import | 本地源码 | `from .scservo_def import *`             |
| 顶层 import | 本地源码 | `from .protocol_packet_handler import *` |

#### `scservo_sdk/protocol_packet_handler.py`
| 时机        | 类别     | import 语句                  |
| ----------- | -------- | ---------------------------- |
| 顶层 import | 本地源码 | `from .scservo_def import *` |

#### `scservo_sdk/group_sync_read.py`
| 时机        | 类别     | import 语句                  |
| ----------- | -------- | ---------------------------- |
| 顶层 import | 本地源码 | `from .scservo_def import *` |

#### `scservo_sdk/group_sync_write.py`
| 时机        | 类别     | import 语句                  |
| ----------- | -------- | ---------------------------- |
| 顶层 import | 本地源码 | `from .scservo_def import *` |

#### `serial/__init__.py`
| 时机        | 类别     | import 语句                                                           |
| ----------- | -------- | --------------------------------------------------------------------- |
| 顶层 import | 标准库   | `from __future__ import absolute_import`                              |
| 顶层 import | 标准库   | `import sys`                                                          |
| 顶层 import | 标准库   | `import importlib`                                                    |
| 顶层 import | 本地源码 | `from serial.serialutil import *`                                     |
| 顶层 import | 本地源码 | `from serial.serialcli import Serial`                                 |
| 顶层 import | 标准库   | `import os`                                                           |
| 顶层 import | 本地源码 | `from serial.serialwin32 import Serial`                               |
| 顶层 import | 本地源码 | `from serial.serialposix import Serial, PosixPollSerial, VTIMESerial` |
| 顶层 import | 本地源码 | `from serial.serialjava import Serial`                                |

#### `serial/serialposix.py`
| 时机              | 类别     | import 语句                                                                                                                |
| ----------------- | -------- | -------------------------------------------------------------------------------------------------------------------------- |
| 顶层 import       | 标准库   | `from __future__ import absolute_import`                                                                                   |
| 顶层 import       | 标准库   | `import errno`                                                                                                             |
| 顶层 import       | 标准库   | `import fcntl`                                                                                                             |
| 顶层 import       | 标准库   | `import os`                                                                                                                |
| 顶层 import       | 标准库   | `import select`                                                                                                            |
| 顶层 import       | 标准库   | `import struct`                                                                                                            |
| 顶层 import       | 标准库   | `import sys`                                                                                                               |
| 顶层 import       | 标准库   | `import termios`                                                                                                           |
| 顶层 import       | 本地源码 | `import serial`                                                                                                            |
| 顶层 import       | 本地源码 | `from serial.serialutil import SerialBase, SerialException, to_bytes, \ PortNotOpenError, SerialTimeoutException, Timeout` |
| 顶层 import       | 标准库   | `import array`                                                                                                             |
| 顶层 import       | 标准库   | `import array`                                                                                                             |
| 函数内延迟 import | 标准库   | `import warnings`                                                                                                          |

#### `serial/serialutil.py`
| 时机              | 类别   | import 语句                              |
| ----------------- | ------ | ---------------------------------------- |
| 顶层 import       | 标准库 | `from __future__ import absolute_import` |
| 顶层 import       | 标准库 | `import io`                              |
| 顶层 import       | 标准库 | `import time`                            |
| 函数内延迟 import | 标准库 | `import array`                           |
| 顶层 import       | 标准库 | `import sys`                             |

统计：97 个文件，741 条 import/from-import 语句。
