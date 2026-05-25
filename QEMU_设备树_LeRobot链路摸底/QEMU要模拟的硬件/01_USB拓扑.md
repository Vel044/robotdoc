# 01 USB 拓扑

## 目标

这个文件只讲硬件链路：一部分来自设备树，一部分来自 USB 运行时枚举。

```text
设备树证据：
/树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.from-dtb.dts

USB 树证据：
/树莓派5真机文件快照/2026-05-18/hardware/lsusb-tree.txt
```

QEMU 如果要模拟当前 Pi 5 + LeRobot 的硬件环境，第一层看设备树里实际启用的板级硬件，第二层看 USB host 工作之后枚举出来的 USB 设备。这个文件只保留硬件链路，用户态依赖放到 LeRobot 运行验证文档里。

## 一、设备树

### 设备树是怎么从树莓派读出来的

这里的设备树证据不是从 Linux 源码里直接抄来的，而是从树莓派真机上读出来后保存到快照目录的。用了两条口径：

```text
1. 读 boot 分区里的 dtb 文件：
   /boot/firmware/bcm2712-rpi-5-b.dtb

2. 读运行中 kernel 暴露出来的设备树：
   /proc/device-tree/
```

第一条口径拿到的是启动分区里的 DTB 文件。它是 Raspberry Pi firmware 启动 kernel 时使用的设备树二进制文件之一：

```bash
scp RaspberryPi:/boot/firmware/bcm2712-rpi-5-b.dtb \
  树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.dtb
```

DTB 是二进制，人不方便直接看，所以再用 `dtc` 反编译成 DTS 文本：

```bash
dtc -I dtb -O dts \
  -o 树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.from-dtb.dts \
  树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.dtb
```

所以这个文件：

```text
树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.from-dtb.dts
```

本质上就是：

```text
树莓派 boot 分区里的 bcm2712-rpi-5-b.dtb
  -> 用 dtc 反编译
  -> 得到方便阅读的 bcm2712-rpi-5-b.from-dtb.dts
```

第二条口径是读 `/proc/device-tree/`。这是当前正在运行的 kernel 解析完设备树以后，对外暴露出来的设备树目录。比如可以这样读关键属性：

```bash
ssh RaspberryPi 'tr -d "\0" < /proc/device-tree/model'
ssh RaspberryPi 'tr "\0" "\n" < /proc/device-tree/compatible'
ssh RaspberryPi 'tr -d "\0" < /proc/device-tree/aliases/usb0'
ssh RaspberryPi 'tr -d "\0" < /proc/device-tree/aliases/usb1'
```

### DTS 和真实运行时设备树的区别

`bcm2712-rpi-5-b.from-dtb.dts` 和 `/proc/device-tree/` 都是设备树证据，但位置不同：

| 证据                           | 含义                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| `bcm2712-rpi-5-b.from-dtb.dts` | boot 分区里的 `bcm2712-rpi-5-b.dtb` 反编译结果，偏启动输入。 |
| `/proc/device-tree/`           | kernel 启动后实际解析到的设备树，偏运行时结果。              |

两者在 USB 主链路上是一致的：

| 节点 / 属性             | boot DTB                                | `/proc/device-tree`                     |
| ----------------------- | --------------------------------------- | --------------------------------------- |
| `/ compatible`          | `raspberrypi,5-model-b`、`brcm,bcm2712` | `raspberrypi,5-model-b`、`brcm,bcm2712` |
| `/aliases/usb0`         | `/axi/pcie@1000120000/rp1/usb@200000`   | `/axi/pcie@1000120000/rp1/usb@200000`   |
| `/aliases/usb1`         | `/axi/pcie@1000120000/rp1/usb@300000`   | `/axi/pcie@1000120000/rp1/usb@300000`   |
| `usb@200000 compatible` | `snps,dwc3`                             | `snps,dwc3`                             |
| `usb@200000 dr_mode`    | `host`                                  | `host`                                  |
| `usb@300000 compatible` | `snps,dwc3`                             | `snps,dwc3`                             |
| `usb@300000 dr_mode`    | `host`                                  | `host`                                  |

主要区别在运行时信息上：

| 项                | boot DTB           | `/proc/device-tree`                                                                                    |
| ----------------- | ------------------ | ------------------------------------------------------------------------------------------------------ |
| `model`           | `Raspberry Pi 5`   | `Raspberry Pi 5 Model B Rev 1.1`                                                                       |
| `chosen/bootargs` | 基础 bootargs      | 包含 `console=ttyAMA10,115200`、`root=PARTUUID=2428fd84-02`、`rootwait` 等最终启动参数。               |
| 运行时补充        | 不含部分运行时字段 | 多出 `serial-number`、`memreserve`、`chosen/bootloader`、`chosen/power`、`linux,initrd-start/end` 等。 |

### 设备树描述的是静态板级硬件

设备树描述的是静态板级硬件：

```text
BCM2712                         SoC
  -> AXI                        片上总线
  -> PCIe pcie@1000120000       PCIe
  -> RP1                        IO芯片
  -> RP1 DWC3 USB host          DWC3 硬件；Linux 中由 xhci-hcd 接管成 xHCI host。
```

### 设备树摘录

下面摘自真机启动 DTB 反编译后的 `bcm2712-rpi-5-b.from-dtb.dts`，只保留和 USB 链路直接有关的节点：

```text
/ {                                                        // 根节点：先识别板子和 SoC。
    compatible = "raspberrypi,5-model-b", "brcm,bcm2712";  // Pi 5 B + BCM2712。
    model = "Raspberry Pi 5";                              // boot DTB 里的板卡名。

    axi {                                                  // BCM2712 片上 AXI 总线。
        compatible = "simple-bus";                         // 子节点按地址展开。
        #address-cells = <0x02>;                           // 地址用 2 个 cell。
        #size-cells = <0x02>;                              // 长度也用 2 个 cell。

        pcie@1000120000 {                                  // BCM2712 PCIe；RP1 挂在后面。
            compatible = "brcm,bcm2712-pcie";              // 匹配 BCM2712 PCIe 驱动。
            reg = <0x10 0x120000 0x00 0x9310>;             // MMIO 地址 0x1000120000，长度 0x9310。
            device_type = "pci";                           // 这是一条 PCI/PCIe 总线。
            status = "okay";                               // 启用这个 PCIe 节点。

            rp1 {                                          // Pi 5 的 RP1 I/O 芯片。
                compatible = "simple-bus";                 // RP1 内部外设继续按地址展开。
                ranges = <0xc0 0x40000000 ...>;            // RP1 MMIO 空间基准。

                usb@200000 {                               // RP1 的第一个 DWC3 USB host。
                    reg = <0xc0 0x40200000 0x00 0x100000>; // RP1 基准 + 0x200000。
                    compatible = "snps,dwc3";              // Synopsys DWC3 USB 控制器。
                    dr_mode = "host";                      // 主机模式，能枚举外接 USB 设备。
                    status = "okay";                       // 启用这个 USB host。
                };

                usb@300000 {                               // RP1 的第二个 DWC3 USB host。
                    reg = <0xc0 0x40300000 0x00 0x100000>; // RP1 基准 + 0x300000。
                    compatible = "snps,dwc3";              // 同样是 DWC3。
                    dr_mode = "host";                      // 同样是 USB host。
                    status = "okay";                       // 启用这个 USB host。
                };
            };
        };
    };
};

// DTS 里没有 xhci-hcd、UVC 摄像头、CDC ACM 串口板、HID 键盘节点。
// DWC3 host 初始化后，在 Linux USB 栈里表现成 xHCI/root hub；
// 具体 USB 外设是 host 工作后运行时枚举出来的。
```

文件后面的 alias / symbol 也把这两个 USB host 明确指到 RP1 后面：

```text
usb0 = "/axi/pcie@1000120000/rp1/usb@200000";
usb1 = "/axi/pcie@1000120000/rp1/usb@300000";
rp1_usb0 = "/axi/pcie@1000120000/rp1/usb@200000";
rp1_usb1 = "/axi/pcie@1000120000/rp1/usb@300000";
```

也就是说，QEMU/新 kernel 不能只从 USB root hub 开始看。对 Pi 5 真硬件来说，USB host 前面还有一段必须成立的板级链路：

```text
BCM2712 PCIe 控制器
  -> RP1 I/O 芯片
  -> RP1 内部 DWC3 USB host
  -> Linux xhci-hcd root hub
```

这些字段可以这样读：

| 字段                        | 意思                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `/axi`                      | 设备树里的片上总线路径。AXI 是 ARM/SoC 里常见的片上互连总线，表示这些外设从 BCM2712 的内部地址空间能访问到。        |
| `pcie@1000120000`           | 设备树节点名。`pcie` 表示这是 PCIe 控制器，`@1000120000` 是 unit-address，用来标识这个节点对应的硬件地址/地址区域。 |
| `compatible`                | 驱动匹配字符串。kernel 用它决定该用哪个驱动处理这个节点。                                                           |
| `brcm`                      | Broadcom 的 vendor 前缀。`brcm,bcm2712-pcie` 表示 BCM2712 上的 PCIe 控制器。                                        |
| `snps`                      | Synopsys 的 vendor 前缀。`snps,dwc3` 表示 Synopsys DesignWare USB3 Controller。                                     |
| `reg = <...>`               | 这个设备的寄存器地址和大小。kernel 驱动会按这里的信息映射 MMIO 寄存器。                                             |
| `status = "okay"`           | 这个硬件节点启用。若是 `disabled`，kernel 一般不会初始化它。                                                        |
| `dr_mode = "host"`          | DWC3 工作在 USB host 模式，也就是主机模式，可以枚举摄像头、串口板、键盘等外设。                                     |
| `rp1`                       | Raspberry Pi 5 的 RP1 I/O 芯片节点。它挂在 BCM2712 的 PCIe 后面。                                                   |
| `usb@200000` / `usb@300000` | RP1 里的两个 USB 控制器节点，`@200000`、`@300000` 是相对这个 RP1 外设空间的节点地址标识。                           |

## 二、USB 拓扑

### USB 树

所以后面的 USB 树是这样导出来的：

```bash
ssh RaspberryPi 'lsusb -t' \
  > 树莓派5真机文件快照/2026-05-18/hardware/lsusb-tree.txt
```

### USB 运行时枚举

USB 设备不是设备树里的固定节点，而是 USB host 工作后运行时枚举出来的：

```text
xhci-hcd root hub
  -> USB 2.0 root hub
      -> 键盘，Driver=usbhid
  -> USB 2.0 root hub
      -> 外接 USB 2.1 Hub
          -> Port 1: 第一个 UVC 摄像头，Driver=uvcvideo
          -> Port 2: 第二个 UVC 摄像头，Driver=uvcvideo
          -> Port 3: 第一个 SO101 串口板，Driver=cdc_acm
          -> Port 4: 第二个 SO101 串口板，Driver=cdc_acm
```

合起来就是：

```text
BCM2712 -> AXI -> PCIe -> RP1 -> DWC3 USB host
  -> xhci-hcd root hub
  -> USB hub
  -> UVC camera / CDC ACM serial / HID keyboard
```

## QEMU 或新 kernel 要面对的硬件层

| 硬件层            | 证据                                                                       | QEMU / 新 kernel 要做到什么                                |
| ----------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| BCM2712 PCIe      | 设备树有 `/axi/pcie@1000120000`，`compatible = "brcm,bcm2712-pcie"`        | 让系统能从 BCM2712 走到挂在 PCIe 后面的 RP1。              |
| RP1               | 设备树里 `rp1` 是 `pcie@1000120000` 的子节点                               | 呈现 Pi 5 的 I/O 芯片入口，后面的 USB host 都在 RP1 下面。 |
| RP1 DWC3 USB host | `usb@200000`、`usb@300000`，`compatible = "snps,dwc3"`，`dr_mode = "host"` | 提供 USB host 控制器，让 Linux 能初始化主机模式 USB。      |
| xHCI root hub     | `lsusb -t` 里 root hub 的 `Driver=xhci-hcd`                                | 让 USB host 在 Linux 里表现成 xHCI root hub。              |
| USB hub           | `lsusb -t` 里外接 Hub 的 `Driver=hub/4p`                                   | 支持 Hub 继续展开下游端口。                                |
| UVC camera        | `Class=Video, Driver=uvcvideo`                                             | 让两个 USB 摄像头按 UVC 设备枚举出来。                     |
| CDC ACM serial    | `Class=Communications/CDC Data, Driver=cdc_acm`                            | 让两块 SO101 串口板按 CDC ACM 设备枚举出来。               |
| HID keyboard      | `Class=Human Interface Device, Driver=usbhid`                              | 支持真机键盘输入，方便登录、shell 操作和现场调试。         |

## 一句话结论

```text
QEMU 要模拟的是让 guest kernel 看到：
BCM2712/RP1/DWC3/xHCI 这一条 USB host 硬件链，
再让 USB 树里枚举出 UVC 摄像头、CDC ACM 串口板和 HID 键盘。
```
