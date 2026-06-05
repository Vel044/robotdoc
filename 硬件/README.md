# 树莓派 5 硬件链路资料整理

本目录集中保存树莓派 5 上 **BCM2712 SoC、RP1 I/O 控制器、PCIe、USB/xHCI** 相关资料。用途是支撑后续写设备树链路、QEMU 硬件建模、USB 摄像头/舵机链路图和答辩 PPT。

当前最关心的主链路是：

```text
BCM2712 SoC
  -> BCM2712 片上总线 / AXI 视图
  -> BCM2712 PCIe host controller
  -> PCIe 2.0 x4
  -> RP1 PCIe endpoint
  -> RP1 内部 AXI bus fabric
  -> USB XHCI Host 0 / USB XHCI Host 1
  -> xhci-hcd
  -> USB 摄像头 / USB 串口舵机链路
```

设备树里的简化对应关系是：

```text
/                                  根节点，描述整块 Raspberry Pi 5 板级硬件
└─ /axi                            BCM2712 给 Linux 呈现的片上总线视图
   └─ pcie@1000120000              BCM2712 PCIe host controller
      └─ RP1                       PCIe 后面的 RP1 I/O 控制器
         ├─ usb@200000             RP1 USB host 0，最终进入 xHCI
         └─ usb@300000             RP1 USB host 1，最终进入 xHCI
```

注意：`usb@200000` 和 `usb@300000` 里的地址是 RP1 侧寄存器窗口/偏移意义上的地址，不是普通内存数组地址。Linux 通过 PCIe BAR/地址映射访问这些寄存器，最终 CPU 的 MMIO store/load 会被 PCIe host controller 转成 PCIe 事务，到达 RP1 内部外设寄存器。

---

## 1. 图片索引

### 1.1 官方 RP1 系统框图

本地文件：[RP1_System_Diagram_官方PDF.png](./图片/RP1_System_Diagram_官方PDF.png)

来源：[RP1 Peripherals datasheet](./资料/RP1外设数据手册_rp1-peripherals.pdf) 第 2 章 Figure 2。

用途：

- 最权威，适合作为硬件链路的依据。
- 左下能看到 `PCIe x4 PHY` 和 `PCIe 2.0 EP`。
- 右侧能看到 `USB XHCI Host 0` 和 `USB XHCI Host 1`。
- 中间能看到 RP1 内部 `AXI Arbiter / bus fabric`。

不足：

- 图太细，不适合直接放 PPT。
- 不包含 Linux 设备树节点名，例如 `/axi`、`pcie@1000120000`、`usb@200000`。

### 1.2 CNX 的 RP1 block diagram

本地文件：[RP1_Block_Diagram_CNX.png](./图片/RP1_Block_Diagram_CNX.png)

来源：[CNX RP1 datasheet/block diagram 文章](./资料/CNX_RP1数据手册与框图文章.html)

用途：

- 布局更干净，适合参考绘制自己的示意图。
- 中央 `Bus Fabric` 很适合作为 RP1 内部总线抽象。
- 左侧直接标出 `USB XHCI Host 0/1` 和对应 USB2/USB3 PHY。

不足：

- 第三方转载/整理图，不如官方 PDF 权威。
- 仍然没有设备树节点和 Linux 驱动层。

### 1.3 Phoronix 的树莓派 5 I/O 总览图

本地文件：[RaspberryPi5_RP1_IO_Overview_Phoronix.png](./图片/RaspberryPi5_RP1_IO_Overview_Phoronix.png)

来源：[Phoronix RP1 Linux RFC 文章](./资料/Phoronix_RP1_Linux_RFC文章.html)

用途：

- 适合讲大方向：`Host CPU BCM2712 -> PCIe Gen2 x4 -> RP1 -> USB/Ethernet/GPIO/CSI/DSI`。
- 可以作为 PPT 第一张“板级硬件拆分”的参考。

不足：

- 太高层，不适合解释 `usb@200000`、`usb@300000` 和 `xhci-hcd`。

### 1.4 官方 RP1 芯片照片

本地文件：[RP1_Southbridge_Photo_官方PDF.png](./图片/RP1_Southbridge_Photo_官方PDF.png)

来源：[RP1 Peripherals datasheet](./资料/RP1外设数据手册_rp1-peripherals.pdf) 第 1 章 Figure 1。

用途：

- 适合 PPT 里说明 RP1 是板上的真实 I/O 芯片，不是 Linux 里的抽象名。
- 不适合解释链路，只适合做“实物定位”。

---

## 2. 文档索引

### 2.1 RP1 官方外设数据手册

本地文件：[RP1外设数据手册_rp1-peripherals.pdf](./资料/RP1外设数据手册_rp1-peripherals.pdf)

用途：

- RP1 最核心资料。
- 第 1 章说明 RP1 通过 PCIe 2.0 x4 连接 BCM2712。
- 第 2 章给出 RP1 system diagram。
- 第 5 章说明 USB Host subsystem 基于 Synopsys `dwc_usb3`，有两个 USB 3.0 xHCI Host 实例。
- 第 6 章说明 RP1 内部功能通过 PCI Express endpoint controller 与 AP 通信。

### 2.2 Raspberry Pi 官方 RP1 I/O controller 文档源码

本地文件：

- [RaspberryPi官方_RP1_IO_controller.adoc](./资料/RaspberryPi官方_RP1_IO_controller.adoc)
- [RaspberryPi官方_IO_controllers入口.adoc](./资料/RaspberryPi官方_IO_controllers入口.adoc)

用途：

- 官方网页文档的 AsciiDoc 源码，本地可读，避免网页 Cloudflare 阻挡。
- 适合引用“RP1 是 Raspberry Pi 5/CM5 内置 I/O controller，负责 USB、Ethernet、GPIO、storage 等外设”。
- 明确 RP1 通过 PCIe 2.0 x4 连接 BCM2712。

### 2.3 Raspberry Pi 官方 BCM2712 处理器文档源码

本地文件：[RaspberryPi官方_BCM2712处理器.adoc](./资料/RaspberryPi官方_BCM2712处理器.adoc)

用途：

- 说明 BCM2712 是树莓派 5 的主 SoC。
- 说明 BCM2712 上的 x4 PCIe 在树莓派 5 上用于连接 RP1 south bridge。
- 适合解释为什么链路开头是 BCM2712，而不是 RP1。

### 2.4 树莓派 5 产品简介

本地文件：[树莓派5产品简介_raspberry-pi-5-product-brief.pdf](./资料/树莓派5产品简介_raspberry-pi-5-product-brief.pdf)

用途：

- 适合写论文/答辩里的硬件平台概述。
- 不适合深挖设备树或 xHCI。

### 2.5 第三方资料

本地文件：

- [CNX_RP1数据手册与框图文章.html](./资料/CNX_RP1数据手册与框图文章.html)
- [Phoronix_RP1_Linux_RFC文章.html](./资料/Phoronix_RP1_Linux_RFC文章.html)
- [DeepWiki_BCM2712_SoC资料.html](./资料/DeepWiki_BCM2712_SoC资料.html)

用途：

- CNX：方便看简化版 RP1 block diagram。
- Phoronix：方便看 Linux/RP1 upstream 背景和高层 I/O 图。
- DeepWiki：方便从 Linux 源码视角理解 BCM2712、PCIe、地址空间和设备树相关文件。

---

## 3. 这几张图和你当前 PPT 图的关系

你当前画的这条链：

```text
BCM2712 SoC
  -> /axi
  -> pcie@1000120000
  -> RP1 I/O 芯片
  -> usb@200000 / usb@300000
  -> xHCI
```

网上没有一张现成图完整表达这条链，因为它跨了三层：

1. **板级硬件层**：BCM2712 和 RP1 是两颗芯片，通过 PCIe 2.0 x4 相连。
2. **RP1 内部硬件层**：RP1 里有 PCIe endpoint、AXI bus fabric、USB XHCI Host 0/1。
3. **Linux 设备树/驱动层**：`/axi`、`pcie@1000120000`、`usb@200000`、`usb@300000`、`xhci-hcd` 是 Linux 对硬件的命名和绑定结果。

因此建议画图时这样处理：

- **第一张总览图**：保留你自己的链路图，讲清楚设备树路径和 Linux 驱动绑定。
- **第二张 RP1 内部图**：参考 [RP1_Block_Diagram_CNX.png](./图片/RP1_Block_Diagram_CNX.png)，把 RP1 内部画成 `PCIe EP -> Bus Fabric -> USB XHCI Host 0/1`。
- **第三张证据图**：引用 [RP1_System_Diagram_官方PDF.png](./图片/RP1_System_Diagram_官方PDF.png)，说明这些模块确实在 RP1 里。

---

## 4. 后续文档建议

后续如果继续拆 `bcm2712-rpi-5-b.dtb` 和 Linux 驱动链路，可以按下面顺序写：

1. **设备树入口**：`/boot/firmware/bcm2712-rpi-5-b.dtb` 反编译成 DTS，定位 `/axi`、`pcie@1000120000`、RP1 子节点。
2. **PCIe 枚举**：说明 BCM2712 PCIe host 如何发现 RP1 endpoint，BAR 如何把 RP1 寄存器窗口映射给 CPU。
3. **USB 节点绑定**：说明 `usb@200000`、`usb@300000` 如何通过 `compatible = "snps,dwc3"` 进入 DWC3/xHCI 链路。
4. **xHCI roothub**：说明 `xhci-hcd` 创建 roothub，USB 摄像头、USB 串口设备如何挂在 roothub 下。
5. **MMIO 语义**：单独讲普通内存写、DMA 内存写、MMIO 寄存器写的区别，以及 CPU store 如何经过 PCIe 到达 RP1 寄存器。

已有相关文档可以继续对接：

- [QEMU要模拟的硬件/01_USB拓扑.md](../QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/01_USB拓扑.md)
- [QEMU要模拟的硬件/03_BCM2712与设备树.md](../QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/03_BCM2712与设备树.md)
- [QEMU要模拟的硬件/04_RP1_USB摄像头舵机链路.md](../QEMU_设备树_LeRobot链路摸底/QEMU要模拟的硬件/04_RP1_USB摄像头舵机链路.md)

---

## 5. 在线来源

- RP1 datasheet: <https://datasheets.raspberrypi.com/rp1/rp1-peripherals.pdf>
- Raspberry Pi 5 product brief: <https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf>
- Raspberry Pi I/O controllers: <https://www.raspberrypi.com/documentation/computers/io-controllers.html>
- Raspberry Pi documentation GitHub: <https://github.com/raspberrypi/documentation>
- CNX RP1 block diagram article: <https://www.cnx-software.com/2023/10/07/raspberry-pi-rp1-datasheet-block-diagram/>
- Phoronix RP1 Linux RFC article: <https://www.phoronix.com/news/Raspberry-Pi-5-RP1-Linux-RFC>
- DeepWiki BCM2712 notes: <https://deepwiki.com/raspberrypi/linux/2.3-bcm2712-soc-%28raspberry-pi-5%29>
