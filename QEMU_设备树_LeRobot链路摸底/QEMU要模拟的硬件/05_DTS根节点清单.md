# 05 DTS 根节点清单

## 数据来源

这个文件列真机 `bcm2712-rpi-5-b.dtb` 的根节点，并解释每个节点大概是什么。后面先做第一轮排除法，用来缩小后续分析范围：

```text
新 kernel 必须初始化 / QEMU 必须模拟 / 可以暂时忽略
```

数据来自：

```text
../树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.dtb
../树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.from-dtb.dts
```

根节点顺序按下面命令读出的 DTB 原始顺序：

```bash
fdtget -l ../树莓派5真机文件快照/2026-05-18/boot/bcm2712-rpi-5-b.dtb /
```

当前 DTB 根节点共 35 个。

## 根节点清单

| 根节点 | 大致含义 | 和新 kernel / QEMU 的关系 | 后续排除法备注 |
| --- | --- | --- | --- |
| `clocks` | 板级固定时钟和时钟源集合，比如 UART、USB、SDHCI 等基础时钟。 | 后续判断哪些时钟会影响启动、串口、存储和 USB 初始化。 | 待判断 |
| `cpus` | CPU 拓扑，描述 4 个 Cortex-A76 CPU 核。 | 后续判断新 kernel 如何识别主核和其他 CPU。 | 待判断 |
| `psci` | ARM PSCI 固件调用接口，通常用于 CPU on/off、重启等。 | 后续判断新 kernel 是否要通过 PSCI 管理多核和重启。 | 待判断 |
| `reserved-memory` | 固件、ATF、CMA 等保留内存区域。 | 后续判断内存分配器必须避开的地址范围。 | 待判断 |
| `soc@107c000000` | BCM2712 SoC 片上外设区域之一。 | 后续判断哪些 BCM2712 片上外设需要初始化或模拟。 | 待判断 |
| `axi` | BCM2712 片上 AXI 总线，PCIe/RP1 链路在这里展开。 | 后续判断 PCIe、RP1、DWC3 USB host 的板级入口。 | 待判断 |
| `timer` | ARMv8 generic timer。 | 后续判断调度 tick、sleep、timeout 依赖的时间源。 | 待判断 |
| `clk-27M` | 27 MHz 固定时钟。 | 后续判断哪些外设引用这个固定时钟。 | 待判断 |
| `clk-108M` | 108 MHz 固定时钟。 | 后续判断哪些外设引用这个固定时钟。 | 待判断 |
| `hvs@107c580000` | 硬件视频/显示合成相关节点。 | 后续判断显示链路是否需要在 QEMU 或新 kernel 阶段处理。 | 待判断 |
| `arm-pmu` | ARM Performance Monitor Unit，CPU 性能计数器。 | 后续判断是否需要支持性能计数和 profiling。 | 待判断 |
| `thermal-zones` | 温度区域和散热策略描述。 | 后续判断温控、降频和风扇策略是否需要处理。 | 待判断 |
| `firmwarekms` | Raspberry Pi firmware KMS / 显示相关节点。 | 后续判断显示相关功能是否进入当前阶段范围。 | 待判断 |
| `phy` | PHY 相关节点，用于部分高速外设的物理层配置。 | 后续判断 USB/PCIe/显示等链路是否引用它。 | 待判断 |
| `memory@0` | 物理内存描述，启动时由 bootloader/firmware 填充或修正。 | 后续判断新 kernel 如何建立物理内存管理。 | 待判断 |
| `leds` | 板载 LED 节点。 | 后续判断状态灯是否需要支持。 | 待判断 |
| `sd-io-1v8-reg` | SD/IO 相关 1.8V regulator。 | 后续判断 SD/IO 存储或外设供电是否依赖它。 | 待判断 |
| `sd-vcc-reg` | SD 卡供电 regulator。 | 后续判断 rootfs 所在存储链路是否依赖它。 | 待判断 |
| `wl-on-reg` | 无线模块上电相关 regulator。 | 后续判断 Wi-Fi/蓝牙是否进入当前阶段范围。 | 待判断 |
| `cam1_clk` | 摄像头 1 相关时钟。 | 后续判断 CSI 摄像头链路是否需要处理。 | 待判断 |
| `cam0_clk` | 摄像头 0 相关时钟。 | 后续判断 CSI 摄像头链路是否需要处理。 | 待判断 |
| `cam0_reg` | 摄像头 0 相关供电 regulator。 | 后续判断 CSI 摄像头供电是否需要处理。 | 待判断 |
| `cam1_reg` | 摄像头 1 相关供电 regulator。 | 后续判断 CSI 摄像头供电是否需要处理。 | 待判断 |
| `cam_dummy_reg` | 摄像头相关 dummy regulator。 | 后续判断摄像头节点里的占位供电是否影响驱动。 | 待判断 |
| `dummy` | 通用 dummy regulator 或占位节点。 | 后续判断哪些外设只是引用占位供电。 | 待判断 |
| `i2c0if` | I2C0 接口相关节点。 | 后续判断板上 I2C 设备是否进入当前阶段范围。 | 待判断 |
| `i2c0mux` | I2C0 mux 相关节点。 | 后续判断 I2C 分路和摄像头/显示检测是否需要处理。 | 待判断 |
| `rp1_firmware` | RP1 firmware 相关节点。 | 后续判断 RP1 初始化是否依赖 firmware/mailbox 交互。 | 待判断 |
| `rp1_vdd_3v3` | RP1 3.3V 供电相关节点。 | 后续判断 RP1 外设供电依赖。 | 待判断 |
| `chosen` | kernel 启动参数、console、initramfs 等启动信息。 | 后续判断 cmdline、stdout-path、initrd 信息如何传给 kernel。 | 待判断 |
| `aliases` | 设备树别名，比如 `usb0`、`usb1` 指向 RP1 USB host。 | 后续判断稳定路径和节点引用如何对应真实硬件链路。 | 待判断 |
| `__overrides__` | overlay 参数覆盖表。 | 后续判断是否需要支持 overlay 参数或只使用最终 DTB。 | 待判断 |
| `cooling_fan` | 散热风扇节点。 | 后续判断温控/风扇是否进入当前阶段范围。 | 待判断 |
| `pwr_button` | 电源按钮节点。 | 后续判断电源键事件是否需要支持。 | 待判断 |
| `__symbols__` | 设备树 label 到节点路径的符号表。 | 后续判断调试、overlay 或引用追踪是否需要使用它。 | 待判断 |

## 第一轮排除法

这一轮只按当前目标筛选：

```text
目标：搞清楚新 kernel 启动、QEMU 模拟、LeRobot USB 链路最先要面对什么。
```

所以先不追显示、CSI 摄像头、无线、LED、风扇、电源键等旁路功能。排除法结果是：

```text
根节点总数：35
第一阶段先看：12
第一阶段后放：18
辅助信息节点：5
```

### 第一阶段先看的 12 个

| 根节点 | 为什么先看 |
| --- | --- |
| `chosen` | 拿 bootargs、stdout-path、initramfs 信息；启动链必须看。 |
| `memory@0` | 建立物理内存管理必须看。 |
| `reserved-memory` | 内存分配前必须知道哪些区域不能碰。 |
| `cpus` | CPU 拓扑和主核/多核启动必须看。 |
| `psci` | 多核启动、关机、重启依赖 PSCI。 |
| `timer` | 调度、sleep、timeout 依赖 ARM generic timer。 |
| `clocks` | UART、存储、USB 等外设可能引用时钟，先保留。 |
| `soc@107c000000` | BCM2712 片上外设区域，属于板级基础结构。 |
| `axi` | PCIe -> RP1 -> DWC3 USB host 的主链路入口。 |
| `rp1_firmware` | RP1 相关 firmware 交互，可能影响 RP1 初始化。 |
| `rp1_vdd_3v3` | RP1 供电节点，可能影响 RP1 外设可用性。 |
| `aliases` | `usb0`、`usb1` 等稳定别名会帮助定位 RP1 USB host。 |

### 第一阶段后放的 18 个

| 根节点 | 为什么后放 |
| --- | --- |
| `clk-27M` | 固定时钟，先通过引用关系间接看；不单独展开。 |
| `clk-108M` | 固定时钟，先通过引用关系间接看；不单独展开。 |
| `hvs@107c580000` | 显示合成相关；不是 LeRobot USB 主链路。 |
| `arm-pmu` | 性能计数器；不影响先启动和 USB 外设出现。 |
| `thermal-zones` | 温控策略；第一阶段先不做温控闭环。 |
| `firmwarekms` | 显示/KMS 相关；不是当前主链路。 |
| `phy` | PHY 节点先只看是否被 USB/PCIe 引用，不单独展开。 |
| `leds` | 状态灯；不影响 LeRobot 主负载。 |
| `sd-io-1v8-reg` | SD/IO regulator，先等确认 rootfs 存储链路后再看。 |
| `sd-vcc-reg` | SD 卡 regulator，先等确认 rootfs 存储链路后再看。 |
| `wl-on-reg` | Wi-Fi/蓝牙供电；不是当前 USB 机器人链路。 |
| `cam1_clk` | CSI 摄像头时钟；当前摄像头走 USB UVC。 |
| `cam0_clk` | CSI 摄像头时钟；当前摄像头走 USB UVC。 |
| `cam0_reg` | CSI 摄像头供电；当前摄像头走 USB UVC。 |
| `cam1_reg` | CSI 摄像头供电；当前摄像头走 USB UVC。 |
| `cam_dummy_reg` | CSI 摄像头占位供电；当前摄像头走 USB UVC。 |
| `i2c0if` | I2C 接口；当前 LeRobot 主链路走 USB。 |
| `i2c0mux` | I2C mux；当前 LeRobot 主链路走 USB。 |

### 辅助信息节点 5 个

| 根节点 | 为什么算辅助信息 |
| --- | --- |
| `__overrides__` | overlay 参数表，不是硬件本体。 |
| `__symbols__` | label 到路径的符号表，不是硬件本体。 |
| `dummy` | 占位 regulator，通常用于满足引用关系。 |
| `cooling_fan` | 风扇设备节点，后续和温控一起看。 |
| `pwr_button` | 电源键输入节点，后续和电源管理一起看。 |

第一轮排除后，后续重点先围绕这条链继续展开：

```text
chosen / memory / reserved-memory / cpus / psci / timer
  -> soc / axi
  -> pcie@1000120000
  -> rp1
  -> rp1 usb@200000 / usb@300000
  -> xhci root hub
  -> USB hub / UVC camera / CDC ACM serial
```
