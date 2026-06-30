# PFTF RPi4 UEFI 在 QEMU raspi4b 下失败定位

本文记录一次排查：目标是让 `QEMU raspi4b` 启动官方 PFTF/RPi4 UEFI 固件 `RPI_EFI.fd`，再由 UEFI 启动 Limine，最后启动 Linux kernel。结论先说：

**官方 PFTF/RPi4 的 `RPI_EFI.fd` 不能在当前 QEMU `raspi4b` 机器上原样跑到 UEFI 菜单或 UEFI Shell。它不是没有执行，而是进入 BL31/EDK2 后，在 DXE 驱动初始化阶段访问了 QEMU 没有实现的 BCM2711 MMIO 寄存器，触发同步异常。**

我们之前能跑通 `QEMU raspi4b -> UEFI -> Limine -> Linux -> initramfs shell`，靠的是专门改过的 QEMU 版本固件：

```text
qemu-rpi4-uefi-limine-linux-test/RPI_EFI_RPi4Qemu_DEBUG_SAFE_SD.fd
```

它和官方 PFTF 固件不是同一个测试对象。

---

## 1. 涉及文件

| 文件 | 作用 |
| --- | --- |
| `qemu-rpi4-uefi-limine-linux-test/rpi4-uefi-v1.52/RPI_EFI.fd` | 官方 PFTF RPi4 UEFI v1.52 固件 |
| `qemu-rpi4-uefi-limine-linux-test/rpi4-uefi-v1.52/config.txt` | 真机 boot firmware 使用的配置 |
| `qemu-rpi4-uefi-limine-linux-test/rpi4-uefi-v1.52/bcm2711-rpi-4-b.dtb` | PFTF 包里的 RPi4 DTB |
| `qemu-rpi4-uefi-limine-linux-test/sd-shell.img` | 已配置 Limine + Linux + minimal initramfs shell 的 SD 镜像 |
| `qemu-rpi4-uefi-limine-linux-test/RPI_EFI_RPi4Qemu_DEBUG_SAFE_SD.fd` | QEMU 专用 UEFI 固件，已验证可启动到 shell |
| `rpi5-uefi-d0/edk2-platforms/Platform/RaspberryPi/RPi4/RPi4Qemu.dsc` | QEMU 专用固件构建配置 |
| `rpi5-uefi-d0/edk2-platforms/Platform/RaspberryPi/RPi4/RPi4Qemu.fdf` | QEMU 专用固件 FDF 打包配置 |

---

## 2. 真机上 PFTF 是怎么被加载的

官方 `config.txt` 里关键字段是：

```ini
arm_64bit=1
enable_uart=1
uart_2ndstage=1
enable_gic=1
armstub=RPI_EFI.fd
device_tree_address=0x3e0000
device_tree_end=0x400000
dtoverlay=miniuart-bt
dtoverlay=upstream-pi4
```

含义：

1. 树莓派真机上电后先跑 SoC/板级固件，不是直接跑 `RPI_EFI.fd`。
2. `config.txt` 里的 `armstub=RPI_EFI.fd` 告诉树莓派 boot firmware：把这个 UEFI 固件当作 ARM stub 加载。
3. `device_tree_address=0x3e0000` 告诉 boot firmware：把设备树放到 `0x3e0000` 附近。
4. 真机 boot firmware 还会做一些 QEMU generic loader 不会自动做的初始化。

所以在 QEMU 里直接 `-kernel RPI_EFI.fd`、`-bios RPI_EFI.fd`、`-device loader ... RPI_EFI.fd` 都只是近似模拟，不等价于完整树莓派 boot firmware。

---

## 3. 失败复现

### 3.1 `-M raspi4b,firmware=RPI_EFI.fd`：没有串口输出

短超时测试命令：

```bash
cd /Users/vel/Work/RobotOS/Lerobot/qemu-rpi4-uefi-limine-linux-test

qemu-system-aarch64 \
  -M raspi4b,firmware=rpi4-uefi-v1.52/RPI_EFI.fd \
  -m 2G \
  -display none \
  -serial file:pftf-v152-machine-firmware-serial.log \
  -no-reboot -no-shutdown \
  -D pftf-v152-machine-firmware-qemu.log
```

现象：

```text
qemu-rpi4-uefi-limine-linux-test/pftf-v152-machine-firmware-serial.log
文件大小：0 bytes
```

也就是 QEMU 接受这个参数组合后没有任何串口输出，5 秒后只能手动杀掉。

### 3.2 `-bios RPI_EFI.fd`：没有串口输出

短超时测试命令：

```bash
cd /Users/vel/Work/RobotOS/Lerobot/qemu-rpi4-uefi-limine-linux-test

qemu-system-aarch64 \
  -M raspi4b \
  -m 2G \
  -display none \
  -serial file:pftf-v152-bios-serial.log \
  -bios rpi4-uefi-v1.52/RPI_EFI.fd \
  -no-reboot -no-shutdown \
  -D pftf-v152-bios-qemu.log
```

现象：

```text
qemu-rpi4-uefi-limine-linux-test/pftf-v152-bios-serial.log
文件大小：0 bytes
```

这说明 `-bios` 也不是加载 PFTF RPi4 armstub 的正确方式。

### 3.3 `-kernel RPI_EFI.fd`：没有串口输出

现象：

```text
qemu-rpi4-uefi-limine-linux-test/pftf-v152-kernel-serial.log
文件大小：0 bytes
```

这说明 `-kernel` 这种入口不适合直接喂官方 PFTF `RPI_EFI.fd`。

### 3.4 generic loader 放到 `0x0`：只能看到 BL31

命令：

```bash
cd /Users/vel/Work/RobotOS/Lerobot/qemu-rpi4-uefi-limine-linux-test

qemu-system-aarch64 \
  -M raspi4b \
  -m 2G \
  -display none \
  -serial file:pftf-v152-loader0-serial.log \
  -device loader,force-raw=on,file=rpi4-uefi-v1.52/RPI_EFI.fd,addr=0x0,cpu-num=0 \
  -no-reboot -no-shutdown \
  -D pftf-v152-loader0-qemu.log
```

串口输出：

```text
NOTICE:  BL31: v2.9(release):v2.9
NOTICE:  BL31: Built : 17:09:05, May 24 2023
```

这说明官方 PFTF 固件不是完全没跑，至少 TF-A/BL31 入口已经执行了。

### 3.5 loader `RPI_EFI.fd@0x0` + DTB `@0x3e0000`：仍然只到 BL31

命令：

```bash
cd /Users/vel/Work/RobotOS/Lerobot/qemu-rpi4-uefi-limine-linux-test

qemu-system-aarch64 \
  -M raspi4b \
  -m 2G \
  -display none \
  -serial file:pftf-v152-loader0-dtb3e-serial.log \
  -drive file=sd-shell.img,if=sd,format=raw \
  -device loader,force-raw=on,file=rpi4-uefi-v1.52/RPI_EFI.fd,addr=0x0,cpu-num=0 \
  -device loader,force-raw=on,file=rpi4-uefi-v1.52/bcm2711-rpi-4-b.dtb,addr=0x3e0000 \
  -no-reboot -no-shutdown \
  -D pftf-v152-loader0-dtb3e-qemu.log
```

串口输出仍然只有：

```text
NOTICE:  BL31: v2.9(release):v2.9
NOTICE:  BL31: Built : 17:09:05, May 24 2023
```

所以问题不是“没有 SD 镜像”或“没有 DTB”这么简单，后面需要看异常现场。

---

## 4. 第一个异常：RNG200 MMIO 写触发 Data Abort

用 GDB/HMP 抓到异常字符串：

```text
Synchronous Exception at 0x000000003966B008
```

故障点反汇编：

```asm
0x3966b000: ldr w0, [x0]   // MmioRead32(Address=x0)，从 x0 指向的 MMIO 地址读 32 bit
0x3966b004: ret            // 返回调用者
0x3966b008: str w1, [x0]   // MmioWrite32(Address=x0, Value=w1)，向 x0 指向的 MMIO 地址写 32 bit
0x3966b00c: ret            // 返回调用者
```

调用点附近：

```asm
0x3966b28c: mov  w1, #0x40000              // w1 = 0x40000，要写入的寄存器值
0x3966b290: mov  x0, #0x4010               // x0 低 16 bit = 0x4010
0x3966b294: movk x0, #0xfe10, lsl #16      // x0 = 0xfe104010
0x3966b298: bl   0x3966b008                // MmioWrite32(0xfe104010, 0x40000)

0x3966b29c: sub  x0, x0, #0x10             // x0 = 0xfe104000
0x3966b2a0: mov  w1, #0x7fff               // w1 = 0x7fff
0x3966b2a4: bl   0x3966b008                // MmioWrite32(0xfe104000, 0x7fff)
```

这里的地址：

```text
0xfe104000
0xfe104010
```

属于 BCM2711/RPi4 的 RNG200 随机数发生器寄存器区域。官方 PFTF 的 `Bcm2838RngDxe` 会初始化这个硬件。

关键点：

1. `str w1, [x0]` 本身只是一条 ARM64 store 指令。
2. 当 `x0` 是普通 RAM 地址时，它是普通内存写。
3. 当 `x0` 是 MMIO 地址时，这个 store 会被 CPU/总线当成设备寄存器写。
4. 在 QEMU 中，如果 `raspi4b` 机器没有为这个 MMIO 地址区注册设备模型，访问就会变成未实现设备访问，最终触发 guest 里的同步异常。

所以第一个明确失败点是：

```text
Bcm2838RngDxe -> MmioWrite32(0xfe104010 / 0xfe104000) -> QEMU raspi4b 没有 RNG200 模型 -> Synchronous Exception
```

---

## 5. 跳过 RNG 后的第二个异常：PCIe Host Bridge MMIO 读触发 Data Abort

为了确认是不是单点问题，曾用 GDB 把 RNG 初始化写跳过去。GDB 日志里能看到：

```text
skip RNG MmioWrite32: addr=0xfe104010 value=0x40000 lr=0x3966b29c
skip RNG MmioWrite32: addr=0xfe104000 value=0x7fff lr=0x3966b2a8
```

跳过 RNG 后，官方 PFTF 又触发第二个异常：

```text
Synchronous Exception at 0x00000000396314D4
```

第二个故障点：

```asm
0x396314d4: ldr w0, [x0]   // MmioRead32(Address=x0)，从设备寄存器读 32 bit
0x396314d8: ret            // 返回调用者
```

异常上下文里能看到关键地址：

```text
FAR = 0xfd509210
```

`FAR` 是 Fault Address Register，表示这次 data abort 访问的地址。`0xfd509210` 落在 BCM2711 PCIe host bridge 寄存器区域附近；PFTF/RPi4 配置里也写着：

```text
gBcm27xxTokenSpaceGuid.PcdBcm27xxPciRegBase|0xfd500000
```

所以第二个明确失败点是：

```text
PciHostBridge / PciSegmentLib -> MmioRead32(0xfd509210) -> QEMU raspi4b 没有完整 BCM2711 PCIe Host 模型 -> Synchronous Exception
```

这说明官方 PFTF 在 QEMU `raspi4b` 里不是“补一个参数就能好”的问题。它会继续碰真实板子上的外设寄存器，而 QEMU 并没有完整实现这些硬件。

---

## 6. 为什么这不是 Limine 的问题

同一个 `sd-shell.img`，换成 QEMU 专用固件可以跑通：

```bash
cd /Users/vel/Work/RobotOS/Lerobot/qemu-rpi4-uefi-limine-linux-test

qemu-system-aarch64 \
  -M raspi4b \
  -m 2G \
  -serial mon:stdio \
  -display none \
  -drive file=sd-shell.img,if=sd,format=raw \
  -device loader,force-raw=on,file=RPI_EFI_RPi4Qemu_DEBUG_SAFE_SD.fd,addr=0x0,cpu-num=0 \
  -device loader,force-raw=on,file=rpi4-uefi-v1.52/bcm2711-rpi-4-b.dtb,addr=0x1f0000 \
  -no-reboot -no-shutdown
```

成功标志：

```text
=== Limine -> Linux -> minimal initramfs shell ===
/ #
```

这条成功链路证明：

1. Limine 配置能被 UEFI 找到。
2. Limine 能加载 Linux kernel 和 initramfs。
3. Linux 能启动到最小 shell。
4. 卡住的层次在“官方 PFTF/RPi4 UEFI 固件适配 QEMU raspi4b”，不是 Limine。

---

## 7. QEMU 专用固件为什么能绕过去

QEMU 专用构建配置里已经把相关路径去掉或绕开了。`RPi4Qemu.dsc` 中的注释：

```text
Networking / RNG / PCI support

Omitted for the QEMU raspi4b bring-up build:
- BcmGenetDxe touches 0xfd580000..0xfd58ffff, which QEMU does not model.
- Bcm2838RngDxe touches RNG200 at 0xfe104000, which QEMU raspi4b lacks.
- The BCM2711 PCIe host bridge at 0xfd500000 is not implemented enough for
  PFTF's PciSegmentLib/PciHostBridgeDxe path.
- NVMe and XHCI depend on that PCIe path.
```

`RPi4Qemu.fdf` 中也有对应说明：

```text
Omitted in the QEMU raspi4b bring-up firmware. QEMU does not model the
corresponding BCM2711 GENET, RNG200, and PCIe register blocks closely
enough for the stock PFTF DXE drivers.
```

也就是说，我们能跑通的固件不是“官方 PFTF 原样成功”，而是做了 QEMU 适配：

1. 去掉会访问未实现 MMIO 的 RNG、GENET、PCIe/NVMe/XHCI 路径。
2. 保留 SD/MMC 路径，让 UEFI BDS 能从 `sd-shell.img` 找到 `EFI/BOOT/BOOTAA64.EFI`。
3. 让 Limine 继续接管启动 Linux。

---

## 8. 当前结论

### 8.1 已确认

1. `-M raspi4b,firmware=RPI_EFI.fd`：短超时测试，串口 0 输出。
2. `-bios RPI_EFI.fd`：短超时测试，串口 0 输出。
3. `-kernel RPI_EFI.fd`：串口 0 输出。
4. `loader RPI_EFI.fd@0x0`：能进入 BL31，但之后卡住。
5. `loader RPI_EFI.fd@0x0 + DTB@0x3e0000 + SD`：仍只到 BL31。
6. GDB 证明官方 PFTF 后续触发同步异常：
   - 第一次：`MmioWrite32(0xfe104010 / 0xfe104000)`，RNG200 未建模。
   - 跳过 RNG 后第二次：`MmioRead32(0xfd509210)`，BCM2711 PCIe host bridge 未完整建模。
7. QEMU 专用 UEFI 固件能启动 Limine/Linux/shell，说明 Limine 和 Linux 侧链路本身可用。

### 8.2 后续路线

有三条路：

| 路线 | 说明 | 适合做什么 |
| --- | --- | --- |
| 继续用 `RPI_EFI_RPi4Qemu_DEBUG_SAFE_SD.fd` | 改固件适配 QEMU，绕过 QEMU 没实现的硬件 | 验证 UEFI -> Limine -> Linux/内核启动链 |
| 改 QEMU `raspi4b` 硬件模型 | 给 QEMU 补 RNG200、BCM2711 PCIe host、GENET 等 MMIO 模型 | 真正模拟官方 PFTF 需要的板级硬件 |
| 用 `virt` + AAVMF/EDK2 | 不模拟树莓派板级硬件，只验证标准 AArch64 UEFI | 验证 Limine/aarch64 EFI loader/自研 kernel 入口 |

如果当前目标是“先把 Limine/aarch64 UEFI/Linux kernel 链路跑通”，第一条和第三条更合适。

如果目标是“官方 PFTF/RPi4 UEFI 在 `QEMU raspi4b` 上原样跑起来”，那就必须走第二条：补 QEMU 的 BCM2711/RPi4 外设模型，至少要处理 RNG200 和 PCIe host bridge，后面还可能遇到 GENET、XHCI、NVMe 等更多设备。
