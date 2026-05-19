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
Raspberry Pi firmware
  -> 读取 /boot/firmware/config.txt
  -> 选择 kernel_2712.img
  -> 选择 bcm2712-rpi-5-b.dtb
  -> 读取 cmdline.txt
  -> 可选加载 initramfs_2712
  -> 把 kernel image 和 dtb 放进内存
  -> 跳到 kernel 入口
  -> kernel 解析 dtb / cmdline
  -> kernel 挂载 rootfs
  -> 进入用户态
```

## 对新 kernel 的要求

| 项 | 新 kernel 要做什么 |
| --- | --- |
| kernel image | 产出 Pi firmware/QEMU 能加载的 ARM64 kernel image。 |
| dtb | 接收并解析 `bcm2712-rpi-5-b.dtb` 或 QEMU 提供的 dtb。 |
| cmdline | 读取 `console=`、`root=`、`rootwait` 等启动参数。 |
| rootfs | 能挂载真实 rootfs，运行用户态程序。 |
| modules | 如果驱动做成模块，`/lib/modules/<kernel-release>` 必须匹配。 |

## 当前真机核对

| 对比项 | 真机结果 | 结论 |
| --- | --- | --- |
| 板卡型号 | `Raspberry Pi 5 Model B Rev 1.1` | 与 Pi 5/BCM2712 目标一致。 |
| boot dtb | `/boot/firmware/bcm2712-rpi-5-b.dtb` 存在 | 与 Pi 5 启动链一致。 |
| chosen console | `stdout-path = serial10:115200n8`，实际 cmdline 展开为 `ttyAMA10,115200` | firmware 会补全启动参数。 |
| 当前 kernel | `6.12.75-v8-16k-TEST-PSELECT6+` | modules 目录必须匹配这个 release。 |

