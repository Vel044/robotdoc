# 树莓派 5 真机文件快照：2026-05-18

## 已复制的原始文件

```text
boot/bcm2712-rpi-5-b.dtb          # 真机 /boot/firmware 里的 Pi 5 dtb
boot/config.txt                   # Raspberry Pi firmware 配置
boot/cmdline.txt                  # /boot/firmware/cmdline.txt
device-tree/proc-device-tree.tar  # 运行时 /proc/device-tree 打包，包含 firmware 修改后的实际设备树属性
```

没有复制完整大文件：

```text
/boot/firmware/kernel_2712.img
/boot/firmware/initramfs_2712
```

这两个目前只记录了 `ls -l` 和 `sha256sum`，见：

```text
boot/boot_files_ls.txt
boot/boot_files_sha256.txt
```

## 本地生成文件

```text
boot/bcm2712-rpi-5-b.from-dtb.dts
```

这是用本地 `dtc` 从 `boot/bcm2712-rpi-5-b.dtb` 反编译出来的文本版设备树，方便阅读。反编译时出现的 `dtc` warning 是设备树格式检查警告，不影响 `.dts` 文件生成。

## 查询输出

```text
proc/uname-a.txt
proc/version.txt
proc/cmdline.txt
proc/cpuinfo.txt

device-tree/model.txt
device-tree/compatible.txt
device-tree/chosen_stdout-path.txt
device-tree/chosen_bootargs.txt
device-tree/alias_usb0.txt
device-tree/alias_usb1.txt
device-tree/alias_serial10.txt
device-tree/alias_console.txt

hardware/lsusb.txt
hardware/lsusb-tree.txt
hardware/v4l2-list-devices.txt
hardware/dev-video-ls.txt
hardware/dev-tty-so101-ls.txt
hardware/udev-video0.txt
hardware/udev-video2.txt
hardware/udev-ttyACM0.txt
hardware/udev-ttyACM1.txt

modules/lsmod.txt
modules/modules.dep
modules/modules.alias
modules/modules.builtin
modules/modules.order
```

## 关键事实

```text
板卡：Raspberry Pi 5 Model B Rev 1.1
compatible：raspberrypi,5-model-b / brcm,bcm2712
kernel：6.12.75-v8-16k-TEST-PSELECT6+
设备树：/boot/firmware/bcm2712-rpi-5-b.dtb
USB host：RP1 DWC3 host -> xhci-hcd root hub
摄像头：/dev/video0、/dev/video2，driver=uvcvideo，USB id=1bcf:2281
舵机串口：/dev/ttyACM0、/dev/ttyACM1，driver=cdc_acm，USB id=1a86:55d3
```

