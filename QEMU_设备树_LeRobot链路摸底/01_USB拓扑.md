# 01 USB 拓扑

## 从 USB 拓扑反推 kernel 要求

这里引用的是 USB 树，不是设备树：

```text
robotdoc/QEMU_设备树_LeRobot链路摸底/树莓派5真机文件快照/2026-05-18/hardware/lsusb-tree.txt
```

`lsusb-tree.txt` 来自树莓派真机上的 `lsusb -t` 输出。它不是说明 BCM2712/RP1 这些板级硬件的 DTS，而是说明当前 USB 总线上 root hub、外接 hub、摄像头、舵机串口板是怎么一层层挂起来的。

这个文件不是只说明“插了哪些 USB 设备”，它真正有用的地方是：它把 LeRobot 外设压到 kernel 的驱动链路暴露出来。

当前 LeRobot 外设的 USB 拓扑可以概括成：

```text
RP1 / DWC3 USB host
  -> xhci-hcd root hub
  -> USB 2.0 root hub
      -> 键盘，Driver=usbhid
  -> USB 2.0 root hub
  -> 外接 USB 2.1 Hub
      -> Port 1: 第一个 UVC 摄像头，Driver=uvcvideo
      -> Port 2: 第二个 UVC 摄像头，Driver=uvcvideo
      -> Port 3: 第一个 SO101 串口板，Driver=cdc_acm
      -> Port 4: 第二个 SO101 串口板，Driver=cdc_acm
```

所以新的 kernel 如果要在树莓派真机上跑完整 LeRobot，不能只做到“能启动”。它至少要支持下面这些层：

| 层              | 真机证据                                                  | 新 kernel 要提供什么                                                         |
| --------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------- |
| USB host        | `Driver=xhci-hcd`，硬件源头是 RP1 DWC3 USB host           | 初始化 RP1/DWC3/xHCI，让 USB root hub 出现。                                 |
| USB hub         | `Driver=hub/4p`                                           | 支持外接 USB Hub，能继续枚举 Hub 后面的设备。                                |
| USB HID 键盘    | `Class=Human Interface Device, Driver=usbhid`             | 支持 USB 键盘输入；它不是 LeRobot 主链路，但是真机登录、调试、应急操作需要。 |
| UVC 摄像头      | `Class=Video, Driver=uvcvideo`                            | 支持 UVC 驱动，把两个 USB 摄像头变成 V4L2 设备。                             |
| V4L2            | `/dev/video0`、`/dev/video2`                              | 提供 `video4linux` 设备节点、V4L2 ioctl、buffer queue、采集接口。            |
| CDC ACM 串口    | `Class=Communications/CDC Data, Driver=cdc_acm`           | 支持 USB CDC ACM，把两块舵机串口板变成 ttyACM 设备。                         |
| TTY / termios   | `/dev/ttyACM0`、`/dev/ttyACM1`                            | 支持串口 `open/read/write/ioctl/select/poll` 和波特率配置。                  |
| devtmpfs / udev | `/dev/video*`、`/dev/ttyACM*`、`/dev/so101_follower_left` | 让用户态能通过稳定路径发现和打开设备。                                       |

对应到 LeRobot：

```text
OpenCV camera
  -> /dev/video0, /dev/video2
  -> V4L2
  -> uvcvideo
  -> USB hub
  -> xhci-hcd / RP1 DWC3

scservo_sdk / pyserial
  -> /dev/ttyACM0, /dev/ttyACM1
  -> TTY / termios
  -> cdc_acm
  -> USB hub
  -> xhci-hcd / RP1 DWC3
```

```text
USB keyboard
  -> usbhid
  -> HID input subsystem
  -> /dev/input/event*
  -> 登录、shell 操作、现场调试
```
