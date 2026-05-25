# 04 RP1 / USB / 摄像头 / 舵机链路

## 准确硬件链路

```text
BCM2712
  -> PCIe pcie2
  -> RP1
  -> RP1 DWC3 USB host, compatible = snps,dwc3
  -> xhci-hcd root hub
  -> UVC camera / CDC ACM serial
```

关键点：

```text
设备树描述 BCM2712、PCIe、RP1、DWC3 USB host。
USB 摄像头和舵机板不是设备树静态节点。
它们是 USB host 工作后运行时枚举出来的设备。
```

## LeRobot 实际用到的外设

| LeRobot 角色 | 设备节点 | USB id | driver | 说明 |
| --- | --- | --- | --- | --- |
| handeye camera | `/dev/video0` | `1bcf:2281` | `uvcvideo` | USB 2.0 Camera，OpenCV index `0`。 |
| fixed camera | `/dev/video2` | `1bcf:2281` | `uvcvideo` | USB 2.0 Camera，OpenCV index `2`。 |
| follower servo | `/dev/ttyACM0` | `1a86:55d3` | `cdc_acm` | `/dev/so101_follower_left -> ttyACM0`。 |
| leader servo | `/dev/ttyACM1` | `1a86:55d3` | `cdc_acm` | `/dev/so101_leader_left -> ttyACM1`。 |

## 摄像头链路

```text
SO101Follower.get_observation()
  -> OpenCVCamera.async_read()
  -> OpenCVCamera.read()
  -> cv2.VideoCapture.read()
  -> /dev/video0, /dev/video2
  -> V4L2
  -> uvcvideo
  -> USB core
  -> xhci-hcd root hub
  -> RP1 DWC3 USB host
```

对新 kernel 的要求：

- 提供 `/dev/video*` 设备节点。
- 支持 V4L2 `ioctl`。
- 支持 videobuf2 buffer queue。
- 支持 UVC over USB。
- 支持 USB host 枚举和传输。

## 舵机串口链路

```text
FeetechMotorsBus
  -> scservo_sdk
  -> pyserial
  -> serialposix
  -> os.open / termios / ioctl / select / os.read / os.write
  -> /dev/ttyACM0, /dev/ttyACM1
  -> cdc_acm
  -> USB core
  -> xhci-hcd root hub
  -> RP1 DWC3 USB host
```

对新 kernel 的要求：

- 提供 `/dev/ttyACM*` 设备节点。
- 支持 TTY/termios。
- 支持 `read/write/ioctl/select`。
- 支持 CDC ACM USB class。
- 支持 USB host 枚举和传输。

## 相关 kernel 源码主路径

```text
USB host:
linux/drivers/usb/dwc3/core.c
linux/drivers/usb/dwc3/host.c
linux/drivers/usb/host/xhci.c
linux/drivers/usb/host/xhci-plat.c

UVC camera:
linux/drivers/media/usb/uvc/uvc_driver.c
linux/drivers/media/usb/uvc/uvc_v4l2.c
linux/drivers/media/usb/uvc/uvc_video.c
linux/drivers/media/usb/uvc/uvc_queue.c
linux/drivers/media/v4l2-core/v4l2-dev.c
linux/drivers/media/v4l2-core/v4l2-ioctl.c
linux/drivers/media/common/videobuf2/videobuf2-core.c

CDC ACM serial:
linux/drivers/usb/class/cdc-acm.c

RP1:
linux/drivers/mfd/rp1.c
linux/drivers/firmware/rp1-fw.c
linux/drivers/mailbox/rp1-mailbox.c
linux/drivers/misc/rp1-pio.c
```

