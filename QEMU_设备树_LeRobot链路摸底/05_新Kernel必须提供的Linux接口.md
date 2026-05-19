# 05 新 Kernel 必须提供的 Linux 接口

这些不是 Python 包清单，而是 LeRobot 用户态会压到 kernel 的接口面。

| 接口面 | 当前触发来源 | 新 kernel 要提供什么 |
| --- | --- | --- |
| ELF / 进程 | Python、conda、torch/cv2/av 原生 `.so` | `execve`、动态链接、`mmap`、`mprotect`、`brk`、signals。 |
| 文件系统 | conda 包、模型目录、dataset、rootfs | VFS、`open/read/write/stat/close`、目录遍历、权限。 |
| 伪文件系统 | `/proc`、`/sys`、`/dev` | 用户态和 udev 依赖这些路径发现系统和设备。 |
| 线程/同步 | Python、PyTorch、OpenCV、数据写入线程 | `clone`、`futex`、调度、signals。 |
| 时间/等待 | `record_loop()`、`busy_wait()`、串口/摄像头 timeout | `clock_gettime`、`nanosleep`、poll/select timeout。 |
| 摄像头 | OpenCV -> V4L2 -> UVC | `/dev/video*`、V4L2 `ioctl`、buffer queue、USB transfer。 |
| 舵机串口 | scservo_sdk -> pyserial -> TTY -> CDC ACM | `/dev/ttyACM*`、termios、`read/write/ioctl/select`。 |
| USB host | 摄像头和舵机都挂 USB | DWC3/xHCI host、USB 枚举、bulk/interrupt/isochronous transfer。 |
| 模块 | `uvcvideo`、`cdc_acm`、`videobuf2_*` | 如果做模块，`/lib/modules/<uname -r>` 要和 kernel release 匹配。 |

## 和 LeRobot 负载的关系

```text
python -m lerobot.record
  -> Python / glibc / dynamic linker
  -> torch CPU inference
  -> OpenCV VideoCapture
  -> pyserial/scservo motor bus
  -> dataset/video writer
```

对应到 kernel：

```text
Python 能不能跑：
  ELF loader / mmap / futex / signals / VFS

摄像头能不能读：
  /dev/video* / V4L2 ioctl / UVC / USB host

舵机能不能动：
  /dev/ttyACM* / TTY / termios / CDC ACM / USB host

数据能不能保存：
  filesystem / write / fsync / mmap / page cache
```

## 最小阶段划分

| 阶段 | 先实现什么 | 说明 |
| --- | --- | --- |
| 最小启动 | ARM64 boot、dtb、cmdline、console、memory、timer、GIC | 能看到 kernel 日志。 |
| 最小用户态 | ELF、VFS、rootfs、`/dev`、`/proc`、`/sys`、基本 syscall | 能跑 shell / Python。 |
| LeRobot mock | Python/conda、torch CPU、mock camera、mock motor | 能跑 `record.py` 主流程。 |
| LeRobot 真硬件 | DWC3/xHCI、UVC、CDC ACM、V4L2、TTY | 能接真实摄像头和舵机。 |

