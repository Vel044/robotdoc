# 实验 07 环境快照

## 基本信息

| 项目 | 记录 |
|---|---|
| 日期 |  |
| 机器 | 树莓派5 |
| 任务 | pick |
| episode 时长 | 约 30 s |
| `chunk_size` | 100 |
| 目标 `fps` | 30 |
| 模型权重路径 |  |

## 系统版本

| 项目 | 命令 | 结果 |
|---|---|---|
| OS | `cat /etc/os-release` |  |
| Kernel | `uname -a` |  |
| Python | `python -V` |  |
| LeRobot | `python -c "import importlib.metadata as m; print(m.version('lerobot'))"` |  |
| PyTorch | `python -c "import torch; print(torch.__version__)"` |  |
| torchvision | `python -c "import torchvision; print(torchvision.__version__)"` |  |
| OpenCV | `python -c "import cv2; print(cv2.__version__)"` |  |
| NumPy | `python -c "import numpy; print(numpy.__version__)"` |  |
| pyserial | `python -c "import serial; print(serial.__version__)"` |  |
| feetech-servo-sdk | `python -c "import importlib.metadata as m; print(m.version('feetech-servo-sdk'))"` |  |

## 硬件和设备节点

| 项目 | 命令 | 结果 |
|---|---|---|
| CPU | `lscpu` |  |
| 摄像头设备 | `v4l2-ctl --list-devices` |  |
| 摄像头参数 | `v4l2-ctl --device=/dev/videoX --all` |  |
| 串口设备 | `ls -l /dev/ttyACM*` |  |
| USB 拓扑 | `lsusb -t` |  |

## 温度和频率

| 项目 | 命令 | 结果 |
|---|---|---|
| 开始温度 | `vcgencmd measure_temp` |  |
| 结束温度 | `vcgencmd measure_temp` |  |
| CPU 频率 | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` |  |
| 降频状态 | `vcgencmd get_throttled` |  |

## 备注

- 摄像头连接位置：
- follower 连接位置：
- 供电方式：
- 是否有后台任务：
- 异常现象：
