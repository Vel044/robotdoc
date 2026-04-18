# LeRobot 观察推理执行循环完整链路分析

> 从 record 开始，在一个 episode 的一个观察推理执行循环内的完整软硬件交互链
>
> 源码路径：`/Users/vel-virtual/Lerobot/lerobot/src/lerobot/`

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [步骤一：观测阶段 (Observation)](#2-步骤一观测阶段-observation)
3. [步骤二：推理阶段 (Inference)](#3-步骤二推理阶段-inference)
4. [步骤三：动作阶段 (Action)](#4-步骤三动作阶段-action)
5. [完整时序图](#5-完整时序图)
6. [源码文件索引](#6-源码文件索引)

---

## 1. 整体架构概览

### 1.1 一个 Episode 的主循环

**源码位置**：`lerobot/record.py:record_loop()`

```python
while timestamp < control_time_s:
    # ============================================================
    # 步骤一：观测阶段 (Observation)
    # ============================================================
    obs = robot.get_observation()
    obs_processed = robot_observation_processor(obs)

    # ============================================================
    # 步骤二：推理阶段 (Inference)
    # ============================================================
    action_values = predict_action(observation, policy, ...)

    # ============================================================
    # 步骤三：动作阶段 (Action)
    # ============================================================
    robot_action_to_send = robot_action_processor((action_values, obs))
    robot.send_action(robot_action_to_send)

    # ============================================================
    # 等待阶段 (Wait)
    # ============================================================
    busy_wait(1 / fps - dt_s)
```

### 1.2 层次架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Python 应用层                                       │
│                                                                              │
│   record.py: record_loop()                                                  │
│         │                                                                    │
│         ├── robot.get_observation()                                         │
│         ├── predict_action()                                                 │
│         └── robot.send_action()                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   观测路径       │  │   推理路径       │  │   动作路径       │
│                 │  │                 │  │                 │
│  摄像头 + 串口   │  │  PyTorch       │  │  串口           │
│       │         │  │       │         │  │       │         │
│       ▼         │  │       ▼         │  │       ▼         │
│  OpenCV V4L2   │  │  torch.nn      │  │  pyserial       │
│       │         │  │       │         │  │       │         │
│       ▼         │  │       ▼         │  │       ▼         │
│  VIDIOC_DQBUF  │  │  parallel_for   │  │  write/read     │
│       │         │  │       │         │  │       │         │
│       ▼         │  │       ▼         │  │       ▼         │
│  kernel V4L2   │  │  ThreadPool     │  │  ioctl          │
│       │         │  │       │         │  │       │         │
│       ▼         │  │       ▼         │  │       ▼         │
│  DMA Buffer    │  │  CPU 矩阵运算   │  │  kernel TTY     │
│                 │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Linux Kernel                                        │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      │
│   │ V4L2 子系统  │   │ TTY 子系统  │   │ 内存管理    │                      │
│   │ (摄像头)     │   │ (串口)      │   │ (mmap)     │                      │
│   └─────────────┘   └─────────────┘   └─────────────┘                      │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      │
│   │ USB 驱动    │   │ USB-Serial  │   │ DMA 引擎   │                      │
│   │ (UVC)       │   │ (cdc_acm)   │   │            │                      │
│   └─────────────┘   └─────────────┘   └─────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          硬件层                                              │
│                                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      │
│   │ USB 摄像头  │   │ USB 舵机板  │   │ ARM CPU    │                      │
│   │ (Camera)    │   │ (Feetech)  │   │ (树莓派)   │                      │
│   └─────────────┘   └─────────────┘   └─────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 步骤一：观测阶段 (Observation)

### 2.1 Python 层调用链

**源码位置**：`lerobot/record.py:254-265`

```python
# ============================================================
# 观测阶段 (Observation Phase)
# ============================================================
obs_start_t = time.perf_counter()
obs = robot.get_observation()           # 获取观测数据
obs_processed = robot_observation_processor(obs)  # 处理观测数据
obs_end_t = time.perf_counter()
```

### 2.2 robot.get_observation() 实现

**源码位置**：`lerobot/robots/so101_follower/so101_follower.py:249-267`

```python
def get_observation(self) -> dict[str, Any]:
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    # ============================================================
    # 第一部分：读取舵机位置 (串口通信)
    # ============================================================
    start = time.perf_counter()
    obs_dict = self.bus.sync_read("Present_Position")  # 读取所有舵机位置
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
    dt_ms = (time.perf_counter() - start) * 1e3

    # ============================================================
    # 第二部分：读取摄像头图像 (V4L2)
    # ============================================================
    for cam_key, cam in self.cameras.items():
        start = time.perf_counter()
        obs_dict[cam_key] = cam.async_read()  # 异步读取图像
        dt_ms = (time.perf_counter() - start) * 1e3

    return obs_dict
```

### 2.3 第一部分：舵机位置读取 (串口)

#### 2.3.1 Python 层

**源码位置**：`lerobot/motors/motors_bus.py:1053-1095`

```python
def sync_read(self, data_name: str, motors: str | list[str] | None = None, ...):
    """从多个舵机同时读取同一寄存器"""
    names = self._get_motors_list(motors)
    ids = [self.motors[motor].id for motor in names]

    # 获取寄存器地址和长度
    addr, length = get_address(self.model_ctrl_table, model, data_name)

    # 执行同步读取
    ids_values, _ = self._sync_read(addr, length, ids, ...)

    return {self._id_to_name(id_): value for id_, value in ids_values.items()}

def _sync_read(self, addr: int, length: int, motor_ids: list[int], ...):
    # 设置同步读取器
    self._setup_sync_reader(motor_ids, addr, length)

    # 发送并接收数据包
    comm = self.sync_reader.txRxPacket()

    # 提取数据
    values = {id_: self.sync_reader.getData(id_, addr, length) for id_ in motor_ids}
    return values, comm
```

#### 2.3.2 PySerial 层

**源码位置**：`lerobot/motors/motors_bus.py:117` (成员变量)

```python
self.ser: serial.Serial  # PySerial 串口对象
```

PySerial 内部调用链：

```
serial.Serial.write(data)
    │
    ▼
serialposix.py: write()
    │
    ▼
os.write(self.fd, data)  # POSIX 系统调用
    │
    ▼
glibc: __write()
    │
    ▼
syscall: write(fd, buf, count)
```

#### 2.3.3 系统调用层

**write 系统调用入口**：`linux/fs/read_write.c:631`

```c
SYSCALL_DEFINE3(write, unsigned int, fd, const char __user *, buf, size_t, count)
{
    return ksys_write(fd, buf, count);
}

ssize_t ksys_write(unsigned int fd, const char __user *buf, size_t count)
{
    struct fd f = fdget_pos(fd);
    
    if (f.file) {
        loff_t pos, *ppos = file_ppos(f.file);
        // 调用文件操作的 write 方法
        ret = vfs_write(f.file, buf, count, ppos);
        fdput_pos(f);
    }
    return ret;
}
```

#### 2.3.4 TTY 子系统

```
vfs_write()
    │
    ▼
tty_write()  // drivers/tty/tty_io.c
    │
    ▼
tty_ldisc_ops->write()  // 线路规程
    │
    ▼
tty_operations->write()  // 驱动特定实现
    │
    ▼
usb_serial_port_write()  // USB Serial 驱动
    │
    ▼
usb_submit_urb()  // 提交 USB 请求块
    │
    ▼
USB 总线 → USB 设备 (舵机板)
```

#### 2.3.5 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 步骤一观测阶段 - 舵机位置读取                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Python 应用层
    │
    │ robot.get_observation()
    │     → bus.sync_read("Present_Position")
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ lerobot/motors/motors_bus.py                                                │
│                                                                              │
│   sync_read() → _sync_read() → sync_reader.txRxPacket()                     │
│                                                                              │
│   构造数据包：                                                                │
│   ┌────────────────────────────────────────────┐                            │
│   │ Header | ID | Length | Instr | Param | CRC │                            │
│   │ 0xFF 0xFF | ... | ... | READ | addr,len | ... │                           │
│   └────────────────────────────────────────────┘                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PySerial (serial/serialposix.py)                                            │
│                                                                              │
│   self.ser.write(packet)                                                     │
│       │                                                                      │
│       ▼                                                                      │
│   os.write(self.fd, packet)  # fd = open("/dev/ttyACM0")                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ glibc (sysdeps/unix/sysv/linux/write.c)                                     │
│                                                                              │
│   __write(fd, buf, count)                                                    │
│       │                                                                      │
│       ▼                                                                      │
│   syscall(__NR_write, fd, buf, count)                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Linux Kernel - 系统调用入口                                                  │
│                                                                              │
│   fs/read_write.c: SYSCALL_DEFINE3(write, ...)                              │
│       │                                                                      │
│       ▼                                                                      │
│   ksys_write(fd, buf, count)                                                │
│       │                                                                      │
│       ▼                                                                      │
│   vfs_write(file, buf, count, pos)                                          │
│       │                                                                      │
│       ▼                                                                      │
│   file->f_op->write()  或  new_sync_write()                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Linux Kernel - TTY 子系统                                                    │
│                                                                              │
│   drivers/tty/tty_io.c: tty_write()                                         │
│       │                                                                      │
│       ├─► tty_ldisc_ref_wait()  获取线路规程                                │
│       │                                                                      │
│       ▼                                                                      │
│   ld->ops->write()  (N_TTY 线路规程)                                         │
│       │                                                                      │
│       ▼                                                                      │
│   tty->ops->write()  (驱动 write 方法)                                       │
│                                                                              │
│   对于 USB Serial 设备 (/dev/ttyACM0):                                       │
│       │                                                                      │
│       ▼                                                                      │
│   drivers/usb/serial/usb-serial.c: serial_write()                           │
│       │                                                                      │
│       ▼                                                                      │
│   usb_serial_port_write(port, buf, count)                                   │
│       │                                                                      │
│       ▼                                                                      │
│   usb_submit_urb(urb)  提交 USB 请求块                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Linux Kernel - USB 核心                                                      │
│                                                                              │
│   drivers/usb/core/hcd.c: usb_hcd_submit_urb()                              │
│       │                                                                      │
│       ▼                                                                      │
│   hcd->driver->urb_enqueue()  主机控制器驱动                                 │
│       │                                                                      │
│       ├─► 分配 DMA 缓冲区                                                    │
│       ├─► 构建 USB 事务                                                      │
│       └─► 提交到 USB 调度器                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 硬件层                                                                       │
│                                                                              │
│   USB 主机控制器 (EHCI/xHCI)                                                 │
│       │                                                                      │
│       ▼                                                                      │
│   USB 总线 → USB 线缆                                                        │
│       │                                                                      │
│       ▼                                                                      │
│   USB 舵机板 (Feetech/飞特)                                                  │
│       │                                                                      │
│       ├─► 解析数据包                                                         │
│       ├─► 读取舵机位置寄存器                                                  │
│       └─► 构造响应包                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    │ (响应数据返回)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Linux Kernel - read 路径                                                     │
│                                                                              │
│   USB 中断处理 → URB 完成                                                    │
│       │                                                                      │
│       ▼                                                                      │
│   tty_flip_buffer_push()  数据推送到 TTY 缓冲区                              │
│       │                                                                      │
│       ▼                                                                      │
│   用户调用 read() 时:                                                        │
│       │                                                                      │
│       ▼                                                                      │
│   tty_read() → ld->ops->read() → copy_to_user()                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 返回 Python 层                                                               │
│                                                                              │
│   serial.Serial.read() 返回响应数据                                          │
│       │                                                                      │
│       ▼                                                                      │
│   sync_reader.getData() 解析响应                                             │
│       │                                                                      │
│       ▼                                                                      │
│   返回 dict: {"motor1.pos": 1234, "motor2.pos": 5678, ...}                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 第二部分：摄像头图像读取 (V4L2)

#### 2.4.1 Python 层

**源码位置**：`lerobot/cameras/opencv/camera_opencv.py`

```python
def async_read(self) -> np.ndarray:
    """异步读取一帧图像"""
    return self.read()  # 调用同步读取

def read(self) -> np.ndarray:
    """读取一帧图像，返回 BGR 格式的 NumPy 数组"""
    ret, frame = self.videocapture.read()
    if not ret:
        raise RuntimeError("Failed to read frame")
    return frame
```

#### 2.4.2 OpenCV 层

**源码位置**：`opencv/modules/videoio/src/cap.cpp:551`

```cpp
bool VideoCapture::read(OutputArray image) {
    if (grab()) {
        retrieve(image);
    }
    return !image.empty();
}

bool VideoCapture::grab() {
    return !icap.empty() ? icap->grabFrame() : false;
}

bool VideoCapture::retrieve(OutputArray image, int channel) {
    if (!icap.empty())
        return icap->retrieveFrame(channel, image);
    return false;
}
```

#### 2.4.3 V4L2 层调用链

详见之前文档《OpenCV视频流数据流分析.md》，核心流程：

```
grabFrame()
    │
    ▼
tryIoctl(VIDIOC_DQBUF)
    │
    ├─► 如果返回 EAGAIN:
    │       │
    │       ▼
    │   select(fd) 等待数据就绪
    │       │
    │       ▼
    │   再次 ioctl(VIDIOC_DQBUF)
    │
    ▼
retrieveFrame()
    │
    ├─► 从 mmap 地址读取数据 (零拷贝)
    │
    └─► convertToRgb() 格式转换
```

#### 2.4.4 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 步骤一观测阶段 - 摄像头图像读取                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Python 应用层
    │
    │ cam.async_read()
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OpenCV Python 绑定 (cv2)                                                    │
│                                                                              │
│   cv2.VideoCapture.read()                                                    │
│       │                                                                      │
│       ▼                                                                      │
│   cv2.VideoCapture.grab() + cv2.VideoCapture.retrieve()                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OpenCV C++ 层 (modules/videoio/src/)                                         │
│                                                                              │
│   VideoCapture::read() (cap.cpp:551)                                        │
│       │                                                                      │
│       ├─► grab() → icap->grabFrame()                                        │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   CvCaptureCAM_V4L::grabFrame() (cap_v4l.cpp:1065)                  │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   read_frame_v4l2() (cap_v4l.cpp:943)                               │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   tryIoctl(VIDIOC_DQBUF, &buf)                                       │
│       │       │                                                              │
│       │       ├─► ioctl(deviceHandle, VIDIOC_DQBUF, &buf)                   │
│       │       │       │                                                      │
│       │       │       └─► 返回 EAGAIN?                                       │
│       │       │               │                                              │
│       │       │               ▼                                              │
│       │       │           select(deviceHandle + 1, &fds, NULL, NULL, &tv)   │
│       │       │               │                                              │
│       │       │               └─► 再次 ioctl                                │
│       │       │                                                              │
│       │       └─► 成功: bufferIndex = buf.index                             │
│       │                                                                      │
│       └─► retrieve() → icap->retrieveFrame()                                │
│               │                                                              │
│               ▼                                                              │
│           CvCaptureCAM_V4L::retrieveFrame() (cap_v4l.cpp:2099)              │
│               │                                                              │
│               ├─► 从 mmap 地址读取数据                                       │
│               │       buffers[bufferIndex].memories[start]                  │
│               │                                                              │
│               └─► convertToRgb() (cap_v4l.cpp:1408)                          │
│                       │                                                      │
│                       ├─► MJPEG: imdecode(libjpeg)                           │
│                       ├─► YUV420: cvtColor(COLOR_YUV2BGR_IYUV)              │
│                       ├─► NV12: cvtColor(COLOR_YUV2BGR_NV12)                │
│                       └─► YUYV: cvtColor(COLOR_YUV2BGR_YUYV)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ glibc (系统调用封装)                                                          │
│                                                                              │
│   ioctl(fd, VIDIOC_DQBUF, &buf)                                             │
│       │                                                                      │
│       ▼                                                                      │
│   syscall(__NR_ioctl, fd, VIDIOC_DQBUF, &buf)                               │
│                                                                              │
│   select(fd)                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│   syscall(__NR_pselect6, ...)                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Linux Kernel - V4L2 子系统                                                   │
│                                                                              │
│   drivers/media/v4l2-core/v4l2-dev.c: v4l2_ioctl()                          │
│       │                                                                      │
│       ▼                                                                      │
│   video_usercopy() 参数处理和验证                                             │
│       │                                                                      │
│       ▼                                                                      │
│   vdev->ioctl_ops->vidioc_dqbuf()                                           │
│       │                                                                      │
│       ▼                                                                      │
│   drivers/media/common/videobuf2/videobuf2-core.c                          │
│       │                                                                      │
│       ▼                                                                      │
│   vb2_core_dqbuf()                                                           │
│       │                                                                      │
│       ├─► 从 done 队列取出已填充的缓冲区                                      │
│       ├─► 填充 buf->bytesused, buf->timestamp                               │
│       └─► 标记缓冲区为 dequeued                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Linux Kernel - USB UVC 驱动                                                  │
│                                                                              │
│   drivers/media/usb/uvc/uvc_v4l2.c: uvc_v4l2_ioctl()                        │
│       │                                                                      │
│       ▼                                                                      │
│   drivers/media/usb/uvc/uvc_video.c: uvc_video_decode()                      │
│       │                                                                      │
│       ├─► 从 USB 端点接收数据                                                 │
│       ├─► 解码视频帧                                                          │
│       └─► 放入 videobuf2 缓冲区                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 硬件层                                                                       │
│                                                                              │
│   USB 摄像头 Sensor                                                          │
│       │                                                                      │
│       ├─► Sensor 采集图像                                                     │
│       ├─► ISP 处理 (可选)                                                    │
│       ├─► 编码 (MJPEG/YUV)                                                   │
│       └─► USB 批量传输到主机                                                  │
│                                                                              │
│   USB 主机控制器                                                             │
│       │                                                                      │
│       ├─► 接收 USB 数据包                                                    │
│       ├─► DMA 写入内核缓冲区                                                 │
│       └─► 触发中断通知 V4L2                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 返回 Python 层                                                               │
│                                                                              │
│   cv2.VideoCapture.read() 返回 (True, frame)                                │
│       │                                                                      │
│       ▼                                                                      │
│   frame: np.ndarray (H x W x 3, BGR, uint8)                                 │
│       │                                                                      │
│       ▼                                                                      │
│   obs_dict[cam_key] = frame                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 步骤二：推理阶段 (Inference)

### 3.1 Python 层调用

**源码位置**：`lerobot/record.py:276-296`

```python
# ============================================================
# 推理阶段 (Inference Phase)
# ============================================================
inference_start_t = time.perf_counter()

if policy is not None:
    action_values = predict_action(
        observation=observation_frame,
        policy=policy,
        device=get_safe_torch_device(policy.config.device),
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=policy.config.use_amp,
        task=single_task,
        robot_type=robot.robot_type,
    )

inference_end_t = time.perf_counter()
```

### 3.2 predict_action 实现

**源码位置**：`lerobot/utils/control_utils.py:125-175`

```python
def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    执行单步推理，从观测预测动作

    流程:
    1. 将 NumPy 数组转换为 PyTorch 张量
    2. 运行预处理器
    3. 输入策略模型
    4. 运行后处理器
    5. 返回动作
    """
    observation = copy(observation)

    with (
        torch.inference_mode(),  # 禁用梯度计算
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # ============================================================
        # 1. 数据预处理：NumPy → PyTorch Tensor
        # ============================================================
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])

            # 图像数据特殊处理
            if "image" in name:
                # uint8 [0,255] → float32 [0,1]
                observation[name] = observation[name].type(torch.float32) / 255
                # HWC → CHW (通道优先)
                observation[name] = observation[name].permute(2, 0, 1).contiguous()

            # 添加 batch 维度
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # ============================================================
        # 2. 运行预处理器
        # ============================================================
        observation = preprocessor(observation)

        # ============================================================
        # 3. 策略推理
        # ============================================================
        action = policy.select_action(observation)

        # ============================================================
        # 4. 运行后处理器
        # ============================================================
        action = postprocessor(action)

        # ============================================================
        # 5. 格式化输出
        # ============================================================
        action = action.squeeze(0).cpu()  # 移除 batch 维度，移到 CPU

        return action
```

### 3.3 策略推理层

**源码位置**：`lerobot/policies/pretrained.py` (概念位置)

```python
def select_action(self, observation: dict[str, Tensor]) -> Tensor:
    """
    从观测选择动作

    内部调用:
    1. policy.forward(observation)  → 神经网络前向传播
    """
    # 调用神经网络模型
    action = self.model(observation)

    return action
```

### 3.4 PyTorch 层

#### 3.4.1 torch.nn.Module.forward()

```
policy.model(observation)
    │
    ▼
torch.nn.Module.__call__()
    │
    ├─► _forward_pre_hooks
    │
    ├─► forward()  ← 用户定义的前向传播
    │       │
    │       ▼
    │   各层计算:
    │       │
    │       ├─► Conv2d: 卷积
    │       ├─► Linear: 全连接 (矩阵乘法)
    │       ├─► LayerNorm: 层归一化
    │       └─► ReLU/GELU: 激活函数
    │
    └─► _forward_hooks
```

#### 3.4.2 矩阵乘法并行化

**源码位置**：`aten/src/ATen/native/cpu/BlasKernel.cpp:375`

```cpp
// GEMM 矩阵乘法：C = A @ B
void gemm_transa_(...) {
    // 按 M 维度（行）并行分块
    parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
        // 每个 worker 线程处理 [begin, end) 行
        for (int64_t i = begin; i < end; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                // 计算点积
                C[i,j] = dot(A[i,:], B[:,j]);
            }
        }
    });
}
```

### 3.5 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 步骤二推理阶段                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Python 应用层
    │
    │ predict_action(observation, policy, ...)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ lerobot/utils/control_utils.py                                              │
│                                                                              │
│   1. 数据转换:                                                               │
│      observation[name] = torch.from_numpy(observation[name])                 │
│      observation[name] = observation[name].to(device)                        │
│                                                                              │
│   2. 预处理:                                                                  │
│      observation = preprocessor(observation)                                 │
│                                                                              │
│   3. 策略推理:                                                                │
│      action = policy.select_action(observation)                             │
│                                                                              │
│   4. 后处理:                                                                  │
│      action = postprocessor(action)                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PyTorch Python API (torch/)                                                 │
│                                                                              │
│   policy.select_action(obs)                                                 │
│       │                                                                      │
│       ▼                                                                      │
│   model.__call__(obs)  ← 触发 forward                                       │
│       │                                                                      │
│       ▼                                                                      │
│   model.forward(obs)                                                        │
│       │                                                                      │
│       ▼                                                                      │
│   各层计算:                                                                   │
│       │                                                                      │
│       ├─► Conv2d: 卷积操作                                                    │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   torch.nn.functional.conv2d()                                       │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   aten::conv2d → im2col + gemm                                       │
│       │                                                                      │
│       ├─► Linear: 全连接层                                                   │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   torch.nn.functional.linear()                                       │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   output = input @ weight.T + bias                                  │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   aten::addmm / aten::mm                                             │
│       │                                                                      │
│       ├─► LayerNorm: 层归一化                                                 │
│       │       │                                                              │
│       │       ▼                                                              │
│       │   aten::layer_norm                                                   │
│       │                                                                      │
│       └─► ReLU/GELU: 激活函数                                                │
│               │                                                              │
│               ▼                                                              │
│           aten::gelu / aten::relu                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PyTorch C++ ATen 层 (aten/src/ATen/)                                         │
│                                                                              │
│   aten::addmm (矩阵乘法 + 加法)                                               │
│       │                                                                      │
│       ▼                                                                      │
│   native/Blas.cpp: addmm_out_cpu()                                           │
│       │                                                                      │
│       ▼                                                                      │
│   native/cpu/BlasKernel.cpp: cpublas_gemm()                                  │
│       │                                                                      │
│       ▼                                                                      │
│   gemm_core_()                                                               │
│       │                                                                      │
│       └─► gemm_notrans_() / gemm_transa_() / ...                            │
│               │                                                              │
│               ▼                                                              │
│           parallel_for(0, m, 1, [&](begin, end) {...})                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PyTorch 并行调度层 (aten/src/ATen/Parallel-inl.h)                            │
│                                                                              │
│   parallel_for<>()                                                           │
│       │                                                                      │
│       ├─► 判断是否需要并行                                                     │
│       │       (numiter > grain_size && num_threads > 1)                      │
│       │                                                                      │
│       └─► invoke_parallel(begin, end, grain_size, f)                         │
│               │                                                              │
│               ├─► calc_num_tasks_and_chunk_size()                            │
│               │       → num_tasks=4, chunk_size=N/4                          │
│               │                                                              │
│               └─► _run_with_pool(task, num_tasks)                            │
│                       │                                                      │
│                       ├─► 主线程: task(0)                                     │
│                       └─► 线程池: task(1)~task(N)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PyTorch 线程池 (c10/core/thread_pool.cpp)                                    │
│                                                                              │
│   c10::ThreadPool                                                            │
│       │                                                                      │
│       ├─► threads_[0]: main_loop(0)  等待任务                               │
│       ├─► threads_[1]: main_loop(1)  等待任务                               │
│       ├─► threads_[2]: main_loop(2)  等待任务                               │
│       └─► threads_[3]: main_loop(3)  等待任务                               │
│                                                                              │
│   任务执行:                                                                   │
│       │                                                                      │
│       ├─► condition_.wait() 等待任务                                         │
│       ├─► 取出任务并执行                                                      │
│       └─► 更新 available_ 计数                                               │
│                                                                              │
│   同步机制:                                                                   │
│       │                                                                      │
│       ├─► mutex_ 保护任务队列                                                 │
│       ├─► condition_ 通知新任务                                               │
│       └─► completed_ 通知全部完成                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CPU 矩阵运算 (ARM64)                                                         │
│                                                                              │
│   矩阵乘法: C[i,j] = Σ A[i,k] * B[k,j]                                      │
│       │                                                                      │
│       ├─► 每个 worker 线程计算若干行                                           │
│       ├─► ARM64 NEON SIMD 优化                                               │
│       │       │                                                              │
│       │       ├─► fp16_gemv_notrans (FP16 矩阵向量乘)                         │
│       │       └─► bf16_dot_with_fp32_arith (BF16 点积)                        │
│       │                                                                      │
│       └─► 无数据竞争（每个线程写入不同行）                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 返回 Python 层                                                               │
│                                                                              │
│   action = policy.select_action(observation) 返回                            │
│       │                                                                      │
│       ▼                                                                      │
│   action: torch.Tensor (batch_size, action_dim)                             │
│       │                                                                      │
│       ▼                                                                      │
│   action = action.squeeze(0).cpu()  # 移除 batch 维度                        │
│       │                                                                      │
│       ▼                                                                      │
│   action: torch.Tensor (action_dim,)                                        │
│       │                                                                      │
│       ▼                                                                      │
│   act_processed_policy: dict[str, float]                                     │
│       {"motor1.pos": 123.4, "motor2.pos": 567.8, ...}                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 步骤三：动作阶段 (Action)

### 4.1 Python 层调用

**源码位置**：`lerobot/record.py:298-322`

```python
# ============================================================
# 动作阶段 (Action Phase)
# ============================================================
action_start_t = time.perf_counter()

if policy is not None:
    # 策略推理得到的动作
    robot_action_to_send = robot_action_processor((act_processed_policy, obs))
else:
    # 示教器得到的动作
    robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

# 发送动作到机器人
_sent_action = robot.send_action(robot_action_to_send)

action_end_t = time.perf_counter()
```

### 4.2 robot.send_action() 实现

**源码位置**：`lerobot/robots/so101_follower/so101_follower.py:269-295`

```python
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    """命令机器人移动到目标关节位置"""
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    # 提取位置目标
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() 
                if key.endswith(".pos")}

    # 可选：限制目标位置不能离当前位置太远
    if self.config.max_relative_target is not None:
        present_pos = self.bus.sync_read("Present_Position")
        goal_pos = ensure_safe_goal_position(goal_present_pos, 
                                               self.config.max_relative_target)

    # ============================================================
    # 同步写入目标位置到所有舵机
    # ============================================================
    self.bus.sync_write("Goal_Position", goal_pos)

    return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

### 4.3 sync_write 实现

**源码位置**：`lerobot/motors/motors_bus.py:1148-1220`

```python
def sync_write(self, data_name: str, values: Value | dict[str, Value], ...):
    """同时向多个舵机写入同一寄存器"""
    names = self._get_motors_list(motors)
    ids = [self.motors[motor].id for motor in names]

    # 获取寄存器地址和长度
    addr, length = get_address(self.model_ctrl_table, model, data_name)

    # 设置同步写入器
    self._setup_sync_writer(ids, addr, length)

    # 为每个舵机添加数据
    for motor, value in values.items():
        id_ = self.motors[motor].id
        data = self._serialize_data(value, length)
        self.sync_writer.changeParam(id_, data)

    # ============================================================
    # 发送数据包
    # ============================================================
    comm = self.sync_writer.txPacket()
```

### 4.4 完整调用链

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 步骤三动作阶段                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Python 应用层
    │
    │ robot.send_action(action)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ lerobot/robots/so101_follower/so101_follower.py                             │
│                                                                              │
│   send_action(action)                                                        │
│       │                                                                      │
│       ├─► 提取 goal_pos                                                      │
│       │                                                                      │
│       ├─► (可选) 安全检查                                                    │
│       │       present_pos = bus.sync_read("Present_Position")                │
│       │                                                                      │
│       └─► bus.sync_write("Goal_Position", goal_pos)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ lerobot/motors/motors_bus.py                                                │
│                                                                              │
│   sync_write(data_name, values)                                              │
│       │                                                                      │
│       ├─► 获取寄存器地址: addr, length                                        │
│       │                                                                      │
│       ├─► 设置同步写入器                                                      │
│       │       sync_writer.start_address = addr                              │
│       │       sync_writer.data_length = length                              │
│       │       sync_writer.addParam(id_)                                      │
│       │                                                                      │
│       └─► 发送数据包                                                          │
│               sync_writer.txPacket()                                         │
│                                                                              │
│   构造数据包 (Feetech 协议):                                                  │
│   ┌────────────────────────────────────────────────────────┐                │
│   │ 0xFF 0xFF | ID | LEN | SYNC_WRITE | ADDR | DATA | CRC │                │
│   └────────────────────────────────────────────────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PySerial → os.write() → glibc → syscall → Kernel TTY → USB → 舵机板          │
│ (与观测阶段舵机读取相同的路径，只是方向是 write)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 硬件层：舵机执行                                                              │
│                                                                              │
│   USB 舵机板收到数据包                                                        │
│       │                                                                      │
│       ├─► 解析 Goal_Position                                                 │
│       ├─► 写入舵机寄存器                                                      │
│       └─► 触发舵机 PID 控制器                                                 │
│               │                                                              │
│               ▼                                                              │
│           电机转动到目标位置                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 完整时序图

```
时间轴 ───────────────────────────────────────────────────────────────────────►

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Python     │  │  Kernel    │  │  Hardware  │  │  设备       │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       │ ============ 步骤一：观测 ============              │
       │                │                │                │
       │ robot.get_observation()          │                │
       │ ───────────────────────────────► │                │
       │                │                │                │
       │   [舵机读取]    │                │                │
       │   bus.sync_read()                │                │
       │ ───────────────────────────────► │                │
       │                │  write/read    │                │
       │                │ ──────────────► │                │
       │                │                │  USB 传输      │
       │                │                │ ─────────────► │
       │                │                │                │ 舵机响应
       │ ◄─────────────────────────────── │                │
       │   返回舵机位置  │                │                │
       │                │                │                │
       │   [摄像头读取]  │                │                │
       │   cam.async_read()               │                │
       │ ───────────────────────────────► │                │
       │                │  VIDIOC_DQBUF  │                │
       │                │ (可能 select)  │                │
       │                │ ──────────────► │                │
       │                │                │  摄像头采集    │
       │                │                │  DMA 传输      │
       │                │ ◄───────────── │                │
       │ ◄─────────────────────────────── │                │
       │   返回图像帧    │                │                │
       │                │                │                │
       │ ============ 步骤二：推理 ============              │
       │                │                │                │
       │ predict_action(observation)      │                │
       │ ───────────────────────────────► │                │
       │                │                │                │
       │   [PyTorch 推理]                │                │
       │   矩阵乘法并行  │                │                │
       │   CPU 计算     │                │                │
       │ ◄─────────────────────────────── │                │
       │   返回动作值    │                │                │
       │                │                │                │
       │ ============ 步骤三：动作 ============              │
       │                │                │                │
       │ robot.send_action(action)        │                │
       │ ───────────────────────────────► │                │
       │                │  write()        │                │
       │                │ ──────────────► │                │
       │                │                │  USB 传输      │
       │                │                │ ─────────────► │
       │                │                │                │ 舵机执行
       │ ◄─────────────────────────────── │                │
       │   动作已发送    │                │                │
       │                │                │                │
       │ ============ 等待下一帧 ============              │
       │                │                │                │
       │ busy_wait(1/fps - dt)            │                │
       │                │                │                │
       ▼                ▼                ▼                ▼
```

---

## 6. 源码文件索引

### Python 应用层
| 文件 | 主要内容 |
|------|----------|
| `lerobot/record.py` | 主循环 record_loop() |
| `lerobot/utils/control_utils.py` | predict_action() |
| `lerobot/robots/so101_follower/so101_follower.py` | get_observation(), send_action() |
| `lerobot/motors/motors_bus.py` | sync_read(), sync_write() |
| `lerobot/cameras/opencv/camera_opencv.py` | async_read() |

### OpenCV C++ 层
| 文件 | 主要内容 |
|------|----------|
| `opencv/modules/videoio/src/cap.cpp` | VideoCapture::read() |
| `opencv/modules/videoio/src/cap_v4l.cpp` | V4L2 后端 |

### PyTorch 层
| 文件 | 主要内容 |
|------|----------|
| `aten/src/ATen/Parallel.h` | 并行 API |
| `aten/src/ATen/Parallel-inl.h` | parallel_for 模板 |
| `c10/core/thread_pool.cpp` | ThreadPool 实现 |
| `aten/src/ATen/native/cpu/BlasKernel.cpp` | GEMM 实现 |

### Linux 内核层
| 文件 | 主要内容 |
|------|----------|
| `fs/read_write.c` | write/read 系统调用 |
| `drivers/tty/tty_io.c` | TTY 子系统 |
| `drivers/media/v4l2-core/v4l2-dev.c` | V4L2 ioctl |
| `drivers/usb/serial/usb-serial.c` | USB Serial 驱动 |

---

*文档生成时间：2026-03-29*