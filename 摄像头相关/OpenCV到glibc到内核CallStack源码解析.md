# OpenCV 到 glibc 到内核 CallStack 源码解析

本文只讨论 LeRobot 中 OpenCV 摄像头“取一帧”的底层路径。上层入口是
`OpenCVCamera._read_loop()` 中的 `self.videocapture.read()`，底层设备是 Linux
V4L2/UVC 摄像头。重点不是解释图像算法，而是说明这条调用链什么时候进内核、传了什么参数、等待发生在哪个队列上。

## 总调用链

在 Raspberry Pi OS / Linux 上，OpenCV 使用 V4L2 后端时，一次 `VideoCapture.read()` 的主路径可以写成：

```text
OpenCVCamera._read_loop()
  -> OpenCVCamera.read()
  -> cv2.VideoCapture.read()
  -> cv::VideoCapture::read(OutputArray image)
  -> cv::VideoCapture::grab()
  -> CvCaptureCAM_V4L::grabFrame()
  -> CvCaptureCAM_V4L::read_frame_v4l2()
  -> CvCaptureCAM_V4L::tryIoctl(VIDIOC_DQBUF, &buf)
  -> glibc ioctl(fd, VIDIOC_DQBUF, struct v4l2_buffer *)
  -> Linux sys_ioctl()
  -> v4l2_ioctl()
  -> video_usercopy()
  -> __video_do_ioctl()
  -> v4l_dqbuf()
  -> uvc_ioctl_dqbuf()
  -> uvc_dequeue_buffer()
  -> vb2_dqbuf()
  -> vb2_core_dqbuf()
  -> __vb2_get_done_vb()
  -> __vb2_wait_for_done_vb()
  -> wait_event_interruptible(q->done_wq, ...)
```

最重要的一点：`VIDIOC_DQBUF` 不是把整张图像从内核复制出来。它的语义是
“从 V4L2 完成队列里取出一个已经填好数据的 buffer 描述符”。在 OpenCV 的 MMAP 模式下，图像数据所在的 buffer 已经通过 `mmap` 映射到用户态；`DQBUF` 主要返回的是 `index`、`bytesused`、`timestamp`、`sequence` 等元数据。若摄像头格式是 MJPEG，JPEG 解码仍在用户态 OpenCV/libjpeg 中完成。

## 1. OpenCV 入口：read 等于 grab 加 retrieve

源码位置：`opencv/modules/videoio/src/cap.cpp`

```cpp
bool VideoCapture::read(OutputArray image)
{
    CV_INSTRUMENT_REGION();

    if (grab())
    {
        retrieve(image);
    } else {
        image.release();
    }
    return !image.empty();
}
```

逐句看：

| 语句 | 含义 |
| --- | --- |
| `OutputArray image` | OpenCV 的输出参数。Python 侧最终拿到的是 `np.ndarray`，中间由 OpenCV 的 Python binding 把 `cv::Mat` 包装出来。 |
| `grab()` | 只负责从摄像头后端抓取一帧。在 V4L2 后端中，它会走到 `VIDIOC_DQBUF`，把一个已完成的 kernel buffer 出队。 |
| `retrieve(image)` | 根据当前像素格式把 buffer 里的数据变成可用图像。MJPEG 摄像头会在这里走 JPEG 解码，得到 BGR `cv::Mat`。 |
| `image.release()` | `grab()` 失败时清空输出。Python 侧表现为 `ret=False` 或空帧。 |
| `return !image.empty()` | 返回本次读帧是否有效。 |

因此，`read()` 不是一个单纯系统调用，它是“内核取 buffer + 用户态解码/转换”的组合。

## 2. OpenCV V4L2：构造 struct v4l2_buffer 并 DQBUF

源码位置：`opencv/modules/videoio/src/cap_v4l.cpp`

```cpp
bool CvCaptureCAM_V4L::read_frame_v4l2()
{
    v4l2_buffer buf = v4l2_buffer();
    v4l2_plane mplanes[VIDEO_MAX_PLANES];

    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;

    if (V4L2_TYPE_IS_MULTIPLANAR(type)) {
        buf.m.planes = mplanes;
        buf.length = VIDEO_MAX_PLANES;
    }

    while (!tryIoctl(VIDIOC_DQBUF, &buf)) {
        int err = errno;
        if (err == EIO && !(buf.flags & (V4L2_BUF_FLAG_QUEUED | V4L2_BUF_FLAG_DONE))) {
            if (!tryIoctl(VIDIOC_QBUF, &buf))
                return false;
            continue;
        }
        returnFrame = false;
        return false;
    }

    CV_Assert(buf.index < req.count);
    ...
}
```

参数和数据含义：

| 名称 | 类型/值 | 含义 |
| --- | --- | --- |
| `buf` | `struct v4l2_buffer` | 用户态和 V4L2 内核驱动交换的 buffer 描述结构。调用前 OpenCV 填 `type/memory`，调用后内核填 `index/bytesused/timestamp/flags` 等字段。 |
| `buf.type` | `type`，通常是 `V4L2_BUF_TYPE_VIDEO_CAPTURE` | 说明这是视频采集队列。若是多平面格式，则会是 `V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE`。 |
| `buf.memory` | `V4L2_MEMORY_MMAP` | 表示 buffer 由内核分配，用户态通过 `mmap()` 映射访问。 |
| `buf.m.planes` | `struct v4l2_plane *` | 多平面格式才用；单平面 MJPEG/YUYV 通常不用这个数组。 |
| `VIDIOC_DQBUF` | `_IOWR('V', 17, struct v4l2_buffer)` | V4L2 出队命令。`_IOWR` 表示用户态传入结构，内核也会把结果写回该结构。 |
| `buf.index` | 内核写回 | 表示这次取到的是 OpenCV buffer 数组里的第几个 buffer。后续 `retrieve()` 会按这个 index 找到映射地址。 |

这里的 `DQBUF` 可以阻塞，也可以立即返回，取决于 fd 是否是非阻塞模式、队列中是否已有完成帧。OpenCV 的代码还在 `tryIoctl()` 里处理 `EAGAIN/EBUSY`。

## 3. OpenCV tryIoctl：ioctl 失败时用 select 等设备就绪

源码位置：`opencv/modules/videoio/src/cap_v4l.cpp`

```cpp
bool CvCaptureCAM_V4L::tryIoctl(unsigned long ioctlCode, void *parameter,
                                bool failIfBusy, int attempts) const
{
    while (true)
    {
        errno = 0;
        int result = ioctl(deviceHandle, ioctlCode, parameter);
        int err = errno;

        if (result != -1)
            return true;

        const bool isBusy = (err == EBUSY);
        if (isBusy && failIfBusy)
            return false;
        if (!(isBusy || errno == EAGAIN))
            return false;
        if (--attempts == 0)
            return false;

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(deviceHandle, &fds);

        struct timeval tv;
        tv.tv_sec = param_v4l_select_timeout;
        tv.tv_usec = 0;

        errno = 0;
        result = select(deviceHandle + 1, &fds, NULL, NULL, &tv);
        err = errno;
        ...
    }
}
```

逐句看关键参数：

| 语句 | 含义 |
| --- | --- |
| `ioctl(deviceHandle, ioctlCode, parameter)` | 进入 glibc `ioctl` 包装函数，最终进入 Linux `sys_ioctl`。在取帧路径中，`ioctlCode=VIDIOC_DQBUF`，`parameter=&buf`。 |
| `deviceHandle` | `/dev/video*` 的文件描述符。OpenCV 打开摄像头时得到。 |
| `EBUSY` | 设备忙，例如队列状态不允许当前操作。 |
| `EAGAIN` | 非阻塞模式下暂时没有帧可取。OpenCV 会用 `select()` 等设备变为可读后重试。 |
| `fd_set fds` | `select()` 的读 fd 集合，只放入这个摄像头 fd。 |
| `select(deviceHandle + 1, &fds, NULL, NULL, &tv)` | 等待 V4L2 fd 可读，超时默认 10 秒。Linux/glibc 下常见会落到 `pselect6` 系统调用。 |

所以摄像头后台线程有两类进内核方式：`ioctl(VIDIOC_DQBUF)` 是真正取 V4L2 buffer；`select/pselect6` 是在还没帧可取时睡眠等待 fd 可读。

## 4. glibc ioctl：变长参数包装成系统调用

源码位置：`glibc-2.42/sysdeps/unix/sysv/linux/ioctl.c`

```c
int
__ioctl (int fd, unsigned long int request, ...)
{
  va_list args;
  va_start (args, request);
  void *arg = va_arg (args, void *);
  va_end (args);

  int r;
  if (!__ioctl_arch (&r, fd, request, arg))
    {
      r = INTERNAL_SYSCALL_CALL (ioctl, fd, request, arg);

      if (__glibc_unlikely (INTERNAL_SYSCALL_ERROR_P (r)))
        {
          __set_errno (-r);
          return -1;
        }
    }
  return r;
}
weak_alias (__ioctl, ioctl)
```

这里没有复杂逻辑：

| 语句 | 含义 |
| --- | --- |
| `int fd` | 摄像头 fd，例如 `/dev/video0`。 |
| `unsigned long request` | 命令号，这里是 `VIDIOC_DQBUF`。 |
| `...` / `va_arg(args, void *)` | `ioctl` 是变长参数接口，第三个参数在这里被取成 `void *arg`，也就是 `struct v4l2_buffer *`。 |
| `__ioctl_arch` | 架构特定钩子。一般返回 false 后走通用 Linux syscall 路径。 |
| `INTERNAL_SYSCALL_CALL(ioctl, fd, request, arg)` | glibc 内部发起 `ioctl` 系统调用。在 ARM64 上最终是按系统调用 ABI 进入内核。 |
| `__set_errno(-r); return -1` | Linux syscall 失败返回负 errno，glibc 把它转换成用户态习惯的 `-1 + errno`。 |

这层只是 C 库包装层，不做 V4L2 语义处理。

## 5. glibc select：OpenCV 等待 fd 可读时会走 pselect6

源码位置：`glibc-2.42/sysdeps/unix/sysv/linux/select.c`

```c
__select64 (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
            struct __timeval64 *timeout)
{
  __time64_t s = timeout != NULL ? timeout->tv_sec : 0;
  int32_t us = timeout != NULL ? timeout->tv_usec : 0;
  ...
  struct __timespec64 ts64, *pts64 = NULL;
  if (timeout != NULL)
    {
      ts64.tv_sec = s;
      ts64.tv_nsec = ns;
      pts64 = &ts64;
    }

  int r = SYSCALL_CANCEL (pselect6_time64, nfds, readfds, writefds, exceptfds,
                          pts64, NULL);
  ...
  return r;
}
```

OpenCV 传入的是：

| 参数 | 实际含义 |
| --- | --- |
| `nfds` | `deviceHandle + 1`，这是 POSIX `select` 要求的最大 fd 加一。 |
| `readfds` | 只包含摄像头 fd，表示等待“可读”。 |
| `writefds` / `exceptfds` | `NULL`，不关心可写和异常集合。 |
| `timeout` | `struct timeval`，OpenCV 默认 10 秒。 |
| `pselect6_time64` | Linux 系统调用名。用户态写 `select()`，在现代 glibc/Linux 上常会由 `pselect6`/`pselect6_time64` 实现。 |

这解释了为什么 ftrace 或 strace 里经常看到 `pselect6`，而不是字面上的 `select`。

## 6. Linux sys_ioctl：从 fd 找到 file，再分发到 V4L2

源码位置：`linux/fs/ioctl.c`

```c
SYSCALL_DEFINE3(ioctl, unsigned int, fd, unsigned int, cmd, unsigned long, arg)
{
    struct fd f = fdget(fd);
    int error;

    if (!fd_file(f))
        return -EBADF;

    error = security_file_ioctl(fd_file(f), cmd, arg);
    if (error)
        goto out;

    error = do_vfs_ioctl(fd_file(f), fd, cmd, arg);
    if (error == -ENOIOCTLCMD)
        error = vfs_ioctl(fd_file(f), cmd, arg);

out:
    fdput(f);
    return error;
}
```

逐句看：

| 语句 | 含义 |
| --- | --- |
| `fdget(fd)` | 根据整数 fd 找到内核里的 `struct file`。用户态 fd 在内核中不能直接用，必须先解析。 |
| `fd_file(f)` | 取出 `struct file *`。如果 fd 无效，返回 `-EBADF`。 |
| `security_file_ioctl(...)` | LSM 安全检查，例如权限、安全策略。 |
| `do_vfs_ioctl(...)` | 先处理一些 VFS 通用 ioctl。 |
| `vfs_ioctl(...)` | 若不是通用 ioctl，则调用该文件的 `file_operations->unlocked_ioctl`。对于 `/dev/video*`，最终进 V4L2 的 `v4l2_ioctl` / `video_ioctl2`。 |
| `fdput(f)` | 释放 fd 引用。 |

## 7. V4L2 分发：video_usercopy 负责拷贝 ioctl 参数结构

源码位置：`linux/drivers/media/v4l2-core/v4l2-dev.c`

```c
static long v4l2_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct video_device *vdev = video_devdata(filp);
    int ret = -ENODEV;

    if (vdev->fops->unlocked_ioctl) {
        if (video_is_registered(vdev))
            ret = vdev->fops->unlocked_ioctl(filp, cmd, arg);
    } else
        ret = -ENOTTY;

    return ret;
}
```

`v4l2_ioctl` 做的事很薄：从 `file` 找 `video_device`，确认设备还注册着，然后继续调用该 video device 的 ioctl 实现。UVC 设备通常使用 V4L2 核心的 `video_ioctl2()`：

源码位置：`linux/drivers/media/v4l2-core/v4l2-ioctl.c`

```c
long
video_usercopy(struct file *file, unsigned int orig_cmd, unsigned long arg,
               v4l2_kioctl func)
{
    char sbuf[128];
    void *mbuf = NULL, *array_buf = NULL;
    void *parg = (void *)arg;
    ...
    unsigned int cmd = video_translate_cmd(orig_cmd);
    const size_t ioc_size = _IOC_SIZE(cmd);

    if (_IOC_DIR(cmd) != _IOC_NONE) {
        if (ioc_size <= sizeof(sbuf)) {
            parg = sbuf;
        } else {
            mbuf = kmalloc(ioc_size, GFP_KERNEL);
            if (NULL == mbuf)
                return -ENOMEM;
            parg = mbuf;
        }

        err = video_get_user((void __user *)arg, parg, cmd,
                             orig_cmd, &always_copy);
        if (err)
            goto out;
    }
    ...
    err = func(file, cmd, parg);
    ...
    if (video_put_user((void __user *)arg, parg, cmd, orig_cmd))
        err = -EFAULT;
out:
    kvfree(array_buf);
    kfree(mbuf);
    return err;
}
```

逐句看核心逻辑：

| 语句 | 含义 |
| --- | --- |
| `orig_cmd` | 用户态传进来的 ioctl 命令，OpenCV 这里是 `VIDIOC_DQBUF`。 |
| `arg` | 用户态指针地址，指向 OpenCV 栈上的 `struct v4l2_buffer buf`。 |
| `_IOC_SIZE(cmd)` | 从 ioctl 命令编码中解析参数结构大小。`VIDIOC_DQBUF` 的结构是 `struct v4l2_buffer`。 |
| `sbuf[128]` | 小结构优先用栈上临时缓冲区，避免分配。`struct v4l2_buffer` 可以走这条路径。 |
| `video_get_user(...)` | 把用户态 `struct v4l2_buffer` 拷贝到内核临时缓冲区。注意这只是拷贝“描述结构”，不是拷贝图像 payload。 |
| `func(file, cmd, parg)` | 调用真正的 V4L2 ioctl 分发函数，这里是 `__video_do_ioctl`。 |
| `video_put_user(...)` | 把内核填好的 `struct v4l2_buffer` 再拷回用户态，让 OpenCV 看到 `index/bytesused/timestamp`。 |

## 8. __video_do_ioctl：把 VIDIOC_DQBUF 映射到 v4l_dqbuf

源码位置：`linux/drivers/media/v4l2-core/v4l2-ioctl.c`

```c
static const struct v4l2_ioctl_info v4l2_ioctls[] = {
    ...
    IOCTL_INFO(VIDIOC_DQBUF, v4l_dqbuf, v4l_print_buffer, INFO_FL_QUEUE),
    ...
};
```

这张表说明 `VIDIOC_DQBUF` 的处理函数是 `v4l_dqbuf`。

```c
static long __video_do_ioctl(struct file *file,
        unsigned int cmd, void *arg)
{
    struct video_device *vfd = video_devdata(file);
    const struct v4l2_ioctl_ops *ops = vfd->ioctl_ops;
    const struct v4l2_ioctl_info *info;
    ...

    if (v4l2_is_known_ioctl(cmd)) {
        info = &v4l2_ioctls[_IOC_NR(cmd)];
        ...
    }
    ...
    if (info != &default_info) {
        ret = info->func(ops, file, fh, arg);
    } else if (!ops->vidioc_default) {
        ret = -ENOTTY;
    } else {
        ret = ops->vidioc_default(...);
    }
    ...
    return ret;
}
```

`info->func` 对 `VIDIOC_DQBUF` 来说就是 `v4l_dqbuf`：

```c
static int v4l_dqbuf(const struct v4l2_ioctl_ops *ops,
                    struct file *file, void *fh, void *arg)
{
    struct v4l2_buffer *p = arg;
    int ret = check_fmt(file, p->type);

    return ret ? ret : ops->vidioc_dqbuf(file, fh, p);
}
```

逐句看：

| 语句 | 含义 |
| --- | --- |
| `struct v4l2_ioctl_ops *ops` | 具体驱动注册给 V4L2 核心的操作表。UVC 驱动会在这里挂上 `uvc_ioctl_dqbuf`。 |
| `struct v4l2_buffer *p = arg` | `video_usercopy` 已经把用户态结构拷到了内核临时缓冲区，这里把它按 `v4l2_buffer` 解释。 |
| `check_fmt(file, p->type)` | 检查 buffer 类型是否与设备当前队列匹配。 |
| `ops->vidioc_dqbuf(file, fh, p)` | 进入具体驱动的 DQBUF 实现。UVC 摄像头对应 `uvc_ioctl_dqbuf`。 |

## 9. UVC 层：进入 uvc_video_queue

源码位置：`linux/drivers/media/usb/uvc/uvc_v4l2.c`

```c
static int uvc_ioctl_dqbuf(struct file *file, void *fh, struct v4l2_buffer *buf)
{
    struct uvc_fh *handle = fh;
    struct uvc_streaming *stream = handle->stream;

    if (!uvc_has_privileges(handle))
        return -EBUSY;

    return uvc_dequeue_buffer(&stream->queue, buf,
                              file->f_flags & O_NONBLOCK);
}
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `file` | `/dev/video*` 对应的 `struct file`。 |
| `fh` | V4L2 file handle，UVC 驱动里是 `struct uvc_fh *`。 |
| `buf` | 要填回给用户态的 `struct v4l2_buffer`。 |
| `handle->stream` | 当前 UVC video streaming 对象，里面包含格式、帧参数、队列、USB URB 解码函数等。 |
| `file->f_flags & O_NONBLOCK` | fd 是否非阻塞。阻塞模式下没有完成帧会睡眠；非阻塞模式下没有完成帧返回 `-EAGAIN`。 |

然后进入 UVC 队列：

源码位置：`linux/drivers/media/usb/uvc/uvc_queue.c`

```c
int uvc_dequeue_buffer(struct uvc_video_queue *queue, struct v4l2_buffer *buf,
                       int nonblocking)
{
    int ret;

    mutex_lock(&queue->mutex);
    ret = vb2_dqbuf(&queue->queue, buf, nonblocking);
    mutex_unlock(&queue->mutex);

    return ret;
}
```

`uvc_video_queue` 是 UVC 对 videobuf2 队列的包装：

```c
struct uvc_video_queue {
    struct vb2_queue queue;
    struct mutex mutex;         /* Protects queue */

    unsigned int flags;
    unsigned int buf_used;

    spinlock_t irqlock;         /* Protects irqqueue */
    struct list_head irqqueue;
};
```

这里的 `queue->mutex` 保护 UVC 队列操作；真正的“有没有帧完成、要不要睡眠”等逻辑在 `vb2_queue` 中。

## 10. videobuf2：done_list 没帧就睡在 done_wq 上

源码位置：`linux/drivers/media/common/videobuf2/videobuf2-v4l2.c`

```c
int vb2_dqbuf(struct vb2_queue *q, struct v4l2_buffer *b, bool nonblocking)
{
    int ret;

    if (vb2_fileio_is_active(q))
        return -EBUSY;

    if (b->type != q->type)
        return -EINVAL;

    ret = vb2_core_dqbuf(q, NULL, b, nonblocking);

    if (!q->is_output &&
        b->flags & V4L2_BUF_FLAG_DONE &&
        b->flags & V4L2_BUF_FLAG_LAST)
        q->last_buffer_dequeued = true;
    ...
}
```

`vb2_dqbuf` 先做类型检查，再调用 `vb2_core_dqbuf`。核心等待发生在 `__vb2_wait_for_done_vb`：

源码位置：`linux/drivers/media/common/videobuf2/videobuf2-core.c`

```c
static int __vb2_wait_for_done_vb(struct vb2_queue *q, int nonblocking)
{
    for (;;) {
        int ret;

        if (q->waiting_in_dqbuf)
            return -EBUSY;
        if (!q->streaming)
            return -EINVAL;
        if (q->error)
            return -EIO;
        if (q->last_buffer_dequeued)
            return -EPIPE;

        if (!list_empty(&q->done_list))
            break;

        if (nonblocking)
            return -EAGAIN;

        q->waiting_in_dqbuf = 1;
        call_void_qop(q, wait_prepare, q);

        ret = wait_event_interruptible(q->done_wq,
                !list_empty(&q->done_list) || !q->streaming ||
                q->error);

        call_void_qop(q, wait_finish, q);
        q->waiting_in_dqbuf = 0;
        if (ret)
            return ret;
    }
    return 0;
}
```

逐句看：

| 语句 | 含义 |
| --- | --- |
| `q->waiting_in_dqbuf` | 防止同一个队列上同时有多个阻塞 DQBUF。 |
| `!q->streaming` | 如果还没 `STREAMON` 或已经 `STREAMOFF`，不能等帧。 |
| `q->error` | 队列进入错误状态，返回 `-EIO`。 |
| `q->last_buffer_dequeued` | 最后一帧已经取过，继续取返回 `-EPIPE`。 |
| `!list_empty(&q->done_list)` | 如果完成队列里已经有 buffer，立即返回，不睡眠。 |
| `nonblocking` | 非阻塞 fd 没有完成帧时直接 `-EAGAIN`。 |
| `call_void_qop(q, wait_prepare, q)` | 睡眠前释放驱动锁，避免睡觉时堵住 streamoff/qbuf 等操作。 |
| `wait_event_interruptible(q->done_wq, ...)` | 这是关键等待点。当前线程进入内核等待队列，直到 `done_list` 非空、停止 streaming 或队列错误。信号可打断，所以叫 interruptible。 |
| `call_void_qop(q, wait_finish, q)` | 被唤醒后重新拿回驱动锁。 |

然后 `__vb2_get_done_vb` 从 `done_list` 拿第一个完成 buffer：

```c
ret = __vb2_wait_for_done_vb(q, nonblocking);
if (ret)
    return ret;

spin_lock_irqsave(&q->done_lock, flags);
*vb = list_first_entry(&q->done_list, struct vb2_buffer, done_entry);
...
```

最后 `vb2_core_dqbuf` 把这个完成 buffer 的元数据填进用户态要求的 `struct v4l2_buffer`：

```c
int vb2_core_dqbuf(struct vb2_queue *q, unsigned int *pindex, void *pb,
                   bool nonblocking)
{
    struct vb2_buffer *vb = NULL;
    int ret;

    ret = __vb2_get_done_vb(q, &vb, pb, nonblocking);
    if (ret < 0)
        return ret;

    switch (vb->state) {
    case VB2_BUF_STATE_DONE:
        break;
    case VB2_BUF_STATE_ERROR:
        break;
    default:
        return -EINVAL;
    }

    call_void_vb_qop(vb, buf_finish, vb);
    vb->prepared = 0;

    if (pindex)
        *pindex = vb->index;

    if (pb)
        call_void_bufop(q, fill_user_buffer, vb, pb);

    list_del(&vb->queued_entry);
    q->queued_count--;
    trace_vb2_dqbuf(q, vb);
    __vb2_dqbuf(vb);
    ...
    return 0;
}
```

`fill_user_buffer` 是把 `vb2_buffer` 的状态、长度、时间戳、index 等转换回 V4L2 uAPI 的 `struct v4l2_buffer`。这仍然是元数据层面的填充，不是 JPEG 解码。

## 11. done_list 是谁填的

UVC 驱动从 USB URB 收到摄像头数据后，会把当前 buffer 标记完成，然后交给 videobuf2：

源码位置：`linux/drivers/media/usb/uvc/uvc_queue.c`

```c
buf->state = buf->error ? UVC_BUF_STATE_ERROR : UVC_BUF_STATE_DONE;
vb2_set_plane_payload(&buf->buf.vb2_buf, 0, buf->bytesused);
vb2_buffer_done(&buf->buf.vb2_buf, buf->error ? VB2_BUF_STATE_ERROR :
                                        VB2_BUF_STATE_DONE);
```

videobuf2 的 `vb2_buffer_done()` 会把 buffer 挂进 `done_list`，再唤醒等在 `done_wq` 上的 DQBUF 线程：

```c
spin_lock_irqsave(&q->done_lock, flags);
if (state == VB2_BUF_STATE_QUEUED) {
    vb->state = VB2_BUF_STATE_QUEUED;
} else {
    list_add_tail(&vb->done_entry, &q->done_list);
    vb->state = state;
}
atomic_dec(&q->owned_by_drv_count);
spin_unlock_irqrestore(&q->done_lock, flags);

switch (state) {
case VB2_BUF_STATE_QUEUED:
    return;
default:
    wake_up(&q->done_wq);
    break;
}
```

这就把“摄像头硬件/USB 已经送来一帧”变成了“阻塞在 `VIDIOC_DQBUF` 的线程可以继续运行”。

## 12. 核心数据结构

### struct v4l2_buffer

源码位置：`linux/include/uapi/linux/videodev2.h`

```c
struct v4l2_buffer {
    __u32           index;
    __u32           type;
    __u32           bytesused;
    __u32           flags;
    __u32           field;
    struct timeval  timestamp;
    struct v4l2_timecode timecode;
    __u32           sequence;

    __u32           memory;
    union {
        __u32           offset;
        unsigned long   userptr;
        struct v4l2_plane *planes;
        __s32           fd;
    } m;
    __u32           length;
    __u32           reserved2;
    union {
        __s32           request_fd;
        __u32           reserved;
    };
};
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `index` | 第几个 V4L2 buffer。OpenCV 用它索引自己的 `buffers[]`。 |
| `type` | buffer 队列类型，例如视频采集。 |
| `bytesused` | payload 实际用了多少字节。MJPEG 时这里通常是压缩 JPEG 数据长度。 |
| `flags` | buffer 状态标志，例如 DONE、QUEUED、ERROR 等。 |
| `timestamp` | 驱动记录的帧时间戳。 |
| `sequence` | 帧序号。 |
| `memory` | buffer 内存模型，OpenCV 这里是 `V4L2_MEMORY_MMAP`。 |
| `m.offset` | MMAP 模式下用于 `mmap()` 的偏移/cookie。 |
| `m.planes` | 多平面模式下指向 `struct v4l2_plane` 数组。 |
| `length` | buffer 总长度，不等于有效 payload 长度。 |

### struct vb2_queue

源码位置：`linux/include/media/videobuf2-core.h`

```c
struct vb2_queue {
    unsigned int            type;
    unsigned int            io_modes;
    ...
    struct list_head        queued_list;
    unsigned int            queued_count;

    atomic_t                owned_by_drv_count;
    struct list_head        done_list;
    spinlock_t              done_lock;
    wait_queue_head_t       done_wq;

    unsigned int            streaming:1;
    unsigned int            error:1;
    unsigned int            waiting_in_dqbuf:1;
    unsigned int            last_buffer_dequeued:1;
    ...
};
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `queued_list` | 用户态已经 QBUF、等待驱动填充的 buffer 列表。 |
| `queued_count` | 当前 queued buffer 数。 |
| `owned_by_drv_count` | 驱动正在持有/填充的 buffer 数。 |
| `done_list` | 已经完成、可以被 DQBUF 出队给用户态的 buffer 列表。 |
| `done_lock` | 保护 `done_list` 的自旋锁。 |
| `done_wq` | 等待完成 buffer 的 wait queue。`wait_event_interruptible` 就睡在这里。 |
| `streaming` | V4L2 stream 是否开启。 |
| `waiting_in_dqbuf` | 是否已经有线程在阻塞 DQBUF。 |

队列初始化时会初始化这两个核心对象：

```c
INIT_LIST_HEAD(&q->done_list);
init_waitqueue_head(&q->done_wq);
```

### struct uvc_video_queue

源码位置：`linux/drivers/media/usb/uvc/uvcvideo.h`

```c
struct uvc_video_queue {
    struct vb2_queue queue;
    struct mutex mutex;         /* Protects queue */

    unsigned int flags;
    unsigned int buf_used;

    spinlock_t irqlock;         /* Protects irqqueue */
    struct list_head irqqueue;
};
```

UVC 不自己重新发明一套视频 buffer 队列，而是把 `vb2_queue` 包在 `uvc_video_queue` 里。UVC 层处理 USB 摄像头协议和 URB 数据到达；videobuf2 层负责通用的 QBUF/DQBUF、done_list、wait queue、mmap buffer 生命周期。

## 13. 和 LeRobot async_read 的关系

LeRobot 主线程调用的 `cam.async_read()` 通常不直接跑上面的整条内核链。它等待的是后台读线程已经准备好的 `latest_frame`：

```text
后台线程: VideoCapture.read() -> V4L2 DQBUF/select -> libjpeg 解码 -> latest_frame
主线程:   new_frame_event.wait() -> frame_lock -> 取 latest_frame
```

所以：

| 位置 | 等待性质 | 是否可能进内核 | 对 obs 抖动的影响 |
| --- | --- | --- | --- |
| `new_frame_event.wait()` | Python 线程事件等待 | 没有新帧时可能进入 pthread/futex 路径 | 如果后台线程慢，会把等待暴露到 obs 阶段 |
| `frame_lock` | 保护 `latest_frame` 的短临界区 | 只有竞争时可能进入内核 | 通常很短，影响小 |
| 后台 `VideoCapture.read()` | 摄像头设备等待和取帧 | `ioctl/select/pselect6` 会进入内核 | 主要受摄像头 fps、USB/V4L2 和解码耗时影响 |

这也是论文 3.1 里的关键解释：obs 表面上是在 `async_read()`，但摄像头真正的内核路径在后台线程中持续运行。主线程只是从线程间缓存取最新帧；当缓存没准备好时，等待才会转移到 `async_read()`。

## 14. 性能结论

1. `VIDIOC_DQBUF` 是控制面 + 元数据出队，不是整帧像素拷贝。
2. 若 done_list 已经有完成帧，`DQBUF` 可以很快返回；若没有完成帧，阻塞点在 `wait_event_interruptible(q->done_wq, ...)`。
3. OpenCV 在 `EAGAIN/EBUSY` 时还会用 `select()` 等待摄像头 fd 可读；glibc 下常见系统调用表现为 `pselect6`。
4. MJPEG 解码不在内核完成，而是在 `retrieve()` 之后的 OpenCV/libjpeg 用户态路径完成，这部分会消耗 ARM CPU。
5. 对 LeRobot 来说，`async_read()` 的等待和后台 `VideoCapture.read()` 的等待不是一回事：前者是线程同步等待，后者是设备/驱动 buffer 等待。

本文与 `OpenCV视频流内部实现.md` 的关系：该文档更偏 OpenCV 内部图像格式转换和 Python `np.ndarray` 生成过程；本文补足 `ioctl/select`、glibc、V4L2、UVC、videobuf2 到内核等待队列的源码级链路。
