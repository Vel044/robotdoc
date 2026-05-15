# VIDIOC_DQBUF 内核链路

本文从 `sys_ioctl()`（ioctl 系统调用入口）开始，追踪 `VIDIOC_DQBUF` 如何经过 V4L2 核心、UVC 驱动、videobuf2 队列，到达 `done_wq` 等待点，以及 USB URB 完成如何唤醒等待者。

---

## 图：ioctl syscall 到内核边界的分层路径

```
ioctl(fd, VIDIOC_DQBUF, &v4l2_buffer)    # OpenCV VideoCapture.read() 调用入口
  │
  ▼
VFS: sys_ioctl()                         # 为什么：Linux 一切皆文件，sys_ioctl 是 ioctl  syscall 入口
  │  fdget(fd) 把 int 转 struct file*，安全检查，分发到 f_op->unlocked_ioctl
  ▼
V4L2: video_ioctl2()                     # 什么：V4L2 核心的 unlocked_ioctl 实现；为什么：所有 V4L2 设备共用这套框架
  │  调用 video_usercopy 拷贝 v4l2_buffer 参数，__video_do_ioctl 查表分发
  ▼
V4L2: video_usercopy()                  # 什么：V4L2 ioctl 参数边界函数；为什么：用户态和内核态之间的拷贝
  │  copy_from_user(v4l2_buffer) → func() → copy_to_user(v4l2_buffer)
  ▼
V4L2: __video_do_ioctl()                # 什么：V4L2 ioctl 分发中枢；为什么：查 v4l2_ioctls[] 表找到 v4l_dqbuf
  │  VIDIOC_DQBUF → v4l_dqbuf() → uvc_ioctl_dqbuf()
  ▼
UVC: uvc_ioctl_dqbuf()                   # 什么：UVC 驱动的 DQBUF 实现；为什么：检查权限后入 videobuf2 队列 UVC=USB Video Class
  │  uvc_has_privileges() 检查，uvc_dequeue_buffer() 加锁调用 vb2
  ▼
vb2: vb2_core_dqbuf()                   # 什么：videobuf2 出队核心；为什么：统一视频 buffer 管理
  │  __vb2_get_done_vb() → __vb2_wait_for_done_vb()
  ▼
vb2: __vb2_wait_for_done_vb()            # 什么：vb2 等待完成 buffer；为什么：done_list 为空时必须等待
  │  wait_event_interruptible(done_wq, ...)  ← 核心等待点
========== kernel sleep boundary ==========
  │  进程进入 TASK_INTERRUPTIBLE 睡眠，等待 URB 完成回调唤醒
  ▼
URB 完成: uvc_video_complete(urb)        # USB 摄像头数据到达，产生完成中断
  │
  ▼
vb2: vb2_buffer_done()                   # 什么：标记 buffer 完成；为什么：把 buffer 移入 done_list
  │  list_add_tail(done_list) + wake_up(done_wq)  ← 唤醒点
========== kernel wake boundary ==========
  │
  ▼
DQBUF 返回: index/bytesused/timestamp   # v4l2_buffer 已填充元数据，返回用户态
```

---

## 1. ioctl 系统调用入口

`ioctl` 是 Linux 内核接收用户态 ioctl 请求的第一个函数。它通过 `fdget()` 把整数 fd 转换成 `struct file`，然后做安全检查和 VFS 分发。

### 1.1 `sys_ioctl()` — 系统调用入口

```c
// linux/fs/ioctl.c:893
// 函数名：SYSCALL_DEFINE3(ioctl) 即 sys_ioctl()
// 作用：ioctl 系统调用入口。从整数 fd 找到 struct file，安全检查后分发到 VFS 或驱动。
// 参数：fd=文件描述符（如 /dev/video0 的 fd）, cmd=ioctl 命令号（如 VIDIOC_DQBUF）, arg=命令参数指针
// 返回值：成功返回 0 或正值，失败返回负 errno（如 -EBADF, -EFAULT, -ENODEV）
SYSCALL_DEFINE3(ioctl, unsigned int, fd, unsigned int, cmd, unsigned long, arg)
{
    struct fd f = fdget(fd);                 // int fd → struct file *（增加引用计数）
    int error;

    if (!fd_file(f))                        // fd 无效或未对应打开的文件
        return -EBADF;                       // 返回 Bad file descriptor

    error = security_file_ioctl(fd_file(f), cmd, arg); // LSM 安全检查（SELinux/AppArmor 等）
    if (error)
        goto out;                           // 安全检查失败，跳到 out

    // 先尝试 VFS 通用 ioctl 处理（针对普通文件的 FIEMAP/FIBMAP 等）
    error = do_vfs_ioctl(fd_file(f), fd, cmd, arg);
    if (error == -ENOIOCTLCMD)               // VFS 没有处理这个 cmd（如 V4L2 专属 ioctl）
        error = vfs_ioctl(fd_file(f), cmd, arg); // 分发到 file_operations->unlocked_ioctl

out:
    fdput(f);                               // 释放 file 引用
    return error;
}
```

对 `/dev/video0` 来说：
- `file->f_op->unlocked_ioctl` 指向 `video_ioctl2()`（V4L2 核心）
- `do_vfs_ioctl()` 不处理 V4L2 专属 ioctl，返回 `-ENOIOCTLCMD`
- 最终进入 `vfs_ioctl()` → `video_ioctl2()`

---

## 2. 完整调用栈

从 OpenCV 的 `VideoCapture.read()` 到摄像头硬件，链路跨越用户态和内核态两大边界。内核外是 OpenCV → libjpeg → glibc 的层层封装（Python 侧不在本链路范围）；内核内是 sys_ioctl → V4L2 → UVC → vb2 的逐级分发。本节用两张调用栈图分别展示内核态的完整函数调用路径。

### 2.1 内核态调用栈（主路径：取帧请求到睡眠）

```text
Linux SYSCALL_DEFINE3(ioctl, fd, cmd, arg)
│  内核 syscall 入口：接收用户态 fd、ioctl cmd 和参数指针（struct v4l2_buffer *）
└── sys_ioctl(fd, VIDIOC_DQBUF, arg) [linux/fs/ioctl.c:893]
    ├── fdget(fd)                            → int fd 转 struct file *
    ├── security_file_ioctl()                → LSM 安全检查
    └── vfs_ioctl(file, cmd, arg)           → V4L2 不走 do_vfs_ioctl
        └── video_ioctl2(file, cmd, arg)   → V4L2 核心 unlocked_ioctl
            └── video_usercopy(file, cmd, arg, __video_do_ioctl)
                ├── video_get_user()                    → copy_from_user(v4l2_buffer)
                │   （用户态 struct v4l2_buffer → 内核临时缓冲区）
                └── __video_do_ioctl(file, cmd, parg)
                    ├── v4l_dqbuf(ops, file, fh, p)    → VIDIOC_DQBUF 处理函数
                    │   ├── check_fmt(file, p->type)
                    │   └── ops->vidioc_dqbuf(file, fh, p)
                    └── uvc_ioctl_dqbuf(file, fh, buf)
                        ├── uvc_has_privileges(handle)  → 权限检查
                        └── uvc_dequeue_buffer(&stream->queue, buf, nonblocking)
                            └── vb2_dqbuf(&queue->queue, buf, nonblocking)
                                ├── vb2_fileio_is_active() → 不是文件 IO 模式
                                ├── b->type != q->type  → 类型检查
                                └── vb2_core_dqbuf(q, NULL, b, nonblocking)
                                    └── __vb2_get_done_vb(q, &vb, pb, nonblocking)
                                        └── __vb2_wait_for_done_vb(q, nonblocking)
                                            └── wait_event_interruptible(done_wq, ...)
                                                │  核心等待点：done_list 为空时，进程在此睡眠
                                                ▼  【进程进入 TASK_INTERRUPTIBLE】
```

### 2.2 内核态调用栈（唤醒路径：URB 完成到 DQBUF 返回）

```text
xhci_irq()                                → USB 主机控制器中断
│  xHCI 硬件产生完成中断
└── usb_hcd_giveback_urb()               → URB 归还回调
    └── uvc_video_complete(urb)           → UVC URB 完成处理 [linux/drivers/media/usb/uvc/uvc_video.c]
        ├── urb->status == 0             → 传输成功
        └── uvc_video_decode(urb)        → 解析 UVC 数据（MJPEG 帧）
            └── uvc_queue_buffer()        → 写入 vb2 buffer
                ├── buf->state = UVC_BUF_STATE_DONE
                └── vb2_buffer_done(vb, VB2_BUF_STATE_DONE)
                    ├── list_add_tail(&vb->done_entry, &q->done_list)
                    │   （buffer 从 drivers 列表移到完成队列）
                    └── wake_up(&q->done_wq)
                        │  核心唤醒点：唤醒等待在 done_wq 上的 DQBUF 进程
                        ▼  【进程从 TASK_INTERRUPTIBLE 唤醒】

// 进程被唤醒后继续执行
__vb2_wait_for_done_vb()                  → wait_event_interruptible 返回（条件满足）
    └── return 0
__vb2_get_done_vb()                       → 从 done_list 取 buffer
    ├── list_first_entry(&done_list)     → 取第一个完成 buffer
    └── list_del(&vb->done_entry)         → 从 done_list 移除
vb2_core_dqbuf()                          → 填充元数据
    ├── vb->state = VB2_BUF_STATE_DONE
    ├── fill_user_buffer(vb, pb)         → 填充 index/bytesused/timestamp
    └── __vb2_dqbuf(vb)                  → buffer 回到空闲池
video_usercopy()                          → video_put_user
    └── copy_to_user()                    → v4l2_buffer 回到用户态
返回用户态 ioctl()                         → 返回 0（成功）
```

---

## 3. VFS 与 V4L2 核心层

### 2.1 `vfs_ioctl()` — VFS ioctl 分发

```c
// linux/fs/ioctl.c:44
// 函数名：vfs_ioctl()
// 作用：VFS 层 ioctl 分发。调用文件类型的 unlocked_ioctl 实现。
// 参数：filp=struct file *, cmd=ioctl 命令号, arg=参数指针
// 返回值：驱动处理结果，或 -ENOTTY（驱动未实现）
long vfs_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    int error = -ENOTTY;                    // 默认错误：驱动没实现

    if (!filp->f_op->unlocked_ioctl)        // 确认驱动实现了 unlocked_ioctl
        goto out;

    error = filp->f_op->unlocked_ioctl(filp, cmd, arg); // 调驱动实现
    if (error == -ENOIOCTLCMD)              // 驱动也不知道这个 cmd
        error = -ENOTTY;                    // 返回 not a tty（ioctl 术语）
out:
    return error;
}
```

对 V4L2 设备，`filp->f_op->unlocked_ioctl` 指向 `video_ioctl2()`。

### 2.2 `video_usercopy()` — 拷贝 ioctl 参数结构

`video_usercopy()` 是 V4L2 核心层最重要的边界函数。它负责：
1. 把用户态的 `struct v4l2_buffer` 拷贝到内核临时缓冲区（`video_get_user()` = `copy_from_user`）
2. 调用驱动的 ioctl 处理函数
3. 把内核填好的结果拷贝回用户态（`video_put_user()` = `copy_to_user`）

注意：这只是拷贝 buffer **描述结构**，不是像素数据。

```c
// linux/drivers/media/v4l2-core/v4l2-ioctl.c
// 函数名：video_usercopy()
// 作用：V4L2 核心 ioctl 参数边界函数。把用户态 struct v4l2_buffer 拷贝到内核，
//       调用驱动处理函数，再把结果拷贝回用户态。
// 参数：file=struct file *, orig_cmd=原始 ioctl 命令号（如 VIDIOC_DQBUF）,
//       arg=用户态 struct v4l2_buffer * 指针, func=实际处理函数（__video_do_ioctl）
// 返回值：成功返回 0，失败返回负 errno
long
video_usercopy(struct file *file, unsigned int orig_cmd, unsigned long arg,
               v4l2_kioctl func)
{
    char sbuf[128];                          // 小结构用栈上缓冲区（struct v4l2_buffer 可用）
    void *mbuf = NULL, *array_buf = NULL;
    void *parg = (void *)arg;                // 指向用户态 struct v4l2_buffer
    ...
    unsigned int cmd = video_translate_cmd(orig_cmd); // 命令翻译（如有 compat 模式）
    const size_t ioc_size = _IOC_SIZE(cmd); // 从 ioctl cmd 编码解析参数结构大小

    // 分配内核临时缓冲区，拷贝用户态数据
    if (_IOC_DIR(cmd) != _IOC_NONE) {
        if (ioc_size <= sizeof(sbuf)) {
            parg = sbuf;                     // 小于 128 字节用栈上缓冲区，避免 kmalloc
        } else {
            mbuf = kmalloc(ioc_size, GFP_KERNEL); // 大结构需要动态分配
            if (NULL == mbuf)
                return -ENOMEM;
            parg = mbuf;
        }

        // 关键拷贝：用户态 → 内核态（v4l2_buffer 描述结构，不是像素数据）
        err = video_get_user((void __user *)arg, parg, cmd,
                             orig_cmd, &always_copy);
        if (err)
            goto out;
    }

    // 调用驱动处理函数 __video_do_ioctl → v4l_dqbuf → uvc_ioctl_dqbuf
    err = func(file, cmd, parg);

    // 关键拷贝：内核态 → 用户态（元数据：index/bytesused/timestamp/flags）
    if (video_put_user((void __user *)arg, parg, cmd, orig_cmd))
        err = -EFAULT;

out:
    kvfree(array_buf);
    kfree(mbuf);                             // 释放临时分配
    return err;
}
```

### 2.3 `__video_do_ioctl()` — V4L2 ioctl 查表分发

`VIDIOC_DQBUF` 在 V4L2 的 ioctl 表里映射到 `v4l_dqbuf`：

```c
// linux/drivers/media/v4l2-core/v4l2-ioctl.c
// V4L2 ioctl 命令表：cmd 编号 → 处理函数的映射
static const struct v4l2_ioctl_info v4l2_ioctls[] = {
    ...
    // VIDIOC_DQBUF 对应的处理函数是 v4l_dqbuf，输出打印函数是 v4l_print_buffer
    IOCTL_INFO(VIDIOC_DQBUF, v4l_dqbuf, v4l_print_buffer, INFO_FL_QUEUE),
    ...
};
```

`__video_do_ioctl()` 通过查表找到 `v4l_dqbuf`：

```c
// linux/drivers/media/v4l2-core/v4l2-ioctl.c
// 函数名：__video_do_ioctl()
// 作用：V4L2 ioctl 分发中枢。通过查表把 ioctl 命令分发给具体处理函数。
// 参数：file=struct file *, cmd=ioctl 命令号（如 VIDIOC_DQBUF）, arg=内核临时缓冲区的 v4l2_buffer *
// 返回值：处理结果
static long __video_do_ioctl(struct file *file, unsigned int cmd, void *arg)
{
    struct video_device *vfd = video_devdata(file); // 从 file* 提取 video_device*
    const struct v4l2_ioctl_ops *ops = vfd->ioctl_ops; // UVC 驱动注册的操作表
    const struct v4l2_ioctl_info *info;
    ...
    enum v4l2_ioctl_info default_info;      // 未知 ioctl 的默认处理

    // 用 cmd 编号查 v4l2_ioctls 表
    if (v4l2_is_known_ioctl(cmd)) {
        info = &v4l2_ioctls[_IOC_NR(cmd)];  // _IOC_NR 从 cmd 编码中提取编号
        ...
    }
    ...
    if (info != &default_info) {
        ret = info->func(ops, file, fh, arg); // 调用 v4l_dqbuf(ops, file, fh, arg)
    } else if (!ops->vidioc_default) {
        ret = -ENOTTY;                       // 驱动没实现默认处理
    } else {
        ret = ops->vidioc_default(...);      // 驱动特定 ioctl
    }
    ...
    return ret;
}
```

### 2.4 `v4l_dqbuf()` — VIDIOC_DQBUF 处理函数

```c
// linux/drivers/media/v4l2-core/v4l2-ioctl.c
// 函数名：v4l_dqbuf()
// 作用：VIDIOC_DQBUF 的 V4L2 核心处理函数。检查 buffer type，然后调用驱动实现。
// 参数：ops=V4L2 ioctl 操作表（UVC 驱动注册）, file=struct file *,
//       fh=驱动文件句柄, arg=内核临时缓冲区的 struct v4l2_buffer *
// 返回值：成功返回 0，失败返回负 errno
static int v4l_dqbuf(const struct v4l2_ioctl_ops *ops,
                     struct file *file, void *fh, void *arg)
{
    struct v4l2_buffer *p = arg;              // 内核临时缓冲区的 v4l2_buffer
    int ret = check_fmt(file, p->type);     // 确认 buffer type 与设备当前格式匹配

    // 调用 UVC 驱动的 uvc_ioctl_dqbuf
    return ret ? ret : ops->vidioc_dqbuf(file, fh, p);
}
```

`ops->vidioc_dqbuf` 对 UVC 摄像头就是 `uvc_ioctl_dqbuf`。

---

## 3. UVC 驱动层

### 3.1 `uvc_ioctl_dqbuf()` — 权限检查后入 vb2 队列

```c
// linux/drivers/media/usb/uvc/uvc_v4l2.c
// 函数名：uvc_ioctl_dqbuf()
// 作用：UVC 驱动的 VIDIOC_DQBUF 实现。检查权限，然后调用 vb2 队列出队。
// 参数：file=struct file *, fh=UVC 文件句柄（包含 stream 信息）, buf=struct v4l2_buffer *
// 返回值：成功返回 0，失败返回负 errno
static int uvc_ioctl_dqbuf(struct file *file, void *fh, struct v4l2_buffer *buf)
{
    struct uvc_fh *handle = fh;              // UVC 驱动的 file handle，包含流信息
    struct uvc_streaming *stream = handle->stream; // 当前视频流

    // 检查当前进程是否有权访问该流（如设备正被其他进程独占）
    if (!uvc_has_privileges(handle))
        return -EBUSY;

    // nonblocking 来自 file->f_flags & O_NONBLOCK，控制阻塞/非阻塞模式
    return uvc_dequeue_buffer(&stream->queue, buf,
                              file->f_flags & O_NONBLOCK);
}
```

### 3.2 `uvc_dequeue_buffer()` — 加锁后调用 vb2

```c
// linux/drivers/media/usb/uvc/uvc_queue.c
// 函数名：uvc_dequeue_buffer()
// 作用：UVC 队列出队 wrapper。加锁保护，然后调用 videobuf2 核心出队。
// 参数：queue=uvc_video_queue *, buf=struct v4l2_buffer *, nonblocking=是否非阻塞
// 返回值：vb2_dqbuf 的返回值
int uvc_dequeue_buffer(struct uvc_video_queue *queue,
                       struct v4l2_buffer *buf, int nonblocking)
{
    int ret;

    mutex_lock(&queue->mutex);               // 保护 UVC 队列并发操作
    ret = vb2_dqbuf(&queue->queue, buf, nonblocking); // vb2 核心出队
    mutex_unlock(&queue->mutex);

    return ret;
}
```

`uvc_video_queue` 是 UVC 对 videobuf2 的简单包装：

```c
// linux/drivers/media/usb/uvc/uvcvideo.h
// struct uvc_video_queue — UVC 视频队列包装
//   queue       : 通用 videobuf2 队列（核心数据结构）
//   mutex       : 保护队列操作的互斥锁
//   irqlock     : 中断上下文中保护 irqqueue 的自旋锁
//   irqqueue    : 中断上下文中入队的 buffer 链表
struct uvc_video_queue {
    struct vb2_queue queue;          // 通用 videobuf2 队列
    struct mutex mutex;              // 保护队列操作
    unsigned int flags;
    unsigned int buf_used;
    spinlock_t irqlock;              // 中断上下文中保护 irqqueue
    struct list_head irqqueue;       // 中断上下文中入队的 buffer
};
```

---

## 4. videobuf2 核心层

videobuf2（vb2）是 Linux 媒体子系统的通用视频 buffer 管理框架。UVC 驱动把 `vb2_queue` 包在 `uvc_video_queue` 里，不自己发明队列逻辑。

### 4.1 `vb2_dqbuf()` — 类型检查后进入核心

```c
// linux/drivers/media/common/videobuf2/videobuf2-v4l2.c
// 函数名：vb2_dqbuf()
// 作用：vb2 层 VIDIOC_DQBUF 入口。检查队列状态和 buffer type，然后调用核心出队。
// 参数：q=struct vb2_queue *, b=struct v4l2_buffer *, nonblocking=是否非阻塞
// 返回值：成功返回 0，失败返回负 errno
int vb2_dqbuf(struct vb2_queue *q, struct v4l2_buffer *b, bool nonblocking)
{
    int ret;

    if (vb2_fileio_is_active(q))             // 文件 IO 模式（如 mmap 测试程序）活跃时不允许
        return -EBUSY;

    if (b->type != q->type)                  // buffer type 必须匹配（如 V4L2_BUF_TYPE_VIDEO_CAPTURE）
        return -EINVAL;

    ret = vb2_core_dqbuf(q, NULL, b, nonblocking); // 调用 vb2 核心出队

    // 处理最后一帧标志（V4L2_BUF_FLAG_LAST）
    if (!q->is_output &&                      // capture 模式
        b->flags & V4L2_BUF_FLAG_DONE &&
        b->flags & V4L2_BUF_FLAG_LAST)
        q->last_buffer_dequeued = true;       // 通知流已结束
    ...
}
```

### 4.2 `vb2_core_dqbuf()` — 整体流程控制

`vb2_core_dqbuf()` 是出队流程的顶层协调者。它先调用 `__vb2_get_done_vb()` 等待或获取一个完成 buffer，然后把 buffer 元数据填回用户态结构：

```c
// linux/drivers/media/common/videobuf2/videobuf2-core.c
// 函数名：vb2_core_dqbuf()
// 作用：vb2 出队核心流程控制。等待（或直接取）完成 buffer，填充元数据回用户态。
// 参数：q=struct vb2_queue *, pindex=输出参数 buffer 索引,
//       pb=struct v4l2_buffer *（内核临时缓冲区）, nonblocking=是否非阻塞
// 返回值：成功返回 0，失败返回负 errno
int vb2_core_dqbuf(struct vb2_queue *q, unsigned int *pindex, void *pb,
                   bool nonblocking)
{
    struct vb2_buffer *vb = NULL;
    int ret;

    // __vb2_get_done_vb 负责等待（如果需要）和从 done_list 取 buffer
    ret = __vb2_get_done_vb(q, &vb, pb, nonblocking);
    if (ret < 0)
        return ret;                           // 错误码：-EAGAIN（nonblocking）等

    // 检查 buffer 状态
    switch (vb->state) {
    case VB2_BUF_STATE_DONE:                  // 传输成功完成
        break;
    case VB2_BUF_STATE_ERROR:                 // 传输错误
        break;
    default:                                  // 异常状态
        return -EINVAL;
    }

    // 驱动后处理回调（如 timestamp 从芯片时间转成系统时间）
    call_void_vb_qop(vb, buf_finish, vb);
    vb->prepared = 0;

    if (pindex)
        *pindex = vb->index;                  // 回传 buffer 索引

    // 填充用户态 v4l2_buffer 结构（元数据：index/bytesused/timestamp/flags）
    if (pb)
        call_void_bufop(q, fill_user_buffer, vb, pb);

    // 从 queued 链表移除，计数器减一
    list_del(&vb->queued_entry);
    q->queued_count--;
    trace_vb2_dqbuf(q, vb);                   // ftrace 调试点
    __vb2_dqbuf(vb);                         // 内部清理，buffer 回到空闲池
    ...
    return 0;
}
```

### 4.3 `__vb2_get_done_vb()` — 从 done_list 取 buffer

`__vb2_get_done_vb()` 先调用 `__vb2_wait_for_done_vb()` 等待或检查完成 buffer，然后从 `done_list` 取出第一个条目：

```c
// linux/drivers/media/common/videobuf2/videobuf2-core.c
// 函数名：__vb2_get_done_vb()
// 作用：从 done_list 取一个完成 buffer。先等待（如果需要），然后出队。
// 参数：q=struct vb2_queue *, vb=输出参数指向取到的 buffer *,
//       pb=struct v4l2_buffer *, nonblocking=是否非阻塞
// 返回值：成功返回 0，失败返回负 errno
static int __vb2_get_done_vb(struct vb2_queue *q, struct vb2_buffer **vb,
                              void *pb, bool nonblocking)
{
    int ret;

    // 如果 done_list 有 buffer 立即返回；否则可能睡眠等待
    ret = __vb2_wait_for_done_vb(q, nonblocking);
    if (ret)
        return ret;                           // -EAGAIN 或错误

    // 取出一个完成 buffer
    spin_lock_irqsave(&q->done_lock, flags);
    // done_list 是 struct list_head链表，vb2_buffer 通过 done_entry 节点链接
    *vb = list_first_entry(&q->done_list, struct vb2_buffer, done_entry);
    // 把 buffer 从 done_list 移除
    list_del(&(*vb)->done_entry);
    // 驱动持有计数减一（之前在 URB 填充时 increment）
    atomic_dec(&q->owned_by_drv_count);
    spin_unlock_irqrestore(&q->done_lock, flags);
    ...
}
```

### 4.4 `__vb2_wait_for_done_vb()` — 核心等待点

这是 `VIDIOC_DQBUF` 在 `done_list` 为空时的实际睡眠点：

```c
// linux/drivers/media/common/videobuf2/videobuf2-core.c
// 函数名：__vb2_wait_for_done_vb()
// 作用：等待一个完成 buffer 入 done_list。如果 done_list 非空立即返回；
//       否则在 done_wq 上睡眠，直到 URB 完成回调唤醒。
// 参数：q=struct vb2_queue *, nonblocking=是否非阻塞模式
// 返回值：成功返回 0，-EAGAIN（非阻塞无帧），被信号打断返回 -ERESTARTSYS
static int __vb2_wait_for_done_vb(struct vb2_queue *q, int nonblocking)
{
    for (;;) {                                // 循环处理 spurious wakeup
        int ret;

        if (q->waiting_in_dqbuf)              // 防止重复等待（dqbuf 嵌套）
            return -EBUSY;
        if (!q->streaming)                    // streaming 未开启（还没 STREAMON）
            return -EINVAL;
        if (q->error)                         // 队列处于错误状态
            return -EIO;
        if (q->last_buffer_dequeued)          // 流已正常结束
            return -EPIPE;

        // done_list 非空，说明有完成帧，直接返回（不睡眠）
        if (!list_empty(&q->done_list))
            break;

        if (nonblocking)                      // 非阻塞模式，无帧可取
            return -EAGAIN;

        q->waiting_in_dqbuf = 1;              // 标记有人正在等待
        // 驱动睡眠前准备：释放持有的锁，避免阻塞 qbuf/streamoff 等操作
        call_void_qop(q, wait_prepare, q);

        // 关键等待：进程在此睡眠，直到 done_list 非空、streaming 停止或队列错误
        // wait_event_interruptible 是可中断睡眠，信号（如 Ctrl+C）可打断
        ret = wait_event_interruptible(q->done_wq,
                !list_empty(&q->done_list) || !q->streaming ||
                q->error);

        // 唤醒后重新获取驱动锁
        call_void_qop(q, wait_finish, q);
        q->waiting_in_dqbuf = 0;              // 清除等待标记
        if (ret)                              // 信号打断（如 SIGINT）
            return ret;
    }
    return 0;
}
```

### 4.5 `struct vb2_queue` — 核心数据结构

```c
// linux/include/media/videobuf2-core.h
// struct vb2_queue — videobuf2 核心队列结构
//   type             : 队列类型（如 V4L2_BUF_TYPE_VIDEO_CAPTURE）
//   io_modes         : 支持的 IO 模式（VB2_MMAP / VB2_USERPTR / VB2_DMABUF）
//   queued_list      : 用户态已 QBUF、等待驱动填充的 buffer 链表
//   queued_count     : 当前 queued buffer 数量
//   owned_by_drv_count : 驱动正在持有/填充的 buffer 数量（atomic）
//   done_list        : 已完成、可以被 DQBUF 的 buffer 链表
//   done_lock        : 保护 done_list 的自旋锁
//   done_wq          : DQBUF 等待队列头（关键），wait_event_interruptible 在此睡眠
//   streaming        : STREAMON 后为 1
//   error            : 队列错误标志
//   waiting_in_dqbuf : 已有线程在阻塞 DQBUF
//   last_buffer_dequeued : 流已结束
struct vb2_queue {
    unsigned int            type;
    unsigned int            io_modes;
    struct list_head        queued_list;
    unsigned int            queued_count;

    atomic_t                owned_by_drv_count;
    struct list_head        done_list;
    spinlock_t              done_lock;
    wait_queue_head_t       done_wq;          // 关键等待队列头

    unsigned int            streaming:1;
    unsigned int            error:1;
    unsigned int            waiting_in_dqbuf:1;
    unsigned int            last_buffer_dequeued:1;
    ...
};
```

队列初始化时：

```c
// linux/drivers/media/common/videobuf2/videobuf2-core.c（vb2_queue_init）
INIT_LIST_HEAD(&q->done_list);          // 初始化完成链表
init_waitqueue_head(&q->done_wq);      // 初始化等待队列头
```

---

## 5. done_list 是谁填的 — URB 完成到 buffer 可用

### 5.1 URB 完成回调链

摄像头数据通过 USB bulk IN 传输。xHCI 把数据 DMA 到 mmap buffer 后，产生完成中断，UVC 驱动在中断上下文处理：

```c
// linux/drivers/media/usb/uvc/uvc_video.c
// 函数名：uvc_video_complete()
// 作用：USB URB 完成回调。处理摄像头数据到达，触发解码和 buffer 填充。
// 参数：urb=struct urb *（USB 请求块）
static void uvc_video_complete(struct urb *urb)
{
    struct uvc_streaming *stream = (struct uvc_streaming *)urb->context;
    struct uvc_video_queue *queue = &stream->queue;
    ...
    switch (urb->status) {
    case 0:                                   // 传输成功
        ret = uvc_video_decode(urb);         // 解析 UVC 数据，写入 vb2 buffer
        break;
    case -ENOENT:                            // URB 被取消（设备拔除）
        ...
    }
    ...
}
```

`uvc_video_decode()` 把 URB 数据写入 vb2 buffer，然后调用 `vb2_buffer_done()`：

```c
// linux/drivers/media/usb/uvc/uvc_queue.c
// 函数名：uvc_queue_buffer_direct()（内部路径）
// 作用：把填充好的 vb2 buffer 标记为完成，通知 videobuf2
// 关键调用：vb2_buffer_done(buf, VB2_BUF_STATE_DONE)
buf->state = buf->error ? UVC_BUF_STATE_ERROR : UVC_BUF_STATE_DONE;
// 设置 buffer 的实际 payload 大小（MJPEG 压缩数据长度）
vb2_set_plane_payload(&buf->buf.vb2_buf, 0, buf->bytesused);
// 通知 videobuf2：buffer 已填充完成，可以 DQBUF 了
vb2_buffer_done(&buf->buf.vb2_buf, buf->error ? VB2_BUF_STATE_ERROR :
                                               VB2_BUF_STATE_DONE);
```

### 5.2 `vb2_buffer_done()` — 移入 done_list 并唤醒

```c
// linux/drivers/media/common/videobuf2/videobuf2-core.c
// 函数名：vb2_buffer_done()
// 作用：标记 buffer 完成，移入 done_list，唤醒等待在 done_wq 上的 DQBUF 进程。
// 参数：vb=struct vb2_buffer *, state=buffer 状态（VB2_BUF_STATE_DONE 或 ERROR）
void vb2_buffer_done(struct vb2_buffer *vb, enum vb2_buffer_state state)
{
    ...
    spin_lock_irqsave(&q->done_lock, flags);

    if (state == VB2_BUF_STATE_QUEUED) {    // 重新入队（如 STREAMOFF 再 STREAMON）
        vb->state = VB2_BUF_STATE_QUEUED;
    } else {
        // 关键：将 buffer 追加到 done_list 尾部
        list_add_tail(&vb->done_entry, &q->done_list);
        vb->state = state;
    }
    // 驱动持有计数减一（之前在入队时 increment）
    atomic_dec(&q->owned_by_drv_count);
    spin_unlock_irqrestore(&q->done_lock, flags);

    switch (state) {
    case VB2_BUF_STATE_QUEUED:
        return;                              // 不唤醒，继续等待
    default:
        wake_up(&q->done_wq);                // 关键唤醒：通知 DQBUF 等待者
        break;
    }
}
```

这就把"USB 摄像头已经送来一帧数据"映射为"等待在 `done_wq` 的进程可以继续执行 `VIDIOC_DQBUF`"。

---

## 6. USB bulk IN 与 DMA scatter-gather

UVC 摄像头使用 USB bulk IN endpoint 传输 MJPEG 压缩流。数据通过 xHCI 的 scatter-gather DMA 直接写入用户态 mmap buffer：

### 6.1 DMA 零拷贝路径

```text
用户态 mmap(fd, buffer_offset) → 用户态虚拟地址
        ↓
vb2_queue 持有 buffer 物理页 → DMA-able
        ↓
xHCI DMA scatter-gather 直接写入物理页
        ↓
URB 完成中断 → uvc_video_complete() → vb2_buffer_done()
        ↓
wake_up(done_wq) → DQBUF 返回元数据（index/bytesused/timestamp）
```

整个数据路径是零拷贝的：摄像头数据直接写入用户态 buffer，URB 完成只是通知 videobuf2"这帧数据已经好了"。

---

## 7. `struct v4l2_buffer` 关键字段

```c
// linux/include/uapi/linux/videodev2.h
// struct v4l2_buffer — V4L2 buffer 描述结构（用户态和内核态通过此结构交换元数据）
//   index      : buffer 索引，OpenCV 用它索引自己的 buffers[]
//   type       : buffer 类型，如 V4L2_BUF_TYPE_VIDEO_CAPTURE
//   bytesused  : 实际 payload 长度（MJPEG 压缩数据长度）
//   flags      : 状态标志，V4L2_BUF_FLAG_DONE / V4L2_BUF_FLAG_QUEUED 等
//   timestamp  : 驱动记录的帧时间戳
//   sequence   : 帧序号
//   memory     : 内存模型，OpenCV 用 V4L2_MEMORY_MMAP
//   m.offset   : MMAP 模式下的 mmap 偏移
//   length     : buffer 总长度（不等于有效 payload 长度）
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
    ...
};
```

---

## 8. 完整调用栈

### 8.1 主路径（阻塞取帧）

```text
// 用户态
glibc ioctl(fd, VIDIOC_DQBUF, &v4l2_buffer)  → SYSCALL → sys_ioctl()

// 内核态
sys_ioctl() [linux/fs/ioctl.c:893]
├── fdget(fd)                            → int fd 转 struct file *
├── security_file_ioctl()                → LSM 安全检查
└── vfs_ioctl(file, cmd, arg)           → 因为 V4L2 不走 do_vfs_ioctl
    └── video_ioctl2(file, cmd, arg)    → V4L2 核心 unlocked_ioctl
        └── video_usercopy(file, cmd, arg, __video_do_ioctl)
            ├── video_get_user()                    → copy_from_user(v4l2_buffer)
            │   （用户态 struct v4l2_buffer → 内核临时缓冲区）
            └── __video_do_ioctl(file, cmd, parg)
                └── v4l_dqbuf(ops, file, fh, p)
                    └── uvc_ioctl_dqbuf(file, fh, buf)
                        └── uvc_dequeue_buffer(&stream->queue, buf, nonblocking)
                            └── vb2_dqbuf(&queue->queue, buf, nonblocking)
                                └── vb2_core_dqbuf(q, NULL, b, nonblocking)
                                    └── __vb2_get_done_vb(q, &vb, pb, nonblocking)
                                        └── __vb2_wait_for_done_vb(q, nonblocking)
                                            └── wait_event_interruptible(done_wq, ...)
                                                │ （done_list 为空时，进程在此睡眠）
```

### 8.2 唤醒路径（URB 完成到 DQBUF 返回）

```text
// USB/中断上下文
xhci_irq()                                → USB 主机控制器中断
└── usb_hcd_giveback_urb()               → URB 完成回调
    └── uvc_video_complete(urb)           → UVC URB 完成处理
        └── uvc_video_decode(urb)        → 解析 UVC 数据
            └── uvc_queue_buffer()        → 把数据写入 vb2 buffer
                └── vb2_buffer_done(vb, VB2_BUF_STATE_DONE)
                    ├── list_add_tail(&vb->done_entry, &q->done_list)
                    │   （buffer 从 drivers 列表移到完成队列）
                    └── wake_up(&q->done_wq)
                        （唤醒等待在 done_wq 上的 DQBUF 进程）

// DQBUF 进程被唤醒后继续执行
__vb2_wait_for_done_vb()                  → wait_event_interruptible 返回（条件满足）
__vb2_get_done_vb()                       → 从 done_list 取 buffer
vb2_core_dqbuf()                          → 填充 index/bytesused/timestamp
video_usercopy()                          → video_put_user 拷贝回用户态
返回用户态 ioctl()                         → 返回 0
```

---

## 9. VIDIOC_DQBUF 和 VIDIOC_QBUF 的区别

| 操作  | ioctl          | 含义                                                             |
| ----- | -------------- | ---------------------------------------------------------------- |
| QBUF  | `VIDIOC_QBUF`  | 把 buffer 入队到驱动队列，等待填充（初始化时每 buffer 调用一次） |
| DQBUF | `VIDIOC_DQBUF` | 从完成队列取出已填充的 buffer（每读一帧调用一次）                |

在 OpenCV 典型的 `V4L2_MEMORY_MMAP` 模式下，初始化时会做一次 `REQBUFS`（分配 buffer） + 多次 `QBUF`（入队到驱动），然后循环调用 `DQBUF`（取完成帧） + `QBUF`（重新入队供驱动下次填充）。

---

## 10. 性能结论

1. **`VIDIOC_DQBUF` 是控制面 + 元数据出队**，不是整帧像素拷贝。若 `done_list` 已经有完成帧，系统调用可以很快返回。

2. **阻塞点在 `wait_event_interruptible(q->done_wq, ...)`**。当 `done_list` 为空时，调用进程在 `done_wq` wait queue 上睡眠，等待 `vb2_buffer_done()` 唤醒。

3. **MJPEG 解码不在内核完成**，在 `retrieve()` 之后的 OpenCV/libjpeg 用户态路径完成，消耗 ARM CPU。

4. **DMA 零拷贝**：xHCI 把 USB bulk 数据直接 DMA 到用户态 mmap buffer，不经过内核额外拷贝。