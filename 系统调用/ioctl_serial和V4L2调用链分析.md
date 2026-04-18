# ioctl 系统调用分析：Serial Flush 与 V4L2 retrieveFrame

> 本文档分析 Linux 内核中 ioctl 系统调用的两条核心路径：
> 1. **Serial Flush**：`TCFLSH` ioctl 如何刷新串口缓冲区
> 2. **V4L2 retrieveFrame**：`VIDIOC_DQBUF/VIDIOC_QBUF` ioctl 如何完成帧的出队/入队
>
> 推理硬件：树莓派 5 (ARM CPU)

---

## 1. Serial Flush (TCFLSH) 调用链

### 1.1 完整调用路径

```
用户空间
  └─ ioctl(fd, TCFLSH, TCIOFLUSH)
      │
      ▼
[内核入口]
  └─ sys_ioctl()           (fs/ioctl.c)
      └─ vfs_ioctl()       (fs/ioctl.c)
          └─ do_vfs_ioctl() (fs/ioctl.c)
              │
              ▼
[tty 层 - 通用 ioctl 分发]
  └─ tty_ioctl()           (drivers/tty/tty_io.c:2679)
      │ 处理 TCFLSH case
      ▼
  └─ tty_perform_flush()   (drivers/tty/tty_ioctl.c:925)
      │
      ▼
  └─ __tty_perform_flush() (drivers/tty/tty_ioctl.c:899)
      │ 根据 arg 分别处理
      ▼
  ┌─ TCIFLUSH  ──→ ld->ops->flush_buffer(tty)  ──→ n_tty_flush_buffer()
  ├─ TCIOFLUSH ──→ 上述输入刷新 + tty_driver_flush_buffer()
  └─ TCOFLUSH  ──→ tty_driver_flush_buffer()
                      │
                      ▼
                  uart_flush_buffer()    (drivers/tty/serial/serial_core.c:666)
                      │
                      ▼
                  port->ops->flush_buffer(port)  (硬件特定实现)
```

### 1.2 tty_ioctl 入口处理

**源码路径**：`linux/drivers/tty/tty_io.c:2679`

```c
long tty_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct tty_struct *tty = file_tty(file);
    // ...
    switch (cmd) {
    // ...
    case TCFLSH:
        switch (arg) {
        case TCIFLUSH:
        case TCIOFLUSH:
            // flush tty buffer and allow ldisc to process ioctl
            tty_buffer_flush(tty, NULL);
            break;
        }
        break;
    // ...
    }
    // ...
}
```

### 1.3 tty_perform_flush 与 __tty_perform_flush

**源码路径**：`linux/drivers/tty/tty_ioctl.c:899`

```c
/* Caller guarantees ldisc reference is held */
static int __tty_perform_flush(struct tty_struct *tty, unsigned long arg)
{
    struct tty_ldisc *ld = tty->ldisc;

    switch (arg) {
    case TCIFLUSH:
        // 刷新输入缓冲区：调用 line discipline 的 flush_buffer
        if (ld && ld->ops->flush_buffer) {
            ld->ops->flush_buffer(tty);
            tty_unthrottle(tty);
        }
        break;
    case TCIOFLUSH:
        // 刷新输入缓冲区
        if (ld && ld->ops->flush_buffer) {
            ld->ops->flush_buffer(tty);
            tty_unthrottle(tty);
        }
        // 刷新输出缓冲区
        fallthrough;
    case TCOFLUSH:
        tty_driver_flush_buffer(tty);
        break;
    default:
        return -EINVAL;
    }
    return 0;
}
```

### 1.4 n_tty_flush_buffer (N_TTY line discipline)

**源码路径**：`linux/drivers/tty/n_tty.c:356`

```c
static void n_tty_flush_buffer(struct tty_struct *tty)
{
    down_write(&tty->termios_rwsem);
    reset_buffer_flags(tty->disc_data);  // 清空 read buffer
    n_tty_kick_worker(tty);               // 唤醒等待数据的进程

    if (tty->link)
        n_tty_packet_mode_flush(tty);
    up_write(&tty->termios_rwsem);
}
```

### 1.5 tty_driver_flush_buffer (驱动层刷新)

**源码路径**：`linux/drivers/tty/tty_io.c:82`

```c
void tty_driver_flush_buffer(struct tty_struct *tty)
{
    if (tty->ops->flush_buffer)
        tty->ops->flush_buffer(tty);
}
```

### 1.6 uart_flush_buffer (Serial Core 层)

**源码路径**：`linux/drivers/tty/serial/serial_core.c:666`

```c
static void uart_flush_buffer(struct tty_struct *tty)
{
    struct uart_state *state = tty->driver_data;
    struct uart_port *port;
    unsigned long flags;

    if (WARN_ON(!state))
        return;

    pr_debug("uart_flush_buffer(%d) called\n", tty->index);

    port = uart_port_lock(state, flags);
    if (!port)
        return;

    // 1. 清空 kernel FIFO (输出缓冲区)
    kfifo_reset(&state->port.xmit_fifo);

    // 2. 调用硬件驱动层刷新（如果有）
    if (port->ops->flush_buffer)
        port->ops->flush_buffer(port);

    uart_port_unlock(port, flags);

    // 3. 唤醒等待写入的进程
    tty_port_tty_wakeup(&state->port);
}
```

### 1.7 各层 flush 含义总结

| ioctl 参数 | 刷新位置 | 调用函数 |
|-----------|---------|---------|
| `TCIFLUSH` | 输入缓冲区 (tty buffer + ldisc) | `tty_buffer_flush()` + `ld->ops->flush_buffer()` |
| `TCOFLUSH` | 输出缓冲区 (driver xmit FIFO) | `tty_driver_flush_buffer()` → `uart_flush_buffer()` |
| `TCIOFLUSH` | 输入 + 输出缓冲区 | 上述两者都调用 |

---

## 2. V4L2 retrieveFrame (DQBUF/QBUF) 调用链

> OpenCV 的 `VideoCapture::grabFrame()` 和 `retrieveFrame()` 底层调用 V4L2 ioctl。
> 对应关系：
> - `grabFrame()` → `VIDIOC_DQBUF` (出队，从内核 DMA 缓冲区取数据)
> - `retrieveFrame()` → `VIDIOC_QBUF` (入队，归还缓冲区)

### 2.1 完整调用路径

```
用户空间 (OpenCV)
  └─ ioctl(fd, VIDIOC_DQBUF, &v4l2_buffer)   // grabFrame
  └─ ioctl(fd, VIDIOC_QBUF, &v4l2_buffer)    // retrieveFrame
      │
      ▼
[内核 V4L2 层]
  └─ video_ioctl2()          (drivers/media/v4l2-core/v4l2-ioctl.c:3522)
      └─ video_usercopy()    (drivers/media/v4l2-core/v4l2-ioctl.c:3412)
          └─ __video_do_ioctl()  (drivers/media/v4l2-core/v4l2-ioctl.c:3055)
              │ 根据 cmd 分发
              ▼
          ┌─ v4l_dqbuf()    → ops->vidioc_dqbuf()
          └─ v4l_qbuf()     → ops->vidioc_qbuf()
              │
              ▼
[videobuf2 层]
  ├─ vb2_dqbuf()             (drivers/media/common/videobuf2/videobuf2-v4l2.c:838)
  │   └─ vb2_core_dqbuf()    (drivers/media/common/videobuf2/videobuf2-core.c)
  │       └─ __vb2_get_done_vb() → 从 done_list 取缓冲区
  │
  └─ vb2_qbuf()              (drivers/media/common/videobuf2/videobuf2-v4l2.c:810)
      └─ vb2_core_qbuf()     (drivers/media/common/videobuf2/videobuf2-core.c)
          └─ 将缓冲区添加到 queued_list
```

### 2.2 video_ioctl2 入口

**源码路径**：`linux/drivers/media/v4l2-core/v4l2-ioctl.c:3522`

```c
long video_ioctl2(struct file *file, unsigned int cmd, unsigned long arg)
{
    return video_usercopy(file, cmd, arg, __video_do_ioctl);
}
```

### 2.3 video_usercopy (参数复制与错误处理)

**源码路径**：`linux/drivers/media/v4l2-core/v4l2-ioctl.c:3412`

```c
long video_usercopy(struct file *file, unsigned int orig_cmd, unsigned long arg,
                   v4l2_kioctl func)
{
    char    sbuf[128];
    void    *mbuf = NULL, *array_buf = NULL;
    void    *parg = (void *)arg;
    long    err = -EINVAL;
    bool    has_array_args;
    size_t  array_size = 0;
    void __user *user_ptr = NULL;
    void    **kernel_ptr = NULL;
    unsigned int cmd = video_translate_cmd(orig_cmd);

    // 1. 复制用户参数到内核缓冲区
    if (_IOC_DIR(cmd) != _IOC_NONE) {
        if (ioc_size <= sizeof(sbuf)) {
            parg = sbuf;
        } else {
            mbuf = kmalloc(ioc_size, GFP_KERNEL);
            parg = mbuf;
        }
        err = video_get_user((void __user *)arg, parg, cmd, orig_cmd, ...);
    }

    // 2. 处理数组参数 (如多平面缓冲区)
    err = check_array_args(cmd, parg, &array_size, &user_ptr, &kernel_ptr);
    has_array_args = err;
    if (has_array_args) {
        array_buf = kvmalloc(array_size, GFP_KERNEL);
        copy_from_user(array_buf, user_ptr, array_size);
        *kernel_ptr = array_buf;
    }

    // 3. 调用 ioctl 处理函数 (核心)
    err = func(file, cmd, parg);

    // 4. 复制结果回用户空间
    if (has_array_args) {
        copy_to_user(user_ptr, array_buf, array_size);
    }
    video_put_user((void __user *)arg, parg, cmd, orig_cmd);
out:
    kvfree(array_buf);
    kfree(mbuf);
    return err;
}
```

### 2.4 __video_do_ioctl (V4L2 ioctl 分发)

**源码路径**：`linux/drivers/media/v4l2-core/v4l2-ioctl.c:3055`

```c
static long __video_do_ioctl(struct file *file, unsigned int cmd, void *arg)
{
    struct video_device *vfd = video_devdata(file);
    const struct v4l2_ioctl_ops *ops = vfd->ioctl_ops;
    struct v4l2_fh *vfh = NULL;
    long ret = -ENOTTY;

    if (test_bit(V4L2_FL_USES_V4L2_FH, &vfd->flags))
        vfh = file->private_data;

    // 根据 cmd 查找对应的处理函数
    if (v4l2_is_known_ioctl(cmd)) {
        info = &v4l2_ioctls[_IOC_NR(cmd)];  // 从表中查找
        // ... 权限检查 ...
        ret = info->func(ops, file, fh, arg);  // 调用处理函数
    }
    // ...
}
```

### 2.5 v4l_qbuf / v4l_dqbuf (V4L2 ioctl 包装)

**源码路径**：`linux/drivers/media/v4l2-core/v4l2-ioctl.c:2176`

```c
static int v4l_qbuf(const struct v4l2_ioctl_ops *ops,
                    struct file *file, void *fh, void *arg)
{
    struct v4l2_buffer *p = arg;
    int ret = check_fmt(file, p->type);
    return ret ? ret : ops->vidioc_qbuf(file, fh, p);
}

static int v4l_dqbuf(const struct v4l2_ioctl_ops *ops,
                    struct file *file, void *fh, void *arg)
{
    struct v4l2_buffer *p = arg;
    int ret = check_fmt(file, p->type);
    return ret ? ret : ops->vidioc_dqbuf(file, fh, p);
}
```

### 2.6 vb2_qbuf (videobuf2 入队)

**源码路径**：`linux/drivers/media/common/videobuf2/videobuf2-v4l2.c:810`

```c
int vb2_qbuf(struct vb2_queue *q, struct media_device *mdev,
             struct v4l2_buffer *b)
{
    struct media_request *req = NULL;
    struct vb2_buffer *vb;
    int ret;

    if (vb2_fileio_is_active(q)) {
        dprintk(q, 1, "file io in progress\n");
        return -EBUSY;
    }

    // 1. 根据 index 获取 vb2_buffer
    vb = vb2_get_buffer(q, b->index);

    // 2. 预处理 (处理 request 对象等)
    ret = vb2_queue_or_prepare_buf(q, mdev, vb, b, false, &req);

    // 3. 核心入队操作
    ret = vb2_core_qbuf(q, vb, b, req);

    if (req)
        media_request_put(req);
    return ret;
}
```

**vb2_core_qbuf 核心逻辑** (`videobuf2-core.c`):

```c
int vb2_core_qbuf(struct vb2_queue *q, struct vb2_buffer *vb,
                  struct v4l2_buffer *b, struct media_request *req)
{
    // ...
    // 1. 将 buffer 添加到 queued_list
    orig_state = vb->state;
    list_add_tail(&vb->queued_entry, &q->queued_list);
    q->queued_count++;
    q->waiting_for_buffers = false;
    vb->state = VB2_BUF_STATE_QUEUED;

    // 2. 如果正在 streaming，将 buffer 交给驱动
    if (q->start_streaming_called)
        __enqueue_in_driver(vb);

    // 3. 填充用户空间 buffer 信息
    if (pb)
        call_void_bufop(q, fill_user_buffer, vb, pb);

    // 4. 如果达到最小 queued buffers，启动 streaming
    if (q->streaming && !q->start_streaming_called &&
        q->queued_count >= q->min_queued_buffers) {
        ret = vb2_start_streaming(q);
        // ...
    }
    return 0;
}
```

### 2.7 vb2_dqbuf (videobuf2 出队)

**源码路径**：`linux/drivers/media/common/videobuf2/videobuf2-v4l2.c:838`

```c
int vb2_dqbuf(struct vb2_queue *q, struct v4l2_buffer *b, bool nonblocking)
{
    int ret;

    if (vb2_fileio_is_active(q))
        return -EBUSY;

    if (b->type != q->type)
        return -EINVAL;

    // 调用核心出队，nonblocking 决定是否阻塞等待
    ret = vb2_core_dqbuf(q, NULL, b, nonblocking);

    if (!q->is_output && b->flags & V4L2_BUF_FLAG_DONE && b->flags & V4L2_BUF_FLAG_LAST)
        q->last_buffer_dequeued = true;

    b->flags &= ~V4L2_BUF_FLAG_DONE;
    return ret;
}
```

**vb2_core_dqbuf 核心逻辑** (`videobuf2-core.c`):

```c
int vb2_core_dqbuf(struct vb2_queue *q, struct vb2_buffer **vb,
                   struct v4l2_buffer *b, bool nonblocking)
{
    // ...
    // 1. 从 done_list 取一个缓冲区 (可能会阻塞等待)
    ret = __vb2_get_done_vb(q, &buffer, pb, nonblocking);

    // 2. 如果 buffer 还在驱动手中，等待驱动完成
    if (buffer->state == VB2_BUF_STATE_DEQUEUED) {
        // 调用驱动的 wait_prepare 释放锁
        call_void_qop(q, wait_prepare, q);
        // 阻塞等待 V4L2_BUF_FLAG_DONE 或超时/错误
        ret = wait_event_interruptible(q->done_wq, ...);
        // 重新获取锁
        call_void_qop(q, wait_finish, q);
    }

    // 3. 将 buffer 标记为 DEQUEUED，填充用户空间信息
    __vb2_dqbuf(buffer);
    call_void_bufop(q, fill_user_buffer, buffer, pb);

    return 0;
}
```

**__vb2_dqbuf** (`videobuf2-core.c:2116`):

```c
static void __vb2_dqbuf(struct vb2_buffer *vb)
{
    struct vb2_queue *q = vb->vb2_queue;

    dprintk(q, 2, "dequeuing buffer %d\n", vb->index);

    // 1. 从 done_list 移除
    list_del(&vb->done_entry);
    q->done_count--;

    // 2. 标记为 DEQUEUED 状态
    vb->state = VB2_BUF_STATE_DEQUEUED;

    trace_vb2_dqbuf(q, vb);
}
```

---

## 3. OpenCV retrieveFrame 与 V4L2 的对应关系

根据 `robotdoc/Analysis/OpenCV视频流数据流分析.md`：

| OpenCV 函数 | V4L2 ioctl | 内核操作 |
|------------|------------|---------|
| `grabFrame()` | `VIDIOC_DQBUF` | 从 DMA 缓冲区队列出队，取回已填充数据的缓冲区 |
| `retrieveFrame()` | `VIDIOC_QBUF` | 将缓冲区重新入队，供下次 DQBUF 使用 |

```cpp
// cap_v4l.cpp:943 - read_frame_v4l2() 出队
while (!tryIoctl(VIDIOC_DQBUF, &buf)) {
    // EAGAIN/EBUSY → select 等待 → 重试
}
buffers[buf.index].buffer = buf;  // 保存缓冲区信息

// cap_v4l.cpp:237 - retrieveFrame() 入队
if (!tryIoctl(VIDIOC_QBUF, &buffers[bufferIndex].buffer)) {
    // 归还缓冲区
}
```

---

## 4. 关键数据结构

### 4.1 v4l2_buffer

```c
struct v4l2_buffer {
    __u32           index;          // 缓冲区索引
    __u32           type;           // V4L2_BUF_TYPE_VIDEO_CAPTURE 等
    __u32           bytesused;      // 已使用的字节数
    __u32           flags;          // V4L2_BUF_FLAG_QUEUED 等
    __u32           field;          // 视频字段
    struct timeval  timestamp;      // 时间戳
    // ... 多平面支持
    union {
        __u32       length;         // 单平面数据长度
        __u32       m.planes;       // 多平面数组
    } m;
};
```

### 4.2 vb2_buffer 状态机

```
VB2_BUF_STATE_DEQUEUED  ←─── vb2_core_dqbuf() ───┐
    │                                            │
    │  (用户空间可见)                             │ __vb2_get_done_vb()
    │                                            │ (阻塞等待)
    │                                            ▼
    │                                    VB2_BUF_STATE_DONE
    │                                            │
    │  vb2_core_qbuf() ───────────────────────────┘
    │        │
    │        ▼
    │  VB2_BUF_STATE_QUEUED ──→ (streaming 时) ──→ VB2_BUF_STATE_DEQUEUED
    │                                              (返回给用户空间)
```

---

## 5. ARM 树莓派 5 上的性能关键点

| 操作 | 系统调用 | 耗时占比 |
|------|---------|---------|
| `VIDIOC_DQBUF` + `select()` | 是 | ~33ms (等待摄像头) |
| `VIDIOC_QBUF` | 是 | <1ms |
| 内核 `vb2_core_dqbuf()` | 否 | <1ms |
| 内核 `__vb2_get_done_vb()` 阻塞唤醒 | 否 | context switch 开销 |

---

## 6. 源码文件索引

| 文件 | 路径 | 说明 |
|------|------|------|
| tty_ioctl | `linux/drivers/tty/tty_io.c:2679` | TTY ioctl 入口 |
| tty_perform_flush | `linux/drivers/tty/tty_ioctl.c:925` | TCFLSH 顶层处理 |
| __tty_perform_flush | `linux/drivers/tty/tty_ioctl.c:899` | TCFLSH 实际刷新逻辑 |
| n_tty_flush_buffer | `linux/drivers/tty/n_tty.c:356` | N_TTY line discipline 刷新 |
| uart_flush_buffer | `linux/drivers/tty/serial/serial_core.c:666` | Serial Core 刷新 |
| video_ioctl2 | `linux/drivers/media/v4l2-core/v4l2-ioctl.c:3522` | V4L2 ioctl 入口 |
| video_usercopy | `linux/drivers/media/v4l2-core/v4l2-ioctl.c:3412` | V4L2 用户参数复制 |
| __video_do_ioctl | `linux/drivers/media/v4l2-core/v4l2-ioctl.c:3055` | V4L2 ioctl 分发 |
| v4l_qbuf/v4l_dqbuf | `linux/drivers/media/v4l2-core/v4l2-ioctl.c:2176` | V4L2 QBUF/DQBUF 包装 |
| vb2_qbuf | `linux/drivers/media/common/videobuf2/videobuf2-v4l2.c:810` | videobuf2 入队 |
| vb2_dqbuf | `linux/drivers/media/common/videobuf2/videobuf2-v4l2.c:838` | videobuf2 出队 |
| vb2_core_qbuf | `linux/drivers/media/common/videobuf2/videobuf2-core.c` | videobuf2 核心入队 |
| vb2_core_dqbuf | `linux/drivers/media/common/videobuf2/videobuf2-core.c` | videobuf2 核心出队 |

---

*文档生成时间：2026-04-01*