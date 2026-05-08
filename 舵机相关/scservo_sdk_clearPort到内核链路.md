# scservo_sdk clearPort 到 Linux 内核链路

本文从 `scservo_sdk/port_handler.py` 的 `PortHandler.clearPort()` 开始，追踪到 Linux 内核完成“等待串口发送队列排空”的完整路径。

结论先说清楚：`clearPort()` 在当前 SDK 里不是清空接收缓冲区，而是调用 `pyserial` 的 `flush()`。在 Linux/POSIX 后端，`flush()` 等价于 `termios.tcdrain(fd)`，glibc 再把它变成 `ioctl(fd, TCSBRK, 1)`。内核收到 `TCSBRK` 且 `arg=1` 后，只等待已排队的输出字节发送完成，不发送 break 信号。

---

## 1. scservo_sdk 入口

`clearPort()` 在 `protocol_packet_handler.txPacket()` 中，每次真正 `writePort()` 之前调用：

```python
# scservo_sdk/protocol_packet_handler.py
port.clearPort()                         # 发送新协议帧前，先等待上一次输出排空
written_packet_length = port.writePort(txpacket)  # 然后把本次协议帧写入串口
```

SDK 起点源码：

```python
# scservo_sdk/port_handler.py
def clearPort(self):                      # self 是 PortHandler，内部持有 self.ser
    self.ser.flush()                      # 调用 pyserial Serial.flush()
```

这里的实际参数只有一个隐含参数：

| 层级 | 参数 | 含义 |
|------|------|------|
| `PortHandler.clearPort()` | `self.ser` | `serial.Serial` 实例，已经打开 `/dev/ttyACM0` |
| `serial.Serial.flush()` | `self.fd` | pyserial 打开的文件描述符，指向 `/dev/ttyACM0` |
| `tcdrain(fd)` | `fd` | 同一个 tty 文件描述符 |
| `ioctl(fd, TCSBRK, 1)` | `TCSBRK` | TTY ioctl 命令，等待发送完成或发送 break |
| `ioctl(fd, TCSBRK, 1)` | `arg=1` | 非零表示只 drain，不发送 break |

在 ARM64 的通用 syscall 表里，`ioctl` 的系统调用号是 `__NR_ioctl=29`。`TCSBRK` 的 ioctl 命令值来自 `linux/include/uapi/asm-generic/ioctls.h`：

```c
#define TCSBRK 0x5409                         // tty ioctl：等待输出完成或发送 break
```

---

## 2. 完整调用栈

### 2.1 内核外调用栈

```text
protocol_packet_handler.txPacket()
│  SDK 协议层：准备发送一帧 Feetech 指令包；在真正写串口前先调用 clearPort()。
└── PortHandler.clearPort()
    │  SDK 串口封装层：不传数据，只要求底层串口把“之前排队的输出”发送完。
    └── serial.Serial.flush()
        │  pyserial POSIX 后端：file-like flush 语义，在串口上实现为等待输出 drain。
        └── termios.tcdrain(fd)
            │  CPython termios 模块入口：释放 GIL 后调用 libc tcdrain(fd)。
            └── CPython termios_tcdrain_impl(fd)
                │  CPython C 实现：把 Python int fd 原样交给 libc，失败时转成 Python 异常。
                └── glibc tcdrain(fd)
                    │  libc termios 实现：把 tcdrain 转成 ioctl(fd, TCSBRK, 1)。
                    └── ioctl(fd, TCSBRK, 1)
                        │  用户态发起系统调用：fd=/dev/ttyACM0，cmd=TCSBRK，arg=1。
                        └── syscall boundary
                            │  从这里进入 Linux 内核：ARM64 上 x8=__NR_ioctl=29。
```

### 2.2 内核内调用栈

```text
Linux SYSCALL_DEFINE3(ioctl, fd, cmd, arg)
│  内核 syscall 入口：用 fd 查到 struct file，并做 LSM 安全检查。
└── do_vfs_ioctl(file, fd, TCSBRK, 1)
    │  VFS 通用 ioctl 分发：先处理通用命令，不认识则返回 -ENOIOCTLCMD。
    └── vfs_ioctl(file, TCSBRK, 1)
        │  VFS 设备分发：调用这个文件的 file_operations->unlocked_ioctl。
        └── file->f_op->unlocked_ioctl()
            │  /dev/ttyACM0 的 f_op 是 tty_fops，所以进入 TTY ioctl。
            └── tty_ioctl(file, TCSBRK, 1)
                │  TTY 命令处理：TCSBRK 且 arg=1 表示只等待输出完成，不发 break。
                ├── tty_wait_until_sent(tty, 0)
                │   │  等待发送队列排空：timeout=0 在这里表示无限等待。
                │   └── wait_event_interruptible_timeout(
                │       tty->write_wait,
                │       !tty_chars_in_buffer(tty),
                │       MAX_SCHEDULE_TIMEOUT)
                │       │  睡在 tty->write_wait 上，直到驱动报告输出缓冲为 0。
                └── tty_chars_in_buffer(tty)
                    │  TTY core 查询具体驱动：还有多少输出字节没有完成。
                    └── tty->ops->chars_in_buffer(tty)
                        │  tty_operations 分发到 CDC ACM 驱动。
                        └── acm_tty_chars_in_buffer(tty)
                            │  CDC ACM 驱动：根据正在使用的写 URB/write buffer 估算剩余输出字节。
```

对 `/dev/ttyACM0` 来说，`tty->ops` 来自 CDC ACM 驱动：

```text
linux/drivers/usb/class/cdc-acm.c
static const struct tty_operations acm_ops = {
    .chars_in_buffer = acm_tty_chars_in_buffer,
    .write           = acm_tty_write,
    .write_room      = acm_tty_write_room,
    ...
}
```

因此内核最终是在 CDC ACM 驱动里判断还有多少 USB bulk OUT 写 URB 没完成。

---

## 3. pyserial 与 CPython 层

### 3.1 pyserial `flush()`

```python
# venv/lib/python3.13/site-packages/serial/serialposix.py
def flush(self):                          # file-like flush，语义是等待输出完成
    if not self.is_open:                  # 如果串口未打开
        raise PortNotOpenError()          # 抛出 pyserial 的端口未打开异常
    termios.tcdrain(self.fd)              # 调 CPython termios.tcdrain(fd)
```

`self.fd` 是 pyserial 在打开串口时得到的 Unix 文件描述符。它指向字符设备 `/dev/ttyACM0`，对应内核中的 `struct file`。

### 3.2 CPython `termios.tcdrain()`

```c
// cpython/Modules/termios.c
static PyObject *
termios_tcdrain_impl(PyObject *module, int fd)  // fd 是 Python 传入的 int 文件描述符
{
    termiosmodulestate *state = PyModule_GetState(module);  // 获取 termios 模块状态
    int r;                                                  // 保存 libc tcdrain 返回值

    Py_BEGIN_ALLOW_THREADS                                  // 释放 GIL，避免阻塞其他 Python 线程
    r = tcdrain(fd);                                        // 调用 libc 的 tcdrain(fd)
    Py_END_ALLOW_THREADS                                    // reacquire GIL

    if (r == -1) {                                          // libc 返回 -1 表示失败
        return PyErr_SetFromErrno(state->TermiosError);     // 把 errno 转成 Python 异常
    }
    Py_RETURN_NONE;                                         // 成功时 Python 返回 None
}
```

CPython 不解释 tty 语义，只负责释放 GIL，并把 `fd` 原样交给 libc。

### 3.3 glibc `tcdrain()`

```c
// glibc-2.42/sysdeps/unix/sysv/linux/tcdrain.c
int
__libc_tcdrain (int fd)                         // fd 是 /dev/ttyACM0 的文件描述符
{
  return SYSCALL_CANCEL (ioctl, fd, TCSBRK, 1);  // 发 ioctl(fd, TCSBRK, 1)
}
weak_alias (__libc_tcdrain, tcdrain)             // tcdrain 是 __libc_tcdrain 的弱别名
```

`TCSBRK` 的语义在内核 `tty_ioctl()` 中决定：

- `arg == 0`：等待输出完成，然后发送 break。
- `arg != 0`：只等待输出完成，不发送 break。

`clearPort()` 走的是 `arg=1`，所以它只是 drain。

---

## 4. Linux ioctl 入口

### 4.1 系统调用入口

```c
// linux/fs/ioctl.c
SYSCALL_DEFINE3(ioctl, unsigned int, fd, unsigned int, cmd, unsigned long, arg)
{
    struct fd f = fdget(fd);                      // 用 fd 查当前进程的 file 表，得到 struct file
    int error;                                    // 保存返回码

    if (!fd_file(f))                              // fd 无效或已经关闭
        return -EBADF;                            // 返回 bad file descriptor

    error = security_file_ioctl(fd_file(f), cmd, arg);  // LSM 安全检查
    if (error)                                    // 如果安全模块拒绝
        goto out;                                 // 直接返回错误

    error = do_vfs_ioctl(fd_file(f), fd, cmd, arg);  // VFS 通用 ioctl 处理
    if (error == -ENOIOCTLCMD)                    // 通用层不认识该命令
        error = vfs_ioctl(fd_file(f), cmd, arg);  // 调用具体 file_operations 的 ioctl

out:
    fdput(f);                                     // 释放 fd 引用
    return error;                                 // 返回 ioctl 结果
}
```

这里的实际参数来自 glibc：

```text
fd  = pyserial 打开的 /dev/ttyACM0 文件描述符
cmd = TCSBRK
arg = 1
__NR_ioctl = 29
```

### 4.2 `/dev/ttyACM0` 的 file_operations

```c
// linux/drivers/tty/tty_io.c
static const struct file_operations tty_fops = {
    .read_iter       = tty_read,        // read() 走 tty_read
    .write_iter      = tty_write,       // write() 走 tty_write
    .poll            = tty_poll,        // select/poll 走 tty_poll
    .unlocked_ioctl  = tty_ioctl,       // ioctl() 走 tty_ioctl
};
```

`struct file_operations` 是 VFS 给每个打开文件挂的操作表。`/dev/ttyACM0` 是字符设备文件，所以 `vfs_ioctl()` 最终通过 `file->f_op->unlocked_ioctl` 调到 `tty_ioctl()`。

---

## 5. TTY 层如何处理 TCSBRK

### 5.1 `tty_ioctl()` 的 `TCSBRK` 分支

```c
// linux/drivers/tty/tty_io.c
long tty_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct tty_struct *tty = file_tty(file);       // 从 struct file 取出对应 tty_struct
    int retval;                                    // 保存中间返回码

    switch (cmd) {                                 // 第一段 switch 做通用预处理
    case TCSBRK:                                   // tcdrain/tcsendbreak 共用该命令
        retval = tty_check_change(tty);            // 检查作业控制和 tty 状态
        if (retval)                                // 如果当前进程不能操作该 tty
            return retval;                         // 返回错误
        tty_wait_until_sent(tty, 0);               // 等待输出队列排空，0 表示无限等待
        if (signal_pending(current))               // 等待中被信号打断
            return -EINTR;                         // 返回 interrupted
        break;                                     // 继续进入第二段 switch
    }

    switch (cmd) {                                 // 第二段 switch 执行命令语义
    case TCSBRK:                                   // SVID 版本：非零 arg 不发送 break
        if (!arg)                                  // arg == 0 才发送 break
            return send_break(tty, 250);           // 发送 250 ms break
        return 0;                                  // arg == 1，tcdrain 完成，直接成功返回
    }
}
```

`clearPort()` 的 `arg=1`，所以 `send_break()` 不会执行。真正做事的是前面的 `tty_wait_until_sent()`。

### 5.2 `tty_wait_until_sent()`

```c
// linux/drivers/tty/tty_ioctl.c
void tty_wait_until_sent(struct tty_struct *tty, long timeout)
{
    if (!timeout)                                  // timeout 为 0
        timeout = MAX_SCHEDULE_TIMEOUT;            // 转成无限等待

    timeout = wait_event_interruptible_timeout(    // 进入可被信号打断的等待
        tty->write_wait,                           // 等待队列：写侧完成事件会唤醒它
        !tty_chars_in_buffer(tty),                 // 条件：驱动输出缓冲区没有剩余字符
        timeout);                                  // 最大等待时间

    if (timeout <= 0)                              // 超时或被信号中断
        return;                                    // 直接返回，调用者检查 signal_pending

    if (timeout == MAX_SCHEDULE_TIMEOUT)           // 如果是无限等待
        timeout = 0;                               // 传给驱动 wait_until_sent 时仍用 0 表示无限

    if (tty->ops->wait_until_sent)                 // 驱动可选提供更底层等待函数
        tty->ops->wait_until_sent(tty, timeout);   // CDC ACM 没有这个钩子，通常不走
}
```

核心是 `!tty_chars_in_buffer(tty)`。它会问具体驱动还有多少字节排在输出队列里。

### 5.3 `tty_chars_in_buffer()`

```c
// linux/drivers/tty/tty_ioctl.c
unsigned int tty_chars_in_buffer(struct tty_struct *tty)
{
    if (tty->ops->chars_in_buffer)                 // 如果具体 tty 驱动提供查询函数
        return tty->ops->chars_in_buffer(tty);     // 调驱动查询剩余输出字节
    return 0;                                      // 没有驱动队列则认为已经排空
}
```

对 `/dev/ttyACM0`，`tty->ops->chars_in_buffer` 指向 `acm_tty_chars_in_buffer()`。

---

## 6. CDC ACM 驱动如何判断发送完成

### 6.1 `tty_operations` 绑定

```c
// linux/drivers/usb/class/cdc-acm.c
static const struct tty_operations acm_ops = {
    .write           = acm_tty_write,              // TTY 写数据时调用
    .write_room      = acm_tty_write_room,         // 查询还能写多少
    .flush_buffer    = acm_tty_flush_buffer,       // 刷新输出队列时调用
    .chars_in_buffer = acm_tty_chars_in_buffer,    // tcdrain 查询剩余输出字节时调用
};
```

`struct tty_operations` 是 TTY 核心和具体硬件驱动之间的接口表。CDC ACM 驱动用它把通用 TTY 操作映射到 USB ACM 设备。

### 6.2 `acm_tty_chars_in_buffer()`

```c
// linux/drivers/usb/class/cdc-acm.c
static unsigned int acm_tty_chars_in_buffer(struct tty_struct *tty)
{
    struct acm *acm = tty->driver_data;            // tty->driver_data 指向 CDC ACM 私有结构

    if (acm->disconnected)                         // 如果 USB 设备已经拔掉
        return 0;                                  // 剩余字节视为 0

    return (ACM_NW - acm_wb_is_avail(acm)) * acm->writesize;
                                                    // 估算正在使用的写缓冲数量 × 每个缓冲大小
}
```

这里有两个重要结构：

| 结构 | 含义 |
|------|------|
| `struct acm` | CDC ACM 设备实例，保存 USB 设备、TTY port、读写 URB、锁、状态位 |
| `struct acm_wb` | write buffer，一次 USB bulk OUT 写请求使用一个 `acm_wb` |

`acm_tty_chars_in_buffer()` 的返回值是估算值，注释里也说明它会 overcount。对 `tcdrain()` 来说，只要还有写缓冲处于使用状态，就继续等待。

### 6.3 USB 写完成如何唤醒等待者

```c
// linux/drivers/usb/class/cdc-acm.c
static void acm_write_bulk(struct urb *urb)
{
    struct acm_wb *wb = urb->context;              // urb->context 指向本次写使用的 acm_wb
    struct acm *acm = wb->instance;                // wb->instance 指向所属 acm 设备
    unsigned long flags;                           // 保存中断标志
    int status = urb->status;                      // USB 传输完成状态

    spin_lock_irqsave(&acm->write_lock, flags);    // 保护写缓冲状态
    acm_write_done(acm, wb);                       // 标记该 write buffer 空闲，减少 transmitting
    spin_unlock_irqrestore(&acm->write_lock, flags); // 释放写锁

    set_bit(EVENT_TTY_WAKEUP, &acm->flags);        // 设置需要唤醒 TTY 写等待者的事件
    schedule_delayed_work(&acm->dwork, 0);         // 调度 workqueue，避免在硬中断上下文做复杂工作
}

static void acm_softint(struct work_struct *work)
{
    struct acm *acm = container_of(work, struct acm, dwork.work); // 由 work 找回 acm

    if (test_and_clear_bit(EVENT_TTY_WAKEUP, &acm->flags))       // 如果有写完成唤醒事件
        tty_port_tty_wakeup(&acm->port);                         // 唤醒 TTY 写等待队列
}
```

这就是 `clearPort()` 能等待“上一帧已经由 USB 写完成”的原因：

1. 上一次 `writePort()` 提交了 USB bulk OUT URB。
2. xHCI/USB 控制器把数据送到 USB 设备后，URB complete。
3. `acm_write_bulk()` 标记写缓冲空闲。
4. `tty_port_tty_wakeup()` 唤醒 `tty->write_wait`。
5. `tty_wait_until_sent()` 重新检查 `tty_chars_in_buffer()`，如果为 0 就返回。

---

## 7. clearPort 究竟完成了什么

`clearPort()` 完成的任务不是“丢弃残留包”，而是“等待之前已经排入 TTY/USB 驱动的输出数据完成发送”。

对舵机链路来说：

- Sync Read 发送前调用 `clearPort()`，避免上一帧还在 USB/TTY 写队列中。
- Sync Write 发送前调用 `clearPort()`，同样等待上一条输出完成。
- 由于 `arg=1`，内核不会向串口线发送 break。
- 如果当前没有未完成的 CDC ACM 写 URB，`tty_chars_in_buffer()` 返回 0，`clearPort()` 很快返回。

这条链路对应的系统调用参数最终是：

```text
ioctl(fd=/dev/ttyACM0, cmd=TCSBRK, arg=1)
```

内核做的核心动作是：

```text
等待 tty->write_wait，直到 acm_tty_chars_in_buffer(tty) == 0
```
