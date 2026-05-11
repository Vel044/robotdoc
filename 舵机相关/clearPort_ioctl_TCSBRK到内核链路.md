# clearPort：ioctl(TCSBRK) 到 Linux 内核链路

本文从 `scservo_sdk/port_handler.py` 的 `PortHandler.clearPort()` 开始，追踪到 Linux 内核完成“等待串口发送队列排空”的完整路径。

结论先说清楚：`clearPort()` 在当前 SDK 里不是清空接收缓冲区，也不是丢弃发送缓冲区，而是调用 `pyserial` 的 `flush()`。在 Linux/POSIX 后端，`flush()` 等价于 `termios.tcdrain(fd)`，glibc 再把它变成 `ioctl(fd, TCSBRK, 1)`。内核收到 `TCSBRK` 且 `arg=1` 后，只等待已排队的输出字节发送完成，不发送 break 信号，也不会提交新的 USB URB。

因此 `clearPort()` 本身不走 `pselect6`；如果 trace 里同时出现 `pselect6`，通常来自前后的 pyserial `read()` 或阻塞 `write()`，不属于 `clearPort()` 的主链路。

和 `writePort()`、`readPort()` 相比，`clearPort()` 的核心不是数据搬运，而是同步等待：它查询 CDC ACM 驱动中是否还有未完成的 USB bulk OUT 写 URB；如果有，就睡在 `tty->write_wait` 上，直到写完成回调释放写缓冲并唤醒等待者。

---

## 图：ioctl(TCSBRK, 1) syscall 到内核边界的分层路径

```
ioctl(fd, TCSBRK, 1)                    # glibc tcdrain() 把 tcdrain 转成这个 ioctl
  │
  ▼
VFS: do_vfs_ioctl()                     # 为什么：所有设备的 ioctl 都经过 VFS 分发
  │  通用命令集不处理 TCSBRK，返回 -ENOIOCTLCMD
  ▼
TTY ioctl: tty_ioctl()                  # 收到 TCSBRK 后调 tty_wait_until_sent(tty, 0)
  │  为什么走 TTY：因为 fd 是 /dev/ttyACM0，对应 tty_fops
  ▼
tty_wait_until_sent()                  # 睡在 tty->write_wait，直到输出缓冲排空
  │  为什么需要等待：确保之前 write 提交的主机侧 USB 写 URB 已完成，再发下一帧
  ▼
tty_chars_in_buffer(tty)                # 查询驱动还有多少字节没发完
  │  为什么调用驱动：因为只有驱动知道 USB URB 队列状态
  ▼
acm_tty_chars_in_buffer()              # CDC ACM 驱动计算正在使用的写缓冲数 × writesize
  │
  ▼
等待写完成回调唤醒                  # acm_write_bulk() -> acm_write_done() -> tty_port_tty_wakeup()
========== kernel boundary ==========
```

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

这里的 `flush()` 容易被误解。它不是 `reset_input_buffer()`，不会清空接收缓冲；也不是 `reset_output_buffer()`，不会主动丢弃已经排队的输出数据。它对应 POSIX `tcdrain()`，语义是阻塞等待输出队列中的数据完成发送。

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

`clearPort()` 的链路很特殊：它不传输数据，只发一个 `ioctl` 命令来等待输出完成。内核外是 pyserial → CPython termios → glibc 的层层转换；内核内是 VFS → TTY ioctl → `tty_wait_until_sent()` → CDC ACM 查询路径。与此同时，上一帧 `writePort()` 提交的 USB bulk OUT URB 会在完成时触发 CDC ACM 写完成回调，回调释放写缓冲并唤醒等待在 `tty->write_wait` 上的 `clearPort()`。本节展示从 Python 到内核的完整调用栈。

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
└── fdget(fd)
    │  fd 表查找：得到 /dev/ttyACM0 对应的 struct file。
    └── security_file_ioctl(file, TCSBRK, 1)
        │  LSM 安全检查。
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
                                    │  CDC ACM 根据正在使用的 write buffer 估算剩余输出字节。

上一帧 USB bulk OUT URB 完成
└── acm_write_bulk(urb)
    │  xHCI/USB core 在写 URB 完成后调用 CDC ACM 完成回调。
    └── acm_write_done(acm, wb)
        │  标记 write buffer 空闲，减少 transmitting，释放 runtime PM 引用。
        └── tty_port_tty_wakeup(&acm->port)
            │  唤醒 tty->write_wait，clearPort() 醒来后重新检查 chars_in_buffer。
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

`clearPort()` 在 Python 侧只做了两件事：`pyserial.Serial.flush()` 把语义转给 `termios.tcdrain()`，CPython 的 termios 模块释放 GIL 后调用 glibc 的 `tcdrain()`。glibc 再把 `tcdrain` 包装成 `ioctl(fd, TCSBRK, 1)`。本节解释 Python 到 glibc 的转换过程。

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

`ioctl` 是 Linux 中设备驱动的"万能"系统调用。`TCSBRK` 不是 VFS 通用命令，所以 `do_vfs_ioctl()` 不认识它，会返回 `-ENOIOCTLCMD`，再由 `vfs_ioctl()` 通过 `tty_fops.unlocked_ioctl` 分发给 `tty_ioctl()`。本节解释 `ioctl` 从 VFS 到 TTY 层的分发路径。

这一层只传递控制命令，不搬运舵机协议帧。`arg=1` 是一个整数值，不是指向用户态数据缓冲的指针，因此这里不存在类似 `write()` 的 `copy_from_iter()` 或类似 `read()` 的 `copy_to_iter()`。

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

这里仍然只是在控制路径上分发命令；真正决定要不要等待的是 TTY 层对 `TCSBRK` 的解释。

---

## 5. TTY 层如何处理 TCSBRK

glibc 把 `tcdrain` 转成 `ioctl(fd, TCSBRK, 1)` 传入内核。TTY 层的 `tty_ioctl()` 收到 `TCSBRK` 后，需要区分是"发送 break"还是"等待输出完成"。由于 `arg=1`，内核不会发送 break，而是调用 `tty_wait_until_sent()` 进入等待，直到驱动报告输出队列已空。本节解释 TTY 层如何解析 `TCSBRK` 命令，并把等待请求转发给具体驱动。

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

因此，`TCSBRK` 在这里并不是“发送特殊串口信号”，而是借用了同一个 ioctl 命令实现 POSIX `tcdrain()` 的等待语义。

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

核心是 `!tty_chars_in_buffer(tty)`。它会问具体驱动还有多少字节排在输出队列里。对 CDC ACM 来说，`acm_ops` 没有提供 `.wait_until_sent` 钩子，因此主要判断就落在 `chars_in_buffer` 是否变为 0。

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

`tty_wait_until_sent()` 反复调用 `tty_chars_in_buffer()` 查询还有多少输出字节未完成。对 `/dev/ttyACM0`，这个查询最终落到 CDC ACM 驱动的 `acm_tty_chars_in_buffer()`。本节解释 CDC ACM 如何根据正在使用的写缓冲估算剩余字节，以及写完成回调如何通过 `tty_port_tty_wakeup()` 唤醒阻塞在 `write_wait` 上的进程。

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

`acm_wb_is_avail()` 会在 `acm->write_lock` 保护下扫描 CDC ACM 的写缓冲数组。只要某个 `acm_wb.use` 仍为 true，就说明对应的 USB 写 URB 还没有完成，`tcdrain()` 就不能认为发送队列已经排空。

```c
// linux/drivers/usb/class/cdc-acm.c
static int acm_wb_is_avail(struct acm *acm)
{
    int i, n = ACM_NW;                              // ACM_NW 是写缓冲个数
    unsigned long flags;

    spin_lock_irqsave(&acm->write_lock, flags);     // 保护写缓冲状态
    for (i = 0; i < ACM_NW; i++)
        if (acm->wb[i].use)                         // 该写缓冲仍被某个 URB 使用
            n--;
    spin_unlock_irqrestore(&acm->write_lock, flags);

    return n;                                      // 返回空闲写缓冲数量
}
```

这里有两个重要结构：

| 结构 | 含义 |
|------|------|
| `struct acm` | CDC ACM 设备实例，保存 USB 设备、TTY port、读写 URB、锁、状态位 |
| `struct acm_wb` | write buffer，一次 USB bulk OUT 写请求使用一个 `acm_wb` |

`acm_tty_chars_in_buffer()` 的返回值是估算值，源码注释里也说明它会 overcount：它按“正在使用的写缓冲数量 × 每个缓冲大小”估算，而不是精确返回实际剩余字节数。对 `tcdrain()` 来说这可以接受，因为它只关心是否还有未完成写缓冲；一旦所有写缓冲都空闲，返回值就变为 0。

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

static void acm_write_done(struct acm *acm, struct acm_wb *wb)
{
    wb->use = false;                               // 写缓冲重新变为空闲
    acm->transmitting--;                           // 正在传输的写请求数减一
    usb_autopm_put_interface_async(acm->control);  // 释放 runtime PM 引用
}

static void acm_softint(struct work_struct *work)
{
    struct acm *acm = container_of(work, struct acm, dwork.work); // 由 work 找回 acm

    if (test_and_clear_bit(EVENT_TTY_WAKEUP, &acm->flags))       // 如果有写完成唤醒事件
        tty_port_tty_wakeup(&acm->port);                         // 唤醒 TTY 写等待队列
}
```

```c
// linux/drivers/tty/tty_port.c
void tty_port_tty_wakeup(struct tty_port *port)
{
    port->client_ops->write_wakeup(port);          // 默认指向 tty_port_default_wakeup()
}

static void tty_port_default_wakeup(struct tty_port *port)
{
    struct tty_struct *tty = tty_port_tty_get(port);

    if (tty) {
        tty_wakeup(tty);                           // 最终唤醒 tty->write_wait
        tty_kref_put(tty);
    }
}

// linux/drivers/tty/tty_io.c
void tty_wakeup(struct tty_struct *tty)
{
    wake_up_interruptible_poll(&tty->write_wait, EPOLLOUT);
}
```

这就是 `clearPort()` 能等待“上一帧已经由 USB 写完成”的原因：

1. 上一次 `writePort()` 提交了 USB bulk OUT URB。
2. xHCI/USB 控制器把数据送到 USB 设备后，URB complete。
3. `acm_write_bulk()` 标记写缓冲空闲。
4. `tty_port_tty_wakeup()` 唤醒 `tty->write_wait`。
5. `tty_wait_until_sent()` 重新检查 `tty_chars_in_buffer()`，如果为 0 就返回。

这里的“写完成”指 USB 写 URB 已经完成，CDC ACM 的写缓冲已经释放；它仍不表示舵机侧已经完成机械动作，只表示本机 USB 串口发送队列已经排空。

---

## 7. clearPort 究竟完成了什么

`clearPort()` 完成的任务不是“丢弃残留包”，而是“等待之前已经排入 TTY/USB 驱动的输出数据完成发送”。

对舵机链路来说：

- Sync Read 发送前调用 `clearPort()`，避免上一帧还在 USB/TTY 写队列中。
- Sync Write 发送前调用 `clearPort()`，同样等待上一条输出完成。
- 由于 `arg=1`，内核不会向串口线发送 break。
- 如果当前没有未完成的 CDC ACM 写 URB，`tty_chars_in_buffer()` 返回 0，`clearPort()` 很快返回。

从系统调用角度看，`clearPort()` 和 `writePort()`、`readPort()` 的差异如下：

| 项目 | `clearPort()` 中是否发生 | 说明 |
| --- | --- | --- |
| 用户态协议帧复制到内核 | 否 | 没有传入舵机协议帧，只传入 `cmd=TCSBRK, arg=1` |
| 内核缓冲之间复制协议字节 | 否 | 不调用 CDC ACM 的 `write()`，也不创建新的写缓冲 |
| 新提交 USB URB | 否 | 它等待之前已经提交的写 URB 完成 |
| 查询 CDC ACM 写缓冲状态 | 是 | 通过 `acm_tty_chars_in_buffer()` 判断是否还有 `acm_wb.use=true` |
| 阻塞等待 | 可能 | 若仍有未完成写 URB，就睡在 `tty->write_wait` |
| 写完成唤醒 | 是 | `acm_write_bulk()` 释放写缓冲后通过 `tty_port_tty_wakeup()` 唤醒等待者 |

因此，`clearPort()` 的耗时主要取决于调用时是否还有上一帧未完成发送。如果上一帧已经完成，它只是一次很快的 `ioctl` 查询；如果上一帧还在 USB/串口物理链路中，它会等待写 URB 完成。

这条链路对应的系统调用参数最终是：

```text
ioctl(fd=/dev/ttyACM0, cmd=TCSBRK, arg=1)
```

内核做的核心动作是：

```text
等待 tty->write_wait，直到 acm_tty_chars_in_buffer(tty) == 0
```

需要特别区分的是，`clearPort()` 等待的是主机侧 TTY/CDC ACM 输出队列排空，不是等待舵机完成动作，也不是清空下一次 `readPort()` 要读取的回包。接收缓冲清理对应的是 pyserial 的 `reset_input_buffer()`，而不是这里的 `flush()`。
