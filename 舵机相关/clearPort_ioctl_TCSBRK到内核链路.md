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

来源：scservo_sdk/protocol_packet_handler.py:相关代码片段（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# scservo_sdk/protocol_packet_handler.py
port.clearPort()                         # 发送新协议帧前，先等待上一次输出排空
written_packet_length = port.writePort(txpacket)  # 然后把本次协议帧写入串口
```

SDK 起点源码：

来源：scservo_sdk/port_handler.py:clearPort（节选：仅保留本链路相关分支，已按当前仓库源码核对）
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

来源：linux/include/uapi/asm-generic/ioctls.h:TCSBRK
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

来源：serial/serialposix.py:flush（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# venv/lib/python3.13/site-packages/serial/serialposix.py
def flush(self):                          # file-like flush，语义是等待输出完成
    if not self.is_open:                  # 如果串口未打开
        raise PortNotOpenError()          # 抛出 pyserial 的端口未打开异常
    termios.tcdrain(self.fd)              # 调 CPython termios.tcdrain(fd)
```

`self.fd` 是 pyserial 在打开串口时得到的 Unix 文件描述符。它指向字符设备 `/dev/ttyACM0`，对应内核中的 `struct file`。

### 3.2 CPython `termios.tcdrain()`

来源：cpython/Modules/termios.c:termios_tcdrain_impl（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// cpython/Modules/termios.c
static PyObject *  // 本行参与当前 C 层路径的控制流或数据准备
termios_tcdrain_impl(PyObject *module, int fd)  // fd 是 Python 传入的 int 文件描述符
{
    termiosmodulestate *state = PyModule_GetState(module);  // 获取 termios 模块状态
    int r;                                                  // 保存 libc tcdrain 返回值

    Py_BEGIN_ALLOW_THREADS                                  // 释放 GIL，避免阻塞其他 Python 线程
    r = tcdrain(fd);                                        // 调用 libc 的 tcdrain(fd)
    Py_END_ALLOW_THREADS                                    // reacquire GIL  // 本行参与当前 C 层路径的控制流或数据准备

    if (r == -1) {                                          // libc 返回 -1 表示失败
        return PyErr_SetFromErrno(state->TermiosError);     // 把 errno 转成 Python 异常
    }
    Py_RETURN_NONE;                                         // 成功时 Python 返回 None
}
```

CPython 不解释 tty 语义，只负责释放 GIL，并把 `fd` 原样交给 libc。

### 3.3 glibc `tcdrain()`

来源：glibc-2.42/sysdeps/unix/sysv/linux/tcdrain.c:__libc_tcdrain（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// glibc-2.42/sysdeps/unix/sysv/linux/tcdrain.c
int  // 本行参与当前 C 层路径的控制流或数据准备
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

来源：linux/fs/ioctl.c:SYSCALL_DEFINE ioctl（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/fs/ioctl.c
SYSCALL_DEFINE3(ioctl, unsigned int, fd, unsigned int, cmd, unsigned long, arg)  // 定义 Linux 系统调用入口，用户态 syscall 会进入这里
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

out:  // 本行参与当前 C 层路径的控制流或数据准备
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

来源：linux/drivers/tty/tty_io.c:相关代码片段（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
static const struct file_operations tty_fops = {  // 调用 TTY 层接口处理串口设备语义
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

来源：linux/drivers/tty/tty_io.c:tty_ioctl（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
long tty_ioctl(struct file *file, unsigned int cmd, unsigned long arg)  // 定义当前层的 C 函数入口
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

来源：linux/drivers/tty/tty_ioctl.c:tty_wait_until_sent（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_ioctl.c
void tty_wait_until_sent(struct tty_struct *tty, long timeout)  // 定义当前层的 C 函数入口
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

`wait_event_interruptible_timeout()` 本身是宏，不是普通函数。它把“等待队列 + 条件表达式 + 超时时间”展开成一段固定等待模板：先检查条件，不满足就把当前任务挂进等待队列，并通过 `schedule_timeout()` 让出 CPU。

来源：linux/include/linux/wait.h:wait_event_interruptible_timeout / __wait_event_interruptible_timeout / ___wait_event（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/include/linux/wait.h（加注释版）
#define wait_event_interruptible_timeout(wq_head, condition, timeout) \
({                                                                    \
    long __ret = timeout;                                             /* 保存剩余等待时间，单位是 jiffies */ \
    might_sleep();                                                    /* 标记这里可能睡眠，调试时可检查非法睡眠场景 */ \
    if (!___wait_cond_timeout(condition))                             /* 先立即检查一次条件；已经满足就不睡 */ \
        __ret = __wait_event_interruptible_timeout(                   /* 条件不满足，进入真正的等待循环 */ \
            wq_head, condition, timeout);                             /* wq_head 是等待队列，condition 会被反复求值 */ \
    __ret;                                                            /* 返回剩余时间、0、1 或 -ERESTARTSYS */ \
})

#define ___wait_cond_timeout(condition)                               \
({                                                                    \
    bool __cond = (condition);                                        /* 重新计算等待条件，比如 !tty_chars_in_buffer(tty) */ \
    if (__cond && !__ret)                                             /* 条件刚好在超时边界成立 */ \
        __ret = 1;                                                    /* 返回 1，表示条件成立但没有剩余时间 */ \
    __cond || !__ret;                                                 /* 条件成立或已经超时，都让外层跳出等待 */ \
})

#define __wait_event_interruptible_timeout(wq_head, condition, timeout) \
    ___wait_event(                                                      /* 调用通用等待模板 */ \
        wq_head,                                                        /* 等待队列：本链路是 tty->write_wait */ \
        ___wait_cond_timeout(condition),                                /* 等待条件：本链路是 !tty_chars_in_buffer(tty) */ \
        TASK_INTERRUPTIBLE,                                             /* 可被信号打断的睡眠状态 */ \
        0,                                                              /* 非 exclusive waiter，唤醒时不独占事件 */ \
        timeout,                                                        /* 初始最大等待时间 */ \
        __ret = schedule_timeout(__ret))                                /* 真正睡眠：让出 CPU，等待唤醒或超时 */

#define ___wait_event(wq_head, condition, state, exclusive, ret, cmd) \
({                                                                   \
    __label__ __out;                                                 /* 宏内部跳转出口 */ \
    struct wait_queue_entry __wq_entry;                              /* 当前任务挂入等待队列用的节点 */ \
    long __ret = ret;                                                /* 本层返回值，初始为 timeout */ \
                                                                     \
    init_wait_entry(&__wq_entry, exclusive ? WQ_FLAG_EXCLUSIVE : 0);  /* 初始化等待节点 */ \
    for (;;) {                                                       /* 被唤醒后还会回到这里重新检查条件 */ \
        long __int = prepare_to_wait_event(                          /* 把当前任务加入等待队列，并设置任务状态 */ \
            &wq_head, &__wq_entry, state);                           /* state 是 TASK_INTERRUPTIBLE */ \
                                                                     \
        if (condition)                                               /* 条件成立：本链路就是输出缓冲已经排空 */ \
            break;                                                   /* 跳出等待循环 */ \
                                                                     \
        if (___wait_is_interruptible(state) && __int) {              /* 如果睡眠可中断且当前有信号待处理 */ \
            __ret = __int;                                           /* 通常是 -ERESTARTSYS */ \
            goto __out;                                              /* 被信号打断，直接退出 */ \
        }                                                            \
                                                                     \
        cmd;                                                         /* 执行 schedule_timeout()：当前任务真正睡眠 */ \
    }                                                                \
    finish_wait(&wq_head, &__wq_entry);                              /* 恢复 TASK_RUNNING，并从等待队列移除 */ \
__out:                                                              \
    __ret;                                                           /* 把等待结果返回给调用者 */ \
})
```

套回本链路后，宏里的关键参数就是：

```c
wq_head   = tty->write_wait
condition = !tty_chars_in_buffer(tty)
state     = TASK_INTERRUPTIBLE
cmd       = __ret = schedule_timeout(__ret)
```

所以这里的“等待”具体就是：`prepare_to_wait_event()` 把当前 `clearPort()` 所在线程挂到 `tty->write_wait`，`schedule_timeout()` 让当前线程睡眠，后续 CDC ACM 写完成回调通过 `tty_wakeup()` 唤醒它；醒来后宏会重新计算 `!tty_chars_in_buffer(tty)`，只有条件成立才退出循环。

#### 5.2.1 为什么调用这个宏就能睡眠等待 event

先把话说白：这里的 “event” 不是一个单独的 `struct event` 对象，也不是“直接等硬件中断”。Linux 这里的 event 是：

```text
等待队列 tty->write_wait 被 wake_up，
并且条件 !tty_chars_in_buffer(tty) 重新检查为 true。
```

所以它等的是两个东西配合：

```text
1. 有人唤醒 tty->write_wait
2. 醒来后发现 tty_chars_in_buffer(tty) == 0
```

套进 `clearPort()` 这条链路，宏展开后可以近似看成下面这段代码：

```c
// wait_event_interruptible_timeout(
//     tty->write_wait,
//     !tty_chars_in_buffer(tty),
//     timeout)
// 展开后的关键逻辑，省略少量宏细节。

struct wait_queue_entry wait;                         // 当前线程要挂到等待队列里的节点
long ret = timeout;                                   // 剩余等待时间

init_wait_entry(&wait, 0);                            // wait.private = current

for (;;) {
    long signal_ret;

    signal_ret = prepare_to_wait_event(               // 把当前线程挂到 tty->write_wait
        &tty->write_wait,
        &wait,
        TASK_INTERRUPTIBLE);                          // 把当前线程状态设成可被信号打断的睡眠态

    if (!tty_chars_in_buffer(tty))                    // 每次睡前/醒后都重新检查：输出缓冲是否已经空
        break;                                        // 条件成立，说明不用再等

    if (signal_ret) {                                 // 有信号打断
        ret = signal_ret;                             // 通常是 -ERESTARTSYS
        goto out;
    }

    ret = schedule_timeout(ret);                      // 真正睡眠：当前 clearPort 线程让出 CPU
}

finish_wait(&tty->write_wait, &wait);                 // 从等待队列移除，并恢复 TASK_RUNNING

out:
return ret;
```

上面这段里面，`init_wait_entry()` 让等待节点记住“睡的是谁”：

来源：linux/kernel/sched/wait.c:init_wait_entry（节选）
```c
void init_wait_entry(struct wait_queue_entry *wq_entry, int flags)
{
    wq_entry->flags = flags;
    wq_entry->private = current;                      // 关键：保存当前 clearPort 线程
    wq_entry->func = autoremove_wake_function;        // 关键：被 wake_up 时调用这个函数
    INIT_LIST_HEAD(&wq_entry->entry);
}
```

`prepare_to_wait_event()` 让当前线程正式“睡到这个队列上”：

来源：linux/kernel/sched/wait.c:prepare_to_wait_event（节选）
```c
long prepare_to_wait_event(struct wait_queue_head *wq_head,
                           struct wait_queue_entry *wq_entry,
                           int state)
{
    unsigned long flags;
    long ret = 0;

    spin_lock_irqsave(&wq_head->lock, flags);         // 锁住等待队列
    if (signal_pending_state(state, current)) {
        list_del_init(&wq_entry->entry);              // 如果已有信号，取消等待
        ret = -ERESTARTSYS;
    } else {
        if (list_empty(&wq_entry->entry))
            __add_wait_queue(wq_head, wq_entry);      // 关键：把 wait 节点挂进 tty->write_wait
        set_current_state(state);                     // 关键：current->__state = TASK_INTERRUPTIBLE
    }
    spin_unlock_irqrestore(&wq_head->lock, flags);

    return ret;
}
```

到这里还没有真正切走 CPU。真正切走 CPU 是下一句：

```c
ret = schedule_timeout(ret);                          // 进入调度器，当前线程不再继续运行
```

所以“睡”的代码层事实是：

```text
wait.private = current
wait.func = autoremove_wake_function
__add_wait_queue(&tty->write_wait, &wait)
set_current_state(TASK_INTERRUPTIBLE)
schedule_timeout()
```

那它怎么醒？写完成后，CDC ACM 先释放写缓冲，然后一路调用到 `tty_wakeup()`：

来源：linux/drivers/usb/class/cdc-acm.c:acm_write_bulk / acm_softint（节选）
```c
static void acm_write_bulk(struct urb *urb)
{
    struct acm_wb *wb = urb->context;
    struct acm *acm = wb->instance;
    unsigned long flags;

    spin_lock_irqsave(&acm->write_lock, flags);
    acm_write_done(acm, wb);                          // 关键：wb->use = false，写缓冲释放
    spin_unlock_irqrestore(&acm->write_lock, flags);

    set_bit(EVENT_TTY_WAKEUP, &acm->flags);           // 标记需要唤醒 TTY 写等待者
    schedule_delayed_work(&acm->dwork, 0);            // 把唤醒动作交给 workqueue
}

static void acm_softint(struct work_struct *work)
{
    struct acm *acm = container_of(work, struct acm, dwork.work);

    if (test_and_clear_bit(EVENT_TTY_WAKEUP, &acm->flags))
        tty_port_tty_wakeup(&acm->port);              // 进入 TTY port 唤醒路径
}
```

来源：linux/drivers/tty/tty_port.c:tty_port_tty_wakeup / tty_port_default_wakeup（节选）
```c
void tty_port_tty_wakeup(struct tty_port *port)
{
    port->client_ops->write_wakeup(port);             // 默认就是 tty_port_default_wakeup()
}

static void tty_port_default_wakeup(struct tty_port *port)
{
    struct tty_struct *tty = tty_port_tty_get(port);

    if (tty) {
        tty_wakeup(tty);                              // 终于调用到 tty_wakeup()
        tty_kref_put(tty);
    }
}
```

`tty_wakeup()` 最后唤醒 `tty->write_wait`：

来源：linux/drivers/tty/tty_io.c:tty_wakeup（节选）
```c
void tty_wakeup(struct tty_struct *tty)
{
    struct tty_ldisc *ld;

    if (test_bit(TTY_DO_WRITE_WAKEUP, &tty->flags)) {
        ld = tty_ldisc_ref(tty);
        if (ld) {
            if (ld->ops->write_wakeup)
                ld->ops->write_wakeup(tty);
            tty_ldisc_deref(ld);
        }
    }

    wake_up_interruptible_poll(&tty->write_wait, EPOLLOUT); // 关键：唤醒睡在 write_wait 上的任务
}
```

`wake_up_interruptible_poll()` 又是宏，最终进入 `__wake_up()`：

来源：linux/include/linux/wait.h:wake_up_interruptible_poll（节选）
```c
#define wake_up_interruptible_poll(x, m) \
    __wake_up(x, TASK_INTERRUPTIBLE, 1, poll_to_key(m))
```

`__wake_up()` 会遍历等待队列，调用每个等待节点的 `func`：

来源：linux/kernel/sched/wait.c:__wake_up / __wake_up_common_lock / autoremove_wake_function（节选）
```c
int __wake_up(struct wait_queue_head *wq_head,
              unsigned int mode,
              int nr_exclusive,
              void *key)
{
    return __wake_up_common_lock(wq_head, mode, nr_exclusive, 0, key);
}

static int __wake_up_common_lock(struct wait_queue_head *wq_head,
                                 unsigned int mode,
                                 int nr_exclusive,
                                 int wake_flags,
                                 void *key)
{
    unsigned long flags;
    int remaining;

    spin_lock_irqsave(&wq_head->lock, flags);         // 锁住 tty->write_wait
    remaining = __wake_up_common(wq_head, mode, nr_exclusive,
                                 wake_flags, key);    // 遍历等待队列，调用 wait.func
    spin_unlock_irqrestore(&wq_head->lock, flags);

    return nr_exclusive - remaining;
}

int autoremove_wake_function(struct wait_queue_entry *wq_entry,
                             unsigned mode,
                             int sync,
                             void *key)
{
    int ret = default_wake_function(wq_entry, mode, sync, key);

    if (ret)
        list_del_init_careful(&wq_entry->entry);      // 唤醒成功后，从等待队列移除

    return ret;
}
```

`default_wake_function()` 会把当初睡下去的 `current` 重新唤醒：

来源：linux/kernel/sched/core.c:default_wake_function / try_to_wake_up（节选）
```c
int default_wake_function(wait_queue_entry_t *curr,
                          unsigned mode,
                          int wake_flags,
                          void *key)
{
    return try_to_wake_up(curr->private, mode, wake_flags);
    // curr->private 就是 init_wait_entry() 里保存的 current，
    // 也就是睡在 tty->write_wait 上的 clearPort 线程。
}

int try_to_wake_up(struct task_struct *p,
                   unsigned int state,
                   int wake_flags)
{
    wake_flags |= WF_TTWU;

    /*
     * 真实源码很长，这里保留关键结果：
     * 如果 p 当前状态匹配 TASK_INTERRUPTIBLE，
     * 就把 p 改回可运行，并放回某个 CPU 的 runqueue。
     */
    cpu = select_task_rq(p, p->wake_cpu, &wake_flags);
    ttwu_queue(p, cpu, wake_flags);                   // 关键：把被唤醒任务放回 runqueue
    return success;
}
```

所以“醒”的代码层事实是：

```text
tty_wakeup()
  -> wake_up_interruptible_poll(&tty->write_wait, EPOLLOUT)
  -> __wake_up(&tty->write_wait, TASK_INTERRUPTIBLE, ...)
  -> 遍历 tty->write_wait 上的 wait_queue_entry
  -> 调 wait.func，也就是 autoremove_wake_function()
  -> default_wake_function()
  -> try_to_wake_up(wait.private, TASK_INTERRUPTIBLE, ...)
  -> wait.private 这个 clearPort 线程重新进入 runqueue
```

最后一定要注意：被唤醒不等于马上返回成功。它只是重新变成“可运行”。等调度器再次切回这个线程后，`schedule_timeout()` 返回，wait 宏会重新执行：

```c
if (!tty_chars_in_buffer(tty))
    break;
```

只有这个条件成立，`clearPort()` 才真的结束等待。对 CDC ACM 来说，这个条件成立的原因是前面 `acm_write_done()` 已经把写缓冲标成空闲：

```c
wb->use = false;
```

因此整个闭环是：

```text
clearPort 线程睡到 tty->write_wait
  -> 等待 “有人 wake_up tty->write_wait”
  -> 写 URB 完成后 CDC ACM 调 tty_wakeup()
  -> tty_wakeup 唤醒 tty->write_wait
  -> clearPort 线程回到 runqueue
  -> 被调度回来后重新检查 chars_in_buffer
  -> chars_in_buffer == 0 才返回
```

#### 5.2.2 `schedule_timeout()` 到 `schedule()` 的调度链路

上面的宏里，`cmd` 实际执行的是：

```c
__ret = schedule_timeout(__ret)
```

这一步才是真正“睡下去”的地方。`prepare_to_wait_event()` 只是把当前任务挂到 `tty->write_wait` 并把任务状态设置成 `TASK_INTERRUPTIBLE`；`schedule_timeout()` 会进入调度器，把 CPU 交给其他可运行任务。

来源：linux/kernel/time/timer.c:schedule_timeout（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/kernel/time/timer.c
signed long __sched schedule_timeout(signed long timeout)  // 当前任务按 timeout 睡眠
{
    struct process_timer timer;                            // 有限超时时使用的内核定时器
    unsigned long expire;                                  // 超时到期的 jiffies

    switch (timeout) {
    case MAX_SCHEDULE_TIMEOUT:                             // clearPort/tcdrain 常走这里：无限等待
        schedule();                                        // 不设置定时器，直接进入调度器睡眠
        goto out;                                          // 被 wake_up 唤醒后从这里返回
    default:
        if (timeout < 0) {                                 // 防御非法 timeout
            __set_current_state(TASK_RUNNING);             // 恢复运行态
            goto out;
        }
    }

    expire = timeout + jiffies;                            // 有限等待：计算到期时间
    timer.task = current;                                  // 定时器到期时唤醒当前任务
    timer_setup_on_stack(&timer.timer, process_timeout, 0);
    __mod_timer(&timer.timer, expire, MOD_TIMER_NOTPENDING);

    schedule();                                            // 进入调度器，当前任务让出 CPU

    del_timer_sync(&timer.timer);                          // 被提前唤醒时删除定时器
    destroy_timer_on_stack(&timer.timer);
    timeout = expire - jiffies;                            // 返回剩余等待时间

out:
    return timeout < 0 ? 0 : timeout;                      // 超时返回 0，提前唤醒返回剩余 jiffies
}
```

在本链路里，`tty_wait_until_sent(tty, 0)` 先把 `timeout=0` 转成 `MAX_SCHEDULE_TIMEOUT`，所以 `schedule_timeout()` 不会设置定时器，而是直接调用 `schedule()`。这表示：如果没有信号，也没有写完成唤醒，它可以一直睡。

`schedule()` 的定义在调度器核心里：

来源：linux/kernel/sched/core.c:schedule / __schedule_loop（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/kernel/sched/core.c
asmlinkage __visible void __sched schedule(void)  // 显式让当前任务进入一次调度
{
    struct task_struct *tsk = current;            // current 是当前 clearPort 所在线程

    if (!task_is_running(tsk))                    // 前面已被设置成 TASK_INTERRUPTIBLE
        sched_submit_work(tsk);                   // 睡眠前提交可能积压的工作

    __schedule_loop(SM_NONE);                     // 进入核心调度循环

    sched_update_worker(tsk);                     // 如果是 worker 线程，恢复相关状态
}

static __always_inline void __schedule_loop(int sched_mode)
{
    do {
        preempt_disable();                        // 调度切换期间禁止抢占
        __schedule(sched_mode);                   // 选择下一个任务并做上下文切换
        sched_preempt_enable_no_resched();        // 恢复抢占，但先不立即重调度
    } while (need_resched());                     // 如果仍需要调度，继续循环
}
```

真正决定“当前任务是不是要被拿下 CPU”的逻辑在 `__schedule()`：

来源：linux/kernel/sched/core.c:__schedule（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/kernel/sched/core.c
static void __sched notrace __schedule(int sched_mode)
{
    struct task_struct *prev, *next;
    unsigned long prev_state;
    struct rq_flags rf;
    struct rq *rq;

    rq = cpu_rq(smp_processor_id());              // 当前 CPU 的 runqueue
    prev = rq->curr;                              // 当前正在 CPU 上跑的任务，也就是 clearPort 线程

    local_irq_disable();                          // 关闭本地中断，保护调度关键区
    rq_lock(rq, &rf);                             // 锁住当前 CPU 的运行队列
    update_rq_clock(rq);                          // 更新 runqueue 时钟

    prev_state = READ_ONCE(prev->__state);        // 读取当前任务状态
    if (sched_mode != SM_PREEMPT && prev_state) { // 非抢占调度，且当前任务不是 TASK_RUNNING
        try_to_block_task(rq, prev, &prev_state); // 把当前任务从可运行队列转为阻塞/睡眠
    }

    next = pick_next_task(rq, prev, &rf);         // 从 runqueue 里选择下一个应该运行的任务

    if (prev != next) {                           // 如果选出来的不是自己
        RCU_INIT_POINTER(rq->curr, next);         // 当前 CPU 的 current 切到 next
        trace_sched_switch(false, prev, next, prev_state);
        rq = context_switch(rq, prev, next, &rf); // 真正切换地址空间、寄存器和内核栈
    } else {
        rq_unpin_lock(rq, &rf);                   // 如果还是自己，就解锁继续跑
        raw_spin_rq_unlock_irq(rq);
    }
}
```

最后的 `context_switch()` 负责真正切换执行现场：

来源：linux/kernel/sched/core.c:context_switch（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/kernel/sched/core.c
context_switch(struct rq *rq, struct task_struct *prev,
               struct task_struct *next, struct rq_flags *rf)
{
    prepare_task_switch(rq, prev, next);          // 调度切换前的通用准备

    if (next->mm)
        switch_mm_irqs_off(prev->active_mm, next->mm, next); // 必要时切换用户地址空间
    else
        enter_lazy_tlb(prev->active_mm, next);    // 下一个是内核线程时使用 lazy TLB

    prepare_lock_switch(rq, next, rf);            // 准备释放 runqueue 锁并切换任务

    switch_to(prev, next, prev);                  // 架构相关：切换寄存器、栈指针、CPU 执行上下文
    barrier();

    return finish_task_switch(prev);              // 切回来后收尾，完成 prev 的切换后处理
}
```

把这段和 `clearPort()` 合起来看，就是：

```text
prepare_to_wait_event()
  -> 当前任务加入 tty->write_wait
  -> 当前任务状态设为 TASK_INTERRUPTIBLE

schedule_timeout(MAX_SCHEDULE_TIMEOUT)
  -> schedule()
  -> __schedule_loop()
  -> __schedule()
  -> try_to_block_task()      # 当前 clearPort 线程不再是可运行任务
  -> pick_next_task()         # 选择别的任务运行
  -> context_switch()
  -> switch_to()              # CPU 真正切到别的任务

CDC ACM 写完成后 tty_wakeup()
  -> wake_up_interruptible_poll(&tty->write_wait, EPOLLOUT)
  -> clearPort 线程重新进入 runqueue
  -> 某次调度再切回 clearPort 线程
  -> schedule_timeout() 返回
  -> wait 宏重新检查 !tty_chars_in_buffer(tty)
```

核心是 `!tty_chars_in_buffer(tty)`。它会问具体驱动还有多少字节排在输出队列里。对 CDC ACM 来说，`acm_ops` 没有提供 `.wait_until_sent` 钩子，因此主要判断就落在 `chars_in_buffer` 是否变为 0。

### 5.3 `tty_chars_in_buffer()`

来源：linux/drivers/tty/tty_ioctl.c:tty_chars_in_buffer（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_ioctl.c
unsigned int tty_chars_in_buffer(struct tty_struct *tty)  // 定义当前层的 C 函数入口
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

来源：linux/drivers/usb/class/cdc-acm.c:相关代码片段（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static const struct tty_operations acm_ops = {  // 调用 TTY 层接口处理串口设备语义
    .write           = acm_tty_write,              // TTY 写数据时调用
    .write_room      = acm_tty_write_room,         // 查询还能写多少
    .flush_buffer    = acm_tty_flush_buffer,       // 刷新输出队列时调用
    .chars_in_buffer = acm_tty_chars_in_buffer,    // tcdrain 查询剩余输出字节时调用
};
```

`struct tty_operations` 是 TTY 核心和具体硬件驱动之间的接口表。CDC ACM 驱动用它把通用 TTY 操作映射到 USB ACM 设备。

### 6.2 `acm_tty_chars_in_buffer()`

来源：linux/drivers/usb/class/cdc-acm.c:acm_tty_chars_in_buffer（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static unsigned int acm_tty_chars_in_buffer(struct tty_struct *tty)  // 定义当前层的 C 函数入口
{
    struct acm *acm = tty->driver_data;            // tty->driver_data 指向 CDC ACM 私有结构

    if (acm->disconnected)                         // 如果 USB 设备已经拔掉
        return 0;                                  // 剩余字节视为 0

    return (ACM_NW - acm_wb_is_avail(acm)) * acm->writesize;  // 把本层处理结果或错误码返回上一层
                                                    // 估算正在使用的写缓冲数量 × 每个缓冲大小
}
```

`acm_wb_is_avail()` 会在 `acm->write_lock` 保护下扫描 CDC ACM 的写缓冲数组。只要某个 `acm_wb.use` 仍为 true，就说明对应的 USB 写 URB 还没有完成，`tcdrain()` 就不能认为发送队列已经排空。

来源：linux/drivers/usb/class/cdc-acm.c:acm_wb_is_avail（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static int acm_wb_is_avail(struct acm *acm)  // 定义当前层的 C 函数入口
{
    int i, n = ACM_NW;                              // ACM_NW 是写缓冲个数
    unsigned long flags;  // 本行参与当前 C 层路径的控制流或数据准备

    spin_lock_irqsave(&acm->write_lock, flags);     // 保护写缓冲状态
    for (i = 0; i < ACM_NW; i++)  // 循环处理队列、缓冲区或 fd 集合
        if (acm->wb[i].use)                         // 该写缓冲仍被某个 URB 使用
            n--;  // 本行参与当前 C 层路径的控制流或数据准备
    spin_unlock_irqrestore(&acm->write_lock, flags);  // 调用下一层 C 函数继续完成当前路径

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

来源：linux/drivers/usb/class/cdc-acm.c:acm_write_bulk（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static void acm_write_bulk(struct urb *urb)  // 定义当前层的 C 函数入口
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

static void acm_write_done(struct acm *acm, struct acm_wb *wb)  // 定义当前层的 C 函数入口
{
    wb->use = false;                               // 写缓冲重新变为空闲
    acm->transmitting--;                           // 正在传输的写请求数减一
    usb_autopm_put_interface_async(acm->control);  // 释放 runtime PM 引用
}

static void acm_softint(struct work_struct *work)  // 定义当前层的 C 函数入口
{
    struct acm *acm = container_of(work, struct acm, dwork.work); // 由 work 找回 acm

    if (test_and_clear_bit(EVENT_TTY_WAKEUP, &acm->flags))       // 如果有写完成唤醒事件
        tty_port_tty_wakeup(&acm->port);                         // 唤醒 TTY 写等待队列
}
```

来源：linux/drivers/tty/tty_port.c:tty_port_tty_wakeup（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_port.c
void tty_port_tty_wakeup(struct tty_port *port)  // 定义当前层的 C 函数入口
{
    port->client_ops->write_wakeup(port);          // 默认指向 tty_port_default_wakeup()
}

static void tty_port_default_wakeup(struct tty_port *port)  // 定义当前层的 C 函数入口
{
    struct tty_struct *tty = tty_port_tty_get(port);  // 定义当前链路涉及的内核数据结构

    if (tty) {  // 检查状态或错误码，决定是否走异常/分支路径
        tty_wakeup(tty);                           // 最终唤醒 tty->write_wait
        tty_kref_put(tty);  // 调用 TTY 层接口处理串口设备语义
    }
}

// linux/drivers/tty/tty_io.c
void tty_wakeup(struct tty_struct *tty)  // 定义当前层的 C 函数入口
{
    wake_up_interruptible_poll(&tty->write_wait, EPOLLOUT);  // 处理 fd 就绪检查和等待唤醒逻辑
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
