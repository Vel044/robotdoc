# writePort：write 系统调用到 Linux 内核链路

本文从 `scservo_sdk/port_handler.py` 的 `PortHandler.writePort(packet)` 开始，重点追踪它触发的 `write()` 系统调用：一帧 Feetech 协议数据如何从 Python 写入 `/dev/ttyACM0`，进入 Linux TTY 层、CDC ACM 驱动、USB core，最后提交给 xHCI 主控。

说明：pyserial 的阻塞写路径在 `os.write()` 之后还可能调用 `select()`，因此 strace/ftrace 里会看到 `pselect6`。这个调用只负责检查可写/取消等待，不是舵机协议字节的数据提交主链路，单独见 [pyserial_select_pselect6到内核链路.md](pyserial_select_pselect6到内核链路.md)。

这里以两个典型包为例：

- Sync Read 请求包：14 字节，`FF FF FE 0A 82 38 02 01 02 03 04 05 06 CS`
- Sync Write 请求包：26 字节，`FF FF FE 16 83 2A 02 [id+data×6] CS`

---

## 图：`write()` 在内核中的主链路

```
VFS
  │  `vfs_write()` 分发到 tty file_operations
  ▼
TTY / N_TTY
  │  `tty_write()` -> `n_tty_write()`
  ▼
CDC ACM
  │  `acm_tty_write()` 把字节写入 URB 缓冲
  ▼
USB core
  │  `usb_submit_urb()`
  ▼
xHCI driver
  │  `xhci_queue_bulk_tx()` 组 TRB 并写入 transfer ring
  │  `xhci_ring_ep_doorbell()` 通知硬件取 TRB
  ▼
xHCI hardware
  │  DMA 读缓冲并发出 USB bulk OUT
```

说明：这张图只画 `write()` 进入内核后的主数据路径。`write(fd, buf, count)` 本身是系统调用入口，不单独作为一层；如果只想表达 `/dev/ttyACM0` 的写链路，用 `VFS -> TTY/N_TTY -> CDC ACM -> USB core -> xHCI driver -> xHCI hardware` 就够了。

---

## 1. scservo_sdk 入口

`writePort()` 被 `protocol_packet_handler.txPacket()` 调用，前面已经由 SDK 填好包头和校验和。

```python
# scservo_sdk/port_handler.py
def writePort(self, packet):             # packet 是 list[int] 或 bytearray，表示完整协议帧
    return self.ser.write(packet)        # 调用 pyserial Serial.write()
```

进入系统调用时的关键参数：

| 场景            | `fd`                      | `buf`         | `count` |
| --------------- | ------------------------- | ------------- | ------- |
| Sync Read 请求  | `/dev/ttyACM0` 文件描述符 | 14 字节协议帧 | 14      |
| Sync Write 请求 | `/dev/ttyACM0` 文件描述符 | 26 字节协议帧 | 26      |

注意：`writePort()` 返回的是内核接受写入的字节数。对于 CDC ACM，它通常表示数据已经复制进驱动写缓冲并提交 USB URB，不等价于舵机已经执行动作。

---

## 2. 完整调用栈

从 Python 的 `writePort()` 到 xHCI 硬件，写链路跨越用户态和内核态两大边界。内核外是 SDK → pyserial → CPython → glibc 的层层封装；内核内是 VFS → TTY → N_TTY → CDC ACM → USB core → xHCI 的逐级分发。本节用两张调用栈图分别展示用户态和内核态的完整函数调用路径，帮助建立全局视角。

### 2.1 内核外调用栈

```text
protocol_packet_handler.txPacket()
│  SDK 协议层：已经填好 Header、ID、Length、Instruction、参数和 Checksum。
└── PortHandler.writePort(txpacket)
    │  SDK 串口封装层：把完整协议帧交给 pyserial；Sync Read 是 14 字节，Sync Write 是 26 字节。
    └── serial.Serial.write(packet)
        │  pyserial POSIX 后端：把 list/bytearray 转成 bytes，并循环调用 os.write 直到写完。
        └── os.write(fd, bytes(packet))
            │  Python 标准库入口：fd 是 /dev/ttyACM0，buf 是协议帧用户态地址，count 是帧长。
            └── CPython os_write_impl(fd, data)
                │  CPython posix 模块：从 Py_buffer 取 data->buf 和 data->len。
                └── _Py_write(fd, buf, len)
                    │  CPython 文件工具层：释放 GIL，处理 EINTR 重试，然后调用 libc write。
                    └── glibc write(fd, buf, count)
                        │  libc syscall 包装：把 fd/buf/count 放入 ARM64 寄存器并触发 __NR_write=64。
                        └── syscall boundary
                            │  从这里进入 Linux 内核：ARM64 上 x8=__NR_write=64。
```

### 2.2 内核内调用栈

```text
Linux SYSCALL_DEFINE3(write, fd, buf, count)
│  内核 syscall 入口：接收用户态 fd、用户态 buf 指针和 count。
└── ksys_write(fd, buf, count)
    │  fdget_pos() + file_ppos()：把 int fd 转成 struct file，并处理文件偏移语义。
    └── vfs_write(file, buf, count, pos)
        │  access_ok() + rw_verify_area()：检查用户地址、写权限和写入范围。
        │  file_start_write()：标记一次文件写入开始；返回路径会执行 file_end_write()。
        └── new_sync_write(file, buf, count, pos)
            │  init_sync_kiocb() + iov_iter_ubuf()：把用户缓冲包装成 kiocb + iov_iter。
            └── filp->f_op->write_iter(iocb, iter)
                │  /dev/ttyACM0 对应 tty_fops.write_iter。
                └── tty_write(iocb, iter)
                    │  TTY file_operations 写入口：转入 file_tty_write。
                    └── file_tty_write(file, iocb, iter)
                        │  tty_ldisc_ref_wait()：找到 tty_struct 和当前行规程 N_TTY。
                        └── iterate_tty_write(ld, tty, file, iter)
                            │  tty_write_lock()：串行化同一个 TTY 的写入。
                            │  copy_from_iter()：从用户态 iov_iter 分块复制到 tty->write_buf。
                            └── ld->ops->write(tty, file, kbuf, nr)
                                │  N_TTY 行规程 write。
                                └── n_tty_write(tty, file, kbuf, nr)
                                    │  termios_rwsem + output_lock；原始模式下基本不改字节。
                                    └── tty->ops->write(tty, b, nr)
                                        │  tty_operations 分发：/dev/ttyACM0 对应 acm_ops.write。
                                        └── acm_tty_write(tty, buf, count)
                                            │  acm_wb_alloc()：取空闲写缓冲。
                                            │  memcpy()：把协议帧复制到 CDC ACM 的 USB 写缓冲。
                                            │  usb_autopm_get_interface_async()：增加 USB runtime PM 引用。
                                            └── acm_start_wb(acm, wb)
                                                │  填 URB 的 transfer_buffer、DMA 地址、transfer_buffer_length 和 USB 设备。
                                                └── usb_submit_urb(wb->urb, GFP_ATOMIC)
                                                    │  USB core：校验 URB、endpoint、方向和长度，标记传输进行中。
                                                    └── usb_hcd_submit_urb(urb, GFP_ATOMIC)
                                                        │  map_urb_for_dma()：处理 DMA 访问关系。
                                                        └── hcd->driver->urb_enqueue()
                                                            │  HCD 分发表：树莓派 5 USB 主控走 xHCI 的 urb_enqueue。
                                                            └── xhci_urb_enqueue()
                                                                │  分配 urb_priv，检查 slot/endpoint 状态，选择传输类型。
                                                                └── xhci_queue_bulk_tx()
                                                                    │  queue_trb()：把 bulk OUT 数据描述成 TRB，放入 transfer ring。
                                                                    └── giveback_first_trb()
                                                                        │  交出首个 TRB 的 cycle bit。
                                                                        └── xhci_ring_ep_doorbell()
                                                                            │  写 doorbell 寄存器，通知硬件开始取 TRB。
```

对 `/dev/ttyACM0`，具体硬件驱动是 `linux/drivers/usb/class/cdc-acm.c`。

---

## 3. pyserial 与 CPython 层

在真正触发系统调用之前，协议帧要穿过 Python 生态的三层封装：pyserial 做串口抽象、CPython 的 `os` 模块做 Python 对象到 C 类型的转换、glibc 做最终的 syscall 包装。这三层都不理解串口语义，只是负责把数据从 Python 对象搬运到内核入口。本节逐层解释数据在离开 Python 之前经历了什么。

### 3.1 pyserial `write()`

```python
# venv/lib/python3.13/site-packages/serial/serialposix.py
def write(self, data):                         # data 是 SDK 传入的协议帧
    if not self.is_open:                       # 串口必须已经打开
        raise PortNotOpenError()               # 未打开则抛异常
    d = to_bytes(data)                         # list[int]/bytearray 转成 bytes
    tx_len = length = len(d)                   # length 是总字节数，Sync Read=14，Sync Write=26
    timeout = Timeout(self._write_timeout)     # 写超时，默认通常是 None
    while tx_len > 0:                          # 循环直到所有字节都写出去
        n = os.write(self.fd, d)               # 触发 POSIX write(fd, buf, count)
        if timeout.is_non_blocking:            # 如果配置了非阻塞写
            return n                           # 只返回本次写入字节数
        d = d[n:]                              # 去掉已经写入的前 n 字节
        tx_len -= n                            # 更新剩余字节数
    return length - len(d)                     # 返回总写入字节数
```

`to_bytes(data)` 之后，用户态 `buf` 是连续字节数组：

```text
Sync Read:  14 字节，count=14
Sync Write: 26 字节，count=26
```

### 3.2 CPython `os.write()`

```c
// cpython/Modules/posixmodule.c
static Py_ssize_t
os_write_impl(PyObject *module, int fd, Py_buffer *data)
{
    return _Py_write(fd, data->buf, data->len);  // fd 原样传入，data->buf 是 bytes 数据地址
}

// cpython/Python/fileutils.c
Py_ssize_t
_Py_write(int fd, const void *buf, size_t count)
{
    _Py_AssertHoldsTstate();                    // 确认当前线程持有 GIL
    assert(!PyErr_Occurred());                  // 调用前不应已有 Python 异常
    return _Py_write_impl(fd, buf, count, 1);   // 进入实际 write 包装，gil_held=1
}

static Py_ssize_t
_Py_write_impl(int fd, const void *buf, size_t count, int gil_held)
{
    Py_ssize_t n;                               // 保存 write 返回值
    int err;                                    // 保存 errno

    do {                                       // EINTR 时重试
        Py_BEGIN_ALLOW_THREADS                  // 释放 GIL
        errno = 0;                              // 清 errno，便于判断本次 syscall 结果
        n = write(fd, buf, count);              // 调用 libc write(fd, buf, count)
        err = errno;                            // 保存 errno
        Py_END_ALLOW_THREADS                    // 恢复 GIL
    } while (n < 0 && err == EINTR &&           // 如果被信号打断
             !PyErr_CheckSignals());            // 且 Python 信号处理器没有抛异常，则重试

    return n;                                   // 返回写入字节数或 -1
}
```

CPython 不理解串口，只把 `fd`、`buf`、`count` 交给 libc。

### 3.3 glibc `write()`

```c
// glibc-2.42/sysdeps/unix/sysv/linux/write.c
ssize_t
__libc_write (int fd, const void *buf, size_t nbytes)
{
  return SYSCALL_CANCEL (write, fd, buf, nbytes); // 发 write 系统调用
}
weak_alias (__libc_write, write)                  // write 是 __libc_write 的别名
```

树莓派 5 是 ARM64，系统调用参数按寄存器传入：

```text
x0 = fd
x1 = buf 用户态地址
x2 = count
x8 = __NR_write = 64
svc #0
```

---

## 4. Linux VFS 写路径

glibc 触发 `write` 系统调用后，数据正式进入 Linux 内核。VFS（虚拟文件系统）是内核所有文件操作的统一入口，负责权限检查、地址校验和分发。对 `/dev/ttyACM0` 来说，VFS 会把写请求交给 `tty_fops.write_iter`，也就是 `tty_write()`。本节解释 `write` 系统调用从入口到 TTY 层的 VFS 路径。

### 4.1 `write` 系统调用入口

`write` 系统调用是内核接收用户态写请求的第一个函数。它通过 fd 表把整数 fd 转换成 `struct file`，然后调用 `vfs_write()` 进入 VFS 通用写逻辑。TTY 设备是 stream（无偏移），所以 `ppos` 通常为 NULL，不需要维护文件位置。

```c
// linux/fs/read_write.c
SYSCALL_DEFINE3(write, unsigned int, fd, const char __user *, buf, size_t, count)
{
    return ksys_write(fd, buf, count);           // 进入通用 fd 写逻辑
}

ssize_t ksys_write(unsigned int fd, const char __user *buf, size_t count)
{
    struct fd f = fdget_pos(fd);                 // fd -> struct file，并处理文件位置锁
    ssize_t ret = -EBADF;                        // 默认错误：fd 无效

    if (fd_file(f)) {                            // 如果 fd 查到了有效 file
        loff_t pos, *ppos = file_ppos(fd_file(f)); // tty 是 stream，ppos 通常为 NULL
        if (ppos) {                              // 普通文件才需要维护偏移
            pos = *ppos;                         // 保存原始偏移
            ppos = &pos;                         // 使用临时偏移变量
        }
        ret = vfs_write(fd_file(f), buf, count, ppos); // 调 VFS 写
        if (ret >= 0 && ppos)                    // 普通文件写成功后
            fd_file(f)->f_pos = pos;             // 回写文件偏移
        fdput_pos(f);                            // 释放 fd 引用
    }

    return ret;                                  // 返回写入字节数或错误码
}
```

`__user` 表示 `buf` 是用户态地址。内核不能直接信任它，只能通过 `copy_from_user()` 或 `iov_iter` 这类机制复制。

### 4.2 `vfs_write()`

```c
// linux/fs/read_write.c
ssize_t vfs_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    ssize_t ret;                                 // 保存返回值

    if (!(file->f_mode & FMODE_WRITE))           // 文件不是以可写方式打开
        return -EBADF;                           // 返回 bad file descriptor
    if (!(file->f_mode & FMODE_CAN_WRITE))       // 文件类型不支持写
        return -EINVAL;                          // 返回 invalid argument
    if (unlikely(!access_ok(buf, count)))        // 检查用户态地址范围是否合法
        return -EFAULT;                          // 用户地址不可访问

    ret = rw_verify_area(WRITE, file, pos, count); // 检查写范围和权限
    if (ret)                                     // 如果检查失败
        return ret;                              // 返回错误
    if (count > MAX_RW_COUNT)                    // 限制单次读写最大长度
        count = MAX_RW_COUNT;                    // 截断到内核允许的最大值

    file_start_write(file);                      // 通知文件系统开始写
    if (file->f_op->write)                       // 如果 file_operations 有老式 write
        ret = file->f_op->write(file, buf, count, pos); // 调老式 write
    else if (file->f_op->write_iter)             // tty_fops 使用 write_iter
        ret = new_sync_write(file, buf, count, pos); // 包装成 kiocb/iov_iter 后调用
    else                                         // 两者都没有
        ret = -EINVAL;                           // 不支持写
    file_end_write(file);                        // 通知文件系统写结束
    return ret;                                  // 返回实际写入字节数
}
```

TTY 的 `file_operations` 是：

```c
// linux/drivers/tty/tty_io.c
static const struct file_operations tty_fops = {
    .write_iter = tty_write,                     // VFS write_iter 入口
    .read_iter  = tty_read,                      // VFS read_iter 入口
    .poll       = tty_poll,                      // select/poll 入口
    .unlocked_ioctl = tty_ioctl,                 // ioctl 入口
};
```

---

## 5. TTY 与 N_TTY 写路径

VFS 把写请求交给 `tty_fops.write_iter` 后，数据正式进入 TTY 子系统。TTY 核心是 Linux 的终端抽象层，负责把通用的文件写操作转换成具体终端设备能理解的写操作。N_TTY 是默认的行规程（line discipline），在串口原始模式下基本不做字节转换，只负责把数据交给底层驱动。本节解释 TTY 如何从 VFS 接收数据，并调用 CDC ACM 驱动的写方法。

### 5.1 `tty_write()`

`tty_write()` 是 `tty_fops.write_iter` 的实现入口。它的职责很简单：从 `struct file` 找到对应的 `tty_struct`，获取当前行规程（通常是 N_TTY），然后把 `iov_iter` 里的用户数据分块交给行规程处理。这里不做任何实际的数据拷贝，只是做设备查找和权限检查。

```c
// linux/drivers/tty/tty_io.c
static ssize_t tty_write(struct kiocb *iocb, struct iov_iter *from)
{
    return file_tty_write(iocb->ki_filp, iocb, from); // 取出 file 后转给 file_tty_write
}

static ssize_t file_tty_write(struct file *file, struct kiocb *iocb, struct iov_iter *from)
{
    struct tty_struct *tty = file_tty(file);      // 从 struct file 找到 tty_struct
    struct tty_ldisc *ld;                         // 当前 line discipline
    ssize_t ret;                                  // 返回值

    if (tty_paranoia_check(tty, file_inode(file), "tty_write")) // tty 状态检查
        return -EIO;                              // tty 异常
    if (!tty || !tty->ops->write || tty_io_error(tty)) // tty 不存在、驱动无 write、或 I/O 错误
        return -EIO;                              // 返回 I/O 错误

    ld = tty_ldisc_ref_wait(tty);                 // 获取当前 line discipline，通常是 N_TTY
    if (!ld)                                      // 如果 line discipline 已经挂起
        return hung_up_tty_write(iocb, from);     // 按 hangup 处理
    if (!ld->ops->write)                          // line discipline 没有 write
        ret = -EIO;                               // 返回错误
    else                                          // 正常情况
        ret = iterate_tty_write(ld, tty, file, from); // 把 iov_iter 分块交给 ldisc 写
    tty_ldisc_deref(ld);                          // 释放 line discipline 引用
    return ret;                                   // 返回写入字节数
}
```

`struct tty_ldisc` 是 TTY 行规程。pyserial 会把串口配置成原始模式，通常仍使用默认的 `N_TTY` 行规程，但关闭输出后处理。

### 5.2 `iterate_tty_write()`

`iterate_tty_write()` 是本次 `write()` 路径里第一次真正的数据复制位置。VFS 传下来的 `iov_iter` 仍然指向用户态缓冲，TTY core 不能把用户态指针直接交给底层驱动，所以先把用户缓冲里的协议帧复制到 `tty->write_buf` 这个内核临时缓冲，再调用行规程的 `ld->ops->write()`。

```c
// linux/drivers/tty/tty_io.c
static ssize_t iterate_tty_write(struct tty_ldisc *ld, struct tty_struct *tty,
                                 struct file *file, struct iov_iter *from)
{
    size_t chunk, count = iov_iter_count(from);      // 用户态本次要写的字节数
    ssize_t ret, written = 0;                        // 返回值和累计写入数

    ret = tty_write_lock(tty, file->f_flags & O_NDELAY); // 串行化同一 TTY 写入
    if (ret < 0)
        return ret;

    chunk = 2048;                                    // TTY 默认分块大小
    if (count < chunk)
        chunk = count;

    if (tty->write_cnt < chunk) {                    // 内核临时写缓冲不够大
        u8 *buf_chunk = kvmalloc(chunk, GFP_KERNEL | __GFP_RETRY_MAYFAIL);
        kvfree(tty->write_buf);
        tty->write_cnt = chunk;
        tty->write_buf = buf_chunk;
    }

    for (;;) {
        size_t size = min(chunk, count);

        // 第一次 CPU 参与的数据复制：
        // 从用户态 buf 指向的协议帧，复制到 TTY 的内核临时缓冲 tty->write_buf。
        if (copy_from_iter(tty->write_buf, size, from) != size)
            break;

        ret = ld->ops->write(tty, file, tty->write_buf, size); // 调 N_TTY 的 write
        if (ret <= 0)
            break;
        written += ret;
        count -= ret;
        if (!count)
            break;
    }

    tty_write_unlock(tty);                           // 释放 TTY 写锁
    return written ? written : ret;
}
```

这里的 `copy_from_iter()` 可以理解为面向 `iov_iter` 的用户态拷贝接口，本质作用等价于“从用户态 `buf` 把字节安全复制进内核”。对于本项目的舵机包，它复制的就是 14 字节 Sync Read 请求或 26 字节 Sync Write 请求。

### 5.3 `n_tty_write()`

`n_tty_write()` 是 N_TTY 行规程的写方法，也是 TTY 层真正处理数据的地方。对于 pyserial 配置的原始模式（`O_OPOST` 关闭），它不做换行转换等输出后处理，直接把字节原样交给底层驱动的 `tty->ops->write()`。如果驱动暂时没空间（`num == 0`），它会睡眠在 `write_wait` 上，等 `acm_write_bulk()` 完成回调唤醒。

```c
// linux/drivers/tty/n_tty.c
static ssize_t n_tty_write(struct tty_struct *tty, struct file *file,
                           const u8 *buf, size_t nr)
{
    const u8 *b = buf;                            // 当前要写的位置
    DEFINE_WAIT_FUNC(wait, woken_wake_function);  // 写等待队列节点
    ssize_t num, retval = 0;                      // num 是单次驱动写入数，retval 是错误码

    down_read(&tty->termios_rwsem);               // 读取 termios 配置期间加读锁
    process_echoes(tty);                          // 处理终端回显残留，串口原始模式通常无影响
    add_wait_queue(&tty->write_wait, &wait);      // 加入写等待队列

    while (1) {                                   // 循环直到所有字节写完或出错
        if (signal_pending(current)) {            // 有信号等待处理
            retval = -ERESTARTSYS;                // 标记可重启系统调用
            break;                                // 退出
        }
        if (tty_hung_up_p(file)) {                // tty 已挂起
            retval = -EIO;                        // I/O 错误
            break;                                // 退出
        }

        if (O_OPOST(tty)) {                       // 如果启用输出后处理
            num = process_output_block(tty, b, nr); // 可能转换换行等
        } else {                                  // pyserial 原始模式通常走这里
            struct n_tty_data *ldata = tty->disc_data; // N_TTY 私有数据

            while (nr > 0) {                      // 还有字节没交给驱动
                mutex_lock(&ldata->output_lock);  // 串行化输出
                num = tty->ops->write(tty, b, nr); // 调 CDC ACM 驱动 acm_tty_write()
                mutex_unlock(&ldata->output_lock);// 释放输出锁
                if (num < 0) {                    // 驱动返回错误
                    retval = num;                 // 保存错误
                    goto break_out;               // 跳到清理
                }
                if (!num)                         // 驱动暂时没有空间
                    break;                        // 去等待 write_wait
                b += num;                         // 前移用户数据指针
                nr -= num;                        // 减少剩余字节数
            }
        }

        if (!nr)                                  // 所有字节已经交给驱动
            break;                                // 写完成
        if (tty_io_nonblock(tty, file)) {         // 非阻塞模式且没写完
            retval = -EAGAIN;                     // 返回 EAGAIN
            break;                                // 退出
        }
        up_read(&tty->termios_rwsem);             // 等待前释放 termios 锁
        wait_woken(&wait, TASK_INTERRUPTIBLE, MAX_SCHEDULE_TIMEOUT); // 等待驱动唤醒
        down_read(&tty->termios_rwsem);           // 醒来后重新取锁
    }

break_out:
    remove_wait_queue(&tty->write_wait, &wait);   // 从写等待队列移除
    up_read(&tty->termios_rwsem);                 // 释放 termios 锁
    return (b - buf) ? b - buf : retval;          // 有写入则返回字节数，否则返回错误
}
```

在本项目中，协议帧只有 14 或 26 字节，远小于 CDC ACM 默认写缓冲，因此通常一次 `acm_tty_write()` 就能接受全部字节。

---

## 6. CDC ACM 驱动写路径

TTY 层把数据交给 `tty->ops->write()`，也就是 CDC ACM 驱动的 `acm_tty_write()`。这是从"软件 TTY 层"到"硬件 USB 层"的交界点。CDC ACM 驱动负责把 TTY 传来的字节打包成 USB URB，通过 USB core 提交给 xHCI 主机控制器。本节解释 CDC ACM 如何分配写缓冲、复制数据、构造 URB，直到 xHCI 接手。

### 6.1 `acm_tty_write()`

```c
// linux/drivers/usb/class/cdc-acm.c
static ssize_t acm_tty_write(struct tty_struct *tty, const u8 *buf,
                             size_t count)
{
    struct acm *acm = tty->driver_data;           // CDC ACM 私有设备结构
    int stat;                                     // 保存函数返回状态
    unsigned long flags;                          // 保存中断标志
    int wbn;                                      // write buffer 编号
    struct acm_wb *wb;                            // 本次写使用的 write buffer

    if (!count)                                   // 写入长度为 0
        return 0;                                 // 直接成功返回 0

    spin_lock_irqsave(&acm->write_lock, flags);   // 保护写缓冲分配
    wbn = acm_wb_alloc(acm);                      // 找一个空闲 acm_wb
    if (wbn < 0) {                                // 没有空闲写缓冲
        spin_unlock_irqrestore(&acm->write_lock, flags); // 释放锁
        return 0;                                 // 返回 0，TTY 上层会等待 write_wait
    }
    // wb = write buffer，CDC ACM 驱动的写缓冲描述符（struct acm_wb）。
    // 每个字段的含义：
    //   buf      - DMA 安全的数据缓冲区（usb_alloc_coherent 分配）
    //   dmah     - buf 的 DMA 总线地址，USB 控制器做 DMA 时用
    //   len      - 本次要发送的字节数
    //   urb      - 挂在这个写缓冲上的 USB Request Block
    //   instance - 指向所属 acm 设备
    //   use      - 标记该写缓冲是否正在被使用
    wb = &acm->wb[wbn];

    if (!acm->dev) {                              // USB 设备不存在
        wb->use = false;                          // 释放写缓冲
        spin_unlock_irqrestore(&acm->write_lock, flags); // 释放锁
        return -ENODEV;                           // 设备不存在
    }

    count = (count > acm->writesize) ? acm->writesize : count; // 不超过单个写缓冲大小
    // buf 来自 TTY 层的内核临时缓冲（tty->write_buf），是普通内核内存。
    // wb->buf 是 CDC ACM 驱动预先分配的 DMA 安全内存（usb_alloc_coherent），
    // 物理连续且可被 USB 控制器直接做 DMA。必须把数据从普通内存复制到 DMA 安全区，
    // URB 才能直接用这块内存做 bulk OUT 传输。
    memcpy(wb->buf, buf, count);
    wb->len = count;                              // 记录本次要发的字节数

    stat = usb_autopm_get_interface_async(acm->control); // 增加 USB 自动电源管理引用
    if (stat) {                                   // 如果电源管理失败
        wb->use = false;                          // 释放写缓冲
        spin_unlock_irqrestore(&acm->write_lock, flags); // 释放锁
        return stat;                              // 返回错误
    }

    if (acm->susp_count) {                        // 如果设备正在挂起
        usb_anchor_urb(wb->urb, &acm->delayed);   // 把 URB 放到延迟队列
        spin_unlock_irqrestore(&acm->write_lock, flags); // 释放锁
        return count;                             // 上层看作已接收
    }

    // acm_start_wb 把 wb 包装成 USB URB 提交给 USB core：
    //   - wb->urb->transfer_buffer = wb->buf    （URB 数据指针指向 DMA 安全缓冲）
    //   - wb->urb->transfer_dma    = wb->dmah   （DMA 映射地址，USB 控制器用）
    //   - wb->urb->transfer_buffer_length = wb->len  （本次传输字节数）
    //   - wb->urb->dev = acm->dev               （目标 USB 设备）
    // 最后调用 usb_submit_urb(wb->urb, GFP_ATOMIC) 提交给 USB core。
    // GFP_ATOMIC 是因为当前持有 acm->write_lock 自旋锁，不能睡眠。
    stat = acm_start_wb(acm, wb);
    spin_unlock_irqrestore(&acm->write_lock, flags); // 释放写锁

    if (stat < 0)                                 // 提交失败
        return stat;                              // 返回错误
    return count;                                 // 成功时返回本次接收的字节数
}
```

关键结构：

| 结构            | 本链路中的作用                              |
| --------------- | ------------------------------------------- |
| `struct acm`    | 一个 CDC ACM 设备实例，连接 TTY 层和 USB 层 |
| `struct acm_wb` | 写缓冲，包含 `buf`、`len`、`urb`、`use` 等  |
| `struct urb`    | USB Request Block，一次 USB 传输请求        |

### 6.2 `acm_start_wb()`

`acm_start_wb()` 把 `acm_wb` 里的数据真正提交给 USB core。它填写 URB 的 `transfer_buffer`（数据地址）、`transfer_dma`（DMA 映射地址）、`transfer_buffer_length`（数据长度）和 `dev`（USB 设备），然后调用 `usb_submit_urb()`。从这里开始，数据离开内核软件缓冲区，进入 USB 主机控制器的硬件调度队列。

```c
// linux/drivers/usb/class/cdc-acm.c
static int acm_start_wb(struct acm *acm, struct acm_wb *wb)
{
    int rc;                                       // usb_submit_urb 返回值

    acm->transmitting++;                         // 记录正在传输的写请求数

    wb->urb->transfer_buffer = wb->buf;          // URB 数据缓冲指向 wb->buf
    wb->urb->transfer_dma = wb->dmah;            // DMA 地址，给 USB 控制器使用
    wb->urb->transfer_buffer_length = wb->len;   // 本次 USB bulk OUT 传输长度
    wb->urb->dev = acm->dev;                     // USB 设备对象

    rc = usb_submit_urb(wb->urb, GFP_ATOMIC);    // 提交 URB 给 USB core
    if (rc < 0) {                                // 如果提交失败
        acm_write_done(acm, wb);                 // 回滚，释放写缓冲
    }
    return rc;                                   // 0 表示成功提交
}
```

此时，Feetech 协议帧已经在 `urb->transfer_buffer` 里。对于 14 字节 Sync Read 请求，`transfer_buffer_length=14`；对于 26 字节 Sync Write 请求，`transfer_buffer_length=26`。

CDC ACM 在创建写 URB 时已经设置了 `URB_NO_TRANSFER_DMA_MAP`，因为 `wb->buf` 来自 `usb_alloc_coherent()`，同时 `wb->dmah` 已经保存了 DMA 地址。因此后面的 USB core 虽然仍会走到 `map_urb_for_dma()`，但这一步不会再把协议帧复制到另一块普通 DMA 缓冲。

### 6.3 USB core 与 xHCI

`usb_submit_urb()` 是 USB core 的入口。USB core 负责校验 URB 合法性、处理 DMA 访问关系，然后把 URB 交给具体的主机控制器驱动（HCD）。树莓派 5 的 USB 主控走 xHCI，所以最终进入 `xhci_urb_enqueue()`，由 xHCI 驱动把 URB 转成硬件可执行的 Transfer Request Block（TRB）。这里的 TRB 是 xHCI 规范里的传输请求描述符，多个 TRB 串在一起组成 transfer ring。

```c
// linux/drivers/usb/core/urb.c
int usb_submit_urb(struct urb *urb, gfp_t mem_flags)
{
    struct usb_device *dev;                       // USB 设备
    struct usb_host_endpoint *ep;                 // USB endpoint

    if (!urb || !urb->complete)                   // URB 必须存在且有完成回调
        return -EINVAL;                           // 参数错误
    if (urb->hcpriv)                              // URB 已经提交过还没完成
        return -EBUSY;                            // 忙

    dev = urb->dev;                               // 取 USB 设备
    if ((!dev) || (dev->state < USB_STATE_UNAUTHENTICATED)) // 设备无效
        return -ENODEV;                           // 设备不存在

    ep = usb_pipe_endpoint(dev, urb->pipe);       // 从 pipe 找 endpoint
    if (!ep)                                      // endpoint 不存在
        return -ENOENT;                           // 返回不存在

    urb->ep = ep;                                 // 记录 endpoint
    urb->status = -EINPROGRESS;                   // 标记传输进行中
    urb->actual_length = 0;                       // 完成前实际传输长度为 0

    return usb_hcd_submit_urb(urb, mem_flags);    // 交给 Host Controller Driver
}
```

`usb_hcd_submit_urb()` 是 USB core 和主机控制器驱动（HCD）之间的桥梁。它先增加 URB 引用计数防止过早释放，然后调用 `map_urb_for_dma()` 处理 DMA 访问关系，最后把 URB 交给具体 HCD 的 `urb_enqueue`。对树莓派 5 来说，这个 HCD 就是 xHCI。注意，对 CDC ACM 写 URB 来说，数据缓冲已设置 `URB_NO_TRANSFER_DMA_MAP`，因此这里不是第三次协议帧复制。

```c
// linux/drivers/usb/core/hcd.c
int usb_hcd_submit_urb(struct urb *urb, gfp_t mem_flags)
{
    int status;                                   // 提交状态
    struct usb_hcd *hcd = bus_to_hcd(urb->dev->bus); // 找到 USB 主机控制器

    usb_get_urb(urb);                             // 增加 URB 引用计数
    atomic_inc(&urb->use_count);                  // 标记 URB 正在使用
    atomic_inc(&urb->dev->urbnum);                // 设备活跃 URB 数加一

    status = map_urb_for_dma(hcd, urb, mem_flags); // 处理 DMA 访问关系
    if (likely(status == 0)) {                    // DMA 访问关系处理成功
        status = hcd->driver->urb_enqueue(hcd, urb, mem_flags); // 调 xHCI enqueue
        if (unlikely(status))                     // 如果 HCD 拒绝
            unmap_urb_for_dma(hcd, urb);          // 回滚 DMA 映射
    }

    return status;                                // 0 表示已经排入 HCD
}
```

`xhci_urb_enqueue()` 是 xHCI 驱动的入口。它为 URB 分配 `urb_priv` 跟踪结构，按 endpoint 类型分发到不同的传输队列。CDC ACM 的数据端点是 bulk，所以进入 `xhci_queue_bulk_tx()`，由它把 URB 数据转换成 xHCI 硬件能理解的 Transfer Request Block（TRB）。

```c
// linux/drivers/usb/host/xhci.c
static int xhci_urb_enqueue(struct usb_hcd *hcd, struct urb *urb, gfp_t mem_flags)
{
    struct xhci_hcd *xhci = hcd_to_xhci(hcd);     // 从通用 HCD 取 xHCI 私有结构
    unsigned int slot_id, ep_index;               // slot 是 USB 设备，ep_index 是 endpoint
    struct urb_priv *urb_priv;                    // xHCI 给 URB 分配的私有跟踪结构
    int num_tds;                                  // Transfer Descriptor 数量

    ep_index = xhci_get_endpoint_index(&urb->ep->desc); // 计算 endpoint 索引
    num_tds = 1;                                  // 普通 bulk 包通常一个 TD
    urb_priv = kzalloc(struct_size(urb_priv, td, num_tds), mem_flags); // 分配私有结构
    if (!urb_priv)                                // 内存不足
        return -ENOMEM;                           // 返回错误

    urb->hcpriv = urb_priv;                       // 把 xHCI 私有结构挂到 URB
    slot_id = urb->dev->slot_id;                  // USB 设备 slot id

    switch (usb_endpoint_type(&urb->ep->desc)) {  // 按 endpoint 类型排队
    case USB_ENDPOINT_XFER_BULK:                  // CDC ACM 数据端点是 bulk
        return xhci_queue_bulk_tx(xhci, GFP_ATOMIC, urb, slot_id, ep_index); // 写入 xHCI 传输环
    }
}
```

`xhci_queue_bulk_tx()` 会把 URB 数据描述成 TRB，并把这些 TRB 放入 endpoint 对应的 xHCI transfer ring。对普通 bulk OUT 来说，它在循环中计算本段数据的 DMA 地址、长度字段和控制字段，然后调用 `queue_trb()` 写入当前 ring 槽位。

```c
// linux/drivers/usb/host/xhci-ring.c
int xhci_queue_bulk_tx(struct xhci_hcd *xhci, gfp_t mem_flags,
        struct urb *urb, int slot_id, unsigned int ep_index)
{
    ...
    addr = (u64) urb->transfer_dma;              // URB 数据缓冲的 DMA 地址
    ...
    length_field = TRB_LEN(trb_buff_len) |       // 本段传输长度
        TRB_TD_SIZE(remainder) |                 // TD 剩余大小提示
        TRB_INTR_TARGET(0);                      // 完成事件送到 interrupter 0

    queue_trb(xhci, ring, more_trbs_coming | need_zero_pkt,
            lower_32_bits(send_addr),            // TRB field[0]：buffer DMA 地址低 32 位
            upper_32_bits(send_addr),            // TRB field[1]：buffer DMA 地址高 32 位
            length_field,                        // TRB field[2]：长度和中断目标
            field);                              // TRB field[3]：类型、cycle、IOC、CHAIN 等
    ...
    giveback_first_trb(xhci, slot_id, ep_index, urb->stream_id,
            start_cycle, start_trb);             // 所有 TRB 写完后，把首个 TRB 交给硬件并敲 doorbell
    return 0;                                    // xHCI 排队完成，沿 write 调用栈返回
}
```

`queue_trb()` 是真正把 TRB 写进 transfer ring 的函数。`ring->enqueue` 指向当前可写的 TRB 槽位，CPU 依次写入四个 32-bit 字段；最后调用 `inc_enq()` 推进 ring 的 enqueue 指针。

```c
// linux/drivers/usb/host/xhci-ring.c
static void queue_trb(struct xhci_hcd *xhci, struct xhci_ring *ring,
        bool more_trbs_coming,
        u32 field1, u32 field2, u32 field3, u32 field4)
{
    struct xhci_generic_trb *trb;

    trb = &ring->enqueue->generic;               // 当前 transfer ring 的写入槽位
    trb->field[0] = cpu_to_le32(field1);         // 数据 buffer DMA 地址低 32 位
    trb->field[1] = cpu_to_le32(field2);         // 数据 buffer DMA 地址高 32 位
    trb->field[2] = cpu_to_le32(field3);         // 长度、TD size、interrupter
    /* make sure TRB is fully written before giving it to the controller */
    wmb();                                      // 保证前三个字段先写完
    trb->field[3] = cpu_to_le32(field4);         // 类型、cycle bit、IOC、CHAIN 等控制位

    trace_xhci_queue_trb(ring, trb);
    inc_enq(xhci, ring, more_trbs_coming);       // 推进 enqueue 指针
}
```

TRB 入环本身还不等于硬件已经开始取任务。`xhci_queue_bulk_tx()` 会在所有 TRB 都写完后调用 `giveback_first_trb()`：这个函数先通过首个 TRB 的 cycle bit 把整串 TRB 一次性交给硬件，再写 doorbell 通知 xHCI 控制器开始处理这个 endpoint 的 transfer ring。

```c
// linux/drivers/usb/host/xhci-ring.c
static void giveback_first_trb(struct xhci_hcd *xhci, int slot_id,
        unsigned int ep_index, unsigned int stream_id, int start_cycle,
        struct xhci_generic_trb *start_trb)
{
    /*
     * Pass all the TRBs to the hardware at once and make sure this write
     * isn't reordered.
     */
    wmb();                                      // 保证前面写入的 TRB 先对硬件可见
    if (start_cycle)
        start_trb->field[3] |= cpu_to_le32(start_cycle); // 交出第一个 TRB 的 cycle bit
    else
        start_trb->field[3] &= cpu_to_le32(~TRB_CYCLE);
    xhci_ring_ep_doorbell(xhci, slot_id, ep_index, stream_id); // 通知硬件取 ring
}
```

真正的“敲门铃”发生在 `xhci_ring_ep_doorbell()`。它先定位当前 USB 设备 slot 对应的 doorbell 寄存器，再把 endpoint 编号和 stream id 编成 doorbell value 写进去。这个 `writel()` 是通知 xHCI 硬件的关键动作：告诉主控“这个 slot 的这个 endpoint 有新的 TRB 可以处理”。

```c
// linux/drivers/usb/host/xhci-ring.c
void xhci_ring_ep_doorbell(struct xhci_hcd *xhci,
        unsigned int slot_id,
        unsigned int ep_index,
        unsigned int stream_id)
{
    __le32 __iomem *db_addr = &xhci->dba->doorbell[slot_id]; // 当前设备 slot 的 doorbell

    readl(db_addr);                              // Pi 4/5 这类非一致性 DMA 平台上做一次序列化
    writel(DB_VALUE(ep_index, stream_id), db_addr); // 写 doorbell 寄存器，通知 xHCI 硬件
    readl(db_addr);                              // flush doorbell write
}
```

```c
// linux/drivers/usb/host/xhci.h
struct xhci_doorbell_array {
    __le32 doorbell[256];                        // doorbell[0] 是命令环，doorbell[slot_id] 是设备 endpoint
};

#define DB_VALUE(ep, stream) ((((ep) + 1) & 0xff) | ((stream) << 16))
```

因此，xHCI 写链路更精确的最后几步是：

```text
xhci_queue_bulk_tx()
  -> queue_trb()                      # 把 URB buffer 地址、长度、标志写成 TRB
  -> giveback_first_trb()             # 通过 cycle bit 把首个 TRB 交给硬件
  -> xhci_ring_ep_doorbell()          # 写 doorbell[slot_id]
  -> xHCI hardware DMA 读取 TRB 和数据缓冲，发出 USB bulk OUT
```

到这里，内核已经把串口协议帧交给 USB 主机控制器，后续由 xHCI 硬件通过 DMA 读取数据缓冲并完成 USB 包发送。

---

## 7. 写完成回调

数据提交给 xHCI 后，`writePort()` 在 `acm_tty_write()` 返回时就已经结束了，它不会等待 USB 传输真正完成。当 xHCI 把 bulk OUT 数据发送出去后，USB 主控会产生完成中断，触发 `acm_write_bulk()` 回调。这个回调的职责是：释放写缓冲、标记传输完成、唤醒可能阻塞在 `write_wait` 上的进程。本节解释完成回调如何打扫战场。

```c
// linux/drivers/usb/class/cdc-acm.c
static void acm_write_bulk(struct urb *urb)
{
    struct acm_wb *wb = urb->context;             // 完成的是哪一个写缓冲
    struct acm *acm = wb->instance;               // 所属 CDC ACM 设备
    unsigned long flags;                          // 保存中断标志

    spin_lock_irqsave(&acm->write_lock, flags);   // 保护写缓冲状态
    acm_write_done(acm, wb);                      // 标记写缓冲空闲，减少 transmitting
    spin_unlock_irqrestore(&acm->write_lock, flags); // 释放锁

    set_bit(EVENT_TTY_WAKEUP, &acm->flags);       // 通知 TTY 写侧可以继续写
    schedule_delayed_work(&acm->dwork, 0);        // 调度 workqueue 唤醒等待者
}
```

`writePort()` 不会等到这个回调才返回。它通常在 `acm_tty_write()` 成功提交 URB 后就返回 `count`。后续如果上层再次调用 `clearPort()`，才会通过 `tcdrain()` 等待这些未完成的写 URB 排空。

---

## 8. writePort 究竟完成了什么

`writePort(packet)` 完成的是：

1. 把 Python 的协议帧字节复制到内核。
2. 通过 TTY/N_TTY 把字节交给 CDC ACM 驱动。
3. CDC ACM 分配 `acm_wb`，复制到 USB 写缓冲。
4. 构造并提交 USB bulk OUT URB。
5. xHCI 把 URB 转成硬件可执行的 bulk transfer。

从 `write()` 系统调用进入内核开始，本次路径里 CPU 参与的数据复制主要有两次：

| 次数 | 源地址 | 目标地址 | 代码位置 | 作用 |
| --- | --- | --- | --- | --- |
| 1 | 用户态 `buf`，也就是 pyserial 传给 `write()` 的协议帧 | TTY core 的 `tty->write_buf` | `copy_from_iter(tty->write_buf, size, from)` | 把用户态字节安全搬进内核临时缓冲 |
| 2 | N_TTY 传下来的 `buf`，实际指向 `tty->write_buf` | CDC ACM 的 `wb->buf` DMA 安全写缓冲 | `memcpy(wb->buf, buf, count)` | 把协议帧放进 USB bulk OUT URB 使用的数据缓冲 |

`map_urb_for_dma()` 和 xHCI 入队主要是在建立 DMA 访问关系、写 TRB 和敲 doorbell，不是再把 14 或 26 字节协议帧做一次普通 CPU `memcpy()`。随后真正把数据送到 USB 总线的是 xHCI 硬件通过 DMA 读取 `wb->buf`。

最终系统调用参数是：

```text
write(fd=/dev/ttyACM0, buf=<用户态协议帧地址>, count=14 或 26)
```

内核完成串口发送的关键对象是：

```text
struct file -> struct tty_struct -> struct tty_ldisc(N_TTY) -> struct tty_operations(acm_ops) -> struct urb
```
