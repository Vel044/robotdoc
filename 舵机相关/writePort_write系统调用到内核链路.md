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
  │  `vfs_write()` -> `new_sync_write()` 分发到 tty file_operations
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

来源：scservo_sdk/port_handler.py:writePort（节选：仅保留本链路相关分支，已按当前仓库源码核对）
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

### 2.2 内核内调用栈（拆分版）

下面把内核路径拆成 4 段。每段最后一行就是下一段的入口，这样既能保持完整链路，又不会把一整条栈压成一根很长的竖线。

#### 2.2.1 syscall 到 TTY 入口

```text
Linux SYSCALL_DEFINE3(write, fd, buf, count)
│  内核 syscall 入口：接收用户态 fd、用户态 buf 指针和 count。
└── ksys_write(fd, buf, count)
    │  fdget_pos() + file_ppos()：把 int fd 转成 struct file，并处理文件偏移语义。
    └── vfs_write(file, buf, count, pos)
        │  access_ok() + rw_verify_area()：检查用户地址、写权限和写入范围。
        │  file_start_write()/file_end_write()：普通文件 freeze 保护；/dev/ttyACM0 是字符设备，实际直接返回。
        └── new_sync_write(file, buf, count, pos)
            │  init_sync_kiocb() + iov_iter_ubuf()：把用户缓冲包装成 kiocb + iov_iter。
            └── filp->f_op->write_iter(iocb, iter)
                │  /dev/ttyACM0 对应 tty_fops.write_iter。
                └── tty_write(iocb, iter)
                    │  TTY file_operations 写入口，下一段继续展开。
```

#### 2.2.2 TTY/N_TTY 到 CDC ACM 入口

```text
tty_write(iocb, iter)
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
                        │  CDC ACM 写入口，下一段继续展开。
```

#### 2.2.3 CDC ACM 到 USB core

```text
acm_tty_write(tty, buf, count)
│  acm_wb_alloc()：取空闲写缓冲。
│  memcpy()：把协议帧复制到 CDC ACM 的 USB 写缓冲。
│  usb_autopm_get_interface_async()：增加 USB runtime PM 引用。
└── acm_start_wb(acm, wb)
    │  填 URB 的 transfer_buffer、DMA 地址、transfer_buffer_length 和 USB 设备。
    └── usb_submit_urb(wb->urb, GFP_ATOMIC)
        │  USB core：校验 URB、endpoint、方向和长度，标记传输进行中。
        └── usb_hcd_submit_urb(urb, GFP_ATOMIC)
            │  HCD 提交入口，下一段继续展开。
```

#### 2.2.4 USB core 到 xHCI doorbell

```text
usb_hcd_submit_urb(urb, GFP_ATOMIC)
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

来源：serial/serialposix.py:write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
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

来源：cpython/Modules/posixmodule.c:os_write_impl / cpython/Python/fileutils.c:_Py_write,_Py_write_impl（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// cpython/Modules/posixmodule.c
static Py_ssize_t  // 本行参与当前 C 层路径的控制流或数据准备
os_write_impl(PyObject *module, int fd, Py_buffer *data)  // 调用下一层 C 函数继续完成当前路径
{
    return _Py_write(fd, data->buf, data->len);  // fd 原样传入，data->buf 是 bytes 数据地址
}

// cpython/Python/fileutils.c
Py_ssize_t  // 本行参与当前 C 层路径的控制流或数据准备
_Py_write(int fd, const void *buf, size_t count)  // 调用下一层 C 函数继续完成当前路径
{
    _Py_AssertHoldsTstate();                    // 确认当前线程持有 GIL
    assert(!PyErr_Occurred());                  // 调用前不应已有 Python 异常
    return _Py_write_impl(fd, buf, count, 1);   // 进入实际 write 包装，gil_held=1
}

static Py_ssize_t  // 本行参与当前 C 层路径的控制流或数据准备
_Py_write_impl(int fd, const void *buf, size_t count, int gil_held)  // 调用下一层 C 函数继续完成当前路径
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

来源：glibc-2.42/sysdeps/unix/sysv/linux/write.c:__libc_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// glibc-2.42/sysdeps/unix/sysv/linux/write.c
ssize_t  // 本行参与当前 C 层路径的控制流或数据准备
__libc_write (int fd, const void *buf, size_t nbytes)  // 调用下一层 C 函数继续完成当前路径
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

来源：linux/fs/read_write.c:SYSCALL_DEFINE write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/fs/read_write.c
SYSCALL_DEFINE3(write, unsigned int, fd, const char __user *, buf, size_t, count)  // 定义 Linux 系统调用入口，用户态 syscall 会进入这里
{
    return ksys_write(fd, buf, count);           // 进入通用 fd 写逻辑
}

ssize_t ksys_write(unsigned int fd, const char __user *buf, size_t count)  // 定义当前层的 C 函数入口
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

#### 4.1.1 `fdget_pos()` / `file_ppos()` / `fdput_pos()`

这三个函数是 syscall 入口里的 fd 管理辅助逻辑，不搬运舵机协议字节。`fdget_pos()` 把整数 fd 换成 `struct file`，必要时锁住文件偏移；`file_ppos()` 判断这个 file 是否需要文件偏移；`fdput_pos()` 在返回前释放 fd 引用和可能拿到的位置锁。对 `/dev/ttyACM0` 来说，TTY 是 `FMODE_STREAM`，所以 `file_ppos()` 返回 `NULL`，后面的 `pos` 分支不会参与串口数据路径。

来源：linux/fs/file.c:fdget_pos / linux/fs/read_write.c:file_ppos / linux/include/linux/file.h:fdput_pos（节选：保留 fd 引用和 stream 偏移分支，已按当前仓库源码核对）
```c
// linux/fs/file.c
struct fd fdget_pos(unsigned int fd)              // 把用户传入的整数 fd 转成内核 struct fd
{
    struct fd f = fdget(fd);                      // 从当前进程 fd 表取 struct file 引用
    struct file *file = fd_file(f);               // 从 struct fd 中取出 struct file 指针

    if (file && file_needs_f_pos_lock(file)) {    // 共享普通文件位置时才需要锁 f_pos
        f.word |= FDPUT_POS_UNLOCK;               // 标记 fdput_pos() 返回时需要解锁
        mutex_lock(&file->f_pos_lock);            // 锁住普通文件的当前位置
    }
    return f;                                     // 返回带引用的 fd 包装
}

// linux/fs/read_write.c
static inline loff_t *file_ppos(struct file *file) // 判断本次 I/O 是否需要文件偏移
{
    return file->f_mode & FMODE_STREAM ? NULL : &file->f_pos; // TTY 是 stream，返回 NULL
}

// linux/include/linux/file.h
static inline void fdput_pos(struct fd f)         // 释放 fdget_pos() 获取的资源
{
    if (f.word & FDPUT_POS_UNLOCK)                // 如果前面锁了普通文件 f_pos
        __f_unlock_pos(fd_file(f));               // 解锁文件位置
    fdput(f);                                     // 释放 fd 引用
}
```

### 4.2 `vfs_write()`

来源：linux/fs/read_write.c:vfs_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/fs/read_write.c
ssize_t vfs_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)  // 定义当前层的 C 函数入口
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

    file_start_write(file);                      // 普通文件获取 freeze 写保护；/dev/ttyACM0 是字符设备，这里实际直接返回
    if (file->f_op->write)                       // 如果 file_operations 有老式 write
        ret = file->f_op->write(file, buf, count, pos); // 调老式 write
    else if (file->f_op->write_iter)             // tty_fops 使用 write_iter
        ret = new_sync_write(file, buf, count, pos); // 包装成 kiocb/iov_iter 后调用
    else                                         // 两者都没有
        ret = -EINVAL;                           // 不支持写
    if (ret > 0) {                               // 写入成功后
        fsnotify_modify(file);                   // 通知文件被修改
        add_wchar(current, ret);                 // 统计当前任务写入的字节数
    }
    inc_syscw(current);                          // 统计当前任务 write 系统调用次数
    file_end_write(file);                        // 普通文件释放 freeze 写保护；/dev/ttyACM0 是字符设备，这里实际直接返回
    return ret;                                  // 返回实际写入字节数
}
```

### 4.3 VFS 辅助检查、通知和统计

`access_ok()`、`rw_verify_area()`、`fsnotify_modify()`、`add_wchar()` 和 `inc_syscw()` 都是 VFS 通用框架的辅助动作，不是 TTY/USB 数据分发点。它们分别负责用户地址范围检查、读写权限/LSM 检查、文件修改通知和任务 I/O 统计。

来源：linux/arch/arm64/include/asm/uaccess.h:access_ok / linux/fs/read_write.c:rw_verify_area（节选：保留 write 链路相关检查，已按当前仓库源码核对）
```c
// linux/arch/arm64/include/asm/uaccess.h
static inline int access_ok(const void __user *addr, unsigned long size) // 检查用户态指针范围是否可能合法
{
    if (IS_ENABLED(CONFIG_ARM64_TAGGED_ADDR_ABI) && // ARM64 tagged address ABI 开启时
        (current->flags & PF_KTHREAD || test_thread_flag(TIF_TAGGED_ADDR))) // 需要处理 tagged 用户地址
        addr = untagged_addr(addr);                // 去掉地址 tag 后再检查范围

    return likely(__access_ok(addr, size));        // 判断 addr+size 是否落在用户态地址空间
}

// linux/fs/read_write.c
int rw_verify_area(int read_write, struct file *file, const loff_t *ppos, size_t count) // 检查读写范围和权限
{
    int mask = read_write == READ ? MAY_READ : MAY_WRITE; // write 链路使用 MAY_WRITE
    int ret;                                      // 保存权限检查结果

    if (unlikely((ssize_t) count < 0))            // count 被解释成负数时非法
        return -EINVAL;                           // 返回参数错误

    if (ppos) {                                   // 普通文件才检查偏移范围
        loff_t pos = *ppos;                       // 读取当前位置
        if (unlikely(pos < 0)) {                  // 负偏移需要额外判断
            if (!unsigned_offsets(file))          // 文件不允许无符号偏移
                return -EINVAL;                   // 返回参数错误
            if (count >= -pos)                    // 写入长度会越过可表示范围
                return -EOVERFLOW;                // 返回溢出
        } else if (unlikely((loff_t) (pos + count) < 0)) { // 正偏移加 count 后溢出
            if (!unsigned_offsets(file))          // 文件不允许无符号偏移
                return -EINVAL;                   // 返回参数错误
        }
    }

    ret = security_file_permission(file, mask);   // LSM/security 层检查是否允许写
    if (ret)                                      // 权限检查失败
        return ret;                               // 直接返回错误

    return fsnotify_file_area_perm(file, mask, ppos, count); // 通知/检查文件区域权限
}
```

来源：linux/include/linux/fsnotify.h:fsnotify_modify / linux/include/linux/sched/xacct.h:add_wchar,inc_syscw（节选：保留 write 统计相关分支，已按当前仓库源码核对）
```c
// linux/include/linux/fsnotify.h
static inline void fsnotify_modify(struct file *file) // VFS 写成功后的修改通知
{
    fsnotify_file(file, FS_MODIFY);               // 通知 inotify/fanotify 等观察者文件被修改
}

// linux/include/linux/sched/xacct.h
static inline void add_wchar(struct task_struct *tsk, ssize_t amt) // 统计当前任务写入的字符数
{
    tsk->ioac.wchar += amt;                       // 增加 task extended accounting 的写字节计数
}

static inline void inc_syscw(struct task_struct *tsk) // 统计当前任务 write 类系统调用次数
{
    tsk->ioac.syscw++;                            // write syscall 计数加一
}
```

### 4.4 `file_start_write()` / `file_end_write()`

`file_start_write()` 是 VFS 通用写路径里的文件系统 freeze 保护。它只对普通文件生效：如果目标是 regular file，就进入 `sb_start_write()`，增加 superblock 的 writer 计数，防止文件系统 freeze 和当前写入并发冲突。

但本链路写的是 `/dev/ttyACM0`，它是字符设备，不是普通文件。因此 `file_start_write()` 和后面的 `file_end_write()` 在这里都会因为 `!S_ISREG(...)` 直接返回，不会进入 `sb_start_write()`，也不会影响后面的 TTY/USB 数据提交链路。

来源：linux/include/linux/fs.h:file_start_write,file_end_write,sb_start_write（节选：保留本链路判断和普通文件分支，已按当前仓库源码核对）
```c
// linux/include/linux/fs.h
static inline void file_start_write(struct file *file) // VFS 写路径开始时调用的 freeze 写保护入口
{
    if (!S_ISREG(file_inode(file)->i_mode))       // /dev/ttyACM0 是字符设备，不是 regular file
        return;                                   // 串口链路在这里直接返回，不拿 superblock freeze 锁
    sb_start_write(file_inode(file)->i_sb);       // 普通文件才增加 superblock writer 计数
}

static inline void file_end_write(struct file *file) // 和 file_start_write() 配对的结束入口
{
    if (!S_ISREG(file_inode(file)->i_mode))       // 字符设备同样不参与 regular file freeze 保护
        return;                                   // 串口链路在这里直接返回
    sb_end_write(file_inode(file)->i_sb);         // 普通文件才释放 superblock writer 计数
}

static inline void sb_start_write(struct super_block *sb) // 普通文件写入时的 superblock freeze 保护
{
    __sb_start_write(sb, SB_FREEZE_WRITE);        // 增加当前 superblock 的写者计数，和 freeze 流程互斥
}
```

所以，对 `writePort()` 来说，这一对函数是“VFS 通用框架经过的保护点”，不是协议帧复制点，也不是进入 TTY 的关键跳转点。真正分发到 TTY 是下一行的 `new_sync_write()`。

### 4.5 `new_sync_write()`

`new_sync_write()` 是 `vfs_write()` 和 `tty_fops.write_iter` 之间容易漏掉的一层。它不再重新解释串口协议，而是把用户态 `buf/count` 包装成同步 `kiocb` 和 `iov_iter`，然后调用 `/dev/ttyACM0` 的 `write_iter`，也就是 `tty_write()`。

来源：linux/fs/read_write.c:new_sync_write（已按当前仓库源码核对）
```c
// linux/fs/read_write.c
static ssize_t new_sync_write(struct file *filp, const char __user *buf, size_t len, loff_t *ppos) // 把 VFS write 请求包装成同步 write_iter 调用
{
    struct kiocb kiocb;                          // 同步 I/O 控制块，携带 file 和当前位置
    struct iov_iter iter;                        // 描述用户态源缓冲的迭代器
    ssize_t ret;                                 // 保存 write_iter 返回值

    init_sync_kiocb(&kiocb, filp);               // 初始化同步 kiocb，并绑定当前 struct file
    kiocb.ki_pos = (ppos ? *ppos : 0);           // 普通文件使用当前位置；TTY stream 的 ppos 通常为 NULL
    iov_iter_ubuf(&iter, ITER_SOURCE, (void __user *)buf, len); // 把用户态写缓冲包装成 ITER_SOURCE

    ret = filp->f_op->write_iter(&kiocb, &iter); // 调用 tty_fops.write_iter，也就是 tty_write()
    BUG_ON(ret == -EIOCBQUEUED);                 // 同步路径不允许返回异步排队状态
    if (ret > 0 && ppos)                         // 普通文件成功写入后才需要更新偏移
        *ppos = kiocb.ki_pos;                    // 把 write_iter 更新后的偏移回写给调用方
    return ret;                                  // 返回写入字节数或错误码
}
```

TTY 的 `file_operations` 是：

来源：linux/drivers/tty/tty_io.c:相关代码片段（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
static const struct file_operations tty_fops = {  // 调用 TTY 层接口处理串口设备语义
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

来源：linux/drivers/tty/tty_io.c:tty_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
static ssize_t tty_write(struct kiocb *iocb, struct iov_iter *from)  // 定义当前层的 C 函数入口
{
    return file_tty_write(iocb->ki_filp, iocb, from); // 取出 file 后转给 file_tty_write
}

static ssize_t file_tty_write(struct file *file, struct kiocb *iocb, struct iov_iter *from)  // 定义当前层的 C 函数入口
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

#### 5.1.1 `tty_ldisc_ref_wait()`

`tty_ldisc_ref_wait()` 是 TTY 层保护行规程生命周期的等待/引用函数。它保证当前线程拿到的 `tty->ldisc` 不会在本次 `write()` 中途被切换或释放。对本链路来说，它通常拿到的就是 N_TTY；它本身不复制协议帧，只是在进入 `iterate_tty_write()` 前稳定行规程对象。

来源：linux/drivers/tty/tty_ldisc.c:tty_ldisc_ref_wait（已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_ldisc.c
struct tty_ldisc *tty_ldisc_ref_wait(struct tty_struct *tty) // 等待并获取当前 tty line discipline
{
    struct tty_ldisc *ld;                         // 保存当前行规程指针

    ldsem_down_read(&tty->ldisc_sem, MAX_SCHEDULE_TIMEOUT); // 以读模式锁住 ldisc 生命周期
    ld = tty->ldisc;                              // 读取当前行规程，通常是 N_TTY
    if (!ld)                                      // 如果行规程已经不存在
        ldsem_up_read(&tty->ldisc_sem);           // 释放刚拿到的读锁
    return ld;                                    // 返回行规程指针或 NULL
}
```

### 5.2 `iterate_tty_write()`

`iterate_tty_write()` 是本次 `write()` 路径里第一次真正的数据复制位置。VFS 传下来的 `iov_iter` 仍然指向用户态缓冲，TTY core 不能把用户态指针直接交给底层驱动，所以先把用户缓冲里的协议帧复制到 `tty->write_buf` 这个内核临时缓冲，再调用行规程的 `ld->ops->write()`。

来源：linux/drivers/tty/tty_io.c:iterate_tty_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
static ssize_t iterate_tty_write(struct tty_ldisc *ld, struct tty_struct *tty,  // 定义当前层的 C 函数入口
                                 struct file *file, struct iov_iter *from)  // 定义当前链路涉及的内核数据结构
{
    size_t chunk, count = iov_iter_count(from);      // 用户态本次要写的字节数
    ssize_t ret, written = 0;                        // 返回值和累计写入数

    ret = tty_write_lock(tty, file->f_flags & O_NDELAY); // 串行化同一 TTY 写入
    if (ret < 0)  // 检查状态或错误码，决定是否走异常/分支路径
        return ret;  // 把本层处理结果或错误码返回上一层

    chunk = 2048;                                    // TTY 默认分块大小
    if (count < chunk)  // 检查状态或错误码，决定是否走异常/分支路径
        chunk = count;  // 更新当前层需要传递的状态、长度、指针或错误码

    if (tty->write_cnt < chunk) {                    // 内核临时写缓冲不够大
        u8 *buf_chunk = kvmalloc(chunk, GFP_KERNEL | __GFP_RETRY_MAYFAIL);  // 更新当前层需要传递的状态、长度、指针或错误码
        kvfree(tty->write_buf);  // 调用下一层 C 函数继续完成当前路径
        tty->write_cnt = chunk;  // 更新当前层需要传递的状态、长度、指针或错误码
        tty->write_buf = buf_chunk;  // 更新当前层需要传递的状态、长度、指针或错误码
    }

    for (;;) {  // 循环处理队列、缓冲区或 fd 集合
        size_t size = min(chunk, count);  // 更新当前层需要传递的状态、长度、指针或错误码

        // 第一次 CPU 参与的数据复制：
        // 从用户态 buf 指向的协议帧，复制到 TTY 的内核临时缓冲 tty->write_buf。
        if (copy_from_iter(tty->write_buf, size, from) != size)  // 检查状态或错误码，决定是否走异常/分支路径
            break;  // 本行参与当前 C 层路径的控制流或数据准备

        ret = ld->ops->write(tty, file, tty->write_buf, size); // 调 N_TTY 的 write
        if (ret <= 0)  // 检查状态或错误码，决定是否走异常/分支路径
            break;  // 本行参与当前 C 层路径的控制流或数据准备
        written += ret;  // 更新当前层需要传递的状态、长度、指针或错误码
        count -= ret;  // 更新当前层需要传递的状态、长度、指针或错误码
        if (!count)  // 检查状态或错误码，决定是否走异常/分支路径
            break;  // 本行参与当前 C 层路径的控制流或数据准备
    }

    tty_write_unlock(tty);                           // 释放 TTY 写锁
    return written ? written : ret;  // 把本层处理结果或错误码返回上一层
}
```

`tty_write_lock()` 和 `copy_from_iter()` 是这个阶段最容易漏看的两个 helper。前者串行化同一个 TTY 的写入，后者才是真正把用户态协议帧复制进 TTY 内核临时缓冲的动作。

来源：linux/drivers/tty/tty_io.c:tty_write_lock / linux/include/linux/uio.h:copy_from_iter（已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
int tty_write_lock(struct tty_struct *tty, bool ndelay) // 串行化同一个 tty 的写路径
{
    if (!mutex_trylock(&tty->atomic_write_lock)) { // 尝试获取 tty 写锁
        if (ndelay)                                // 非阻塞写不能等待锁
            return -EAGAIN;                        // 立即返回稍后再试
        if (mutex_lock_interruptible(&tty->atomic_write_lock)) // 阻塞等待写锁，可被信号打断
            return -ERESTARTSYS;                   // 等锁期间被信号打断
    }
    return 0;                                      // 成功获得写锁
}

// linux/include/linux/uio.h
static __always_inline __must_check
size_t copy_from_iter(void *addr, size_t bytes, struct iov_iter *i) // 从 iov_iter 指向的用户缓冲复制到内核缓冲
{
    if (check_copy_size(addr, bytes, false))       // 检查目标内核缓冲大小是否可信
        return _copy_from_iter(addr, bytes, i);    // 执行真正的用户态到内核态复制
    return 0;                                      // 检查失败则不复制
}
```

这里的 `copy_from_iter()` 可以理解为面向 `iov_iter` 的用户态拷贝接口，本质作用等价于“从用户态 `buf` 把字节安全复制进内核”。对于本项目的舵机包，它复制的就是 14 字节 Sync Read 请求或 26 字节 Sync Write 请求。

### 5.3 `n_tty_write()`

`n_tty_write()` 是 N_TTY 行规程的写方法，也是 TTY 层真正处理数据的地方。对于 pyserial 配置的原始模式（`O_OPOST` 关闭），它不做换行转换等输出后处理，直接把字节原样交给底层驱动的 `tty->ops->write()`。如果驱动暂时没空间（`num == 0`），它会睡眠在 `write_wait` 上，等 `acm_write_bulk()` 完成回调唤醒。

来源：linux/drivers/tty/n_tty.c:n_tty_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/n_tty.c
static ssize_t n_tty_write(struct tty_struct *tty, struct file *file,  // 定义当前层的 C 函数入口
                           const u8 *buf, size_t nr)  // 本行参与当前 C 层路径的控制流或数据准备
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

break_out:  // 本行参与当前 C 层路径的控制流或数据准备
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

来源：linux/drivers/usb/class/cdc-acm.c:acm_tty_write（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static ssize_t acm_tty_write(struct tty_struct *tty, const u8 *buf,  // 定义当前层的 C 函数入口
                             size_t count)  // 本行参与当前 C 层路径的控制流或数据准备
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
    wb = &acm->wb[wbn];  // 更新当前层需要传递的状态、长度、指针或错误码

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
    memcpy(wb->buf, buf, count);  // 调用下一层 C 函数继续完成当前路径
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
    stat = acm_start_wb(acm, wb);  // 更新当前层需要传递的状态、长度、指针或错误码
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

来源：linux/drivers/usb/class/cdc-acm.c:acm_start_wb（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static int acm_start_wb(struct acm *acm, struct acm_wb *wb)  // 定义当前层的 C 函数入口
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

### 6.3 USB core 提交 URB

`usb_submit_urb()` 是 USB core 的入口。USB core 负责校验 URB 合法性、记录 endpoint 状态，然后把 URB 交给通用 HCD 提交函数。树莓派 5 的 USB 主控走 xHCI，但这一小节先只看到 USB core/HCD 边界；xHCI 内部如何把 URB 转成 TRB、敲 doorbell 放到第 7 章展开。

来源：linux/drivers/usb/core/urb.c:usb_submit_urb（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/core/urb.c
int usb_submit_urb(struct urb *urb, gfp_t mem_flags)  // 定义当前层的 C 函数入口
{
    struct usb_device *dev;                       // USB 设备
    struct usb_host_endpoint *ep;                 // USB endpoint  // 定义当前链路涉及的内核数据结构

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

来源：linux/drivers/usb/core/hcd.c:usb_hcd_submit_urb（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/core/hcd.c
int usb_hcd_submit_urb(struct urb *urb, gfp_t mem_flags)  // 定义当前层的 C 函数入口
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

### 6.4 DMA 映射与 transfer_dma

`map_urb_for_dma()` 这一跳的真实分发也要算进完整链路。树莓派 5 使用 xHCI，`hcd->driver->map_urb_for_dma` 会先进入 `xhci_map_urb_for_dma()`；普通 CDC ACM 写包是 14/26 字节 bulk OUT，不满足 xHCI 的 Immediate Transfer（最多 8 字节 OUT）条件，最终落到通用 `usb_hcd_map_urb_for_dma()`。因为 CDC ACM 写 URB 的缓冲来自 `usb_alloc_coherent()` 并设置了 `URB_NO_TRANSFER_DMA_MAP`，通用映射函数会保留已有 `transfer_dma`，不会再复制协议帧。

来源：linux/drivers/usb/core/hcd.c:map_urb_for_dma / linux/drivers/usb/host/xhci.c:xhci_map_urb_for_dma / linux/drivers/usb/core/hcd.c:usb_hcd_map_urb_for_dma（节选：仅保留 CDC ACM 写链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/core/hcd.c
static int map_urb_for_dma(struct usb_hcd *hcd, struct urb *urb,  // USB core 调用的 DMA 映射分发入口
                           gfp_t mem_flags)       // 内存分配上下文，写完成回调里常见 GFP_ATOMIC
{
    if (hcd->driver->map_urb_for_dma)             // xHCI HCD 注册了自己的 DMA 映射钩子
        return hcd->driver->map_urb_for_dma(hcd, urb, mem_flags); // 先进入 xhci_map_urb_for_dma()
    else                                          // 其他没有自定义钩子的 HCD
        return usb_hcd_map_urb_for_dma(hcd, urb, mem_flags); // 直接走通用 DMA 映射
}

// linux/drivers/usb/host/xhci.c
static int xhci_map_urb_for_dma(struct usb_hcd *hcd, struct urb *urb, // xHCI 的 DMA 映射钩子
                                gfp_t mem_flags)  // 继续传给通用映射或临时缓冲分配
{
    struct xhci_hcd *xhci;                        // xHCI 主控私有结构

    xhci = hcd_to_xhci(hcd);                      // 从通用 HCD 取 xHCI 对象

    if (xhci_urb_suitable_for_idt(urb))           // 只有最多 8 字节的 OUT 传输才可能使用 Immediate Transfer
        return 0;                                 // IDT 会把小数据塞进 TRB 字段；14/26 字节写包不会走这里

    if (xhci->quirks & XHCI_SG_TRB_CACHE_SIZE_QUIRK) { // 某些 xHCI 需要临时缓冲规避 SG/TRB 缓存限制
        if (xhci_urb_temp_buffer_required(hcd, urb)) // 判断当前 URB 是否需要 bounce buffer
            return xhci_map_temp_buffer(hcd, urb); // 需要时分配并映射临时缓冲
    }
    return usb_hcd_map_urb_for_dma(hcd, urb, mem_flags); // CDC ACM 常规路径进入通用 DMA 映射
}

// linux/drivers/usb/core/hcd.c
int usb_hcd_map_urb_for_dma(struct usb_hcd *hcd, struct urb *urb, // 通用 HCD DMA 映射函数
                            gfp_t mem_flags)     // 映射失败时可能按该上下文分配资源
{
    enum dma_data_direction dir;                  // DMA 方向：OUT 给设备，IN 从设备来
    int ret = 0;                                  // 默认映射成功

    dir = usb_urb_dir_in(urb) ? DMA_FROM_DEVICE : DMA_TO_DEVICE; // 写 URB 是 DMA_TO_DEVICE
    if (urb->transfer_buffer_length != 0          // 有实际数据缓冲才需要考虑映射
        && !(urb->transfer_flags & URB_NO_TRANSFER_DMA_MAP)) { // CDC ACM coherent 缓冲已设置该标志，因此跳过本分支
        if (hcd->localmem_pool) {                 // HCD 有本地内存池时可分配 coherent 缓冲
            ret = hcd_alloc_coherent(urb->dev->bus, mem_flags, // 分配 HCD 可 DMA 的缓冲
                                     &urb->transfer_dma, // 返回 DMA 地址
                                     &urb->transfer_buffer, // 返回 CPU 可访问地址
                                     urb->transfer_buffer_length, dir); // 按 URB 长度和方向分配
            if (ret == 0)                         // 分配成功
                urb->transfer_flags |= URB_MAP_LOCAL; // 记录后续完成时需要释放本地映射
        } else if (hcd_uses_dma(hcd)) {           // 普通 DMA HCD 走 dma_map_* 路径
            /* 这里省略 sg/page/single 三个通用映射分支；CDC ACM 写 URB 因 URB_NO_TRANSFER_DMA_MAP 不进入。 */ // 节选说明
        }
        if (ret && (urb->transfer_flags & (URB_SETUP_MAP_SINGLE | // 数据映射失败且 setup 包已映射时
                URB_SETUP_MAP_LOCAL)))            // 检查 setup 包是否需要回滚
            usb_hcd_unmap_urb_for_dma(hcd, urb);  // 回滚已经建立的 DMA 映射
    }
    return ret;                                   // CDC ACM 写 URB 通常返回 0，并保留驱动已有 transfer_dma
}
```

## 7. xHCI 主控写路径

前面 USB core 已经把 CDC ACM 写 URB 交给 HCD。对树莓派 5 来说，具体 HCD 是 xHCI；从这里开始，重点从“USB core 提交”切换为“xHCI 如何把 URB 描述成硬件可执行任务，并通过 doorbell MMIO 寄存器通知控制器”。

### 7.1 xHCI URB 入队

`xhci_urb_enqueue()` 是 xHCI 驱动的入口。它为 URB 分配 `urb_priv` 跟踪结构，按 endpoint 类型分发不同的传输队列。CDC ACM 的数据端点是 bulk，所以进入 `xhci_queue_bulk_tx()`，由它把 URB 数据转换成 xHCI 硬件能理解的 Transfer Request Blo到ck（TRB）。

来源：linux/drivers/usb/host/xhci.c:xhci_urb_enqueue（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/host/xhci.c
static int xhci_urb_enqueue(struct usb_hcd *hcd, struct urb *urb, gfp_t mem_flags)  // 定义当前层的 C 函数入口
{
    struct xhci_hcd *xhci = hcd_to_xhci(hcd);     // 从通用 HCD 取 xHCI 私有结构
    unsigned long flags;                          // 保存中断标志，保护 xHCI ring 操作
    int ret = 0;                                  // 保存入队结果
    unsigned int slot_id, ep_index;               // slot 是 USB 设备，ep_index 是 endpoint
    unsigned int *ep_state;                       // endpoint 状态位
    struct urb_priv *urb_priv;                    // xHCI 给 URB 分配的私有跟踪结构
    int num_tds;                                  // Transfer Descriptor 数量

    ep_index = xhci_get_endpoint_index(&urb->ep->desc); // 计算 endpoint 索引
    num_tds = 1;                                  // 普通 bulk 包通常一个 TD
    urb_priv = kzalloc(struct_size(urb_priv, td, num_tds), mem_flags); // 分配私有结构
    if (!urb_priv)                                // 内存不足
        return -ENOMEM;                           // 返回错误

    urb->hcpriv = urb_priv;                       // 把 xHCI 私有结构挂到 URB
    trace_xhci_urb_enqueue(urb);                  // 记录 xHCI URB 入队 tracepoint

    spin_lock_irqsave(&xhci->lock, flags);        // 锁住 xHCI 状态和 transfer ring
    ret = xhci_check_args(hcd, urb->dev, urb->ep, true, true, __func__); // 检查设备和 endpoint 参数
    if (ret <= 0) {                               // 参数非法或 endpoint 不可用
        ret = ret ? ret : -EINVAL;                // 0 也转换成错误码
        goto free_priv;                           // 释放 urb_priv 并返回
    }
    slot_id = urb->dev->slot_id;                  // USB 设备 slot id
    ep_state = &xhci->devs[slot_id]->eps[ep_index].ep_state; // 找到 endpoint 状态
    if (*ep_state & (EP_GETTING_STREAMS | EP_GETTING_NO_STREAMS)) { // endpoint 正在切换 streams 状态
        ret = -EINVAL;                            // 此时不能接收新的 URB
        goto free_priv;                           // 释放 urb_priv 并返回
    }
    if (*ep_state & EP_SOFT_CLEAR_TOGGLE) {       // endpoint 正在手工清 toggle
        ret = -EINVAL;                            // 避免在 toggle 状态不稳定时入队
        goto free_priv;                           // 释放 urb_priv 并返回
    }

    switch (usb_endpoint_type(&urb->ep->desc)) {  // 按 endpoint 类型排队
    case USB_ENDPOINT_XFER_BULK:                  // CDC ACM 数据端点是 bulk
        ret = xhci_queue_bulk_tx(xhci, GFP_ATOMIC, urb, slot_id, ep_index); // 写入 xHCI 传输环
        break;                                    // bulk 分支结束
    }

    if (ret) {                                    // 入队失败
free_priv:
        xhci_urb_free_priv(urb_priv);             // 释放 xHCI URB 私有结构
        urb->hcpriv = NULL;                       // 清掉 URB 上的 HCD 私有指针
    }
    spin_unlock_irqrestore(&xhci->lock, flags);   // 释放 xHCI 锁
    return ret;                                   // 0 表示 URB 已排进 xHCI ring
}
```

### 7.2 TRB 生成与 transfer ring

`xhci_queue_bulk_tx()` 会把 URB 数据描述成 TRB，并把这些 TRB 放入 endpoint 对应的 xHCI transfer ring。对普通 bulk OUT 来说，它在循环中计算本段数据的 DMA 地址、长度字段和控制字段，然后调用 `queue_trb()` 写入当前 ring 槽位。

来源：linux/drivers/usb/host/xhci-ring.c:xhci_queue_bulk_tx（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/host/xhci-ring.c
int xhci_queue_bulk_tx(struct xhci_hcd *xhci, gfp_t mem_flags,  // 定义当前层的 C 函数入口
        struct urb *urb, int slot_id, unsigned int ep_index)  // 定义当前链路涉及的内核数据结构
{
    ...  // 本行参与当前 C 层路径的控制流或数据准备
    addr = (u64) urb->transfer_dma;              // URB 数据缓冲的 DMA 地址
    ...  // 本行参与当前 C 层路径的控制流或数据准备
    length_field = TRB_LEN(trb_buff_len) |       // 本段传输长度
        TRB_TD_SIZE(remainder) |                 // TD 剩余大小提示
        TRB_INTR_TARGET(0);                      // 完成事件送到 interrupter 0

    queue_trb(xhci, ring, more_trbs_coming | need_zero_pkt,  // 进入 xHCI 主控队列或门铃通知路径
            lower_32_bits(send_addr),            // TRB field[0]：buffer DMA 地址低 32 位
            upper_32_bits(send_addr),            // TRB field[1]：buffer DMA 地址高 32 位
            length_field,                        // TRB field[2]：长度和中断目标
            field);                              // TRB field[3]：类型、cycle、IOC、CHAIN 等
    ...  // 本行参与当前 C 层路径的控制流或数据准备
    giveback_first_trb(xhci, slot_id, ep_index, urb->stream_id,  // 进入 xHCI 主控队列或门铃通知路径
            start_cycle, start_trb);             // 所有 TRB 写完后，把首个 TRB 交给硬件并敲 doorbell
    return 0;                                    // xHCI 排队完成，沿 write 调用栈返回
}
```

`queue_trb()` 是真正把 TRB 写进 transfer ring 的函数。`ring->enqueue` 指向当前可写的 TRB 槽位，CPU 依次写入四个 32-bit 字段；最后调用 `inc_enq()` 推进 ring 的 enqueue 指针。

来源：linux/drivers/usb/host/xhci-ring.c:queue_trb（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/host/xhci-ring.c
static void queue_trb(struct xhci_hcd *xhci, struct xhci_ring *ring,  // 定义当前层的 C 函数入口
        bool more_trbs_coming,  // 本行参与当前 C 层路径的控制流或数据准备
        u32 field1, u32 field2, u32 field3, u32 field4)  // 本行参与当前 C 层路径的控制流或数据准备
{
    struct xhci_generic_trb *trb;  // 定义当前链路涉及的内核数据结构

    trb = &ring->enqueue->generic;               // 当前 transfer ring 的写入槽位
    trb->field[0] = cpu_to_le32(field1);         // 数据 buffer DMA 地址低 32 位
    trb->field[1] = cpu_to_le32(field2);         // 数据 buffer DMA 地址高 32 位
    trb->field[2] = cpu_to_le32(field3);         // 长度、TD size、interrupter
    /* make sure TRB is fully written before giving it to the controller */
    wmb();                                      // 保证前三个字段先写完
    trb->field[3] = cpu_to_le32(field4);         // 类型、cycle bit、IOC、CHAIN 等控制位

    trace_xhci_queue_trb(ring, trb);  // 进入 xHCI 主控队列或门铃通知路径
    inc_enq(xhci, ring, more_trbs_coming);       // 推进 enqueue 指针
}
```

TRB 入环本身还不等于硬件已经开始取任务。`xhci_queue_bulk_tx()` 会在所有 TRB 都写完后调用 `giveback_first_trb()`：这个函数先通过首个 TRB 的 cycle bit 把整串 TRB 一次性交给硬件，再写 doorbell 通知 xHCI 控制器开始处理这个 endpoint 的 transfer ring。

来源：linux/drivers/usb/host/xhci-ring.c:giveback_first_trb（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/host/xhci-ring.c
static void giveback_first_trb(struct xhci_hcd *xhci, int slot_id,  // 定义当前层的 C 函数入口
        unsigned int ep_index, unsigned int stream_id, int start_cycle,  // 本行参与当前 C 层路径的控制流或数据准备
        struct xhci_generic_trb *start_trb)  // 定义当前链路涉及的内核数据结构
{
    /*
     * Pass all the TRBs to the hardware at once and make sure this write
     * isn't reordered.
     */
    wmb();                                      // 保证前面写入的 TRB 先对硬件可见
    if (start_cycle)  // 检查状态或错误码，决定是否走异常/分支路径
        start_trb->field[3] |= cpu_to_le32(start_cycle); // 交出第一个 TRB 的 cycle bit
    else  // 处理前面条件不满足时的备选路径
        start_trb->field[3] &= cpu_to_le32(~TRB_CYCLE);  // 更新当前层需要传递的状态、长度、指针或错误码
    xhci_ring_ep_doorbell(xhci, slot_id, ep_index, stream_id); // 通知硬件取 ring
}
```

### 7.3 xHCI doorbell 寄存器写全链路

真正的“敲门铃”发生在 `xhci_ring_ep_doorbell()`。这一节只追一件事：CPU 到底怎样把一个 32 位 doorbell value 写进 xHCI 控制器的 MMIO 寄存器。完整展开链路如下：

```text
xhci_ring_ep_doorbell()
  -> 计算 db_addr = &xhci->dba->doorbell[slot_id]
  -> 计算 value = DB_VALUE(ep_index, stream_id)
  -> readl(db_addr)                 # 写前序列化，最终是 ldr/ldar
  -> writel(value, db_addr)          # 真正写 MMIO，最终是 dmb oshst + str
  -> readl(db_addr)                 # 写后读回，flush posted MMIO write
```

#### 7.3.1 函数入口参数与 endpoint 状态检查

##### 7.3.1.1 函数入口总览

来源：linux/drivers/usb/host/xhci-ring.c:xhci_ring_ep_doorbell（按当前仓库源码核对）
```c
// linux/drivers/usb/host/xhci-ring.c
void xhci_ring_ep_doorbell(struct xhci_hcd *xhci,          // xHCI 主控的软件对象，保存寄存器基址、设备表和 ring 状态
        unsigned int slot_id,                              // 当前 USB 设备在 xHCI 中的 slot 编号
        unsigned int ep_index,                             // 当前 endpoint 在 xHCI 驱动内部的索引
        unsigned int stream_id)                            // USB streams 的 stream 编号；普通 CDC ACM bulk OUT 通常为 0
{
    __le32 __iomem *db_addr = &xhci->dba->doorbell[slot_id]; // 计算当前 slot 的 doorbell MMIO 寄存器地址
    struct xhci_virt_ep *ep = &xhci->devs[slot_id]->eps[ep_index]; // 找到当前 slot/endpoint 对应的软件状态对象
    unsigned int ep_state = ep->ep_state;                   // 读取 endpoint 状态位，决定能不能敲 doorbell

    if ((ep_state & EP_STOP_CMD_PENDING) || (ep_state & SET_DEQ_PENDING) || // 如果 endpoint 正在停止或正在设置 dequeue 指针
        (ep_state & EP_HALTED) || (ep_state & EP_CLEARING_TT))              // 或者 endpoint 已 halt / 正在清 TT 状态
        return;                                             // 此时不能敲 doorbell，直接返回，不写寄存器

    trace_xhci_ring_ep_doorbell(slot_id, DB_VALUE(ep_index, stream_id)); // 记录 tracepoint，只用于追踪，不通知硬件

    readl(db_addr);                                         // 写前读 doorbell，做 PCIe/MMIO 序列化
    writel(DB_VALUE(ep_index, stream_id), db_addr);          // 把 doorbell value 写入 xHCI doorbell MMIO 寄存器
    readl(db_addr);                                         // 写后读 doorbell，flush posted MMIO write
}
```

##### 7.3.1.2 四个入口参数

入口参数逐个拆开：

```text
xhci       # 当前 xHCI 主控的软件对象，里面保存 MMIO 基址、设备表、transfer ring 等状态
slot_id    # xHCI 给当前 USB 设备分配的 slot 编号；doorbell[slot_id] 就是该设备的门铃寄存器
ep_index   # xHCI 驱动内部的 endpoint 索引；DB_VALUE 会把它转换成 doorbell 的 Endpoint Target
stream_id  # USB streams 的 stream 编号；CDC ACM 普通 bulk OUT 通常为 0
```

##### 7.3.1.3 `ep` 是什么，为什么要先取它

`ep` 是 `endpoint` 的缩写。这里的 `ep` 不是 USB 设备本身，也不是舵机，而是“当前 USB 设备里某一条 endpoint 通道”在 xHCI 驱动里的软件状态对象。`writePort` 写舵机时，最终走的是 CDC ACM 数据接口的 bulk OUT endpoint，所以这里的 `ep` 可以理解成“这个 USB 串口板的 bulk OUT 传输通道状态”。

这一行按层级拆开是：

```text
xhci                              # xHCI 主控对象
xhci->devs                        # xHCI 维护的 slot_id -> USB 设备对象数组
xhci->devs[slot_id]               # 当前 slot_id 对应的 USB 设备对象
xhci->devs[slot_id]->eps          # 当前 USB 设备的 endpoint 软件状态数组
xhci->devs[slot_id]->eps[ep_index] # 当前 endpoint 的软件状态对象
&xhci->devs[slot_id]->eps[ep_index] # 取这个 endpoint 状态对象的地址
ep                                # 用 ep 指针保存这个地址，后面读 ep->ep_state
```

对应的结构关系在源码里是这样定义的：

来源：linux/drivers/usb/host/xhci.h:xhci_hcd / xhci_virt_device / xhci_virt_ep（节选）
```c
// linux/drivers/usb/host/xhci.h
struct xhci_hcd {                                      // xHCI 主控的软件总对象
    struct xhci_virt_device *devs[MAX_HC_SLOTS];       // slot_id 到 USB 设备软件对象的映射表
};

#define EP_CTX_PER_DEV 31                              // 每个 xHCI 设备最多跟踪 31 个 endpoint context

struct xhci_virt_device {                              // xHCI 中一个 USB 设备的软件对象
    int slot_id;                                       // 这个 USB 设备在 xHCI 里的 slot 编号
    struct usb_device *udev;                           // USB core 层的 USB 设备对象
    struct xhci_container_ctx *out_ctx;                // 硬件输出 context，xHCI 会更新它
    struct xhci_container_ctx *in_ctx;                 // 给 xHCI 命令使用的输入 context
    struct xhci_virt_ep eps[EP_CTX_PER_DEV];           // 这个 USB 设备的 endpoint 软件状态数组
};

struct xhci_virt_ep {                                  // xHCI 中一个 endpoint 的软件状态对象
    struct xhci_virt_device *vdev;                     // parent：指回所属 USB 设备对象
    unsigned int ep_index;                             // 当前 endpoint 在 eps[] 数组里的索引
    struct xhci_ring *ring;                            // 当前 endpoint 的 transfer ring，TRB 就排在这里
    struct xhci_stream_info *stream_info;              // streams 模式的信息；普通 CDC ACM bulk OUT 通常不用
    struct xhci_ring *new_ring;                        // 配置 endpoint 失败时用于恢复状态的临时 ring
    unsigned int err_count;                            // endpoint 相关错误计数
    unsigned int ep_state;                             // endpoint 软件状态位，决定能不能继续排 URB / 敲 doorbell
#define SET_DEQ_PENDING      (1 << 0)                  // 正在设置 transfer ring dequeue pointer
#define EP_HALTED            (1 << 1)                  // endpoint halt/stall，通常需要错误恢复
#define EP_STOP_CMD_PENDING  (1 << 2)                  // Stop Endpoint 命令已经挂起，常见于取消 URB
#define EP_GETTING_STREAMS   (1 << 3)                  // 正在切到 streams 模式，暂时不要 enqueue URB
#define EP_HAS_STREAMS       (1 << 4)                  // endpoint 当前已经使用 streams
#define EP_GETTING_NO_STREAMS (1 << 5)                 // 正在退出 streams 模式，暂时不要 enqueue URB
#define EP_HARD_CLEAR_TOGGLE (1 << 6)                  // 需要硬清 data toggle 状态
#define EP_SOFT_CLEAR_TOGGLE (1 << 7)                  // 需要软清 data toggle 状态
#define EP_CLEARING_TT       (1 << 8)                  // 正在清 hub Transaction Translator 缓冲
    struct list_head cancelled_td_list;                // 已取消 TD 链表，取消 URB 时会用到
    struct xhci_hcd *xhci;                             // 指回 xHCI 主控对象
    struct xhci_segment *queued_deq_seg;               // 已提交 Set TR Dequeue 命令对应的 ring segment
    union xhci_trb *queued_deq_ptr;                    // 已提交 Set TR Dequeue 命令对应的 dequeue TRB 指针
};
```

这些 `eps[]` 元素是在分配 xHCI 虚拟设备时初始化的：

来源：linux/drivers/usb/host/xhci-mem.c:xhci_alloc_virt_device（节选）
```c
// linux/drivers/usb/host/xhci-mem.c
for (i = 0; i < 31; i++) {                            // 遍历这个 USB 设备的 31 个 endpoint 软件槽位
    dev->eps[i].ep_index = i;                         // 记录 endpoint 索引，后面 ep_index 就能反查自己
    dev->eps[i].vdev = dev;                           // 记录 parent USB 设备对象
    dev->eps[i].xhci = xhci;                          // 记录所属 xHCI 主控对象
    INIT_LIST_HEAD(&dev->eps[i].cancelled_td_list);   // 初始化该 endpoint 的取消 TD 链表
    INIT_LIST_HEAD(&dev->eps[i].bw_endpoint_list);    // 初始化该 endpoint 的带宽管理链表
}

dev->eps[0].ring = xhci_ring_alloc(xhci, 2, 1, TYPE_CTRL, 0, flags); // 给控制 endpoint 0 分配默认 control ring
```

所以，`struct xhci_virt_ep *ep = &xhci->devs[slot_id]->eps[ep_index];` 这一行的目的就是：先拿到“当前 slot 的当前 endpoint 状态对象”，后面才能检查 `ep->ep_state`，并且 xHCI 其他路径也能通过 `ep->ring` 找到这个 endpoint 的 transfer ring。

##### 7.3.1.4 endpoint 状态检查

`ep_state` 检查是为了避免在 endpoint 正处于停止、重设 dequeue、halt 或清 TT 状态时重新敲门铃：

```text
EP_STOP_CMD_PENDING  # 已经有 Stop Endpoint 命令挂起
SET_DEQ_PENDING      # 正在设置 transfer ring dequeue pointer
EP_HALTED            # endpoint 已 halt
EP_CLEARING_TT       # 正在清 transaction translator 相关状态
```

这些分支如果命中，函数直接 `return`，不会写 doorbell 寄存器。真正会写寄存器的路径必须先通过这些状态检查。

##### 7.3.1.5 tracepoint：源码怎么判断是否写入 buffer

`trace_xhci_ring_ep_doorbell(slot_id, DB_VALUE(ep_index, stream_id))` 只记录 tracepoint。它会再次计算一次 doorbell value 供追踪使用，不负责通知硬件；真正通知硬件的是后面的 `writel()`。

这一行也拆开看：

```c
trace_xhci_ring_ep_doorbell(slot_id, DB_VALUE(ep_index, stream_id)); // 把 slot_id 和 doorbell value 送进 Linux tracepoint 系统
```

其中 `DB_VALUE(ep_index, stream_id)` 还是前面那套 doorbell value 计算：

```text
doorbell = ((ep_index + 1) & 0xff) | (stream_id << 16)
```

这个 tracepoint 的定义在 `xhci-trace.h`：

来源：linux/drivers/usb/host/xhci-trace.h:xhci_log_doorbell / xhci_ring_ep_doorbell（节选）
```c
// linux/drivers/usb/host/xhci-trace.h
DECLARE_EVENT_CLASS(xhci_log_doorbell,                 // 定义一类 xHCI doorbell trace event 模板
    TP_PROTO(u32 slot, u32 doorbell),                   // 这个 trace event 接收两个参数：slot 和 doorbell
    TP_ARGS(slot, doorbell),                            // 调用 tracepoint 时传入的实参就是 slot、doorbell
    TP_STRUCT__entry(                                   // 定义 trace ring buffer 里要保存哪些字段
        __field(u32, slot)                              // 保存 slot_id
        __field(u32, doorbell)                          // 保存 doorbell value
    ),
    TP_fast_assign(                                     // 定义 tracepoint 触发时如何把参数写入 trace entry
        __entry->slot = slot;                           // 把函数参数 slot 存到 trace entry
        __entry->doorbell = doorbell;                   // 把函数参数 doorbell 存到 trace entry
    ),
    TP_printk("Ring doorbell for %s",                   // 定义用户读取 trace 时怎么格式化显示
          xhci_decode_doorbell(__get_buf(XHCI_MSG_MAX), // 申请一段临时字符串缓冲，用来生成可读文本
                               __entry->slot,           // 把 trace entry 里的 slot 传给解码函数
                               __entry->doorbell)       // 把 trace entry 里的 doorbell 传给解码函数
    )
);

DEFINE_EVENT(xhci_log_doorbell, xhci_ring_ep_doorbell,  // 基于 doorbell 模板生成 xhci_ring_ep_doorbell 事件
     TP_PROTO(u32 slot, u32 doorbell),                  // 生成的事件参数类型仍然是 slot、doorbell
     TP_ARGS(slot, doorbell)                            // 生成的事件调用实参仍然是 slot、doorbell
);
```

`DECLARE_EVENT_CLASS` 是“定义模板”：字段怎么存、怎么打印都写在这里。`DEFINE_EVENT` 是“用这个模板生成一个具体事件”。所以 `trace_xhci_ring_ep_doorbell()` 不是手写的普通函数，而是 Linux trace 宏根据 `DEFINE_EVENT(..., xhci_ring_ep_doorbell, ...)` 生成出来的 trace 调用入口。

这个具体事件背后有一个 `struct tracepoint` 对象，里面最关键的是 `key` 和 `funcs`。`key` 用来快速判断这个 tracepoint 有没有打开；`funcs` 是打开后要调用的 probe/回调列表。

来源：linux/include/linux/tracepoint-defs.h:struct tracepoint（节选）
```c
// linux/include/linux/tracepoint-defs.h
struct tracepoint {                                  // 一个内核 tracepoint 的运行时对象
    const char *name;                                // tracepoint 名字，例如 xhci_ring_ep_doorbell
    struct static_key key;                           // static key：快速判断这个 tracepoint 当前是否启用
    struct static_call_key *static_call_key;         // static call 优化用的 key，减少启用时的间接调用开销
    void *static_call_tramp;                         // static call 跳板地址
    void *iterator;                                  // 遍历 funcs 回调列表的函数
    void *probestub;                                 // probe stub，给 ftrace/perf 等注册回调用
    int (*regfunc)(void);                            // 第一次启用时的注册钩子
    void (*unregfunc)(void);                         // 最后一次关闭时的注销钩子
    struct tracepoint_func __rcu *funcs;             // 已注册的 trace 回调数组；未启用时通常是 NULL
};
```

这个对象默认就是“关闭”的。源码里创建 tracepoint 时把 `.key` 初始化成 `STATIC_KEY_INIT_FALSE`，把 `.funcs` 初始化成 `NULL`：

来源：linux/include/linux/tracepoint.h:DEFINE_TRACE_FN（节选）
```c
// linux/include/linux/tracepoint.h
#define DEFINE_TRACE_FN(_name, _reg, _unreg, proto, args) /* 为 _name 生成 tracepoint 对象和迭代函数 */ \
    static const char __tpstrtab_##_name[] /* 保存 tracepoint 名字字符串 */ \
    __section("__tracepoints_strings") = #_name; /* 把名字放进 __tracepoints_strings section */ \
    extern struct static_call_key STATIC_CALL_KEY(tp_func_##_name); /* 声明 static call key */ \
    int __traceiter_##_name(void *__data, proto); /* 声明遍历 probe 回调的 iterator */ \
    void __probestub_##_name(void *__data, proto); /* 声明 probe stub */ \
    struct tracepoint __tracepoint_##_name __used /* 定义真正的 tracepoint 对象 */ \
    __section("__tracepoints") = { /* 放进 __tracepoints section，供内核 tracing 框架发现 */ \
        .name = __tpstrtab_##_name, /* tracepoint 名字 */ \
        .key = STATIC_KEY_INIT_FALSE, /* 默认关闭：static_key_false() 初始不命中 */ \
        .static_call_key = &STATIC_CALL_KEY(tp_func_##_name), /* static call 优化所需 key */ \
        .static_call_tramp = STATIC_CALL_TRAMP_ADDR(tp_func_##_name), /* static call 跳板 */ \
        .iterator = &__traceiter_##_name, /* 启用后用于遍历 funcs 回调列表 */ \
        .probestub = &__probestub_##_name, /* ftrace/perf 注册 probe 用的 stub */ \
        .regfunc = _reg, /* tracepoint 启用时的注册函数 */ \
        .unregfunc = _unreg, /* tracepoint 关闭时的注销函数 */ \
        .funcs = NULL }; /* 默认没有任何回调，所以不会写 trace buffer */ \
    __TRACEPOINT_ENTRY(_name); /* 把该 tracepoint 放入 tracepoint 指针表 */
```

代入本事件名后，关键初始状态就是：

```text
__tracepoint_xhci_ring_ep_doorbell.name  = "xhci_ring_ep_doorbell"
__tracepoint_xhci_ring_ep_doorbell.key   = false
__tracepoint_xhci_ring_ep_doorbell.funcs = NULL
```

追到通用 tracepoint 宏，真实的判断在 `__DECLARE_TRACE` 里：

来源：linux/include/linux/tracepoint.h:__DECLARE_TRACE（节选）
```c
// linux/include/linux/tracepoint.h
#define __DECLARE_TRACE(name, proto, args, cond, data_proto) /* 声明并生成 trace_##name() 入口 */ \
    extern int __traceiter_##name(data_proto); /* 声明 trace 回调 iterator */ \
    DECLARE_STATIC_CALL(tp_func_##name, __traceiter_##name); /* 声明 static call 优化入口 */ \
    extern struct tracepoint __tracepoint_##name; /* 声明这个 tracepoint 的运行时对象 */ \
    static inline void trace_##name(proto) /* 生成 trace_xxx(...) 调用入口 */ \
    { /* trace_##name 函数体开始 */ \
        if (static_key_false(&__tracepoint_##name.key)) /* 判断该 tracepoint 当前是否启用 */ \
            __DO_TRACE(name, /* 启用时才进入 __DO_TRACE，未启用则不写 buffer */ \
                TP_ARGS(args), /* 把调用点参数传给 trace 回调 */ \
                TP_CONDITION(cond), 0); /* 检查 trace 条件，普通路径 rcuidle=0 */ \
    } /* trace_##name 函数体结束 */ \
    static inline bool trace_##name##_enabled(void) /* 生成 trace_xxx_enabled() 查询入口 */ \
    { /* enabled 查询函数体开始 */ \
        return static_key_false(&__tracepoint_##name.key); /* 返回同一个 static key 状态 */ \
    } /* enabled 查询函数体结束 */
```

把 `name` 代入 `xhci_ring_ep_doorbell` 后，调用点会生成这样的入口函数：

来源：linux/include/linux/tracepoint.h:__DECLARE_TRACE（节选，按本事件名代入理解）
```c
// linux/include/linux/tracepoint.h
static inline void trace_xhci_ring_ep_doorbell(u32 slot, u32 doorbell) // tracepoint 调用入口，参数就是 slot 和 doorbell
{
    if (static_key_false(&__tracepoint_xhci_ring_ep_doorbell.key))     // 只在 tracepoint 被启用时才进入 __DO_TRACE()
        __DO_TRACE(xhci_ring_ep_doorbell,                              // 调用注册到该 tracepoint 上的回调
            TP_ARGS(slot, doorbell),                                   // 把 slot、doorbell 作为回调参数
            TP_CONDITION(true), 0);                                    // 条件为 true，普通非 rcuidle trace
}
```

这就是“是否写入 trace buffer”的判断点。没有执行下面这种启用动作时：

```bash
echo 1 > /sys/kernel/debug/tracing/events/xhci-hcd/xhci_ring_ep_doorbell/enable
```

`static_key_false(&__tracepoint_xhci_ring_ep_doorbell.key)` 不命中，函数直接跳过，不进入 `__DO_TRACE()`，也就不会把 `slot_id` 和 `doorbell` 写进 trace buffer。启用之后，tracing 框架注册 probe，并把 static key 打开；下一次执行到这个调用点，才会进入 `__DO_TRACE()`。

`__DO_TRACE()` 的核心动作是：

来源：linux/include/linux/tracepoint.h:__DO_TRACE（节选）
```c
// linux/include/linux/tracepoint.h
#define __DO_TRACE(name, args, cond, rcuidle) /* 定义真正执行 tracepoint 回调的宏 */ \
    do { /* 用 do/while 包住多条语句 */                                      \
        int __maybe_unused __idx = 0; /* rcuidle 路径用到的 SRCU 索引，普通路径基本不用 */ \
        if (!(cond)) /* 如果 trace 条件不成立 */                             \
            return; /* 直接返回，不记录 trace */                              \
        preempt_disable_notrace(); /* 关闭抢占，避免 trace 回调执行中被普通调度打断 */ \
        __DO_TRACE_CALL(name, TP_ARGS(args)); /* 调用注册到该 tracepoint 上的 probe/回调 */ \
        preempt_enable_notrace(); /* 恢复抢占 */                              \
    } while (0) /* 宏整体表现为一条语句 */
```

trace entry 最终怎么显示，靠 `xhci_decode_doorbell()`：

来源：linux/drivers/usb/host/xhci.h:xhci_decode_doorbell（节选）
```c
// linux/drivers/usb/host/xhci.h
static inline const char *xhci_decode_doorbell(char *str, u32 slot, u32 doorbell) // 把 slot/doorbell 转成人能看的字符串
{
    u8 ep;                                                // 保存 doorbell 低 8 位的 endpoint target
    u16 stream;                                           // 保存 doorbell 高 16 位的 stream id
    int ret;                                              // 保存 sprintf 已写入的字符数

    ep = (doorbell & 0xff);                               // 取 bits 0..7：Endpoint Target
    stream = doorbell >> 16;                              // 取 bits 16..31：Stream ID

    if (slot == 0) {                                      // slot 0 表示 xHCI Command Ring doorbell
        sprintf(str, "Command Ring %d", doorbell);        // 格式化命令环 doorbell
        return str;                                       // 返回格式化后的字符串
    }
    ret = sprintf(str, "Slot %d ", slot);                 // 普通设备 doorbell：先写 Slot 编号
    if (ep > 0 && ep < 32)                                // 合法 endpoint target 范围
        ret = sprintf(str + ret, "ep%d%s",                // 继续拼 endpoint 编号和方向
                  ep / 2,                                 // 把 xHCI doorbell endpoint target 转成显示用 endpoint 编号
                  ep % 2 ? "in" : "out");                // 奇数显示 in，偶数显示 out
    else if (ep == 0 || ep < 248)                         // 0 或部分值是保留范围
        ret = sprintf(str + ret, "Reserved %d", ep);      // 显示 Reserved
    else                                                  // 其他高值留给厂商定义
        ret = sprintf(str + ret, "Vendor Defined %d", ep);// 显示 Vendor Defined
    if (stream)                                           // 如果 stream id 非 0
        ret = sprintf(str + ret, " Stream %d", stream);   // 继续追加 Stream 编号

    return str;                                           // 返回最终字符串
}
```

所以这行 trace 的真实作用是：

```text
输入：slot_id、doorbell value
保存：slot 字段、doorbell 字段
trace 输出格式：读取 /sys/kernel/debug/tracing/trace 或 trace_pipe 时显示为 Ring doorbell for Slot X epYin/epYout [Stream Z]
硬件效果：没有
寄存器效果：没有
USB 传输效果：没有
```

它只是给调试工具看的记录点。真正让 xHCI 控制器动作起来的仍然是后面的：

```c
writel(DB_VALUE(ep_index, stream_id), db_addr); // 真正把 doorbell value 写入 MMIO 寄存器，通知硬件
```

#### 7.3.2 doorbell 寄存器地址怎么算出来

`db_addr` 来自这一行：

```c
__le32 __iomem *db_addr = &xhci->dba->doorbell[slot_id]; // 取 doorbell array 中 slot_id 对应的 32 位 MMIO 寄存器地址
```

要理解它，先看 `xhci->dba` 怎样初始化。

来源：linux/drivers/usb/host/xhci-mem.c:xhci_mem_init（节选）
```c
// linux/drivers/usb/host/xhci-mem.c
val = readl(&xhci->cap_regs->db_off);              // 从 xHCI capability register 读取 DBOFF Doorbell Offset：doorbell array 偏移
val &= DBOFF_MASK;                                 // 清除 DBOFF 低 2 个保留位，只保留有效偏移
xhci->dba = (void __iomem *) xhci->cap_regs + val; // 用 capability register 基址加偏移，得到 doorbell array 的 MMIO 基址
```

来源：linux/drivers/usb/host/xhci-caps.h:DBOFF_MASK（节选）
```c
// linux/drivers/usb/host/xhci-caps.h
/* db_off bitmask - bits 0:1 reserved */ // DBOFF 的低 2 位是保留位，不能参与地址计算
#define DBOFF_MASK (~0x3)                 // 位掩码：把低 2 位清 0，保留高位偏移
```

这里的计算含义是：

```text
raw_db_off = readl(&xhci->cap_regs->db_off)
val        = raw_db_off & ~0x3
xhci->dba  = xhci->cap_regs + val
```

`xhci->cap_regs` 是 xHCI capability register 区域的 MMIO 基址，`db_off` 是 xHCI 规范给出的 Doorbell Array Offset。低 2 位是保留位，所以用 `DBOFF_MASK` 清掉。最终 `xhci->dba` 指向 xHCI doorbell array 的 MMIO 虚拟地址。

doorbell array 的结构是：

来源：linux/drivers/usb/host/xhci.h:xhci_doorbell_array（节选）
```c
// linux/drivers/usb/host/xhci.h
struct xhci_doorbell_array {              // xHCI doorbell array 的内核映射结构
    __le32 doorbell[256];                  // 256 个 32 位 little-endian doorbell 寄存器槽位
};
```

因此：

```c
&xhci->dba->doorbell[slot_id]              // 取第 slot_id 个 doorbell 槽位的地址
```

按 C 指针运算展开就是：

```text
db_addr = (u8 __iomem *)xhci->dba + slot_id * sizeof(__le32)
        = (u8 __iomem *)xhci->dba + slot_id * 4
```

这不是普通 RAM 地址，而是内核映射出来的设备寄存器地址。CPU 对这个地址执行 load/store，会经过设备内存/MMIO 通路到 xHCI 控制器，而不是写内存条。

#### 7.3.3 doorbell value 怎么算出来

写入 doorbell 寄存器的 32 位值来自 `DB_VALUE(ep_index, stream_id)`。

来源：linux/drivers/usb/host/xhci.h:DB_VALUE（节选）
```c
// linux/drivers/usb/host/xhci.h
#define DB_VALUE(ep, stream) ((((ep) + 1) & 0xff) | ((stream) << 16)) // 低 8 位写 Endpoint Target，高 16 位写 Stream ID
```

按位展开：

```text
value = (((ep_index + 1) & 0xff) | (stream_id << 16))

bits  0..7   Endpoint Target = ep_index + 1
bits  8..15  Reserved = 0
bits 16..31  Stream ID
```

对 CDC ACM 的普通 bulk OUT 来说，`stream_id` 通常是 0（这里面很长，反正只要是写就是0），所以：

```text
value = ((ep_index + 1) & 0xff) | (0 << 16)
      = (ep_index + 1) & 0xff
```

举例：

```text
假设 CDC ACM bulk OUT 是 endpoint 1 OUT：

epnum     = 1
direction = 0                 # OUT
ep_index  = 1 * 2 + 0 - 1 = 1
stream_id = 0

Endpoint Target = (1 + 1) & 0xff = 2
Stream ID       = 0 << 16 = 0
value           = 0x00000002
```


这个 value 不携带舵机协议帧，也不携带数据地址。协议帧已经在 CDC ACM 的 `wb->buf` 里，TRB 里保存了 `wb->buf` 的 DMA 地址和长度。doorbell value 只告诉 xHCI：“哪个 slot 的哪个 endpoint/stream 有新的 transfer ring 条目要处理。”

#### 7.3.4 第一次 readl：为什么写前还要读

`xhci_ring_ep_doorbell()` 在 `writel()` 前先执行：

```c
readl(db_addr);                            // 从 doorbell MMIO 地址读 32 位值；这里主要用读操作做序列化
```

源码注释说得很直接：在 Pi 4/Pi 5 这类非一致性 DMA + PCIe 场景里，TRB 写入、屏障完成、doorbell MMIO 写到达 Root Complex、endpoint DMA engine 看到 system RAM 中的新 TRB，几件事之间存在理论竞态。写前读一次 doorbell MMIO 地址，可以制造一次跨链路 round-trip，用来序列化 CPU 状态。

`readl()` 的通用定义是：

来源：linux/include/asm-generic/io.h:readl（节选）
```c
// linux/include/asm-generic/io.h
static inline u32 readl(const volatile void __iomem *addr)      // 定义 32 位 MMIO 读函数，addr 是 I/O 内存地址
{
    u32 val;                                                    // 保存从设备寄存器读出的 32 位值

    if (rwmmio_tracepoint_enabled(rwmmio_read))                 // 如果启用了 MMIO read tracepoint
        log_read_mmio(32, addr, _THIS_IP_, _RET_IP_);           // 记录一次 32 位 MMIO 读，便于调试追踪
    __io_br();                                                  // 读前屏障/架构钩子；ARM64 当前为空
    val = __le32_to_cpu((__le32 __force)__raw_readl(addr));      // 底层读取 MMIO，再从 little-endian 转成 CPU endian
    __io_ar(val);                                               // 读后屏障/架构钩子；ARM64 上会处理 DMA/设备读后的顺序
    if (rwmmio_tracepoint_enabled(rwmmio_post_read))            // 如果启用了 MMIO post-read tracepoint
        log_post_read_mmio(val, 32, addr, _THIS_IP_, _RET_IP_); // 记录读完成后的值和地址
    return val;                                                 // 返回读到的 32 位寄存器值
}
```

ARM64 的底层读函数是：

来源：linux/arch/arm64/include/asm/io.h:__raw_readl（节选）
```c
// linux/arch/arm64/include/asm/io.h
static __always_inline u32 __raw_readl(const volatile void __iomem *addr) // ARM64 底层 32 位 MMIO 读，强制内联
{
    u32 val;                                                    // 保存汇编指令从 MMIO 地址读出的 32 位值
    asm volatile(ALTERNATIVE("ldr %w0, [%1]",                   // 默认生成 ldr：从 addr 指向的设备地址读取 32 位
                 "ldar %w0, [%1]",                              // 如果命中 workaround，则替换为 acquire load
                 ARM64_WORKAROUND_DEVICE_LOAD_ACQUIRE)          // 控制是否把 ldr 替换成 ldar 的 ARM64 workaround 条件
             : "=r" (val) : "r" (addr));                        // 输出 val 用通用寄存器，输入 addr 用通用寄存器
    return val;                                                 // 把读出的值返回给 readl()
}
```

正常情况下最终类似：

```asm
ldr w8, [x9]                            // 从 x9 指向的 MMIO 地址读 32 位到 w8
```

如果启用了对应 ARM64 workaround，可能替换成 acquire 语义的：

```asm
ldar w8, [x9]                           // acquire 语义读取：从 x9 指向的 MMIO 地址读 32 位到 w8
```

这里的 `x9` 是 `db_addr` 对应的 MMIO 虚拟地址。因为地址属于设备寄存器映射，这次 `ldr/ldar` 不是读普通内存，而是读 xHCI doorbell MMIO 寄存器。读出的值本身没有被业务使用；重要的是这次 MMIO 读造成的顺序化效果。

#### 7.3.5 writel：真正写 doorbell 寄存器

真正写寄存器的是：

```c
writel(DB_VALUE(ep_index, stream_id), db_addr); // 把计算出的 doorbell value 写入 db_addr 指向的 MMIO 寄存器
```

先把它拆成两个中间值：

```text
value   = DB_VALUE(ep_index, stream_id)
db_addr = &xhci->dba->doorbell[slot_id]
```

然后进入 `writel()`。

来源：linux/include/asm-generic/io.h:writel（节选）
```c
// linux/include/asm-generic/io.h
static inline void writel(u32 value, volatile void __iomem *addr) // 定义 32 位 MMIO 写函数，value 是要写的值，addr 是设备寄存器地址
{
    if (rwmmio_tracepoint_enabled(rwmmio_write))                  // 如果启用了 MMIO write tracepoint
        log_write_mmio(value, 32, addr, _THIS_IP_, _RET_IP_);     // 记录写入值、宽度、地址和调用位置
    __io_bw();                                                    // 写前屏障：确保前面的 DMA/普通内存写先对设备可见
    __raw_writel((u32 __force)__cpu_to_le32(value), addr);        // 转成 little-endian 32 位后，执行底层 MMIO store
    __io_aw();                                                    // 写后架构钩子；ARM64 上为空操作
    if (rwmmio_tracepoint_enabled(rwmmio_post_write))             // 如果启用了 MMIO post-write tracepoint
        log_post_write_mmio(value, 32, addr, _THIS_IP_, _RET_IP_);// 记录 MMIO 写完成后的 trace 信息
}
```

trace 分支只用于 MMIO 访问追踪。核心三步是：

```text
__io_bw()
__raw_writel((u32 __force)__cpu_to_le32(value), addr)
__io_aw()
```

ARM64 的 I/O barrier 定义是：

来源：linux/arch/arm64/include/asm/io.h:IO barriers（节选）
```c
// linux/arch/arm64/include/asm/io.h
#define __io_bw()      dma_wmb()       // MMIO 写前屏障：ARM64 上转到 DMA 写屏障
#define __io_br(v)                     // MMIO 读前钩子：ARM64 这里为空
#define __io_aw(v)                     // MMIO 写后钩子：ARM64 这里为空
```

注意：ARM64 上空的是 `__io_aw(v)`，不是 `__io_bw()`。`__io_bw()` 会继续展开成 DMA 写屏障：

来源：linux/include/asm-generic/barrier.h:dma_wmb 与 linux/arch/arm64/include/asm/barrier.h:__dma_wmb（节选）
```c
// linux/include/asm-generic/barrier.h
#ifdef __dma_wmb                                      // 如果当前架构提供了底层 DMA 写屏障
#define dma_wmb()  do { kcsan_wmb(); __dma_wmb(); } while (0) // 先通知 KCSAN，再执行真实架构屏障
#endif                                                // 结束 __dma_wmb 条件定义

// linux/arch/arm64/include/asm/barrier.h
#define dmb(opt)      asm volatile("dmb " #opt : : : "memory") // 生成 ARM64 dmb 屏障指令，并带 compiler memory clobber
#define __dma_wmb()   dmb(oshst)                                // DMA 写屏障：Outer Shareable 域内排序 store
```

所以写前屏障主链路是：

```text
__io_bw()
  -> dma_wmb()
      -> kcsan_wmb()      # 未开 KCSAN 时为空；开 KCSAN 时给竞态检测器建模
      -> __dma_wmb()
          -> dmb(oshst)
```

`dmb oshst` 的作用是：在 Outer Shareable 域内，对 store 建立顺序。放在 doorbell 前，就是保证前面写入 transfer ring 的 TRB 先对设备可见，然后才允许 CPU 发出后面的 MMIO doorbell 写。

接着看字节序转换：

来源：linux/include/uapi/linux/byteorder/little_endian.h:__cpu_to_le32（节选）
```c
// linux/include/uapi/linux/byteorder/little_endian.h
#define __cpu_to_le32(x) ((__force __le32)(__u32)(x)) // 小端 CPU 上不换字节，只把 x 标记为 little-endian 32 位
```

树莓派 5 的 ARM64 Linux 通常是 little-endian，所以 `__cpu_to_le32(value)` 只是把 `value` 转成 `__le32` 类型标记，不会交换字节。若是 big-endian CPU，同名宏会执行字节交换，保证写给 xHCI 的 32 位寄存器值仍按 little-endian 解释。

最后进入 ARM64 的底层写函数：

来源：linux/arch/arm64/include/asm/io.h:__raw_writel（节选）
```c
// linux/arch/arm64/include/asm/io.h
static __always_inline void __raw_writel(u32 val, volatile void __iomem *addr) // ARM64 底层 32 位 MMIO 写，强制内联
{
    volatile u32 __iomem *ptr = addr;                         // 把 void __iomem 地址转成 32 位 MMIO 指针
    asm volatile("str %w0, %1" : : "rZ" (val), "Qo" (*ptr));  // 生成 str 指令，把 val 写到 ptr 指向的 MMIO 地址
}
```

最终汇编效果类似：

```asm
dmb oshst                               // DMA 写屏障：保证前面的 TRB 写入先对设备可见
str w8, [x9]                            // 把 w8 中的 32 位 doorbell value 写到 x9 指向的 MMIO 地址
```

其中：

```text
w8  # 存放 32 位 doorbell value
x9  # 存放 db_addr，即 xHCI doorbell[slot_id] 的 MMIO 地址
```

如果 `x9` 指向普通 RAM，`str` 就是普通内存写；但这里 `x9` 指向 xHCI doorbell MMIO 区域，所以这次 store 通过设备内存映射通路到达 xHCI 控制器。硬件看到这个 32 位写入后，就知道对应 slot/endpoint 的 transfer ring 有新 TRB 可以取。

#### 7.3.6 第二次 readl：flush posted MMIO write

`writel()` 后还有一次：

```c
readl(db_addr);                         // 写后再次读同一个 MMIO 地址，用来 flush posted MMIO write
```

这次读不是为了使用 doorbell 的返回值，而是为了 flush posted MMIO write。很多 MMIO 写在总线/Root Complex 侧可能是 posted write：CPU 发出 store 后，写请求可能先进入中间缓冲，CPU 侧看起来已经完成，但设备侧未必立刻收到。紧跟一次对同一 MMIO 区域的读，会迫使系统把前面的 posted write 推到设备侧，因为读请求必须等前面的相关写到达后才能返回一致的设备状态。

因此，三条 MMIO 操作的意义分别是：

```text
readl(db_addr)       # 写前序列化，确保 CPU/TRB/DMA 可见性状态更稳
writel(value, addr)  # 真正写 xHCI doorbell 寄存器
readl(db_addr)       # 写后 flush，确保 posted doorbell write 到达控制器
```

最终从 C 到汇编/计算可以压缩成：

```text
value   = ((ep_index + 1) & 0xff) | (stream_id << 16)
db_addr = (u8 __iomem *)xhci->dba + slot_id * 4

readl(db_addr)
  -> __raw_readl(db_addr)
  -> ldr/ldar wN, [db_addr]

writel(value, db_addr)
  -> __io_bw()
      -> dma_wmb()
          -> kcsan_wmb()
          -> __dma_wmb()
              -> dmb oshst
  -> __cpu_to_le32(value)
  -> __raw_writel(value, db_addr)
      -> str wN, [db_addr]
  -> __io_aw(v)     # ARM64 上为空

readl(db_addr)
  -> __raw_readl(db_addr)
  -> ldr/ldar wN, [db_addr]
```

到这里，CPU 已经把一个 32 位 doorbell value 写进 xHCI 的 doorbell MMIO 寄存器。xHCI 硬件随后根据 slot/endpoint 去 DMA 读取 transfer ring 中的 TRB，再根据 TRB 里的 DMA 地址读取 `wb->buf`，最终发出 USB bulk OUT。

### 7.4 ARM64 MMIO 访问宏与汇编补充

这一节集中放 `7.3` 用到的通用宏和汇编语法，避免主链路被背景知识打断。

#### 7.4.1 `volatile`、`__iomem`、`__force`

`volatile u32 __iomem *ptr = addr;` 本身不读写硬件，只是在 C 类型层面把 `addr` 从“未知宽度的 MMIO 地址”变成“指向 32 位 MMIO 寄存器的指针”。其中：

```text
volatile  # 告诉编译器：这个对象有外部副作用，不要把访问随便优化掉、合并或假设它像普通内存一样稳定
u32       # 这次访问宽度是 32 位
__iomem   # 内核 sparse 静态检查标记：这是 I/O 内存地址，不是普通 RAM 指针
```

`__iomem` 和 `__force` 的定义主要服务于 sparse 静态检查：

来源：linux/include/linux/compiler_types.h:__iomem 与 __force（节选）
```c
#ifdef __CHECKER__                                      // sparse 静态检查器启用时进入该分支
# define __iomem  __attribute__((noderef, address_space(__iomem))) // 标记 I/O 内存地址空间，禁止当普通指针解引用
# define __force  __attribute__((force))                // 允许显式强制转换，告诉 sparse 这是有意转换
#else                                                   // 普通 C 编译器编译内核时进入该分支
# define __iomem                                        // 普通编译时展开为空，不生成机器指令
# define __force                                        // 普通编译时展开为空，不生成机器指令
#endif                                                  // 结束 __CHECKER__ 条件分支
```

普通编译时，`__iomem` 和 `__force` 多数情况下展开为空；但在 sparse 检查下，它们能防止把普通内存指针和 I/O 内存指针混用。

#### 7.4.2 `__always_inline`

来源：linux/include/linux/compiler_attributes.h:__always_inline（节选）
```c
#define __always_inline inline __attribute__((__always_inline__)) // 要求编译器尽量强制内联该函数
```

`__raw_writel()` 前面的 `__always_inline` 要求编译器把函数体直接展开到调用点，而不是生成一次 `bl __raw_writel` 这样的函数调用。因此最终代码更接近：

```text
dmb oshst
str wN, [db_addr]
```

而不是：

```text
dmb oshst
bl  __raw_writel
ret
```

#### 7.4.3 inline asm 约束怎么读

`__raw_writel()` 里的汇编是：

```c
asm volatile("str %w0, %1" : : "rZ" (val), "Qo" (*ptr)); // 内联汇编：把 val 用 32 位 store 写到 ptr 指向的地址
```

GCC/Clang inline asm 的格式是：

```text
asm volatile("汇编模板" : 输出操作数 : 输入操作数 : clobber 列表)
```

本句没有输出操作数，也没有显式 clobber 列表，只有两个输入操作数：

```text
"rZ" (val)    # 第 0 个输入操作数：要写出的 32 位值，放入通用寄存器；如果是 0，可用架构允许的 zero 形式
"Qo" (*ptr)   # 第 1 个输入操作数：`str` 可以接受的内存操作数，地址来自 ptr 指向的 MMIO 寄存器
```

模板里的 `str %w0, %1` 中：

```text
%w0  # 第 0 个操作数的 32 位 W 寄存器形式
%1   # 第 1 个操作数对应的内存地址操作数
```

最终效果类似：

```asm
str w8, [x9]                            // 把 w8 中的 32 位值写入 x9 指向的 MMIO 地址
```

`__raw_readl()` 的汇编是：

```c
asm volatile(ALTERNATIVE("ldr %w0, [%1]",                  // 默认模板：用 ldr 从 addr 指向地址读 32 位
             "ldar %w0, [%1]",                              // 替换模板：用 ldar 做 acquire 语义读取
             ARM64_WORKAROUND_DEVICE_LOAD_ACQUIRE)          // ARM64 workaround 条件，决定是否替换为 ldar
         : "=r" (val) : "r" (addr));                        // 输出到 val，输入是 addr
```

这里有一个输出操作数：

```text
"=r" (val)  # 输出：把读到的 32 位值放入通用寄存器，再写回 C 变量 val
"r" (addr)  # 输入：把 MMIO 地址放入通用寄存器
```

正常生成 `ldr`；如果命中 `ARM64_WORKAROUND_DEVICE_LOAD_ACQUIRE`，运行时替换为 `ldar`。

#### 7.4.4 `kcsan_wmb()`、`__dma_wmb()` 与 `do { } while (0)`

`dma_wmb()` 的定义是：

来源：linux/include/asm-generic/barrier.h:dma_wmb（节选）
```c
// linux/include/asm-generic/barrier.h
#ifdef __dma_wmb                                      // 如果架构已经定义了底层 __dma_wmb()
#define dma_wmb()  do { kcsan_wmb(); __dma_wmb(); } while (0) // 定义通用 dma_wmb：先给 KCSAN 建模，再执行真实 DMA 写屏障
#endif                                                // 结束 __dma_wmb 条件分支
```

`do { ... } while (0)` 不是为了循环，而是把多条语句包成“像一条语句一样”的宏，避免 `if/else` 展开后语法错乱：

```c
if (cond)                                  // 如果条件成立
    dma_wmb();                             // 宏展开后仍表现为一条语句
else                                       // 否则分支能正确匹配这个 if
    foo();                                 // 执行其他逻辑
```

`kcsan_wmb()` 是给 KCSAN 数据竞争检测器建模用的。普通未打开 KCSAN 弱内存检测时，它是空操作：

来源：linux/include/linux/kcsan-checks.h:kcsan_wmb（节选）
```c
// linux/include/linux/kcsan-checks.h
// 如果开启 KCSAN 弱内存模型，并且使用 ThreadSanitizer 插桩，就走 signal fence 建模路径。
#if defined(CONFIG_KCSAN_WEAK_MEMORY) && defined(__SANITIZE_THREAD__)
// 定义把内核屏障映射成 signal fence 的辅助宏；注意续行反斜杠必须在行尾。
#define __KCSAN_BARRIER_TO_SIGNAL_FENCE(name) /* 定义 signal fence 辅助宏，name 决定 fence 类型 */ \
    do { /* 用 do/while(0) 把多条语句包成一条宏语句 */                         \
        barrier(); /* 编译器屏障：阻止这之前的 C 访问被编译器搬到后面 */          \
        __atomic_signal_fence(__KCSAN_BARRIER_TO_SIGNAL_FENCE_##name); /* 给 ThreadSanitizer/KCSAN 一个屏障事件 */ \
        barrier(); /* 编译器屏障：阻止这之后的 C 访问被编译器搬到前面 */          \
    } while (0) /* 宏整体表现为一条语句，方便放进 if/else */
// KCSAN 写屏障建模：把 kcsan_wmb() 映射成 wmb 类型 signal fence。
#define kcsan_wmb()     __KCSAN_BARRIER_TO_SIGNAL_FENCE(wmb)
// 如果开启 KCSAN 弱内存模型，并且使用显式 barrier 插桩，就绑定到 __kcsan_wmb。
#elif defined(CONFIG_KCSAN_WEAK_MEMORY) && defined(__KCSAN_INSTRUMENT_BARRIERS__)
// 把 kcsan_wmb 绑定到真实插桩函数/符号。
#define kcsan_wmb       __kcsan_wmb
// 普通构建或未开启 KCSAN 弱内存屏障建模时，走空操作。
#else
// 普通构建里为空操作，不生成硬件指令。
#define kcsan_wmb()     do { } while (0)
// 结束 KCSAN 条件分支。
#endif
```

`__dma_wmb()` 才是 ARM64 上真正生成硬件屏障指令的部分：

来源：linux/arch/arm64/include/asm/barrier.h:__dma_wmb（节选）
```c
// linux/arch/arm64/include/asm/barrier.h
#define dmb(opt)      asm volatile("dmb " #opt : : : "memory") // 生成 ARM64 dmb 屏障，opt 被拼成屏障域/类型
#define __dma_wmb()   dmb(oshst)                                // DMA 写屏障：Outer Shareable 域内对 store 排序
```

因此，`writel()` 写 doorbell 前的实际硬件顺序保障来自：

```asm
dmb oshst                               // ARM64 数据内存屏障：Outer Shareable 域内排序 store
```

#### 7.4.5 小端转换 `__cpu_to_le32()`

`writel()` 里有：

```c
__raw_writel((u32 __force)__cpu_to_le32(value), addr); // 转成 little-endian 32 位后，写到 addr 指向的 MMIO 地址
```

小端 ARM64 上定义为：

来源：linux/include/uapi/linux/byteorder/little_endian.h:__cpu_to_le32（节选）
```c
#define __cpu_to_le32(x) ((__force __le32)(__u32)(x)) // 小端 CPU 上只做类型转换和 endian 标记，不交换字节
```

拆开理解：

```text
(__u32)(x)       # 先把 x 转成 32 位无符号整数
(__le32)(...)    # 再把它标记为 little-endian 32-bit 值
__force          # 告诉 sparse：这里是有意做 endian 类型转换
```

树莓派 5 的 ARM64 Linux 通常是 little-endian，所以这里不发生字节交换；它保证同一份驱动在 big-endian CPU 上也能把 xHCI 要求的 little-endian 32 位值写出去。

---

## 8. 写完成回调

数据提交给 xHCI 后，`writePort()` 在 `acm_tty_write()` 返回时就已经结束了，它不会等待 USB 传输真正完成。当 xHCI 把 bulk OUT 数据发送出去后，USB 主控会产生完成中断，触发 `acm_write_bulk()` 回调。这个回调的职责是：释放写缓冲、标记传输完成、唤醒可能阻塞在 `write_wait` 上的进程。本节解释完成回调如何打扫战场。

来源：linux/drivers/usb/class/cdc-acm.c:acm_write_bulk（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static void acm_write_bulk(struct urb *urb)  // 定义当前层的 C 函数入口
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

## 9. writePort 究竟完成了什么

`writePort(packet)` 完成的是：

1. 把 Python 的协议帧字节复制到内核。
2. 通过 TTY/N_TTY 把字节交给 CDC ACM 驱动。
3. CDC ACM 分配 `acm_wb`，复制到 USB 写缓冲。
4. 构造并提交 USB bulk OUT URB。
5. xHCI 把 URB 转成硬件可执行的 bulk transfer。

从 `write()` 系统调用进入内核开始，本次路径里 CPU 参与的数据复制主要有两次：

| 次数 | 源地址                                                | 目标地址                            | 代码位置                                     | 作用                                         |
| ---- | ----------------------------------------------------- | ----------------------------------- | -------------------------------------------- | -------------------------------------------- |
| 1    | 用户态 `buf`，也就是 pyserial 传给 `write()` 的协议帧 | TTY core 的 `tty->write_buf`        | `copy_from_iter(tty->write_buf, size, from)` | 把用户态字节安全搬进内核临时缓冲             |
| 2    | N_TTY 传下来的 `buf`，实际指向 `tty->write_buf`       | CDC ACM 的 `wb->buf` DMA 安全写缓冲 | `memcpy(wb->buf, buf, count)`                | 把协议帧放进 USB bulk OUT URB 使用的数据缓冲 |

`map_urb_for_dma()` 和 xHCI 入队主要是在建立 DMA 访问关系、写 TRB 和敲 doorbell，不是再把 14 或 26 字节协议帧做一次普通 CPU `memcpy()`。随后真正把数据送到 USB 总线的是 xHCI 硬件通过 DMA 读取 `wb->buf`。

最终系统调用参数是：

```text
write(fd=/dev/ttyACM0, buf=<用户态协议帧地址>, count=14 或 26)
```

内核完成串口发送的关键对象是：

```text
struct file -> struct tty_struct -> struct tty_ldisc(N_TTY) -> struct tty_operations(acm_ops) -> struct urb
```
