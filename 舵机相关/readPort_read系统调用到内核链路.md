# readPort：read 系统调用到 Linux 内核链路

本文从 `scservo_sdk/port_handler.py` 的 `PortHandler.readPort(length)` 开始，重点解释它触发的 `read()` 系统调用：一次舵机回包如何从 USB CDC ACM 设备进入内核缓冲区，再被 pyserial 通过 `select + read` 取回 Python。

读路径可以分成一个后台接收阶段和两个用户态系统调用阶段：

1. USB 后台接收：CDC ACM 驱动提前提交 bulk IN URB，xHCI 收到舵机回包后通过 DMA 写入 URB 缓冲，完成回调再把数据推入 TTY/N_TTY 接收缓冲。
2. `select.select([fd, pipe], [], [], 0)`：检查 `/dev/ttyACM0` 是否已有数据可读，内核路径是 `pselect6 -> do_select -> vfs_poll -> tty_poll -> n_tty_poll`。
3. `os.read(fd, size)`：真正把数据从 N_TTY 缓冲区复制到用户态，内核路径是 `read -> ksys_read -> vfs_read -> new_sync_read -> tty_read -> n_tty_read -> copy_to_iter`。

其中 `select()` 在 Linux 内核中通常表现为 `pselect6`，它只是 `read()` 之前的可读性检查；协议字节真正返回用户态发生在后面的 `read()` 中。`pselect6` 的公共链路单独见 [pyserial_select_pselect6到内核链路.md](pyserial_select_pselect6到内核链路.md)。

本次源码核对后的补齐点：

| 链路段 | 核对结论 |
| --- | --- |
| VFS 读分发 | 已补 `new_sync_read()`，明确 `vfs_read -> new_sync_read -> tty_fops.read_iter` |
| 读 URB 预提交 | 已补 `acm_submit_read_urb -> usb_submit_urb -> usb_hcd_submit_urb -> xhci_urb_enqueue -> xhci_queue_bulk_tx` |
| flip buffer 复制 | 已补 `tty_insert_flip_string()` 到 `__tty_insert_flip_string_flags()` 的复制路径 |
| CPython 来源 | 来源标识覆盖 `os_read_impl/_Py_read`，不再只标入口函数 |

---

## 图：read() syscall 调用路径

```
read(fd, buf, count)                    # Python/CPython 调用入口
  │
  ▼
VFS: vfs_read()                         # 权限检查后分发到 tty_fops.read_iter
  │
  ▼
VFS: new_sync_read()                    # 包装 kiocb/iov_iter 后调用 read_iter
  │
  ▼
TTY: tty_read()                         # TTY 读入口，获取当前 line discipline
  │
  ▼
N_TTY: n_tty_read()                     # 从 ldata->read_buf 消费已经到达的字节
  │
  ▼
copy_from_read_buf()                    # 从 N_TTY 环形 read_buf 拷到 TTY 内核临时缓冲
  │
  ▼
copy_to_iter()                          # 再复制到 CPython 分配的用户态 bytes 缓冲
```

上面这张图只保留 `read syscall` 的同步调用路径，不再把 USB 回包进入 `read_buf` 的后台异步接收链路画进同一张图里；异步接收过程单独放在后文的“USB 回包进入内核缓冲区”一节说明。

---

## 1. scservo_sdk 入口

`readPort()` 在 `protocol_packet_handler.rxPacket()` 的循环中被反复调用：

来源：scservo_sdk/protocol_packet_handler.py:相关代码片段（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# scservo_sdk/protocol_packet_handler.py
rxpacket.extend(port.readPort(wait_length - rx_length))  # 非阻塞读取还缺的字节
rx_length = len(rxpacket)                                # 更新当前已收到长度
```

SDK 起点源码：

来源：scservo_sdk/port_handler.py:readPort（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# scservo_sdk/port_handler.py
def readPort(self, length):                  # length 是本次最多希望读取的字节数
    if (sys.version_info > (3, 0)):           # Python 3 分支
        return self.ser.read(length)         # 调 pyserial Serial.read(size)
    else:                                    # Python 2 分支
        return [ord(ch) for ch in self.ser.read(length)] # 转成 int 列表
```

在本项目里 `serial.Serial(..., timeout=0)`，因此 `readPort()` 是非阻塞读：有多少读多少，没有数据就返回空 `bytes`，整体超时由 SDK 的 `isPacketTimeout()` 控制。

---

## 2. 完整调用栈

读链路分三个阶段：USB 回包先通过预提交的 bulk IN URB 异步进入内核接收缓冲；随后 pyserial 通过 `select` 检查 fd 是否可读；最后通过 `read` 把已经到达 N_TTY 缓冲区的字节复制回用户态。内核外是 SDK → pyserial → CPython → glibc；`read syscall` 的同步调用主链是 VFS → TTY → N_TTY → `copy_from_read_buf` → `copy_to_iter`。异步接收链则是 xHCI → USB core → CDC ACM → TTY flip buffer → N_TTY read_buf。本节用五张调用栈图展示完整路径。

### 2.1 就绪检查阶段：内核外调用栈

```text
PortHandler.readPort(length)
│  SDK 串口封装层：请求最多读 length 字节；timeout=0，所以这次调用不能长期阻塞。
└── serial.Serial.read(size)
    │  pyserial 读逻辑：先 select 判断 fd 是否可读，避免直接 read 空转或阻塞。
    └── select.select([fd, pipe_abort_read_r], [], [], 0)
        │  用户态就绪检查：监视串口 fd 和取消读管道 fd，timeout=0 表示立即返回。
        └── CPython select_select_impl()
            │  把 Python fd 列表转成 fd_set 位图，计算 nfds=max(fd)+1。
            └── glibc select()
                │  libc select 包装：把 timeval {0,0} 转成 timespec {0,0}。
                └── pselect6_time64 / pselect6 syscall
                    │  用户态进入内核：__NR_pselect6=72，传入 fd_set、超时时间和 sigmask=NULL。
                    └── syscall boundary
                        │  从这里进入 Linux 内核。
```

### 2.2 就绪检查阶段：内核内调用栈

```text
Linux SYSCALL_DEFINE6(pselect6, n, inp, outp, exp, tsp, sig)
│  内核 syscall 入口：从用户态读取 sigmask 打包参数和 timespec。
└── do_pselect()
    │  处理 pselect 的时间和信号屏蔽语义，再进入 select 核心。
    └── core_sys_select()
        │  把用户态 fd_set 复制成内核 fd_set_bits，并准备结果位图。
        └── do_select()
            │  遍历被置位的 fd；timeout=0 时只轮询一遍，不睡眠。
            └── vfs_poll(file, wait)
                │  VFS poll 分发：调用 struct file 的 poll 方法。
                └── file->f_op->poll()
                    │  /dev/ttyACM0 的 file_operations 是 tty_fops，所以进入 tty_poll。
                    └── tty_poll(file, wait)
                        │  TTY poll 入口：获取当前 line discipline。
                        └── ld->ops->poll()
                            │  默认行规程 N_TTY 的 poll 方法。
                            └── n_tty_poll(tty, file, wait)
                                │  检查 N_TTY read_buf 是否已有数据；有则返回 EPOLLIN。
```

### 2.3 数据读取阶段：内核外调用栈

```text
select 返回 fd 可读
│  pyserial 确认串口 fd 当前有数据，才进入真正 read。
└── serial.Serial.read()
    │  继续同一个 pyserial read 循环，准备从串口 fd 取 bytes。
    └── os.read(fd, size)
        │  Python 标准库入口：fd=/dev/ttyACM0，size=wait_length-rx_length。
        └── CPython os_read_impl(fd, length)
            │  分配 Python bytes 写入器，准备接收内核复制回来的数据。
            └── _Py_read(fd, buf, count)
                │  释放 GIL，处理 EINTR 重试，然后调用 libc read。
                └── glibc read(fd, buf, count)
                    │  libc syscall 包装：触发 __NR_read=63。
                    └── syscall boundary
                        │  从这里进入 Linux 内核。
```

### 2.4 数据读取阶段：内核内调用栈

```text
Linux SYSCALL_DEFINE3(read, fd, buf, count)
│  内核 syscall 入口：接收 fd、用户态目标 buf、期望读取 count。
└── ksys_read(fd, buf, count)
    │  fdget_pos() + file_ppos()：把 int fd 转成 struct file；TTY 是 stream，通常没有文件偏移。
    └── vfs_read(file, buf, count, pos)
        │  access_ok() + rw_verify_area()：检查用户态目标地址、读权限和读取范围。
        └── new_sync_read(file, buf, count, pos)
            │  init_sync_kiocb() + iov_iter_ubuf()：把用户态目标缓冲包装成 kiocb + iov_iter。
            └── filp->f_op->read_iter(iocb, iter)
                │  /dev/ttyACM0 对应 tty_fops.read_iter。
                └── tty_read(iocb, iter)
                    │  TTY read 入口：获取 tty_struct 和当前 N_TTY 行规程。
                    └── iterate_tty_read(ld, tty, file, iter)
                        │  准备 64 字节 kernel_buf，调用行规程 read。
                        └── n_tty_read(tty, file, kernel_buf, nr, cookie, offset)
                            │  从 N_TTY read_buf 消费已经到达的串口字节。
                            └── copy_from_read_buf()
                                │  memcpy()：从 N_TTY 环形 read_buf 复制到 kernel_buf。
                                └── copy_to_iter(kernel_buf, size, iter)
                                    │  把 kernel_buf 中的字节复制到 CPython 分配的用户态 bytes 缓冲。
```

### 2.5 USB 回包进入内核缓冲区

```text
tty 打开/激活
│  CDC ACM 驱动提前准备接收通道，不等用户态 read() 才提交 USB 请求。
└── acm_port_activate()
    │  提交控制 URB，并批量提交 bulk IN 读 URB。
    └── acm_submit_read_urbs()
        │  遍历接收 URB 池。
        └── acm_submit_read_urb()
            │  每个空闲读 URB 都交给 USB core。
            └── usb_submit_urb()
                │  USB core 校验 URB 并设置 IN 方向。
                └── usb_hcd_submit_urb()
                    │  map_urb_for_dma() 处理 DMA 访问关系。
                    └── xhci_urb_enqueue()
                        │  bulk IN endpoint 也排进 xHCI transfer ring。
                        └── xhci_queue_bulk_tx()
                            │  写 TRB 并敲 doorbell，硬件随后等待设备回包。

舵机回包到达 USB bulk IN endpoint
│  硬件层：STS3215 回包经 USB CDC ACM 数据 IN 端点返回主机。
└── xHCI 完成读 URB
    │  USB 主控把收到的字节 DMA 到 URB transfer_buffer，并产生完成事件。
    └── USB core 调用 urb->complete
        │  USB core 统一回调提交者注册的完成函数。
        └── acm_read_bulk_callback(urb)
            │  CDC ACM 读完成回调：检查状态、统计活跃时间、处理收到的数据，并重新提交读 URB。
            └── acm_process_read_urb(acm, urb)
                │  把 USB 层收到的字节推入 TTY 接收路径。
                ├── tty_insert_flip_string(&acm->port, urb->transfer_buffer, urb->actual_length)
                │   │  复制 urb->transfer_buffer 中的 actual_length 字节到 TTY flip buffer。
                └── tty_flip_buffer_push(&acm->port)
                    │  通知 TTY core 有新输入，调度 line discipline 消费 flip buffer。
                    └── flush_to_ldisc()
                        │  workqueue 中把 flip buffer 交给当前行规程。
                        └── n_tty_receive_buf_common()
                            │  原始模式下基本不解释字节，复制/写入 N_TTY read_buf。
                            └── wake_up_interruptible_poll(&tty->read_wait, EPOLLIN)
                                │  唤醒等待读者；之后 n_tty_poll() 会看到 EPOLLIN。
```

---

## 3. pyserial 与 select 阶段

pyserial 的读逻辑是"先 poll 再 read"：先用 `select` 检查 `/dev/ttyACM0` 是否有数据，确认可读后再调用 `os.read()` 取数据。`timeout=0` 时 `select` 只做一次立即轮询，不会阻塞。本节解释 pyserial 的读循环、CPython 的 `select` 实现，以及 glibc 如何把 `select` 转成 `pselect6` 系统调用。

### 3.1 pyserial `read()`

来源：serial/serialposix.py:read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```python
# venv/lib/python3.13/site-packages/serial/serialposix.py
def read(self, size=1):                           # size 是 SDK 本次想读的最大字节数
    if not self.is_open:                          # 串口必须打开
        raise PortNotOpenError()                  # 未打开则报错
    read = bytearray()                            # 用户态累积缓冲区
    timeout = Timeout(self._timeout)              # self._timeout=0，表示非阻塞
    while len(read) < size:                       # 没读够 size 时循环
        ready, _, _ = select.select(              # 先问内核 fd 是否可读
            [self.fd, self.pipe_abort_read_r],    # 读集合：串口 fd + 取消读的管道 fd
            [],                                   # 写集合为空
            [],                                   # 异常集合为空
            timeout.time_left())                  # timeout=0，立即返回
        if self.pipe_abort_read_r in ready:       # 如果取消读管道就绪
            os.read(self.pipe_abort_read_r, 1000) # 清掉取消信号
            break                                 # 退出
        if not ready:                             # 没有任何 fd 就绪
            break                                 # 非阻塞返回空 bytes
        buf = os.read(self.fd, size - len(read))  # 串口 fd 可读，真正 read()
        if not buf:                               # select 说可读但 read 返回空
            raise SerialException(...)            # 通常表示设备断开或多进程抢占
        read.extend(buf)                          # 累积读到的字节
        if timeout.expired():                     # 超时保护
            break                                 # 退出循环
    return bytes(read)                            # 返回实际读到的字节，可能短于 size
```

`timeout=0` 时，pyserial 使用 `select()` 做一次立即轮询，不让 `read()` 长时间阻塞。

### 3.2 CPython `select.select()`

来源：cpython/Modules/selectmodule.c:select_select_impl（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// cpython/Modules/selectmodule.c
static PyObject *  // 本行参与当前 C 层路径的控制流或数据准备
select_select_impl(PyObject *module, PyObject *rlist, PyObject *wlist,  // 处理 fd 就绪检查和等待唤醒逻辑
                   PyObject *xlist, PyObject *timeout_obj)  // 本行参与当前 C 层路径的控制流或数据准备
{
    pylist rfd2obj[FD_SETSIZE + 1];               // Python 对象和 fd 的映射表
    pylist wfd2obj[FD_SETSIZE + 1];               // 写 fd 映射表，本链路为空
    pylist efd2obj[FD_SETSIZE + 1];               // 异常 fd 映射表，本链路为空
    fd_set ifdset, ofdset, efdset;                // 1024 bit fd 位图
    struct timeval tv, *tvp;                      // 超时时间，timeout=0 时 tv={0,0}
    int imax, omax, emax, max;                    // 最大 fd + 1
    int n;                                        // select 返回值

    rfd2obj[0].sentinel = -1;                     // 初始化读映射表
    wfd2obj[0].sentinel = -1;                     // 初始化写映射表
    efd2obj[0].sentinel = -1;                     // 初始化异常映射表
    imax = seq2set(rlist, &ifdset, rfd2obj);      // 把 [fd, pipe] 转成 ifdset 位图
    omax = seq2set(wlist, &ofdset, wfd2obj);      // 写集合为空，omax=0
    emax = seq2set(xlist, &efdset, efd2obj);      // 异常集合为空，emax=0

    max = imax;                                   // max 是最大监视 fd + 1

    Py_BEGIN_ALLOW_THREADS                        // 释放 GIL
    errno = 0;                                    // 清 errno
    n = select(max, &ifdset, NULL, NULL, tvp);    // 调 libc select()
    Py_END_ALLOW_THREADS                          // 恢复 GIL

    return ret;                                   // 把就绪 fd 转回 Python list
}
```

此时进入 libc 的参数含义：

```text
nfds    = max(self.fd, pipe_abort_read_r) + 1
readfds = bit(self.fd)=1, bit(pipe_abort_read_r)=1
writefds = NULL
exceptfds = NULL
timeout = {tv_sec=0, tv_usec=0}
```

### 3.3 glibc `select()` 到 `pselect6`

来源：glibc-2.42/sysdeps/unix/sysv/linux/select.c:__select64（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// glibc-2.42/sysdeps/unix/sysv/linux/select.c
int  // 本行参与当前 C 层路径的控制流或数据准备
__select64 (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,  // 处理 fd 就绪检查和等待唤醒逻辑
            struct __timeval64 *timeout)  // 定义当前链路涉及的内核数据结构
{
  __time64_t s = timeout != NULL ? timeout->tv_sec : 0; // 秒
  int32_t us = timeout != NULL ? timeout->tv_usec : 0;  // 微秒
  int32_t ns;                                           // 纳秒

  s += us / USEC_PER_SEC;                               // timeval 秒归一化
  us = us % USEC_PER_SEC;                               // 剩余微秒
  ns = us * NSEC_PER_USEC;                              // 微秒转纳秒

  struct __timespec64 ts64, *pts64 = NULL;              // pselect6 使用 timespec
  if (timeout != NULL) {                                // pyserial timeout=0，会传非 NULL
    ts64.tv_sec = s;                                    // 0  // 更新当前层需要传递的状态、长度、指针或错误码
    ts64.tv_nsec = ns;                                  // 0  // 更新当前层需要传递的状态、长度、指针或错误码
    pts64 = &ts64;                                      // 指向栈上 timespec
  }

  int r = SYSCALL_CANCEL(pselect6_time64, nfds, readfds, writefds,  // 处理 fd 就绪检查和等待唤醒逻辑
                         exceptfds, pts64, NULL);       // sigmask=NULL  // 更新当前层需要传递的状态、长度、指针或错误码
  return r;                                             // 返回就绪 fd 数
}
```

对本项目，内核看到的 `pselect6` 参数是：

```text
__NR_pselect6 = 72
n    = max(fd, pipe_fd) + 1
inp  = 用户态 fd_set 地址，里面置位了 ttyACM0 fd 和 pipe fd
outp = NULL
exp  = NULL
tsp  = 用户态 timespec 地址，值为 {0, 0}
sig  = NULL
```

---

## 4. Linux pselect6 与 TTY poll

`pselect6` 进入内核后，VFS 遍历所有被监视的 fd，调用每个文件的 `poll` 方法。对 `/dev/ttyACM0`，最终调用的是 `n_tty_poll()`，它检查 N_TTY 的 `read_buf` 是否有数据。本节解释 Linux select 核心机制，以及 TTY 层如何判断串口 fd 是否可读。

### 4.1 `pselect6` 入口

来源：linux/fs/select.c:SYSCALL_DEFINE pselect6（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/fs/select.c
SYSCALL_DEFINE6(pselect6, int, n, fd_set __user *, inp, fd_set __user *, outp,  // 定义 Linux 系统调用入口，用户态 syscall 会进入这里
                fd_set __user *, exp, struct __kernel_timespec __user *, tsp,  // 本行参与当前 C 层路径的控制流或数据准备
                void __user *, sig)  // 本行参与当前 C 层路径的控制流或数据准备
{
    struct sigset_argpack x = {NULL, 0};          // pselect 的 sigmask 打包参数

    if (get_sigset_argpack(&x, sig))              // sig=NULL 时不读取
        return -EFAULT;                           // sig 指针非法才失败

    return do_pselect(n, inp, outp, exp, tsp, x.p, x.size, PT_TIMESPEC);  // 定义当前层的 C 函数入口
                                                   // 进入 select 核心
}
```

`__user` 表示这些指针指向用户态内存，内核必须用 `copy_from_user()` 间接读取。

### 4.2 `core_sys_select()`

来源：linux/fs/select.c:core_sys_select（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/fs/select.c
int core_sys_select(int n, fd_set __user *inp, fd_set __user *outp,  // 定义当前层的 C 函数入口
                    fd_set __user *exp, struct timespec64 *end_time)  // 本行参与当前 C 层路径的控制流或数据准备
{
    fd_set_bits fds;                              // 内核内部 fd 位图集合
    void *bits;                                   // 位图内存
    size_t size;                                  // 每个位图大小
    long stack_fds[SELECT_STACK_ALLOC/sizeof(long)]; // 小 fd 集合用栈上空间

    size = FDS_BYTES(n);                          // 计算 n 个 fd 需要多少字节
    bits = stack_fds;                             // 默认使用栈上位图

    fds.in      = bits;                           // 输入读集合
    fds.out     = bits + size;                    // 输入写集合
    fds.ex      = bits + 2*size;                  // 输入异常集合
    fds.res_in  = bits + 3*size;                  // 输出读就绪集合
    fds.res_out = bits + 4*size;                  // 输出写就绪集合
    fds.res_ex  = bits + 5*size;                  // 输出异常就绪集合

    get_fd_set(n, inp, fds.in);                   // 从用户态复制 readfds
    get_fd_set(n, outp, fds.out);                 // outp=NULL 时结果为空
    get_fd_set(n, exp, fds.ex);                   // exp=NULL 时结果为空

    zero_fd_set(n, fds.res_in);                   // 清空输出读集合
    zero_fd_set(n, fds.res_out);                  // 清空输出写集合
    zero_fd_set(n, fds.res_ex);                   // 清空输出异常集合

    return do_select(n, &fds, end_time);          // 执行真正轮询
}
```

`fd_set_bits` 是内核内部结构，保存输入 fd 位图和结果 fd 位图。pyserial 只关心读集合。

### 4.3 `do_select()`

来源：linux/fs/select.c:do_select（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/fs/select.c
static int do_select(int n, fd_set_bits *fds, struct timespec64 *end_time)  // 定义当前层的 C 函数入口
{
    struct poll_wqueues table;                    // 保存 poll 等待队列
    poll_table *wait;                             // 传给各 fd 的 poll 回调
    int retval = 0;                               // 就绪 fd 数

    poll_initwait(&table);                        // 初始化等待队列
    wait = &table.pt;                             // 取 poll_table

    if (end_time && !end_time->tv_sec && !end_time->tv_nsec) { // timeout={0,0}  // 检查状态或错误码，决定是否走异常/分支路径
        wait->_qproc = NULL;                      // 不挂等待队列，只立即轮询
    }

    for (;;) {                                    // select 主循环
        for (每一个被置位的 fd) {                 // 遍历 fd_set 中的 bit
            struct fd f = fdget(i);               // fd -> struct file  // 定义当前链路涉及的内核数据结构
            if (fd_file(f)) {                     // fd 有效
                wait_key_set(wait, in, out, bit, 0); // 设置关心的事件类型
                mask = vfs_poll(fd_file(f), wait); // 调具体文件的 poll
                fdput(f);                         // 释放 fd 引用
            }
            if ((mask & POLLIN_SET) && (in & bit)) { // 如果读就绪
                res_in |= bit;                    // 标记结果位图
                retval++;                         // 就绪数加一
            }
        }
        if (retval)                               // 有 fd 就绪
            break;                                // 返回
        if (timeout 已到)                         // pyserial timeout=0 立即到
            break;                                // 返回 0
        poll_schedule_timeout(...);               // 非零超时时才睡眠
    }

    poll_freewait(&table);                        // 释放等待队列资源
    return retval;                                // 返回就绪 fd 数
}
```

对于 `timeout=0`，`do_select()` 不睡眠，只检查一遍当前状态。

### 4.4 `tty_poll()` 与 `n_tty_poll()`

来源：linux/drivers/tty/tty_io.c:tty_poll（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
static __poll_t tty_poll(struct file *filp, poll_table *wait)  // 定义当前层的 C 函数入口
{
    struct tty_struct *tty = file_tty(filp);      // file -> tty_struct  // 定义当前链路涉及的内核数据结构
    struct tty_ldisc *ld;                         // line discipline  // 定义当前链路涉及的内核数据结构
    __poll_t ret = 0;                             // poll 结果掩码

    ld = tty_ldisc_ref_wait(tty);                 // 获取 N_TTY 行规程
    if (!ld)                                      // 如果没有行规程
        return hung_up_tty_poll(filp, wait);      // 按挂起处理
    if (ld->ops->poll)                            // N_TTY 提供 poll
        ret = ld->ops->poll(tty, filp, wait);     // 调 n_tty_poll()
    tty_ldisc_deref(ld);                          // 释放行规程引用
    return ret;                                   // 返回 EPOLLIN 等 bit
}
```

来源：linux/drivers/tty/n_tty.c:n_tty_poll（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/n_tty.c
static __poll_t n_tty_poll(struct tty_struct *tty, struct file *file,  // 定义当前层的 C 函数入口
                           poll_table *wait)  // 处理 fd 就绪检查和等待唤醒逻辑
{
    __poll_t mask = 0;                            // 就绪事件掩码

    poll_wait(file, &tty->read_wait, wait);       // 注册读等待队列
    poll_wait(file, &tty->write_wait, wait);      // 注册写等待队列

    if (input_available_p(tty, 1))                // N_TTY read_buf 中有可读数据
        mask |= EPOLLIN | EPOLLRDNORM;            // 返回读就绪
    else {                                        // 暂时没看到数据
        tty_buffer_flush_work(tty->port);         // 推动 flip buffer 工作处理
        if (input_available_p(tty, 1))            // 再检查一次
            mask |= EPOLLIN | EPOLLRDNORM;        // 有数据则读就绪
    }

    if (tty->ops->write && !tty_is_writelocked(tty) &&  // 检查状态或错误码，决定是否走异常/分支路径
        tty_chars_in_buffer(tty) < WAKEUP_CHARS &&  // 调用 TTY 层接口处理串口设备语义
        tty_write_room(tty) > 0)                  // 写侧也可写时
        mask |= EPOLLOUT | EPOLLWRNORM;           // 返回写就绪

    return mask;                                  // pyserial 只关心 EPOLLIN
}
```

如果 `n_tty_poll()` 返回 `EPOLLIN`，CPython `select.select()` 会把串口 fd 放进 `ready` 列表，pyserial 随后调用 `os.read()`。

---

## 5. USB CDC ACM 接收路径

与 write 不同，读 URB 是在串口打开时就预提交给 USB core 的。舵机回包到达 USB bulk IN endpoint 后，xHCI 通过 DMA 把收到的数据写入 URB 接收缓冲，并触发 CDC ACM 驱动的完成回调。回调把数据推入 TTY flip buffer，再经 N_TTY 行规程放入 `read_buf`。只有数据进入 `read_buf` 后，上层的 `select` 和 `read` 才能看到它。本节解释数据从 USB 线到内核缓冲区的完整路径。

### 5.1 打开 tty 时预提交读 URB

来源：linux/drivers/usb/class/cdc-acm.c:acm_port_activate（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static int acm_port_activate(struct tty_port *port, struct tty_struct *tty)  // 定义当前层的 C 函数入口
{
    struct acm *acm = container_of(port, struct acm, port); // tty_port -> acm  // 定义当前链路涉及的内核数据结构

    acm->ctrlurb->dev = acm->dev;                 // 控制中断 URB 绑定 USB 设备
    usb_submit_urb(acm->ctrlurb, GFP_KERNEL);     // 提交控制状态 URB

    clear_bit(ACM_THROTTLED, &acm->flags);        // 确保未被流控暂停
    acm_submit_read_urbs(acm, GFP_KERNEL);        // 提交所有 bulk IN 读 URB
    return 0;                                     // 激活完成
}
```

### 5.2 提交读 URB

来源：linux/drivers/usb/class/cdc-acm.c:acm_submit_read_urb（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static int acm_submit_read_urb(struct acm *acm, int index, gfp_t mem_flags)  // 定义当前层的 C 函数入口
{
    int res;                                      // usb_submit_urb 返回值

    if (!test_and_clear_bit(index, &acm->read_urbs_free)) // 该 URB 不空闲
        return 0;                                 // 已经提交过，不重复提交

    res = usb_submit_urb(acm->read_urbs[index], mem_flags); // 提交 bulk IN URB
    if (res) {                                    // 提交失败
        set_bit(index, &acm->read_urbs_free);     // 标回空闲
        return res;                               // 返回错误
    }

    return 0;                                     // 成功提交
}

static int acm_submit_read_urbs(struct acm *acm, gfp_t mem_flags)  // 定义当前层的 C 函数入口
{
    int i;                                        // URB 下标

    for (i = 0; i < acm->rx_buflimit; ++i)        // 遍历所有接收 URB
        acm_submit_read_urb(acm, i, mem_flags);   // 逐个提交给 USB core

    return 0;                                     // 所有读 URB 已提交
}
```

CDC ACM 的读不是用户调用 `read()` 时才向 USB 设备要数据，而是驱动提前把 bulk IN URB 提交给 USB core。舵机回包到达时，USB 主控把数据填进 URB 的 `transfer_buffer`。

这些接收 URB 的缓冲区在设备初始化时已经通过 `usb_alloc_coherent()` 分配，并设置了 `URB_NO_TRANSFER_DMA_MAP`。因此 USB core 提交读 URB 时主要处理 DMA 访问关系，不会再额外分配一块普通缓冲来复制舵机回包。

来源：linux/drivers/usb/class/cdc-acm.c:相关代码片段（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
rb->base = usb_alloc_coherent(acm->dev, readsize, GFP_KERNEL, &rb->dma);  // 更新当前层需要传递的状态、长度、指针或错误码
urb->transfer_flags |= URB_NO_TRANSFER_DMA_MAP;  // 更新当前层需要传递的状态、长度、指针或错误码
urb->transfer_dma = rb->dma;  // 更新当前层需要传递的状态、长度、指针或错误码
usb_fill_bulk_urb(urb, acm->dev, acm->in, rb->base,  // 调用下一层 C 函数继续完成当前路径
                  acm->readsize, acm_read_bulk_callback, rb);  // 调用下一层 C 函数继续完成当前路径
```

### 5.3 读 URB 进入 USB core/xHCI 队列

`acm_submit_read_urb()` 调用的 `usb_submit_urb()` 和写路径共用 USB core/HCD/xHCI 入队逻辑。区别是这里的 endpoint 是 bulk IN，URB 被预先排到 xHCI transfer ring 后，硬件等待设备返回数据；后面的 `read()` 系统调用只是消费已经到达 TTY/N_TTY 缓冲区的数据，不会在当下才创建这条 USB 传输。

来源：linux/drivers/usb/core/urb.c:usb_submit_urb（节选：仅保留 bulk IN 读 URB 相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/core/urb.c
int usb_submit_urb(struct urb *urb, gfp_t mem_flags) // USB core 接收驱动提交的异步传输请求
{
    int xfertype, max;                            // endpoint 类型和最大包长
    struct usb_device *dev;                       // URB 绑定的 USB 设备
    struct usb_host_endpoint *ep;                 // pipe 对应的 endpoint
    int is_out;                                   // 方向标记；bulk IN 读 URB 为 0
    unsigned int allowed;                         // 当前传输类型允许的 URB 标志

    if (!urb || !urb->complete)                   // URB 必须存在并带完成回调
        return -EINVAL;                           // 参数错误，不能提交
    if (urb->hcpriv) {                            // hcpriv 非空表示 URB 已经在 HCD 手里
        WARN_ONCE(1, "URB %pK submitted while active\n", urb); // 警告重复提交
        return -EBUSY;                            // 返回忙
    }

    dev = urb->dev;                               // 取出 USB 设备
    if ((!dev) || (dev->state < USB_STATE_UNAUTHENTICATED)) // 设备不存在或状态太早
        return -ENODEV;                           // 设备不可用

    ep = usb_pipe_endpoint(dev, urb->pipe);       // 从 pipe 找到 bulk IN endpoint
    if (!ep)                                      // endpoint 查不到
        return -ENOENT;                           // 返回不存在

    urb->ep = ep;                                 // 把 endpoint 缓存在 URB 上
    urb->status = -EINPROGRESS;                   // 标记传输已经进入进行中状态
    urb->actual_length = 0;                       // 完成前实际接收长度清零

    xfertype = usb_endpoint_type(&ep->desc);      // 读取 endpoint 类型
    if (xfertype == USB_ENDPOINT_XFER_CONTROL) {  // 控制端点分支
        /* CDC ACM 数据读 URB 是 bulk IN，不走控制端点分支。 */ // 节选说明
    } else {                                      // 非控制端点
        is_out = usb_endpoint_dir_out(&ep->desc); // bulk IN endpoint 得到 is_out=0
    }

    urb->transfer_flags &= ~(URB_DIR_MASK | URB_DMA_MAP_SINGLE | // 清掉上次提交遗留的方向和 DMA 标志
            URB_DMA_MAP_PAGE | URB_DMA_MAP_SG | URB_MAP_LOCAL | // 清掉 page/sg/local 映射标志
            URB_SETUP_MAP_SINGLE | URB_SETUP_MAP_LOCAL |        // 清掉控制包映射标志
            URB_DMA_SG_COMBINED);                // 清掉合并 SG 标志
    urb->transfer_flags |= (is_out ? URB_DIR_OUT : URB_DIR_IN); // bulk IN 读 URB 设置 URB_DIR_IN

    if (xfertype != USB_ENDPOINT_XFER_CONTROL && // 非控制端点
            dev->state < USB_STATE_CONFIGURED)   // 设备必须已配置
        return -ENODEV;                           // 未配置设备不能提交数据 URB

    max = usb_endpoint_maxp(&ep->desc);           // 读取 endpoint 最大包长
    if (max <= 0)                                 // endpoint 描述符异常
        return -EMSGSIZE;                         // 包长非法

    allowed = (URB_NO_TRANSFER_DMA_MAP | URB_NO_INTERRUPT | URB_DIR_MASK | // 基础允许标志
            URB_FREE_BUFFER);                     // 允许完成时释放缓冲
    switch (xfertype) {                           // 按传输类型补充允许标志
    case USB_ENDPOINT_XFER_BULK:                  // CDC ACM 读数据端点是 bulk
    case USB_ENDPOINT_XFER_INT:                   // interrupt 和 bulk 共享部分规则
        if (is_out)                               // OUT 方向才允许 ZERO_PACKET
            allowed |= URB_ZERO_PACKET;           // bulk IN 不会加这个标志
        fallthrough;                              // 继续执行非 isoc 默认规则
    default:                                      // 所有非 isochronous endpoint
        if (!is_out)                              // IN 方向
            allowed |= URB_SHORT_NOT_OK;          // 允许驱动要求短包视为错误
        break;                                    // 规则处理结束
    }
    allowed &= urb->transfer_flags;               // 只保留当前 URB 实际设置且允许的标志

    return usb_hcd_submit_urb(urb, mem_flags);    // 交给 Host Controller Driver，树莓派 5 上进入 xHCI
}
```

来源：linux/drivers/usb/core/hcd.c:usb_hcd_submit_urb（节选：仅保留 bulk IN 读 URB 相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/core/hcd.c
int usb_hcd_submit_urb(struct urb *urb, gfp_t mem_flags) // USB core 到 HCD 的提交入口
{
    int status;                                   // HCD 提交结果
    struct usb_hcd *hcd = bus_to_hcd(urb->dev->bus); // 从 USB bus 找到主机控制器

    usb_get_urb(urb);                             // 增加 URB 引用，HCD 完成前不能释放
    atomic_inc(&urb->use_count);                  // 标记 URB 正在被 HCD 使用
    atomic_inc(&urb->dev->urbnum);                // 设备当前活跃 URB 数加一
    usbmon_urb_submit(&hcd->self, urb);           // 给 usbmon 记录提交事件

    status = map_urb_for_dma(hcd, urb, mem_flags); // 建立或确认 DMA 访问关系
    if (likely(status == 0)) {                    // DMA 关系处理成功
        status = hcd->driver->urb_enqueue(hcd, urb, mem_flags); // 调 HCD 的 urb_enqueue，xHCI 下就是 xhci_urb_enqueue()
        if (unlikely(status))                     // HCD 入队失败
            unmap_urb_for_dma(hcd, urb);          // 回滚 DMA 映射关系
    }

    if (unlikely(status)) {                       // 提交流程失败
        usbmon_urb_submit_error(&hcd->self, urb, status); // 给 usbmon 记录错误
        urb->hcpriv = NULL;                       // 清掉 HCD 私有指针
        INIT_LIST_HEAD(&urb->urb_list);           // 重置 URB 链表节点
        atomic_dec(&urb->use_count);              // 撤销使用计数
    }

    return status;                                // 0 表示读 URB 已交给 HCD，完成会异步回调
}
```

读 URB 的 DMA 映射分发和写 URB 走同一套函数，只是方向变成 `DMA_FROM_DEVICE`。CDC ACM 读缓冲同样来自 `usb_alloc_coherent()` 并设置了 `URB_NO_TRANSFER_DMA_MAP`，因此这里确认/沿用已有 DMA 地址，而不是在用户态 `read()` 时再复制一份 USB 接收缓冲。

来源：linux/drivers/usb/core/hcd.c:map_urb_for_dma / linux/drivers/usb/host/xhci.c:xhci_map_urb_for_dma / linux/drivers/usb/core/hcd.c:usb_hcd_map_urb_for_dma（节选：仅保留 CDC ACM 读链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/core/hcd.c
static int map_urb_for_dma(struct usb_hcd *hcd, struct urb *urb, // USB core 调用的 DMA 映射分发入口
                           gfp_t mem_flags)      // 提交读 URB 时通常是 GFP_KERNEL 或 GFP_ATOMIC
{
    if (hcd->driver->map_urb_for_dma)             // xHCI HCD 注册了自己的映射钩子
        return hcd->driver->map_urb_for_dma(hcd, urb, mem_flags); // 先进入 xhci_map_urb_for_dma()
    else                                          // 没有 HCD 私有钩子
        return usb_hcd_map_urb_for_dma(hcd, urb, mem_flags); // 直接走通用映射
}

// linux/drivers/usb/host/xhci.c
static int xhci_map_urb_for_dma(struct usb_hcd *hcd, struct urb *urb, // xHCI 的 DMA 映射钩子
                                gfp_t mem_flags)  // 原样传给通用映射
{
    struct xhci_hcd *xhci;                        // xHCI 主控私有结构

    xhci = hcd_to_xhci(hcd);                      // 从通用 HCD 取 xHCI 对象

    if (xhci_urb_suitable_for_idt(urb))           // IDT 只适用于很小的 OUT 传输
        return 0;                                 // bulk IN 读 URB 不会走这个返回

    if (xhci->quirks & XHCI_SG_TRB_CACHE_SIZE_QUIRK) { // 少数 xHCI 需要临时缓冲规避 SG/TRB 缓存限制
        if (xhci_urb_temp_buffer_required(hcd, urb)) // 判断是否需要 bounce buffer
            return xhci_map_temp_buffer(hcd, urb); // 需要时先分配临时缓冲
    }
    return usb_hcd_map_urb_for_dma(hcd, urb, mem_flags); // 常规 bulk IN 读 URB 进入通用映射
}

// linux/drivers/usb/core/hcd.c
int usb_hcd_map_urb_for_dma(struct usb_hcd *hcd, struct urb *urb, // 通用 HCD DMA 映射函数
                            gfp_t mem_flags)     // 映射所需的内存分配上下文
{
    enum dma_data_direction dir;                  // DMA 方向
    int ret = 0;                                  // 默认成功

    dir = usb_urb_dir_in(urb) ? DMA_FROM_DEVICE : DMA_TO_DEVICE; // bulk IN 读 URB 是 DMA_FROM_DEVICE
    if (urb->transfer_buffer_length != 0          // 有接收缓冲才需要处理映射
        && !(urb->transfer_flags & URB_NO_TRANSFER_DMA_MAP)) { // CDC ACM 读 URB 设置了该标志，所以跳过重映射分支
        /* 通用 sg/page/single 映射分支在这里；本链路使用 coherent 缓冲，不进入。 */ // 节选说明
    }
    return ret;                                   // 返回 0 后继续交给 xHCI urb_enqueue
}
```

来源：linux/drivers/usb/host/xhci.c:xhci_urb_enqueue（节选：仅保留 bulk IN 读 URB 相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/host/xhci.c
static int xhci_urb_enqueue(struct usb_hcd *hcd, struct urb *urb, gfp_t mem_flags) // xHCI 接收 URB 的入口
{
    struct xhci_hcd *xhci = hcd_to_xhci(hcd);     // 从通用 HCD 取 xHCI 私有结构
    unsigned long flags;                          // 保存中断标志
    int ret = 0;                                  // 入队结果
    unsigned int slot_id, ep_index;               // USB 设备 slot 和 endpoint 索引
    struct urb_priv *urb_priv;                    // xHCI 跟踪 URB/TD 的私有结构
    int num_tds;                                  // 本 URB 需要的 TD 数量

    ep_index = xhci_get_endpoint_index(&urb->ep->desc); // 根据 endpoint 描述符计算 ep_index
    num_tds = 1;                                  // 普通 bulk IN 读 URB 通常一个 TD
    urb_priv = kzalloc(struct_size(urb_priv, td, num_tds), mem_flags); // 分配 xHCI 私有跟踪结构
    if (!urb_priv)                                // 分配失败
        return -ENOMEM;                           // 返回内存不足

    urb_priv->num_tds = num_tds;                  // 记录 TD 数量
    urb_priv->num_tds_done = 0;                   // 完成 TD 计数清零
    urb->hcpriv = urb_priv;                       // 把 xHCI 私有结构挂到 URB 上

    spin_lock_irqsave(&xhci->lock, flags);        // 锁住 xHCI 状态和 transfer ring
    ret = xhci_check_args(hcd, urb->dev, urb->ep, true, true, __func__); // 检查设备和 endpoint 是否可用
    if (ret <= 0) {                               // 参数或状态非法
        ret = ret ? ret : -EINVAL;                // 规范化错误码
        goto free_priv;                           // 释放私有结构并返回
    }

    slot_id = urb->dev->slot_id;                  // 读取 xHCI slot id
    switch (usb_endpoint_type(&urb->ep->desc)) {  // 按 endpoint 类型分发
    case USB_ENDPOINT_XFER_BULK:                  // CDC ACM bulk IN 进入这个分支
        ret = xhci_queue_bulk_tx(xhci, GFP_ATOMIC, urb, slot_id, ep_index); // 把读 URB 转成 bulk transfer TRB
        break;                                    // bulk 分支结束
    }

    if (ret) {                                    // 入队失败
free_priv:
        xhci_urb_free_priv(urb_priv);             // 释放 xHCI 私有结构
        urb->hcpriv = NULL;                       // 清掉 URB 私有指针
    }
    spin_unlock_irqrestore(&xhci->lock, flags);   // 释放 xHCI 锁
    return ret;                                   // 0 表示 URB 已排进 xHCI transfer ring
}
```

来源：linux/drivers/usb/host/xhci-ring.c:xhci_queue_bulk_tx（节选：仅保留 bulk IN 读 URB 相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/host/xhci-ring.c
int xhci_queue_bulk_tx(struct xhci_hcd *xhci, gfp_t mem_flags, // 把 bulk URB 写入 xHCI transfer ring
        struct urb *urb, int slot_id, unsigned int ep_index) // URB、设备 slot 和 endpoint 索引
{
    struct xhci_ring *ring;                       // endpoint 对应的 transfer ring
    struct urb_priv *urb_priv;                    // 前面分配的 URB 私有结构
    struct xhci_td *td;                           // 当前 Transfer Descriptor
    struct xhci_generic_trb *start_trb;           // 本 TD 的第一个 TRB
    bool more_trbs_coming = true;                 // 是否还有后续 TRB
    bool first_trb = true;                        // 首个 TRB 的 cycle bit 延后交给硬件
    unsigned int start_cycle, enqd_len, trb_buff_len, full_len; // 首 TRB cycle、已入队长度、本 TRB 长度、总长度
    unsigned int num_trbs;                       // 这个 URB 需要写入的 TRB 数量
    u32 field, length_field, remainder;           // TRB 控制字段和长度字段
    u64 addr, send_addr;                          // DMA 地址
    int ret;                                      // prepare_transfer 返回值

    ring = xhci_urb_to_transfer_ring(xhci, urb);  // 找到 bulk IN endpoint 的 transfer ring
    if (!ring)                                    // ring 不存在
        return -EINVAL;                           // endpoint 状态异常

    full_len = urb->transfer_buffer_length;       // 本次读 URB 可接收的最大字节数
    num_trbs = count_trbs_needed(urb);            // 根据接收缓冲长度和边界计算需要的 TRB 数
    addr = (u64) urb->transfer_dma;               // CDC ACM 读缓冲的 DMA 地址
    send_addr = addr;                             // 当前 TRB 要写入的 DMA 地址
    ret = prepare_transfer(xhci, xhci->devs[slot_id], // 为 endpoint ring 预留 TRB 空间
            ep_index, urb->stream_id,             // 指定 endpoint 和 stream
            num_trbs, urb, 0, mem_flags);         // 按 TRB 数预留一个 TD 的 ring 空间
    if (unlikely(ret < 0))                        // ring 空间准备失败
        return ret;                               // 返回 HCD 错误码

    urb_priv = urb->hcpriv;                       // 取 xHCI 私有结构
    td = &urb_priv->td[0];                        // 普通 bulk URB 的第一个 TD
    start_trb = &ring->enqueue->generic;          // 保存第一个 TRB 地址
    start_cycle = ring->cycle_state;              // 保存首 TRB 当前 cycle，等所有 TRB 写完后再交给硬件

    for (enqd_len = 0; first_trb || enqd_len < full_len; // 遍历，把整个接收缓冲描述成一个或多个 TRB
            enqd_len += trb_buff_len) {           // 每轮推进一个 TRB 长度
        field = TRB_TYPE(TRB_NORMAL);             // bulk 传输使用 Normal TRB
        trb_buff_len = TRB_BUFF_LEN_UP_TO_BOUNDARY(addr); // 单个 TRB 不能跨 64KB 边界
        if (enqd_len + trb_buff_len > full_len)   // 最后一段不能超过 URB 总长度
            trb_buff_len = full_len - enqd_len;   // 截断成剩余长度

        if (first_trb) {                          // 第一个 TRB
            first_trb = false;                    // 后续循环不再是首 TRB
        } else {                                  // 后续 TRB
            field |= ring->cycle_state;           // 非首 TRB 直接带当前 cycle 状态
        }

        if (enqd_len + trb_buff_len >= full_len) { // 最后一个 TRB
            field &= ~TRB_CHAIN;                  // 清掉 CHAIN，表示 TD 结束
            field |= TRB_IOC;                     // 完成时产生事件
            more_trbs_coming = false;             // 没有后续 TRB
            td->last_trb = ring->enqueue;         // 记录 TD 最后一个 TRB
            td->last_trb_seg = ring->enq_seg;     // 记录最后 TRB 所在 segment
        }

        if (usb_urb_dir_in(urb))                  // bulk IN 读 URB
            field |= TRB_ISP;                     // 短包也产生事件，便于收到实际长度后完成

        remainder = xhci_td_remainder(xhci, enqd_len, trb_buff_len, // 计算 TD 剩余量
                                      full_len, urb, more_trbs_coming); // 供控制器调度和事件生成使用
        length_field = TRB_LEN(trb_buff_len) |    // 当前 TRB 可接收长度
            TRB_TD_SIZE(remainder) |              // TD 剩余大小提示
            TRB_INTR_TARGET(0);                   // 完成事件发给 interrupter 0

        queue_trb(xhci, ring, more_trbs_coming,   // 把 TRB 写入 transfer ring
                  lower_32_bits(send_addr),       // buffer DMA 地址低 32 位
                  upper_32_bits(send_addr),       // buffer DMA 地址高 32 位
                  length_field,                   // 长度和中断目标字段
                  field);                         // 类型、cycle、IOC、ISP 等控制位
        addr += trb_buff_len;                     // 推进下一个 TRB 的 DMA 地址
        send_addr = addr;                         // 更新下一轮写入地址
    }

    giveback_first_trb(xhci, slot_id, ep_index, urb->stream_id, // 交出首 TRB 并敲 doorbell
            start_cycle, start_trb);              // 通知 xHCI 硬件这个 bulk IN URB 可以执行
    return 0;                                     // 入队完成；后续由完成事件触发 acm_read_bulk_callback()
}
```

### 5.4 URB 完成回调

来源：linux/drivers/usb/class/cdc-acm.c:acm_read_bulk_callback（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static void acm_read_bulk_callback(struct urb *urb)  // 定义当前层的 C 函数入口
{
    struct acm_rb *rb = urb->context;             // 接收缓冲描述符
    struct acm *acm = rb->instance;               // 所属 CDC ACM 设备
    int status = urb->status;                     // USB 传输状态

    switch (status) {                             // 按完成状态处理
    case 0:                                       // 正常完成
        usb_mark_last_busy(acm->dev);             // 更新 USB 自动电源管理活跃时间
        acm_process_read_urb(acm, urb);           // 把 URB 数据推给 TTY
        break;                                    // 完成正常路径
    case -EPIPE:                                  // endpoint stall  // 处理当前 switch 命中的具体命令分支
        set_bit(EVENT_RX_STALL, &acm->flags);     // 标记需要清 halt
        return;                                   // 暂停后续提交
    }

    if (test_bit(ACM_THROTTLED, &acm->flags))     // 如果 TTY 已经节流
        return;                                   // 不继续提交读 URB

    acm_submit_read_urb(acm, rb->index, GFP_ATOMIC); // 重新提交同一个 URB，继续接收后续数据
}
```

### 5.5 推入 TTY flip buffer

来源：linux/drivers/usb/class/cdc-acm.c:acm_process_read_urb（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/usb/class/cdc-acm.c
static void acm_process_read_urb(struct acm *acm, struct urb *urb)  // 定义当前层的 C 函数入口
{
    unsigned long flags;                          // 保存中断标志

    if (!urb->actual_length)                      // 本次 URB 没有数据
        return;                                   // 直接返回

    spin_lock_irqsave(&acm->read_lock, flags);    // 保护接收路径
    tty_insert_flip_string(&acm->port,            // 把 USB 收到的字节插入 TTY flip buffer
                           urb->transfer_buffer,  // 源数据：USB URB 缓冲
                           urb->actual_length);   // 字节数：实际收到长度
    spin_unlock_irqrestore(&acm->read_lock, flags); // 释放接收锁

    tty_flip_buffer_push(&acm->port);             // 通知 TTY 核心处理新数据
}
```

这一步把 USB 层收到的舵机状态包变成 TTY 层可读数据。后续 N_TTY 的 receive buffer 有数据，`n_tty_poll()` 才会返回 `EPOLLIN`。

`tty_insert_flip_string()` 是 CDC ACM 到 TTY flip buffer 的 CPU 复制点。它在头文件中是 inline 包装，最终进入 `__tty_insert_flip_string_flags()`，该函数按 TTY buffer 可用空间分段 `memcpy()`。

来源：linux/include/linux/tty_flip.h:tty_insert_flip_string / linux/drivers/tty/tty_buffer.c:__tty_insert_flip_string_flags（节选：仅保留普通字节 TTY_NORMAL 路径，已按当前仓库源码核对）
```c
// linux/include/linux/tty_flip.h
static inline size_t tty_insert_flip_string(struct tty_port *port, // CDC ACM 调用的普通字节入队接口
                                            const u8 *chars, size_t size) // chars 指向 URB transfer_buffer
{
    return tty_insert_flip_string_fixed_flag(port, chars, TTY_NORMAL, size); // 普通串口字节统一标成 TTY_NORMAL
}

static inline size_t tty_insert_flip_string_fixed_flag(struct tty_port *port, // 带固定 flag 的 flip buffer 入队接口
                                                       const u8 *chars, u8 flag, // flag 对所有字符相同
                                                       size_t size) // 要复制的字节数
{
    return __tty_insert_flip_string_flags(port, chars, &flag, false, size); // 进入真正复制函数，flags 不随字符变化
}

// linux/drivers/tty/tty_buffer.c
size_t __tty_insert_flip_string_flags(struct tty_port *port, const u8 *chars, // 把驱动收到的字节复制进 flip buffer
                                      const u8 *flags, bool mutable_flags, // flags 描述每个字节的状态
                                      size_t size) // 目标复制长度
{
    bool need_flags = mutable_flags || flags[0] != TTY_NORMAL; // 普通字节不需要额外 flag buffer
    size_t copied = 0;                           // 已复制字节数

    do {                                         // 可能跨多个 tty_buffer 分段复制
        size_t goal = min_t(size_t, size - copied, TTY_BUFFER_PAGE); // 本轮最多申请一页大小空间
        size_t space = __tty_buffer_request_room(port, goal, need_flags); // 向 flip buffer 申请可写空间
        struct tty_buffer *tb = port->buf.tail;  // 当前写入的 flip buffer

        if (unlikely(space == 0))                // 没有可用空间
            break;                               // 停止复制，返回已复制长度

        memcpy(char_buf_ptr(tb, tb->used), chars, space); // URB transfer_buffer -> TTY flip buffer

        if (mutable_flags) {                     // 每个字节有独立 flag
            memcpy(flag_buf_ptr(tb, tb->used), flags, space); // 同步复制 flag 数组
            flags += space;                      // 推进 flag 源指针
        } else if (tb->flags) {                  // 当前 buffer 分配了 flag 区
            memset(flag_buf_ptr(tb, tb->used), flags[0], space); // 写入固定 flag
        } else {                                 // 普通 TTY_NORMAL 且无 flag 区
            WARN_ON_ONCE(need_flags);            // 理论上不需要 flag，却发现状态不一致时报警
        }

        tb->used += space;                       // 推进 flip buffer 已用长度
        copied += space;                         // 累加本次已经复制的字节数
        chars += space;                          // 推进 URB 缓冲源指针
    } while (unlikely(size > copied));            // 还有未复制字节时继续申请下一个 buffer

    return copied;                               // 返回实际进入 flip buffer 的字节数
}
```

### 5.6 从 flip buffer 到 N_TTY read_buf

`tty_flip_buffer_push()` 不是直接让用户态 `read()` 读取 URB 缓冲，而是把 TTY flip buffer 的处理工作放到 workqueue。`flush_to_ldisc()` 之后会把 flip buffer 中的数据交给当前行规程；本项目通常仍使用 N_TTY 行规程，并处于原始模式，因此 N_TTY 基本不解释舵机字节，只把它们写入自己的 `read_buf`，随后唤醒等待在 `read_wait` 上的读者。

来源：linux/drivers/tty/tty_buffer.c:tty_flip_buffer_push（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_buffer.c
void tty_flip_buffer_push(struct tty_port *port)  // 定义当前层的 C 函数入口
{
    struct tty_bufhead *buf = &port->buf;  // 定义当前链路涉及的内核数据结构

    tty_flip_buffer_commit(buf->tail);             // 提交 flip buffer 中的新字节
    queue_work(system_unbound_wq, &buf->work);     // 调度 flush_to_ldisc()
}

static void flush_to_ldisc(struct work_struct *work)  // 定义当前层的 C 函数入口
{
    struct tty_port *port = container_of(work, struct tty_port, buf.work);  // 定义当前链路涉及的内核数据结构
    struct tty_buffer *head = port->buf.head;  // 定义当前链路涉及的内核数据结构
    size_t count = smp_load_acquire(&head->commit) - head->read;  // 更新当前层需要传递的状态、长度、指针或错误码

    receive_buf(port, head, count);                // 把 flip buffer 交给 line discipline
}

static size_t tty_ldisc_receive_buf(struct tty_ldisc *ld, const u8 *p,  // 定义当前层的 C 函数入口
                                    const u8 *f, size_t count)  // 本行参与当前 C 层路径的控制流或数据准备
{
    if (ld->ops->receive_buf2)  // 检查状态或错误码，决定是否走异常/分支路径
        count = ld->ops->receive_buf2(ld->tty, p, f, count); // N_TTY receive_buf2  // 更新当前层需要传递的状态、长度、指针或错误码
    return count;  // 把本层处理结果或错误码返回上一层
}
```

来源：linux/drivers/tty/n_tty.c:n_tty_receive_buf_common（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/n_tty.c
static size_t n_tty_receive_buf_common(struct tty_struct *tty, const u8 *cp,  // 定义当前层的 C 函数入口
                                       const u8 *fp, size_t count, bool flow)  // 本行参与当前 C 层路径的控制流或数据准备
{
    struct n_tty_data *ldata = tty->disc_data;  // 定义当前链路涉及的内核数据结构

    down_read(&tty->termios_rwsem);                // 读取 termios 配置
    __receive_buf(tty, cp, fp, count);             // 原始模式下把字节写入 read_buf
    smp_store_release(&ldata->commit_head, ldata->read_head); // 发布给 read/poll 侧
    wake_up_interruptible_poll(&tty->read_wait, EPOLLIN | EPOLLRDNORM);  // 处理 fd 就绪检查和等待唤醒逻辑
    up_read(&tty->termios_rwsem);  // 调用下一层 C 函数继续完成当前路径

    return count;  // 把本层处理结果或错误码返回上一层
}
```

这一段会引入从 TTY flip buffer 到 N_TTY `read_buf` 的内核内复制或逐字节入队。它发生在 USB 完成回调之后、用户态 `select()` 看到可读之前。

---

## 6. read 系统调用取数据

pyserial 先用 `select` 确认 fd 可读，然后调用 `os.read()` 把数据从内核取回 Python。本章追踪 `read` 系统调用从 CPython 到 glibc、再到 Linux VFS、TTY 层、N_TTY 行规程的完整路径，解释内核如何把已经到达 `read_buf` 的字节复制到用户态。

### 6.1 CPython 与 glibc

来源：cpython/Modules/posixmodule.c:os_read_impl / cpython/Python/fileutils.c:_Py_read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// cpython/Modules/posixmodule.c
static PyObject *  // 本行参与当前 C 层路径的控制流或数据准备
os_read_impl(PyObject *module, int fd, Py_ssize_t length)  // 调用下一层 C 函数继续完成当前路径
{
    PyBytesWriter *writer = PyBytesWriter_Create(length); // 分配 Python bytes 写入器
    Py_ssize_t n = _Py_read(fd, PyBytesWriter_GetData(writer), length); // 调 _Py_read
    return PyBytesWriter_FinishWithSize(writer, n);       // 按实际读到字节数构造 bytes
}

// cpython/Python/fileutils.c
Py_ssize_t  // 本行参与当前 C 层路径的控制流或数据准备
_Py_read(int fd, void *buf, size_t count)  // 调用下一层 C 函数继续完成当前路径
{
    Py_ssize_t n;                                // read 返回值
    int err;                                     // errno  // 本行参与当前 C 层路径的控制流或数据准备

    do {                                        // EINTR 重试循环
        Py_BEGIN_ALLOW_THREADS                   // 释放 GIL
        errno = 0;                               // 清 errno
        n = read(fd, buf, count);                // 调 libc read(fd, buf, count)
        err = errno;                             // 保存 errno
        Py_END_ALLOW_THREADS                     // 恢复 GIL
    } while (n < 0 && err == EINTR &&            // 信号打断
             !PyErr_CheckSignals());             // Python 信号处理器未抛异常则重试

    return n;                                    // 返回读到字节数或 -1
}
```

来源：glibc-2.42/sysdeps/unix/sysv/linux/read.c:__libc_read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// glibc-2.42/sysdeps/unix/sysv/linux/read.c
ssize_t  // 本行参与当前 C 层路径的控制流或数据准备
__libc_read (int fd, void *buf, size_t nbytes)  // 调用下一层 C 函数继续完成当前路径
{
  return SYSCALL_CANCEL (read, fd, buf, nbytes); // 发 read 系统调用
}
weak_alias (__libc_read, read)                   // read 是 __libc_read 的别名
```

### 6.2 Linux VFS read

来源：linux/fs/read_write.c:SYSCALL_DEFINE read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/fs/read_write.c
SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)  // 定义 Linux 系统调用入口，用户态 syscall 会进入这里
{
    return ksys_read(fd, buf, count);             // 进入通用 read 逻辑
}

ssize_t ksys_read(unsigned int fd, char __user *buf, size_t count)  // 定义当前层的 C 函数入口
{
    struct fd f = fdget_pos(fd);                  // fd -> struct file  // 定义当前链路涉及的内核数据结构
    ssize_t ret = -EBADF;                         // 默认 fd 错误

    if (fd_file(f)) {                             // fd 有效
        loff_t pos, *ppos = file_ppos(fd_file(f)); // tty 是 stream，ppos 通常 NULL
        if (ppos) {                               // 普通文件才需要维护文件偏移
            pos = *ppos;                          // 保存原始偏移
            ppos = &pos;                          // 使用临时偏移变量
        }
        ret = vfs_read(fd_file(f), buf, count, ppos); // 调 VFS read
        if (ret >= 0 && ppos)                     // 普通文件读取成功后
            fd_file(f)->f_pos = pos;              // 回写文件偏移
        fdput_pos(f);                             // 释放 fd 引用
    }
    return ret;                                   // 返回读到字节数或错误
}

ssize_t vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)  // 定义当前层的 C 函数入口
{
    ssize_t ret;                                  // 返回值

    if (!(file->f_mode & FMODE_READ))             // 文件不是可读方式打开
        return -EBADF;                            // 返回 fd 错误
    if (!(file->f_mode & FMODE_CAN_READ))         // 文件类型不支持读
        return -EINVAL;                           // 返回参数错误
    if (unlikely(!access_ok(buf, count)))         // 用户态目标地址不合法
        return -EFAULT;                           // 返回坏地址

    ret = rw_verify_area(READ, file, pos, count); // 检查读范围和权限
    if (ret)  // 检查状态或错误码，决定是否走异常/分支路径
        return ret;  // 把本层处理结果或错误码返回上一层
    if (count > MAX_RW_COUNT)                     // 限制单次读写最大长度
        count = MAX_RW_COUNT;  // 更新当前层需要传递的状态、长度、指针或错误码

    if (file->f_op->read)                         // 老式 read
        ret = file->f_op->read(file, buf, count, pos); // 调 read
    else if (file->f_op->read_iter)               // tty_fops 使用 read_iter
        ret = new_sync_read(file, buf, count, pos); // 包装成 iov_iter
    else                                          // 没有读方法
        ret = -EINVAL;                            // 不支持

    if (ret > 0) {                                // 成功读取
        fsnotify_access(file);                    // 文件访问通知
        add_rchar(current, ret);                  // 统计当前任务读取字节数
    }
    inc_syscr(current);                           // 统计 read 系统调用次数

    return ret;                                   // 返回实际字节数
}
```

### 6.3 `new_sync_read()`

`new_sync_read()` 是 `vfs_read()` 和 `tty_fops.read_iter` 之间的同步包装层。它把 CPython 传下来的用户态目标缓冲包装成 `iov_iter`，然后调用 `/dev/ttyACM0` 的 `read_iter`，也就是 `tty_read()`。

来源：linux/fs/read_write.c:new_sync_read（已按当前仓库源码核对）
```c
// linux/fs/read_write.c
static ssize_t new_sync_read(struct file *filp, char __user *buf, size_t len, loff_t *ppos) // 把 VFS read 请求包装成同步 read_iter 调用
{
    struct kiocb kiocb;                          // 同步 I/O 控制块，携带 file 和当前位置
    struct iov_iter iter;                        // 描述用户态目标缓冲的迭代器
    ssize_t ret;                                 // 保存 read_iter 返回值

    init_sync_kiocb(&kiocb, filp);               // 初始化同步 kiocb，并绑定当前 struct file
    kiocb.ki_pos = (ppos ? *ppos : 0);           // 普通文件使用当前位置；TTY stream 的 ppos 通常为 NULL
    iov_iter_ubuf(&iter, ITER_DEST, buf, len);   // 把用户态接收缓冲包装成 ITER_DEST

    ret = filp->f_op->read_iter(&kiocb, &iter);  // 调用 tty_fops.read_iter，也就是 tty_read()
    BUG_ON(ret == -EIOCBQUEUED);                 // 同步 read 路径不允许返回异步排队状态
    if (ppos)                                    // 普通文件需要维护偏移
        *ppos = kiocb.ki_pos;                    // 把 read_iter 更新后的偏移回写给调用方
    return ret;                                  // 返回读到字节数或错误码
}
```

### 6.4 TTY read 与 N_TTY read

来源：linux/drivers/tty/tty_io.c:tty_read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
static ssize_t tty_read(struct kiocb *iocb, struct iov_iter *to)  // 定义当前层的 C 函数入口
{
    struct file *file = iocb->ki_filp;            // 当前打开的 tty 文件
    struct tty_struct *tty = file_tty(file);      // file -> tty_struct  // 定义当前链路涉及的内核数据结构
    struct tty_ldisc *ld;                         // 当前 line discipline
    ssize_t ret;                                  // 返回值

    if (!tty || tty_io_error(tty))                // tty 不存在或 I/O 错误
        return -EIO;                              // 返回错误

    ld = tty_ldisc_ref_wait(tty);                 // 获取 N_TTY 行规程
    if (!ld)                                      // 如果没有行规程
        return hung_up_tty_read(iocb, to);        // 挂起处理
    ret = -EIO;                                   // 默认错误
    if (ld->ops->read)                            // N_TTY 提供 read
        ret = iterate_tty_read(ld, tty, file, to); // 分块读
    tty_ldisc_deref(ld);                          // 释放行规程引用

    return ret;                                   // 返回读到字节数
}
```

`tty_read()` 本身不直接把数据复制到用户态，而是调用 `iterate_tty_read()`。这里有一个固定的 64 字节 `kernel_buf`：N_TTY 先把 `read_buf` 中的数据复制到这个内核临时缓冲，TTY core 再用 `copy_to_iter()` 把数据复制到用户态 `bytes` 缓冲。

来源：linux/drivers/tty/tty_io.c:iterate_tty_read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/tty_io.c
static ssize_t iterate_tty_read(struct tty_ldisc *ld, struct tty_struct *tty,  // 定义当前层的 C 函数入口
                                struct file *file, struct iov_iter *to)  // 定义当前链路涉及的内核数据结构
{
    void *cookie = NULL;                           // N_TTY 跨次继续读的状态
    unsigned long offset = 0;                      // 已复制到用户态的偏移
    size_t copied, count = iov_iter_count(to);     // 用户本次请求读取的字节数
    u8 kernel_buf[64];                             // TTY core 的内核临时缓冲
    ssize_t retval = 0;  // 更新当前层需要传递的状态、长度、指针或错误码

    do {  // 本行参与当前 C 层路径的控制流或数据准备
        ssize_t size = min(count, sizeof(kernel_buf));  // 更新当前层需要传递的状态、长度、指针或错误码

        size = ld->ops->read(tty, file, kernel_buf, size, &cookie, offset);  // 更新当前层需要传递的状态、长度、指针或错误码
        if (!size)  // 检查状态或错误码，决定是否走异常/分支路径
            break;  // 本行参与当前 C 层路径的控制流或数据准备
        if (size < 0) {  // 检查状态或错误码，决定是否走异常/分支路径
            retval = retval ? retval : size;  // 更新当前层需要传递的状态、长度、指针或错误码
            break;  // 本行参与当前 C 层路径的控制流或数据准备
        }

        copied = copy_to_iter(kernel_buf, size, to); // 从内核临时缓冲复制到用户态 iov_iter
        offset += copied;  // 更新当前层需要传递的状态、长度、指针或错误码
        count -= copied;  // 更新当前层需要传递的状态、长度、指针或错误码
        if (unlikely(copied != size)) {  // 检查状态或错误码，决定是否走异常/分支路径
            count = 0;  // 更新当前层需要传递的状态、长度、指针或错误码
            retval = -EFAULT;  // 更新当前层需要传递的状态、长度、指针或错误码
        }
    } while (cookie);  // 调用下一层 C 函数继续完成当前路径

    memzero_explicit(kernel_buf, sizeof(kernel_buf));  // 调用下一层 C 函数继续完成当前路径
    return offset ? offset : retval;  // 把本层处理结果或错误码返回上一层
}
```

来源：linux/drivers/tty/n_tty.c:n_tty_read（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/n_tty.c
static ssize_t n_tty_read(struct tty_struct *tty, struct file *file, u8 *kbuf,  // 定义当前层的 C 函数入口
                          size_t nr, void **cookie, unsigned long offset)  // 本行参与当前 C 层路径的控制流或数据准备
{
    struct n_tty_data *ldata = tty->disc_data;    // N_TTY 私有数据，包含 read_buf
    u8 *kb = kbuf;                                // 内核临时缓冲写入位置
    DEFINE_WAIT_FUNC(wait, woken_wake_function);  // 读等待队列节点
    ssize_t retval;                               // 返回值

    if (file->f_flags & O_NONBLOCK) {             // pyserial 打开 fd 时使用非阻塞语义
        if (!mutex_trylock(&ldata->atomic_read_lock)) // 读锁被占用
            return -EAGAIN;                       // 非阻塞返回 EAGAIN
    } else {                                      // 阻塞 fd
        mutex_lock_interruptible(&ldata->atomic_read_lock); // 可被信号打断地取锁
    }

    down_read(&tty->termios_rwsem);               // 读 termios 配置
    add_wait_queue(&tty->read_wait, &wait);       // 加入读等待队列

    while (nr) {                                  // 还有用户请求的空间
        if (!input_available_p(tty, 0)) {         // N_TTY 缓冲区没有数据
            if (tty_io_nonblock(tty, file)) {     // 非阻塞 read
                retval = -EAGAIN;                 // 立即返回 EAGAIN
                break;                            // 退出
            }
            wait_woken(&wait, TASK_INTERRUPTIBLE, MAX_SCHEDULE_TIMEOUT); // 阻塞等待
            continue;                             // 醒来后重新检查
        }

        if (copy_from_read_buf(tty, &kb, &nr)) {  // 从 ldata->read_buf 复制到 kbuf
            remove_wait_queue(&tty->read_wait, &wait); // 移除等待节点
            *cookie = cookie;                     // 如果还有数据，支持继续读
            return kb - kbuf;                     // 返回本次复制字节数
        }
    }

    remove_wait_queue(&tty->read_wait, &wait);    // 清理等待队列
    mutex_unlock(&ldata->atomic_read_lock);       // 释放读锁
    if (kb - kbuf)                                // 如果读到数据
        retval = kb - kbuf;                       // 返回字节数
    return retval;                                // 返回字节数或错误
}
```

`copy_from_read_buf()` 内部通过 `memcpy()` 从 N_TTY 的环形 `read_buf` 复制到 `iterate_tty_read()` 提供的 `kernel_buf`。因此，同步 `read()` 路径中 CPU 参与的主要复制是 `read_buf -> kernel_buf -> 用户态 bytes`。

来源：linux/drivers/tty/n_tty.c:copy_from_read_buf（节选：仅保留本链路相关分支，已按当前仓库源码核对）
```c
// linux/drivers/tty/n_tty.c
static bool copy_from_read_buf(const struct tty_struct *tty, u8 **kbp,  // 定义当前层的 C 函数入口
                               size_t *nr)  // 本行参与当前 C 层路径的控制流或数据准备
{
    struct n_tty_data *ldata = tty->disc_data;  // 定义当前链路涉及的内核数据结构
    size_t head = smp_load_acquire(&ldata->commit_head);  // 更新当前层需要传递的状态、长度、指针或错误码
    size_t tail = MASK(ldata->read_tail);  // 更新当前层需要传递的状态、长度、指针或错误码
    size_t n = min3(head - ldata->read_tail, N_TTY_BUF_SIZE - tail, *nr);  // 更新当前层需要传递的状态、长度、指针或错误码
    u8 *from = read_buf_addr(ldata, tail);  // 更新当前层需要传递的状态、长度、指针或错误码

    memcpy(*kbp, from, n);                         // N_TTY read_buf -> kernel_buf  // 调用下一层 C 函数继续完成当前路径
    zero_buffer(tty, from, n);                     // 清掉已消费区域
    smp_store_release(&ldata->read_tail, ldata->read_tail + n);  // 调用下一层 C 函数继续完成当前路径
    *kbp += n;
    *nr -= n;

    return head != ldata->read_tail;  // 把本层处理结果或错误码返回上一层
}
```

在 `readPort(length)` 中，pyserial 已经用 `select()` 确认 fd 可读，因此正常情况下 `n_tty_read()` 会直接从 `ldata->read_buf` 复制数据并返回，不会睡眠。如果 select 之后数据被其他线程抢走或设备断开，非阻塞 read 可能返回 `EAGAIN` 或空读。

---

## 7. 关键数据结构

| 结构 | 位置 | 本链路中的含义 |
|------|------|----------------|
| `fd_set` | 用户态 CPython/glibc | `select()` 的 bitset，bit N 表示 fd N |
| `fd_set_bits` | `linux/fs/select.c` | 内核复制后的 read/write/except 输入集合和结果集合 |
| `struct file` | VFS | 一个打开的 `/dev/ttyACM0` 文件实例，挂 `tty_fops` |
| `struct tty_struct` | TTY core | 一个 tty 设备实例，保存 `tty->ops`、`tty->driver_data`、等待队列 |
| `struct tty_ldisc` | TTY line discipline | 当前行规程，通常是 `N_TTY`，提供 `read/write/poll` |
| `struct n_tty_data` | `n_tty.c` | N_TTY 私有缓冲，`read_buf` 保存已经收到但用户还没读走的字节 |
| `struct acm` | `cdc-acm.c` | CDC ACM USB 串口设备，连接 USB 和 TTY |
| `struct urb` | USB core | USB Request Block，bulk IN URB 收舵机回包 |
| `struct usb_hcd` | USB core/HCD | USB 主机控制器抽象，负责把 URB 分发给 xHCI |
| `struct xhci_hcd` / TRB | xHCI driver | 树莓派 5 USB 主控的传输环和硬件请求描述 |

---

## 8. readPort 究竟完成了什么

`readPort(length)` 完成的是“非阻塞取走当前已经进入 TTY 缓冲区的舵机回包字节”：

1. CDC ACM 驱动提前提交 bulk IN URB。
2. 舵机通过 USB 返回状态包，xHCI 完成 URB。
3. `acm_read_bulk_callback()` 把 URB 数据推入 TTY flip buffer。
4. N_TTY 把 flip buffer 数据放入 `n_tty_data.read_buf`。
5. pyserial 先用 `pselect6` 立即检查 fd 是否可读。
6. 如果可读，再用 `read(fd, buf, count)` 把数据复制回 Python。
7. SDK 的 `rxPacket()` 把这些 bytes 拼成 Feetech 状态包并校验。

从“舵机回包进入主机”到“Python 拿到 bytes”，数据搬运可以分为下面几步：

| 阶段 | 源地址 | 目标地址 | 代码位置 | 说明 |
| --- | --- | --- | --- | --- |
| USB 接收 | USB 总线数据 | CDC ACM 读 URB 的 `transfer_buffer` | xHCI DMA | 硬件 DMA 写入，不是 CPU `memcpy()` |
| CDC ACM 到 TTY | URB `transfer_buffer` | TTY flip buffer | `tty_insert_flip_string()` | CPU 复制收到的舵机回包字节 |
| TTY 到 N_TTY | TTY flip buffer | N_TTY `read_buf` | `flush_to_ldisc()` / `n_tty_receive_buf_common()` | 原始模式下基本不解释协议，只放入行规程读缓冲 |
| N_TTY 到 TTY 临时缓冲 | N_TTY `read_buf` | `iterate_tty_read()` 的 `kernel_buf` | `copy_from_read_buf()` | `read()` 同步路径中的第一次 CPU 复制 |
| 内核到用户态 | `kernel_buf` | CPython `bytes` 缓冲 | `copy_to_iter()` | `read()` 同步路径中的第二次 CPU 复制 |

`select()` 阶段也会复制 fd 位图等控制信息，但它不搬运舵机回包内容；真正把状态包字节带回 Python 的是后面的 `read()`。

本链路最终涉及的系统调用参数是：

```text
pselect6(
    n=max(fd, pipe_fd)+1,
    inp=包含 ttyACM0 fd 和 pipe fd 的 fd_set,
    outp=NULL,
    exp=NULL,
    tsp={0,0},
    sig=NULL
)

read(
    __NR_read=63,
    fd=/dev/ttyACM0,
    buf=<Python bytes 写入器的用户态地址>,
    count=wait_length - rx_length
)
```
