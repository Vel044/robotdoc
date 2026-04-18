# Feetech 舵机串口 read/write/select/ioctl 系统调用接口分析

> 本文档追踪 Feetech 舵机读写过程中，从 scservo_sdk → pyserial(serialposix.py) → CPython → glibc → Linux 内核 的四个核心系统调用的完整参数传递链路。

---

## 一、调用链总览

```
pyserial (serial/serialposix.py)
  os.read(fd, size)         ──────────────────────────────────────────────────────┐
  os.write(fd, data)        ─────────────────────────────────────────────────────┤
  select.select(r,w,x,to)  ─────────────────────────────────────────────────────┤
  termios.tcdrain(fd)      ─────────────────────────────────────────────────────┤
  fcntl.ioctl(fd,cmd,arg)  ─────────────────────────────────────────────────────┘
         │
         ▼
CPython (Modules/)
  os.read      → posixmodule.c: os_read_impl()    → Python/fileutils.c: _Py_read()
  os.write     → posixmodule.c: os_write_impl()   → Python/fileutils.c: _Py_write()
  select.select → selectmodule.c: select_select_impl()
  tcdrain      → termios.c: termios_tcdrain_impl()
  ioctl        → fcntlmodule.c: fcntl_ioctl_impl()
         │
         ▼
glibc (sysdeps/unix/sysv/linux/)
  __libc_read()    → SYSCALL_CANCEL(read, fd, buf, nbytes)
  __libc_write()   → SYSCALL_CANCEL(write, fd, buf, nbytes)
  __select64()     → SYSCALL_CANCEL(pselect6, nfds, readfds, writefds, ...)
  __libc_tcdrain() → SYSCALL_CANCEL(ioctl, fd, TCSBRK, 1)    ← 本质是 ioctl
  __ioctl()        → 直接 svc #0 (ARM 系统调用)
         │
         ▼
Linux 内核
  sys_read()    → VFS → tty_read() → n_tty_read()
  sys_write()   → VFS → tty_write() → n_tty_write()
  sys_select()  → kern_select() → core_sys_select() → do_select() → vfs_poll()
  sys_ioctl()   → tty_ioctl() → n_tty_ioctl() / tty_tiocmset() / ...
```

---

## 二、read() 系统调用

### 2.1 第一层：pyserial (serial/serialposix.py:553)

```python
# serial/serialposix.py: Serial.read()
def read(self, size=1):
    read = bytearray()
    timeout = Timeout(self._timeout)

    while len(read) < size:
        # ① select 等待 fd 可读
        ready, _, _ = select.select(
            [self.fd], [], [], timeout.time_left())
        if not ready:  # 超时
            break

        # ② os.read() ———— 系统调用 ————
        buf = os.read(self.fd, size - len(read))
        read.extend(buf)

        # ③ 超时则退出
        if timeout.expired():
            break

    return bytes(read)
```

**参数：**
```
fd    = self.fd    (int)         文件描述符
count = size - len(read) (int)   本次最多读取字节数（剩余需要的量）
```

---

### 2.2 第二层：CPython os_read_impl (cpython/Modules/posixmodule.c:11950)

```c
// cpython/Modules/posixmodule.c
//
// 参数说明：
//   module : PyObject*  ← Python 的 `os` 模块对象（CPython 自动传入，与实例方法的 self 类似）
//   fd     : int        ← 文件描述符
//   length : Py_ssize_t ← 读取字节数
static PyObject *
os_read_impl(PyObject *module, int fd, Py_ssize_t length)
{
    if (length < 0) { errno = EINVAL; return posix_error(); }
    length = Py_MIN(length, _PY_READ_MAX);
    // 创建一个 length 字节的 PyBytes 对象作为接收缓冲区
    PyBytesWriter *writer = PyBytesWriter_Create(length);
    // 调用 _Py_read(fd, buf, count)
    Py_ssize_t n = _Py_read(fd, PyBytesWriter_GetData(writer), length);
    if (n == -1) { PyBytesWriter_Discard(writer); return NULL; }

    return PyBytesWriter_FinishWithSize(writer, n);
}
```

---

### 2.3 第三层：_Py_read (cpython/Python/fileutils.c:1865)

```c
// cpython/Python/fileutils.c
//
// 参数说明：
//   fd    : int       文件描述符
//   buf   : void*     接收缓冲区的地址（CPython 分配的 PyBytes 内部地址）
//   count : size_t    最多读取的字节数
Py_ssize_t
_Py_read(int fd, void *buf, size_t count)
{
    Py_ssize_t n; int err; int async_err = 0;
    if (count > _PY_READ_MAX) { count = _PY_READ_MAX; }

    _Py_BEGIN_SUPPRESS_IPH
    do {
        Py_BEGIN_ALLOW_THREADS           // 释放 GIL
        errno = 0;
        n = read(fd, buf, count);       // ← 调用 glibc __libc_read()
        err = errno;
        Py_END_ALLOW_THREADS            // 重新获取 GIL
    } while (n < 0 && err == EINTR &&
             !(async_err = PyErr_CheckSignals()));
    _Py_END_SUPPRESS_IPH
    if (async_err) { return -1; }
    return n;
}
```

---

### 2.4 第四层：glibc __libc_read → svc #0 (glibc-2.42/sysdeps/unix/sysv/linux/read.c)

```c
// glibc-2.42/sysdeps/unix/sysv/linux/read.c
ssize_t
__libc_read(int fd, void *buf, size_t nbytes)
{
  return SYSCALL_CANCEL(read, fd, buf, nbytes);
}
```

**SYSCALL_CANCEL 展开链（用户态 → 内核态的完整过程）：**

```
_Py_read() 中调用 read(fd, buf, count)
  │
  ├──→ glibc __libc_read()         ← 仍处于用户态
  │       return SYSCALL_CANCEL(read, fd, buf, nbytes);
  │
  ├──→ glibc __SYSCALL_CANCEL3     ← 宏展开
  │
  ├──→ glibc __syscall_cancel()    ← 最后一个取消处理块
  │       INTERNAL_SYSCALL_NCS_CALL(__NR_read, fd, buf, nbytes);
  │
  ├──→ ARM: svc #0                ← ⚡ 触发陷阱，CPU 特权级切换
  │       (Supervisor Call，ARM 的系统调用指令)
  │
  └────→ Linux 内核 sys_read()    ← ⚡ 开始在内核态执行
              参数传递：x0=fd, x1=buf, x2=count
```

**关键点：glibc 和内核之间没有函数调用关系。`svc #0` 是一条 CPU 指令，触发 CPU 从用户态切换到内核态，CPU 自动跳转到 sys_read 入口。两层之间是 CPU 硬件级别的切换，不是软件函数调用。**

---

### 2.5 Linux 内核 sys_read (linux/fs/read_write.c)

```c
// linux/fs/read_write.c
SYSCALL_DEFINE3(read, unsigned int, fd, char __user *, buf, size_t, count)
{
    struct fd f = fdget_pos(fd);
    if (f.file)
        ret = vfs_read(f.file, buf, count, &f.file->f_pos);
    fdput_pos(f);
    return ret;
}

// vfs_read() → file->f_op->read_iter() → tty_read() → iterate_tty_read()
// → n_tty_read()
//   数据来源：tty->disc_data->read_buf (环形缓冲区)
```

**参数对照（用户态 → 内核态）：**

| 层次 | fd | buf | count |
|------|----|-----|-------|
| pyserial | `self.fd` | CPython 分配 | `size - len(read)` |
| CPython os_read_impl | `fd` | `PyBytesWriter_GetData()` | `length` |
| _Py_read | `fd` | `buf` | `count` |
| glibc | `fd` | `buf` | `nbytes` |
| sys_read | `fd` | `buf` (用户态地址) | `count` |

---

## 三、write() 系统调用

### 3.1 第一层：pyserial (serial/serialposix.py:628)

```python
# serial/serialposix.py: Serial.write()
def write(self, data):
    d = to_bytes(data)
    tx_len = length = len(d)
    timeout = Timeout(self._write_timeout)

    while tx_len > 0:
        # ① os.write() ———— 系统调用 ————
        n = os.write(self.fd, d)       # 将用户态缓冲区写入串口驱动
        if timeout.is_non_blocking:
            return n                     # 非阻塞模式：立即返回已写入字节数

        # ② select 等待 fd 变为"可写"
        ready, _, _ = select.select([], [self.fd], [], timeout.time_left())
        if not ready:  # 超时
            break

        # ③ 丢弃已发送部分，继续发送剩余字节
        d = d[n:]; tx_len -= n

    return length - len(d)
```

**参数：**
```
fd   = self.fd   (int)       文件描述符
data = bytes 对象             CPython 自动提取 data->buf 和 data->len
```

---

### 3.2 第二层：CPython os_write_impl (cpython/Modules/posixmodule.c:12295)

```c
// cpython/Modules/posixmodule.c
//
// 参数说明：
//   module : PyObject*  ← Python 的 `os` 模块对象
//   fd     : int        ← 文件描述符
//   data   : Py_buffer* ← Python bytes/bytearray 的缓冲协议指针
//                        data->buf = 数据起始地址, data->len = 数据长度
static Py_ssize_t
os_write_impl(PyObject *module, int fd, Py_buffer *data)
{
    return _Py_write(fd, data->buf, data->len);
}
```

---

### 3.3 第三层：_Py_write_impl (cpython/Python/fileutils.c:1924)

```c
// cpython/Python/fileutils.c
//
// 参数说明：
//   fd      : int       文件描述符
//   buf     : void*     数据缓冲区的地址
//   count   : size_t    数据长度
//   gil_held: int       GIL 是否持有（此处为 true）
static Py_ssize_t
_Py_write_impl(int fd, const void *buf, size_t count, int gil_held)
{
    Py_ssize_t n; int err; int async_err = 0;
    if (count > _PY_WRITE_MAX) { count = _PY_WRITE_MAX; }

    if (gil_held) {
        do {
            Py_BEGIN_ALLOW_THREADS
            errno = 0;
            n = write(fd, buf, count);   // ← 调用 glibc __libc_write()
            err = errno;
            Py_END_ALLOW_THREADS
        } while (n < 0 && err == EINTR &&
                 !(async_err = PyErr_CheckSignals()));
    }
    return n;
}
```

---

### 3.4 第四层：glibc __libc_write → svc #0 (glibc-2.42/sysdeps/unix/sysv/linux/write.c)

```c
// glibc-2.42/sysdeps/unix/sysv/linux/write.c
ssize_t
__libc_write(int fd, const void *buf, size_t nbytes)
{
  return SYSCALL_CANCEL(write, fd, buf, nbytes);
}
```

**SYSCALL_CANCEL 展开链（与 read 完全相同的过程）：**

```
_Py_write_impl() 中调用 write(fd, buf, count)
  │
  ├──→ glibc __libc_write()         ← 用户态
  │       return SYSCALL_CANCEL(write, fd, buf, nbytes);
  │
  ├──→ __SYSCALL_CANCEL3 → __syscall_cancel()
  │
  ├──→ ARM: svc #0                  ← ⚡ 特权级切换
  │
  └────→ Linux 内核 sys_write()     ← ⚡ 内核态执行
              参数传递：x0=fd, x1=buf, x2=count
```

---

### 3.5 Linux 内核 sys_write (linux/fs/read_write.c)

```c
// linux/fs/read_write.c
SYSCALL_DEFINE3(write, unsigned int, fd, const char __user *, buf, size_t, count)
{
    struct fd f = fdget_pos(fd);
    if (f.file)
        ret = vfs_write(f.file, buf, count, &f.file->f_pos);
    fdput_pos(f);
    return ret;
}

// vfs_write() → tty_write() → n_tty_write() → uart_write()
//   数据去向：写入 UART 循环缓冲区，触发硬件发送
```

---

## 四、select() 系统调用

### 4.1 第一层：pyserial (serial/serialposix.py:553 内部)

```python
# pyserial 内部 read() 中使用 select 等待串口可读
ready, _, _ = select.select(
    [self.fd, self.pipe_abort_read_r],  # rlist: 监听可读
    [],                                   # wlist: 不监听可写
    [],                                   # xlist: 不监听异常
    timeout.time_left())                  # timeout: float 秒，None=永久
```

**参数：**
```
rlist   = [self.fd, self.pipe_abort_read_r]   Python list → CPython 转为 fd_set 位图
wlist   = []                                   空列表
xlist   = []                                   空列表
timeout = timeout.time_left()                  float 秒或 None
```

---

### 4.2 第二层：CPython select_select_impl (cpython/Modules/selectmodule.c:278)

```c
// cpython/Modules/selectmodule.c
//
// 参数说明：
//   module       : PyObject*  ← Python 的 `select` 模块对象
//   rlist        : PyObject*  ← 监听"可读" fd 的 Python list
//   wlist        : PyObject*  ← 监听"可写" fd 的 Python list
//   xlist        : PyObject*  ← 监听"异常" fd 的 Python list
//   timeout_obj  : PyObject*  ← Python float/int/None，超时时间（秒）
static PyObject *
select_select_impl(PyObject *module, PyObject *rlist, PyObject *wlist,
                   PyObject *xlist, PyObject *timeout_obj)
{
    fd_set ifdset, ofdset, efdset;
    struct timeval tv, *tvp;

    // ① Python list → fd_set 位图
    imax = seq2set(rlist, &ifdset, rfd2obj);  // 遍历 rlist，将每个 fd 置入 ifdset
    omax = seq2set(wlist, &ofdset, wfd2obj);
    emax = seq2set(xlist, &efdset, efd2obj);
    max = max(imax, omax, emax);

    // ② timeout 转换：float → struct timeval
    if (timeout_obj == Py_None)
        tvp = NULL;                           // NULL = 永久等待
    else
        tvp = &tv;                            // tv_sec + tv_usec

    // ③ 调用 glibc select()
    n = select(max, &ifdset, &ofdset, &efdset, tvp);

    // ④ fd_set 位图 → Python list（FD_ISSET 检测每个 fd）
    // 返回 [rlist_result, wlist_result, xlist_result]
}
```

**传给 glibc select() 的五个参数：**
```
max      = max_fd + 1        (int)          参与检测的最大 fd 编号 + 1
readfds  = &ifdset           (fd_set*)      监听可读 fd 位图（传入待检测，传出就绪状态）
writefds = &ofdset           (fd_set*)      监听可写 fd 位图
exceptfds = &efdset          (fd_set*)      监听异常 fd 位图
timeout  = tvp               (struct timeval*) NULL=永久等待
```

**fd_set 类型定义：**
```c
// linux/include/uapi/linux/posix_types.h:23-27
#define __FD_SETSIZE 1024
typedef struct {
    unsigned long fds_bits[1024 / (8 * sizeof(long))];  // ARM64: [16] (16*64=1024 bits)
} __kernel_fd_set;
// linux/include/linux/types.h:20
typedef __kernel_fd_set fd_set;
```

**struct timeval 类型定义：**
```c
// linux/include/uapi/linux/time.h:17-20 (用户态)
struct timeval {
    __kernel_old_time_t tv_sec;     // 秒
    __kernel_suseconds_t tv_usec;   // 微秒
};
```

---

### 4.3 第三层：glibc __select64 → pselect6 → svc #0 (glibc-2.42/sysdeps/unix/sysv/linux/select.c)

```c
// glibc-2.42/sysdeps/unix/sysv/linux/select.c
int
__select64(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
           struct __timeval64 *timeout)
{
    // struct __timeval64 { long long tv_sec; long long tv_usec; }
    int r = SYSCALL_CANCEL(pselect6, nfds, readfds, writefds, exceptfds,
                           pts64, NULL);   // ← 内部调用 pselect6 系统调用
    return r;
}
```

**注意：glibc `select()` 内部实际调用的是 `pselect6()` 系统调用（多一个 sigmask 参数）。**

**完整调用链：**

```
select_select_impl() 中调用 select(max, &ifdset, ...)
  │
  ├──→ glibc __select64()
  │       return SYSCALL_CANCEL(pselect6, nfds, readfds, writefds, exceptfds, pts64, NULL);
  │
  ├──→ __SYSCALL_CANCEL6 → __syscall_cancel()
  │
  ├──→ ARM: svc #0                  ← ⚡ pselect6 系统调用
  │
  └────→ Linux 内核 sys_pselect6()  ← ⚡ 内核态
              参数传递：x0=nfds, x1=readfds, x2=writefds, x3=exceptfds, x4=timeout, x5=sigmask
```

---

### 4.4 Linux 内核 sys_select / sys_pselect6 (linux/fs/select.c:836)

```c
// linux/fs/select.c
//
// 参数说明：
//   n    : int      ← 最大 fd 编号 + 1
//   inp  : fd_set __user *  ← 监听可读 fd 位图（用户态指针）
//   outp : fd_set __user *  ← 监听可写 fd 位图
//   exp  : fd_set __user *  ← 监听异常 fd 位图
//   tvp  : struct __kernel_old_timeval __user * ← 超时时间（秒+微秒）
//
// struct __kernel_old_timeval 定义（linux/include/uapi/linux/time_types.h:24-29）：
//   struct __kernel_old_timeval {
//       __kernel_long_t tv_sec;    // 秒
//       __kernel_long_t tv_usec;   // 微秒
//   };

SYSCALL_DEFINE5(select, int, n, fd_set __user *, inp,
                fd_set __user *, outp, fd_set __user *, exp,
                struct __kernel_old_timeval __user *, tvp)
{
    return kern_select(n, inp, outp, exp, tvp);
}

// kern_select() → core_sys_select() → do_select()
//
// 核心轮询循环 do_select() (linux/fs/select.c:477):
static noinline_for_stack int do_select(int n, fd_set_bits *fds,
                                         struct timespec64 *end_time)
{
    for (;;) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < BITS_PER_LONG; ++j, ++i) {
                f = fdget(i);
                mask = vfs_poll(f.file, wait);  // ← 对每个 fd 调用 vfs_poll
                fdput(f);
                // 检查 POLLIN/POLLOUT/POLLEX 位
            }
            cond_resched();
        }
        if (retval || timed_out || signal_pending(current)) break;
        // 睡眠等待，直到有 fd 就绪或超时
        poll_schedule_timeout(&table, TASK_INTERRUPTIBLE, to, slack);
    }
    return retval;
}

// 对 TTY 设备：vfs_poll() → tty_poll() → 关联到 tty->read_wait / write_wait
// 收到数据时内核调用 wake_up_interruptible(tty->read_wait)，poll 返回就绪
```

---

## 五、tcdrain() 系统调用

### 5.1 第一层：pyserial (serial/serialposix.py:flush())

```python
# serial/serialposix.py: Serial.flush()
def flush(self):
    if not self.is_open: raise PortNotOpenError()
    termios.tcdrain(self.fd)   # ← 等待输出缓冲区排空
```

**参数：**
```
fd = self.fd (int)  文件描述符
```

---

### 5.2 第二层：CPython termios_tcdrain_impl (cpython/Modules/termios.c:324)

```c
// cpython/Modules/termios.c
//
// 参数说明：
//   module : PyObject*  ← Python 的 `termios` 模块对象
//   fd     : int        ← 文件描述符
static PyObject *
termios_tcdrain_impl(PyObject *module, int fd)
{
    termiosmodulestate *state = PyModule_GetState(module);
    int r;
    Py_BEGIN_ALLOW_THREADS
    r = tcdrain(fd);           // ← 调用 glibc tcdrain()
    Py_END_ALLOW_THREADS
    if (r == -1) { return PyErr_SetFromErrno(state->TermiosError); }
    Py_RETURN_NONE;
}
```

---

### 5.3 第三层：glibc __libc_tcdrain → ioctl → svc #0 (glibc-2.42/sysdeps/unix/sysv/linux/tcdrain.c)

```c
// glibc-2.42/sysdeps/unix/sysv/linux/tcdrain.c
int
__libc_tcdrain(int fd)
{
    /* With an argument of 1, TCSBRK for output to be drain.  */
    return SYSCALL_CANCEL(ioctl, fd, TCSBRK, 1);  // ← 本质是 ioctl！
}
```

**关键发现：`tcdrain()` 在 glibc 内部实现为 `ioctl(fd, TCSBRK=0x5409, 1)`，不是独立的系统调用。**

**完整调用链：**

```
termios_tcdrain_impl() 中调用 tcdrain(fd)
  │
  ├──→ glibc __libc_tcdrain()
  │       return SYSCALL_CANCEL(ioctl, fd, TCSBRK, 1);
  │
  ├──→ __SYSCALL_CANCEL3 → __syscall_cancel()
  │
  ├──→ ARM: svc #0                  ← ⚡ ioctl 系统调用
  │
  └────→ Linux 内核 sys_ioctl()     ← ⚡ 内核态
              参数传递：x0=fd, x1=TCSBRK(0x5409), x2=1
```

---

### 5.4 第四层：Linux 内核 sys_ioctl → tty_ioctl (linux/drivers/tty/tty_ioctl.c)

```c
// linux/drivers/tty/tty_ioctl.c
static long tty_mode_ioctl(struct tty_struct *tty, unsigned int cmd,
                             unsigned long arg)
{
    switch (cmd) {
    case TCSBRK:   // arg=1 时等价于 tcdrain
        return tty_wait_until_sent(tty, arg ? 0 : DURATION);
    }
}

// linux/drivers/tty/tty_io.c
static void tty_wait_until_sent(struct tty_struct *tty, long timeout)
{
    if (!tty->ops->wait_until_sent) return;
    tty->ops->wait_until_sent(tty, timeout);
    // → uart_wait_until_sent()
    // → 等待 UART 硬件 FIFO + 发送缓冲区完全排空
}
```

---

## 六、ioctl() 系统调用（以 TIOCINQ 为例）

### 6.1 第一层：pyserial (serial/serialposix.py:549)

```python
# serial/serialposix.py: Serial.in_waiting
@property
def in_waiting(self):
    s = fcntl.ioctl(self.fd, TIOCINQ, TIOCM_zero_str)
    #               ↑         ↑          ↑
    #            串口fd    0x541B     4字节零值缓冲区
    return struct.unpack('I', s)[0]
```

**参数：**
```
fd   = self.fd             (int)          文件描述符
cmd  = TIOCINQ = 0x541B    (unsigned long) ioctl 命令码
arg  = TIOCM_zero_str      (bytes)        4字节零值缓冲区（内核写入结果）
```

---

### 6.2 第二层：CPython fcntl_ioctl_impl (cpython/Modules/fcntlmodule.c:197)

```c
// cpython/Modules/fcntlmodule.c
//
// 参数说明：
//   module     : PyObject*      ← Python 的 `fcntl` 模块对象
//   fd         : int            ← 文件描述符
//   code       : unsigned long  ← ioctl 命令码（如 TIOCINQ=0x541B）
//   arg        : PyObject*      ← ioctl 参数（int、bytes 或 None）
//   mutate_arg : int            ← 标记 arg 是否会被内核修改
static PyObject *
fcntl_ioctl_impl(PyObject *module, int fd, unsigned long code,
                  PyObject *arg, int mutate_arg)
{
    int ret; int async_err = 0;

    if (arg == NULL || PyIndex_Check(arg)) {
        // ① arg 是整数（如 TIOCMBIS 的位标志参数）
        int int_arg = 0;
        if (arg != NULL) { PyArg_Parse(arg, "i", &int_arg); }
        do {
            Py_BEGIN_ALLOW_THREADS
            ret = ioctl(fd, code, int_arg);   // ← 调用 glibc ioctl()
            Py_END_ALLOW_THREADS
        } while (ret == -1 && errno == EINTR && !(async_err = PyErr_CheckSignals()));
        return PyLong_FromLong(ret);
    }

    if (PyUnicode_Check(arg) || PyObject_CheckBuffer(arg)) {
        // ② arg 是可变缓冲区（如 TIOCINQ 使用此路径）
        char buf[IOCTL_BUFSZ+GUARDSZ];  // 1024 + guard
        memcpy(buf, ptr, len);
        memcpy(buf + len, guard, GUARDSZ);   // 溢出保护
        do {
            Py_BEGIN_ALLOW_THREADS
            ret = ioctl(fd, code, ptr);      // ← 内核写入结果到 buf
            Py_END_ALLOW_THREADS
        } while (ret == -1 && errno == EINTR && !(async_err = PyErr_CheckSignals()));
        if (ptr == buf) { memcpy(view.buf, buf, len); }  // 复制回 Python 对象
        return PyLong_FromLong(ret);
    }
}
```

**对于 TIOCINQ（`arg = 4字节缓冲区`）：**
```
fd   = self.fd
code = 0x541B (TIOCINQ)
ptr  = 4字节零值缓冲区地址
      内核 ioctl 处理函数将输入缓冲区字节数写入 *ptr
```

---

### 6.3 第三层：glibc __ioctl → svc #0 (glibc-2.42/sysdeps/unix/sysv/linux/aarch64/ioctl.S)

```assembly
/* glibc-2.42/sysdeps/unix/sysv/linux/aarch64/ioctl.S */
ENTRY(__ioctl)
    MOV     x3, x2          /* arg  → x3 */
    MOV     x2, x1          /* code → x2 */
    MOV     x1, x0          /* fd   → x1 */
    MOV     x0, #NR_ioctl   /* syscall number → x0 */
    svc     #0x00           /* supervisor call → 陷入内核 */
END(__ioctl)
```

**完整调用链：**

```
fcntl_ioctl_impl() 中调用 ioctl(fd, code, ptr)
  │
  ├──→ glibc __ioctl()            ← 用户态汇编函数
  │       MOV x3, x2              // arg → x3
  │       MOV x2, x1              // code → x2
  │       MOV x1, x0              // fd → x1
  │       MOV x0, #NR_ioctl       // syscall# → x0
  │       svc #0x00               // ⚡ 直接触发系统调用
  │
  └────→ Linux 内核 sys_ioctl()   ← ⚡ 内核态
              参数传递：x0=fd, x1=cmd, x2=arg
```

**注意：`ioctl()` 绕过了 SYSCALL_CANCEL 宏，因为 ioctl 不支持线程取消语义（没有取消点），所以直接用汇编执行 `svc #0`。**

---

### 6.4 第四层：Linux 内核 sys_ioctl (linux/fs/ioctl.c → tty_ioctl)

**ioctl 的三个参数含义：**

| 参数 | 类型 | 含义 |
|------|------|------|
| `fd` | `unsigned int` | 文件描述符，指向 TTY 设备 |
| `cmd` | `unsigned int` | **操作码**（如 `0x5409` = TCSBRK），告诉内核执行哪个操作 |
| `arg` | `unsigned long` | **操作数**（含义完全取决于 cmd）|

**cmd 的编码格式（32-bit）：**

```
cmd 是一个 32 位整数，编码为：
  [31:30] 方向    = 00=无数据传输, 01=写, 10=读, 11=读写
  [29:16] size    = 数据大小
  [15:8]  type    = 类型字符（如 'T'=0x54 表示终端控制）
  [7:0]   nr      = 命令编号

对于 TCSBRK = 0x5409：
  0x54 = 'T' (终端控制类型)
  0x09 = 9  (命令编号)
  方向=无(size=0)
  合并：0x54 << 8 | 0x09 = 0x5409
```

**arg=1 传入 TCSBRK 的含义（tcdrain 的实现）：**

```c
// linux/drivers/tty/tty_ioctl.c:2762
case TCSBRK:   /* SVID version: non-zero arg --> no break */
    /* non-zero arg means wait for all output data
     * to be sent (performed above) but don't send break.
     * This is used by the tcdrain() termios function.
     */
    if (!arg)
        tty->ops->break_ctl(tty, -1);  // arg=0 → 发送 break 信号
    // arg ≠ 0 → 不发送 break，只等待输出排空（tcdrain 走这个分支）
    return tty_wait_until_sent(tty, 0);  // 等待 TX FIFO + 发送缓冲区排空
```

**总结：`tcdrain(fd)` → `ioctl(fd, TCSBRK=0x5409, 1)`**
- `cmd = 0x5409 (TCSBRK)` = "Terminal break control"
- `arg = 1` = "不发送 break，只等待输出排空"
- tcdrain 正是利用了这个语义：等待所有数据从 UART 硬件 FIFO 发送出去

---

```c
// linux/fs/ioctl.c
SYSCALL_DEFINE3(ioctl, unsigned int, fd, unsigned int, cmd, unsigned long, arg)
{
    struct fd f = fdget(fd);
    if (!f.file) return -EBADF;
    error = vfs_ioctl(f.file, fd, cmd, arg);
    fdput(f);
    return error;
}

// vfs_ioctl() → filp->f_op->unlocked_ioctl() → tty_ioctl()

// linux/drivers/tty/tty_ioctl.c
long tty_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct tty_struct *tty = file_tty(file);
    switch (cmd) {
    case TIOCINQ:   // = FIONREAD = 0x541B
        return n_tty_ioctl(tty, cmd, arg);
        // → put_user(ldata->read_cnt, (unsigned int __user *)arg)
        //   内核将 read_cnt 值写入用户态缓冲区 arg
    case TCSBRK:    // arg=1 时被 tcdrain() 调用
        return tty_wait_until_sent(tty, arg ? 0 : DURATION);
        // arg=1 → tty_wait_until_sent(tty, 0) 等待输出排空
        // arg=0 → 发送 break 信号
    case TIOCMBIS:
    case TIOCMBIC:
    case TIOCMSET:
        return tty_tiocmset(tty, cmd, (void __user *)arg);
    }
}
```

---

## 七、Feetech 舵机场景系统调用序列

```
时间线  用户态(pyserial)              系统调用              内核动作
──────  ───────────────────────────  ───────────────       ─────────
  t0    flush()
        → termios.tcdrain(fd)       → ioctl(fd,TCSBRK,1)  tty_wait_until_sent()
                                                                          等待 TX FIFO 排空

  t1    write(packet)
        → os.write(fd, packet)      → write(fd,buf,count)  tty_write()→n_tty_write()
                                                                          → uart_write()→启动TX

  t2    read(length)                ┐
        while len(read) < length:   │
  t2.1  select.select([fd],...to)  → pselect6(nfds,...)   do_select()→vfs_poll()
                                     │                      睡眠直到 fd 可读或超时
  t2.2  os.read(fd, remaining)     → read(fd,buf,count)  n_tty_read()→copy_from_read_buf()
                                     │                      从 read_buf 取出数据
  t2.3  if len(read) < length:     │
        → 回到 t2.1 (循环)          │

  t3    in_waiting
        → fcntl.ioctl(fd,TIOCINQ)  → ioctl(fd,0x541B,&n)  n_tty_ioctl(TIOCINQ)
                                                                          返回 read_cnt
```

---

## 八、ioctl 常量值对照

| 常量名 | pyserial 定义 | 内核定义 | 值 | 用途 |
|--------|-------------|---------|----|------|
| `TIOCINQ` | `getattr(termios,'FIONREAD',0x541B)` | `#define FIONREAD 0x541B` | `0x541B` | 查询输入缓冲区字节数 |
| `TIOCOUTQ` | `getattr(termios,'TIOCOUTQ',0x5411)` | `#define TIOCOUTQ 0x5411` | `0x5411` | 查询输出缓冲区字节数 |
| `TIOCMGET` | `termios.TIOCMGET` | `#define TIOCMGET 0x5415` | `0x5415` | 读取 modem 状态 |
| `TIOCMBIS` | `termios.TIOCMBIS` | `#define TIOCMBIS 0x5416` | `0x5416` | 设置 modem 位 |
| `TIOCMBIC` | `termios.TIOCMBIC` | `#define TIOCMBIC 0x5417` | `0x5417` | 清除 modem 位 |
| `TIOCMSET` | `termios.TIOCMSET` | `#define TIOCMSET 0x5418` | `0x5418` | 全量设置 modem |
| `TCSBRK` | `termios.TCSBRK` | `#define TCSBRK 0x5409` | `0x5409` | tcdrain 的底层 ioctl |
| `TCFLSH` | `termios.TCFLSH` | `#define TCFLSH 0x540B` | `0x540B` | 刷新缓冲区 |

---

## 九、源码路径汇总

| 层次 | 组件 | 文件路径 |
|------|------|----------|
| pyserial | 串口读写 | `serial/serialposix.py` |
| CPython | os.read / os.write | `cpython/Modules/posixmodule.c:11950 / 12295` |
| CPython | _Py_read / _Py_write | `cpython/Python/fileutils.c:1865 / 1924` |
| CPython | select.select | `cpython/Modules/selectmodule.c:278` |
| CPython | termios.tcdrain | `cpython/Modules/termios.c:324` |
| CPython | fcntl.ioctl | `cpython/Modules/fcntlmodule.c:197` |
| glibc | __libc_read / __libc_write | `glibc-2.42/sysdeps/unix/sysv/linux/read.c / write.c` |
| glibc | __select64 (= pselect6) | `glibc-2.42/sysdeps/unix/sysv/linux/select.c` |
| glibc | __libc_tcdrain (= ioctl) | `glibc-2.42/sysdeps/unix/sysv/linux/tcdrain.c` |
| glibc | __ioctl (ARM64 asm) | `glibc-2.42/sysdeps/unix/sysv/linux/aarch64/ioctl.S` |
| glibc | SYSCALL_CANCEL 宏 | `glibc-2.42/sysdeps/unix/sysdep.h:243-253` |
| 内核 | sys_read / sys_write | `linux/fs/read_write.c` |
| 内核 | sys_select / sys_pselect6 | `linux/fs/select.c:836 / 925` |
| 内核 | sys_ioctl → tty_ioctl | `linux/fs/ioctl.c` + `linux/drivers/tty/tty_ioctl.c` |
| 内核 | tty_read / tty_write | `linux/drivers/tty/tty_io.c` |
| 内核 | n_tty_read / n_tty_write | `linux/drivers/tty/n_tty.c` |
| 内核 | fd_set / timeval / termios | `linux/include/uapi/linux/posix_types.h` / `linux/time.h` / `linux/termbits.h` |
| 内核 | ioctl 常量 | `linux/include/uapi/asm-generic/ioctls.h` |
