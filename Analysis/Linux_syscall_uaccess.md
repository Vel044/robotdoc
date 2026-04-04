# Linux 系统调用参数传递与 `__user` 标注：以 `select`/`pselect6` 为例

本文以 `select` → `pselect6` 这条链路为例，**从 Python 序列化库（pyserial）开始**，逐层展开经过 CPython → glibc → 内核的完整调用链。

---

## 1. 调用全貌

### 完整路径总览（以 LeRobot 串口读写为例）

```
Python (用户代码层)
 │
 ├─ select 路径
 │    serial.Serial.read()
 │    └─ select.select(rlist, wlist, xlist, timeout)
 │         └─ CPython select_select_impl()
 │              └─ glibc __select64()
 │                   └─ pselect6 syscall → kernel
 │
 └─ read/write 路径
      serial.Serial.read()
      └─ os.read(fd, size)
           └─ CPython _Py_read(fd, buf, count)
                └─ glibc read(fd, buf, count)
                     └─ read syscall → kernel

内核（内核态层）
 └─ SYSCALL_DEFINE6(pselect6) / SYSCALL_DEFINE3(read)
      └─ do_pselect() / ksys_read()
           └─ core_sys_select() / vfs_read()
                └─ do_select() / 文件系统/驱动
                     └─ USB-Serial / V4L2 驱动
                          └─ 硬件（SCServo / 摄像头）
```

---

## 2. 逐层展开：select 路径

### 2.1 路径 A：pyserial → select（等待可读）

> LeRobot 串口通过 `select.select()` 判断串口数据是否可读。

源码路径：[venv/lib/python3.13/site-packages/serial/serialposix.py:565](../venv/lib/python3.13/site-packages/serial/serialposix.py#L565)

```python
def read(self, size=1):
    # ... 初始化 ...
    while len(read) < size:
        try:
            # 关键调用：select.select() 判断 fd 是否可读
            ready, _, _ = select.select([self.fd, self.pipe_abort_read_r], [], [], timeout.time_left())
            if self.pipe_abort_read_r in ready:
                os.read(self.pipe_abort_read_r, 1000)
                break
            if not ready:
                break   # timeout
            buf = os.read(self.fd, size - len(read))  # 读串口
        # ...
```

| 调用层次 | 函数 | 传入参数 |
|---------|------|--------|
| 第 1 层 | `select.select(rlist, wlist, xlist, timeout)` | Python list / float |
| 第 2 层 | CPython `select_select_impl()` | Python 对象 → C fd_set |
| 第 3 层 | glibc `__select64()` | fd_set*, timeval* |
| 第 4 层 | `pselect6` syscall | 6 个寄存器参数 |
| 第 5 层 | 内核 `SYSCALL_DEFINE6(pselect6)` | `__user` 指针处理 |

---

### 2.1.1 pyserial 关键调用：`select.select()` 的完整参数解析

源码路径：[venv/lib/python3.13/site-packages/serial/serialposix.py:565](../venv/lib/python3.13/site-packages/serial/serialposix.py#L565)

```python
ready, _, _ = select.select(
    [self.fd, self.pipe_abort_read_r],   # rlist（读事件监视列表）
    [],                                    # wlist（写事件监视列表，空）
    [],                                    # xlist（异常事件监视列表，空）
    timeout.time_left()                    # timeout（超时，秒为单位）
)
```

#### 传入参数详解

| 位置 | 表达式 | Python 类型 | 实际值含义 |
|------|--------|-----------|----------|
| arg1 | `[self.fd, self.pipe_abort_read_r]` | `list[int]` | 要监视的 **fd 列表**。`self.fd` = 串口文件描述符（`int`）；`self.pipe_abort_read_r` = 中断管道的读端 fd（用于取消阻塞） |
| arg2 | `[]` | `list[int]` | 写监视列表，此处为空（不需要监视写事件） |
| arg3 | `[]` | `list[int]` | 异常监视列表，此处为空 |
| arg4 | `timeout.time_left()` | `float` 或 `None` | 剩余超时秒数（`float`），`None` 表示无限等待 |

#### `Timeout.time_left()` 的实现

源码路径：[venv/lib/python3.13/site-packages/serial/serialutil.py:141-154](../venv/lib/python3.13/site-packages/serial/serialutil.py#L141)

```python
class Timeout(object):
    def __init__(self, duration):
        self.is_infinite = (duration is None)     # None → 无限等待
        self.is_non_blocking = (duration == 0)    # 0 → 非阻塞
        self.duration = duration                    # 原始超时秒数（float）
        self.target_time = self.TIME() + duration  # 到期时间 = 当前单调时间 + 超时

    def time_left(self):
        if self.is_non_blocking:
            return 0          # 非阻塞，立即返回
        elif self.is_infinite:
            return None       # 无限等待
        else:
            delta = self.target_time - self.TIME()  # 剩余 = 到期 - 当前
            return max(0, delta)
```

**返回值**：

| 返回值位置 | 类型 | 含义 |
|-----------|------|------|
| `ready` | `list[int]` | **已就绪**的读 fd 列表。LeRobot 里通常是 `[self.fd]`（数据可读）或 `[self.pipe_abort_read_r]`（要中断） |
| `_`（wlist） | `list[int]` | 已就绪的写 fd 列表，此处忽略 |
| `_`（xlist） | `list[int]` | 已就绪的异常 fd 列表，此处忽略 |

---

### 2.1.2 CPython：`select_select_impl()` 完整参数解析

源码路径：[cpython/Modules/selectmodule.c:278](../cpython/Modules/selectmodule.c#L278)

```c
static PyObject *
select_select_impl(PyObject *module, 
                  PyObject *rlist, 
                  PyObject *wlist,
                  PyObject *xlist, 
                  PyObject *timeout_obj)
```

#### 传入参数详解

| 位置 | 形参名 | CPython C 类型 | Python 传入值 | 说明 |
|------|--------|--------------|-------------|------|
| arg1 | `module` | `PyObject *` | 模块对象（`select` 模块） | CPython 模块对象 |
| arg2 | `rlist` | `PyObject *` | `list[int]` | 读监视 fd 列表 |
| arg3 | `wlist` | `PyObject *` | `list[int]` | 写监视 fd 列表 |
| arg4 | `xlist` | `PyObject *` | `list[int]` | 异常监视 fd 列表 |
| arg5 | `timeout_obj` | `PyObject *` | `float` / `int` / `None` | 超时（秒），对应 `timeout.time_left()` |

#### 内部关键变量

```c
// cpython/Modules/selectmodule.c:296-300
fd_set ifdset, ofdset, efdset;     // ★ fd_set = 位图结构，共 1024 bits
struct timeval tv, *tvp;            // ★ struct timeval = 超时时间结构
int imax, omax, emax, max;          // 各列表最大 fd 编号 + 1
PyTime_t timeout, deadline = 0;     // ★ PyTime_t = int64_t（纳秒级精度）
```

#### CPython 内部类型定义

```c
// cpython/Include/cpython/pytime.h:10
typedef int64_t PyTime_t;   // ★ CPython 内部时间类型，始终是 int64_t（纳秒）
```

#### `struct timeval` 定义

```c
// linux/include/uapi/linux/time.h:17
struct timeval {
    __kernel_old_time_t tv_sec;      // ★ 秒（32/64 位取决于架构）
    __kernel_suseconds_t tv_usec;   // ★ 微秒（-999999 ~ 999999）
};
```

#### `fd_set` 定义

```c
// linux/include/uapi/linux/posix_types.h:25
typedef struct {
    unsigned long fds_bits[__FD_SETSIZE / (8 * sizeof(long))];
    // ARM64: sizeof(long) = 8 字节
    // __FD_SETSIZE = 1024
    // fds_bits[1024 / (8 * 8)] = fds_bits[16]
    // 每个 unsigned long = 64 bits，16 × 64 = 1024 bits
} __kernel_fd_set;

// glibc/linux 通过 typedef 使用
// linux/include/linux/types.h:20
typedef __kernel_fd_set fd_set;
```

**fd_set 本质**：一个 1024 位的固定大小位图。每 bit 对应一个 fd 编号（0~1023）。第 N 个 bit = 1 表示"监视第 N 个 fd"。

#### `pylist` 结构（CPython 内部）

```c
// cpython/Modules/selectmodule.c:123-127
typedef struct {
    PyObject *obj;       // 指向 Python fd 对象的引用
    SOCKET fd;           // ★ fd 编号（int）
    int sentinel;        // -1 = 哨兵（列表结尾标记）
} pylist;
```

`sentinel = -1` 表示 pylist 数组的结尾，`fd2obj[0].sentinel = -1` 初始化。

---

### 2.1.3 glibc：`__select64()` 完整参数解析

源码路径：[glibc-2.42/sysdeps/unix/sysv/linux/select.c:32-73](../glibc-2.42/sysdeps/unix/sysv/linux/select.c#L32)

```c
int
__select64 (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
            struct __timeval64 *timeout)
```

#### 传入参数详解

| 位置 | 形参名 | glibc C 类型 | 来源（CPython 传入值） | 说明 |
|------|--------|------------|--------------------|------|
| arg1 | `nfds` | `int` | `imax > omax ? imax : (emax > omax ? emax : omax)` | 最大监视 fd + 1 |
| arg2 | `readfds` | `fd_set *` | `&ifdset` | 读监视位图（**栈上** `fd_set` 的地址） |
| arg3 | `writefds` | `fd_set *` | `&ofdset` | 写监视位图 |
| arg4 | `exceptfds` | `fd_set *` | `&efdset` | 异常监视位图 |
| arg5 | `timeout` | `struct __timeval64 *` | `pts64`（指向栈上 `ts64`） | 超时时间（**注意不是 `timeval*`，是 `__timeval64*`**） |

#### glibc `__timeval64` 结构定义

```c
// glibc-2.42/include/struct___timeval64.h:11
#if __TIMESIZE == 64     // ARM64 上 __TIMESIZE == 64，__timeval64 == timeval
struct __timeval64 {
    __time64_t tv_sec;         // ★ 秒（64 位有符号整数）
    __suseconds64_t tv_usec;   // ★ 微秒（64 位有符号整数，范围 -999999~999999）
};
// __time64_t = __SQUAD_TYPE = long int (ARM64 64位平台上)
// __suseconds64_t = __SUSECONDS64_T_TYPE = __SQUAD_TYPE = long int
#endif
```

#### glibc 时间相关类型推导（ARM64）

```c
// glibc-2.42/bits/typesizes.h
__TIME_T_TYPE       = __SLONGWORD_TYPE = long int      // 64 位
__SUSECONDS64_T_TYPE = __SQUAD_TYPE    = long int      // 64 位

// 在 ARM64 (LP64) 上：
// sizeof(long) = 8 bytes
// __time64_t   = long int = int64_t
// __suseconds64_t = long int = int64_t

// 所以 __timeval64 在 ARM64 上等价于：
struct __timeval64 {
    int64_t tv_sec;    // 秒
    int64_t tv_usec;   // 微秒
};
```

#### 宏定义（glibc 时间换算）

```c
// glibc-2.42/include/time.h
#define USEC_PER_SEC    1000000L     // 1 秒 = 100 万微秒
#define NSEC_PER_SEC    1000000000L  // 1 秒 = 10 亿纳秒
#define NSEC_PER_USEC   1000L        // 1 微秒 = 1000 纳秒
#define INT64_MAX       9223372036854775807L  // 64 位有符号最大正整数
```

#### `SYSCALL_CANCEL()` 展开 → 调用 `pselect6_time64`

```c
// glibc/sysdeps/unix/sysdep.h:251
#define SYSCALL_CANCEL(...) \
    __SYSCALL_CANCEL_CALL(__VA_ARGS__)
// → __syscall_cancel(arg1,...,arg6, __NR_pselect6_time64)
// 最终发出：svc #0  (ARM64) 或 syscall (x86)
```

**`pselect6_time64` 调用**：

```c
int r = SYSCALL_CANCEL(pselect6_time64,
    nfds,           // arg1: int          → ARM64 x0
    readfds,        // arg2: fd_set*      → ARM64 x1
    writefds,       // arg3: fd_set*      → ARM64 x2
    exceptfds,      // arg4: fd_set*      → ARM64 x3
    pts64,          // arg5: timespec*    → ARM64 x4
    NULL            // arg6: sigmask ptr   → ARM64 x5
);
// x8 = __NR_pselect6_time64 (= 72 on ARM64)
```

#### 内核 `struct __kernel_timespec` 定义

```c
// linux/include/uapi/linux/time_types.h:7
struct __kernel_timespec {
    __kernel_time64_t  tv_sec;   // ★ 秒（64 位有符号）
    long long          tv_nsec;  // ★ 纳秒（64 位，0~999999999）
};
// __kernel_time64_t = __kernel_long_t = long int (ARM64)
```

#### glibc `__timespec64` 结构定义

```c
// glibc-2.42/include/struct___timespec64.h:15
#if __TIMESIZE == 64
// ARM64 上直接 alias 为 timespec
# define __timespec64 timespec
#else
struct __timespec64 {
    __time64_t tv_sec;    // ★ 秒
    // 小端序：tv_nsec(32bit) + padding(32bit) = 8 字节对齐
    __int32_t tv_nsec;    // ★ 纳秒（32 位！）
    __int32_t :32;        // 填充对齐
};
#endif
```

> **注意**：glibc 的 `__timespec64`（32 位平台）和内核的 `__kernel_timespec` 在 32 位平台上有细微差异（纳秒字段宽度不同），glibc 在 32 位平台上会做转换。ARM64 上两者等价。

---

### 2.1.4 内核：`SYSCALL_DEFINE6(pselect6)` 完整参数解析

源码路径：[linux/fs/select.c:925](../linux/fs/select.c#L925)

```c
SYSCALL_DEFINE6(pselect6, int, n,
                fd_set __user *, inp,
                fd_set __user *, outp,
                fd_set __user *, exp,
                struct __kernel_timespec __user *, tsp,
                void __user *, sig)
```

#### 传入参数详解

| 位置 | 形参名 | 内核 C 类型 | glibc 传入值 | 是否 `__user` | 访问方式 |
|------|--------|-----------|------------|-------------|---------|
| arg1 | `n` | `int` | `nfds` | **否**（标量） | 直接使用 |
| arg2 | `inp` | `fd_set __user *` | `readfds`（栈地址） | **是** | `get_fd_set()` = `copy_from_user` |
| arg3 | `outp` | `fd_set __user *` | `writefds`（栈地址） | **是** | `get_fd_set()` = `copy_from_user` |
| arg4 | `exp` | `fd_set __user *` | `exceptfds`（栈地址） | **是** | `get_fd_set()` = `copy_from_user` |
| arg5 | `tsp` | `struct __kernel_timespec __user *` | `pts64`（栈地址） | **是** | `get_timespec64()` = `get_user` |
| arg6 | `sig` | `void __user *` | `NULL` | **是** | `get_sigset_argpack()` |

#### 关键：`__user` 标注的含义

`__user` 是 Linux 内核的**地址空间标注（address-space annotation）**，告诉内核开发者：
- 这个指针**来自用户态**，地址属于用户进程虚拟地址空间
- **不能直接解引用**（`*inp` 是非法操作）
- 必须通过 `copy_from_user` / `copy_to_user` / `get_user` 系列函数访问

#### 内核如何访问 `__user` 指针

```c
// linux/fs/select.c:783-785
if ((ret = get_fd_set(n, inp, fds.in)) ||   // 从用户 inp 复制 fd_set
    (ret = get_fd_set(n, outp, fds.out)) ||  // 从用户 outp 复制 fd_set
    (ret = get_fd_set(n, exp, fds.ex)))       // 从用户 exp 复制 fd_set
    goto out;

// linux/fs/select.c:802-805
if (set_fd_set(n, inp, fds.res_in) ||   // 把结果从内核写回用户 inp
    set_fd_set(n, outp, fds.res_out) ||  // 把结果从内核写回用户 outp
    set_fd_set(n, exp, fds.res_ex))      // 把结果从内核写回用户 exp
    ret = -EFAULT;
```

其中 `get_fd_set()` 和 `set_fd_set()` 就是 `copy_from_user` / `copy_to_user` 的封装。

---

### 2.1.5 完整数据流：从 pyserial 到内核

```
Python                              CPython栈                      glibc栈                    内核
─────────────────────────────────────────────────────────────────────────────────────────────────────

select.select(
    [self.fd, ...],                rlist (PyObject*) ─────────────────────────────────────►
    [],                             wlist (PyObject*) ─────────────────────────────────────►
    [],                             xlist (PyObject*) ─────────────────────────────────────►
    timeout.time_left() → float     timeout_obj (PyObject*) ──────────────────────────────►
)                                     │
                                     │ seq2set() 把 Python list → fd_set 位图
                                     ▼
                                 fd_set ifdset ───────────→ &ifdset (fd_set*) ──────────────► inp (fd_set __user *)
                                 fd_set ofdset ───────────→ &ofdset (fd_set*) ─────────────► outp (fd_set __user *)
                                 fd_set efdset ───────────→ &efdset (fd_set*) ─────────────► exp  (fd_set __user *)
                                     │
                                     │ _PyTime_FromSecondsObject() + _PyTime_AsTimeval()
                                     ▼
                                 struct timeval tv ──→ pts64=&ts64 ─────────────────────────► tsp (timespec __user *)
                                     │
                                     │ 内核会修改 pts64 指向的内存（剩余时间写回）
                                     ▼
                            SYSCALL_CANCEL(pselect6_time64, nfds, &ifdset, &ofdset, &efdset, &ts64, NULL)
                                              ↓ svc #0，x8=72
                                     ┌──────────────────────────────────────────────────────────┐
                                     │  内核 SYSCALL_DEFINE6(pselect6, n, inp, outp, exp, tsp, sig) │
                                     │   ├─ n = nfds                                    (直接用) │
                                     │   ├─ inp = &ifdset   → get_fd_set() = copy_from_user       │
                                     │   ├─ outp = &ofdset  → get_fd_set() = copy_from_user       │
                                     │   ├─ exp  = &efdset   → get_fd_set() = copy_from_user       │
                                     │   ├─ tsp  = &ts64    → get_timespec64() = get_user        │
                                     │   └─ sig  = NULL     → 跳过 sigmask 设置               │
                                     │   ├─ do_select() 等待 fd 可读...                               │
                                     │   ├─ 内核写回剩余时间到 tsp 指向的 ts64                        │
                                     └──────────────────────────────────────────────────────────┘
                                              ↓
                                 pts64 指向的 ts64.tv_sec / ts64.tv_nsec 已被内核修改
                                              ↓
                            TIMESPEC_TO_TIMEVAL(timeout, pts64)
                                     │
                                     ▼
                                 返回值 (int) = 已就绪 fd 数量
```


### 2.3 完整参数对齐：select 路径（从 Python 到内核）

```
层  Python/CPython/glibc                          形参/实参类型              传内核时
───────────────────────────────────────────────────────────────────────────────────────────────
第1层  Python: select.select(rlist, wlist, xlist, timeout)
       ├─ rlist  = list of fd(int)                  Python list
       ├─ wlist  = list of fd(int)
       ├─ xlist  = list of fd(int)
       └─ timeout = float / None                   秒为单位的浮点数

第2层  CPython select_select_impl(rlist, wlist, xlist, timeout_obj)
       ├─ 从 Python list 构造 `fd_set ifdset`       C struct fd_set
       ├─ 从 timeout_obj 构造 `struct timeval tv`  C struct timeval
       └─ 调用 n = select(max, &ifdset, &ofdset, &efdset, &tv)

第3层  glibc __select64(nfds, readfds, writefds, exceptfds, timeout)
       ├─ readfds   = &ifdset   (fd_set*)            fd_set __user *
       ├─ writefds  = &ofdset   (fd_set*)            fd_set __user *
       ├─ exceptfds = &efdset   (fd_set*)            fd_set __user *
       ├─ timeout   = &tv       (struct __timeval64*) → pts64 → 栈上 ts64
       └─ 调用 SYSCALL_CANCEL(pselect6_time64, nfds, r,w,e, pts64, NULL)

第4层  内核 SYSCALL_DEFINE6(pselect6, n, inp,outp,exp, tsp, sig)
       ├─ n=arg1     = nfds      (int)                标量 → x0
       ├─ inp=arg2   = readfds   (fd_set*)            fd_set __user * → x1
       ├─ outp=arg3  = writefds  (fd_set*)            fd_set __user * → x2
       ├─ exp=arg4   = exceptfds (fd_set*)            fd_set __user * → x3
       ├─ tsp=arg5   = pts64     (timespec*)          timespec __user * → x4
       └─ sig=arg6   = NULL      (void*)              void __user * → x5
                                                         x8 = 72 (syscall nr)
                                                         svc #0  ← ARM64 syscall
```

### 2.5 `__select64` 全流程逐行解析

---

## 2. glibc：`__select64` 全流程逐行解析

源码路径：[glibc-2.42/sysdeps/unix/sysv/linux/select.c](glibc-2.42/sysdeps/unix/sysv/linux/select.c)

### 2.1 函数原型与传入参数一览

```c
int
__select64 (int nfds,
            fd_set *readfds, 
            fd_set *writefds, 
            fd_set *exceptfds,
            struct __timeval64 *timeout)
```

| 形参名 | 类型 | 来源（调用者） | 含义 |
|--------|------|--------------|------|
| `nfds` | `int` | 调用者传入 | 要监视的最大 fd 编号 +1。例如监视 fd 3/4/5，则 `nfds=6` |
| `readfds` | `fd_set *` | 调用者传入 | 指向用户态 `fd_set` 缓冲区的指针，要监视哪些 fd 的**可读**事件 |
| `writefds` | `fd_set *` | 调用者传入 | 指向用户态 `fd_set` 缓冲区的指针，要监视哪些 fd 的**可写**事件 |
| `exceptfds` | `fd_set *` | 调用者传入 | 指向用户态 `fd_set` 缓冲区的指针，要监视哪些 fd 的**异常**事件 |
| `timeout` | `struct __timeval64 *` | 调用者传入 | 指向用户态 `struct __timeval64` 的指针，设置等待超时。`NULL` 表示无限等待 |

> `fd_set` 本质是一个位图，Linux 通常用 1024 位的数组实现（`NFDBITS` = `sizeof(unsigned long) * 8`），每个比特位对应一个 fd 编号。



## 5. `__user` 标注在内核"起作用"的位置

### 5.1 内核定义时：`__user` 是给静态分析工具看的

```c
// linux/fs/select.c:925
SYSCALL_DEFINE6(pselect6, int, n,
    fd_set __user *, inp,   // ← 标注 inp 是用户态地址
    fd_set __user *, outp,  // ← 标注 outp 是用户态地址
    fd_set __user *, exp,   // ← 标注 exp 是用户态地址
    struct __kernel_timespec __user *, tsp,  // ← 标注 tsp 是用户态地址
    void __user *, sig)     // ← 标注 sig 是用户态地址
```

`__user` 是 Linux 内核的 **地址空间标注（address-space annotation）**，主要作用：
- **Sparse 静态检查**：告知 sparse "这个指针指向用户态地址"，sparse 检查你是否直接解引用
- **代码可读性**：人读代码时能立即识别"这是用户态传来的"
- **不改变机器码**：`__user` 本身不生成额外的机器指令，它只是给编译器/分析工具看的提示

### 5.2 内核调用时：哪些参数必须用 uaccess API 访问

| 参数 | 内核如何访问 | 是否必须用 uaccess API？ |
|------|------------|----------------------|
| `n` | 直接用 `int` 值 | **不是指针，无需 uaccess** |
| `inp` / `outp` / `exp` | `get_fd_set()` 读入，`set_fd_set()` 写回 | **必须**——通过 `copy_from_user` / `copy_to_user` |
| `tsp` | `get_timespec64()` | **必须**——通过 `get_user` / `copy_from_user` |
| `sig` | `get_sigset_argpack()` 读取 | **必须**——通过 `unsafe_get_user` |

也就是说：**所有 `__user` 标注的指针参数，在内核实现层必须通过 uaccess API 访问**。

### 5.3 glibc 调用时：`__user` 对 glibc 来说毫无意义

glibc 是**用户态代码**，它不知道什么是 `__user`：

```c
// glibc/sysdeps/unix/sysv/linux/pselect.c
// glibc 里没有 __user 这种东西，只有普通指针

__syscall_ulong_t data[2] = {
    (uintptr_t) sigmask,   // 普通用户态指针
    __NSIG_BYTES
};
// 直接把指针值放到寄存器里发起 syscall
SYSCALL_CANCEL (pselect6_time64, nfds, readfds, writefds, exceptfds,
                timeout, data);
```

glibc 的参数和内核定义一一对应：
- 内核说 `fd_set __user *` → glibc 传 `fd_set *`（**值完全一样，只是没有 `__user` 标注**）
- `__user` 是内核源码的内部标注，glibc 编译时完全看不到

---

**B. 某参数传 NULL，内核里还需要 `__user` 标注吗？**
> **仍然需要**。`__user` 标注的是"指针的来源地址空间"，不是"指针当前是否为空"。内核收到 NULL 后确实会检查跳过，但定义时仍然必须标 `__user`，因为：
> 1. 函数签名要告诉内核实现者"这是用户态地址，需要 uaccess"
> 2. sparse 工具要看 `__user` 来判断"这参数如果被解引用是否安全"
> 3. NULL 只是"不访问"的简写，不代表它不是用户态指针

### 6.2 总结：定义 vs 调用

| 场景 | `__user` 是否存在 | 结论 |
|------|----------------|------|
| 内核定义 `SYSCALL_DEFINE6(pselect6)` | **必须有**（4 个指针参数全标） | 规范要求 + 静态检查必需 |
| glibc 调用侧 | **不存在**（glibc 没有 `__user`） | glibc 是用户态代码，所有指针天然都是用户态指针 |
| 能否传 NULL 代替某个指针 | **可以**（内核会检查 `if (tsp)` / `if (from)` 跳过访问） | NULL 值本身不是省略 `__user` |
| 标量参数如 `n` / `sigsetsize` | **不需要 `__user`** | 不是指针，不涉及跨地址空间访问 |

### 6.3 内核里哪些参数"天然不需要 `__user`"

只有**标量值（scalar）** 不需要 `__user`：
- `int n`（fd 数量）—— 直接放寄存器，内核直接用
- `size_t sigsetsize`（信号集大小）—— 同上
- `enum poll_time_type type`（内部枚举）—— 同上

所有**指针类型**（指向用户态缓冲区的地址），**都应该标 `__user`**。

