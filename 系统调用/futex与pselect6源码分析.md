# futex 与 pselect6 系统调用源码分析

> 本文档整理自 Linux 内核源码，详细分析 futex 和 pselect6 两个系统调用的入口、接口语义、核心数据结构。
> 
> 源码路径：`/Users/vel-virtual/Lerobot/linux/`

---

## 目录

1. [futex 系统调用](#1-futex-系统调用)
2. [pselect6 系统调用](#2-pselect6-系统调用)
3. [相关数据结构](#3-相关数据结构)
4. [调用链路图](#4-调用链路图)

---

## 1. futex 系统调用

### 1.1 系统调用入口

**源码位置**：`kernel/futex/syscalls.c:188`

```c
SYSCALL_DEFINE6(futex, u32 __user *, uaddr, int, op, u32, val,
        const struct __kernel_timespec __user *, utime,
        u32 __user *, uaddr2, u32, val3)
{
    int ret, cmd = op & FUTEX_CMD_MASK;
    ktime_t t, *tp = NULL;
    struct timespec64 ts;

    // 处理超时参数
    if (utime && futex_cmd_has_timeout(cmd)) {
        if (get_timespec64(&ts, utime))
            return -EFAULT;
        ret = futex_init_timeout(cmd, op, &ts, &t);
        if (ret)
            return ret;
        tp = &t;
    }

    // 调用核心实现
    ret = do_futex(uaddr, op, val, tp, uaddr2, (unsigned long)utime, val3);
    return ret;
}
```

### 1.2 接口语义

**函数原型**：
```c
long sys_futex(u32 *uaddr, int op, u32 val, 
               struct __kernel_timespec *utime,
               u32 *uaddr2, u32 val3);
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `uaddr` | `u32 __user *` | futex 变量的用户空间地址 |
| `op` | `int` | 操作码（见操作类型表） |
| `val` | `u32` | 操作相关的值（等待时的期望值/唤醒时的数量） |
| `utime` | `struct __kernel_timespec *` | 超时时间（绝对或相对时间） |
| `uaddr2` | `u32 __user *` | 第二个 futex 地址（requeue 操作用） |
| `val3` | `u32` | 第三个值（比较唤醒用） |

### 1.3 操作类型定义

**源码位置**：`include/uapi/linux/futex.h`

```c
/* 基本操作 */
#define FUTEX_WAIT          0   /* 等待 futex 值变为期望值 */
#define FUTEX_WAKE          1   /* 唤醒指定数量的等待者 */
#define FUTEX_FD            2   /* 创建与 futex 关联的 fd（已废弃） */
#define FUTEX_REQUEUE       3   /* 重新排队等待者到另一个 futex */
#define FUTEX_CMP_REQUEUE   4   /* 比较后重新排队 */
#define FUTEX_WAKE_OP       5   /* 唤醒并执行原子操作 */

/* 优先级继承操作（PI futex） */
#define FUTEX_LOCK_PI       6   /* 获取 PI 锁 */
#define FUTEX_UNLOCK_PI     7   /* 释放 PI 锁 */
#define FUTEX_TRYLOCK_PI    8   /* 尝试获取 PI 锁 */

/* 位集操作 */
#define FUTEX_WAIT_BITSET   9   /* 带位集等待 */
#define FUTEX_WAKE_BITSET   10  /* 带位集唤醒 */

/* PI requeue 操作 */
#define FUTEX_WAIT_REQUEUE_PI   11  /* 等待并 requeue 到 PI 锁 */
#define FUTEX_CMP_REQUEUE_PI    12  /* 比较并 requeue 到 PI 锁 */
#define FUTEX_LOCK_PI2          13  /* PI 锁变体 */

/* 操作修饰符 */
#define FUTEX_PRIVATE_FLAG  128  /* 进程私有 futex（不共享） */
#define FUTEX_CLOCK_REALTIME 256 /* 使用实时时钟 */
#define FUTEX_CMD_MASK      (~(FUTEX_PRIVATE_FLAG | FUTEX_CLOCK_REALTIME))
```

### 1.4 核心实现函数

**源码位置**：`kernel/futex/syscalls.c:82`

```c
long do_futex(u32 __user *uaddr, int op, u32 val, ktime_t *timeout,
        u32 __user *uaddr2, u32 val2, u32 val3)
{
    unsigned int flags = futex_to_flags(op);
    int cmd = op & FUTEX_CMD_MASK;

    switch (cmd) {
    case FUTEX_WAIT:
        val3 = FUTEX_BITSET_MATCH_ANY;
        fallthrough;
    case FUTEX_WAIT_BITSET:
        return futex_wait(uaddr, flags, val, timeout, val3);
        
    case FUTEX_WAKE:
        val3 = FUTEX_BITSET_MATCH_ANY;
        fallthrough;
    case FUTEX_WAKE_BITSET:
        return futex_wake(uaddr, flags, val, val3);
        
    case FUTEX_REQUEUE:
        return futex_requeue(uaddr, flags, uaddr2, flags, val, val2, NULL, 0);
        
    case FUTEX_CMP_REQUEUE:
        return futex_requeue(uaddr, flags, uaddr2, flags, val, val2, &val3, 0);
        
    case FUTEX_WAKE_OP:
        return futex_wake_op(uaddr, flags, uaddr2, val, val2, val3);
        
    case FUTEX_LOCK_PI:
        flags |= FLAGS_CLOCKRT;
        fallthrough;
    case FUTEX_LOCK_PI2:
        return futex_lock_pi(uaddr, flags, timeout, 0);
        
    case FUTEX_UNLOCK_PI:
        return futex_unlock_pi(uaddr, flags);
        
    case FUTEX_TRYLOCK_PI:
        return futex_lock_pi(uaddr, flags, NULL, 1);
        
    case FUTEX_WAIT_REQUEUE_PI:
        val3 = FUTEX_BITSET_MATCH_ANY;
        return futex_wait_requeue_pi(uaddr, flags, val, timeout, val3, uaddr2);
        
    case FUTEX_CMP_REQUEUE_PI:
        return futex_requeue(uaddr, flags, uaddr2, flags, val, val2, &val3, 1);
    }
    return -ENOSYS;
}
```

### 1.5 futex 核心数据结构

#### 1.5.1 futex_key - futex 唯一标识

**源码位置**：`include/linux/futex.h`

```c
/*
 * futex 通过 key 来匹配。key 类型取决于是否为共享映射。
 * offset按 u32 大小（4字节）对齐，用低 2 位表示 key 类型：
 *   00: 私有进程 futex (PTHREAD_PROCESS_PRIVATE)
 *   01: 共享 futex，映射到文件
 *   10: 共享 futex，私有映射到 mm
 */
#define FUT_OFF_INODE    1  /* key 引用 inode */
#define FUT_OFF_MMSHARED 2  /* key 引用 mm */

union futex_key {
    struct {
        u64 i_seq;              /* inode 序列号 */
        unsigned long pgoff;    /* 页偏移 */
        unsigned int offset;    /* 页内偏移 */
    } shared;
    
    struct {
        union {
            struct mm_struct *mm;  /* 内存描述符 */
            u64 __tmp;
        };
        unsigned long address;     /* 虚拟地址 */
        unsigned int offset;       /* 页内偏移 */
    } private;
    
    struct {
        u64 ptr;               /* 指针值 */
        unsigned long word;    /* 字值 */
        unsigned int offset;   /* 偏移 */
    } both;
};
```

**为什么这样设计**：
- 私有 futex：通过 `mm + address` 标识，同一进程内唯一
- 共享 futex：通过 `inode + pgoff` 标识，跨进程唯一
- 使用 union 节省内存，三种视角访问同一数据

#### 1.5.2 futex_q - 等待队列条目

**源码位置**：`kernel/futex/futex.h`

```c
/**
 * struct futex_q - 哈希的 futex 等待队列条目，每个等待任务一个
 * @list:       按优先级排序的等待链表
 * @task:       等待的任务
 * @lock_ptr:   哈希桶锁
 * @key:        futex 被哈希的 key
 * @pi_state:   可选的优先级继承状态
 * @bitset:     位集唤醒用
 */
struct futex_q {
    struct plist_node list;      /* 优先级链表节点 */
    struct task_struct *task;    /* 等待进程 */
    spinlock_t *lock_ptr;        /* 指向哈希桶锁 */
    futex_wake_fn *wake;         /* 唤醒处理函数 */
    void *wake_data;             /* 唤醒数据 */
    union futex_key key;         /* futex 标识 */
    struct futex_pi_state *pi_state;  /* PI 状态 */
    struct rt_mutex_waiter *rt_waiter;
    union futex_key *requeue_pi_key;
    u32 bitset;                  /* 位集 */
    atomic_t requeue_state;
};
```

**为什么需要这个结构**：
- 一个 futex 可能被多个线程等待，每个线程一个 `futex_q`
- 按**优先级**排序（PI futex 需要优先唤醒高优先级线程）
- 包含唤醒函数指针，支持自定义唤醒逻辑

#### 1.5.3 futex_hash_bucket - 哈希桶

**源码位置**：`kernel/futex/futex.h`

```c
/*
 * 哈希桶由所有哈希到同一位置的 futex_key 共享。
 * 每个 key 可能有多个 futex_q 结构（每个等待任务一个）。
 */
struct futex_hash_bucket {
    atomic_t waiters;        /* 等待者数量 */
    spinlock_t lock;         /* 保护链表的自旋锁 */
    struct plist_head chain; /* 等待队列链表 */
} ____cacheline_aligned_in_smp;
```

**为什么这样设计**：
- 全局哈希表将 futex 分散到多个桶，减少锁竞争
- `____cacheline_aligned_in_smp`：每个桶独占缓存行，避免 false sharing
- `waiters` 原子计数用于快速判断是否有等待者

#### 1.5.4 futex_pi_state - 优先级继承状态

```c
/*
 * 优先级继承状态：
 * 当高优先级线程等待低优先级线程持有的锁时，
 * 临时提升持有者优先级，避免优先级反转。
 */
struct futex_pi_state {
    struct list_head list;         /* 链入任务的 pi_state_list */
    struct rt_mutex_base pi_mutex; /* 实时互斥锁 */
    struct task_struct *owner;     /* 锁持有者 */
    refcount_t refcount;           /* 引用计数 */
    union futex_key key;           /* 关联的 futex key */
} __randomize_layout;
```

**为什么需要 PI**：
- 解决优先级反转问题：低优先级持有锁，高优先级等待时
- 临时提升低优先级持有者到等待者的优先级
- 实时系统必需特性

---

## 2. pselect6 系统调用

### 2.1 系统调用入口

**源码位置**：`fs/select.c:925`

```c
SYSCALL_DEFINE6(pselect6, int, n, fd_set __user *, inp, fd_set __user *, outp,
        fd_set __user *, exp, struct __kernel_timespec __user *, tsp,
        void __user *, sig)
{
    struct sigset_argpack x = {NULL, 0};

    // 解析信号掩码参数
    if (get_sigset_argpack(&x, sig))
        return -EFAULT;

    return do_pselect(n, inp, outp, exp, tsp, x.p, x.size, PT_TIMESPEC);
}
```

### 2.2 接口语义

**函数原型**：
```c
long sys_pselect6(int n, fd_set *inp, fd_set *outp, fd_set *exp,
                  struct __kernel_timespec *tsp, void *sig);
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `n` | `int` | 最大文件描述符编号 + 1 |
| `inp` | `fd_set *` | 关心读事件的 fd 集合 |
| `outp` | `fd_set *` | 关心写事件的 fd 集合 |
| `exp` | `fd_set *` | 关心异常事件的 fd 集合 |
| `tsp` | `struct __kernel_timespec *` | 超时时间（纳秒精度） |
| `sig` | `void *` | 指向 `struct sigset_argpack` 的指针 |

### 2.3 sigset_argpack - 第6个参数的封装

**源码位置**：`fs/select.c:909`

```c
/*
 * 大多数架构无法处理 7 参数系统调用。
 * 所以第6个参数是指向结构的指针，包含 sigset_t 指针和大小。
 */
struct sigset_argpack {
    sigset_t __user *p;    /* 信号掩码指针 */
    size_t size;           /* sigset_t 大小 */
};
```

**为什么这样设计**：
- 大多数架构只支持最多 6 个系统调用参数
- pselect6 需要传递 7 个参数（信号掩码指针 + 掩码大小）
- 将这两个参数封装到一个结构体中，通过第 6 个参数传递

### 2.4 核心实现函数 do_select

**源码位置**：`fs/select.c:486`（带详细注释版本）

```c
static noinline_for_stack int do_select(int n, fd_set_bits *fds, 
                                         struct timespec64 *end_time)
{
    ktime_t expire, *to = NULL;
    struct poll_wqueues table;
    poll_table *wait;
    int retval, i, timed_out = 0;
    u64 slack = 0;
    
    // 第一步：获取有效的最大 fd 编号
    rcu_read_lock();
    retval = max_select_fd(n, fds);
    rcu_read_unlock();
    if (retval < 0)
        return retval;
    n = retval;

    // 第二步：初始化 poll_wqueues 结构
    poll_initwait(&table);
    wait = &table.pt;

    // 第三步：处理零超时情况（立即返回）
    if (end_time && !end_time->tv_sec && !end_time->tv_nsec) {
        wait->_qproc = NULL;
        timed_out = 1;
    }

    // 第四步：估算调度精度
    if (end_time && !timed_out)
        slack = select_estimate_accuracy(end_time);

    retval = 0;
    
    // 第五步：主循环
    for (;;) {
        unsigned long *rinp, *routp, *rexp, *inp, *outp, *exp;
        
        inp = fds->in; outp = fds->out; exp = fds->ex;
        rinp = fds->res_in; routp = fds->res_out; rexp = fds->res_ex;

        // 外层循环：遍历所有 fd 位图块
        for (i = 0; i < n; ++rinp, ++routp, ++rexp) {
            unsigned long in, out, ex, all_bits, bit = 1, j;
            unsigned long res_in = 0, res_out = 0, res_ex = 0;
            __poll_t mask;

            in = *inp++; out = *outp++; ex = *exp++;
            all_bits = in | out | ex;
            if (all_bits == 0) {
                i += BITS_PER_LONG;
                continue;
            }

            // 内层循环：遍历位图块中的每个 fd
            for (j = 0; j < BITS_PER_LONG; ++j, ++i, bit <<= 1) {
                struct fd f;
                if (i >= n)
                    break;
                if (!(bit & all_bits))
                    continue;

                mask = EPOLLNVAL;
                f = fdget(i);
                if (fd_file(f)) {
                    // 关键：调用 vfs_poll 检查文件状态
                    wait_key_set(wait, in, out, bit, busy_flag);
                    mask = vfs_poll(fd_file(f), wait);
                    fdput(f);
                }

                // 检查并记录就绪状态
                if ((mask & POLLIN_SET) && (in & bit)) {
                    res_in |= bit;
                    retval++;
                    wait->_qproc = NULL;
                }
                if ((mask & POLLOUT_SET) && (out & bit)) {
                    res_out |= bit;
                    retval++;
                    wait->_qproc = NULL;
                }
                if ((mask & POLLEX_SET) && (ex & bit)) {
                    res_ex |= bit;
                    retval++;
                    wait->_qproc = NULL;
                }
            }

            // 保存结果
            if (res_in)  *rinp = res_in;
            if (res_out) *routp = res_out;
            if (res_ex)  *rexp = res_ex;
        }

        wait->_qproc = NULL;
        
        // 检查退出条件
        if (retval || timed_out || signal_pending(current))
            break;
        if (table.error) {
            retval = table.error;
            break;
        }

        // 设置超时并睡眠
        if (end_time && !to) {
            expire = timespec64_to_ktime(*end_time);
            to = &expire;
        }

        // 关键：进入睡眠等待
        if (!poll_schedule_timeout(&table, TASK_INTERRUPTIBLE, to, slack))
            timed_out = 1;
    }

    poll_freewait(&table);
    return retval;
}
```

### 2.5 pselect6 vs select 的区别

| 特性 | select | pselect6 |
|------|--------|----------|
| 超时精度 | 微秒（timeval） | 纳秒（timespec） |
| 信号掩码 | 不支持 | 原子设置信号掩码 |
| 超时更新 | 修改为剩余时间 | 不修改（POSIX 要求） |
| 参数传递 | 5 个参数 | 6 个参数（sig 封装） |

---

## 3. 相关数据结构

### 3.1 fd_set - 文件描述符集合

**源码位置**：`include/uapi/linux/posix_types.h`

```c
/*
 * fd_set: 文件描述符集合
 * 
 * 每个 fd 占用 1 个 bit，总共支持 1024 个 fd
 * 位操作宏：FD_SET, FD_CLR, FD_ISSET, FD_ZERO
 */
#define __FD_SETSIZE    1024

typedef struct {
    unsigned long fds_bits[__FD_SETSIZE / (8 * sizeof(long))];
} __kernel_fd_set;

/* 在 64 位系统上：fds_bits[1024/64] = fds_bits[16] */
/* 每个 unsigned long 可以表示 64 个 fd */
```

**为什么是 1024**：
- 历史原因，早期系统足够用
- 超过 1024 需要用 poll 或 epoll

**位操作示意**：
```c
fd = 5;
// 设置 fd 5
fds_bits[5/64] |= (1UL << (5 % 64));  // 即 fds_bits[0] 的第 5 位

// 检查 fd 5
fds_bits[5/64] & (1UL << (5 % 64))
```

### 3.2 fd_set_bits - 内核使用的 fd 集合

**源码位置**：`fs/select.c:395`

```c
/*
 * 可扩展版本的 fd_set
 * 包含输入集合和输出（结果）集合
 */
typedef struct {
    unsigned long *in, *out, *ex;        /* 输入：读/写/异常 */
    unsigned long *res_in, *res_out, *res_ex;  /* 输出：就绪结果 */
} fd_set_bits;

/* 辅助宏 */
#define FDS_BITPERLONG  (8*sizeof(long))
#define FDS_LONGS(nr)   (((nr)+FDS_BITPERLONG-1)/FDS_BITPERLONG)
#define FDS_BYTES(nr)   (FDS_LONGS(nr)*sizeof(long))
```

### 3.3 fdtable - 文件描述符表

**源码位置**：`include/linux/fdtable.h`

```c
/*
 * fdtable: 进程的文件描述符表
 * 
 * 每个进程有一个 files_struct，其中包含 fdtable
 * fdtable 维护了 fd 到 file 结构的映射
 */
struct fdtable {
    unsigned int max_fds;              /* 最大 fd 数量 */
    struct file __rcu **fd;            /* fd 数组：fd[fd_number] = file* */
    unsigned long *close_on_exec;      /* exec 时关闭的 fd 位图 */
    unsigned long *open_fds;           /* 已打开的 fd 位图 */
    unsigned long *full_fds_bits;      /* 用于快速查找空闲 fd */
    struct rcu_head rcu;               /* RCU 回收用 */
};

/*
 * files_struct: 进程打开文件表
 */
struct files_struct {
    atomic_t count;                    /* 引用计数 */
    bool resize_in_progress;           /* 正在调整大小 */
    wait_queue_head_t resize_wait;     /* 调整等待队列 */

    struct fdtable __rcu *fdt;         /* 指向当前 fdtable */
    struct fdtable fdtab;              /* 内嵌的默认 fdtable */

    spinlock_t file_lock;              /* 保护整个结构 */
    unsigned int next_fd;              /* 下一个可用 fd */
    
    /* 内嵌的默认位图和 fd 数组 */
    unsigned long close_on_exec_init[1];
    unsigned long open_fds_init[1];
    unsigned long full_fds_bits_init[1];
    struct file __rcu *fd_array[NR_OPEN_DEFAULT];  /* 默认 64 个 */
};
```

**为什么有两层（files_struct + fdtable）**：
- 大多数进程打开的 fd 数量很少（< 64），使用内嵌的 `fdtab` 足够
- 当 fd 数量超过默认值时，动态分配更大的 fdtable
- RCU 保护实现无锁读取

### 3.4 poll_wqueues - poll 等待队列管理

**源码位置**：`include/linux/poll.h`

```c
/*
 * poll_wqueues: select/poll 的等待队列管理结构
 * 
 * 管理所有被监控 fd 的等待队列项
 */
struct poll_wqueues {
    poll_table pt;                     /* poll 表（包含回调函数） */
    struct poll_table_page *table;     /* 额外分配的页面链表 */
    struct task_struct *polling_task;  /* 执行 poll 的任务 */
    int triggered;                     /* 是否有 fd 就绪触发 */
    int error;                         /* 错误码 */
    int inline_index;                  /* 内嵌条目索引 */
    struct poll_table_entry inline_entries[N_INLINE_POLL_ENTRIES]; /* 内嵌条目 */
};

/*
 * poll_table_entry: 单个 fd 的等待队列条目
 */
struct poll_table_entry {
    struct file *filp;                 /* 文件指针 */
    __poll_t key;                      /* 感兴趣的事件掩码 */
    wait_queue_entry_t wait;           /* 等待队列条目 */
    wait_queue_head_t *wait_address;   /* 等待队列头 */
};
```

**工作原理**：
1. `poll_wqueues` 管理所有被监控 fd 的等待队列
2. 对每个 fd，创建一个 `poll_table_entry`，加入该 fd 的等待队列
3. 当 fd 就绪时，驱动程序调用 `wake_up`，唤醒等待进程
4. `triggered` 标志让进程从睡眠中醒来

### 3.5 __kernel_timespec - 时间结构

**源码位置**：`include/uapi/linux/time_types.h`

```c
/*
 * 64 位时间结构，解决 y2038 问题
 */
struct __kernel_timespec {
    __kernel_time64_t tv_sec;   /* 秒（64 位） */
    long long tv_nsec;          /* 纳秒 */
};
```

---

## 4. 调用链路图

### 4.1 futex 调用链路

```
用户空间
    │
    │ futex(&uaddr, FUTEX_WAIT, val, timeout, ...)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                        内核空间                               │
│                                                              │
│  SYSCALL_DEFINE6(futex)          ← kernel/futex/syscalls.c  │
│         │                                                    │
│         ▼                                                    │
│  do_futex()                       ← 核心分发函数              │
│         │                                                    │
│         ├─ FUTEX_WAIT ──► futex_wait()                       │
│         │                         │                          │
│         │                         ▼                          │
│         │                    futex_wait_setup()              │
│         │                         │                          │
│         │                         ▼                          │
│         │                    futex_queue()  ← 加入哈希桶      │
│         │                         │                          │
│         │                         ▼                          │
│         │                    futex_wait_queue() ← 睡眠等待    │
│         │                                                    │
│         ├─ FUTEX_WAKE ──► futex_wake()                       │
│         │                         │                          │
│         │                         ▼                          │
│         │                    futex_wake_mark() ← 标记唤醒     │
│         │                                                    │
│         └─ 其他操作...                                       │
│                                                              │
│  关键数据结构：                                               │
│  ├─ union futex_key       ← 唯一标识 futex                   │
│  ├─ struct futex_q        ← 每个等待者一个                    │
│  ├─ struct futex_hash_bucket ← 哈希桶                        │
│  └─ struct futex_pi_state ← 优先级继承状态                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 pselect6 调用链路

```
用户空间
    │
    │ pselect6(n, readfds, writefds, exceptfds, timeout, sigmask)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                        内核空间                               │
│                                                              │
│  SYSCALL_DEFINE6(pselect6)        ← fs/select.c              │
│         │                                                    │
│         ├─ get_sigset_argpack() ← 解析第6个参数               │
│         │                                                    │
│         ▼                                                    │
│  do_pselect()                                                 │
│         │                                                    │
│         ├─ get_timespec64()    ← 获取超时时间                 │
│         ├─ poll_select_set_timeout() ← 设置截止时间           │
│         ├─ set_user_sigmask()  ← 原子设置信号掩码             │
│         │                                                    │
│         ▼                                                    │
│  core_sys_select()                                           │
│         │                                                    │
│         ├─ 分配 fd_set_bits（栈上或堆上）                      │
│         ├─ get_fd_set()        ← 从用户空间复制 fd 集合        │
│         │                                                    │
│         ▼                                                    │
│  do_select()                    ← 核心实现                    │
│         │                                                    │
│         ├─ max_select_fd()     ← 获取有效最大 fd              │
│         ├─ poll_initwait()     ← 初始化等待队列               │
│         │                                                    │
│         │  ┌────────────────────────────────┐                │
│         │  │ 主循环 for(;;)                   │                │
│         │  │                                 │                │
│         │  │  遍历所有 fd:                    │                │
│         │  │    vfs_poll(fd, wait)           │                │
│         │  │         │                       │                │
│         │  │         ▼                       │                │
│         │  │    file->f_op->poll() ← 驱动poll│                │
│         │  │         │                       │                │
│         │  │         ▼                       │                │
│         │  │    检查返回的事件掩码             │                │
│         │  │    记录就绪的 fd                 │                │
│         │  │                                 │                │
│         │  │  如果没有就绪 fd:                │                │
│         │  │    poll_schedule_timeout() ← 睡眠│                │
│         │  │         │                       │                │
│         │  │         ▼                       │                │
│         │  │    schedule_hrtimeout_range()   │                │
│         │  │                                 │                │
│         │  └────────────────────────────────┘                │
│         │                                                    │
│         ▼                                                    │
│  poll_freewait()              ← 清理等待队列                  │
│         │                                                    │
│         ▼                                                    │
│  poll_select_finish()         ← 更新剩余时间并恢复信号掩码      │
│                                                              │
│  关键数据结构：                                               │
│  ├─ struct fd_set_bits       ← fd 集合包装                   │
│  ├─ struct poll_wqueues      ← 等待队列管理                   │
│  ├─ struct fdtable           ← 进程 fd 表                    │
│  └─ struct sigset_argpack    ← 信号掩码封装                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录：源码文件索引

| 文件 | 路径 | 主要内容 |
|------|------|----------|
| futex 系统调用 | `kernel/futex/syscalls.c` | sys_futex 入口、do_futex 分发 |
| futex 头文件 | `include/uapi/linux/futex.h` | 操作码定义、robust list 结构 |
| futex 内部头文件 | `kernel/futex/futex.h` | futex_q, futex_hash_bucket 定义 |
| futex 核心 | `include/linux/futex.h` | futex_key 定义 |
| pselect6 系统调用 | `fs/select.c` | sys_pselect6、do_select |
| poll 头文件 | `include/linux/poll.h` | poll_wqueues, poll_table_entry |
| fd 表头文件 | `include/linux/fdtable.h` | fdtable, files_struct |
| fd_set 定义 | `include/uapi/linux/posix_types.h` | __kernel_fd_set |
| 时间类型 | `include/uapi/linux/time_types.h` | __kernel_timespec |

---

*文档生成时间：2026-03-27*
*Linux 内核版本：基于源码目录分析*
