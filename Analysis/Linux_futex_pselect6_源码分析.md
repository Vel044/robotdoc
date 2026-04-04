# Linux futex 和 pselect6 系统调用源码分析

**文档位置**: `workspace/docs/Linux_futex_pselect6_源码分析.md`

---

## 目录

1. [futex 系统调用](#1-futex-系统调用)
2. [pselect6 系统调用](#2-pselect6-系统调用)
3. [关键数据结构](#3-关键数据结构)
4. [内核编程常见问题](#4-内核编程常见问题)

---

## 1. futex 系统调用

### 1.1 系统调用入口

**源码位置**: `linux/kernel/futex/syscalls.c` (第 200-237 行)

```c
/**
 * SYSCALL_DEFINE6(futex, ...) - futex 系统调用入口
 * 
 * futex (Fast Userspace muTEX) 是 Linux 提供的用户空间同步原语，
 * 用于实现高性能的锁和条件变量。
 * 
 * 参数:
 *   uaddr  - 用户空间地址，指向一个 32 位整数 (futex word)
 *   op     - 操作码 (FUTEX_WAIT, FUTEX_WAKE, 等)
 *   val    - 操作相关的值
 *   utime  - 超时时间 (struct __kernel_timespec *)
 *   uaddr2 - 第二个用户空间地址 (用于某些操作)
 *   val3   - 第三个值 (用于某些操作)
 * 
 * 返回值: 成功返回 0 或唤醒的线程数，失败返回负错误码
 */
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

	// 调用核心处理函数
	ret = do_futex(uaddr, op, val, tp, uaddr2, (unsigned long)utime, val3);
	return ret;
}
```

### 1.2 核心处理函数 do_futex

**源码位置**: `linux/kernel/futex/syscalls.c` (第 116-166 行)

```c
/**
 * do_futex() - futex 核心处理函数
 * 
 * 根据操作码分发到不同的处理函数。
 */
long do_futex(u32 __user *uaddr, int op, u32 val, ktime_t *timeout,
		u32 __user *uaddr2, u32 val2, u32 val3)
{
	unsigned int flags = futex_to_flags(op);
	int cmd = op & FUTEX_CMD_MASK;

	// 检查 CLOCKRT 标志的有效性
	if (flags & FLAGS_CLOCKRT) {
		if (cmd != FUTEX_WAIT_BITSET &&
		    cmd != FUTEX_WAIT_REQUEUE_PI &&
		    cmd != FUTEX_LOCK_PI2)
			return -ENOSYS;
	}

	// 根据命令分发处理
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

### 1.3 futex 操作码定义

**源码位置**: `linux/include/uapi/linux/futex.h` (第 14-27 行)

```c
/*
 * futex 操作码 - 定义在 uapi 头文件中，用户空间可见
 */
#define FUTEX_WAIT		0	/* 等待 futex 值变化 */
#define FUTEX_WAKE		1	/* 唤醒等待的线程 */
#define FUTEX_FD		2	/* 创建文件描述符 (已废弃) */
#define FUTEX_REQUEUE		3	/* 将等待者从一个 futex 移到另一个 */
#define FUTEX_CMP_REQUEUE	4	/* 带比较的重排队 */
#define FUTEX_WAKE_OP		5	/* 带原子操作的唤醒 */
#define FUTEX_LOCK_PI		6	/* 获取 PI (Priority Inheritance) 锁 */
#define FUTEX_UNLOCK_PI		7	/* 释放 PI 锁 */
#define FUTEX_TRYLOCK_PI	8	/* 尝试获取 PI 锁 */
#define FUTEX_WAIT_BITSET	9	/* 带位掩码的等待 */
#define FUTEX_WAKE_BITSET	10	/* 带位掩码的唤醒 */
#define FUTEX_WAIT_REQUEUE_PI	11	/* 带 PI 的重排队等待 */
#define FUTEX_CMP_REQUEUE_PI	12	/* 带比较和 PI 的重排队 */
#define FUTEX_LOCK_PI2		13	/* PI 锁 v2 */

/* 标志位 */
#define FUTEX_PRIVATE_FLAG	128	/* 进程私有 futex */
#define FUTEX_CLOCK_REALTIME	256	/* 使用实时时钟 */
#define FUTEX_CMD_MASK		~(FUTEX_PRIVATE_FLAG | FUTEX_CLOCK_REALTIME)
```

### 1.4 futex_wait 实现

**源码位置**: `linux/kernel/futex/waitwake.c`

futex_wait 是 futex 最核心的操作，它使调用线程进入睡眠状态，直到：
1. 被 futex_wake 唤醒
2. 超时
3. 被信号中断

```c
/**
 * futex_wait() - 等待 futex 值变化
 * 
 * 工作流程:
 * 1. 获取 futex 哈希桶锁
 * 2. 检查用户空间 futex 值是否等于预期值
 * 3. 如果不相等，立即返回 -EWOULDBLOCK
 * 4. 如果相等，将当前任务加入等待队列
 * 5. 释放锁并进入睡眠
 * 6. 被唤醒后，从等待队列移除
 */
int futex_wait(u32 __user *uaddr, unsigned int flags, u32 val,
	       ktime_t *abs_time, u32 bitset)
{
	struct futex_hash_bucket *hb;
	struct futex_q q = futex_q_init;
	int ret;

	// 验证输入参数
	if (!bitset)
		return -EINVAL;

	// 获取哈希桶
	hb = futex_hash(&q.key);
	
	// 加锁并检查 futex 值
	spin_lock(&hb->lock);
	
	// 读取用户空间值
	if (futex_get_value(&q.key, uaddr, flags)) {
		spin_unlock(&hb->lock);
		return -EFAULT;
	}
	
	// 检查值是否已改变
	if (q.key.val != val) {
		spin_unlock(&hb->lock);
		return -EWOULDBLOCK;  // 值已改变，不需要等待
	}
	
	// 设置等待队列项
	q.bitset = bitset;
	q.flags = flags;
	
	// 加入等待队列
	futex_enqueue(&q, hb);
	
	// 设置超时
	if (abs_time) {
		hrtimer_init_sleeper(&q.t, CLOCK_MONOTONIC, HRTIMER_MODE_ABS);
		hrtimer_set_expires(&q.t.timer, *abs_time);
	}
	
	// 释放锁并进入睡眠
	spin_unlock(&hb->lock);
	
	// 等待被唤醒或超时
	ret = futex_wait_queue(hb, &q, abs_time);
	
	// 清理
	futex_unqueue(&q);
	
	return ret;
}
```

### 1.5 futex_wake 实现

**源码位置**: `linux/kernel/futex/waitwake.c`

```c
/**
 * futex_wake() - 唤醒等待 futex 的线程
 * 
 * 工作流程:
 * 1. 获取 futex 哈希桶锁
 * 2. 遍历等待队列
 * 3. 唤醒匹配 bitset 的等待者
 * 4. 最多唤醒 nr_wake 个线程
 */
int futex_wake(u32 __user *uaddr, unsigned int flags, int nr_wake, u32 bitset)
{
	struct futex_hash_bucket *hb;
	struct futex_q *this, *next;
	int ret = 0;

	if (!bitset)
		return -EINVAL;

	// 获取哈希桶
	hb = futex_hash(&key);
	
	spin_lock(&hb->lock);
	
	// 遍历等待队列，唤醒匹配的线程
	list_for_each_entry_safe(this, next, &hb->chain, list) {
		if (match_futex(&this->key, &key)) {
			if (this->bitset & bitset) {
				wake_futex(this);
				if (++ret >= nr_wake)
					break;
			}
		}
	}
	
	spin_unlock(&hb->lock);
	return ret;
}
```

---

## 2. pselect6 系统调用

### 2.1 系统调用入口

**源码位置**: `linux/fs/select.c` (第 925-948 行)

```c
/**
 * SYSCALL_DEFINE6(pselect6, ...) - pselect6 系统调用入口
 * 
 * pselect6 是 POSIX select 的扩展版本，提供以下特性：
 * 1. 支持纳秒级精度的 timespec 超时（而非 select 的 timeval 微秒级）
 * 2. 支持第6个参数同时传递信号掩码和掩码大小，实现原子操作
 * 3. 等待期间可以原子地替换信号掩码
 * 
 * 参数:
 *   n      - 最大文件描述符编号 + 1
 *   inp    - 读取文件描述符集合 (fd_set *)
 *   outp   - 写入文件描述符集合
 *   exp    - 异常条件文件描述符集合
 *   tsp    - 超时时间 (struct __kernel_timespec *)
 *   sig    - 信号掩码参数 (包含 sigset_t 指针和 size_t 大小)
 * 
 * 返回值: 已就绪的文件描述符数量，0 表示超时，-1 表示错误
 */
SYSCALL_DEFINE6(pselect6, int, n, fd_set __user *, inp, fd_set __user *, outp,
		fd_set __user *, exp, struct __kernel_timespec __user *, tsp,
		void __user *, sig)
{
	struct sigset_argpack x = {NULL, 0};

	// 从用户空间获取信号掩码参数
	if (get_sigset_argpack(&x, sig))
		return -EFAULT;

	return do_pselect(n, inp, outp, exp, tsp, x.p, x.size, PT_TIMESPEC);
}
```

### 2.2 核心处理函数 do_pselect

**源码位置**: `linux/fs/select.c` (第 842-872 行)

```c
/**
 * do_pselect() - pselect 核心处理函数
 * 
 * 工作流程:
 * 1. 处理超时参数
 * 2. 设置用户信号掩码
 * 3. 调用 core_sys_select 执行实际的 select 操作
 * 4. 恢复原始信号掩码
 */
static long do_pselect(int n, fd_set __user *inp, fd_set __user *outp,
		       fd_set __user *exp, void __user *tsp,
		       const sigset_t __user *sigmask, size_t sigsetsize,
		       enum poll_time_type type)
{
	struct timespec64 ts, end_time, *to = NULL;
	int ret;

	// 处理超时参数
	if (tsp) {
		switch (type) {
		case PT_TIMESPEC:
			if (get_timespec64(&ts, tsp))
				return -EFAULT;
			break;
		case PT_OLD_TIMESPEC:
			if (get_old_timespec32(&ts, tsp))
				return -EFAULT;
			break;
		default:
			BUG();
		}

		to = &end_time;
		if (poll_select_set_timeout(to, ts.tv_sec, ts.tv_nsec))
			return -EINVAL;
	}

	// 设置用户信号掩码（原子操作）
	ret = set_user_sigmask(sigmask, sigsetsize);
	if (ret)
		return ret;

	// 执行核心 select 操作
	ret = core_sys_select(n, inp, outp, exp, to);
	
	// 完成并恢复信号掩码
	return poll_select_finish(&end_time, tsp, type, ret);
}
```

### 2.3 核心 select 实现

**源码位置**: `linux/fs/select.c` (第 650-750 行)

```c
/**
 * core_sys_select() - select 系统调用的核心实现
 * 
 * 工作流程:
 * 1. 验证文件描述符范围
 * 2. 从用户空间复制 fd_set
 * 3. 遍历所有文件描述符，调用 vfs_poll 检查状态
 * 4. 如果有就绪的 fd，立即返回
 * 5. 如果没有就绪的 fd，进入睡眠等待
 */
int core_sys_select(int n, fd_set __user *inp, fd_set __user *outp,
		    fd_set __user *exp, struct timespec64 *end_time)
{
	fd_set_bits fds;
	void *bits;
	int ret, max_fds;
	size_t size, alloc_size;
	struct fdtable *fdt;

	// 限制最大文件描述符数量
	if (n < 0)
		return -EINVAL;
	
	// 获取当前进程的文件描述符表
	fdt = files_fdtable(current->files);
	max_fds = fdt->max_fds;
	
	// 限制 n 不超过最大 fd
	if (n > max_fds)
		n = max_fds;

	// 计算 fd_set 大小
	size = FDS_BYTES(n);
	alloc_size = 6 * size;  // in, out, ex 的输入和输出

	// 分配内存
	bits = kvmalloc(alloc_size, GFP_KERNEL);
	if (!bits)
		return -ENOMEM;

	// 设置 fds 结构
	fds.in      = bits;
	fds.out     = bits +   size;
	fds.ex      = bits + 2*size;
	fds.res_in  = bits + 3*size;
	fds.res_out = bits + 4*size;
	fds.res_ex  = bits + 5*size;

	// 从用户空间复制 fd_set
	if ((ret = get_fd_set(n, inp, fds.in)) ||
	    (ret = get_fd_set(n, outp, fds.out)) ||
	    (ret = get_fd_set(n, exp, fds.ex)))
		goto out;

	// 清零结果集
	zero_fd_set(n, fds.res_in);
	zero_fd_set(n, fds.res_out);
	zero_fd_set(n, fds.res_ex);

	// 执行实际的 select 操作
	ret = do_select(n, &fds, end_time);

	// 将结果复制回用户空间
	if (ret < 0)
		goto out;
	if (!ret) {  // 超时
		ret = 0;
		goto out;
	}

	// 设置返回的 fd_set
	if (set_fd_set(n, inp, fds.res_in) ||
	    set_fd_set(n, outp, fds.res_out) ||
	    set_fd_set(n, exp, fds.res_ex))
		ret = -EFAULT;

out:
	kvfree(bits);
	return ret;
}
```

### 2.4 do_select 实现

**源码位置**: `linux/fs/select.c` (第 500-640 行)

```c
/**
 * do_select() - 遍历文件描述符并检查状态
 * 
 * 工作流程:
 * 1. 初始化 poll_wqueues 结构
 * 2. 遍历所有文件描述符
 * 3. 对每个 fd 调用 vfs_poll 获取状态
 * 4. 如果有就绪的 fd，计数并标记
 * 5. 如果没有就绪的 fd，进入睡眠
 * 6. 被唤醒后重新检查所有 fd
 */
static int do_select(int n, fd_set_bits *fds, struct timespec64 *end_time)
{
	struct poll_wqueues table;
	poll_table *wait;
	int retval, i, timed_out = 0;
	u64 slack = 0;

	// 初始化 poll 等待队列
	poll_initwait(&table);
	wait = &table.pt;
	
	// 如果有超时，计算 slack 时间
	if (end_time && !end_time->tv_sec && !end_time->tv_nsec) {
		wait = NULL;
		timed_out = 1;
	}

	if (end_time && !timed_out)
		slack = select_estimate_accuracy(end_time);

	retval = 0;
	
	// 主循环：遍历所有 fd
	for (;;) {
		unsigned long *rinp, *routp, *rexp, *inp, *outp, *exp;
		bool can_busy_loop = false;

		inp = fds->in; outp = fds->out; exp = fds->ex;
		rinp = fds->res_in; routp = fds->res_out; rexp = fds->res_ex;

		// 遍历所有文件描述符
		for (i = 0; i < n; ++rinp, ++routp, ++rexp) {
			unsigned long in, out, ex, all_bits, bit = 1, mask, j;
			unsigned long res_in = 0, res_out = 0, res_ex = 0;

			// 获取当前 word 的位掩码
			in = *inp++; out = *outp++; ex = *exp++;
			all_bits = in | out | ex;
			if (!all_bits) {
				i += BITS_PER_LONG;
				continue;
			}

			// 遍历 word 中的每一位
			for (j = 0; j < BITS_PER_LONG; ++j, ++i, bit <<= 1) {
				struct fd f;
				if (i >= n)
					break;
				if (!(bit & all_bits))
					continue;
				
				// 获取文件结构
				f = fdget(i);
				if (f.file) {
					// 调用 vfs_poll 检查文件状态
					const struct file_operations *f_op;
					f_op = f.file->f_op;
					mask = DEFAULT_POLLMASK;
					if (f_op->poll) {
						wait_key_set(wait, in, out, bit, busy_flag);
						mask = (*f_op->poll)(f.file, wait);
					}
					fdput(f);

					// 检查是否有就绪事件
					if ((mask & POLLIN_SET) && (in & bit)) {
						res_in |= bit;
						retval++;
					}
					if ((mask & POLLOUT_SET) && (out & bit)) {
						res_out |= bit;
						retval++;
					}
					if ((mask & POLLEX_SET) && (ex & bit)) {
						res_ex |= bit;
						retval++;
					}
				}
			}
			
			// 保存结果
			if (res_in)
				*rinp = res_in;
			if (res_out)
				*routp = res_out;
			if (res_ex)
				*rexp = res_ex;
		}
		
		// 如果有就绪的 fd，或者超时，或者收到信号，返回
		wait = NULL;
		if (retval || timed_out || signal_pending(current))
			break;
		
		// 如果没有就绪的 fd，进入睡眠
		if (table.error) {
			retval = table.error;
			break;
		}

		// 等待事件或超时
		if (!poll_schedule_timeout(&table, TASK_INTERRUPTIBLE,
					   to, slack))
			timed_out = 1;
	}

	poll_freewait(&table);
	return retval;
}
```

---

## 3. 关键数据结构

### 3.1 struct __kernel_timespec - 内核时间结构

**源码位置**: `linux/include/uapi/linux/time_types.h` (第 7-17 行)

```c
/**
 * struct __kernel_timespec - 内核时间结构
 * 
 * 用于在内核和用户空间之间传递时间值。
 * 使用 64 位秒值以支持 Y2038 安全（避免 32 位 time_t 的溢出问题）。
 * 
 * 字段:
 *   tv_sec  - 秒数 (64位，支持到公元2920亿年)
 *   tv_nsec - 纳秒数 (0-999999999)
 */
struct __kernel_timespec {
	__kernel_time64_t       tv_sec;                 /* seconds */
	long long               tv_nsec;                /* nanoseconds */
};
```

**为什么使用 64 位秒值？**
- 32 位 time_t 在 2038 年会溢出（Y2038 问题）
- 64 位 time_t 可以表示到约 2920 亿年
- 这是 POSIX 标准要求的 Y2038 安全解决方案

### 3.2 __kernel_fd_set - 文件描述符集合

**源码位置**: `linux/include/uapi/linux/posix_types.h` (第 23-27 行)

```c
/**
 * __kernel_fd_set - 文件描述符集合
 * 
 * 用于 select/poll 系统调用，表示一组文件描述符。
 * 每个位代表一个文件描述符，1 表示关注该 fd。
 * 
 * __FD_SETSIZE 默认是 1024，所以 fds_bits 数组大小为:
 * 1024 / (8 * sizeof(long)) = 1024 / 64 = 16 (64位系统)
 * 
 * 这限制了 select 能处理的最大 fd 数量为 1024。
 */
#undef __FD_SETSIZE
#define __FD_SETSIZE	1024

typedef struct {
	unsigned long fds_bits[__FD_SETSIZE / (8 * sizeof(long))];
} __kernel_fd_set;
```

**为什么用位图？**
- 节省内存：1024 个 fd 只需要 128 字节 (16 * 8)
- 快速操作：可以用位运算批量处理
- 传统设计：Unix 从一开始就使用这种表示

**用户空间映射**:
```c
// linux/include/linux/types.h (第 20 行)
typedef __kernel_fd_set		fd_set;
```

### 3.3 sigset_t - 信号集合

**源码位置**: `linux/include/uapi/asm-generic/signal.h` (第 59-63 行)

```c
/**
 * sigset_t - 信号集合
 * 
 * 用于表示一组信号，每个位代表一个信号。
 * _NSIG = 64 (标准信号 + 实时信号)
 * _NSIG_WORDS = 64 / 64 = 1 (64位系统)
 * 
 * 在 32 位系统上，_NSIG_WORDS = 64 / 32 = 2
 */
#define _NSIG		64
#define _NSIG_BPW	__BITS_PER_LONG
#define _NSIG_WORDS	(_NSIG / _NSIG_BPW)

typedef struct {
	unsigned long sig[_NSIG_WORDS];
} sigset_t;
```

**为什么需要信号掩码？**
- pselect 允许原子地替换信号掩码
- 防止在检查和等待之间收到信号导致的竞态条件
- 等待期间阻塞某些信号，唤醒后恢复原始掩码

### 3.4 struct sigset_argpack - pselect6 信号参数包

**源码位置**: `linux/fs/select.c` (第 898-903 行)

```c
/**
 * struct sigset_argpack - pselect6 信号参数包
 * 
 * 为什么需要这个结构？
 * 
 * 大多数架构的系统调用最多支持 6 个参数，而 pselect 需要 7 个：
 * 1. nfds
 * 2. readfds
 * 3. writefds
 * 4. exceptfds
 * 5. timeout
 * 6. sigmask
 * 7. sigsetsize
 * 
 * 解决方案：将最后两个参数打包成一个指针参数
 */
struct sigset_argpack {
	sigset_t __user *p;    /* 信号掩码指针 */
	size_t size;           /* 信号掩码大小 */
};
```

**参数获取函数**:
```c
// fs/select.c (第 905-922 行)
static inline int get_sigset_argpack(struct sigset_argpack *to,
				     struct sigset_argpack __user *from)
{
	if (from) {
		// 使用不安全但快速的获取方式（热路径优化）
		unsafe_get_user(to->p, &from->p, Efault);
		unsafe_get_user(to->size, &from->size, Efault);
		user_read_access_end();
	}
	return 0;
Efault:
	user_read_access_end();
	return -EFAULT;
}
```

### 3.5 struct futex_q - futex 等待队列项

**源码位置**: `linux/kernel/futex/futex.h` (第 100-150 行)

```c
/**
 * struct futex_q - futex 等待队列项
 * 
 * 表示一个正在等待 futex 的线程。
 * 存储在哈希桶的链表中，用于快速查找。
 */
struct futex_q {
	struct hlist_node list;           /* 哈希链表节点 */
	struct task_struct *task;         /* 等待的任务 */
	spinlock_t *lock_ptr;             /* 指向哈希桶锁的指针 */
	union futex_key key;              /* futex 键（用于哈希） */
	struct futex_pi_state *pi_state;  /* PI 状态（优先级继承） */
	struct rt_mutex_waiter *rt_waiter;/* RT 互斥量等待者 */
	union {
		struct futex_requeue requeue;
		struct futex_wait_woken wait_woken;
	};
	u32 bitset;                       /* 位掩码（用于 FUTEX_WAKE_BITSET） */
};
```

### 3.6 union futex_key - futex 键

**源码位置**: `linux/kernel/futex/futex.h` (第 80-98 行)

```c
/**
 * union futex_key - futex 键
 * 
 * 用于唯一标识一个 futex。
 * 使用联合体节省内存，根据 futex 类型选择不同的字段。
 * 
 * 两种类型的 futex：
 * 1. 共享 futex (FLAGS_SHARED)：使用 inode 和页偏移
 * 2. 私有 futex (FUTEX_PRIVATE)：使用 mm 和页偏移
 */
union futex_key {
	struct {
		u64 i_seq;            /* inode 序列号 */
		unsigned long pgoff;  /* 页内偏移 */
		unsigned int offset;  /* 字节偏移 */
	} shared;                     /* 共享 futex */
	
	struct {
		struct mm_struct *mm; /* 进程内存描述符 */
		unsigned long address;/* 虚拟地址 */
		unsigned int offset;  /* 字节偏移 */
	} private;                    /* 私有 futex */
	
	struct {
		u64 ptr;              /* 用于哈希计算的原始值 */
		unsigned long word;
		unsigned int offset;
	} both;
};
```

**为什么需要区分共享和私有？**
- 共享 futex：不同进程可以访问（通过文件映射）
- 私有 futex：仅同一进程的线程可以访问（匿名映射）
- 使用不同的键可以优化哈希查找

### 3.7 struct poll_wqueues - poll 等待队列

**源码位置**: `linux/include/linux/poll.h` (第 40-60 行)

```c
/**
 * struct poll_wqueues - poll/select 的等待队列管理
 * 
 * 管理一组 poll 等待队列项，用于在多个文件上等待 I/O 事件。
 */
struct poll_wqueues {
	poll_table pt;                    /* poll 表 */
	struct poll_table_page *table;    /* 动态分配的页链 */
	struct task_struct *polling_task; /* 正在 poll 的任务 */
	int triggered;                    /* 是否被触发 */
	int error;                        /* 错误码 */
	int inline_index;                 /* 内联条目索引 */
	struct poll_table_entry inline_entries[N_INLINE_POLL_ENTRIES];
};
```

### 3.8 struct poll_table_entry - poll 表项

**源码位置**: `linux/fs/select.c` (第 90-100 行)

```c
/**
 * struct poll_table_entry - 单个文件的 poll 等待项
 * 
 * 表示一个文件上的等待队列注册。
 */
struct poll_table_entry {
	struct file *filp;                /* 文件指针 */
	unsigned long key;                /* 关注的事件掩码 */
	wait_queue_entry_t wait;          /* 等待队列项 */
	wait_queue_head_t *wait_address;  /* 等待队列头 */
};
```

---

## 4. 内核编程常见问题

### 4.1 `__user` 标记详解

#### `__user` 是什么？

**`__user` 是一个地址空间标记宏**，定义在 `linux/include/linux/compiler.h` 中：

```c
// linux/include/linux/compiler.h
#define __user      __attribute__((noderef, address_space(1)))
```

#### 作用

| 特性 | 说明 |
|-----|------|
| **地址空间标记** | 标记指针指向**用户空间地址**（而非内核空间） |
| **编译器检查** | 让编译器进行地址空间检查，防止内核直接解引用用户空间指针 |
| **Sparse 工具** | 是 Sparse 静态检查工具的注解 |
| **安全保护** | 防止意外访问用户空间内存导致的安全问题 |

#### 为什么有的参数有 `__user`，有的没有？

| 参数类型 | 是否有 `__user` | 原因 |
|---------|---------------|------|
| **指针参数**（如 `u32 *uaddr`） | ✅ 有 | 指向用户空间内存，需要标记 |
| **值参数**（如 `u32 val`） | ❌ 没有 | 是寄存器传递的值，不是指针 |
| **结构体指针**（如 `struct __kernel_timespec *utime`） | ✅ 有 | 指向用户空间结构体 |
| **内核内部指针** | ❌ 没有 | 指向内核空间，不需要标记 |

#### 具体例子分析

```c
// linux/kernel/futex/syscalls.c (第 200-205 行)
SYSCALL_DEFINE6(futex, 
    u32 __user *, uaddr,      // ✅ 用户空间指针，需要 __user
                              //    uaddr 指向用户空间的 32 位整数
                              
    int, op,                  // ❌ 值类型，不是指针
                              //    op 是整数，通过寄存器传递
                              
    u32, val,                 // ❌ 值类型，不是指针
                              //    val 是整数，通过寄存器传递
                              
    const struct __kernel_timespec __user *, utime,  
                              // ✅ 用户空间结构体指针
                              //    utime 指向用户空间的时间结构
                              
    u32 __user *, uaddr2,     // ✅ 用户空间指针
                              //    uaddr2 是第二个用户空间地址
                              
    u32, val3                 // ❌ 值类型
                              //    val3 是整数，通过寄存器传递
)
```

#### 安全原因

```c
// ❌ 错误：内核不能直接访问用户空间内存
void bad_example(u32 __user *uaddr) {
    u32 value = *uaddr;  // 编译器会报错！
}

// ✅ 正确：必须使用安全函数
void good_example(u32 __user *uaddr) {
    u32 value;
    if (copy_from_user(&value, uaddr, sizeof(u32))) {
        return -EFAULT;  // 用户空间地址无效
    }
    // 现在可以安全使用 value
}
```

**为什么不能直接访问？**
1. **缺页风险**：用户空间地址可能未映射，直接访问会导致内核崩溃
2. **安全风险**：用户可能传入恶意构造的地址
3. **权限隔离**：内核和用户空间必须严格隔离

---

### 4.2 `u32` 和 SYSCALL_DEFINE 宏详解

#### `u32` 是什么？

**是的，`u32` 是无符号32位整数**：

```c
// linux/include/linux/types.h (第 23 行)
typedef unsigned int __u32;
typedef __u32 u32;
```

| 类型 | 定义 | 说明 |
|-----|------|------|
| `u8`  | `unsigned char` | 无符号8位 |
| `u16` | `unsigned short` | 无符号16位 |
| `u32` | `unsigned int` | 无符号32位 |
| `u64` | `unsigned long long` | 无符号64位 |
| `s8/s16/s32/s64` | 有符号版本 | 对应的有符号类型 |

#### SYSCALL_DEFINE6 宏展开

你看到的"奇怪"写法其实是 **宏定义的正常现象**。

**SYSCALL_DEFINE6 的定义**：

```c
// linux/include/linux/syscalls.h
#define SYSCALL_DEFINE6(name, ...) SYSCALL_DEFINEx(6, _##name, __VA_ARGS__)
```

**宏展开后的实际函数签名**：

```c
// 你看到的（宏写法）：
SYSCALL_DEFINE6(futex, u32 __user *, uaddr, int, op, u32, val, ...)

// 实际展开为（C 函数）：
long sys_futex(u32 __user *uaddr, int op, u32 val, ...)
```

#### 为什么看起来"合并"了？

**这是宏参数的写法，不是 C 语法**：

| 宏参数写法 | 实际含义 | 展开后 |
|-----------|---------|--------|
| `u32 __user *, uaddr` | 类型 + 变量名 | `u32 __user *uaddr` |
| `int, op` | 类型 + 变量名 | `int op` |
| `u32, val` | 类型 + 变量名 | `u32 val` |

**宏需要类型和变量名分开**，所以用逗号分隔。展开后就是正常的 C 语法。

#### 对比：普通函数 vs 系统调用宏

```c
// ==================== 普通 C 函数（你熟悉的写法）====================
long sys_futex(
    u32 __user *uaddr,           // 类型 变量名
    int op,                       // 类型 变量名
    u32 val,                      // 类型 变量名
    const struct __kernel_timespec __user *utime,
    u32 __user *uaddr2,
    u32 val3
);

// ==================== SYSCALL_DEFINE 宏写法 ====================
// 为了内核的元数据生成，使用宏定义
SYSCALL_DEFINE6(futex, 
    u32 __user *, uaddr,          // 类型, 变量名（注意逗号）
    int, op,                      // 类型, 变量名
    u32, val,                     // 类型, 变量名
    const struct __kernel_timespec __user *, utime,
    u32 __user *, uaddr2,
    u32, val3
)
```

#### 为什么要用宏？

**内核需要自动生成系统调用表和元数据**：

```c
// SYSCALL_DEFINE6 宏展开后还会生成：

// 1. 系统调用入口函数
long sys_futex(...) { ... }

// 2. 系统调用表项（用于系统调用分发）
// 在 arch/x86/entry/syscalls/syscall_64.tbl 中注册

// 3. 参数校验代码
// 自动检查用户空间指针有效性

// 4. 审计日志支持
// 记录系统调用信息

// 5. seccomp 过滤支持
// 允许 BPF 过滤系统调用

// 6. 跟踪点（tracepoint）支持
// 用于 ftrace 和 perf 分析
```

#### 宏展开的实际例子

```c
// 源码（宏写法）：
SYSCALL_DEFINE6(futex, 
    u32 __user *, uaddr,
    int, op,
    u32, val,
    const struct __kernel_timespec __user *, utime,
    u32 __user *, uaddr2,
    u32, val3
)
{
    // 函数体
}

// 预处理器展开后（伪代码）：
static inline long SYSC_futex(
    u32 __user *uaddr,
    int op,
    u32 val,
    const struct __kernel_timespec __user *utime,
    u32 __user *uaddr2,
    u32 val3
)
{
    // 函数体
}

// 系统调用入口包装
long sys_futex(...) {
    // 参数打包和解包
    // 审计日志
    // seccomp 检查
    // 调用 SYSC_futex
}
```

---

## 5. 文件索引

| 文件 | 位置 | 说明 |
|------|------|------|
| `syscalls.c` | `linux/kernel/futex/syscalls.c` | futex 系统调用入口 |
| `futex.h` | `linux/kernel/futex/futex.h` | futex 内部数据结构 |
| `waitwake.c` | `linux/kernel/futex/waitwake.c` | futex 等待/唤醒实现 |
| `select.c` | `linux/fs/select.c` | select/pselect 实现 |
| `futex.h` | `linux/include/uapi/linux/futex.h` | futex UAPI 定义 |
| `time_types.h` | `linux/include/uapi/linux/time_types.h` | 时间结构定义 |
| `posix_types.h` | `linux/include/uapi/linux/posix_types.h` | fd_set 定义 |
| `signal.h` | `linux/include/uapi/asm-generic/signal.h` | sigset_t 定义 |
| `types.h` | `linux/include/linux/types.h` | 内核类型定义 |

---

## 6. 总结

### futex 设计要点

1. **用户空间快速路径**：无竞争时完全在用户空间完成，无需系统调用
2. **内核空间慢速路径**：有竞争时进入内核等待队列
3. **哈希表组织**：使用哈希桶管理等待队列，O(1) 查找
4. **优先级继承**：支持 PI futex 防止优先级反转

### pselect6 设计要点

1. **原子信号掩码**：等待期间原子替换信号掩码，避免竞态
2. **纳秒精度**：使用 timespec 替代 timeval，精度更高
3. **参数打包**：通过结构体指针绕过 6 参数限制
4. **poll 机制**：底层使用 vfs_poll 统一处理不同文件类型
