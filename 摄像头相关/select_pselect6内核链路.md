# select/pselect6 内核链路

本文从 glibc `select()` 调用 `pselect6` 系统调用开始，追踪 `sys_pselect6()` 如何经过 `core_sys_select()`、`do_select()`、`vfs_poll()`，最终到达进程调度器的 `schedule_timeout()`，使进程进入 `TASK_INTERRUPTIBLE` 睡眠。

触发背景：当 `ioctl(VIDIOC_DQBUF)` 返回 EAGAIN（非阻塞模式无帧可取）时，OpenCV 用 `select` 等待摄像头 fd 变为可读。

---

## 图：select/pselect6 syscall 到内核边界的分层路径

```
select(nfds, &readfds, NULL, NULL, &timeout)  # OpenCV tryIoctl 中 EAGAIN 时调用
  │
  ▼
glibc: __select64()                          # 为什么：glibc 把 select 适配到 pselect6 syscall
  │  把 struct timeval 转成 struct timespec，调用 pselect6 syscall
  ▼
syscall: pselect6(nfds, inp, outp, exp, tsp, sig)  # ARM64: x8=__NR_pselect6=72
  │
  ▼
sys_pselect6() [linux/fs/select.c:960]       # 什么：pselect6 syscall 入口；为什么：解析 sigmask 参数
  │  get_sigset_argpack(&x, sig) → sig=NULL 跳过
  ▼
do_pselect()                                 # 什么：设置信号掩码和超时；为什么：封装参数准备
  │  get_timespec64(tsp) → set_user_sigmask() → core_sys_select()
  ▼
core_sys_select()                            # 什么：把用户态 fd_set 复制到内核；为什么：小分配用栈大分配用 vmalloc
  │  get_fd_set() = copy_from_user(fd_set) → do_select() → set_fd_set() = copy_to_user(fd_set)
  ▼
do_select()                                  # 什么：核心 I/O 多路复用；为什么：遍历所有 fd，vfs_poll 检查就绪状态
  │  vfs_poll() 检查 fd → 无就绪则 poll_schedule_timeout() 进入睡眠
  ▼
poll_schedule_timeout()                     # 什么：设置进程状态并调用调度器；为什么：使进程从 RUNNING 转为睡眠
  │  set_current_state(TASK_INTERRUPTIBLE) → schedule_hrtimeout_range()
========== kernel sleep boundary ==========
  │  进程进入 TASK_INTERRUPTIBLE 睡眠，等待 wake_up() 或超时或信号唤醒
  ▼
wake_up(done_wq)                            # UVC 驱动在 URB 完成时调用
  │  uvc_video_complete() → vb2_buffer_done() → wake_up(&stream->queue.done_wq)
========== kernel wake boundary ==========
  │
  ▼
do_select() 继续循环                        # 再次 vfs_poll() 检查 fd 状态
  │  done_list 非空 → vfs_poll 返回 POLLIN → retval++
  ▼
返回: retval (就绪 fd 数量)                  # copy_to_user(fd_set) → 用户态
```

---

## 1. 系统调用入口

### 1.1 glibc `__select64()` — 用户态入口

```c
// glibc-2.42/sysdeps/unix/sysv/linux/select.c
// 函数名：__select64()
// 作用：glibc select() 的实现。实际上调用 pselect6 syscall。
// 参数：nfds=最大 fd+1, readfds=读 fd 集合, writefds=写 fd 集合,
//       exceptfds=异常 fd 集合, timeout=超时时间
// 返回值：就绪 fd 数量，0=超时，-1=错误
__select64 (int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds,
            struct __timeval64 *timeout)
{
  __time64_t s = timeout != NULL ? timeout->tv_sec : 0;
  int32_t us = timeout != NULL ? timeout->tv_usec : 0;
  ...
  struct __timespec64 ts64, *pts64 = NULL;
  if (timeout != NULL) {
    ts64.tv_sec = s;
    ts64.tv_nsec = ns;
    pts64 = &ts64;
  }

  // 实际走 pselect6 syscall（支持纳秒级超时）
  int r = SYSCALL_CANCEL (pselect6_time64, nfds, readfds, writefds, exceptfds,
                          pts64, NULL);  // 第6参数 sig=NULL（无 sigmask）
  ...
  return r;
}
```

ARM64 上 pselect6 系统调用参数：
```text
x0 = nfds                          (最大 fd + 1)
x1 = readfds                       (用户态 fd_set * 读集合)
x2 = writefds                      (用户态 fd_set * 写集合，通常 NULL)
x3 = exceptfds                     (用户态 fd_set * 异常集合，通常 NULL)
x4 = timeout                       (用户态 struct __kernel_timespec * 超时)
x5 = sig                           (sigmask 打包结构地址，NULL 表示不设置信号掩码)
x8 = __NR_pselect6 = 72            (ARM64 syscall 号)
svc #0
```

### 1.2 `sys_pselect6()` — 内核 syscall 入口

```c
// linux/fs/select.c:960
// 函数名：SYSCALL_DEFINE6(pselect6) 即 sys_pselect6()
// 作用：pselect6 系统调用入口。解析 sigmask 参数，调用 do_pselect。
// 参数：n=nfds, inp=读 fd 集合, outp=写 fd 集合, exp=异常 fd 集合,
//       tsp=超时 timespec, sig=sigmask 打包结构（{sigmask_ptr, size} 的地址）
// 返回值：>0=就绪 fd 数量，0=超时，-1=错误
SYSCALL_DEFINE6(pselect6, int, n, fd_set __user *, inp, fd_set __user *, outp,
		fd_set __user *, exp, struct __kernel_timespec __user *, tsp,
		void __user *, sig)
{
	struct sigset_argpack x = {NULL, 0};

	// 从用户态 sig 指针读取 sigmask 打包结构
	// sig → data[0]=sigmask*, data[1]=size
	// LeRobot select() 不设置 sigmask → sig=NULL → x={NULL, 0}
	if (get_sigset_argpack(&x, sig))
		return -EFAULT;

	// 调用核心处理函数
	return do_pselect(n, inp, outp, exp, tsp, x.p, x.size, PT_TIMESPEC);
}
```

### 1.3 `do_pselect()` — 参数准备和信号掩码处理

```c
// linux/fs/select.c:842
// 函数名：do_pselect()
// 作用：解析超时时间，设置信号掩码，调用 core_sys_select。
// 参数：n=nfds, inp/outp/exp=fd 集合, tsp=用户态超时结构,
//       sigmask=信号掩码指针, sigsetsize=信号掩码大小, type=时间类型
// 返回值：core_sys_select 的返回值
static long do_pselect(int n, fd_set __user *inp, fd_set __user *outp,
		       fd_set __user *exp, void __user *tsp,
		       const sigset_t __user *sigmask, size_t sigsetsize,
		       enum poll_time_type type)
{
	struct timespec64 ts, end_time, *to = NULL;
	int ret;

	if (tsp) {
		// 从用户态 timespec 读取超时时间
		switch (type) {
		case PT_TIMESPEC:
			if (get_timespec64(&ts, tsp))
				return -EFAULT;
			break;
		...
		}
		// 计算绝对超时时间（当前时间 + 相对超时）
		to = &end_time;
		if (poll_select_set_timeout(to, ts.tv_sec, ts.tv_nsec))
			return -EINVAL;
	}

	// 设置信号掩码（原子操作，避免竞态）
	ret = set_user_sigmask(sigmask, sigsetsize);
	if (ret)
		return ret;

	// 进入核心 select 处理
	ret = core_sys_select(n, inp, outp, exp, to);
	return poll_select_finish(&end_time, tsp, type, ret);
}
```

---

## 2. 核心 select 处理

### 2.1 `core_sys_select()` — fd 集合复制到内核

```c
// linux/fs/select.c:735
// 函数名：core_sys_select()
// 作用：分配内核位图，把用户态 fd_set 复制到内核，然后调用 do_select。
// 参数：n=nfds, inp/outp/exp=用户态 fd 集合, end_time=绝对超时时间
// 返回值：>0=就绪 fd 数量，0=超时，<0=错误
int core_sys_select(int n, fd_set __user *inp, fd_set __user *outp,
			   fd_set __user *exp, struct timespec64 *end_time)
{
	fd_set_bits fds;                   // 内核位图结构（6 个位图）
	void *bits;                        // 位图数据区
	int ret, max_fds;
	size_t size, alloc_size;
	struct fdtable *fdt;
	long stack_fds[SELECT_STACK_ALLOC/sizeof(long)]; // 栈上小型分配

	ret = -EINVAL;
	if (n < 0)                         // nfds 不能为负
		goto out_nofds;

	// 获取当前进程 fdtable 的 max_fds
	rcu_read_lock();
	fdt = files_fdtable(current->files);
	max_fds = fdt->max_fds;
	rcu_read_unlock();
	if (n > max_fds)                   // 不能超过进程最大 fd
		n = max_fds;

	// 计算位图大小：FDS_BYTES(n) = (n/8) 字节（向上取整）
	size = FDS_BYTES(n);
	bits = stack_fds;                 // 小型分配用栈
	if (size > sizeof(stack_fds) / 6) {
		// 大型分配用 kvmalloc（允许睡眠）
		alloc_size = 6 * size;
		bits = kvmalloc(alloc_size, GFP_KERNEL);
		if (!bits)
			goto out_nofds;
	}

	// fds_bits 结构：6 个位图（in/out/ex 各有输入和输出）
	//   fds.in      = bits
	//   fds.out     = bits + size
	//   fds.ex      = bits + 2*size
	//   fds.res_in  = bits + 3*size  （输出：就绪的读 fd）
	//   fds.res_out = bits + 4*size  （输出：就绪的写 fd）
	//   fds.res_ex  = bits + 5*size  （输出：就绪的异常 fd）
	fds.in      = bits;
	fds.out     = bits +   size;
	fds.ex      = bits + 2*size;
	fds.res_in  = bits + 3*size;
	fds.res_out = bits + 4*size;
	fds.res_ex  = bits + 5*size;

	// 把用户态 fd_set 复制到内核位图
	if ((ret = get_fd_set(n, inp, fds.in)) ||
	    (ret = get_fd_set(n, outp, fds.out)) ||
	    (ret = get_fd_set(n, exp, fds.ex)))
		goto out;

	// 初始化结果位图为 0
	zero_fd_set(n, fds.res_in);
	zero_fd_set(n, fds.res_out);
	zero_fd_set(n, fds.res_ex);

	// 进入核心轮询/等待
	ret = do_select(n, &fds, end_time);

	if (ret < 0)                       // do_select 出错
		goto out;
	if (!ret) {                         // 超时，无就绪 fd
		ret = -ERESTARTNOHAND;
		if (signal_pending(current))   // 但有信号等待
			goto out;
		ret = 0;
	}

	// 把结果位图复制回用户态
	if (set_fd_set(n, inp, fds.res_in) ||
	    set_fd_set(n, outp, fds.res_out) ||
	    set (n, exp, fds.res_ex))
		ret = -EFAULT;

out:
	if (bits != stack_fds)
		kvfree(bits);                  // 释放大块分配
out_nofds:
	return ret;
}
```

`fd_set` 结构（用户态和内核通信用）：
```c
// linux/include/uapi/linux/posix_types.h
// __kernel_fd_set 是 fd_set 的内核表示
// fds_bits[16] = 16 * 64bits = 1024bits → 每 bit 对应 fd 0~1023
typedef struct { unsigned long fds_bits[16]; } __kernel_fd_set;
```

---

## 3. 完整调用栈

### 3.1 内核态调用栈（主路径：select 进入睡眠）

```text
Linux SYSCALL_DEFINE6(pselect6, n, inp, outp, exp, tsp, sig)
│  内核 syscall 入口：n=nfds, inp=读 fd 集合, tsp=超时时间
└── sys_pselect6() [linux/fs/select.c:960]
    ├── get_sigset_argpack(&x, sig)       → sig=NULL，跳过
    └── do_pselect(n, inp, outp, exp, tsp, x.p, x.size, PT_TIMESPEC)
        ├── get_timespec64(tsp, &ts)       → 从用户态读超时
        └── core_sys_select(n, inp, outp, exp, end_time)
            ├── get_fd_set(n, inp, fds.in)    → copy_from_user(fd_set) 用户态→内核
            ├── get_fd_set(n, outp, fds.out)
            ├── get_fd_set(n, exp, fds.ex)
            └── do_select(n, &fds, end_time)
                ├── max_select_fd(n, fds)     → 获取有效最大 fd
                ├── poll_initwait(&table)    → 初始化 poll_wqueues，pt._qproc=__pollwait
                └── for (;;) {
                        // 遍历所有 fd，调用 vfs_poll 检查状态
                        for (i = 0; i < n; ++rinp, ++routp, ++rexp) {
                            for (j = 0; j < BITS_PER_LONG; ++j, ++i, bit <<= 1) {
                                f = fdget(i)                → 取 struct file *
                                mask = vfs_poll(f, wait)   → 检查 fd 状态
                                    └── video_poll(f, wait) → V4L2 驱动 poll
                                        └── uvc_poll(f, wait)
                                            └── poll_wait(f, &done_wq, wait)
                                                └── __pollwait()
                                                    └── add_wait_queue(done_wq, &entry->wait)
                                fdput(f)
                                if (mask & POLLIN_SET && in & bit) {
                                    res_in |= bit; retval++; wait->_qproc = NULL;
                                }
                            }
                        }
                        if (retval || timed_out || signal_pending(current)) break;
                        // 无 fd 就绪 → 进入睡眠
                        poll_schedule_timeout(&table, TASK_INTERRUPTIBLE, to, slack)
                            ├── set_current_state(TASK_INTERRUPTIBLE)
                            └── schedule_hrtimeout_range()
                                │  核心等待点：【进程进入 TASK_INTERRUPTIBLE 睡眠】
                                ▼  等待 wake_up(done_wq) 或 超时 或 信号
                └── poll_freewait(&table)
            └── set_fd_set(n, inp, fds.res_in) → copy_to_user(fd_set) 内核→用户态
```

### 3.2 内核态调用栈（唤醒路径：wake_up 到 select 返回）

```text
// 中断/驱动上下文
uvc_video_complete(urb)                       → USB URB 完成回调 [uvc_video.c]
└── uvc_video_decode(urb)
    └── uvc_queue_buffer()
        └── vb2_buffer_done(vb, VB2_BUF_STATE_DONE)
            ├── list_add_tail(&vb->done_entry, &q->done_list)
            └── wake_up(&q->done_wq)
                │  核心唤醒点：唤醒等待在 done_wq 上的 do_select 进程
                ▼  【进程从 TASK_INTERRUPTIBLE 唤醒】

// do_select 被唤醒后
poll_schedule_timeout()                       → 返回（triggered=1）
    └── smp_store_mb(pwq->triggered, 0)       → 清零 triggered
do_select() 继续 for 循环                    → 再次调用 vfs_poll()
    └── vfs_poll(file, wait)                   → uvc_poll()
        └── poll_wait(file, &done_wq, wait)
            // done_list 非空 → 返回 POLLIN
            // res_in |= bit; retval++; wait->_qproc = NULL;
if (retval || timed_out || signal_pending(current)) break;
// 退出主循环
poll_freewait(&table)
return retval                                  → 返回用户态（就绪 fd 数量）
```

---

## 4. 核心轮询与等待

### 3.1 `do_select()` — 实际 select 逻辑

`do_select()` 是 select 系统的核心。它遍历所有被监控的 fd，调用 `vfs_poll()` 检查是否就绪；如果没有 fd 就绪，就调用 `poll_schedule_timeout()` 进入睡眠。

```c
// linux/fs/select.c:477
// 函数名：do_select()
// 作用：核心 I/O 多路复用函数。监控多个 fd，直到其中某个就绪或超时。
// 参数：n=nfds（最大 fd+1）, fds=内核位图结构, end_time=绝对超时时间
// 返回值：>0=就绪 fd 数量，0=超时，<0=错误
static noinline_for_stack int do_select(int n, fd_set_bits *fds, struct timespec64 *end_time)
{
	// ========== 变量定义 ==========
	ktime_t expire, *to = NULL;         // expire=超时到期时间
	struct poll_wqueues table;          // poll_wqueues：管理等待队列结构
	poll_table *wait;                   // wait：poll 表指针
	int retval, i, timed_out = 0;      // retval=返回值，i=循环计数
	u64 slack = 0;                       // slack=调度精度估算值

	// ========== 第一步：获取有效最大 fd ==========
	rcu_read_lock();
	retval = max_select_fd(n, fds);    // max_select_fd：获取有效最大 fd
	rcu_read_lock();

	if (retval < 0)
		return retval;
	n = retval;                          // 裁剪到实际最大 fd

	// ========== 第二步：初始化 poll_wqueues ==========
	poll_initwait(&table);              // 初始化等待队列结构
	wait = &table.pt;

	// ========== 第三步：处理零超时 ==========
	if (end_time && !end_time->tv_sec && !end_time->tv_nsec) {
		wait->_qproc = NULL;            // 禁用回调，直接轮询
		timed_out = 1;                  // 标记立即超时
	}

	// ========== 第四步：估算超时调度精度 ==========
	if (end_time && !timed_out)
		slack = select_estimate_accuracy(end_time);

	// ========== 第五步：主循环 ==========
	retval = 0;
	for (;;) {
		unsigned long *rinp, *routp, *rexp, *inp, *outp, *exp;
		...

		inp = fds->in; outp = fds->out; exp = fds->ex;    // 输入位图
		rinp = fds->res_in; routp = fds->res_out; rexp = fds->res_ex; // 输出位图

		// 遍历每个 fd 位图块（每块 64 bits）
		for (i = 0; i < n; ++rinp, ++routp, ++rexp) {
			unsigned long in, out, ex, all_bits, bit = 1, j;
			unsigned long res_in = 0, res_out = 0, res_ex = 0;
			__poll_t mask;

			in = *inp++; out = *outp++; ex = *exp++; // 读取当前块
			all_bits = in | out | ex;               // 合并所有感兴趣的位
			if (all_bits == 0) {
				i += BITS_PER_LONG;                  // 无感兴趣位，跳过
				continue;
			}

			// 遍历当前块中的每个 fd
			for (j = 0; j < BITS_PER_LONG; ++j, ++i, bit <<= 1) {
				struct fd f;
				if (i >= n)
					break;
				if (!(bit & all_bits))              // 不感兴趣，跳过
					continue;

				mask = EPOLLNVAL;
				f = fdget(i);                       // 取 fd 对应的 file
				if (fd_file(f)) {
					// 设置等待键值，用于回调时判断哪些 fd 就绪
					wait_key_set(wait, in, out, bit, busy_flag);
					// 关键：调用 vfs_poll 检查文件状态
					// 对 /dev/video0，vfs_poll → video_poll → uvc_poll
					// uvc_poll 检查 vb2_queue 的 done_list 是否非空
					mask = vfs_poll(fd_file(f), wait);
					fdput(f);
				}

				// 检查读就绪（POLLIN）
				if ((mask & POLLIN_SET) && (in & bit)) {
					res_in |= bit;
					retval++;
					wait->_qproc = NULL;             // 有 fd 就绪，禁用回调
				}
				// 检查写就绪（POLLOUT）
				if ((mask & POLLOUT_SET) && (out & bit)) {
					res_out |= bit;
					retval++;
					wait->_qproc = NULL;
				}
				// 检查异常（POLLEX）
				if ((mask & POLLEX_SET) && (ex & bit)) {
					res_ex |= bit;
					retval++;
					wait->_qproc = NULL;
				}
			}

			// 保存结果位图
			if (res_in) *rinp = res_in;
			if (res_out) *routp = res_out;
			if (res_ex) *rexp = res_ex;

			cond_resched();                 // 让出 CPU
		}

		// ========== 检查退出条件 ==========
		wait->_qproc = NULL;               // 重置回调
		if (retval || timed_out || signal_pending(current)) // 有就绪 或 超时 或 信号
			break;
		if (table.error) {                 // 错误
			retval = table.error;
			break;
		}

		// ========== 进入睡眠等待 ==========
		if (end_time && !to) {
			expire = timespec64_to_ktime(*end_time); // 转为 ktime
			to = &expire;
		}

		// 关键等待：进程在此睡眠，直到有 fd 就绪、超时或被信号唤醒
		// poll_schedule_timeout 设置状态为 TASK_INTERRUPTIBLE 后调用 schedule_timeout
		if (!poll_schedule_timeout(&table, TASK_INTERRUPTIBLE, to, slack))
			timed_out = 1;                 // 返回 false = 超时
	}

	poll_freewait(&table);                // 清理等待队列
	return retval;                        // 返回就绪 fd 数量
}
```

### 3.2 `poll_schedule_timeout()` — 进入睡眠

`do_select()` 调用 `poll_schedule_timeout()` 进入睡眠。这是 select 路径的核心等待点。

```c
// linux/fs/select.c:235
// 函数名：poll_schedule_timeout()
// 作用：设置进程状态为 TASK_INTERRUPTIBLE，调用 schedule_timeout 进入睡眠。
// 参数：pwq=poll_wqueues 结构, state=睡眠状态, expires=超时时间, slack=调度精度
// 返回值：-EINTR（被信号打断）或 0（超时）
static int poll_schedule_timeout(struct poll_wqueues *pwq, int state,
				  ktime_t *expires, unsigned long slack)
{
	int rc = -EINTR;

	set_current_state(state);            // 设置进程状态为 TASK_INTERRUPTIBLE
	if (!pwq->triggered)                 // 如果没有被立即唤醒（如已有 fd 就绪）
		rc = schedule_hrtimeout_range(expires, slack, HRTIMER_MODE_ABS);
	__set_current_state(TASK_RUNNING);   // 恢复到 RUNNING

	// smp_store_mb 清零 triggered，准备下一轮
	smp_store_mb(pwq->triggered, 0);

	return rc;
}
```

进程进入睡眠后，当以下任一条件满足时被唤醒：
- **`wake_up()` 被调用**（如 V4L2 fd 收到帧，uvc_poll 调用的等待被触发）
- **超时到期**（hrtimer 到期触发）
- **收到信号**（如 SIGINT）

### 3.3 `vfs_poll()` — VFS 层调用驱动 poll

对 `/dev/video0`，`vfs_poll()` 会调用 V4L2 驱动的 `video_poll()`，最终检查 videobuf2 的 `done_list` 是否非空：

```c
// linux/fs/select.c 最终调用
mask = vfs_poll(fd_file(f), wait);       // 检查 fd 是否就绪
```

```c
// linux/include/linux/poll.h
// vfs_poll 是内联函数，调用文件类型的 poll 方法
static inline __poll_t vfs_poll(struct file *file, poll_table *wait)
{
	if (file->f_op->poll)                 // 确认驱动实现了 poll
		return file->f_op->poll(file, wait); // 调驱动 poll（如 video_poll）
	return DEFAULT_POLLMASK;              // 默认返回可读可写
}
```

对 V4L2 设备，最终调用 `video_poll()` → `uvc_poll()`。`uvc_poll` 检查 `done_list` 是否有完成帧：

```c
// V4L2 驱动路径（简化）
static unsigned int uvc_poll(struct file *file, struct poll_table_struct *wait)
{
	...
	poll_wait(file, &stream->queue.done_wq, wait); // 把进程加入 done_wq 等待队列
	// 如果 done_list 非空，返回 POLLIN
	if (!list_empty(&stream->queue.done_list))
		return POLLIN;
	// 否则返回 0，让 do_select 进入睡眠
	return 0;
}
```

---

## 4. poll_wqueues 与等待队列

### 4.1 `struct poll_wqueues` — 核心数据结构

```c
// linux/include/linux/poll.h
// struct poll_wqueues — poll/select 等待队列管理结构
//   pt         : poll_table，传递给 vfs_poll 的等待表
//   polling_task: 当前轮询进程
//   triggered  : 标记是否被立即唤醒（避免不必要的睡眠）
//   error      : 错误码
//   table      : 动态分配的 poll_table_page 链表
//   inline_entries: 内联的小型 poll_table_entry 数组（避免小分配）
struct poll_wqueues {
    poll_table pt;
    struct task_struct *polling_task;
    int triggered;
    int error;
    struct poll_table_page *table;
    struct poll_table_entry *inline_entries;
};
```

### 4.2 `poll_initwait()` — 初始化

```c
// linux/fs/select.c:120
// 函数名：poll_initwait()
// 作用：初始化 poll_wqueues，设置回调函数为 __pollwait
void poll_initwait(struct poll_wqueues *pwq)
{
	init_poll_funcptr(&pwq->pt, __pollwait); // pt._qproc = __pollwait
	pwq->polling_task = current;
	pwq->triggered = 0;
	pwq->error = 0;
	pwq->table = NULL;
	pwq->inline_index = 0;
}
```

### 4.3 `__pollwait()` — 把进程加入设备等待队列

```c
// linux/fs/select.c:220
// 函数名：__pollwait()
// 作用：poll/select 的回调函数。把当前进程加入设备驱动的等待队列。
//       当设备变为就绪时，驱动调用 wake_up() 唤醒进程。
// 参数：filp=struct file *, wait_address=设备等待队列头（如 vb2_queue.done_wq）, p=poll_table
static void __pollwait(struct file *filp, wait_queue_head_t *wait_address,
				poll_table *p)
{
	struct poll_wqueues *pwq = container_of(p, struct poll_wqueues, pt);
	struct poll_table_entry *entry = poll_get_entry(pwq);
	if (!entry)
		return;
	entry->filp = get_file(filp);
	entry->wait_address = wait_address;        // 如 stream->queue.done_wq
	entry->key = p->_key;                      // POLLIN/POLLOUT 等
	init_waitqueue_func_entry(&entry->wait, pollwake); // 唤醒函数
	entry->wait.private = pwq;
	add_wait_queue(wait_address, &entry->wait); // 加入设备等待队列
}
```

---

## 5. select 和 pselect6 的区别

| 特性 | select | pselect6 |
|------|--------|----------|
| 超时精度 | `struct timeval`（微秒） | `struct timespec`（纳秒） |
| 信号掩码 | 通过单独 sigprocmask 调用（非原子） | 第6参数原子传递 sigmask |
| syscall | `__NR_select=64` | `__NR_pselect6=72` |

LeRobot 的 Python `select()` 在 glibc 实际走 `pselect6`：

```text
用户态 select() → glibc __select64() → pselect6 syscall → sys_pselect6()
```

---

## 6. 完整调用栈

### 6.1 select 系统调用主路径

```text
// 用户态
glibc select(fd, &readfds, NULL, NULL, &timeout)
    └── __select64() → SYSCALL_CANCEL(pselect6_time64, ...)

// 内核态
sys_pselect6() [linux/fs/select.c:960]
├── get_sigset_argpack(&x, sig)       → sig=NULL，跳过
└── do_pselect(n, inp, outp, exp, tsp, x.p, x.size, PT_TIMESPEC)
    ├── get_timespec64(tsp, &ts)       → 读用户态超时
    ├── set_user_sigmask(sigmask, size) → sig=NULL，跳过
    └── core_sys_select(n, inp, outp, exp, end_time)
        ├── get_fd_set(n, inp, fds.in)    → copy_from_user(fd_set)
        ├── get_fd_set(n, outp, fds.out)
        ├── get_fd_set(n, exp, fds.ex)
        └── do_select(n, &fds, end_time)
            ├── max_select_fd()            → 获取有效最大 fd
            ├── poll_initwait(&table)      → 初始化等待队列
            ├── for (;;) {
            │   ├── for (i=0; i<n; i++) {
            │   │   ├── fdget(i)           → 取 struct file *
            │   │   └── vfs_poll(file, wait) → 检查 fd 状态
            │   │       └── video_poll()   → V4L2 驱动 poll
            │   │           └── uvc_poll()
            │   │               └── poll_wait(file, &done_wq, wait)
            │   │                   └── __pollwait()
            │   │                       └── add_wait_queue(done_wq, &entry->wait)
            │   │               // done_list 非空 → POLLIN → 返回
            │   }
            │   ├── if (retval || timed_out || signal_pending) break;
            │   └── poll_schedule_timeout(&table, TASK_INTERRUPTIBLE, to, slack)
            │       ├── set_current_state(TASK_INTERRUPTIBLE)
            │       └── schedule_hrtimeout_range()
            │           // 进程进入 TASK_INTERRUPTIBLE 睡眠
            │           // 被 wake_up(done_wq) 或 hrtimer 或信号唤醒
            └── poll_freewait(&table)
        └── set_fd_set(n, inp, fds.res_in) → copy_to_user(fd_set)
```

### 6.2 唤醒路径（从 V4L2 done_list 非空）

```text
// 中断/驱动上下文
uvc_video_complete(urb)                       → USB URB 完成
└── uvc_video_decode(urb)
    └── vb2_buffer_done(vb, VB2_BUF_STATE_DONE)
        ├── list_add_tail(done_list)
        └── wake_up(&stream->queue.done_wq)
            // done_wq 上的进程被唤醒

// select 进程被唤醒后
poll_schedule_timeout()                       → 返回（triggered=1）
do_select()                                   → 再次循环
    └── vfs_poll(file, wait)                  → uvc_poll()
        └── poll_wait(file, &done_wq, wait)  // __pollwait 已注册
            // done_list 非空，返回 POLLIN
            // retval++，发现有 fd 就绪
// 退出主循环
return retval                                 → 返回用户态
```

---

## 7. 性能结论

1. **`select` 本质是轮询 + 睡眠**。`do_select()` 先用 `vfs_poll()` 轮询一遍所有 fd，如果都没有就绪，才进入 `poll_schedule_timeout()` 睡眠等待。

2. **关键等待点在 `schedule_hrtimeout_range()`**。这是 Linux 调度器的函数，使进程进入 `TASK_INTERRUPTIBLE` 状态，直到超时到期或被唤醒。

3. **唤醒源是 `done_wq`**。当 URB 完成时，UVC 驱动调用 `wake_up(&stream->queue.done_wq)` 唤醒等待在该队列上的进程。

4. **`select` 不直接监控 V4L2 done_list**。它通过 `vfs_poll()` 间接检查。在 OpenCV 的非阻塞取帧模式下，DQBUF 返回 EAGAIN 后调用 select，select 用 poll 表机制把进程注册到 done_wq 等待队列。

5. **性能影响**：`select` 的精度是毫秒级（受 hrtimer 和调度器影响），频繁调用 select 但帧率不高时会造成无效的上下文切换。